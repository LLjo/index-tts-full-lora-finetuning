#!/usr/bin/env python3
"""Wyoming TTS bridge for IndexTTS2.

Listens on TCP for Wyoming events from Home Assistant, translates `synthesize`
requests into HTTP POSTs against the local IndexTTS API, and streams the
resulting raw int16 PCM back as Wyoming `audio-chunk` events.

Designed for the lowest possible TTFA:
  * Uses the `?raw_pcm=true` flag on /inference/stream so there is no WAV
    header to strip per chunk.
  * Caches the currently-merged speaker LoRA — only fires /models/load/<name>
    when the requested voice changes, so a steady stream of HA queries against
    the same voice pays the LoRA-merge cost only once.
  * httpx.AsyncClient with HTTP/1.1 keep-alive so the TCP connection to the
    API stays warm across utterances.
  * Forwards each PCM chunk to HA immediately as it arrives — no batching or
    buffering beyond what asyncio queues do on their own.

Voices advertised to HA are pulled live from /speakers and filtered to
speakers with a real (new-trainer) character LoRA. Legacy verbatim/pattern
adapters are NOT exposed — they were trained on a different input
distribution and would surprise users.

Env vars:
  INDEXTTS_API_URL         (default: http://localhost:8000)
  INDEXTTS_WYOMING_PORT    (default: 10200)
  INDEXTTS_STREAMING_PRESET    (default: fast_quality)
  INDEXTTS_SOLVER_OVERRIDE     (optional, e.g. single_step)
  INDEXTTS_DIFFUSION_STEPS_OVERRIDE  (optional int, e.g. 1)
  INDEXTTS_CFG_OVERRIDE        (optional float, e.g. 0.0)
  INDEXTTS_REFERENCE_AUDIO     (optional absolute path; if set, used for all
                               speakers regardless of which one HA picks.
                               Otherwise the bridge auto-picks the first audio
                               clip from training/<speaker>/dataset/audio/.)
  INDEXTTS_LANGUAGE        (default: en)
  INDEXTTS_ATTRIB_NAME     (default: IndexTTS2)
  INDEXTTS_ATTRIB_URL      (default: https://github.com/index-tts/index-tts)

Reference-audio resolution chain (per synthesize call):
  1. If INDEXTTS_REFERENCE_AUDIO is set, upload that file. The API caches
     conditioning by path so the cost amortizes to ~2ms per call.
  2. Else auto-pick the first audio clip from
     training/<speaker>/dataset/audio/. Same caching.
  3. Stored speaker_embeddings.pt path is intentionally NOT preferred — on
     this codebase it tends to be both slower (load cost) and lower-quality
     than the audio-upload path. The `.pt` load is now cached in-process so
     it's at least not slow on the second call, but the audio-derived
     conditioning produces better-sounding output.

Usage (manual):
    INDEXTTS_API_URL=http://localhost:8000 \\
    INDEXTTS_WYOMING_PORT=10200 \\
    python tools/wyoming_indextts.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeStart,
    SynthesizeChunk,
    SynthesizeStop,
    SynthesizeStopped,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG = logging.getLogger("wyoming-indextts")

# Audio format we negotiate with HA. Matches the IndexTTS streaming output:
# 22050 Hz, signed 16-bit LE, mono. The /inference/stream?raw_pcm=true endpoint
# emits exactly this.
SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # bytes
CHANNELS = 1

# Chunk granularity to forward to HA. The API already emits well-sized chunks;
# we slice to at most this many PCM samples per Wyoming audio-chunk so HA's
# media player can start playing the leading edge without waiting on a fat
# packet. 1024 samples = ~46ms of audio.
WYOMING_CHUNK_SAMPLES = 1024
WYOMING_CHUNK_BYTES = WYOMING_CHUNK_SAMPLES * SAMPLE_WIDTH * CHANNELS

# Shared httpx client — keep-alive across utterances. Set per process.
_HTTP_CLIENT: httpx.AsyncClient | None = None
_LOAD_LOCK: asyncio.Lock | None = None
_CURRENT_LOADED_VOICE: str | None = None  # last speaker we ran /models/load/<>


def _settings() -> dict:
    ref = os.environ.get("INDEXTTS_REFERENCE_AUDIO") or None
    return {
        "api_url": os.environ.get("INDEXTTS_API_URL", "http://localhost:8000").rstrip("/"),
        "wyoming_port": int(os.environ.get("INDEXTTS_WYOMING_PORT", "10200")),
        "preset": os.environ.get("INDEXTTS_STREAMING_PRESET", "fast_quality"),
        "solver_override": os.environ.get("INDEXTTS_SOLVER_OVERRIDE") or None,
        "diffusion_steps_override": _opt_int(os.environ.get("INDEXTTS_DIFFUSION_STEPS_OVERRIDE")),
        "cfg_override": _opt_float(os.environ.get("INDEXTTS_CFG_OVERRIDE")),
        "reference_audio": Path(ref).resolve() if ref else None,
        "language": os.environ.get("INDEXTTS_LANGUAGE", "en"),
        "attrib_name": os.environ.get("INDEXTTS_ATTRIB_NAME", "IndexTTS2"),
        "attrib_url": os.environ.get("INDEXTTS_ATTRIB_URL", "https://github.com/index-tts/index-tts"),
    }


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _resolve_reference_audio(voice_name: str, settings: dict) -> Path | None:
    """Decide which reference audio to attach to the synth request.

    Stored speaker_embeddings.pt is deliberately NOT consulted here: on this
    codebase the audio-upload path produces better-sounding output and is
    similarly fast (cond cache memoizes by file path on the second call).
    Falling back to embeddings would silently change the quality envelope.
    """
    speaker_dir = PROJECT_ROOT / "training" / voice_name

    # 1. Explicit env override
    if settings.get("reference_audio"):
        p = settings["reference_audio"]
        if p.exists():
            return p
        LOG.warning("INDEXTTS_REFERENCE_AUDIO=%s does not exist; falling back", p)

    # 2. First audio file in this speaker's dataset/audio/
    audio_dir = speaker_dir / "dataset" / "audio"
    if audio_dir.exists():
        for p in sorted(audio_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
                return p

    LOG.error(
        "No reference audio available for speaker '%s'. Drop a clip into %s/ "
        "or set INDEXTTS_REFERENCE_AUDIO.",
        voice_name, audio_dir,
    )
    return None


def _opt_int(v):
    try: return int(v) if v is not None and v != "" else None
    except ValueError: return None


def _opt_float(v):
    try: return float(v) if v is not None and v != "" else None
    except ValueError: return None


async def _fetch_voices(api_url: str) -> list[TtsVoice]:
    """Hit /speakers, filter to character-LoRA kind, build Wyoming voice list."""
    settings = _settings()
    attribution = Attribution(name=settings["attrib_name"], url=settings["attrib_url"])
    try:
        client = await _client()
        r = await client.get(f"{api_url}/speakers", timeout=10)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        LOG.warning("Could not load voice list from %s/speakers: %s", api_url, e)
        return []

    voices: list[TtsVoice] = []
    for row in rows:
        # Only advertise speakers with a real (new) character LoRA. Legacy
        # verbatim/pattern adapters load fine at inference but would behave
        # unpredictably for HA users expecting "this voice talks like X".
        if row.get("gpt_lora_kind") != "character":
            continue
        voices.append(TtsVoice(
            name=row["name"],
            description=f"IndexTTS2 character LoRA · {row['name']}",
            attribution=attribution,
            installed=True,
            version=None,
            languages=[settings["language"]],
        ))
    if not voices:
        LOG.warning("No character-LoRA speakers found. HA will see an empty voice list.")
    else:
        LOG.info("Advertising %d voice(s) to HA: %s",
                 len(voices), ", ".join(v.name for v in voices))
    return voices


async def _client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=600, write=10, pool=10))
    return _HTTP_CLIENT


async def _ensure_voice_loaded(api_url: str, voice_name: str) -> None:
    """Call /models/load/<voice> only when the requested voice changed.

    The IndexTTS load_lora() implementation restores the base GPT state first
    and then merges the new LoRA, so swapping is safe to call repeatedly — but
    each call still costs ~1s. Caching here makes back-to-back HA queries
    against the same voice essentially free.
    """
    global _CURRENT_LOADED_VOICE, _LOAD_LOCK
    if _LOAD_LOCK is None:
        _LOAD_LOCK = asyncio.Lock()
    if _CURRENT_LOADED_VOICE == voice_name:
        return
    async with _LOAD_LOCK:
        if _CURRENT_LOADED_VOICE == voice_name:  # re-check after acquiring
            return
        client = await _client()
        LOG.info("Loading speaker LoRA: %s", voice_name)
        r = await client.post(f"{api_url}/models/load/{voice_name}", timeout=60)
        if not r.is_success:
            LOG.error("Speaker load failed (%s): %s", r.status_code, r.text[:200])
            r.raise_for_status()
        _CURRENT_LOADED_VOICE = voice_name


def _build_request_body(text: str, voice_name: str, settings: dict) -> dict:
    """Construct the JSON body for /inference/stream.

    Mirrors what the WebUI sends. use_patterns=True so the API attaches whatever
    pattern_embedding / character LoRA the speaker has. The character LoRA gets
    merged via /models/load/<voice> — see _ensure_voice_loaded.
    """
    body: dict = {
        "text": text,
        "speaker": voice_name,
        "use_patterns": True,
        "verbose": False,
        "streaming_preset": settings["preset"],
    }
    if settings["solver_override"]:
        body["solver_override"] = settings["solver_override"]
    if settings["diffusion_steps_override"] is not None:
        body["diffusion_steps_override"] = settings["diffusion_steps_override"]
    if settings["cfg_override"] is not None:
        body["inference_cfg_override"] = settings["cfg_override"]
    return body


class IndexTTSHandler(AsyncEventHandler):
    """One handler per HA connection. Wyoming server spawns a new instance for
    each TCP client; we hold no per-handler state beyond what the protocol
    requires."""

    def __init__(self, *args, settings: dict, voices: list[TtsVoice], **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.voices = voices
        # Streaming-synthesis state. HA 2025.10+ sends synthesize-start,
        # one or more synthesize-chunk, then synthesize-stop instead of a
        # single legacy `synthesize` event when supports_synthesize_streaming
        # is advertised. We buffer chunks here and flush on stop.
        self._stream_voice: str | None = None
        self._stream_buf: list[str] = []

    def _pick_voice(self, requested) -> str | None:
        if requested is not None and getattr(requested, "name", None):
            return requested.name
        return self.voices[0].name if self.voices else None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            info = Info(tts=[TtsProgram(
                name="indextts2",
                description="IndexTTS2 — character-LoRA TTS with distilled CFM streaming",
                attribution=Attribution(
                    name=self.settings["attrib_name"],
                    url=self.settings["attrib_url"],
                ),
                installed=True,
                voices=self.voices,
                version=None,
                supports_synthesize_streaming=True,
            )])
            await self.write_event(info.event())
            return True

        # Legacy single-shot path (HA < 2025.10 or any client not using streaming).
        # HA 2026.x dispatches both the streaming sequence and a redundant legacy
        # Synthesize for the same utterance — if we're mid-stream, skip this one
        # to avoid running inference twice.
        if Synthesize.is_type(event.type):
            if self._stream_voice is not None:
                LOG.info("[bridge] ignoring legacy Synthesize (streaming session active)")
                return True
            synth = Synthesize.from_event(event)
            voice_name = self._pick_voice(synth.voice)
            if voice_name is None:
                LOG.error("Synthesize requested but no voices configured")
                return True
            try:
                await self._synthesize(synth.text, voice_name)
            except Exception:
                LOG.exception("Synthesize failed")
            return True

        # Streaming path: synthesize-start / -chunk / -stop.
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._stream_voice = self._pick_voice(start.voice)
            self._stream_buf = []
            LOG.info("[STREAM] synthesize-start  voice=%s", self._stream_voice)
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            self._stream_buf.append(chunk.text)
            LOG.info("[STREAM] synthesize-chunk  text=%r  total_chunks=%d",
                     chunk.text[:60], len(self._stream_buf))
            return True

        if SynthesizeStop.is_type(event.type):
            voice_name = self._stream_voice
            text = "".join(self._stream_buf)
            self._stream_voice = None
            self._stream_buf = []
            LOG.info("[STREAM] synthesize-stop  voice=%s  text_len=%d  text=%r",
                     voice_name, len(text), text[:80])
            if not text:
                LOG.warning("synthesize-stop received with empty text buffer")
                await self.write_event(SynthesizeStopped().event())
                return True
            if voice_name is None:
                LOG.error("synthesize-stop with no voice configured")
                await self.write_event(SynthesizeStopped().event())
                return True
            try:
                await self._synthesize(text, voice_name)
            except Exception:
                LOG.exception("Streaming synthesize failed")
            await self.write_event(SynthesizeStopped().event())
            LOG.info("[STREAM] sent SynthesizeStopped")
            return True

        # Unknown event — log it so we can see if HA is sending something we
        # don't yet handle.
        LOG.debug("[bridge] unhandled event type=%s", event.type)
        return True

    async def _synthesize(self, text: str, voice_name: str) -> None:
        api_url = self.settings["api_url"]
        # 1. Swap LoRA if needed (cached — usually a no-op on repeated calls)
        await _ensure_voice_loaded(api_url, voice_name)

        # 2. Decide whether to attach a reference audio. Stored embeddings →
        #    no upload (fastest); otherwise upload one wav per request and
        #    rely on the API's cond cache to keep extraction at ~2ms after
        #    the first call.
        body = _build_request_body(text, voice_name, self.settings)
        ref_path = _resolve_reference_audio(voice_name, self.settings)
        files: dict = {"request_json": (None, json.dumps(body))}
        ref_fh = None
        if ref_path is not None:
            ref_fh = open(ref_path, "rb")
            files["audio_file"] = (ref_path.name, ref_fh, "audio/wav")
            # cond_cache_key is the stable identifier the API uses to memoize
            # speaker conditioning. Path of the on-disk ref clip is perfect
            # because it doesn't change between requests for the same voice.
            body["cond_cache_key"] = str(ref_path)
            files["request_json"] = (None, json.dumps(body))

        client = await _client()
        LOG.info("Synthesize  voice=%s  ref=%s  text=%r",
                 voice_name, ref_path.name if ref_path else "<embeddings>", text[:80])

        await self.write_event(AudioStart(
            rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS,
        ).event())

        try:
            async with client.stream(
                "POST",
                f"{api_url}/inference/stream?raw_pcm=true",
                files=files,
            ) as resp:
                if not resp.is_success:
                    body_text = (await resp.aread()).decode("utf-8", "replace")
                    LOG.error("stream %s: %s", resp.status_code, body_text[:200])
                    return

                # Re-chunk into Wyoming-friendly sizes. The API already emits
                # short chunks, but its boundary doesn't have to match
                # WYOMING_CHUNK_BYTES; we re-slice for consistent audio-chunk
                # framing. Leftover bytes are kept in `leftover` between iters.
                leftover = b""
                async for raw in resp.aiter_bytes(chunk_size=WYOMING_CHUNK_BYTES):
                    if not raw:
                        continue
                    buf = leftover + raw
                    # Emit whole frames; keep odd trailing byte if any (shouldn't
                    # happen at 16-bit alignment, but guard regardless).
                    n_full = (len(buf) // SAMPLE_WIDTH) * SAMPLE_WIDTH
                    if n_full:
                        # Slice into up-to-WYOMING_CHUNK_BYTES pieces
                        i = 0
                        while i < n_full:
                            piece = buf[i : i + WYOMING_CHUNK_BYTES]
                            await self.write_event(AudioChunk(
                                rate=SAMPLE_RATE,
                                width=SAMPLE_WIDTH,
                                channels=CHANNELS,
                                audio=piece,
                            ).event())
                            i += len(piece)
                    leftover = buf[n_full:]

                if leftover:
                    # Unlikely (16-bit aligned upstream) — pad with one zero byte
                    LOG.warning("Trailing %d bytes of unaligned PCM dropped", len(leftover))
        finally:
            if ref_fh is not None:
                try: ref_fh.close()
                except Exception: pass
            await self.write_event(AudioStop().event())


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default=os.environ.get("INDEXTTS_LOG_LEVEL", "INFO"))
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    settings = _settings()
    LOG.info("Settings: %s", {k: v for k, v in settings.items() if "url" in k or "port" in k})

    # Warm the HTTP client; failure here is non-fatal (we retry on each query).
    voices = await _fetch_voices(settings["api_url"])

    def handler_factory(reader, writer):
        return IndexTTSHandler(reader, writer, settings=settings, voices=voices)

    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{settings['wyoming_port']}")
    LOG.info("Wyoming bridge listening on tcp://0.0.0.0:%d", settings["wyoming_port"])
    try:
        await server.run(handler_factory)
    finally:
        client = _HTTP_CLIENT
        if client is not None:
            await client.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
