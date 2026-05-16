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
import re
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


# Boundary = sentence terminator + whitespace, OR a paragraph break.
# Negative lookbehind on digits prevents splitting numbered lists ("1. Foo").
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<![0-9])[.!?]+(?=\s)|\n{2,}')
# Sentences shorter than this get fused with the next one. Smaller = faster
# time-to-first-audio; larger = better intra-sentence prosody. We use a very
# small threshold for the FIRST sentence (so e.g. "Sure!" emits immediately
# and the user hears something fast) and a larger one for subsequent sentences
# (so body prose doesn't get chopped into awkward fragments).
_FIRST_SENTENCE_MIN_CHARS = 5
_SUBSEQUENT_SENTENCE_MIN_CHARS = 30


class _SentenceStream:
    """Incremental sentence extractor for streaming LLM output.

    Feed text chunks via `feed()`; it returns any complete sentences that have
    accumulated since the last call. Call `flush()` at end-of-stream to get
    whatever's still buffered."""

    def __init__(self) -> None:
        self._buf = ""
        self._search_from = 0  # next regex search starts here (skips rejected boundaries)
        self._emitted = 0  # how many sentences we've returned in this session

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            m = _SENTENCE_BOUNDARY_RE.search(self._buf, self._search_from)
            if not m:
                break
            cut = m.end()
            sentence = self._buf[:cut].strip()
            min_chars = (_FIRST_SENTENCE_MIN_CHARS if self._emitted == 0
                         else _SUBSEQUENT_SENTENCE_MIN_CHARS)
            if len(sentence) >= min_chars:
                out.append(sentence)
                self._emitted += 1
                self._buf = self._buf[cut:].lstrip()
                self._search_from = 0
            else:
                # Too short — advance past this boundary so we keep accumulating,
                # but leave the buffer's actual chars (incl. the terminator)
                # intact for when we eventually emit.
                self._search_from = cut
        return out

    def flush(self) -> str | None:
        remaining = self._buf.strip()
        self._buf = ""
        self._search_from = 0
        return remaining if remaining else None

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


async def _fetch_active_distilled_speaker(api_url: str) -> str | None:
    """Query /distill/active-status to find which speaker's distilled CFM
    student is currently loaded in the running model. Returns the speaker
    name (so we can auto-apply single_step + 1 step + CFG=0 for that voice)
    or None if no distilled student is in use.

    Cached at bridge startup — if the user activates a different student via
    the WebUI after startup, they need to restart the bridge."""
    try:
        client = await _client()
        r = await client.get(f"{api_url}/distill/active-status", timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        LOG.info("Could not query /distill/active-status (%s) — distilled "
                 "auto-detection disabled this run.", e)
        return None
    if not data.get("in_use"):
        LOG.info("No distilled CFM student is active. Using preset solver.")
        return None
    speaker = data.get("speaker_match")
    if speaker:
        LOG.info("Distilled student active for speaker=%s — will auto-apply "
                 "single_step / 1 step / CFG=0 for that voice.", speaker)
    else:
        LOG.info("Distilled student is active but speaker is unknown — "
                 "single_step won't be auto-applied (set INDEXTTS_SOLVER_OVERRIDE "
                 "manually if you want it).")
    return speaker


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

    # Precedence for solver / steps / cfg:
    #   1. Explicit env-var override (e.g. INDEXTTS_SOLVER_OVERRIDE).
    #   2. Auto-detect: if a distilled CFM student is active in the running
    #      model AND it belongs to the voice we're synthesizing, use
    #      single_step / 1 step / CFG=0 — the regime the student was trained
    #      for. Saves several diffusion roundtrips per chunk.
    #   3. Otherwise leave it to the preset.
    solver = settings.get("solver_override")
    steps = settings.get("diffusion_steps_override")
    cfg = settings.get("cfg_override")

    if voice_name == settings.get("distilled_speaker"):
        if solver is None:
            solver = "single_step"
        if steps is None:
            steps = 1
        if cfg is None:
            cfg = 0.0

    if solver:
        body["solver_override"] = solver
    if steps is not None:
        body["diffusion_steps_override"] = steps
    if cfg is not None:
        body["inference_cfg_override"] = cfg
    return body


class IndexTTSHandler(AsyncEventHandler):
    """One handler per HA connection. Wyoming server spawns a new instance for
    each TCP client; we hold no per-handler state beyond what the protocol
    requires."""

    def __init__(self, *args, settings: dict, voices: list[TtsVoice], **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.voices = voices
        # ── Streaming-synthesis state (HA 2025.10+ synthesize-start/chunk/stop) ──
        # We split incoming text into sentences and pipeline their synthesis:
        # each sentence's API call runs concurrently in the background, while a
        # single writer task drains audio chunks to Wyoming in sentence order.
        # This eliminates inter-sentence gaps: by the time the writer finishes
        # emitting sentence N's audio, sentence N+1's API call is already in
        # flight (and may have data buffered) so N+1's first byte hits Wyoming
        # immediately.
        self._stream_voice: str | None = None
        self._stream_sentences: _SentenceStream | None = None
        # Outer FIFO of per-sentence chunk queues; the writer task drains in order.
        self._stream_pipeline_q: asyncio.Queue | None = None
        self._stream_writer_task: asyncio.Task | None = None
        # Limit concurrent API calls. The API is GPU-bound and effectively
        # serial, so big concurrency wastes uploads/memory; 2 = current + 1
        # warm lookahead, which is enough to fill any gap.
        self._stream_synth_sem: asyncio.Semaphore = asyncio.Semaphore(2)

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
        # Each completed sentence kicks off a background API call whose audio
        # chunks land in a per-sentence queue. A single writer task drains
        # those queues IN ORDER, emitting one AudioStart at the very first
        # byte and one AudioStop at end-of-stream.
        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._stream_voice = self._pick_voice(start.voice)
            self._stream_sentences = _SentenceStream()
            self._stream_pipeline_q = asyncio.Queue()
            self._stream_writer_task = asyncio.create_task(
                self._stream_writer(self._stream_pipeline_q)
            )
            LOG.info("[STREAM] synthesize-start  voice=%s", self._stream_voice)
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            if self._stream_sentences is None or self._stream_voice is None:
                LOG.warning("synthesize-chunk received outside a streaming session")
                return True
            for sentence in self._stream_sentences.feed(chunk.text):
                self._kickoff_sentence(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            sentences = self._stream_sentences
            pipeline_q = self._stream_pipeline_q
            writer_task = self._stream_writer_task

            # Flush trailing text and queue it as the final sentence.
            remaining = sentences.flush() if sentences else None
            LOG.info("[STREAM] synthesize-stop  trailing=%r",
                     (remaining or "")[:80])
            if remaining:
                self._kickoff_sentence(remaining)

            # Clear handler state so a fresh session can start cleanly even
            # while the writer is still draining.
            self._stream_voice = None
            self._stream_sentences = None
            self._stream_pipeline_q = None
            self._stream_writer_task = None

            # Signal end-of-stream and wait for writer to finish (it emits
            # AudioStop on its way out if any audio was actually written).
            if pipeline_q is not None:
                await pipeline_q.put(None)
            if writer_task is not None:
                try:
                    await writer_task
                except Exception:
                    LOG.exception("stream writer task failed")

            await self.write_event(SynthesizeStopped().event())
            LOG.info("[STREAM] sent SynthesizeStopped")
            return True

        # Unknown event — log it so we can see if HA is sending something we
        # don't yet handle.
        LOG.debug("[bridge] unhandled event type=%s", event.type)
        return True

    def _kickoff_sentence(self, sentence: str) -> None:
        """Fire-and-forget a synthesis task for `sentence` and register its
        per-sentence chunk queue with the pipeline writer. The chunk handler
        returns immediately; the writer drains queues in arrival order."""
        if self._stream_pipeline_q is None or self._stream_voice is None:
            LOG.warning("_kickoff_sentence with no active stream — dropping %r", sentence[:60])
            return
        chunks_q: asyncio.Queue = asyncio.Queue()
        # Stamp the queue with the sentence text so the writer can log it.
        chunks_q._sentence = sentence  # type: ignore[attr-defined]
        self._stream_pipeline_q.put_nowait(chunks_q)
        asyncio.create_task(
            self._synth_to_queue(sentence, self._stream_voice, chunks_q)
        )
        LOG.info("[STREAM] queued sentence  text=%r", sentence[:80])

    async def _synth_to_queue(
        self,
        sentence: str,
        voice_name: str,
        chunks_q: asyncio.Queue,
    ) -> None:
        """POST the sentence to the TTS API and push PCM pieces into chunks_q.
        Always terminates with a None sentinel so the writer doesn't hang."""
        try:
            async with self._stream_synth_sem:
                await self._do_post_and_queue(sentence, voice_name, chunks_q)
        except Exception:
            LOG.exception("synth_to_queue failed for sentence=%r", sentence[:60])
        finally:
            chunks_q.put_nowait(None)

    async def _do_post_and_queue(
        self,
        sentence: str,
        voice_name: str,
        chunks_q: asyncio.Queue,
    ) -> None:
        api_url = self.settings["api_url"]
        await _ensure_voice_loaded(api_url, voice_name)
        body = _build_request_body(sentence, voice_name, self.settings)
        ref_path = _resolve_reference_audio(voice_name, self.settings)
        files: dict = {"request_json": (None, json.dumps(body))}
        ref_fh = None
        if ref_path is not None:
            ref_fh = open(ref_path, "rb")
            files["audio_file"] = (ref_path.name, ref_fh, "audio/wav")
            body["cond_cache_key"] = str(ref_path)
            files["request_json"] = (None, json.dumps(body))
        client = await _client()
        LOG.info("Synthesize  voice=%s  ref=%s  text=%r",
                 voice_name, ref_path.name if ref_path else "<embeddings>", sentence[:80])
        try:
            async with client.stream(
                "POST", f"{api_url}/inference/stream?raw_pcm=true", files=files,
            ) as resp:
                if not resp.is_success:
                    body_text = (await resp.aread()).decode("utf-8", "replace")
                    LOG.error("stream %s: %s", resp.status_code, body_text[:200])
                    return
                leftover = b""
                async for raw in resp.aiter_bytes(chunk_size=WYOMING_CHUNK_BYTES):
                    if not raw:
                        continue
                    buf = leftover + raw
                    n_full = (len(buf) // SAMPLE_WIDTH) * SAMPLE_WIDTH
                    if n_full:
                        i = 0
                        while i < n_full:
                            piece = buf[i : i + WYOMING_CHUNK_BYTES]
                            chunks_q.put_nowait(piece)
                            i += len(piece)
                    leftover = buf[n_full:]
                if leftover:
                    LOG.warning("Trailing %d bytes of unaligned PCM dropped", len(leftover))
        finally:
            if ref_fh is not None:
                try: ref_fh.close()
                except Exception: pass

    async def _stream_writer(self, pipeline_q: asyncio.Queue) -> None:
        """Drain per-sentence chunk queues from `pipeline_q` in order and
        write AudioChunks to Wyoming. Emits one AudioStart at the first
        chunk and one AudioStop when the pipeline closes."""
        audio_started = False
        try:
            while True:
                chunks_q = await pipeline_q.get()
                if chunks_q is None:
                    break
                sentence = getattr(chunks_q, "_sentence", "")
                first_for_sentence = True
                while True:
                    piece = await chunks_q.get()
                    if piece is None:
                        break
                    if not audio_started:
                        await self.write_event(AudioStart(
                            rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS,
                        ).event())
                        audio_started = True
                        LOG.info("[STREAM] first AudioChunk → Wyoming")
                    if first_for_sentence:
                        LOG.info("[STREAM] writing audio for %r", sentence[:60])
                        first_for_sentence = False
                    await self.write_event(AudioChunk(
                        rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS, audio=piece,
                    ).event())
        finally:
            if audio_started:
                await self.write_event(AudioStop().event())
                LOG.info("[STREAM] AudioStop → Wyoming")

    async def _synthesize(
        self,
        text: str,
        voice_name: str,
    ) -> None:
        """Single-shot synthesis path: legacy `Synthesize` events. Emits a
        complete AudioStart → chunks → AudioStop session for this one text."""
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

    # Auto-detect which speaker (if any) has a distilled CFM student loaded.
    # Stash on settings so _build_request_body can apply single_step overrides
    # for that one voice. Querying once at startup keeps the per-request path
    # synchronous; restart the bridge if you activate a different student.
    settings["distilled_speaker"] = await _fetch_active_distilled_speaker(settings["api_url"])

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
