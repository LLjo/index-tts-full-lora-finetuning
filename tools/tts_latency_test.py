#!/usr/bin/env python3
"""TTS playback latency A/B test.

Serves a tiny HTML page with two buttons:
  A. <audio src=URL>             — what voice-satellite-card-integration does
  B. fetch + AudioContext        — the proposed streaming path

Both buttons fetch the SAME streaming WAV URL and time how long it takes
from button-click to the first audible sample. Result decides whether the
~2s playback delay is in the <audio> element's pre-roll buffer (in which
case the AudioContext approach is worth implementing in the integration).

Usage:
    ./run uv run python tools/tts_latency_test.py
Then open http://localhost:9000 in any browser (including the tablet that
runs HA — just substitute the IP). Type text, click each button, watch the
log.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEXTTS_URL = os.environ.get("INDEXTTS_API_URL", "http://localhost:8000")
DEFAULT_VOICE = os.environ.get("INDEXTTS_DEFAULT_SPEAKER", "ozzyv6")


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TTS playback latency test</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 30px auto; padding: 16px; color: #222; }
  h1 { font-size: 20px; }
  textarea, button, input { font-size: 16px; padding: 10px; box-sizing: border-box; }
  textarea { width: 100%; height: 70px; }
  input { width: 100%; }
  button { display: block; width: 100%; margin-top: 8px; cursor: pointer; }
  #log { background: #111; color: #ddd; padding: 12px; white-space: pre-wrap; font-family: monospace;
         font-size: 13px; max-height: 360px; overflow: auto; margin-top: 12px; border-radius: 4px; }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  .row > * { flex: 1; }
  small { color: #666; }
</style>
</head>
<body>
<h1>TTS playback latency test</h1>
<p><small>Both buttons GET the same streaming WAV. The number to compare is
<b>time-to-first-audible-sample</b>.</small></p>

<label>Text:</label>
<textarea id="text">Sure! Here's a quick recipe. Mix flour, sugar, and baking powder. Add an egg and milk. Cook on a greased pan for two to three minutes per side. Serve warm.</textarea>

<label style="margin-top:8px; display:block;">Voice:</label>
<input id="voice" value="__DEFAULT_VOICE__" />

<div class="row">
  <button id="btnA">A: &lt;audio src=URL&gt; (current behavior)</button>
  <button id="btnB">B: fetch + AudioContext (proposed)</button>
</div>
<button id="btnStop" style="background:#fee;">Stop</button>

<div id="log"></div>

<script>
const logEl = document.getElementById('log');
function log(msg) {
  const t = new Date().toISOString().slice(11, 23);
  logEl.textContent += `[${t}] ${msg}\\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

let activeA = null;       // HTMLAudioElement
let activeB = null;       // { stop }

function buildUrl() {
  const text = document.getElementById('text').value;
  const voice = document.getElementById('voice').value;
  // Cachebust each click so neither path benefits from cached bytes.
  return `/tts?text=${encodeURIComponent(text)}&voice=${encodeURIComponent(voice)}&_=${Date.now()}`;
}

// ── A. <audio src=URL> ──────────────────────────────────────────────────
document.getElementById('btnA').onclick = async () => {
  stopAll();
  const url = buildUrl();
  log('A: clicked. starting fetch via <audio src>.');
  const t0 = performance.now();
  const audio = new Audio();
  activeA = audio;
  audio.addEventListener('playing', () => {
    log(`A: <audio> 'playing' event @ ${(performance.now() - t0).toFixed(0)} ms`);
  }, { once: true });
  audio.addEventListener('canplay', () => {
    log(`A: <audio> 'canplay' event @ ${(performance.now() - t0).toFixed(0)} ms`);
  }, { once: true });
  audio.addEventListener('ended', () => {
    log(`A: <audio> 'ended' @ ${(performance.now() - t0).toFixed(0)} ms (total)`);
  }, { once: true });
  audio.addEventListener('error', (e) => log(`A: <audio> error: ${audio.error?.message || e}`));
  audio.src = url;
  try { await audio.play(); }
  catch (e) { log('A: play() rejected: ' + e); }
};

// ── B. fetch + AudioContext (low-latency) ───────────────────────────────
document.getElementById('btnB').onclick = async () => {
  stopAll();
  const url = buildUrl();
  log('B: clicked. starting fetch + AudioContext stream.');
  const t0 = performance.now();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  if (ctx.state === 'suspended') await ctx.resume();
  const abort = new AbortController();
  let playHead = -1, firstFired = false, stopped = false;
  let sampleRate = 0, channels = 0, bitsPerSample = 0;
  let headerParsed = false;
  let leftover = new Uint8Array(0);

  activeB = { stop: () => { stopped = true; abort.abort(); ctx.close().catch(() => {}); } };

  function parseHeader(bytes) {
    if (bytes.length < 44) return -1;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (dv.getUint32(0, false) !== 0x52494646) throw new Error('not RIFF');
    if (dv.getUint32(8, false) !== 0x57415645) throw new Error('not WAVE');
    let off = 12;
    while (off + 8 <= bytes.length) {
      const id = dv.getUint32(off, false);
      const size = dv.getUint32(off + 4, true);
      if (id === 0x666d7420) {
        channels = dv.getUint16(off + 10, true);
        sampleRate = dv.getUint32(off + 12, true);
        bitsPerSample = dv.getUint16(off + 22, true);
        off += 8 + size;
      } else if (id === 0x64617461) {
        log(`B: WAV header parsed: ${sampleRate} Hz, ${channels} ch, ${bitsPerSample}-bit @ ${(performance.now() - t0).toFixed(0)} ms`);
        return off + 8;
      } else {
        off += 8 + size;
      }
    }
    return -1;
  }

  function schedulePcm(pcm) {
    if (stopped || bitsPerSample !== 16) return;
    const frames = pcm.length / 2 / channels;
    if (frames === 0) return;
    const buf = ctx.createBuffer(channels, frames, sampleRate);
    const dv = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (let c = 0; c < channels; c++) {
      const out = buf.getChannelData(c);
      for (let i = 0; i < frames; i++) {
        out[i] = dv.getInt16((i * channels + c) * 2, true) / 32768;
      }
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const now = ctx.currentTime;
    if (playHead < 0) { playHead = now + 0.02; }
    else if (playHead < now) { playHead = now; }
    src.start(playHead);
    if (!firstFired) {
      firstFired = true;
      const delayMs = (playHead - now) * 1000;
      setTimeout(() => {
        log(`B: first sample audible @ ${(performance.now() - t0).toFixed(0)} ms`);
      }, delayMs);
    }
    playHead += buf.duration;
  }

  try {
    const resp = await fetch(url, { signal: abort.signal });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    log(`B: response headers received @ ${(performance.now() - t0).toFixed(0)} ms (type=${resp.headers.get('content-type')})`);
    const reader = resp.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done || stopped) break;
      const merged = new Uint8Array(leftover.length + value.length);
      merged.set(leftover, 0); merged.set(value, leftover.length);
      leftover = merged;
      if (!headerParsed) {
        const off = parseHeader(leftover);
        if (off < 0) continue;
        headerParsed = true;
        leftover = leftover.subarray(off);
      }
      const frameBytes = (bitsPerSample / 8) * channels;
      const whole = Math.floor(leftover.length / frameBytes) * frameBytes;
      if (whole > 0) {
        schedulePcm(leftover.subarray(0, whole));
        leftover = leftover.subarray(whole);
      }
    }
    log(`B: stream ended @ ${(performance.now() - t0).toFixed(0)} ms (total bytes done)`);
  } catch (e) {
    if (e.name === 'AbortError') return;
    log('B: error: ' + e.message);
  }
};

function stopAll() {
  if (activeA) { try { activeA.pause(); activeA.removeAttribute('src'); activeA.load(); } catch {} activeA = null; }
  if (activeB) { try { activeB.stop(); } catch {} activeB = null; }
}
document.getElementById('btnStop').onclick = () => { stopAll(); log('stopped.'); };
</script>
</body>
</html>
"""


def _make_app(indextts_url: str) -> FastAPI:
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML.replace("__DEFAULT_VOICE__", DEFAULT_VOICE)

    @app.get("/tts")
    async def tts(
        text: str = Query(...),
        voice: str = Query(DEFAULT_VOICE),
    ):
        """Proxy GET → POST. IndexTTS /inference/stream expects a multipart
        POST; the browser needs a streamable GET. We bridge them and forward
        chunks unchanged so the browser sees a real chunked WAV stream."""
        speaker_dir = PROJECT_ROOT / "training" / voice / "dataset" / "audio"
        ref_path: Optional[Path] = None
        if speaker_dir.exists():
            for p in sorted(speaker_dir.iterdir()):
                if p.suffix.lower() in {".wav", ".mp3", ".flac"}:
                    ref_path = p
                    break

        body = {
            "text": text,
            "speaker": voice,
            "use_patterns": True,
            "streaming_preset": "fast_quality",
        }
        files = {"request_json": (None, json.dumps(body))}
        ref_fh = None
        if ref_path is not None:
            ref_fh = open(ref_path, "rb")
            files["audio_file"] = (ref_path.name, ref_fh, "audio/wav")
            body["cond_cache_key"] = str(ref_path)
            files["request_json"] = (None, json.dumps(body))

        async def stream_response():
            timeout = httpx.Timeout(connect=5, read=600, write=10, pool=10)
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    async with client.stream(
                        "POST",
                        f"{indextts_url}/inference/stream",
                        files=files,
                    ) as resp:
                        if not resp.is_success:
                            body_text = await resp.aread()
                            yield f"[ERROR] upstream {resp.status_code}: {body_text!r}".encode()
                            return
                        async for chunk in resp.aiter_bytes(chunk_size=4096):
                            yield chunk
                finally:
                    if ref_fh is not None:
                        try: ref_fh.close()
                        except Exception: pass

        return StreamingResponse(stream_response(), media_type="audio/wav")

    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--indextts-url", default=DEFAULT_INDEXTTS_URL)
    args = p.parse_args()

    import uvicorn
    print(f"\n  🧪 Open http://localhost:{args.port} (or http://<this-host>:{args.port})", flush=True)
    print(f"     Proxying POSTs to {args.indextts_url}", flush=True)
    print(f"     Default voice: {DEFAULT_VOICE}\n", flush=True)

    uvicorn.run(_make_app(args.indextts_url), host="0.0.0.0", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
