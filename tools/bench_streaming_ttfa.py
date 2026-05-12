"""
Client-side TTFA benchmark for the /inference/stream endpoint.

Measures wall-clock time from request send to first audio byte received, across a
set of streaming presets. Use this to verify Phase 1 latency wins against a running
API server.

Usage:
    # Pattern speaker (uses stored speaker embeddings, no audio_file needed)
    python tools/bench_streaming_ttfa.py \\
        --url http://localhost:8000/inference/stream \\
        --speaker myspeaker --use-patterns \\
        --text "Hello, this is a streaming latency test." \\
        --presets ultra_fast fast balanced \\
        --runs 5

    # Plain reference audio path
    python tools/bench_streaming_ttfa.py \\
        --url http://localhost:8000/inference/stream \\
        --audio path/to/speaker.wav \\
        --text "Hello, this is a streaming latency test."

The first run per preset is treated as a warmup and excluded from the reported
median, because torch.compile / kernel-init / KV-cache allocation only happens once.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: `requests` is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


# WAV header is 44 bytes; first audio sample starts at byte 45. We treat "first
# non-header byte received" as the TTFA signal.
WAV_HEADER_BYTES = 44


def stream_once(
    url: str,
    text: str,
    preset: str,
    speaker: Optional[str],
    use_patterns: bool,
    audio_path: Optional[Path],
    verbose: bool,
    timeout: float,
) -> dict:
    """Hit the streaming endpoint once and return timing measurements."""
    request_json = {
        "text": text,
        "speaker": speaker,
        "use_patterns": use_patterns,
        "streaming_preset": preset,
        "verbose": verbose,
    }

    files = {"request_json": (None, json.dumps(request_json))}
    if audio_path is not None:
        files["audio_file"] = (audio_path.name, audio_path.read_bytes(), "audio/wav")

    t0 = time.perf_counter()
    resp = requests.post(url, files=files, stream=True, timeout=timeout)
    resp.raise_for_status()

    ttfa_header = None
    ttfa_audio = None
    bytes_received = 0
    total_audio_bytes = 0

    for chunk in resp.iter_content(chunk_size=512):
        if not chunk:
            continue
        now = time.perf_counter()
        prev = bytes_received
        bytes_received += len(chunk)

        if ttfa_header is None:
            ttfa_header = now - t0

        # First byte past the WAV header is "first audio sample"
        if ttfa_audio is None and bytes_received > WAV_HEADER_BYTES:
            ttfa_audio = now - t0

        if prev >= WAV_HEADER_BYTES:
            total_audio_bytes += len(chunk)
        elif bytes_received > WAV_HEADER_BYTES:
            total_audio_bytes += bytes_received - WAV_HEADER_BYTES

    total_time = time.perf_counter() - t0
    # 22050 Hz, mono, int16 → 44100 bytes per second of audio.
    audio_seconds = total_audio_bytes / 44100.0

    return {
        "preset": preset,
        "ttfa_header_ms": (ttfa_header or 0) * 1000,
        "ttfa_audio_ms": (ttfa_audio or 0) * 1000,
        "total_time_ms": total_time * 1000,
        "audio_seconds": audio_seconds,
        "rtf": (total_time / audio_seconds) if audio_seconds > 0 else float("inf"),
    }


def run_preset(args, preset: str) -> list[dict]:
    results = []
    audio_path = Path(args.audio) if args.audio else None

    print(f"\n=== preset: {preset} ===")
    for i in range(args.runs):
        is_warmup = i == 0 and args.runs > 1
        tag = "warmup" if is_warmup else f"run {i}"
        try:
            r = stream_once(
                url=args.url,
                text=args.text,
                preset=preset,
                speaker=args.speaker,
                use_patterns=args.use_patterns,
                audio_path=audio_path,
                verbose=args.server_verbose,
                timeout=args.timeout,
            )
        except requests.HTTPError as e:
            print(f"  {tag}: HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return results
        except Exception as e:
            print(f"  {tag}: ERROR {type(e).__name__}: {e}")
            return results

        print(
            f"  {tag:>8}: TTFA(audio)={r['ttfa_audio_ms']:.1f} ms  "
            f"TTFA(headers)={r['ttfa_header_ms']:.1f} ms  "
            f"total={r['total_time_ms']:.1f} ms  "
            f"audio={r['audio_seconds']:.2f}s  RTF={r['rtf']:.2f}"
        )
        r["is_warmup"] = is_warmup
        results.append(r)

    return results


def summarize(all_results: dict[str, list[dict]]) -> None:
    print("\n=== summary (excluding warmup runs) ===")
    print(f"{'preset':<12} {'TTFA_audio(ms)':>16} {'total(ms)':>12} {'RTF':>7}")
    for preset, runs in all_results.items():
        scored = [r for r in runs if not r.get("is_warmup")]
        if not scored:
            print(f"{preset:<12} (no successful non-warmup runs)")
            continue
        ttfas = [r["ttfa_audio_ms"] for r in scored]
        totals = [r["total_time_ms"] for r in scored]
        rtfs = [r["rtf"] for r in scored if r["rtf"] != float("inf")]
        print(
            f"{preset:<12} "
            f"{statistics.median(ttfas):>16.1f} "
            f"{statistics.median(totals):>12.1f} "
            f"{(statistics.median(rtfs) if rtfs else 0):>7.2f}"
        )


def warmup_pipeline(args) -> None:
    """Hit the warmup endpoint once so CUDA graph capture / JIT happen before timing."""
    warmup_url = args.url.rsplit("/inference/", 1)[0] + "/inference/warmup"
    payload = {
        "speaker": args.speaker,
        "use_patterns": args.use_patterns,
        "text": "Warmup.",
        "streaming_preset": args.presets[0],
    }
    files = {"request_json": (None, json.dumps(payload))}
    if args.audio:
        ap = Path(args.audio)
        files["audio_file"] = (ap.name, ap.read_bytes(), "audio/wav")
    print(f"Warming up via {warmup_url} ...")
    try:
        r = requests.post(warmup_url, files=files, timeout=args.timeout)
        r.raise_for_status()
        info = r.json()
        print(
            f"  warmup done: TTFA={info.get('ttfa_ms')}ms "
            f"total={info.get('total_time_ms')}ms "
            f"chunks={info.get('chunks')} "
            f"audio={info.get('audio_seconds')}s"
        )
    except Exception as e:
        print(f"  warmup failed (continuing anyway): {e}")


def diagnostics_once(args, preset: str) -> dict:
    """Hit /inference/stream/diagnostics and get a JSON timing breakdown."""
    diag_url = args.url.rsplit("/inference/", 1)[0] + "/inference/stream/diagnostics"
    payload = {
        "text": args.text,
        "speaker": args.speaker,
        "use_patterns": args.use_patterns,
        "streaming_preset": preset,
        "verbose": args.server_verbose,
    }
    files = {"request_json": (None, json.dumps(payload))}
    if args.audio:
        ap = Path(args.audio)
        files["audio_file"] = (ap.name, ap.read_bytes(), "audio/wav")
    r = requests.post(diag_url, files=files, timeout=args.timeout)
    r.raise_for_status()
    return r.json()


def run_preset_diagnostics(args, preset: str) -> list[dict]:
    print(f"\n=== preset: {preset} (diagnostics) ===")
    results = []
    for i in range(args.runs):
        is_warmup = i == 0 and args.runs > 1
        tag = "warmup" if is_warmup else f"run {i}"
        try:
            info = diagnostics_once(args, preset)
        except requests.HTTPError as e:
            print(f"  {tag}: HTTP {e.response.status_code}: {e.response.text[:200]}")
            return results
        except Exception as e:
            print(f"  {tag}: ERROR {type(e).__name__}: {e}")
            return results
        print(
            f"  {tag:>8}: TTFA={info['ttfa_ms']} ms  "
            f"total={info['total_time_ms']} ms  "
            f"chunks={info['chunk_count']}  "
            f"audio={info['audio_seconds']}s  "
            f"RTF={info['rtf']}  "
            f"accel={info['accel_engine_active']}"
        )
        info["is_warmup"] = is_warmup
        info["preset"] = preset
        info["ttfa_audio_ms"] = info["ttfa_ms"]  # column-compat with audio-byte path
        info["total_time_ms"] = info["total_time_ms"]
        info["rtf"] = info["rtf"]
        results.append(info)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000/inference/stream")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--audio", help="Path to speaker reference WAV (when not using patterns)")
    p.add_argument("--speaker", help="Speaker name (required with --use-patterns)")
    p.add_argument("--use-patterns", action="store_true", help="Use trained pattern embeddings")
    p.add_argument("--presets", nargs="+",
                   default=["ultra_fast", "fast", "balanced"],
                   help="Presets to benchmark")
    p.add_argument("--runs", type=int, default=5,
                   help="Runs per preset; first is treated as warmup")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--server-verbose", action="store_true",
                   help="Ask the server to log per-stage timings (watch the server stdout)")
    p.add_argument("--mode", choices=["diagnostics", "audio"], default="diagnostics",
                   help="Use the JSON /inference/stream/diagnostics endpoint (default) "
                        "or stream raw audio and time the first byte")
    p.add_argument("--no-warmup-endpoint", action="store_true",
                   help="Skip hitting /inference/warmup before benchmarking")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.use_patterns and not args.speaker:
        print("ERROR: --use-patterns requires --speaker", file=sys.stderr)
        return 2
    if not args.use_patterns and not args.audio:
        print("ERROR: provide --audio when not using --use-patterns", file=sys.stderr)
        return 2

    if not args.no_warmup_endpoint:
        warmup_pipeline(args)

    all_results = {}
    runner = run_preset_diagnostics if args.mode == "diagnostics" else run_preset
    for preset in args.presets:
        all_results[preset] = runner(args, preset)

    summarize(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
