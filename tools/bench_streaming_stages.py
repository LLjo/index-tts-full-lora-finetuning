"""Per-stage timing benchmark across streaming presets.

Hits /inference/stream/diagnostics for each preset, runs N times, drops the
first as warmup, prints a stage-by-stage median table so you can see exactly
where TTFA is spent (cond extract / GPT prefill / CFM / BigVGAN / yield).

Usage:
    python tools/bench_streaming_stages.py \\
        --url http://localhost:8000 \\
        --speaker ozzy --use-patterns \\
        --presets ultra_fast ultra_fast_distilled balanced balanced_distilled \\
        --runs 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import requests


STAGE_ORDER = [
    "request_start_ms",
    "conditioning_done_ms",
    "threads_starting_ms",
    "gpt_first_token_ms",
    "chunk1_dispatched_ms",
    "chunk1_synth_start_ms",
    "chunk1_gpt_latent_done_ms",
    "chunk1_length_reg_done_ms",
    "chunk1_cfm_done_ms",
    "chunk1_bigvgan_done_ms",
    "chunk1_synth_done_ms",
    "chunk1_yielded_ms",
]


def call_diagnostics(
    base_url: str,
    text: str,
    preset: str,
    speaker: Optional[str],
    use_patterns: bool,
    audio_path: Optional[Path],
    timeout: float,
) -> dict:
    request_json = {
        "text": text,
        "streaming_preset": preset,
        "verbose": False,
    }
    if speaker:
        request_json["speaker"] = speaker
    if use_patterns:
        request_json["use_patterns"] = True

    files = {}
    data = {"request_json": json.dumps(request_json)}
    if audio_path is not None:
        files["audio_file"] = (audio_path.name, audio_path.open("rb"), "audio/wav")

    r = requests.post(
        f"{base_url.rstrip('/')}/inference/stream/diagnostics",
        data=data,
        files=files if files else None,
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"diagnostics({preset}) → {r.status_code}: {r.text[:300]}")
    return r.json()


def warm_all(base_url: str, presets: list[str], speaker: Optional[str], use_patterns: bool, audio_path: Optional[Path]) -> None:
    """Run the warmup endpoint once for the full preset list to capture CUDA graphs."""
    request_json = {
        "text": "Hello, this is a warmup pass to capture CUDA graphs and just-in-time compile kernels for streaming.",
        "presets": presets,
    }
    if speaker:
        request_json["speaker"] = speaker
    if use_patterns:
        request_json["use_patterns"] = True

    data = {"request_json": json.dumps(request_json)}
    files = {}
    if audio_path is not None:
        files["audio_file"] = (audio_path.name, audio_path.open("rb"), "audio/wav")

    print(f"[warmup] warming {len(presets)} presets (this can take 60-120s on first call)...")
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url.rstrip('/')}/inference/warmup",
        data=data,
        files=files if files else None,
        timeout=600,
    )
    dt = time.perf_counter() - t0
    if not r.ok:
        print(f"[warmup] FAILED {r.status_code}: {r.text[:300]}", file=sys.stderr)
    else:
        print(f"[warmup] done in {dt:.1f}s")


def pct_delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None or a == 0:
        return "    n/a"
    return f"{(b - a) / a * 100:+6.1f}%"


def fmt(v: Optional[float]) -> str:
    return f"{v:7.1f}" if v is not None else "      —"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--text", default="The quick brown fox jumps over the lazy dog while we measure first-audio latency on every streaming preset.")
    p.add_argument("--speaker", default=None)
    p.add_argument("--use-patterns", action="store_true")
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--presets", nargs="+", default=["ultra_fast", "ultra_fast_distilled", "balanced", "balanced_distilled"])
    p.add_argument("--runs", type=int, default=4, help="Total runs per preset; first is dropped as warmup")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--skip-warmup", action="store_true")
    args = p.parse_args()

    if not args.use_patterns and args.audio is None:
        print("ERROR: need either --use-patterns + --speaker or --audio", file=sys.stderr)
        return 2

    if not args.skip_warmup:
        warm_all(args.url, args.presets, args.speaker, args.use_patterns, args.audio)

    per_preset_runs: dict[str, list[dict]] = {p: [] for p in args.presets}
    for preset in args.presets:
        print(f"\n=== {preset} ===")
        for i in range(args.runs):
            tag = "warmup" if i == 0 else f"run {i}"
            t0 = time.perf_counter()
            try:
                result = call_diagnostics(args.url, args.text, preset, args.speaker, args.use_patterns, args.audio, args.timeout)
            except Exception as e:
                print(f"  {tag}: ERROR {e}")
                continue
            wall = (time.perf_counter() - t0) * 1000
            print(f"  {tag}: TTFA={result.get('ttfa_ms')}ms total={result.get('total_time_ms')}ms RTF={result.get('rtf')} wall={wall:.0f}ms")
            if i > 0:
                per_preset_runs[preset].append(result)

    # Build a stage-level median table.
    print("\n\n=== stage-level medians (ms from request start) ===")
    header = f"{'stage':<32}" + "".join(f"{p:>16}" for p in args.presets)
    print(header)
    print("-" * len(header))
    medians: dict[str, dict[str, Optional[float]]] = {}
    for preset in args.presets:
        runs = per_preset_runs[preset]
        if not runs:
            medians[preset] = {s: None for s in STAGE_ORDER}
            continue
        stage_med: dict[str, Optional[float]] = {}
        for stage in STAGE_ORDER:
            vals = [r["stages"].get(stage) for r in runs if r.get("stages", {}).get(stage) is not None]
            stage_med[stage] = statistics.median(vals) if vals else None
        medians[preset] = stage_med

    for stage in STAGE_ORDER:
        row = f"{stage:<32}"
        for preset in args.presets:
            row += f"{fmt(medians[preset].get(stage)):>16}"
        print(row)

    # Per-stage delta (chunk1 internals) — how much time does each phase take?
    print("\n=== chunk-1 phase deltas (ms each phase took) ===")
    phase_pairs = [
        ("setup", "request_start_ms", "conditioning_done_ms"),
        ("threads_init", "conditioning_done_ms", "threads_starting_ms"),
        ("gpt_first_tok", "threads_starting_ms", "gpt_first_token_ms"),
        ("gpt_to_dispatch", "gpt_first_token_ms", "chunk1_dispatched_ms"),
        ("dispatch_to_synth", "chunk1_dispatched_ms", "chunk1_synth_start_ms"),
        ("gpt_latent", "chunk1_synth_start_ms", "chunk1_gpt_latent_done_ms"),
        ("length_reg", "chunk1_gpt_latent_done_ms", "chunk1_length_reg_done_ms"),
        ("cfm", "chunk1_length_reg_done_ms", "chunk1_cfm_done_ms"),
        ("bigvgan", "chunk1_cfm_done_ms", "chunk1_bigvgan_done_ms"),
        ("synth_done_to_yield", "chunk1_synth_done_ms", "chunk1_yielded_ms"),
    ]
    header = f"{'phase':<22}" + "".join(f"{p:>16}" for p in args.presets)
    print(header)
    print("-" * len(header))
    for name, start_key, end_key in phase_pairs:
        row = f"{name:<22}"
        for preset in args.presets:
            s = medians[preset].get(start_key)
            e = medians[preset].get(end_key)
            row += f"{fmt(e - s) if s is not None and e is not None else '      —':>16}"
        print(row)

    # Top-line summary.
    print("\n=== TTFA + RTF summary (median across non-warmup runs) ===")
    print(f"{'preset':<28}{'TTFA(ms)':>12}{'total(ms)':>12}{'RTF':>8}{'chunks':>9}")
    for preset in args.presets:
        runs = per_preset_runs[preset]
        if not runs:
            print(f"{preset:<28}{'n/a':>12}")
            continue
        ttfa = statistics.median([r["ttfa_ms"] for r in runs if r.get("ttfa_ms") is not None])
        tot = statistics.median([r["total_time_ms"] for r in runs if r.get("total_time_ms") is not None])
        rtf_vals = [r["rtf"] for r in runs if r.get("rtf") is not None]
        rtf = statistics.median(rtf_vals) if rtf_vals else float("nan")
        chunks = statistics.median([r["chunk_count"] for r in runs if r.get("chunk_count") is not None])
        print(f"{preset:<28}{ttfa:>12.1f}{tot:>12.1f}{rtf:>8.2f}{int(chunks):>9d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
