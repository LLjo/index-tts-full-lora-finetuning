#!/usr/bin/env python3
"""
Smoke test for the Phase 3 distilled-CFM inference plumbing.

Goal: verify the integration plumbing (checkpoint overlay, single_step solver,
config flag, end-to-end inference) works BEFORE we spend GPU time on training.
Loads the teacher snapshot AS IF it were a student and runs single-step inference.

What "success" means here:
- The model loads with the distilled-checkpoint overlay applied.
- An inference with solver="single_step" completes without exceptions.
- An output wav is written.
- The audio will sound LIKE GARBAGE — that's the expected behavior because the
  unmodified teacher's flow is not straight, so a single Euler step from z to t=1
  gives noise. This test is about the pipeline, not the audio quality.

After this passes, you can confidently spend GPU hours on real reflow training
knowing the inference side will work.

Usage:
    # First snapshot the teacher (one-time)
    python tools/snapshot_cfm_teacher.py

    # Then run this
    python tools/smoke_test_distilled_pipeline.py \\
        --audio-prompt examples/some_reference.wav \\
        --text "This is a smoke test."
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-prompt", type=Path, required=True, help="Reference audio for the speaker.")
    parser.add_argument("--text", type=str, default="This is a smoke test of the distilled pipeline.")
    parser.add_argument(
        "--distilled-checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "s2mel_teacher.pth",
        help="Path to the distilled CFM checkpoint. Defaults to the teacher snapshot — "
             "remember the audio will be garbage with the teacher; this only validates plumbing.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "smoke_test_output.wav")
    parser.add_argument("--steps", type=int, default=1, help="Use 1 for single_step, more for sanity comparison.")
    args = parser.parse_args()

    if not args.distilled_checkpoint.exists():
        print(
            f"ERROR: distilled checkpoint not found: {args.distilled_checkpoint}\n"
            f"       Run `python tools/snapshot_cfm_teacher.py` first.",
            file=sys.stderr,
        )
        return 1

    from indextts.infer_v2 import IndexTTS2

    print(">> Loading IndexTTS2 with distilled CFM overlay")
    t_load = time.perf_counter()
    tts = IndexTTS2(
        cfg_path=str(PROJECT_ROOT / "checkpoints" / "config.yaml"),
        model_dir=str(PROJECT_ROOT / "checkpoints"),
        use_fp16=torch.cuda.is_available(),
        use_cuda_kernel=torch.cuda.is_available(),
        use_accel=False,  # skip accel for smoke test simplicity
        use_torch_compile=False,
        s2mel_distilled_checkpoint=str(args.distilled_checkpoint),
    )
    print(f">> Model load: {time.perf_counter() - t_load:.2f}s")

    print(">> Running single-step inference (output will be garbage with teacher overlay — that's fine)")

    # Reach in via the streaming generator with the single_step solver.
    from indextts.streaming_v2 import (
        StreamingConfigV2,
        StreamingMode,
        streaming_inference_v2,
    )

    config = StreamingConfigV2(
        mode=StreamingMode.FAST_CHUNKS,
        min_chunk_tokens=30,
        chunk_tokens=40,
        max_chunk_tokens=100,
        first_chunk_diffusion_steps=args.steps,
        diffusion_steps=args.steps,
        first_chunk_cfg_rate=0.0,
        inference_cfg_rate=0.0,  # CFG generally not used with single-step distilled students
        solver="single_step",
        crossfade_samples=512,
        verbose=True,
    )

    chunks = []
    t0 = time.perf_counter()
    first_chunk_at = None
    for wav in streaming_inference_v2(
        tts=tts,
        text=args.text,
        audio_prompt=str(args.audio_prompt),
        config=config,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter() - t0
        chunks.append(wav)
    t_total = time.perf_counter() - t0

    if not chunks:
        print("ERROR: zero chunks produced — pipeline broke before any audio.", file=sys.stderr)
        return 1

    full_wav = torch.cat(chunks, dim=-1).cpu().to(torch.int16)
    import torchaudio
    torchaudio.save(str(args.output), full_wav, 22050)

    audio_secs = full_wav.shape[-1] / 22050.0
    print(f">> Smoke test passed:")
    print(f"   Chunks: {len(chunks)}")
    print(f"   TTFA: {first_chunk_at * 1000:.1f} ms")
    print(f"   Total: {t_total * 1000:.1f} ms")
    print(f"   Audio duration: {audio_secs:.2f} s")
    print(f"   Output: {args.output} (expected to sound bad — pipeline OK, model is teacher-init not student-trained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
