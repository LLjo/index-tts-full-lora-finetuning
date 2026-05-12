#!/usr/bin/env python3
"""
Side-by-side A/B eval between teacher CFM and distilled student.

Synthesizes the same text twice (once with teacher, once with student), saves
both wavs next to each other, computes mel RMSE between them, and times each.

Use this after training to decide whether the student is good enough to ship.
Especially important for the LoRA stutter voice — the roadmap calls out that
distillation tends to smooth fine acoustic details. Eyeball the mels and ears
on the audio before flipping the student into production.

Usage:
    python tools/ab_eval_cfm.py \\
        --audio-prompt examples/ozzy_ref.wav \\
        --text "The patterns were learned from the verbatim transcripts." \\
        --student-checkpoint training/ozzy/cfm_reflow_student/best.pth \\
        --student-steps 1 \\
        --teacher-steps 10 \\
        --teacher-solver heun \\
        --output-dir ab_results/ozzy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-prompt", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True,
                        help="Path to the distilled student CFM checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--student-steps", type=int, default=1)
    parser.add_argument("--student-solver", default="single_step", choices=["single_step", "euler", "heun"])
    parser.add_argument("--student-cfg", type=float, default=0.0)

    parser.add_argument("--teacher-steps", type=int, default=10)
    parser.add_argument("--teacher-solver", default="heun", choices=["euler", "heun"])
    parser.add_argument("--teacher-cfg", type=float, default=0.7)

    parser.add_argument("--seed", type=int, default=1234,
                        help="Same seed used for both runs so the GPT decode is identical "
                             "and only the CFM differs in the output.")
    return parser.parse_args()


def run_synthesis(tts, args, distilled_checkpoint, solver, steps, cfg_rate):
    from indextts.streaming_v2 import StreamingConfigV2, StreamingMode, streaming_inference_v2

    config = StreamingConfigV2(
        mode=StreamingMode.FAST_CHUNKS,
        min_chunk_tokens=30,
        chunk_tokens=40,
        max_chunk_tokens=100,
        first_chunk_diffusion_steps=steps,
        diffusion_steps=steps,
        first_chunk_cfg_rate=cfg_rate,
        inference_cfg_rate=cfg_rate,
        solver=solver,
        crossfade_samples=512,
        verbose=False,
    )

    torch.manual_seed(args.seed)
    chunks = []
    t0 = time.perf_counter()
    first_chunk_at = None
    for wav in streaming_inference_v2(
        tts=tts,
        text=args.text,
        audio_prompt=str(args.audio_prompt),
        config=config,
        temperature=0.8, top_p=0.8, top_k=30,
    ):
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter() - t0
        chunks.append(wav)
    return torch.cat(chunks, dim=-1).cpu(), first_chunk_at, time.perf_counter() - t0


def mel_rmse(wav_a: torch.Tensor, wav_b: torch.Tensor, sr: int = 22050) -> float:
    """RMSE between log-mel spectrograms of the two wavs (length-aligned).

    Crude but fast quality proxy: low RMSE means the student is acoustically
    close to the teacher. Don't trust this in isolation — listen to the audio.
    """
    from indextts.s2mel.modules.audio import mel_spectrogram
    n = min(wav_a.shape[-1], wav_b.shape[-1])
    a = wav_a[..., :n].float() / 32767.0
    b = wav_b[..., :n].float() / 32767.0
    mel_args = dict(
        n_fft=1024, win_size=1024, hop_size=256, num_mels=80,
        sampling_rate=sr, fmin=0, fmax=None, center=False,
    )
    mel_a = mel_spectrogram(a.unsqueeze(0) if a.dim() == 1 else a, **mel_args)
    mel_b = mel_spectrogram(b.unsqueeze(0) if b.dim() == 1 else b, **mel_args)
    n_frames = min(mel_a.shape[-1], mel_b.shape[-1])
    return torch.sqrt(((mel_a[..., :n_frames] - mel_b[..., :n_frames]) ** 2).mean()).item()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from indextts.infer_v2 import IndexTTS2

    print(">> Loading teacher (base s2mel only)")
    tts_teacher = IndexTTS2(
        cfg_path=str(PROJECT_ROOT / "checkpoints" / "config.yaml"),
        model_dir=str(PROJECT_ROOT / "checkpoints"),
        use_fp16=torch.cuda.is_available(),
        use_cuda_kernel=torch.cuda.is_available(),
        use_accel=True,
        use_torch_compile=False,
        s2mel_distilled_checkpoint=None,
    )

    print(f"\n>> [TEACHER] solver={args.teacher_solver} steps={args.teacher_steps} cfg={args.teacher_cfg}")
    wav_teacher, ttfa_t, total_t = run_synthesis(
        tts_teacher, args,
        distilled_checkpoint=None,
        solver=args.teacher_solver,
        steps=args.teacher_steps,
        cfg_rate=args.teacher_cfg,
    )
    teacher_path = args.output_dir / "teacher.wav"
    torchaudio.save(str(teacher_path), wav_teacher.to(torch.int16), 22050)
    print(f"   TTFA: {ttfa_t * 1000:.1f} ms")
    print(f"   Total: {total_t * 1000:.1f} ms")
    print(f"   Audio: {wav_teacher.shape[-1] / 22050:.2f} s → {teacher_path}")

    del tts_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n>> Loading student (with distilled CFM overlay)")
    tts_student = IndexTTS2(
        cfg_path=str(PROJECT_ROOT / "checkpoints" / "config.yaml"),
        model_dir=str(PROJECT_ROOT / "checkpoints"),
        use_fp16=torch.cuda.is_available(),
        use_cuda_kernel=torch.cuda.is_available(),
        use_accel=True,
        use_torch_compile=False,
        s2mel_distilled_checkpoint=str(args.student_checkpoint),
    )

    print(f"\n>> [STUDENT] solver={args.student_solver} steps={args.student_steps} cfg={args.student_cfg}")
    wav_student, ttfa_s, total_s = run_synthesis(
        tts_student, args,
        distilled_checkpoint=str(args.student_checkpoint),
        solver=args.student_solver,
        steps=args.student_steps,
        cfg_rate=args.student_cfg,
    )
    student_path = args.output_dir / "student.wav"
    torchaudio.save(str(student_path), wav_student.to(torch.int16), 22050)
    print(f"   TTFA: {ttfa_s * 1000:.1f} ms")
    print(f"   Total: {total_s * 1000:.1f} ms")
    print(f"   Audio: {wav_student.shape[-1] / 22050:.2f} s → {student_path}")

    rmse = mel_rmse(wav_teacher, wav_student)

    print("\n=== A/B comparison ===")
    print(f"  TTFA   teacher={ttfa_t * 1000:.1f} ms   student={ttfa_s * 1000:.1f} ms   diff={(ttfa_t - ttfa_s) * 1000:+.1f} ms")
    print(f"  Total  teacher={total_t * 1000:.1f} ms  student={total_s * 1000:.1f} ms  diff={(total_t - total_s) * 1000:+.1f} ms")
    print(f"  Mel RMSE: {rmse:.3f}  (lower = student closer to teacher)")
    print(f"  Files: {teacher_path}  vs  {student_path}")
    print("\nListen to both. The mel RMSE is a sanity number, not a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
