#!/usr/bin/env python3
"""
Generate paired (z, x_final) training data for Phase 3 reflow distillation.

What it produces:
    For each input (text, audio_prompt) pair we run the full TTS pipeline
    end-to-end with the teacher CFM at high step count (default 50 Euler steps,
    CFG=0.7 by default). We hook the CFM's inference call to capture:
        z         — initial noise [1, 80, T]
        mu        — concat conditioning [1, T, 512]
        prompt    — reference mel [1, 80, T_prompt]
        style     — style vec [1, 192]
        x_lens    — [T]
        x_final   — teacher mel output [1, 80, T]

    Each pair is saved as one .pt file. Use `tools/train_cfm_reflow.py` to train.

Manifest format (JSONL, one record per line):
    {"id": "utt_001", "audio_prompt": "...wav", "text": "Hello world"}
    {"id": "utt_002", "audio_prompt": "...wav", "text": "Another utterance"}

Optional fields:
    "speaker_embeddings": "path/to/speaker_embeddings.pt"  # skip audio extraction
    "n_samples": 4   # how many z draws per conditioning; default --n-samples
    "max_mel_tokens": 600

Why per-conditioning multi-sample: reflow learns the trajectory from each z to
its corresponding x_final. Drawing several z's per conditioning gives the
student more coverage of the noise distribution for the SAME mu/prompt/style —
typically 3-8 z's per conditioning gives a good signal/cost ratio.

Recommended dataset size:
    - 5k pairs: enough to validate training works
    - 50k pairs: real reflow round
    - 200k pairs: high-quality result

Usage:
    # First snapshot the teacher
    python tools/snapshot_cfm_teacher.py

    # Build a manifest from a transcripts CSV (one-time)
    # (left to the caller — produce a JSONL with audio_prompt + text)

    # Generate pairs
    python tools/generate_reflow_pairs.py \\
        --manifest training/ozzy/reflow_manifest.jsonl \\
        --teacher checkpoints/s2mel_teacher.pth \\
        --output-dir training/ozzy/reflow_pairs \\
        --n-samples 4 \\
        --teacher-steps 50 \\
        --teacher-cfg 0.7

    # Resume — the script skips any sample whose output already exists
    python tools/generate_reflow_pairs.py ...  # same command, picks up where it left off
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest of (text, audio_prompt) records.")
    parser.add_argument("--teacher", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "s2mel_teacher.pth",
                        help="Teacher CFM checkpoint (overlay onto the base s2mel).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write paired .pt files.")
    parser.add_argument("--n-samples", type=int, default=4, help="How many z draws per conditioning.")
    parser.add_argument("--teacher-steps", type=int, default=50, help="Teacher Euler step count (more = higher fidelity, more compute).")
    parser.add_argument("--teacher-cfg", type=float, default=0.7, help="Teacher CFG rate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="GPT sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-mel-tokens", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N manifest entries (for smoke testing).")
    parser.add_argument("--store-fp16", action="store_true", default=True,
                        help="Save tensors as fp16 to halve disk usage (default: True).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: skipping malformed line {i}: {e}", file=sys.stderr)
    return records


class CFMTeacherHook:
    """Monkey-patch the CFM.inference so each call records (z, mu, prompt, style, x_lens, x_final).

    We hook at the inference level (not at solve_euler) so we get a single pre-CFG
    z and the final integrated x. The hook captures and returns x_final as the
    original method would.
    """
    def __init__(self, cfm_module: torch.nn.Module, teacher_steps: int, teacher_cfg: float):
        self.cfm = cfm_module
        self.teacher_steps = teacher_steps
        self.teacher_cfg = teacher_cfg
        self._original_inference = cfm_module.inference
        self.captures: List[Dict[str, torch.Tensor]] = []

    def _hooked_inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, solver_type="euler"):
        """Drop-in replacement for CFM.inference that captures the (z, x_final) pair.

        We override `n_timesteps` and `inference_cfg_rate` to the teacher-config
        values regardless of what the caller passed — paired-data generation
        always uses the slow high-fidelity teacher path. We override solver
        to Euler so the pair is canonical (matches what reflow expects).
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.cfm.in_channels, T], device=mu.device) * temperature

        # Snapshot inputs before the CFM mutates them in place (solve_*_ modifies x[..., :prompt_len] = 0).
        # We clone to be safe — these are the inputs we'll save with the pair.
        z_save = z.detach().clone()
        mu_save = mu.detach().clone()
        prompt_save = prompt.detach().clone()
        style_save = style.detach().clone()

        t_span = torch.linspace(0, 1, self.teacher_steps + 1, device=mu.device)
        x_final = self.cfm.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, self.teacher_cfg)

        self.captures.append({
            "z": z_save.cpu(),
            "mu": mu_save.cpu(),
            "prompt": prompt_save.cpu(),
            "style": style_save.cpu(),
            "x_lens": x_lens.detach().cpu(),
            "x_final": x_final.detach().cpu(),
        })
        return x_final

    def __enter__(self):
        self.cfm.inference = self._hooked_inference
        return self

    def __exit__(self, *exc):
        self.cfm.inference = self._original_inference
        self.captures.clear()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.teacher.exists():
        print(
            f"ERROR: teacher checkpoint not found: {args.teacher}\n"
            f"       Run `python tools/snapshot_cfm_teacher.py` first.",
            file=sys.stderr,
        )
        return 1

    records = load_manifest(args.manifest)
    if args.limit:
        records = records[: args.limit]
    print(f">> Loaded {len(records)} manifest records")

    from indextts.infer_v2 import IndexTTS2

    print(">> Loading TTS with teacher CFM overlay")
    tts = IndexTTS2(
        cfg_path=str(PROJECT_ROOT / "checkpoints" / "config.yaml"),
        model_dir=str(PROJECT_ROOT / "checkpoints"),
        use_fp16=torch.cuda.is_available(),
        use_cuda_kernel=torch.cuda.is_available(),
        use_accel=False,
        use_torch_compile=False,
        s2mel_distilled_checkpoint=str(args.teacher),
    )

    save_dtype = torch.float16 if args.store_fp16 else torch.float32

    cfm = tts.s2mel.models['cfm']
    hook = CFMTeacherHook(cfm, args.teacher_steps, args.teacher_cfg)

    total_pairs = 0
    total_skipped = 0
    total_errors = 0
    t_start = time.perf_counter()

    for rec_idx, rec in enumerate(records):
        utt_id = rec.get("id") or f"utt_{rec_idx:06d}"
        text = rec["text"]
        audio_prompt = rec.get("audio_prompt")
        speaker_embeddings_path = rec.get("speaker_embeddings")
        n_samples = int(rec.get("n_samples", args.n_samples))

        # Skip when all sample outputs already exist (resumable)
        all_exist = all(
            (args.output_dir / f"{utt_id}_s{s}.pt").exists()
            for s in range(n_samples)
        )
        if all_exist:
            total_skipped += n_samples
            if args.verbose:
                print(f"   [{rec_idx + 1}/{len(records)}] {utt_id}: all {n_samples} samples present, skipping")
            continue

        speaker_embeddings = None
        if speaker_embeddings_path:
            speaker_embeddings = torch.load(speaker_embeddings_path, map_location="cpu")

        for s in range(n_samples):
            out_path = args.output_dir / f"{utt_id}_s{s}.pt"
            if out_path.exists():
                total_skipped += 1
                continue
            try:
                # Run a single full TTS pass; the hook records the (z, x_final) pair.
                with hook, torch.no_grad():
                    # Use the non-streaming inference for simplicity. We call it once
                    # and only need its side effect of triggering CFM.inference.
                    _wav = tts.infer(
                        spk_audio_prompt=audio_prompt,
                        text=text,
                        output_path=None,  # don't write wav — we want the CFM inputs/outputs
                        verbose=args.verbose,
                        max_text_tokens_per_segment=120,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=True,
                        speaker_embeddings=speaker_embeddings,
                    )
                    if not hook.captures:
                        print(f"WARN: no CFM capture for {utt_id}_s{s}", file=sys.stderr)
                        continue
                    # Most utterances trigger CFM once (single-segment); if it ran
                    # multiple times, save the first segment only. Reflow doesn't
                    # need segment-level alignment — each pair is independent.
                    cap = hook.captures[0]
                    payload = {
                        "z":       cap["z"].to(save_dtype),
                        "mu":      cap["mu"].to(save_dtype),
                        "prompt":  cap["prompt"].to(save_dtype),
                        "style":   cap["style"].to(save_dtype),
                        "x_lens":  cap["x_lens"],  # int dtype, leave alone
                        "x_final": cap["x_final"].to(save_dtype),
                        "meta": {
                            "id": utt_id,
                            "sample_idx": s,
                            "text": text,
                            "teacher_steps": args.teacher_steps,
                            "teacher_cfg": args.teacher_cfg,
                        },
                    }
                    torch.save(payload, out_path)
                    total_pairs += 1
            except Exception:
                total_errors += 1
                print(f"ERROR generating {utt_id}_s{s}:", file=sys.stderr)
                traceback.print_exc()
                continue

        elapsed = time.perf_counter() - t_start
        rate = total_pairs / elapsed if elapsed > 0 else 0
        eta_pairs = (len(records) * args.n_samples) - total_pairs - total_skipped
        eta_secs = eta_pairs / rate if rate > 0 else 0
        print(
            f"[{rec_idx + 1}/{len(records)}] {utt_id}: "
            f"pairs={total_pairs} skipped={total_skipped} errors={total_errors} "
            f"rate={rate:.2f}/s eta={eta_secs / 60:.1f}min"
        )

    print(f"\n>> Done. Wrote {total_pairs} new pairs to {args.output_dir}")
    print(f"   Skipped (already existed): {total_skipped}")
    print(f"   Errors: {total_errors}")
    return 0 if total_errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
