#!/usr/bin/env python3
"""
Snapshot the current S2Mel checkpoint as the teacher for Phase 3 distillation.

What it does:
- Copies `checkpoints/s2mel.pth` to `checkpoints/s2mel_teacher.pth`.
- Optionally merges an S2Mel LoRA into the snapshot so the teacher already speaks
  the trained voice (stutter etc.) — recommended for per-voice student training.
- Idempotent: skips the copy if the destination is already up to date and warns
  before overwriting unless --force is passed.

Why we keep a snapshot:
- The training loop overwrites s2mel.pth in place. Without an immutable teacher
  copy on disk, paired-data generation and the final A/B eval lose their reference.
- Per-voice distillation needs the LoRA baked into the teacher's weights before
  generating paired data — otherwise the student learns to imitate the *base*
  voice and you lose the stutter.

Usage:
    # Plain base teacher
    python tools/snapshot_cfm_teacher.py

    # Bake in a speaker's S2Mel LoRA so the teacher already does the stutter
    python tools/snapshot_cfm_teacher.py --merge-lora training/ozzy/s2mel_lora/final_checkpoint

    # Custom paths
    python tools/snapshot_cfm_teacher.py \\
        --source checkpoints/s2mel.pth \\
        --dest checkpoints/s2mel_teacher_ozzy.pth \\
        --merge-lora training/ozzy/s2mel_lora/best_checkpoint
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            buf = fp.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def merge_lora_into_checkpoint(
    base_ckpt_path: Path,
    lora_path: Path,
    out_path: Path,
) -> None:
    """Load base s2mel, merge LoRA weights, save as a plain (non-LoRA) checkpoint.

    The output is in the same dict format that `load_checkpoint2` consumes
    (`{"net": {<submodule>: state_dict}, ...}`), so it can be loaded by IndexTTS2
    with no awareness that it was ever a LoRA. This is what we want for the
    distillation teacher.
    """
    from omegaconf import OmegaConf

    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    from indextts.utils.s2mel_lora_utils import (
        apply_lora_to_s2mel,
        load_s2mel_lora_checkpoint,
    )

    print(f">> Loading base s2mel from {base_ckpt_path}")
    cfg = OmegaConf.load(PROJECT_ROOT / "checkpoints" / "config.yaml")
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel, None, str(base_ckpt_path), load_only_params=True, is_distributed=False
    )

    print(f">> Applying + merging LoRA from {lora_path}")
    # apply_lora_to_s2mel + load_s2mel_lora_checkpoint should restore the trained
    # LoRA on top of the base, then we ask PEFT to merge it back into the base
    # weights so the saved checkpoint has no LoRA dependency.
    s2mel = apply_lora_to_s2mel(s2mel, rank=16)  # rank ignored when loading
    s2mel = load_s2mel_lora_checkpoint(s2mel, lora_path, merge_weights=True)

    print(f">> Saving merged teacher to {out_path}")
    state = {"net": {}}
    for key in s2mel.models:
        state["net"][key] = s2mel.models[key].state_dict()
    torch.save(state, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "checkpoints" / "s2mel.pth",
        help="Base S2Mel checkpoint to snapshot.",
    )
    parser.add_argument(
        "--dest", type=Path, default=PROJECT_ROOT / "checkpoints" / "s2mel_teacher.pth",
        help="Destination path for the teacher snapshot.",
    )
    parser.add_argument(
        "--merge-lora", type=Path, default=None,
        help="Optional LoRA checkpoint dir to merge into the teacher (per-voice teacher).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the destination even if it already exists.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: source checkpoint not found: {args.source}", file=sys.stderr)
        return 1

    if args.dest.exists() and not args.force:
        if args.merge_lora is None:
            src_hash = file_digest(args.source)
            dst_hash = file_digest(args.dest)
            if src_hash == dst_hash:
                print(f">> Teacher already up to date: {args.dest}")
                return 0
        print(
            f"ERROR: {args.dest} already exists and is different from source.\n"
            f"       Pass --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    args.dest.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_lora is None:
        print(f">> Copying {args.source} → {args.dest}")
        shutil.copy2(args.source, args.dest)
    else:
        if not args.merge_lora.exists():
            print(f"ERROR: LoRA path not found: {args.merge_lora}", file=sys.stderr)
            return 1
        merge_lora_into_checkpoint(args.source, args.merge_lora, args.dest)

    size_mb = args.dest.stat().st_size / 1024 / 1024
    print(f">> Done. Teacher snapshot is {size_mb:.1f} MB at {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
