#!/usr/bin/env python3
"""
Train a reflow-distilled CFM student from paired (z, x_final) data.

Background:
    Flow matching learns dx/dt = v(x, t). With "vanilla" random pairings of (z, x_1),
    the learned trajectory is curved, so inference needs many integration steps.
    Reflow (Liu et al.) generates paired data by running the TEACHER once at high
    step count, then trains a STUDENT on the *paired* (z, x_final) trajectory.
    The paired trajectory is approximately straight, so the student can do 1-2
    step inference with quality close to many-step teacher.

What this script does:
    - Loads paired data from disk (output of tools/generate_reflow_pairs.py).
    - Initializes a fresh CFM from base s2mel weights (warm start) — the student
      has the same architecture as the teacher, just retrains the estimator
      (DiT). The length_regulator and gpt_layer aren't part of the CFM so they
      stay frozen at base values and are NOT modified here.
    - Trains the student with the standard flow-matching loss but using the
      PAIRED (z, x_final) instead of random (z, x_1) — this is what makes reflow
      different from plain CFM training.
    - Saves checkpoints in the load_checkpoint2-compatible format:
        {"net": {"cfm": cfm_state_dict}}
      which IndexTTS2 will overlay onto base s2mel when loaded via
      s2mel_distilled_checkpoint=...

Loss:
    For each paired sample (z, x_final, mu, prompt, style):
      1. Apply prompt mask: z[:, :, :prompt_len] = 0
      2. Sample t ~ U[0,1]
      3. y_t = (1-t)*z + t*x_final
      4. u = x_final - z  (target velocity)
      5. v_pred = estimator(y_t, prompt_x, x_lens, t, style, mu)
      6. loss = MSE(v_pred, u) over non-prompt frames

Usage:
    python tools/train_cfm_reflow.py \\
        --pairs-dir training/ozzy/reflow_pairs \\
        --output-dir training/ozzy/cfm_reflow_student \\
        --epochs 40 \\
        --batch-size 4 \\
        --grad-accumulation 4 \\
        --learning-rate 5e-5

    # Resume from a saved checkpoint
    python tools/train_cfm_reflow.py ... --resume training/ozzy/cfm_reflow_student/checkpoint_e10.pth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", type=Path, required=True, help="Directory of *.pt paired files from generate_reflow_pairs.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save student checkpoints.")
    parser.add_argument("--base-s2mel", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "s2mel.pth",
                        help="Base s2mel.pth — student CFM is warm-started from this.")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "config.yaml")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a previously-saved student checkpoint.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--amp", action="store_true", default=True, help="Use bf16 mixed precision (default: True).")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--save-every-epochs", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PairFile:
    path: Path


class ReflowPairsDataset(Dataset):
    """Loads paired (z, mu, prompt, style, x_lens, x_final) .pt files from disk.

    Each .pt file is a single sample (one z draw + one teacher x_final + the
    associated conditioning). Variable T per sample — the collate function pads.
    Conditioning is identical across multiple z draws for the same utterance,
    which is fine — duplicate conditioning across the batch just gives the
    student more noise→data pairs per conditioning, which is the reflow signal.
    """

    def __init__(self, files: List[PairFile]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        payload = torch.load(self.files[idx].path, map_location="cpu", weights_only=False)
        return payload


def reflow_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Pad-right to the longest sample in the batch.

    Shapes after collate:
      z:        [B, 80, T_max]
      x_final:  [B, 80, T_max]
      mu:       [B, T_max, 512]
      prompt:   [B, 80, T_prompt_max]
      style:    [B, 192]
      x_lens:   [B]
      prompt_lens: [B]  (derived from per-sample prompt.shape[-1])
    """
    B = len(batch)
    n_channels = batch[0]["z"].shape[1]
    mu_dim = batch[0]["mu"].shape[-1]

    T_max = max(int(item["x_lens"].item() if item["x_lens"].dim() == 0 else item["x_lens"][0].item()) for item in batch)
    Tp_max = max(item["prompt"].shape[-1] for item in batch)

    z = torch.zeros(B, n_channels, T_max, dtype=torch.float32)
    x_final = torch.zeros(B, n_channels, T_max, dtype=torch.float32)
    mu = torch.zeros(B, T_max, mu_dim, dtype=torch.float32)
    prompt = torch.zeros(B, n_channels, Tp_max, dtype=torch.float32)
    style = torch.zeros(B, 192, dtype=torch.float32)
    x_lens = torch.zeros(B, dtype=torch.long)
    prompt_lens = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        T = int(item["x_lens"].item() if item["x_lens"].dim() == 0 else item["x_lens"][0].item())
        Tp = item["prompt"].shape[-1]
        z[i, :, :T] = item["z"][0, :, :T].to(torch.float32)
        x_final[i, :, :T] = item["x_final"][0, :, :T].to(torch.float32)
        mu[i, :T, :] = item["mu"][0, :T, :].to(torch.float32)
        prompt[i, :, :Tp] = item["prompt"][0, :, :Tp].to(torch.float32)
        style[i] = item["style"][0].to(torch.float32)
        x_lens[i] = T
        prompt_lens[i] = Tp

    return {
        "z": z,
        "x_final": x_final,
        "mu": mu,
        "prompt": prompt,
        "style": style,
        "x_lens": x_lens,
        "prompt_lens": prompt_lens,
    }


def build_student_s2mel(cfg, base_s2mel_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the base s2mel as the warm-start point for the student.

    We need the whole s2mel scaffold (length_regulator, gpt_layer, cfm) loaded
    even though we only train the CFM, because the saved checkpoint format
    re-uses the load_checkpoint2 dict-of-state-dicts layout.
    """
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel, None, str(base_s2mel_path),
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    s2mel = s2mel.to(device)
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=64, max_seq_length=8192)
    return s2mel


def save_student(s2mel: torch.nn.Module, out_path: Path, meta: Dict[str, Any]):
    """Save the student in the same format IndexTTS2/load_checkpoint2 expects.

    Only the CFM submodule's params are saved — IndexTTS2's overlay code
    will leave the rest of s2mel at its base values.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "net": {"cfm": s2mel.models["cfm"].state_dict()},
        "meta": meta,
    }
    torch.save(state, out_path)


def reflow_step(
    s2mel: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    sigma_min: float = 1e-6,
) -> torch.Tensor:
    """Compute the reflow flow-matching loss for one batch."""
    z = batch["z"].to(device)              # [B, 80, T]
    x_final = batch["x_final"].to(device)  # [B, 80, T]
    mu = batch["mu"].to(device)            # [B, T, 512]
    prompt = batch["prompt"].to(device)    # [B, 80, T_prompt]
    style = batch["style"].to(device)      # [B, 192]
    x_lens = batch["x_lens"].to(device)    # [B]
    prompt_lens = batch["prompt_lens"].to(device)  # [B]

    B = z.size(0)

    # Sample t per-sample
    t = torch.rand(B, 1, 1, device=device, dtype=z.dtype)

    # Reflow interpolation along the PAIRED trajectory (NOT random z+x_1)
    # y_t = (1-t)*z + t*x_final
    y = (1 - (1 - sigma_min) * t) * z + t * x_final
    # Target velocity u = x_final - z (with sigma_min ≈ 0 it's exactly the chord)
    u = x_final - (1 - sigma_min) * z

    # Apply prompt-region masking to match inference: zero out y in prompt region
    # and put the reference mel in the prompt scaffold tensor.
    prompt_x = torch.zeros_like(z)
    for b in range(B):
        pl = int(prompt_lens[b].item())
        prompt_x[b, :, :pl] = prompt[b, :, :pl]
        y[b, :, :pl] = 0
        # Also zero mu in prompt region IF the model was trained with zero_prompt_speech_token
        # (mirrors solve_euler's behavior). The CFM module knows whether to do this.

    cfm = s2mel.models["cfm"]
    if getattr(cfm, "zero_prompt_speech_token", False):
        for b in range(B):
            pl = int(prompt_lens[b].item())
            mu[b, :pl, :] = 0

    # DiT.forward signature: (x, prompt_x, x_lens, t, style, cond, mask_content=False)
    # Do NOT pass prompt_lens here — there's no such argument. The prompt-region
    # masking is already applied to y and (optionally) mu above.
    v_pred = cfm.estimator(y, prompt_x, x_lens, t.squeeze(1).squeeze(1), style, mu)

    # Mask loss to non-prompt, non-padding region.
    loss_per_sample = []
    for b in range(B):
        pl = int(prompt_lens[b].item())
        tl = int(x_lens[b].item())
        if tl <= pl:
            continue  # nothing to learn on this sample
        pred_slice = v_pred[b, :, pl:tl]
        target_slice = u[b, :, pl:tl]
        loss_per_sample.append(F.mse_loss(pred_slice, target_slice))

    if not loss_per_sample:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(loss_per_sample).mean()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if not args.pairs_dir.exists():
        print(f"ERROR: pairs dir not found: {args.pairs_dir}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    # Collect paired files
    pair_files = sorted(args.pairs_dir.glob("*.pt"))
    if not pair_files:
        print(f"ERROR: no .pt files in {args.pairs_dir}", file=sys.stderr)
        return 1
    print(f">> Found {len(pair_files)} paired samples")

    # Train/val split
    rng = random.Random(args.seed)
    rng.shuffle(pair_files)
    n_val = max(1, int(len(pair_files) * args.val_split))
    val_files = [PairFile(p) for p in pair_files[:n_val]]
    train_files = [PairFile(p) for p in pair_files[n_val:]]
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = ReflowPairsDataset(train_files)
    val_ds = ReflowPairsDataset(val_files)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=reflow_collate,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=reflow_collate,
    )

    # Build student warm-started from base
    cfg = OmegaConf.load(args.config)
    s2mel = build_student_s2mel(cfg, args.base_s2mel, device)

    if args.resume:
        print(f">> Resuming from {args.resume}")
        from indextts.s2mel.modules.commons import load_checkpoint2
        load_checkpoint2(
            s2mel, None, str(args.resume),
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )

    # Only the CFM is trainable
    for p in s2mel.parameters():
        p.requires_grad_(False)
    cfm = s2mel.models["cfm"]
    for p in cfm.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in cfm.parameters() if p.requires_grad)
    print(f">> Trainable parameters: {trainable / 1e6:.2f}M")

    optimizer = AdamW(
        [p for p in cfm.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = (len(train_loader) // args.grad_accumulation) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))
    amp_dtype = torch.bfloat16 if (args.amp and torch.cuda.is_bf16_supported()) else torch.float16

    global_step = 0
    best_val = float("inf")

    history = []

    for epoch in range(1, args.epochs + 1):
        cfm.train()
        running = []
        t_epoch = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            accumulate = (batch_idx + 1) % args.grad_accumulation != 0

            with torch.amp.autocast(device.type, enabled=(args.amp and device.type == "cuda"), dtype=amp_dtype):
                loss = reflow_step(s2mel, batch, device)

            loss_scaled = loss / args.grad_accumulation
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if not accumulate:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cfm.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[epoch {epoch} step {global_step}] "
                        f"loss={loss.item():.4f} lr={lr:.2e}",
                        flush=True,
                    )

            running.append(loss.item())

        epoch_time = time.perf_counter() - t_epoch
        train_loss = float(np.mean(running)) if running else float("nan")

        # Validation
        cfm.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast(device.type, enabled=(args.amp and device.type == "cuda"), dtype=amp_dtype):
                    vloss = reflow_step(s2mel, batch, device)
                val_losses.append(vloss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_secs": epoch_time})
        with open(args.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"== epoch {epoch}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
            f"time={epoch_time:.1f}s ==",
            flush=True,
        )

        # Save periodic + best
        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        if epoch % args.save_every_epochs == 0 or epoch == args.epochs:
            save_student(s2mel, args.output_dir / f"checkpoint_e{epoch:03d}.pth", meta)
        if val_loss < best_val:
            best_val = val_loss
            save_student(s2mel, args.output_dir / "best.pth", meta)
            print(f"   new best val: {best_val:.4f} → saved best.pth", flush=True)

    print(">> Training complete. Latest checkpoint and best.pth written to", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
