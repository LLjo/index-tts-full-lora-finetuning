#!/usr/bin/env python3
"""
Full fine-tuning script for IndexTTS2 (GPT module) with custom voice data.

Unlike LoRA training which only trains adapter weights, this script fine-tunes
the ENTIRE model. This provides stronger voice adaptation but:
- Requires more GPU memory
- Risk of catastrophic forgetting
- Creates full-size checkpoints (~1GB+)

The script expects preprocessed manifests from tools/prepare_lora_dataset.py
(single-sample format is supported).

Usage:
    python tools/train_gpt_finetune.py \
        --train-manifest my_voice_project/train_manifest.jsonl \
        --val-manifest my_voice_project/val_manifest.jsonl \
        --output-dir trained_checkpoints/my_voice \
        --epochs 50 \
        --learning-rate 1e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tune IndexTTS2 GPT on custom voice data.")
    
    # Data arguments
    parser.add_argument(
        "--train-manifest",
        dest="train_manifests",
        action="append",
        type=str,
        required=True,
        help="Training manifest JSONL (from prepare_lora_dataset.py). Can be repeated for multiple datasets.",
    )
    parser.add_argument(
        "--val-manifest",
        dest="val_manifests",
        action="append",
        type=str,
        required=True,
        help="Validation manifest JSONL. Can be repeated for multiple datasets.",
    )
    
    # Model arguments
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/bpe.model"), help="SentencePiece model path.")
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_checkpoints"), help="Directory for checkpoints/logs.")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size per optimization step (smaller for full fine-tuning).")
    parser.add_argument("--grad-accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (more epochs for full fine-tuning).")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (lower for full fine-tuning to avoid forgetting).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=200, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimizer steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=0, help="Validation frequency in steps (0 = once per epoch).")
    parser.add_argument("--save-every", type=int, default=500, help="Steps between checkpoint saves.")
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Number of recent checkpoints to keep.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--text-loss-weight", type=float, default=0.2, help="Weight for text CE loss.")
    parser.add_argument("--mel-loss-weight", type=float, default=0.8, help="Weight for semantic CE loss.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto' for latest.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    
    # Fine-tuning specific
    parser.add_argument("--freeze-conditioning", action="store_true", help="Freeze conditioning encoders (train only GPT transformer).")
    parser.add_argument("--freeze-embeddings", action="store_true", help="Freeze embedding layers.")
    
    return parser.parse_args()


@dataclass
class ManifestSpec:
    path: Path
    language: Optional[str] = None


def parse_manifest_specs(entries: Sequence[str], flag_name: str) -> List[ManifestSpec]:
    if not entries:
        raise ValueError(f"{flag_name} requires at least one manifest path.")
    specs: List[ManifestSpec] = []
    for raw in entries:
        value = raw.strip()
        path = Path(value).expanduser()
        specs.append(ManifestSpec(path=path, language=None))
    return specs


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    sample_type: str = "single"
    manifest_path: Optional[Path] = None


class FinetuneDataset(Dataset):
    """Dataset for full fine-tuning. Supports both single and paired manifests."""
    
    def __init__(self, manifests: Sequence[ManifestSpec]):
        if isinstance(manifests, ManifestSpec):
            manifests = [manifests]
        manifest_list = list(manifests)
        if not manifest_list:
            raise ValueError("No manifest paths supplied.")

        self.samples: List[Sample] = []
        self.bad_indices: Set[int] = set()

        for spec in manifest_list:
            self._load_single_manifest(spec)

        if not self.samples:
            manifest_paths = ", ".join(str(spec.path) for spec in manifest_list)
            raise RuntimeError(f"No entries found in the provided manifests: {manifest_paths}")

    @staticmethod
    def _resolve_path(base_dir: Path, value: str) -> Path:
        if not value:
            raise ValueError("Empty path provided in manifest record.")
        path = Path(value)
        if path.is_absolute():
            return path
        return (base_dir / path).expanduser()

    def _load_single_manifest(self, spec: ManifestSpec) -> None:
        manifest_path = spec.path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        local_count = 0
        base_dir = manifest_path.parent

        print(f"[Info] Parsing manifest {manifest_path} ...")
        
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                # Support both single and paired manifest formats
                is_paired = "prompt_condition_path" in record and "target_codes_path" in record
                
                if is_paired:
                    # Paired format (prompt/target)
                    emo_path = record.get("prompt_emo_vec_path") or record.get("target_emo_vec_path")
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["target_text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["target_codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["prompt_condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, emo_path),
                        text_len=int(record["target_text_len"]),
                        code_len=int(record["target_code_len"]),
                        condition_len=int(record.get("prompt_condition_len", 32)),
                        sample_type="paired",
                        manifest_path=manifest_path,
                    )
                else:
                    # Single format (from prepare_lora_dataset.py)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, record["emo_vec_path"]),
                        text_len=int(record["text_len"]),
                        code_len=int(record["code_len"]),
                        condition_len=int(record.get("condition_len", 32)),
                        sample_type="single",
                        manifest_path=manifest_path,
                    )

                self.samples.append(sample)
                local_count += 1

        print(f"[Info] Loaded {local_count} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError("Dataset is empty.")

        sample = self.samples[idx]
        
        try:
            text_ids = np.load(sample.text_ids_path, allow_pickle=False)
            codes = np.load(sample.codes_path, allow_pickle=False)
            condition = np.load(sample.condition_path, allow_pickle=False)
            emo_vec = np.load(sample.emo_vec_path, allow_pickle=False)

            if text_ids.size == 0 or codes.size == 0 or condition.size == 0 or emo_vec.size == 0:
                raise ValueError("Encountered empty feature file.")

            # Validate codes shape - must be 1D
            if codes.ndim != 1:
                raise ValueError(f"Invalid codes shape {codes.shape}. Expected 1D array.")

            text_ids = text_ids.astype(np.int64, copy=False)
            codes = codes.astype(np.int64, copy=False)
            condition = condition.astype(np.float32, copy=False)
            emo_vec = emo_vec.astype(np.float32, copy=False)

            return {
                "id": sample.id,
                "text_ids": torch.from_numpy(text_ids),
                "codes": torch.from_numpy(codes),
                "condition": torch.from_numpy(condition),
                "emo_vec": torch.from_numpy(emo_vec),
                "text_len": torch.tensor(sample.text_len, dtype=torch.long),
                "code_len": torch.tensor(sample.code_len, dtype=torch.long),
                "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
            }

        except Exception as exc:
            raise RuntimeError(f"Failed to load sample '{sample.id}': {exc}")


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.stack(condition_tensors, dim=0)
    emo_stacked = torch.stack(emo_tensors, dim=0)

    text_lengths = torch.stack([item["text_len"] for item in batch])
    code_lengths = torch.stack([item["code_len"] for item in batch])
    cond_lengths = torch.stack([item["condition_len"] for item in batch])

    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
    }


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(cfg_path: Path, tokenizer: TextTokenizer, base_checkpoint: Path, device: torch.device) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size

    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    raw_state_dict = checkpoint.get("model", checkpoint)

    # Filter out unwanted keys
    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state_dict[new_key] = value
    
    # Detect actual model_dim from checkpoint
    if "mel_pos_embedding.emb.weight" in filtered_state_dict:
        checkpoint_model_dim = filtered_state_dict["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != checkpoint_model_dim:
            print(f"[Warn] Config specifies model_dim={cfg.gpt.model_dim}, but checkpoint uses {checkpoint_model_dim}")
            print(f"[Info] Using checkpoint's model_dim={checkpoint_model_dim} for compatibility")
            cfg.gpt.model_dim = checkpoint_model_dim
    
    model = UnifiedVoice(**cfg.gpt)
    
    # Handle vocabulary size mismatches
    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,
        "text_head.weight": model.text_head.weight,
        "text_head.bias": model.text_head.bias,
    }
    for key, param in resizable_keys.items():
        weight = filtered_state_dict.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[: slices[0]].copy_(weight[: slices[0]])
            else:
                param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
        filtered_state_dict[key] = param.detach().clone()

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys during load: {unexpected}")

    return model.to(device)


def compute_losses(
    model: UnifiedVoice,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)
    
    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Get max supported sequence lengths from position embeddings
    max_text_supported = model.text_pos_embedding.emb.num_embeddings
    max_mel_supported = model.mel_pos_embedding.emb.num_embeddings
    
    # Crop sequences if needed
    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    if text_inputs.size(1) + 2 > max_text_supported:
        max_text_len = max_text_supported - 2
        text_inputs = text_inputs[:, :max_text_len]
        text_lengths = torch.clamp(text_lengths, max=max_text_len)
    
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    if mel_inputs.size(1) + 2 > max_mel_supported:
        max_mel_len = max_mel_supported - 2
        mel_inputs = mel_inputs[:, :max_mel_len]
        code_lengths = torch.clamp(code_lengths, max=max_mel_len)
    
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )

    duration_ctrl = model.speed_emb(torch.ones_like(use_speed))
    duration_free = model.speed_emb(torch.zeros_like(use_speed))
    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1), duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )

    text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    metrics = {}
    with torch.no_grad():
        mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_targets_flat = mel_targets.reshape(-1)
        mel_mask_flat = mel_mask.reshape(-1)
        if mel_mask_flat.any():
            valid_logits = mel_logits_flat[mel_mask_flat]
            valid_targets = mel_targets_flat[mel_mask_flat]
            top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
        else:
            top1 = 0.0
        metrics["mel_top1"] = top1

    return text_loss, mel_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    recent_checkpoints: List[str],
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "recent_checkpoints": recent_checkpoints,
        "best_val": best_val,
    }
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def evaluate(
    model: UnifiedVoice,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            text_loss, mel_loss, metrics = compute_losses(model, batch, device)
            bsz = batch["text_ids"].size(0)
            totals["text_loss"] += text_loss.item() * bsz
            totals["mel_loss"] += mel_loss.item() * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
    if was_training:
        model.train()
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def freeze_module(module: nn.Module, name: str):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False
    print(f"  Frozen: {name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = log_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    # Load tokenizer and model
    print("\n[Step 1/4] Loading model...")
    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(args.config, tokenizer, args.base_checkpoint, device)
    print(f"Model loaded from: {args.base_checkpoint}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optional: Freeze certain parts of the model
    if args.freeze_conditioning:
        print("\nFreezing conditioning encoders:")
        freeze_module(model.conditioning_encoder, "conditioning_encoder")
        freeze_module(model.perceiver_encoder, "perceiver_encoder")
        freeze_module(model.emo_conditioning_encoder, "emo_conditioning_encoder")
        freeze_module(model.emo_perceiver_encoder, "emo_perceiver_encoder")
    
    if args.freeze_embeddings:
        print("\nFreezing embeddings:")
        freeze_module(model.text_embedding, "text_embedding")
        freeze_module(model.mel_embedding, "mel_embedding")
        freeze_module(model.text_pos_embedding, "text_pos_embedding")
        freeze_module(model.mel_pos_embedding, "mel_pos_embedding")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Load datasets
    print("\n[Step 2/4] Loading datasets...")
    train_specs = parse_manifest_specs(args.train_manifests, "--train-manifest")
    val_specs = parse_manifest_specs(args.val_manifests, "--val-manifest")

    train_dataset = FinetuneDataset(train_specs)
    val_dataset = FinetuneDataset(val_specs)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )

    # Setup optimizer (only for trainable params)
    print("\n[Step 3/4] Setting up training...")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(train_loader)) // max(1, args.grad_accumulation)
    total_steps = max(total_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP enabled")

    global_step = 0
    start_epoch = 0
    best_val = math.inf
    recent_checkpoints: List[str] = []
    last_saved_step: Optional[int] = None

    # Resume from checkpoint if specified
    resume_path: Optional[str] = None
    if args.resume:
        if args.resume == "auto":
            candidate = output_dir / "latest.pth"
            if candidate.exists():
                resume_path = str(candidate)
        else:
            resume_path = args.resume
    
    if resume_path:
        print(f"\nResuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)
        recent_checkpoints = checkpoint.get("recent_checkpoints", [])
        best_val = checkpoint.get("best_val", math.inf)
        last_saved_step = checkpoint.get("step")
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"\n[Step 4/4] Starting full fine-tuning:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps: {total_steps}")
    print()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                debug_batch = (epoch == start_epoch and batch_idx == 0)
                text_loss, mel_loss, metrics = compute_losses(model, batch, device, debug=debug_batch)
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
            
            if use_amp:
                scaler.scale(loss / args.grad_accumulation).backward()
            else:
                (loss / args.grad_accumulation).backward()

            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        args.grad_clip
                    )
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # Logging
                if global_step % args.log_interval == 0:
                    writer.add_scalar("train/text_loss", text_loss.item(), global_step)
                    writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"text_loss={text_loss.item():.4f} mel_loss={mel_loss.item():.4f} "
                        f"mel_top1={metrics['mel_top1']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                # Step-based validation
                if args.val_interval > 0 and global_step % args.val_interval == 0:
                    val_metrics = evaluate(model, val_loader, device)
                    writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
                    writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
                    writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
                    print(
                        f"[Val] epoch={epoch + 1} step={global_step} "
                        f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                        f"mel_top1={val_metrics['mel_top1']:.4f}"
                    )
                    if val_metrics["mel_loss"] < best_val:
                        best_val = val_metrics["mel_loss"]
                        # Save best model
                        best_path = output_dir / "best_model.pth"
                        # save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints, best_val)
                        print(f"New best model! val_loss={best_val:.4f}")

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_path = output_dir / f"model_step{global_step}.pth"
                    recent_checkpoints.append(str(ckpt_path))
                    save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints, best_val)
                    
                    # Also save as latest
                    latest_path = output_dir / "latest.pth"
                    save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints, best_val)
                    
                    # Remove old checkpoints
                    while len(recent_checkpoints) > args.keep_checkpoints:
                        obsolete = recent_checkpoints.pop(0)
                        try:
                            os.remove(obsolete)
                            print(f"Removed old checkpoint: {obsolete}")
                        except OSError:
                            pass
                    last_saved_step = global_step

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        # End-of-epoch validation
        if args.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
            writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
            writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
            print(
                f"[Val] epoch={epoch + 1} step={global_step} "
                f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                f"mel_top1={val_metrics['mel_top1']:.4f}"
            )
            if val_metrics["mel_loss"] < best_val:
                best_val = val_metrics["mel_loss"]
                best_path = output_dir / "best_model.pth"
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, recent_checkpoints, best_val)
                print(f"New best model! val_loss={best_val:.4f}")

    # Final checkpoint
    if global_step > 0 and last_saved_step != global_step:
        ckpt_path = output_dir / f"model_step{global_step}.pth"
        recent_checkpoints.append(str(ckpt_path))
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, args.epochs, global_step, recent_checkpoints, best_val)
        
        latest_path = output_dir / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, args.epochs, global_step, recent_checkpoints, best_val)
        
        while len(recent_checkpoints) > args.keep_checkpoints:
            obsolete = recent_checkpoints.pop(0)
            try:
                os.remove(obsolete)
            except OSError:
                pass

    writer.close()
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"  Final checkpoint: {output_dir / 'latest.pth'}")
    print(f"  Best checkpoint: {output_dir / 'best_model.pth'}")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"\nTo use for inference, copy gpt.pth checkpoint:")
    print(f"  cp {output_dir}/best_model.pth checkpoints/gpt_finetuned.pth")
    print(f"\nThen modify your inference code to use the finetuned model")


if __name__ == "__main__":
    main()