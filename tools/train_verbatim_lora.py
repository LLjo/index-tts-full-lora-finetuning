#!/usr/bin/env python3
"""
Verbatim LoRA Training for IndexTTS2

Train LoRA adapters on verbatim transcriptions to reproduce stutters and 
speech imperfections at inference time.

The Key Insight:
================
Standard training: clean text → audio codes
    Problem: Model learns clean speech, no stutters

Verbatim training: stuttered text → audio codes  
    Solution: Model learns "I I I was" → stuttered audio codes

This means at inference:
    - Input "I I I was going going to" → Output has stutters!
    - Input "I was going to" → Output is clean

The model learns the DIRECT TEXT-TO-STUTTER mapping!

Architecture (IndexTTS2 v2.0):
=============================
Stage 1 (GPT) - WHAT WE TRAIN:
    text_tokens + conditioning → semantic_codes
    
    We train LoRA on:
    - GPT transformer layers (c_attn, c_proj, c_fc)
    - Optionally: conditioning encoders, heads
    
Stage 2 (S2Mel + BigVGAN) - UNCHANGED:
    semantic_codes + reference features → audio waveform

Usage:
    python tools/train_verbatim_lora.py --speaker ozzy --epochs 20
    
    # Advanced options
    python tools/train_verbatim_lora.py \\
        --speaker ozzy \\
        --epochs 30 \\
        --lora-rank 16 \\
        --batch-size 4 \\
        --learning-rate 5e-4 \\
        --stutter-weight 2.0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.lora_utils import (
    apply_lora_to_model,
    save_lora_checkpoint,
    get_trainable_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA on verbatim transcriptions for stutter reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Speaker/data
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--train-manifest", type=Path, help="Custom train manifest")
    parser.add_argument("--val-manifest", type=Path, help="Custom val manifest")
    
    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--include-conditioning", action="store_true", default=True,
                        help="Apply LoRA to conditioning encoders")
    parser.add_argument("--include-heads", action="store_true", default=False,
                        help="Apply LoRA to output heads")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grad-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    
    # Loss weights
    parser.add_argument("--text-weight", type=float, default=0.2,
                        help="Weight for text loss (default: 0.2)")
    parser.add_argument("--stutter-weight", type=float, default=2.0,
                        help="Extra weight for samples with stutters (default: 2.0)")
    
    # Output
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model paths
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"))
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/bpe.model"))
    
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class VerbatimSample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    has_repetitions: bool = False
    has_fillers: bool = False
    repetition_count: int = 0


class VerbatimDataset(Dataset):
    """Dataset for verbatim training."""
    
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.base_dir = manifest_path.parent
        self.samples: List[VerbatimSample] = []
        self._load_manifest()
    
    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.base_dir / path
    
    def _load_manifest(self):
        with open(self.manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                sample = VerbatimSample(
                    id=record["id"],
                    text_ids_path=self._resolve_path(record["text_ids_path"]),
                    codes_path=self._resolve_path(record["codes_path"]),
                    condition_path=self._resolve_path(record["condition_path"]),
                    emo_vec_path=self._resolve_path(record["emo_vec_path"]),
                    text_len=int(record["text_len"]),
                    code_len=int(record["code_len"]),
                    condition_len=int(record.get("condition_len", 32)),
                    has_repetitions=record.get("has_repetitions", False),
                    has_fillers=record.get("has_fillers", False),
                    repetition_count=record.get("repetition_count", 0),
                )
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {self.manifest_path}")
        
        # Count stutter samples
        stutter_count = sum(1 for s in self.samples if s.has_repetitions or s.has_fillers)
        print(f"  Samples with stutters: {stutter_count}/{len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        text_ids = np.load(sample.text_ids_path, allow_pickle=False).astype(np.int64)
        codes = np.load(sample.codes_path, allow_pickle=False).astype(np.int64)
        condition = np.load(sample.condition_path, allow_pickle=False).astype(np.float32)
        emo_vec = np.load(sample.emo_vec_path, allow_pickle=False).astype(np.float32)
        
        return {
            "id": sample.id,
            "text_ids": torch.from_numpy(text_ids),
            "codes": torch.from_numpy(codes),
            "condition": torch.from_numpy(condition),
            "emo_vec": torch.from_numpy(emo_vec),
            "text_len": torch.tensor(sample.text_len, dtype=torch.long),
            "code_len": torch.tensor(sample.code_len, dtype=torch.long),
            "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
            "has_stutters": sample.has_repetitions or sample.has_fillers,
            "repetition_count": sample.repetition_count,
        }


def collate_batch(batch: List[Dict]) -> Dict:
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
    
    has_stutters = torch.tensor([item["has_stutters"] for item in batch], dtype=torch.bool)
    
    return {
        "ids": [item["id"] for item in batch],
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
        "has_stutters": has_stutters,
    }


def load_tokenizer(path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    return TextTokenizer(str(path), normalizer)


def build_model(cfg_path: Path, checkpoint_path: Path, device: torch.device) -> UnifiedVoice:
    """Load base GPT model."""
    cfg = OmegaConf.load(cfg_path)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint)
    
    # Detect model_dim from checkpoint
    if "mel_pos_embedding.emb.weight" in raw_state:
        checkpoint_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != checkpoint_dim:
            cfg.gpt.model_dim = checkpoint_dim
            print(f"  Detected model_dim: {checkpoint_dim}")
    
    # Filter state dict to remove inference model and LoRA artifacts
    filtered_state = {}
    for key, value in raw_state.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state[new_key] = value
    
    model = UnifiedVoice(**cfg.gpt)
    
    # Handle vocab size mismatch
    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,
        "text_head.weight": model.text_head.weight,
        "text_head.bias": model.text_head.bias,
    }
    for key, param in resizable_keys.items():
        weight = filtered_state.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[:slices[0]].copy_(weight[:slices[0]])
            else:
                param[:slices[0], :slices[1]].copy_(weight[:slices[0], :slices[1]])
        filtered_state[key] = param.detach().clone()
    
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {missing[:5]}...")
    
    return model.to(device)


def compute_loss(
    model: nn.Module,
    batch: Dict,
    device: torch.device,
    text_weight: float = 0.2,
    stutter_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training loss for verbatim text → codes mapping.
    
    Key: We're training the model to predict semantic codes from verbatim text.
    The verbatim text contains stutters, so the model learns the mapping!
    """
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)
    has_stutters = batch["has_stutters"].to(device)
    
    batch_size = text_ids.size(0)
    
    # Build conditioning
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
    duration_ctrl = base_model.speed_emb(torch.ones_like(use_speed))
    duration_free = base_model.speed_emb(torch.zeros_like(use_speed))
    
    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1), 
         duration_ctrl.unsqueeze(1), 
         duration_free.unsqueeze(1)),
        dim=1,
    )
    
    # Position embedding limits
    max_text = base_model.text_pos_embedding.emb.num_embeddings
    max_mel = base_model.mel_pos_embedding.emb.num_embeddings
    
    # Process text inputs (the VERBATIM text tokens!)
    text_inputs = base_model.set_text_padding(text_ids.clone(), text_lengths)
    if text_inputs.size(1) + 2 > max_text:
        max_len = max_text - 2
        text_inputs = text_inputs[:, :max_len]
        text_lengths = torch.clamp(text_lengths, max=max_len)
    text_inputs = F.pad(text_inputs, (0, 1), value=base_model.stop_text_token)
    text_inputs, text_targets = base_model.build_aligned_inputs_and_targets(
        text_inputs, base_model.start_text_token, base_model.stop_text_token
    )
    
    # Process mel codes (the audio semantic codes)
    mel_inputs = base_model.set_mel_padding(codes.clone(), code_lengths)
    if mel_inputs.size(1) + 2 > max_mel:
        max_len = max_mel - 2
        mel_inputs = mel_inputs[:, :max_len]
        code_lengths = torch.clamp(code_lengths, max=max_len)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=base_model.stop_mel_token)
    mel_inputs, mel_targets = base_model.build_aligned_inputs_and_targets(
        mel_inputs, base_model.start_mel_token, base_model.stop_mel_token
    )
    
    # Embed
    text_emb = base_model.text_embedding(text_inputs) + base_model.text_pos_embedding(text_inputs)
    mel_emb = base_model.mel_embedding(mel_inputs) + base_model.mel_pos_embedding(mel_inputs)
    
    # Get logits
    text_logits, mel_logits = base_model.get_logits(
        conds, text_emb, base_model.text_head, mel_emb, base_model.mel_head
    )
    
    # Create masks for actual sequence lengths (not padding)
    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )
    
    # Compute cross-entropy losses
    text_ce = F.cross_entropy(text_logits, text_targets, reduction='none')
    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction='none')
    
    # Apply masks
    text_loss_per_sample = (text_ce * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1)
    mel_loss_per_sample = (mel_ce * mel_mask).sum(dim=1) / mel_mask.sum(dim=1).clamp_min(1)
    
    # Apply extra weight to samples with stutters!
    # This makes the model prioritize learning stutter patterns
    sample_weights = torch.ones(batch_size, device=device)
    sample_weights[has_stutters] = stutter_weight
    
    text_loss = (text_loss_per_sample * sample_weights).mean()
    mel_loss = (mel_loss_per_sample * sample_weights).mean()
    
    # Combined loss
    total_loss = text_weight * text_loss + (1.0 - text_weight) * mel_loss
    
    # Compute accuracy metrics
    with torch.no_grad():
        mel_pred = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_tgt = mel_targets.reshape(-1)
        mel_m = mel_mask.reshape(-1)
        
        if mel_m.any():
            top1 = (mel_pred[mel_m].argmax(-1) == mel_tgt[mel_m]).float().mean().item()
            # Top-10 accuracy
            top10_pred = mel_pred[mel_m].topk(10, dim=-1).indices
            top10 = (top10_pred == mel_tgt[mel_m].unsqueeze(1)).any(dim=1).float().mean().item()
        else:
            top1 = 0.0
            top10 = 0.0
    
    metrics = {
        "text_loss": text_loss.item(),
        "mel_loss": mel_loss.item(),
        "mel_top1": top1,
        "mel_top10": top10,
        "stutter_samples": has_stutters.sum().item(),
    }
    
    return total_loss, metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    text_weight: float,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    totals = {"mel_loss": 0.0, "mel_top1": 0.0, "mel_top10": 0.0}
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            _, metrics = compute_loss(
                model, batch, device, text_weight, stutter_weight=1.0
            )
            bsz = batch["text_ids"].size(0)
            totals["mel_loss"] += metrics["mel_loss"] * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            totals["mel_top10"] += metrics["mel_top10"] * bsz
            count += bsz
    
    model.train()
    
    return {k: v / max(count, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    
    # Try verbatim dataset first, then fall back to v3/v2
    if args.train_manifest:
        train_manifest = args.train_manifest
    else:
        candidates = [
            speaker_dir / "dataset" / "processed_verbatim" / "train_manifest.jsonl",
            speaker_dir / "dataset" / "processed_v3" / "train_manifest.jsonl",
            speaker_dir / "dataset" / "processed_v2" / "train_manifest.jsonl",
        ]
        train_manifest = None
        for c in candidates:
            if c.exists():
                train_manifest = c
                break
    
    if args.val_manifest:
        val_manifest = args.val_manifest
    else:
        val_manifest = train_manifest.parent / "val_manifest.jsonl" if train_manifest else None
    
    if not train_manifest or not train_manifest.exists():
        print(f"❌ Manifest not found.")
        print("\nFirst prepare your verbatim dataset:")
        print(f"  python tools/prepare_verbatim_dataset.py --speaker {args.speaker}")
        sys.exit(1)
    
    output_dir = args.output_dir or speaker_dir / "verbatim_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs" / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print("=" * 60)
    print("VERBATIM LORA TRAINING")
    print("=" * 60)
    print(f"""
Speaker: {args.speaker}
LoRA rank: {args.lora_rank}
LoRA alpha: {args.lora_alpha}
Epochs: {args.epochs}
Batch size: {args.batch_size} × {args.grad_accumulation} = {args.batch_size * args.grad_accumulation} effective
Learning rate: {args.learning_rate}
Stutter weight: {args.stutter_weight}x

Train manifest: {train_manifest}
Output: {output_dir}
""")
    
    # Load base model
    print("\n[1/3] Loading base model...")
    model = build_model(args.config, args.base_checkpoint, device)
    
    # Apply LoRA (PEFT handles freezing base model automatically)
    print("\n[2/3] Applying LoRA adapters...")
    model = apply_lora_to_model(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        include_gpt=True,
        include_conditioning=args.include_conditioning,
        include_heads=args.include_heads,
    )
    
    # CRITICAL: Enable input gradients for gradient checkpointing to work with LoRA
    # Since UnifiedVoice doesn't have enable_input_require_grads(), we implement it manually
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    
    # Get the base model from PEFT wrapper
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    
    # Add hook to embedding layer to make outputs require grad
    base_model.text_embedding.register_forward_hook(make_inputs_require_grad)
    base_model.mel_embedding.register_forward_hook(make_inputs_require_grad)
    
    param_stats = get_trainable_parameters(model)
    trainable = param_stats["trainable_params"]
    total = param_stats["all_params"]
    print(f"  Trainable: {trainable:,} / {total:,} ({param_stats['trainable_percentage']:.2f}%)")
    
    # Load datasets
    print("\n[3/3] Loading datasets...")
    train_dataset = VerbatimDataset(train_manifest)
    val_dataset = VerbatimDataset(val_manifest) if val_manifest and val_manifest.exists() else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Scheduler
    total_steps = args.epochs * len(train_loader) // args.grad_accumulation
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # AMP
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_stutter_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, metrics = compute_loss(
                    model, batch, device,
                    text_weight=args.text_weight,
                    stutter_weight=args.stutter_weight,
                )
            
            scaled_loss = loss / args.grad_accumulation
            
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            bsz = batch["text_ids"].size(0)
            epoch_loss += loss.item() * bsz
            epoch_samples += bsz
            epoch_stutter_samples += metrics["stutter_samples"]
            
            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
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
                    writer.add_scalar("train/loss", metrics["mel_loss"], global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    writer.add_scalar("train/mel_top10", metrics["mel_top10"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                
                pbar.set_postfix({
                    "loss": f"{metrics['mel_loss']:.4f}",
                    "top1": f"{metrics['mel_top1']:.3f}",
                    "top10": f"{metrics['mel_top10']:.3f}",
                })
                
                # Checkpoint
                if global_step % args.save_interval == 0:
                    ckpt_dir = output_dir / f"checkpoint_step{global_step}"
                    save_lora_checkpoint(model, ckpt_dir, {"step": global_step})
        
        # End of epoch
        avg_train_loss = epoch_loss / max(epoch_samples, 1)
        
        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"stutter_samples={epoch_stutter_samples}/{epoch_samples}")
        
        # Validation
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, args.text_weight)
            
            writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
            writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
            writer.add_scalar("val/mel_top10", val_metrics["mel_top10"], global_step)
            
            print(f"  val_loss={val_metrics['mel_loss']:.4f}, "
                  f"val_top1={val_metrics['mel_top1']:.3f}")
            
            if val_metrics["mel_loss"] < best_val_loss:
                best_val_loss = val_metrics["mel_loss"]
                
                best_dir = output_dir / "best_checkpoint"
                save_lora_checkpoint(model, best_dir, {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "speaker": args.speaker,
                })
                print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    
    # Save final checkpoint
    final_dir = output_dir / "final_checkpoint"
    save_lora_checkpoint(model, final_dir, {
        "epochs": args.epochs,
        "final_val_loss": val_metrics["mel_loss"] if val_loader else avg_train_loss,
        "speaker": args.speaker,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    })
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
Best validation loss: {best_val_loss:.4f}

Output files:
  Final LoRA: {final_dir}
  Best LoRA: {output_dir / 'best_checkpoint'}

HOW TO USE:
===========
1. For inference WITH stutters, input verbatim text:

    from indextts import IndexTTS2
    
    tts = IndexTTS2(lora_path="{final_dir}")
    
    # Input with stutters → Output has stutters!
    tts.infer(
        spk_audio_prompt="reference.wav",
        text="I I I was going going to the store...",
        output_path="output.wav"
    )

2. For clean speech, input clean text:

    tts.infer(
        spk_audio_prompt="reference.wav", 
        text="I was going to the store.",
        output_path="output_clean.wav"
    )

The model has learned the DIRECT mapping:
  stuttered text → stuttered speech
  clean text → clean speech
""")


if __name__ == "__main__":
    main()