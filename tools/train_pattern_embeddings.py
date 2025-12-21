#!/usr/bin/env python3
"""
Pattern Embedding Training for IndexTTS2

This is THE NEW approach to training speaking patterns (stutters, pauses, etc.)
that ACTUALLY WORKS at inference time!

THE KEY DIFFERENCE:
==================
Old approach: Train LoRA on codes, hope patterns emerge
- Problem: Patterns tied to specific conditioning, don't transfer to inference

New approach: Train LEARNABLE PATTERN EMBEDDINGS
- Pattern embedding is injected into GPT conditioning
- Same embedding used in training AND inference
- Patterns transfer because the "trigger" is preserved!

How it works:
1. PatternEmbedding: Learnable tokens that encode "speak with THIS person's patterns"
2. Pattern-aware loss: Explicitly rewards pause/pattern reproduction
3. Combined training: LoRA + PatternEmbedding trained together

Usage:
    # Prepare dataset (extracts pattern features)
    python tools/prepare_pattern_dataset_v3.py --speaker ozzy
    
    # Train pattern embedding + LoRA
    python tools/train_pattern_embeddings.py \
        --speaker ozzy \
        --epochs 40 \
        --pattern-tokens 8
    
    # Inference (patterns will appear!)
    python tools/infer_with_patterns.py \
        --speaker ozzy \
        --text "Hello world"
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.lora_utils import (
    apply_lora_to_model,
    save_lora_checkpoint,
    get_trainable_parameters,
)
from indextts.pattern_embeddings import (
    PatternEmbedding,
    PatternExtractor,
    PatternAwareLoss,
    PatternFeatures,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Pattern Embeddings for speaking pattern reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Speaker/data
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--train-manifest", type=Path, help="Custom train manifest")
    parser.add_argument("--val-manifest", type=Path, help="Custom val manifest")
    
    # Pattern embedding
    parser.add_argument("--pattern-tokens", type=int, default=8,
                        help="Number of learnable pattern tokens (default: 8)")
    parser.add_argument("--pattern-lr", type=float, default=1e-3,
                        help="Learning rate for pattern embedding (default: 1e-3)")
    parser.add_argument("--injection-mode", choices=["add", "prepend", "replace_first"],
                        default="add", help="How to inject pattern embedding")
    
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true",
                        help="Train only pattern embedding, no LoRA")
    
    # Training
    parser.add_argument("--epochs", type=int, default=40,
                        help="Training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grad-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate for LoRA (default: 5e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    
    # Loss weights
    parser.add_argument("--pause-weight", type=float, default=2.0,
                        help="Weight for pause loss (default: 2.0)")
    parser.add_argument("--rate-weight", type=float, default=0.5,
                        help="Weight for rate loss (default: 0.5)")
    
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
class PatternSample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    # Pattern-specific
    pattern_features_path: Optional[Path] = None


class PatternDataset(Dataset):
    """Dataset that loads pattern features alongside standard features."""
    
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.base_dir = manifest_path.parent
        self.samples: List[PatternSample] = []
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
                
                sample = PatternSample(
                    id=record["id"],
                    text_ids_path=self._resolve_path(record["text_ids_path"]),
                    codes_path=self._resolve_path(record["codes_path"]),
                    condition_path=self._resolve_path(record["condition_path"]),
                    emo_vec_path=self._resolve_path(record["emo_vec_path"]),
                    text_len=int(record["text_len"]),
                    code_len=int(record["code_len"]),
                    condition_len=int(record.get("condition_len", 32)),
                    pattern_features_path=(
                        self._resolve_path(record["pattern_features_path"])
                        if "pattern_features_path" in record else None
                    ),
                )
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {self.manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        text_ids = np.load(sample.text_ids_path, allow_pickle=False).astype(np.int64)
        codes = np.load(sample.codes_path, allow_pickle=False).astype(np.int64)
        condition = np.load(sample.condition_path, allow_pickle=False).astype(np.float32)
        emo_vec = np.load(sample.emo_vec_path, allow_pickle=False).astype(np.float32)
        
        # Load pattern features if available
        pattern_features = None
        if sample.pattern_features_path and sample.pattern_features_path.exists():
            pf_data = np.load(sample.pattern_features_path, allow_pickle=True).item()
            pattern_features = PatternFeatures(**pf_data)
        
        return {
            "id": sample.id,
            "text_ids": torch.from_numpy(text_ids),
            "codes": torch.from_numpy(codes),
            "condition": torch.from_numpy(condition),
            "emo_vec": torch.from_numpy(emo_vec),
            "text_len": torch.tensor(sample.text_len, dtype=torch.long),
            "code_len": torch.tensor(sample.code_len, dtype=torch.long),
            "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
            "pattern_features": pattern_features,
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
    
    # Collect pattern features (may be None for some samples)
    pattern_features = [item["pattern_features"] for item in batch]
    
    return {
        "ids": [item["id"] for item in batch],
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
        "pattern_features": pattern_features,
    }


def load_tokenizer(path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    return TextTokenizer(str(path), normalizer)


def build_model(cfg_path: Path, tokenizer: TextTokenizer, checkpoint_path: Path, device: torch.device) -> UnifiedVoice:
    """Load base GPT model."""
    cfg = OmegaConf.load(cfg_path)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint)
    
    # Detect model_dim from checkpoint
    if "mel_pos_embedding.emb.weight" in raw_state:
        checkpoint_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != checkpoint_dim:
            cfg.gpt.model_dim = checkpoint_dim
    
    # Filter state dict
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


def compute_pattern_aware_loss(
    model: nn.Module,
    pattern_embedding: PatternEmbedding,
    batch: Dict,
    device: torch.device,
    loss_fn: PatternAwareLoss,
    injection_mode: str = "add",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with pattern embedding injection.
    
    This is the KEY difference from standard training!
    """
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)
    pattern_features = batch.get("pattern_features")
    
    batch_size = text_ids.size(0)
    
    # === INJECT PATTERN EMBEDDING ===
    # This is what makes patterns learnable and transferable!
    pattern_conditioned = pattern_embedding.get_injection_embedding(
        condition,
        injection_mode=injection_mode,
    )
    
    # Build conditioning with pattern injection
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
    duration_ctrl = base_model.speed_emb(torch.ones_like(use_speed))
    duration_free = base_model.speed_emb(torch.zeros_like(use_speed))
    
    # Add emotion and build final conditioning
    conds = torch.cat(
        (pattern_conditioned + emo_vec.unsqueeze(1), 
         duration_ctrl.unsqueeze(1), 
         duration_free.unsqueeze(1)),
        dim=1,
    )
    
    # Process inputs (same as standard training)
    max_text = base_model.text_pos_embedding.emb.num_embeddings
    max_mel = base_model.mel_pos_embedding.emb.num_embeddings
    
    text_inputs = base_model.set_text_padding(text_ids.clone(), text_lengths)
    if text_inputs.size(1) + 2 > max_text:
        max_len = max_text - 2
        text_inputs = text_inputs[:, :max_len]
        text_lengths = torch.clamp(text_lengths, max=max_len)
    text_inputs = F.pad(text_inputs, (0, 1), value=base_model.stop_text_token)
    text_inputs, text_targets = base_model.build_aligned_inputs_and_targets(
        text_inputs, base_model.start_text_token, base_model.stop_text_token
    )
    
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
    
    # Masks
    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )
    
    # === PATTERN-AWARE LOSS ===
    mel_loss, mel_metrics = loss_fn(mel_logits, mel_targets, mel_mask, pattern_features)
    
    # Standard text loss
    text_ce = F.cross_entropy(text_logits, text_targets, reduction='none')
    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    
    # Combined
    total_loss = 0.2 * text_loss + 0.8 * mel_loss
    
    # Metrics
    with torch.no_grad():
        mel_pred = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_tgt = mel_targets.reshape(-1)
        mel_m = mel_mask.reshape(-1)
        if mel_m.any():
            top1 = (mel_pred[mel_m].argmax(-1) == mel_tgt[mel_m]).float().mean().item()
        else:
            top1 = 0.0
    
    metrics = {
        "text_loss": text_loss.item(),
        "mel_loss": mel_metrics["total_loss"],
        "mel_ce": mel_metrics.get("ce_loss", 0.0),
        "pause_loss": mel_metrics.get("pause_loss", 0.0),
        "rate_loss": mel_metrics.get("rate_loss", 0.0),
        "mel_top1": top1,
        "pattern_scale": pattern_embedding.pattern_scale.item(),
    }
    
    return total_loss, metrics


def evaluate(
    model: nn.Module,
    pattern_embedding: PatternEmbedding,
    loader: DataLoader,
    device: torch.device,
    loss_fn: PatternAwareLoss,
    injection_mode: str,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    pattern_embedding.eval()
    
    totals = {"mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            _, metrics = compute_pattern_aware_loss(
                model, pattern_embedding, batch, device, loss_fn, injection_mode
            )
            bsz = batch["text_ids"].size(0)
            totals["mel_loss"] += metrics["mel_loss"] * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
    
    model.train()
    pattern_embedding.train()
    
    return {k: v / max(count, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    
    train_manifest = args.train_manifest or speaker_dir / "dataset" / "processed_v3" / "train_manifest.jsonl"
    val_manifest = args.val_manifest or speaker_dir / "dataset" / "processed_v3" / "val_manifest.jsonl"
    
    # Fall back to v2 if v3 doesn't exist
    if not train_manifest.exists():
        train_manifest = speaker_dir / "dataset" / "processed_v2" / "train_manifest.jsonl"
        val_manifest = speaker_dir / "dataset" / "processed_v2" / "val_manifest.jsonl"
    
    if not train_manifest.exists():
        print(f"❌ Manifest not found: {train_manifest}")
        print("\nFirst prepare your dataset:")
        print(f"  python tools/prepare_pattern_dataset_v3.py --speaker {args.speaker}")
        sys.exit(1)
    
    output_dir = args.output_dir or speaker_dir / "pattern_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs" / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print("=" * 60)
    print("PATTERN EMBEDDING TRAINING")
    print("=" * 60)
    print(f"""
Speaker: {args.speaker}
Pattern tokens: {args.pattern_tokens}
Injection mode: {args.injection_mode}
LoRA rank: {args.lora_rank if not args.no_lora else 'DISABLED'}
Epochs: {args.epochs}
""")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Load base model
    print("\n[1/4] Loading base model...")
    model = build_model(args.config, tokenizer, args.base_checkpoint, device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA (optional)
    if not args.no_lora:
        print("\n[2/4] Applying LoRA adapters...")
        model = apply_lora_to_model(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            include_gpt=True,
            include_conditioning=True,
            include_heads=True,
        )
        model_params = [p for p in model.parameters() if p.requires_grad]
    else:
        print("\n[2/4] LoRA disabled, training pattern embedding only...")
        model_params = []
    
    # Create pattern embedding
    print("\n[3/4] Creating pattern embedding...")
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    model_dim = base_model.model_dim
    
    pattern_embedding = PatternEmbedding(
        model_dim=model_dim,
        num_pattern_tokens=args.pattern_tokens,
    ).to(device)
    
    print(f"  Pattern tokens: {args.pattern_tokens}")
    print(f"  Model dimension: {model_dim}")
    print(f"  Pattern embedding size: {sum(p.numel() for p in pattern_embedding.parameters()):,}")
    
    # Load datasets
    print("\n[4/4] Loading datasets...")
    train_dataset = PatternDataset(train_manifest)
    val_dataset = PatternDataset(val_manifest)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    
    # Optimizer - separate LR for pattern embedding
    param_groups = []
    if model_params:
        param_groups.append({
            'params': model_params,
            'lr': args.learning_rate,
            'weight_decay': 0.01,
        })
    param_groups.append({
        'params': pattern_embedding.parameters(),
        'lr': args.pattern_lr,  # Often higher for pattern embedding
        'weight_decay': 0.0,  # No decay for embedding
    })
    
    optimizer = AdamW(param_groups)
    
    # Scheduler
    total_steps = args.epochs * len(train_loader) // args.grad_accumulation
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Loss function with pattern-aware components
    loss_fn = PatternAwareLoss(
        pause_weight=args.pause_weight,
        rate_weight=args.rate_weight,
    )
    
    # AMP
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    model.train()
    pattern_embedding.train()
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, metrics = compute_pattern_aware_loss(
                    model, pattern_embedding, batch, device, loss_fn, args.injection_mode
                )
            
            scaled_loss = loss / args.grad_accumulation
            
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            epoch_loss += loss.item() * batch["text_ids"].size(0)
            epoch_samples += batch["text_ids"].size(0)
            
            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(pattern_embedding.parameters(), args.grad_clip)
                
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
                    writer.add_scalar("train/pattern_scale", metrics["pattern_scale"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    
                    if metrics.get("pause_loss", 0) > 0:
                        writer.add_scalar("train/pause_loss", metrics["pause_loss"], global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['mel_loss']:.4f}",
                    "top1": f"{metrics['mel_top1']:.3f}",
                    "scale": f"{metrics['pattern_scale']:.3f}",
                })
                
                # Checkpoint
                if global_step % args.save_interval == 0:
                    ckpt_dir = output_dir / f"checkpoint_step{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    
                    # Save LoRA
                    if not args.no_lora:
                        save_lora_checkpoint(model, ckpt_dir / "lora", {"step": global_step})
                    
                    # Save pattern embedding
                    pattern_embedding.save(
                        ckpt_dir / "pattern_embedding.pt",
                        metadata={"step": global_step, "epoch": epoch}
                    )
        
        # End of epoch
        avg_train_loss = epoch_loss / max(epoch_samples, 1)
        
        # Validation
        val_metrics = evaluate(model, pattern_embedding, val_loader, device, loss_fn, args.injection_mode)
        
        writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
        writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
        
        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_metrics['mel_loss']:.4f}")
        
        if val_metrics["mel_loss"] < best_val_loss:
            best_val_loss = val_metrics["mel_loss"]
            
            # Save best checkpoint
            best_dir = output_dir / "best_checkpoint"
            best_dir.mkdir(exist_ok=True)
            
            if not args.no_lora:
                save_lora_checkpoint(model, best_dir / "lora", {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                })
            
            pattern_embedding.save(
                best_dir / "pattern_embedding.pt",
                metadata={
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "pattern_tokens": args.pattern_tokens,
                    "injection_mode": args.injection_mode,
                }
            )
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    
    # Save final checkpoint
    final_dir = output_dir / "final_checkpoint"
    final_dir.mkdir(exist_ok=True)
    
    if not args.no_lora:
        save_lora_checkpoint(model, final_dir / "lora", {
            "epochs": args.epochs,
            "final_val_loss": val_metrics["mel_loss"],
        })
    
    pattern_embedding.save(
        final_dir / "pattern_embedding.pt",
        metadata={
            "epochs": args.epochs,
            "final_val_loss": val_metrics["mel_loss"],
            "pattern_tokens": args.pattern_tokens,
            "injection_mode": args.injection_mode,
            "speaker": args.speaker,
        }
    )
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
Best validation loss: {best_val_loss:.4f}

Output files:
  Pattern embedding: {final_dir / 'pattern_embedding.pt'}
  {'LoRA checkpoint: ' + str(final_dir / 'lora') if not args.no_lora else '(No LoRA)'}
  Best checkpoint: {output_dir / 'best_checkpoint'}

To use for inference:
    python tools/infer_with_patterns.py \\
        --speaker {args.speaker} \\
        --text "Your text here"

WHY THIS WORKS:
The pattern embedding is a learnable "trigger" that tells the model
"speak with this person's patterns". It's used both in training AND 
inference, so patterns transfer!
""")


if __name__ == "__main__":
    main()