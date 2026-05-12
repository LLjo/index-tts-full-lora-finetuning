#!/usr/bin/env python3
"""
S2Mel LoRA Training for IndexTTS2 - PROSODIC PATTERN LEARNING

This script trains LoRA adapters on the S2Mel (Semantic-to-Mel) diffusion model
to learn prosodic patterns like stutters, pauses, and speech rhythm.

WHY S2Mel Training is the KEY for Stutters:
============================================
1. GPT Stage: text -> semantic codes (WHAT to say)
   - Training GPT only changes the semantic mapping
   - Stutters in text don't reliably transfer to acoustic patterns
   
2. S2Mel Stage: semantic codes -> mel spectrogram (HOW to say it) ← WE TRAIN THIS!
   - This is where prosodic patterns actually live
   - Training S2Mel learns the ACTUAL acoustic patterns of stutters
   - The mel spectrogram contains timing, rhythm, prosody

The S2Mel model uses Conditional Flow Matching (CFM) with a DiT transformer.
By training LoRA adapters on:
- DiT attention layers: learns mel pattern generation
- WaveNet layers: learns fine acoustic details
- Both: the model learns YOUR SPEAKER'S unique stutter patterns!

Architecture Deep Dive:
=======================
CFM (Conditional Flow Matching) contains:
├── DiT (Diffusion Transformer)
│   ├── transformer (gpt_fast style transformer)
│   │   ├── layers[0-N].attention (wqkv, wo) ← TRAIN THIS
│   │   └── layers[0-N].feed_forward (w1, w2, w3) ← TRAIN THIS
│   ├── x_embedder (input projection)
│   ├── cond_projection (condition projection)
│   ├── wavenet (for final refinement) ← TRAIN THIS
│   └── timestep/style embedders
├── Length Regulator (optional training)
└── GPT Layer (projection from GPT space)

The diffusion process:
1. Sample noise z ~ N(0,1)
2. Interpolate: y = (1-t)*z + t*x1 (where x1 is target mel)
3. DiT predicts the flow: dphi/dt
4. Solve ODE to get final mel

By training with your verbatim audio's mel spectrograms, the model learns
to generate mels that include YOUR speaker's stutter patterns!

Usage:
======
    # Basic training
    python tools/train_s2mel_lora.py --speaker ozzy
    
    # With options
    python tools/train_s2mel_lora.py \\
        --speaker ozzy \\
        --epochs 30 \\
        --lora-rank 16 \\
        --batch-size 2 \\
        --learning-rate 1e-4 \\
        --include-wavenet
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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.utils.s2mel_lora_utils import (
    apply_lora_to_s2mel,
    save_s2mel_lora_checkpoint,
    get_s2mel_trainable_parameters,
    print_s2mel_model_structure,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train S2Mel LoRA for prosodic pattern learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--train-manifest", type=Path, help="Custom train manifest")
    parser.add_argument("--val-manifest", type=Path, help="Custom validation manifest")
    parser.add_argument("--reference-features", type=Path,
                        help="Reference features file for consistent style")
    
    # GPT-aligned training (RECOMMENDED for stutter learning)
    parser.add_argument("--use-gpt-aligned", action="store_true", default=False,
                        help="Use GPT-aligned dataset (prepared with prepare_gpt_aligned_dataset.py). "
                             "RECOMMENDED for learning stutters - uses same representations as inference!")
    
    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (8-32 recommended, default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha scaling (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    
    # Module selection
    parser.add_argument("--include-dit", action="store_true", default=True,
                        help="Apply LoRA to DiT transformer (RECOMMENDED)")
    parser.add_argument("--include-wavenet", action="store_true", default=False,
                        help="NOT SUPPORTED - WaveNet uses custom Conv wrappers incompatible with PEFT")
    parser.add_argument("--include-length-regulator", action="store_true", default=False,
                        help="Apply LoRA to length regulator")
    parser.add_argument("--include-gpt-layer", action="store_true", default=False,
                        help="Apply LoRA to GPT projection layer")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (default: 2 - S2Mel is memory intensive)")
    parser.add_argument("--grad-accumulation", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision (default: True)")
    
    # Loss configuration
    parser.add_argument("--stutter-weight", type=float, default=2.0,
                        help="Extra weight for samples with stutters (default: 2.0)")
    parser.add_argument("--use-reference-style", action="store_true", default=True,
                        help="Use consistent reference style during training")
    
    # CFM/Diffusion settings
    parser.add_argument("--diffusion-steps", type=int, default=10,
                        help="Diffusion steps during training (default: 10)")
    
    # Output
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model paths
    parser.add_argument("--config", type=Path, 
                        default=PROJECT_ROOT / "checkpoints" / "config.yaml")
    parser.add_argument("--model-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class S2MelSample:
    """A single S2Mel training sample."""
    id: str
    mel_path: Path
    semantic_codes_path: Path
    style_path: Path
    prompt_condition_path: Path
    semantic_emb_path: Path
    mel_length: int
    semantic_length: int
    has_stutters: bool = False
    stutter_count: int = 0


@dataclass
class GPTAlignedSample:
    """A single GPT-aligned training sample (matches inference representation)."""
    id: str
    mel_path: Path
    s2mel_cond_path: Path
    style_path: Path
    mel_length: int
    code_length: int
    duration_ratio: float
    has_stutters: bool = False
    stutter_count: int = 0


class S2MelDataset(Dataset):
    """
    Dataset for S2Mel LoRA training.
    
    Each sample contains:
    - mel: Target mel spectrogram (what we want to generate)
    - semantic_codes: Semantic codes from W2V-BERT
    - style: Global style vector from CAMPPlus
    - prompt_condition: Conditioning from length regulator
    
    If reference_style is provided, uses it for ALL samples (recommended
    for consistent training that matches inference).
    """
    
    def __init__(
        self, 
        manifest_path: Path, 
        reference_style: Optional[torch.Tensor] = None,
        reference_prompt: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
    ):
        self.manifest_path = manifest_path
        self.reference_style = reference_style
        self.reference_prompt = reference_prompt
        self.reference_mel = reference_mel
        self.samples: List[S2MelSample] = []
        self._load_manifest()
    
    def _load_manifest(self):
        with open(self.manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                sample = S2MelSample(
                    id=record["id"],
                    mel_path=Path(record["mel_path"]),
                    semantic_codes_path=Path(record["semantic_codes_path"]),
                    style_path=Path(record["style_path"]),
                    prompt_condition_path=Path(record["prompt_condition_path"]),
                    semantic_emb_path=Path(record["semantic_emb_path"]),
                    mel_length=record["mel_length"],
                    semantic_length=record["semantic_length"],
                    has_stutters=record.get("has_stutters", False),
                    stutter_count=record.get("stutter_count", 0),
                )
                self.samples.append(sample)
        
        print(f"  Loaded {len(self.samples)} samples from {self.manifest_path}")
        stutter_count = sum(1 for s in self.samples if s.has_stutters)
        print(f"  Samples with stutters: {stutter_count}/{len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load target mel (what we want to learn to generate)
        mel = torch.from_numpy(
            np.load(sample.mel_path, allow_pickle=False).astype(np.float32)
        )
        
        # Load semantic condition
        semantic_emb = torch.from_numpy(
            np.load(sample.semantic_emb_path, allow_pickle=False).astype(np.float32)
        )
        
        # Style - use reference if available, else per-sample
        if self.reference_style is not None:
            style = self.reference_style.clone()
        else:
            style = torch.from_numpy(
                np.load(sample.style_path, allow_pickle=False).astype(np.float32)
            )
        
        # Prompt condition (for reference in CFM)
        if self.reference_prompt is not None:
            prompt_condition = self.reference_prompt.clone()
        else:
            prompt_condition = torch.from_numpy(
                np.load(sample.prompt_condition_path, allow_pickle=False).astype(np.float32)
            )
        
        # Reference mel for prompt
        if self.reference_mel is not None:
            ref_mel = self.reference_mel.clone()
        else:
            # Use first part of target mel as reference
            ref_mel = mel[..., :min(mel.shape[-1] // 3, 200)]
        
        return {
            "id": sample.id,
            "mel": mel.squeeze(0),  # (80, T)
            "semantic_emb": semantic_emb.squeeze(0),  # (T_semantic, 1024)
            "style": style.squeeze(0),  # (192,)
            "prompt_condition": prompt_condition.squeeze(0),  # (T_prompt, 768)
            "ref_mel": ref_mel.squeeze(0),  # (80, T_ref)
            "mel_length": sample.mel_length,
            "semantic_length": sample.semantic_length,
            "has_stutters": sample.has_stutters,
            "stutter_count": sample.stutter_count,
        }


class GPTAlignedDataset(Dataset):
    """
    Dataset for GPT-aligned S2Mel LoRA training.
    
    This dataset uses pre-computed GPT-style conditioning that MATCHES
    what happens at inference time. This is the KEY to making training
    transfer to inference!
    
    Each sample contains:
    - mel: Target mel spectrogram (from actual audio)
    - s2mel_cond: GPT-aligned conditioning (from GPT codes + latent)
    - style: Global style vector
    
    The s2mel_cond was generated using the SAME process as inference:
    Text → GPT → codes → vq2emb + latent → length_regulator → cond
    """
    
    def __init__(
        self,
        manifest_path: Path,
        reference_style: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
    ):
        self.manifest_path = manifest_path
        self.reference_style = reference_style
        self.reference_mel = reference_mel
        self.samples: List[GPTAlignedSample] = []
        self._load_manifest()
    
    def _load_manifest(self):
        with open(self.manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                sample = GPTAlignedSample(
                    id=record["id"],
                    mel_path=Path(record["mel_path"]),
                    s2mel_cond_path=Path(record["s2mel_cond_path"]),
                    style_path=Path(record["style_path"]),
                    mel_length=record["mel_length"],
                    code_length=record["code_length"],
                    duration_ratio=record.get("duration_ratio", 1.72),
                    has_stutters=record.get("has_stutters", False),
                    stutter_count=record.get("stutter_count", 0),
                )
                self.samples.append(sample)
        
        print(f"  Loaded {len(self.samples)} GPT-aligned samples from {self.manifest_path}")
        stutter_count = sum(1 for s in self.samples if s.has_stutters)
        print(f"  Samples with stutters: {stutter_count}/{len(self.samples)}")
        
        # Calculate average duration ratio
        avg_ratio = np.mean([s.duration_ratio for s in self.samples])
        print(f"  Average duration ratio (mel/code): {avg_ratio:.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load target mel (what we want to learn to generate)
        mel = torch.from_numpy(
            np.load(sample.mel_path, allow_pickle=False).astype(np.float32)
        )
        
        # Load GPT-aligned S2Mel conditioning (THIS IS THE KEY!)
        # This conditioning was generated using the SAME process as inference
        s2mel_cond = torch.from_numpy(
            np.load(sample.s2mel_cond_path, allow_pickle=False).astype(np.float32)
        )
        
        # Style - use reference if available, else per-sample
        if self.reference_style is not None:
            style = self.reference_style.clone()
        else:
            style = torch.from_numpy(
                np.load(sample.style_path, allow_pickle=False).astype(np.float32)
            )
        
        # Reference mel for prompt
        if self.reference_mel is not None:
            ref_mel = self.reference_mel.clone()
        else:
            # Use first part of target mel as reference
            ref_mel = mel[..., :min(mel.shape[-1] // 3, 200)]
        
        return {
            "id": sample.id,
            "mel": mel.squeeze(0),  # (80, T)
            "s2mel_cond": s2mel_cond.squeeze(0),  # (T, 768) - PRE-COMPUTED GPT-ALIGNED!
            "style": style.squeeze(0),  # (192,)
            "ref_mel": ref_mel.squeeze(0),  # (80, T_ref)
            "mel_length": sample.mel_length,
            "has_stutters": sample.has_stutters,
            "stutter_count": sample.stutter_count,
            "duration_ratio": sample.duration_ratio,
        }


def collate_gpt_aligned_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate GPT-aligned batch with padding.
    
    For GPT-aligned training, the s2mel_cond is PRE-COMPUTED to match
    inference exactly. We just need to pad it.
    
    CRITICAL: cond and mel must have the same temporal dimension!
    The CFM expects cond.shape[1] == mel.shape[2]
    """
    # Find max lengths
    max_mel_len = max(item["mel"].shape[-1] for item in batch)
    max_ref_mel_len = max(item["ref_mel"].shape[-1] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    # CRITICAL: Use max_mel_len for BOTH mel and cond to ensure they match!
    mels = torch.zeros(batch_size, 80, max_mel_len)
    s2mel_conds = torch.zeros(batch_size, max_mel_len, batch[0]["s2mel_cond"].shape[-1])
    styles = torch.zeros(batch_size, 192)
    ref_mels = torch.zeros(batch_size, 80, max_ref_mel_len)
    
    mel_lengths = []
    ref_mel_lengths = []
    has_stutters = []
    
    for i, item in enumerate(batch):
        mel_len = item["mel"].shape[-1]
        cond_len = item["s2mel_cond"].shape[0]
        ref_mel_len = item["ref_mel"].shape[-1]
        
        # Ensure cond length matches mel length
        # If cond is shorter, pad it. If longer, truncate it.
        actual_cond_len = min(cond_len, mel_len)
        
        mels[i, :, :mel_len] = item["mel"]
        s2mel_conds[i, :actual_cond_len, :] = item["s2mel_cond"][:actual_cond_len, :]
        
        # If cond was shorter than mel, we already padded with zeros
        # If cond was longer, we truncated to mel_len
        
        styles[i, :] = item["style"]
        ref_mels[i, :, :ref_mel_len] = item["ref_mel"]
        
        mel_lengths.append(mel_len)
        ref_mel_lengths.append(ref_mel_len)
        has_stutters.append(item["has_stutters"])
    
    return {
        "ids": [item["id"] for item in batch],
        "mel": mels,  # (B, 80, T_mel)
        "s2mel_cond": s2mel_conds,  # (B, T_mel, 768) - MATCHED to mel length!
        "style": styles,  # (B, 192)
        "ref_mel": ref_mels,  # (B, 80, T_ref)
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
        "ref_mel_lengths": torch.tensor(ref_mel_lengths, dtype=torch.long),
        "has_stutters": torch.tensor(has_stutters, dtype=torch.bool),
    }


def collate_s2mel_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate S2Mel batch with padding.
    
    S2Mel expects:
    - x1: (B, 80, T) - target mel
    - mu: (B, T, dim) - semantic condition
    - style: (B, 192) - global style
    - prompt: (B, 80, T_prompt) - reference mel
    """
    # Find max lengths
    max_mel_len = max(item["mel"].shape[-1] for item in batch)
    max_semantic_len = max(item["semantic_emb"].shape[0] for item in batch)
    max_prompt_len = max(item["prompt_condition"].shape[0] for item in batch)
    max_ref_mel_len = max(item["ref_mel"].shape[-1] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    mels = torch.zeros(batch_size, 80, max_mel_len)
    semantic_embs = torch.zeros(batch_size, max_semantic_len, batch[0]["semantic_emb"].shape[-1])
    styles = torch.zeros(batch_size, 192)
    prompt_conditions = torch.zeros(batch_size, max_prompt_len, batch[0]["prompt_condition"].shape[-1])
    ref_mels = torch.zeros(batch_size, 80, max_ref_mel_len)
    
    mel_lengths = []
    semantic_lengths = []
    prompt_lengths = []
    ref_mel_lengths = []
    has_stutters = []
    
    for i, item in enumerate(batch):
        mel_len = item["mel"].shape[-1]
        semantic_len = item["semantic_emb"].shape[0]
        prompt_len = item["prompt_condition"].shape[0]
        ref_mel_len = item["ref_mel"].shape[-1]
        
        mels[i, :, :mel_len] = item["mel"]
        semantic_embs[i, :semantic_len, :] = item["semantic_emb"]
        styles[i, :] = item["style"]
        prompt_conditions[i, :prompt_len, :] = item["prompt_condition"]
        ref_mels[i, :, :ref_mel_len] = item["ref_mel"]
        
        mel_lengths.append(mel_len)
        semantic_lengths.append(semantic_len)
        prompt_lengths.append(prompt_len)
        ref_mel_lengths.append(ref_mel_len)
        has_stutters.append(item["has_stutters"])
    
    return {
        "ids": [item["id"] for item in batch],
        "mel": mels,  # (B, 80, T)
        "semantic_emb": semantic_embs,  # (B, T_semantic, dim)
        "style": styles,  # (B, 192)
        "prompt_condition": prompt_conditions,  # (B, T_prompt, dim)
        "ref_mel": ref_mels,  # (B, 80, T_ref)
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
        "semantic_lengths": torch.tensor(semantic_lengths, dtype=torch.long),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "ref_mel_lengths": torch.tensor(ref_mel_lengths, dtype=torch.long),
        "has_stutters": torch.tensor(has_stutters, dtype=torch.bool),
    }


def build_s2mel_model(cfg_path: Path, model_dir: Path, device: torch.device) -> nn.Module:
    """Load base S2Mel model."""
    cfg = OmegaConf.load(cfg_path)
    
    s2mel_path = str(model_dir / cfg.s2mel_checkpoint)
    
    print(f"  Loading S2Mel from: {s2mel_path}")
    
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel, None, s2mel_path,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    
    # Move to device FIRST so setup_caches uses correct device
    s2mel = s2mel.to(device)
    
    # Setup caches (this creates freqs_cis buffer on the correct device)
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=4, max_seq_length=8192)
    
    # Ensure all DiT buffers are on the correct device
    dit = s2mel.models['cfm'].estimator
    if hasattr(dit, 'input_pos'):
        dit.input_pos = dit.input_pos.to(device)
    if hasattr(dit, 'transformer'):
        if hasattr(dit.transformer, 'freqs_cis') and dit.transformer.freqs_cis is not None:
            dit.transformer.freqs_cis = dit.transformer.freqs_cis.to(device)
        if hasattr(dit.transformer, 'causal_mask') and dit.transformer.causal_mask is not None:
            dit.transformer.causal_mask = dit.transformer.causal_mask.to(device)
    
    return s2mel


def compute_cfm_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    stutter_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Conditional Flow Matching loss for S2Mel training.
    
    The CFM loss trains the model to predict the flow from noise to target mel.
    This is how the diffusion model learns the mel patterns including stutters!
    
    Process:
    1. Sample random timestep t ~ U(0,1)
    2. Interpolate: y = (1 - (1-σ_min)*t) * z + t * x1
    3. Compute target flow: u = x1 - (1-σ_min) * z
    4. DiT predicts flow from y
    5. Loss = ||predicted_flow - target_flow||
    """
    # Access the underlying model if wrapped by PEFT
    if hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model'):
            base = base.model
    else:
        base = model
        
    cfm = base.models['cfm']
    length_regulator = base.models['length_regulator']
    
    # Move to device
    x1 = batch["mel"].to(device)  # Target mel (B, 80, T)
    semantic_emb = batch["semantic_emb"].to(device)  # (B, T_semantic, 1024)
    style = batch["style"].to(device)  # (B, 192)
    ref_mel = batch["ref_mel"].to(device)  # (B, 80, T_ref)
    mel_lengths = batch["mel_lengths"].to(device)
    prompt_lengths = batch["ref_mel_lengths"].to(device)
    has_stutters = batch["has_stutters"].to(device)
    
    B = x1.size(0)
    
    # Ensure DiT transformer buffers are on correct device (needed for training)
    if hasattr(cfm, 'estimator') and hasattr(cfm.estimator, 'transformer'):
        transformer = cfm.estimator.transformer
        if hasattr(transformer, 'freqs_cis') and transformer.freqs_cis is not None:
            if transformer.freqs_cis.device != device:
                transformer.freqs_cis = transformer.freqs_cis.to(device)
        if hasattr(transformer, 'causal_mask') and transformer.causal_mask is not None:
            if transformer.causal_mask.device != device:
                transformer.causal_mask = transformer.causal_mask.to(device)
    if hasattr(cfm, 'estimator') and hasattr(cfm.estimator, 'input_pos'):
        if cfm.estimator.input_pos is not None and cfm.estimator.input_pos.device != device:
            cfm.estimator.input_pos = cfm.estimator.input_pos.to(device)
    
    # Get condition through length regulator
    # First transpose semantic_emb for length regulator: (B, T, C) -> need (B, 1, T) codes
    # But we have embeddings, so we use them directly as continuous condition
    mu = semantic_emb.transpose(1, 2)  # (B, 1024, T_semantic)
    
    # Regress to mel length
    target_lengths = mel_lengths
    cond = length_regulator(
        mu.transpose(1, 2),  # (B, T_semantic, 1024)
        ylens=target_lengths,
        n_quantizers=3,
        f0=None,
    )[0]  # (B, T_mel, 768)
    
    # Combine with prompt condition
    # For training, we use the target mel's first part as prompt
    prompt_mel_len = prompt_lengths.max().item()
    
    # The CFM forward pass computes the flow matching loss
    # CFM.forward(x1, x_lens, prompt_lens, mu, style)
    loss, _ = cfm(
        x1,  # (B, 80, T) - target mel
        mel_lengths,  # mel lengths
        prompt_lengths,  # prompt lengths
        cond,  # (B, T_mel, 768) - semantic condition
        style,  # (B, 192) - global style
    )
    
    # Apply stutter weighting
    if stutter_weight != 1.0 and has_stutters.any():
        # Compute per-sample loss would require modifying CFM
        # For now, apply global weight if batch has stutters
        stutter_ratio = has_stutters.float().mean()
        effective_weight = 1.0 + (stutter_weight - 1.0) * stutter_ratio
        loss = loss * effective_weight
    
    metrics = {
        "cfm_loss": loss.item(),
        "stutter_samples": has_stutters.sum().item(),
    }
    
    return loss, metrics


def compute_gpt_aligned_cfm_loss(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    stutter_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute CFM loss for GPT-aligned training.
    
    This is the KEY function for making stutter training work!
    
    The difference from standard training:
    - Standard: Uses W2V-BERT embeddings → length_regulator → cond
    - GPT-aligned: Uses PRE-COMPUTED cond (from GPT codes + latent like inference)
    
    This ensures training uses the SAME conditioning representation as inference,
    so the learned patterns actually transfer!
    """
    # Access the underlying model if wrapped by PEFT
    if hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model'):
            base = base.model
    else:
        base = model
        
    cfm = base.models['cfm']
    
    # Move to device
    x1 = batch["mel"].to(device)  # Target mel (B, 80, T)
    cond = batch["s2mel_cond"].to(device)  # PRE-COMPUTED GPT-aligned condition (B, T, 768)
    style = batch["style"].to(device)  # (B, 192)
    mel_lengths = batch["mel_lengths"].to(device)
    prompt_lengths = batch["ref_mel_lengths"].to(device)
    has_stutters = batch["has_stutters"].to(device)
    
    B = x1.size(0)
    
    # Ensure DiT transformer buffers are on correct device
    if hasattr(cfm, 'estimator') and hasattr(cfm.estimator, 'transformer'):
        transformer = cfm.estimator.transformer
        if hasattr(transformer, 'freqs_cis') and transformer.freqs_cis is not None:
            if transformer.freqs_cis.device != device:
                transformer.freqs_cis = transformer.freqs_cis.to(device)
        if hasattr(transformer, 'causal_mask') and transformer.causal_mask is not None:
            if transformer.causal_mask.device != device:
                transformer.causal_mask = transformer.causal_mask.to(device)
    if hasattr(cfm, 'estimator') and hasattr(cfm.estimator, 'input_pos'):
        if cfm.estimator.input_pos is not None and cfm.estimator.input_pos.device != device:
            cfm.estimator.input_pos = cfm.estimator.input_pos.to(device)
    
    # The CFM forward pass computes the flow matching loss
    # cond is ALREADY computed from GPT codes - no need for length_regulator!
    loss, _ = cfm(
        x1,  # (B, 80, T) - target mel
        mel_lengths,  # mel lengths
        prompt_lengths,  # prompt lengths
        cond,  # (B, T_mel, 768) - GPT-ALIGNED condition (this is the key!)
        style,  # (B, 192) - global style
    )
    
    # Apply stutter weighting
    if stutter_weight != 1.0 and has_stutters.any():
        stutter_ratio = has_stutters.float().mean()
        effective_weight = 1.0 + (stutter_weight - 1.0) * stutter_ratio
        loss = loss * effective_weight
    
    metrics = {
        "cfm_loss": loss.item(),
        "stutter_samples": has_stutters.sum().item(),
    }
    
    return loss, metrics


def evaluate_gpt_aligned(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on GPT-aligned validation set."""
    was_training = model.training
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            loss, _ = compute_gpt_aligned_cfm_loss(model, batch, device, stutter_weight=1.0)
            bsz = batch["mel"].size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz
    
    if was_training:
        model.train()
    
    return {
        "val_loss": total_loss / max(total_samples, 1),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Note: We keep model in training mode because DiT's forward has a bug where
    prompt_lens is passed to mask_content parameter. In training mode,
    `if not self.training and mask_content` short-circuits correctly.
    Using torch.no_grad() still prevents gradient computation.
    """
    # Don't call model.eval() - DiT's forward expects training mode
    # because prompt_lens tensor gets passed to mask_content parameter
    was_training = model.training
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            loss, _ = compute_cfm_loss(model, batch, device, stutter_weight=1.0)
            bsz = batch["mel"].size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz
    
    # Restore original mode (should already be training)
    if was_training:
        model.train()
    
    return {
        "val_loss": total_loss / max(total_samples, 1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    
    # Determine dataset paths based on training mode
    use_gpt_aligned = args.use_gpt_aligned
    
    if args.train_manifest:
        train_manifest = args.train_manifest
    elif use_gpt_aligned:
        train_manifest = speaker_dir / "dataset" / "processed_gpt_aligned" / "train_manifest.jsonl"
    else:
        train_manifest = speaker_dir / "dataset" / "processed_s2mel" / "train_manifest.jsonl"
    
    if args.val_manifest:
        val_manifest = args.val_manifest
    elif use_gpt_aligned:
        val_manifest = speaker_dir / "dataset" / "processed_gpt_aligned" / "val_manifest.jsonl"
    else:
        val_manifest = speaker_dir / "dataset" / "processed_s2mel" / "val_manifest.jsonl"
    
    if not train_manifest.exists():
        print(f"❌ Train manifest not found: {train_manifest}")
        if use_gpt_aligned:
            print("\nPrepare GPT-aligned dataset first (RECOMMENDED for stutters):")
            print(f"  python tools/prepare_gpt_aligned_dataset.py --speaker {args.speaker}")
        else:
            print("\nPrepare S2Mel dataset first:")
            print(f"  python tools/prepare_s2mel_dataset.py --speaker {args.speaker}")
        sys.exit(1)
    
    output_dir = args.output_dir or speaker_dir / "s2mel_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs" / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print("=" * 60)
    print("S2MEL LORA TRAINING - PROSODIC PATTERN LEARNING")
    print("=" * 60)
    
    training_mode = "GPT-ALIGNED (RECOMMENDED)" if use_gpt_aligned else "Standard W2V-BERT"
    print(f"""
Speaker: {args.speaker}
Training Mode: {training_mode}
LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}
Targets: DiT={args.include_dit}, WaveNet={args.include_wavenet}
Epochs: {args.epochs}
Batch size: {args.batch_size} × {args.grad_accumulation} = {args.batch_size * args.grad_accumulation} effective
Learning rate: {args.learning_rate}
Stutter weight: {args.stutter_weight}x

Train manifest: {train_manifest}
Output: {output_dir}
""")
    
    if use_gpt_aligned:
        print("📢 USING GPT-ALIGNED TRAINING!")
        print("   This uses the SAME conditioning as inference, so patterns transfer!")
        print()
    
    # Load reference features for consistent style
    reference_style = None
    reference_prompt = None
    reference_mel = None
    
    ref_features_path = args.reference_features or (
        speaker_dir / "dataset" / "processed_s2mel" / "reference_features.pt"
    )
    if args.use_reference_style and ref_features_path.exists():
        print("\n[0/5] Loading reference features for consistent style...")
        ref_features = torch.load(ref_features_path)
        reference_style = ref_features.get("style")
        reference_prompt = ref_features.get("prompt_condition")
        reference_mel = ref_features.get("mel")
        print(f"  ✓ Using consistent reference style: {reference_style.shape}")
    
    # Load base model
    print("\n[1/5] Loading base S2Mel model...")
    model = build_s2mel_model(args.config, args.model_dir, device)
    
    if args.verbose:
        print_s2mel_model_structure(model)
    
    # Apply LoRA
    print("\n[2/5] Applying LoRA adapters to S2Mel...")
    model = apply_lora_to_s2mel(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        include_dit=args.include_dit,
        include_wavenet=args.include_wavenet,
        include_length_regulator=args.include_length_regulator,
        include_gpt_layer=args.include_gpt_layer,
        verbose=True,
    )
    
    # CRITICAL: Re-sync all buffers to device after LoRA application
    # PEFT may leave some buffers on CPU
    model = model.to(device)
    
    # Ensure DiT buffers are on correct device
    def move_dit_buffers_to_device(model, device):
        """Move all DiT registered buffers to the correct device."""
        # Access the underlying model if wrapped by PEFT
        if hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model'):
                base = base.model
        else:
            base = model
            
        if hasattr(base, 'models') and 'cfm' in base.models:
            dit = base.models['cfm'].estimator
            # Move input_pos buffer
            if hasattr(dit, 'input_pos') and dit.input_pos is not None:
                dit.input_pos = dit.input_pos.to(device)
            # Move transformer buffers
            if hasattr(dit, 'transformer'):
                transformer = dit.transformer
                if hasattr(transformer, 'freqs_cis') and transformer.freqs_cis is not None:
                    transformer.freqs_cis = transformer.freqs_cis.to(device)
                if hasattr(transformer, 'causal_mask') and transformer.causal_mask is not None:
                    transformer.causal_mask = transformer.causal_mask.to(device)
    
    move_dit_buffers_to_device(model, device)
    print("  DiT buffers synced to device")
    
    param_stats = get_s2mel_trainable_parameters(model)
    print(f"\n  Trainable: {param_stats['trainable_params']:,} / {param_stats['all_params']:,} "
          f"({param_stats['trainable_percentage']:.2f}%)")
    print(f"  By component: {param_stats['by_component']}")
    
    # Load datasets
    print("\n[3/5] Loading datasets...")
    
    if use_gpt_aligned:
        # GPT-aligned dataset (RECOMMENDED for stutters)
        train_dataset = GPTAlignedDataset(
            train_manifest,
            reference_style=reference_style,
            reference_mel=reference_mel,
        )
        
        val_dataset = None
        if val_manifest.exists():
            val_dataset = GPTAlignedDataset(
                val_manifest,
                reference_style=reference_style,
                reference_mel=reference_mel,
            )
        
        collate_fn = collate_gpt_aligned_batch
    else:
        # Standard S2Mel dataset (W2V-BERT based)
        train_dataset = S2MelDataset(
            train_manifest,
            reference_style=reference_style,
            reference_prompt=reference_prompt,
            reference_mel=reference_mel,
        )
        
        val_dataset = None
        if val_manifest.exists():
            val_dataset = S2MelDataset(
                val_manifest,
                reference_style=reference_style,
                reference_prompt=reference_prompt,
                reference_mel=reference_mel,
            )
        
        collate_fn = collate_s2mel_batch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    
    # Calculate schedule
    batches_per_epoch = len(train_loader)
    effective_grad_accum = min(args.grad_accumulation, batches_per_epoch)
    steps_per_epoch = max(1, batches_per_epoch // effective_grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    
    print(f"\n  Training schedule:")
    print(f"    Batches per epoch: {batches_per_epoch}")
    print(f"    Steps per epoch: {steps_per_epoch}")
    print(f"    Total steps: {total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )
    
    # AMP
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING S2MEL TRAINING")
    print("=" * 60)
    print("\nThis trains the DIFFUSION MODEL to learn mel spectrogram patterns.")
    print("Stutters, pauses, and prosody are encoded in mel spectrograms!")
    
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_stutter_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        accumulated_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Use GPT-aligned loss if using GPT-aligned dataset
                if use_gpt_aligned:
                    loss, metrics = compute_gpt_aligned_cfm_loss(
                        model, batch, device,
                        stutter_weight=args.stutter_weight,
                    )
                else:
                    loss, metrics = compute_cfm_loss(
                        model, batch, device,
                        stutter_weight=args.stutter_weight,
                    )
            
            scaled_loss = loss / effective_grad_accum
            
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            bsz = batch["mel"].size(0)
            epoch_loss += loss.item() * bsz
            epoch_samples += bsz
            epoch_stutter_samples += metrics["stutter_samples"]
            accumulated_batches += 1
            
            # Optimizer step
            is_accumulation_step = accumulated_batches >= effective_grad_accum
            is_last_batch = (batch_idx + 1) == len(train_loader)
            
            if is_accumulation_step or is_last_batch:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, args.grad_clip
                    )
                else:
                    grad_norm = 0.0
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                
                global_step += 1
                
                # Log
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/cfm_loss", metrics["cfm_loss"], global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                
                pbar.set_postfix({
                    "loss": f"{metrics['cfm_loss']:.4f}",
                    "lr": f"{current_lr:.2e}",
                })
                
                # Save checkpoint
                if global_step % args.save_interval == 0:
                    ckpt_dir = output_dir / f"checkpoint_step{global_step}"
                    save_s2mel_lora_checkpoint(model, ckpt_dir, {
                        "step": global_step,
                        "epoch": epoch,
                    })
        
        # End of epoch
        avg_train_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"stutter_samples={epoch_stutter_samples}/{epoch_samples}")
        
        # Validation
        if val_loader:
            if use_gpt_aligned:
                val_metrics = evaluate_gpt_aligned(model, val_loader, device)
            else:
                val_metrics = evaluate(model, val_loader, device)
            writer.add_scalar("val/cfm_loss", val_metrics["val_loss"], global_step)
            print(f"  val_loss={val_metrics['val_loss']:.4f}")
            
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_dir = output_dir / "best_checkpoint"
                save_s2mel_lora_checkpoint(model, best_dir, {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "speaker": args.speaker,
                })
                print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    
    # Save final checkpoint
    final_dir = output_dir / "final_checkpoint"
    save_s2mel_lora_checkpoint(model, final_dir, {
        "epochs": args.epochs,
        "final_val_loss": val_metrics["val_loss"] if val_loader else avg_train_loss,
        "speaker": args.speaker,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    })
    
    # Copy reference features to checkpoint dirs
    if ref_features_path.exists():
        import shutil
        for ckpt_dir in [output_dir / "best_checkpoint", output_dir / "final_checkpoint"]:
            shutil.copy(ref_features_path, ckpt_dir / "reference_features.pt")
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("S2MEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
Best validation loss: {best_val_loss:.4f}

Output files:
  Final: {final_dir}
  Best: {output_dir / 'best_checkpoint'}
""")
    
    print(f"""
WHAT YOU HAVE TRAINED:
======================
The S2Mel diffusion model now knows how to generate mel spectrograms
with your speaker's prosodic patterns - including stutters!

The model learned from ACTUAL mel spectrograms of stuttered speech,
so it can reproduce those patterns in generated audio.

HOW TO USE:
===========
# Combined inference with S2Mel LoRA
python tools/infer_with_s2mel_lora.py \\
    --speaker {args.speaker} \\
    --text "I I I was going going to the store..."

# Or programmatically:
from tools.infer_with_s2mel_lora import S2MelLoRAInference

infer = S2MelLoRAInference(
    s2mel_lora_path="{final_dir}",
    reference_audio="path/to/reference.wav"
)
infer.generate("Your text here", "output.wav")

WHY THIS WORKS FOR STUTTERS:
============================
1. S2Mel generates mel spectrograms from semantic codes
2. Mel specs contain the ACTUAL acoustic patterns (timing, prosody)
3. By training on verbatim audio mels, the model learns stutter patterns
4. These patterns are now embedded in the diffusion generation process!

Unlike GPT training (which only affects semantic codes), S2Mel training
directly learns the acoustic realization - the actual sound of speech.
""")


if __name__ == "__main__":
    main()