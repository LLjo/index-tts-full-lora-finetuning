"""
LoRA (Low-Rank Adaptation) utilities for IndexTTS fine-tuning.

This module provides functions for:
- Applying LoRA to UnifiedVoice models
- Saving/loading LoRA checkpoints
- Merging LoRA weights for deployment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, List
import warnings

import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def get_lora_target_modules(
    include_embeddings: bool = False,
    include_heads: bool = True,
    include_conditioning: bool = True,
    include_gpt: bool = True,
    custom_modules: Optional[List[str]] = None,
) -> List[str]:
    """
    Get the default target modules for LoRA fine-tuning in UnifiedVoice.
    
    IMPORTANT: PEFT matches modules by their FINAL NAME, not full path.
    For example, "c_attn" matches all modules named "c_attn" regardless of path.
    
    Args:
        include_embeddings: Whether to include text/mel embedding layers
        include_heads: Whether to include output prediction heads
        include_conditioning: Whether to include conditioning/voice encoders (IMPORTANT for voice adaptation!)
        include_gpt: Whether to include GPT transformer layers (CRITICAL for generation!)
        custom_modules: Custom list of module names to target (overrides defaults)
    
    Returns:
        List of module names to apply LoRA to
    """
    if custom_modules is not None:
        return custom_modules
    
    target_modules = []
    
    # GPT2 transformer - THE CORE of voice generation!
    # These Conv1D layers handle the autoregressive text-to-semantic-token generation
    # This is WHERE THE VOICE IS LEARNED
    if include_gpt:
        target_modules.extend([
            "c_attn",    # Q, K, V projections in GPT self-attention (Conv1D)
            "c_proj",    # Output projection in GPT attention AND MLP (Conv1D)
            "c_fc",      # First MLP layer in GPT (Conv1D)
        ])
    
    # Conditioning encoders - important for voice style adaptation
    # These process the reference audio to create speaker embeddings
    if include_conditioning:
        target_modules.extend([
            # Conformer encoder layers (in conditioning_encoder and emo_conditioning_encoder)
            "linear_q",       # Self-attention query
            "linear_k",       # Self-attention key
            "linear_v",       # Self-attention value
            "linear_out",     # Self-attention output
            "linear_pos",     # Positional encoding
            "w_1",            # Feed-forward first layer
            "w_2",            # Feed-forward second layer
            "pointwise_conv1", # Conformer convolution
            "pointwise_conv2", # Conformer convolution
            
            # Perceiver resampler layers (compresses conditioning)
            "to_q",           # Cross-attention query
            "to_kv",          # Cross-attention key/value
            "to_out",         # Cross-attention output
            "proj_context",   # Context projection
            
            # Emotion vector transformation
            "emovec_layer",   # Emotion vector projection
            "emo_layer",      # Emotion layer
        ])
    
    # Optional: Include embedding layers for stronger speaker adaptation
    if include_embeddings:
        target_modules.extend([
            "text_embedding",
            "mel_embedding",
        ])
    
    # Optional: Include output heads for prediction refinement
    if include_heads:
        target_modules.extend([
            "text_head",
            "mel_head",
        ])
    
    return target_modules


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    include_embeddings: bool = False,
    include_heads: bool = True,
    include_conditioning: bool = True,
    include_gpt: bool = True,
    bias: str = "none",
) -> PeftModel:
    """
    Apply LoRA adapters to a UnifiedVoice model.
    
    Args:
        model: The base UnifiedVoice model to adapt
        lora_rank: Rank of the LoRA decomposition (higher = more capacity, slower)
        lora_alpha: Scaling factor for LoRA weights (usually 2*rank)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Custom list of modules to target (overrides defaults)
        include_embeddings: Whether to include embedding layers
        include_heads: Whether to include output heads
        include_conditioning: Whether to include conditioning/voice encoders (recommended for voice cloning!)
        include_gpt: Whether to include GPT transformer layers (CRITICAL for voice generation!)
        bias: Bias training strategy: "none", "all", or "lora_only"
    
    Returns:
        The model wrapped with PEFT LoRA adapters
        
    Raises:
        ValueError: If no LoRA adapters were successfully applied
    """
    if target_modules is None:
        target_modules = get_lora_target_modules(
            include_embeddings=include_embeddings,
            include_heads=include_heads,
            include_conditioning=include_conditioning,
            include_gpt=include_gpt,
        )
    
    # Show what modules we're targeting
    print(f"[LoRA] Target modules: {target_modules}")
    
    # Find which target modules actually exist in the model
    model_modules = {name.split('.')[-1] for name, _ in model.named_modules()}
    matched = [m for m in target_modules if m in model_modules]
    unmatched = [m for m in target_modules if m not in model_modules]
    
    if unmatched:
        print(f"[LoRA] Warning: These target modules were not found in model: {unmatched}")
    
    if not matched:
        raise ValueError(
            f"No target modules matched! Targets: {target_modules}\n"
            f"Available module names (last part): {sorted(model_modules)[:50]}..."
        )
    
    print(f"[LoRA] Matched target modules: {matched}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    # Validate that LoRA was actually applied
    lora_params = [n for n, _ in peft_model.named_parameters() if 'lora_' in n.lower()]
    if not lora_params:
        raise ValueError(
            "No LoRA parameters were created! PEFT may not have matched any modules. "
            f"Target modules were: {target_modules}"
        )
    
    # Count LoRA layers
    lora_a_count = sum(1 for n in lora_params if 'lora_a' in n.lower() or 'lora_A' in n)
    lora_b_count = sum(1 for n in lora_params if 'lora_b' in n.lower() or 'lora_B' in n)
    
    print(f"[LoRA] Successfully created {lora_a_count} LoRA adapter pairs")
    
    # Check for GPT layers specifically (critical for voice)
    gpt_lora = [n for n in lora_params if 'gpt.h.' in n]
    if include_gpt and not gpt_lora:
        warnings.warn(
            "WARNING: No GPT transformer layers were adapted by LoRA! "
            "Voice generation may not be affected. "
            "Check that target_modules includes 'c_attn', 'c_proj', 'c_fc'."
        )
    elif gpt_lora:
        gpt_layers = set(n.split('.')[3] for n in gpt_lora if 'gpt.h.' in n)
        print(f"[LoRA] GPT layers with LoRA adapters: {len(gpt_layers)} transformer layers")
    
    # Print trainable parameters info
    peft_model.print_trainable_parameters()
    
    return peft_model


def save_lora_checkpoint(
    model: PeftModel,
    save_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save LoRA checkpoint (adapter weights only, not the full model).
    
    Args:
        model: PEFT model with LoRA adapters
        save_path: Directory to save the checkpoint to
        metadata: Optional metadata to include (training config, stats, etc.)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save PEFT adapter weights and config
    model.save_pretrained(save_path)
    
    # Save additional metadata if provided
    if metadata is not None:
        metadata_path = save_path / "training_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"LoRA checkpoint saved to: {save_path}")
    print(f"  - adapter_config.json: LoRA configuration")
    print(f"  - adapter_model.bin: LoRA weights (~1-10 MB)")
    if metadata is not None:
        print(f"  - training_metadata.json: Training metadata")


def load_lora_checkpoint(
    base_model: torch.nn.Module,
    lora_path: Path,
    merge_weights: bool = False,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Load LoRA checkpoint and apply to base model.
    
    Args:
        base_model: The base UnifiedVoice model to load adapters into
        lora_path: Path to the LoRA checkpoint directory
        merge_weights: If True, merge LoRA into base model (for deployment)
        device: Device to load the model to (if None, uses model's current device)
    
    Returns:
        Model with LoRA adapters loaded (or merged)
    """
    lora_path = Path(lora_path)
    
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    
    # Check for required files
    config_path = lora_path / "adapter_config.json"
    model_path = lora_path / "adapter_model.bin"
    
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")
    if not model_path.exists():
        # Try safetensors format
        model_path = lora_path / "adapter_model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"adapter_model.bin/safetensors not found in {lora_path}")
    
    # Load LoRA adapters
    print(f"Loading LoRA checkpoint from: {lora_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(lora_path),
        device_map=device,
    )
    
    if merge_weights:
        print("Merging LoRA weights into base model...")
        peft_model = peft_model.merge_and_unload()
        print("LoRA weights merged successfully")
    else:
        print("LoRA adapters loaded successfully (not merged)")
    
    # Load metadata if available
    metadata_path = lora_path / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Training metadata loaded: {metadata.get('epochs', 'N/A')} epochs, "
              f"{metadata.get('final_loss', 'N/A')} final loss")
    
    return peft_model


def merge_lora_weights(
    peft_model: PeftModel,
    output_path: Optional[Path] = None,
) -> torch.nn.Module:
    """
    Merge LoRA weights into the base model for deployment.
    
    This is useful when you want to deploy a single model file without
    needing to load adapters at runtime.
    
    Args:
        peft_model: PEFT model with LoRA adapters
        output_path: Optional path to save the merged model
    
    Returns:
        Merged model (base model with LoRA weights integrated)
    """
    print("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the merged model
        torch.save(
            {"model": merged_model.state_dict()},
            output_path
        )
        print(f"Merged model saved to: {output_path}")
    
    return merged_model


def list_lora_checkpoints(lora_dir: Path) -> List[Dict[str, str]]:
    """
    List all LoRA checkpoints in a directory.
    
    Args:
        lora_dir: Directory containing LoRA checkpoints
    
    Returns:
        List of checkpoint info dictionaries with paths and metadata
    """
    lora_dir = Path(lora_dir)
    
    if not lora_dir.exists():
        warnings.warn(f"LoRA directory does not exist: {lora_dir}")
        return []
    
    checkpoints = []
    
    # Find all subdirectories containing adapter_config.json
    for subdir in lora_dir.iterdir():
        if subdir.is_dir():
            config_path = subdir / "adapter_config.json"
            if config_path.exists():
                checkpoint_info = {
                    "path": str(subdir),
                    "name": subdir.name,
                }
                
                # Load config
                with open(config_path, "r") as f:
                    config = json.load(f)
                checkpoint_info["config"] = config
                
                # Load metadata if available
                metadata_path = subdir / "training_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    checkpoint_info["metadata"] = metadata
                
                checkpoints.append(checkpoint_info)
    
    return checkpoints


def get_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get statistics about trainable parameters in a model.
    
    Args:
        model: PyTorch model (with or without LoRA)
    
    Returns:
        Dictionary with parameter counts and percentages
    """
    trainable_params = 0
    all_params = 0
    
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": 100 * trainable_params / all_params if all_params > 0 else 0,
    }