"""
S2Mel LoRA Utilities for IndexTTS2

This module provides utilities for applying LoRA (Low-Rank Adaptation) to the S2Mel 
(Semantic-to-Mel) model, which is the KEY component for learning prosodic patterns 
like stutters, pauses, and speech rhythm.

The Architecture:
=================
S2Mel consists of several components:
1. CFM (Conditional Flow Matching) - Contains the DiT diffusion model
2. DiT (Diffusion Transformer) - The main transformer that generates mel spectrograms
3. Length Regulator - Controls timing/duration
4. GPT Layer - Projects GPT latent to S2Mel space

For prosodic pattern learning, we primarily target:
- DiT Transformer layers (where mel patterns are learned)
- Optionally: Length regulator (for duration learning)

Why S2Mel Training Works for Stutters:
=====================================
1. GPT stage: text -> semantic codes (WHAT to say)
2. S2Mel stage: semantic codes -> mel spectrogram (HOW to say it)

Stutters/pauses are in the "HOW" - the prosodic realization.
Training S2Mel with verbatim mel spectrograms teaches it the actual acoustic patterns!
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. S2Mel LoRA training unavailable.")
    print("Install with: pip install peft")


def get_s2mel_lora_target_modules(
    include_dit: bool = True,
    include_wavenet: bool = False,  # Changed default to False - WaveNet uses custom Conv wrappers incompatible with LoRA
    include_length_regulator: bool = False,
    include_gpt_layer: bool = False,
) -> List[str]:
    """
    Get the list of module names to apply LoRA to in S2Mel.
    
    The DiT (Diffusion Transformer) is the most important component for
    learning prosodic patterns.
    
    IMPORTANT: WaveNet layers are NOT supported because they use custom SConv1d
    wrappers that access kernel_size directly, which breaks with PEFT wrapping.
    Only Linear layers in the DiT transformer are safe to target.
    
    Args:
        include_dit: Include DiT transformer attention layers (RECOMMENDED)
        include_wavenet: DEPRECATED - WaveNet layers are NOT compatible with PEFT
        include_length_regulator: Include length regulator (for duration learning)
        include_gpt_layer: Include GPT projection layer
    
    Returns:
        List of module names for LoRA targeting
    """
    target_modules = []
    
    if include_dit:
        # DiT Transformer - main component for mel pattern generation
        # The transformer uses gpt_fast style attention
        # ONLY target Linear layers, NOT Conv layers
        target_modules.extend([
            # Main attention layers in DiT Transformer (Linear layers)
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.attention\.wqkv",
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.attention\.wo",
            # Feed-forward layers (Linear layers)
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.feed_forward\.w1",
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.feed_forward\.w2",
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.feed_forward\.w3",
            # AdaptiveLayerNorm projection layers
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.attention_norm\.project_layer",
            r"models\.cfm\.estimator\.transformer\.layers\.\d+\.ffn_norm\.project_layer",
            r"models\.cfm\.estimator\.transformer\.norm\.project_layer",
        ])
        
        # Input/output projections (Linear layers only - skip Conv layers!)
        target_modules.extend([
            r"models\.cfm\.estimator\.cond_projection",
            r"models\.cfm\.estimator\.cond_x_merge_linear",
            r"models\.cfm\.estimator\.skip_linear",
        ])
    
    if include_wavenet:
        # WaveNet layers - WARNING: NOT COMPATIBLE WITH PEFT!
        # These use SConv1d which accesses .conv.conv.kernel_size directly
        # PEFT wrapping breaks this attribute access
        print("  WARNING: include_wavenet=True is not supported! WaveNet uses custom Conv wrappers.")
        print("           Skipping WaveNet layers to avoid AttributeError.")
        # Do NOT add any WaveNet modules
    
    if include_length_regulator:
        # Length regulator - controls duration/timing
        # Only include Linear layers, not Conv layers
        target_modules.extend([
            # None - length regulator primarily uses Conv layers
        ])
    
    if include_gpt_layer:
        # GPT latent projection (these are standard Linear layers)
        target_modules.extend([
            r"models\.gpt_layer\.0",
            r"models\.gpt_layer\.1",
            r"models\.gpt_layer\.2",
        ])
    
    return target_modules


def find_all_linear_modules(model: nn.Module, prefix: str = "") -> Dict[str, nn.Module]:
    """
    Find all Linear and Conv1d modules in the model that could be LoRA targets.
    
    Args:
        model: The model to search
        prefix: Current module path prefix
        
    Returns:
        Dict mapping module path to module
    """
    linear_modules = {}
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            linear_modules[full_name] = module
        else:
            # Recurse into children
            linear_modules.update(find_all_linear_modules(module, full_name))
    
    return linear_modules


def get_trainable_modules_info(model: nn.Module) -> Dict[str, Dict]:
    """
    Get information about all potentially trainable modules in S2Mel.
    
    Useful for understanding what can be trained and deciding LoRA targets.
    """
    info = {}
    linear_modules = find_all_linear_modules(model)
    
    for name, module in linear_modules.items():
        if isinstance(module, nn.Linear):
            info[name] = {
                "type": "Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "params": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
            }
        elif isinstance(module, nn.Conv1d):
            info[name] = {
                "type": "Conv1d",
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size[0],
                "params": module.weight.numel() + (module.bias.numel() if module.bias is not None else 0),
            }
    
    return info


def apply_lora_to_s2mel(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    include_dit: bool = True,
    include_wavenet: bool = True,
    include_length_regulator: bool = False,
    include_gpt_layer: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA adapters to the S2Mel model for efficient fine-tuning.
    
    This is the KEY function for enabling prosodic pattern learning:
    - DiT learns the mel pattern generation
    - WaveNet refines acoustic details
    
    Args:
        model: S2Mel model (MyModel instance)
        lora_rank: LoRA rank (higher = more capacity, more params)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        include_dit: Apply LoRA to DiT transformer (HIGHLY RECOMMENDED)
        include_wavenet: Apply LoRA to WaveNet layers
        include_length_regulator: Apply LoRA to length regulator
        include_gpt_layer: Apply LoRA to GPT projection layer
        verbose: Print detailed information
    
    Returns:
        Model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT library required for LoRA. Install with: pip install peft")
    
    if verbose:
        print(f"\n[S2Mel LoRA] Applying LoRA adapters...")
        print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
        print(f"  Targets: DiT={include_dit}, WaveNet={include_wavenet}, "
              f"LengthReg={include_length_regulator}, GPTLayer={include_gpt_layer}")
    
    # Get target modules based on configuration
    target_patterns = get_s2mel_lora_target_modules(
        include_dit=include_dit,
        include_wavenet=include_wavenet,
        include_length_regulator=include_length_regulator,
        include_gpt_layer=include_gpt_layer,
    )
    
    # Find actual module names in the model
    all_modules = find_all_linear_modules(model)
    target_modules = []
    
    for pattern in target_patterns:
        regex = re.compile(pattern)
        for module_name in all_modules.keys():
            if regex.match(module_name):
                # Convert to PEFT-compatible format (dots to underscores isn't needed actually)
                target_modules.append(module_name)
    
    # Remove duplicates while preserving order
    target_modules = list(dict.fromkeys(target_modules))
    
    if verbose:
        print(f"\n  Found {len(target_modules)} target modules:")
        for m in target_modules[:10]:
            print(f"    - {m}")
        if len(target_modules) > 10:
            print(f"    ... and {len(target_modules) - 10} more")
    
    if not target_modules:
        print("  Warning: No target modules found! Trying fallback approach...")
        # Fallback: find all Linear modules
        target_modules = [name for name, m in all_modules.items() 
                         if isinstance(m, nn.Linear) and 'cfm' in name]
        if verbose:
            print(f"  Fallback found {len(target_modules)} modules")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        # Note: S2Mel is not a CausalLM, but we use SEQ_2_SEQ_LM as closest
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"\n  LoRA applied successfully!")
        print(f"  Trainable params: {trainable_params:,} / {all_params:,} "
              f"({100 * trainable_params / all_params:.2f}%)")
    
    return model


def save_s2mel_lora_checkpoint(
    model: nn.Module,
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save S2Mel LoRA checkpoint.
    
    Args:
        model: S2Mel model with LoRA adapters
        output_dir: Directory to save checkpoint
        metadata: Optional metadata to save with checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapter weights
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
        print(f"  Saved S2Mel LoRA adapter to: {output_dir}")
    else:
        # Manual save for non-PEFT wrapped models
        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state[name] = param.detach().cpu()
        
        torch.save(lora_state, output_dir / "s2mel_lora_weights.pt")
        print(f"  Saved S2Mel LoRA weights to: {output_dir / 's2mel_lora_weights.pt'}")
    
    # Save metadata
    if metadata:
        import json
        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_s2mel_lora_checkpoint(
    model: nn.Module,
    lora_path: Path,
    merge_weights: bool = True,
    device: str = "cuda",
    verbose: bool = True,
) -> nn.Module:
    """
    Load S2Mel LoRA checkpoint into model.
    
    Args:
        model: Base S2Mel model
        lora_path: Path to LoRA checkpoint directory
        merge_weights: If True, merge LoRA weights into base model for faster inference
        device: Device to load to
        verbose: Print information
    
    Returns:
        Model with LoRA weights loaded
    """
    lora_path = Path(lora_path)
    
    if verbose:
        print(f"\n[S2Mel LoRA] Loading from: {lora_path}")
    
    # Check for PEFT adapter
    adapter_config = lora_path / "adapter_config.json"
    adapter_model = lora_path / "adapter_model.bin"
    adapter_model_safetensors = lora_path / "adapter_model.safetensors"
    manual_weights = lora_path / "s2mel_lora_weights.pt"
    
    if adapter_config.exists() and (adapter_model.exists() or adapter_model_safetensors.exists()):
        # PEFT format
        if verbose:
            print("  Loading PEFT adapter format...")
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=not merge_weights,
        )
        
        if merge_weights:
            if verbose:
                print("  Merging LoRA weights into base model...")
            model = model.merge_and_unload()
    
    elif manual_weights.exists():
        # Manual format
        if verbose:
            print("  Loading manual LoRA weights format...")
        
        lora_state = torch.load(manual_weights, map_location=device)
        
        # Load into model
        current_state = model.state_dict()
        for name, param in lora_state.items():
            if name in current_state:
                current_state[name].copy_(param)
            else:
                print(f"  Warning: {name} not found in model state")
        
        if merge_weights:
            # For manual format, weights are already merged style (additive)
            pass
    else:
        raise FileNotFoundError(f"No LoRA checkpoint found at {lora_path}")
    
    if verbose:
        print("  S2Mel LoRA loaded successfully!")
    
    return model


def get_s2mel_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Get statistics about trainable parameters in S2Mel model.
    
    Returns:
        Dict with parameter statistics
    """
    trainable_params = 0
    all_params = 0
    trainable_by_component = {
        "cfm": 0,
        "length_regulator": 0,
        "gpt_layer": 0,
        "other": 0,
    }
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
            # Categorize
            if "cfm" in name:
                trainable_by_component["cfm"] += param.numel()
            elif "length_regulator" in name:
                trainable_by_component["length_regulator"] += param.numel()
            elif "gpt_layer" in name:
                trainable_by_component["gpt_layer"] += param.numel()
            else:
                trainable_by_component["other"] += param.numel()
    
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": 100 * trainable_params / max(all_params, 1),
        "by_component": trainable_by_component,
    }


def print_s2mel_model_structure(model: nn.Module, max_depth: int = 3) -> None:
    """
    Print S2Mel model structure for debugging.
    
    Args:
        model: S2Mel model
        max_depth: Maximum depth to print
    """
    def _print_module(module, prefix="", depth=0):
        if depth >= max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Get parameter count
            params = sum(p.numel() for p in child.parameters(recurse=False))
            trainable = sum(p.numel() for p in child.parameters(recurse=False) if p.requires_grad)
            
            indent = "  " * depth
            if params > 0:
                print(f"{indent}{name}: {type(child).__name__} "
                      f"(params={params:,}, trainable={trainable:,})")
            else:
                print(f"{indent}{name}: {type(child).__name__}")
            
            _print_module(child, full_name, depth + 1)
    
    print("\n=== S2Mel Model Structure ===")
    _print_module(model)
    print("=" * 40)