#!/usr/bin/env python3
"""
Quick validation script to verify a LoRA checkpoint has the correct weights.

This script checks:
1. That the checkpoint contains GPT transformer weights (critical for voice)
2. The weights are non-zero and varied (actually trained, not initialized)
3. The checkpoint can be loaded correctly

Usage:
    python tools/verify_lora_checkpoint.py trained_lora/my_voice/final_checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify LoRA checkpoint")
    parser.add_argument("checkpoint_path", type=Path, help="Path to LoRA checkpoint directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint_path
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Check for required files
    config_path = checkpoint_path / "adapter_config.json"
    weights_bin = checkpoint_path / "adapter_model.bin"
    weights_safetensors = checkpoint_path / "adapter_model.safetensors"
    
    if not config_path.exists():
        print(f"❌ Missing adapter_config.json in {checkpoint_path}")
        sys.exit(1)
    
    if not weights_bin.exists() and not weights_safetensors.exists():
        print(f"❌ Missing adapter weights in {checkpoint_path}")
        sys.exit(1)
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"   LoRA rank: {config.get('r', 'N/A')}")
    print(f"   LoRA alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"   Target modules: {len(config.get('target_modules', []))}")
    
    # Load weights
    if weights_safetensors.exists():
        from safetensors.torch import load_file
        state = load_file(str(weights_safetensors))
        weight_file = "safetensors"
    else:
        import torch
        state = torch.load(weights_bin, map_location='cpu')
        weight_file = "bin"
    
    print(f"   Weight format: {weight_file}")
    print(f"   Total weight tensors: {len(state)}")
    
    # Categorize weights
    gpt_weights = [k for k in state.keys() if 'gpt.h.' in k]
    conditioning_weights = [k for k in state.keys() if 'conditioning_encoder' in k or 'perceiver_encoder' in k]
    head_weights = [k for k in state.keys() if 'head' in k]
    embedding_weights = [k for k in state.keys() if 'embedding' in k]
    other_weights = [k for k in state.keys() if k not in gpt_weights + conditioning_weights + head_weights + embedding_weights]
    
    print(f"\n📊 Weight Breakdown:")
    print(f"   GPT transformer: {len(gpt_weights)} tensors")
    print(f"   Conditioning encoders: {len(conditioning_weights)} tensors")  
    print(f"   Output heads: {len(head_weights)} tensors")
    print(f"   Embeddings: {len(embedding_weights)} tensors")
    print(f"   Other: {len(other_weights)} tensors")
    
    # Critical check: GPT weights
    if len(gpt_weights) == 0:
        print(f"\n❌ CRITICAL: No GPT transformer weights found!")
        print(f"   This LoRA checkpoint cannot effectively change the voice.")
        print(f"   The model was likely trained with incorrect target modules.")
        print(f"\n   To fix: Retrain with the updated lora_utils.py that includes")
        print(f"   target modules like 'c_attn', 'c_proj', 'c_fc'")
        sys.exit(1)
    
    print(f"\n✅ GPT transformer weights present!")
    
    # Check weight statistics
    import torch
    
    all_values = []
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            all_values.append(tensor.flatten().float().numpy())
    
    if all_values:
        concat = np.concatenate(all_values)
        mean = np.mean(concat)
        std = np.std(concat)
        min_val = np.min(concat)
        max_val = np.max(concat)
        non_zero = np.count_nonzero(concat) / len(concat) * 100
        
        print(f"\n📈 Weight Statistics:")
        print(f"   Mean: {mean:.6f}")
        print(f"   Std:  {std:.6f}")
        print(f"   Min:  {min_val:.6f}")
        print(f"   Max:  {max_val:.6f}")
        print(f"   Non-zero: {non_zero:.1f}%")
        
        # Check if weights look trained vs initialized
        if std < 0.001:
            print(f"\n⚠️  WARNING: Very low weight variance - model may not be trained")
        elif non_zero < 50:
            print(f"\n⚠️  WARNING: Many zero weights - training may have issues")
        else:
            print(f"\n✅ Weights appear trained (good variance, mostly non-zero)")
    
    # Verbose output
    if args.verbose:
        print(f"\n📋 All Weight Keys:")
        for k in sorted(state.keys()):
            tensor = state[k]
            if isinstance(tensor, torch.Tensor):
                print(f"   {k}: {tuple(tensor.shape)}")
    
    # Summary
    print(f"\n{'='*60}")
    if len(gpt_weights) > 0:
        print(f"✅ CHECKPOINT VALID - Contains {len(gpt_weights)} GPT weights")
        print(f"   This checkpoint should affect voice generation.")
    else:
        print(f"❌ CHECKPOINT INVALID - Missing GPT weights")
        print(f"   Retrain with updated target modules.")
    
    # Count unique GPT layers
    gpt_layers = set()
    for w in gpt_weights:
        # Extract layer number from paths like "base_model.model.gpt.h.0.attn.c_attn.lora_A"
        parts = w.split('.')
        try:
            h_idx = parts.index('h')
            layer_num = parts[h_idx + 1]
            gpt_layers.add(int(layer_num))
        except (ValueError, IndexError):
            pass
    
    if gpt_layers:
        print(f"   GPT layers adapted: {len(gpt_layers)} (layers {min(gpt_layers)}-{max(gpt_layers)})")


if __name__ == "__main__":
    main()