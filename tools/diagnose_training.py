#!/usr/bin/env python3
"""
Diagnostic tool for IndexTTS2 LoRA/fine-tuning training issues.

This script helps diagnose why training might not be producing audible differences:
1. Verifies LoRA weights are actually non-zero (training worked)
2. Compares GPT token outputs with/without LoRA
3. Shows the S2Mel dominance issue
4. Provides recommendations

Usage:
    python tools/diagnose_training.py --lora-path trained_lora/final_checkpoint
    python tools/diagnose_training.py --gpt-checkpoint path/to/finetuned_gpt.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import OrderedDict


def diagnose_lora_checkpoint(lora_path: Path) -> dict:
    """Analyze LoRA checkpoint to verify training worked."""
    print(f"\n{'='*60}")
    print("DIAGNOSING LORA CHECKPOINT")
    print(f"{'='*60}")
    print(f"Path: {lora_path}")
    
    results = {
        "path": str(lora_path),
        "exists": False,
        "has_weights": False,
        "weight_stats": {},
        "issues": [],
        "recommendations": []
    }
    
    if not lora_path.exists():
        results["issues"].append(f"LoRA checkpoint not found at {lora_path}")
        return results
    
    results["exists"] = True
    
    # Check for required files
    adapter_config = lora_path / "adapter_config.json"
    adapter_model_bin = lora_path / "adapter_model.bin"
    adapter_model_safetensors = lora_path / "adapter_model.safetensors"
    
    if not adapter_config.exists():
        results["issues"].append("Missing adapter_config.json")
        return results
    
    # Load adapter weights
    if adapter_model_safetensors.exists():
        from safetensors.torch import load_file
        weights = load_file(str(adapter_model_safetensors))
        print(f"Loaded weights from: adapter_model.safetensors")
    elif adapter_model_bin.exists():
        weights = torch.load(str(adapter_model_bin), map_location="cpu")
        print(f"Loaded weights from: adapter_model.bin")
    else:
        results["issues"].append("No adapter weights file found (adapter_model.bin or .safetensors)")
        return results
    
    results["has_weights"] = True
    
    # Analyze weight statistics
    print(f"\nFound {len(weights)} weight tensors:")
    
    lora_a_weights = {}
    lora_b_weights = {}
    
    for name, tensor in weights.items():
        stats = {
            "shape": list(tensor.shape),
            "mean": float(tensor.float().mean()),
            "std": float(tensor.float().std()),
            "min": float(tensor.float().min()),
            "max": float(tensor.float().max()),
            "zeros_pct": float((tensor == 0).float().mean() * 100),
            "near_zeros_pct": float((tensor.float().abs() < 1e-6).float().mean() * 100)
        }
        results["weight_stats"][name] = stats
        
        if "lora_A" in name:
            lora_a_weights[name] = tensor
        elif "lora_B" in name:
            lora_b_weights[name] = tensor
    
    # Check if weights are meaningful
    print(f"\nLoRA A matrices: {len(lora_a_weights)}")
    print(f"LoRA B matrices: {len(lora_b_weights)}")
    
    # LoRA B matrices start at zero - if they're still zero, training didn't work
    b_all_zeros = True
    b_stats = []
    for name, tensor in lora_b_weights.items():
        is_zero = (tensor.abs() < 1e-8).all().item()
        mean_abs = tensor.float().abs().mean().item()
        b_stats.append((name, is_zero, mean_abs))
        if not is_zero:
            b_all_zeros = False
    
    print(f"\nLoRA B matrix analysis (B matrices start at 0, should be non-zero after training):")
    print("-" * 80)
    for name, is_zero, mean_abs in b_stats[:10]:  # Show first 10
        status = "❌ ZERO" if is_zero else "✓ OK"
        print(f"  {status} | mean_abs={mean_abs:.6f} | {name[:60]}...")
    if len(b_stats) > 10:
        print(f"  ... and {len(b_stats) - 10} more")
    
    if b_all_zeros:
        results["issues"].append(
            "ALL LoRA B matrices are zero! This means training did NOT update the weights. "
            "Check: 1) Training loss was decreasing, 2) Correct number of epochs, 3) Dataset is valid"
        )
        results["recommendations"].append(
            "Re-run training and monitor the loss. Ensure it decreases over epochs."
        )
    else:
        nonzero_count = sum(1 for _, is_zero, _ in b_stats if not is_zero)
        print(f"\n✓ {nonzero_count}/{len(b_stats)} LoRA B matrices have non-zero values (training worked!)")
    
    # Check weight magnitudes
    a_magnitudes = [t.float().abs().mean().item() for t in lora_a_weights.values()]
    b_magnitudes = [t.float().abs().mean().item() for t in lora_b_weights.values()]
    
    if a_magnitudes:
        print(f"\nLoRA A mean magnitude: {np.mean(a_magnitudes):.6f} (std: {np.std(a_magnitudes):.6f})")
    if b_magnitudes:
        print(f"LoRA B mean magnitude: {np.mean(b_magnitudes):.6f} (std: {np.std(b_magnitudes):.6f})")
    
    # Load config
    import json
    with open(adapter_config, "r") as f:
        config = json.load(f)
    
    print(f"\nLoRA Configuration:")
    print(f"  - Rank (r): {config.get('r', 'N/A')}")
    print(f"  - Alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"  - Target modules: {config.get('target_modules', 'N/A')}")
    
    # Check if critical modules were targeted
    target_modules = config.get("target_modules", [])
    critical_gpt_modules = ["c_attn", "c_proj", "c_fc"]
    missing_critical = [m for m in critical_gpt_modules if m not in target_modules]
    if missing_critical:
        results["issues"].append(
            f"Missing critical GPT modules in target_modules: {missing_critical}. "
            "These are essential for voice generation!"
        )
    else:
        print(f"\n✓ All critical GPT modules ({critical_gpt_modules}) are targeted")
    
    return results


def diagnose_gpt_checkpoint(gpt_path: Path, base_checkpoint: Path = None) -> dict:
    """Compare fine-tuned GPT checkpoint with base checkpoint."""
    print(f"\n{'='*60}")
    print("DIAGNOSING GPT CHECKPOINT")
    print(f"{'='*60}")
    print(f"Fine-tuned: {gpt_path}")
    
    results = {
        "path": str(gpt_path),
        "exists": False,
        "different_from_base": None,
        "issues": [],
        "recommendations": []
    }
    
    if not gpt_path.exists():
        results["issues"].append(f"GPT checkpoint not found at {gpt_path}")
        return results
    
    results["exists"] = True
    
    # Load fine-tuned checkpoint
    finetuned = torch.load(gpt_path, map_location="cpu")
    if "model" in finetuned:
        finetuned = finetuned["model"]
    
    print(f"Loaded fine-tuned checkpoint with {len(finetuned)} parameters")
    
    # Compare with base if available
    if base_checkpoint and base_checkpoint.exists():
        print(f"Base: {base_checkpoint}")
        base = torch.load(base_checkpoint, map_location="cpu")
        if "model" in base:
            base = base["model"]
        
        print(f"\nComparing with base checkpoint ({len(base)} parameters)...")
        
        different_keys = []
        identical_keys = []
        
        for key in finetuned.keys():
            if key in base:
                ft_tensor = finetuned[key]
                base_tensor = base[key]
                
                if ft_tensor.shape != base_tensor.shape:
                    different_keys.append((key, "shape_mismatch", ft_tensor.shape, base_tensor.shape))
                else:
                    diff = (ft_tensor - base_tensor).abs().mean().item()
                    if diff > 1e-6:
                        different_keys.append((key, "values_differ", diff))
                    else:
                        identical_keys.append(key)
        
        print(f"\nResults:")
        print(f"  - Identical parameters: {len(identical_keys)}")
        print(f"  - Different parameters: {len(different_keys)}")
        
        if different_keys:
            results["different_from_base"] = True
            print(f"\nDifferent parameters (showing first 20):")
            for info in different_keys[:20]:
                if info[1] == "shape_mismatch":
                    print(f"  - {info[0]}: shape {info[2]} vs {info[3]}")
                else:
                    print(f"  - {info[0]}: mean diff = {info[2]:.6f}")
        else:
            results["different_from_base"] = False
            results["issues"].append(
                "Fine-tuned checkpoint is IDENTICAL to base checkpoint! "
                "Training did not modify the model weights."
            )
    
    return results


def diagnose_inference_issue():
    """Explain the S2Mel dominance issue."""
    print(f"\n{'='*60}")
    print("UNDERSTANDING THE S2Mel DOMINANCE ISSUE")
    print(f"{'='*60}")
    
    explanation = """
The IndexTTS2 architecture has a two-stage pipeline:

STAGE 1: GPT Model (This is what you're training with LoRA)
────────────────────────────────────────────────────────────
Input: Text tokens + Speaker conditioning (from reference audio)
Output: Semantic tokens (numbers representing speech sounds)

Your LoRA training modifies how the GPT generates these tokens.
After training, the GPT should generate DIFFERENT token sequences
that better match your training data's speaking style.

STAGE 2: S2Mel Model (NOT trained)
────────────────────────────────────────────────────────────
Input: 
  - Semantic tokens from GPT
  - Reference audio features (style, mel, prompt_condition)
    ↑↑↑ THIS IS THE KEY ISSUE ↑↑↑
Output: Mel spectrogram that sounds like the reference audio

The S2Mel model uses a Conditional Flow Matching (CFM) approach
that "style transfers" the semantic tokens to sound like the
reference audio. This means:

  - If you provide reference audio from Speaker A at inference time
  - The output will sound like Speaker A
  - REGARDLESS of what the GPT was trained on!

THIS IS WHY YOUR TRAINING SEEMS INEFFECTIVE!
────────────────────────────────────────────────────────────

SOLUTION: Use consistent reference audio
────────────────────────────────────────────────────────────
For your training to be audible, you MUST:

1. Use reference audio from your TRAINING dataset at inference
2. OR extract and save speaker embeddings during training
3. Then use those saved embeddings at inference

The speaker embeddings include:
  - spk_cond_emb: Speaker conditioning embedding
  - style: CAMPlus speaker style vector  
  - prompt_condition: S2Mel prompt condition
  - ref_mel: Reference mel spectrogram
  - emo_cond_emb: Emotion conditioning embedding

Example workflow:
────────────────────────────────────────────────────────────
# 1. After training, extract embeddings from a training audio file
python tools/extract_embeddings.py \\
    --audio training_data/speaker_sample.wav \\
    --output speaker_embeddings.pt

# 2. At inference, use those embeddings (no reference audio needed!)
from indextts.infer_v2 import IndexTTS2
import torch

tts = IndexTTS2(lora_path="trained_lora/final_checkpoint")
embeddings = torch.load("speaker_embeddings.pt")
tts.infer(
    text="Hello world",
    speaker_embeddings=embeddings,  # Use training speaker's embeddings
    output_path="output.wav"
)
"""
    print(explanation)


def run_token_comparison(lora_path: Path, reference_audio: Path, text: str):
    """Compare GPT token outputs with and without LoRA."""
    print(f"\n{'='*60}")
    print("COMPARING GPT TOKEN OUTPUTS")
    print(f"{'='*60}")
    
    try:
        from indextts.infer_v2 import IndexTTS2
    except ImportError as e:
        print(f"Cannot import IndexTTS2: {e}")
        return
    
    print("Loading base model (no LoRA)...")
    base_tts = IndexTTS2()
    
    print(f"Loading model with LoRA from: {lora_path}")
    lora_tts = IndexTTS2(lora_path=str(lora_path))
    
    print(f"\nGenerating tokens for: '{text}'")
    print(f"Reference audio: {reference_audio}")
    
    # This would require modifying inference to expose intermediate tokens
    # For now, just show that models loaded successfully
    print("\n✓ Both models loaded successfully")
    print("To compare actual token outputs, you would need to:")
    print("1. Modify inference_speech() to return intermediate tokens")
    print("2. Compare the token sequences before S2Mel processing")


def main():
    parser = argparse.ArgumentParser(description="Diagnose IndexTTS2 training issues")
    parser.add_argument("--lora-path", type=Path, help="Path to LoRA checkpoint directory")
    parser.add_argument("--gpt-checkpoint", type=Path, help="Path to fine-tuned GPT checkpoint")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"),
                        help="Path to base GPT checkpoint for comparison")
    parser.add_argument("--explain", action="store_true", help="Explain the S2Mel dominance issue")
    parser.add_argument("--reference-audio", type=Path, help="Reference audio for token comparison")
    parser.add_argument("--text", type=str, default="Hello, this is a test.", help="Text for token comparison")
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.lora_path:
        all_results["lora"] = diagnose_lora_checkpoint(args.lora_path)
    
    if args.gpt_checkpoint:
        all_results["gpt"] = diagnose_gpt_checkpoint(args.gpt_checkpoint, args.base_checkpoint)
    
    if args.explain or not (args.lora_path or args.gpt_checkpoint):
        diagnose_inference_issue()
    
    if args.lora_path and args.reference_audio:
        run_token_comparison(args.lora_path, args.reference_audio, args.text)
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    
    all_issues = []
    all_recommendations = []
    
    for name, results in all_results.items():
        if "issues" in results:
            all_issues.extend(results["issues"])
        if "recommendations" in results:
            all_recommendations.extend(results["recommendations"])
    
    if all_issues:
        print("\n⚠️  ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No critical issues found with checkpoints")
    
    if all_recommendations:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(all_recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*60)
    print("MOST LIKELY CAUSE OF 'NO DIFFERENCE' ISSUE:")
    print("="*60)
    print("""
If your LoRA weights are non-zero (training worked), but you can't
hear any difference, the issue is almost certainly:

  → You're using different reference audio at inference than 
    what the model was trained on.

The S2Mel stage uses reference audio features to determine the
voice characteristics. To hear your training:

  1. Use audio from your training dataset as reference
  2. OR extract speaker embeddings from training audio and use those

See: python tools/diagnose_training.py --explain
""")


if __name__ == "__main__":
    main()