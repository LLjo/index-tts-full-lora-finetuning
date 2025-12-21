#!/usr/bin/env python3
"""
Validation script for LoRA checkpoints.

This script helps you:
1. Validate LoRA checkpoint integrity
2. Compare multiple checkpoints
3. Generate test samples
4. Evaluate voice quality metrics

Usage:
    python tools/validate_lora.py \
        --lora-path trained_lora/my_voice/final_checkpoint \
        --reference-audio reference.wav \
        --test-texts test_sentences.txt \
        --output-dir validation_results
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from indextts.infer_v2 import IndexTTS2
from indextts.utils.lora_utils import list_lora_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LoRA checkpoints")
    parser.add_argument(
        "--lora-path",
        type=Path,
        help="Path to LoRA checkpoint directory (or parent directory containing multiple checkpoints)"
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        required=True,
        help="Reference audio for voice prompt"
    )
    parser.add_argument(
        "--test-texts",
        type=Path,
        help="File with test sentences (one per line). If not provided, uses default tests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Base model directory"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also generate samples with base model (no LoRA) for comparison"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps, xpu)"
    )
    return parser.parse_args()


def load_test_texts(test_file: Path = None):
    """Load test sentences from file or return defaults."""
    if test_file and test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    
    # Default test sentences (multilingual)
    return [
        "Hello, this is a test of the fine-tuned voice.",
        "The quick brown fox jumps over the lazy dog.",
        "这是一个中文测试句子。",
        "How does the quality sound with longer sentences?",
        "Testing emotional expression with excitement!",
        "A calm and neutral statement for comparison.",
    ]


def validate_checkpoint(lora_path: Path, model_dir: Path, device: str):
    """Validate that a checkpoint can be loaded correctly."""
    print(f"\nValidating checkpoint: {lora_path}")
    
    try:
        # Check required files
        config_file = lora_path / "adapter_config.json"
        model_file = lora_path / "adapter_model.bin"
        safetensors_file = lora_path / "adapter_model.safetensors"
        
        if not config_file.exists():
            print(f"  ✗ Missing adapter_config.json")
            return False
        
        if not model_file.exists() and not safetensors_file.exists():
            print(f"  ✗ Missing adapter weights (adapter_model.bin or .safetensors)")
            return False
        
        print(f"  ✓ Required files present")
        
        # Load config
        with open(config_file, "r") as f:
            config = json.load(f)
        
        print(f"  ✓ LoRA config valid")
        print(f"    - Rank: {config.get('r', 'N/A')}")
        print(f"    - Alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"    - Target modules: {len(config.get('target_modules', []))}")
        
        # Try loading the model
        try:
            tts = IndexTTS2(
                model_dir=str(model_dir),
                lora_path=str(lora_path),
                device=device
            )
            print(f"  ✓ Successfully loaded into IndexTTS2")
            return True
        except Exception as e:
            print(f"  ✗ Failed to load into IndexTTS2: {e}")
            return False
            
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False


def generate_samples(
    lora_path: Path,
    reference_audio: Path,
    test_texts: list,
    output_dir: Path,
    model_dir: Path,
    checkpoint_name: str,
    device: str
):
    """Generate test samples from a checkpoint."""
    print(f"\nGenerating samples for: {checkpoint_name}")
    
    # Create output directory
    sample_dir = output_dir / checkpoint_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    tts = IndexTTS2(
        model_dir=str(model_dir),
        lora_path=str(lora_path) if lora_path else None,
        device=device
    )
    
    # Generate samples
    results = []
    for i, text in enumerate(test_texts):
        output_file = sample_dir / f"sample_{i:02d}.wav"
        
        print(f"  Generating {i+1}/{len(test_texts)}: {text[:50]}...")
        
        try:
            tts.infer(
                spk_audio_prompt=str(reference_audio),
                text=text,
                output_path=str(output_file),
                verbose=False
            )
            
            results.append({
                "index": i,
                "text": text,
                "output": str(output_file),
                "success": True
            })
            print(f"    ✓ Saved: {output_file.name}")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results.append({
                "index": i,
                "text": text,
                "error": str(e),
                "success": False
            })
    
    # Save results
    results_file = sample_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    success_count = sum(1 for r in results if r["success"])
    print(f"  Generated {success_count}/{len(test_texts)} samples successfully")
    
    return results


def main():
    args = parse_args()
    
    print("=" * 80)
    print("LoRA Checkpoint Validation")
    print("=" * 80)
    
    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"\nDevice: {device}")
    print(f"Reference audio: {args.reference_audio}")
    print(f"Output directory: {args.output_dir}")
    
    # Load test texts
    test_texts = load_test_texts(args.test_texts)
    print(f"Test sentences: {len(test_texts)}")
    
    # Find checkpoints to validate
    checkpoints = []
    
    if args.lora_path:
        lora_path = args.lora_path.resolve()
        
        # Check if it's a single checkpoint or directory of checkpoints
        if (lora_path / "adapter_config.json").exists():
            # Single checkpoint
            checkpoints.append(("checkpoint", lora_path))
        else:
            # Directory of checkpoints
            found = list_lora_checkpoints(lora_path)
            if not found:
                print(f"\nNo LoRA checkpoints found in: {lora_path}")
                print("Looking for directories containing adapter_config.json")
                sys.exit(1)
            
            checkpoints = [(Path(ckpt["path"]).name, Path(ckpt["path"])) for ckpt in found]
            print(f"\nFound {len(checkpoints)} checkpoints:")
            for name, path in checkpoints:
                print(f"  - {name}")
    
    # Add base model if requested
    if args.compare_base:
        checkpoints.insert(0, ("base_model", None))
    
    if not checkpoints:
        print("\nNo checkpoints to validate!")
        print("Specify --lora-path to a checkpoint or directory of checkpoints")
        sys.exit(1)
    
    # Validate each checkpoint
    print("\n" + "=" * 80)
    print("VALIDATION PHASE")
    print("=" * 80)
    
    valid_checkpoints = []
    for name, path in checkpoints:
        if path is None:
            # Base model, skip validation
            valid_checkpoints.append((name, path))
            print(f"\nBase model: will be used for comparison")
        else:
            is_valid = validate_checkpoint(path, args.model_dir, device)
            if is_valid:
                valid_checkpoints.append((name, path))
    
    if not valid_checkpoints:
        print("\n✗ No valid checkpoints found!")
        sys.exit(1)
    
    print(f"\n✓ {len(valid_checkpoints)} checkpoint(s) validated successfully")
    
    # Generate samples
    print("\n" + "=" * 80)
    print("SAMPLE GENERATION PHASE")
    print("=" * 80)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for name, path in valid_checkpoints:
        results = generate_samples(
            lora_path=path,
            reference_audio=args.reference_audio,
            test_texts=test_texts,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            checkpoint_name=name,
            device=device
        )
        all_results[name] = results
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = {
        "reference_audio": str(args.reference_audio),
        "test_sentences": len(test_texts),
        "checkpoints_validated": len(valid_checkpoints),
        "device": device,
        "results": all_results
    }
    
    summary_file = args.output_dir / "validation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nValidation complete!")
    print(f"  Checkpoints tested: {len(valid_checkpoints)}")
    print(f"  Samples per checkpoint: {len(test_texts)}")
    print(f"  Total samples generated: {sum(len(r) for r in all_results.values())}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  Summary: {summary_file}")
    
    for name in all_results.keys():
        print(f"  {name}/: Generated samples")
    
    print("\nNext steps:")
    print("  1. Listen to generated samples in each checkpoint directory")
    print("  2. Compare quality across checkpoints")
    print("  3. Choose the best checkpoint for your use case")
    print(f"  4. Review summary: {summary_file}")


if __name__ == "__main__":
    main()