#!/usr/bin/env python3
"""
Extract Global Pattern Conditioning for Speaking Pattern Training

This is the KEY STEP to make speaking pattern training work!

The Problem:
============
Regular training extracts conditioning from each audio file individually.
At inference, you use a DIFFERENT audio file → DIFFERENT conditioning.
The patterns are tied to specific conditioning vectors the model never sees.

The Solution:
=============
Extract GLOBAL conditioning from your training audio ONCE, then use it for:
1. ALL training samples (so patterns aren't tied to individual embeddings)
2. Inference (so the model sees the same conditioning it was trained with)

Usage:
    # Extract conditioning from your training audio
    python tools/extract_pattern_conditioning.py \
        --speaker ozzy \
        --output training/ozzy/pattern_conditioning.pt
    
    # Or specify audio directory directly
    python tools/extract_pattern_conditioning.py \
        --audio-dir path/to/audio \
        --output conditioning.pt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Extract global pattern conditioning for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument("--speaker", "-s",
                        help="Speaker name (uses training/{speaker}/dataset/audio)")
    parser.add_argument("--audio-dir", type=Path,
                        help="Custom audio directory (overrides --speaker)")
    
    # Output
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path (default: training/{speaker}/pattern_conditioning.pt)")
    
    # Extraction options
    parser.add_argument("--max-files", type=int, default=20,
                        help="Maximum audio files to use (default: 20)")
    parser.add_argument("--method", choices=["average", "longest", "first"], default="average",
                        help="How to combine multiple audio files (default: average)")
    
    # Export options
    parser.add_argument("--export-npy", action="store_true",
                        help="Also export as .npy files for manual training")
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="Directory for .npy export (default: same as output)")
    
    args = parser.parse_args()
    
    # Determine audio directory
    if args.speaker:
        audio_dir = PROJECT_ROOT / "training" / args.speaker / "dataset" / "audio"
        default_output = PROJECT_ROOT / "training" / args.speaker / "pattern_conditioning.pt"
    elif args.audio_dir:
        audio_dir = args.audio_dir
        default_output = audio_dir.parent / "pattern_conditioning.pt"
    else:
        parser.error("Either --speaker or --audio-dir is required")
    
    output_path = args.output or default_output
    
    # Validate
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("PATTERN CONDITIONING EXTRACTION")
    print("=" * 60)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Output path: {output_path}")
    print(f"Method: {args.method}")
    print(f"Max files: {args.max_files}")
    
    # Import here to avoid slow startup
    from indextts.infer_v2 import IndexTTS2
    from indextts.pattern_conditioning import PatternConditioningStore, export_conditioning_for_training
    
    # Find audio files
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    audio_files = sorted(audio_files)[:args.max_files]
    
    if not audio_files:
        print(f"❌ No audio files found in: {audio_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Load model
    print("\nLoading IndexTTS2 model...")
    tts = IndexTTS2(use_cuda_kernel=False)
    
    # Extract conditioning
    print("\nExtracting pattern conditioning...")
    store = PatternConditioningStore(tts)
    conditioning = store.extract_global_conditioning(
        [str(f) for f in audio_files],
        method=args.method,
        verbose=True
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)
    
    # Export as npy if requested
    if args.export_npy:
        export_dir = args.export_dir or output_path.parent
        print("\nExporting to numpy format...")
        export_conditioning_for_training(output_path, export_dir)
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"""
Pattern conditioning saved to: {output_path}

NEXT STEPS for Pattern Training:
================================

1. Prepare dataset with GLOBAL conditioning:
   python tools/prepare_pattern_dataset_v2.py \\
       --speaker {args.speaker or 'your_speaker'} \\
       --pattern-conditioning {output_path}

2. Train with the prepared dataset:
   python tools/train_gpt_lora.py \\
       --train-manifest training/{args.speaker or 'speaker'}/dataset/processed_v2/train_manifest.jsonl \\
       --val-manifest training/{args.speaker or 'speaker'}/dataset/processed_v2/val_manifest.jsonl \\
       --lora-rank 32 --epochs 30

3. Inference using the SAME conditioning:
   python tools/infer.py \\
       --lora-path training/{args.speaker or 'speaker'}/lora/final_checkpoint \\
       --pattern-conditioning {output_path} \\
       --text "Your text here"

WHY THIS WORKS:
===============
By using the SAME conditioning for all training samples AND inference,
the model learns:
  "When I see THIS conditioning + any text → add speaker's patterns"

Instead of:
  "When I see conditioning_X + text_X → patterns (only for sample X)"
""")


if __name__ == "__main__":
    main()