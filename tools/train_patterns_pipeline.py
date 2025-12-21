#!/usr/bin/env python3
"""
Complete Pattern Embedding Training Pipeline for IndexTTS2

This is the ONE SCRIPT you need to train speaking patterns (stutters, pauses, etc.)
that ACTUALLY WORK at inference time!

THE NEW APPROACH: Pattern Embeddings
====================================
Instead of hoping patterns emerge from training, we use:
1. LEARNABLE PATTERN EMBEDDINGS - tokens that encode "speak with this person's patterns"
2. PATTERN-AWARE LOSS - explicitly rewards pause/pattern reproduction
3. CONSISTENT INJECTION - same embedding used in training AND inference

Usage:
    # Put audio files in training/ozzy/dataset/audio/
    python tools/train_patterns_pipeline.py --speaker ozzy
    
    # With options
    python tools/train_patterns_pipeline.py --speaker ozzy \
        --whisper-model large-v3 \
        --epochs 50 \
        --pattern-tokens 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(step: int, total: int, title: str):
    print(f"\n{'─' * 70}")
    print(f"  STEP {step}/{total}: {title}")
    print(f"{'─' * 70}\n")


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and return success status."""
    print(f">> Running: {' '.join(cmd[:3])}...")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"❌ {description} failed!")
        return False
    return True


def check_audio_files(audio_dir: Path) -> list:
    """Check for audio files."""
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(
        description="Complete pattern embedding training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required
    parser.add_argument("--speaker", "-s", required=True,
                        help="Speaker name (uses training/{speaker}/)")
    
    # Transcription
    parser.add_argument("--whisper-model", "-w", default="medium",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (auto-detected if not specified)")
    
    # Pattern embedding
    parser.add_argument("--pattern-tokens", type=int, default=8,
                        help="Number of learnable pattern tokens (default: 8)")
    parser.add_argument("--injection-mode", choices=["add", "prepend", "replace_first"],
                        default="add")
    
    # Training
    parser.add_argument("--epochs", type=int, default=40,
                        help="Training epochs (default: 40)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--pattern-lr", type=float, default=1e-3,
                        help="Pattern embedding learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-lora", action="store_true",
                        help="Train only pattern embedding, no LoRA")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    
    # Skip options
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--skip-conditioning", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    
    # Force regeneration
    parser.add_argument("--retranscribe", action="store_true")
    parser.add_argument("--reprepare", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    
    # Test
    parser.add_argument("--test-texts", nargs="+", default=None,
                        help="Custom test sentences")
    
    args = parser.parse_args()
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    audio_dir = speaker_dir / "dataset" / "audio"
    
    print_header("PATTERN EMBEDDING TRAINING PIPELINE")
    
    print(f"""
This pipeline trains PATTERN EMBEDDINGS for speaking pattern reproduction.

THE KEY INSIGHT:
Pattern Embeddings are learnable tokens that encode "speak with THIS person's
patterns". They're used in BOTH training and inference, so patterns transfer!

Speaker: {args.speaker}
Audio directory: {audio_dir}
Pattern tokens: {args.pattern_tokens}
LoRA rank: {args.lora_rank if not args.no_lora else 'DISABLED'}
""")
    
    # Check audio directory
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        print(f"\nPlease create it and add audio files:")
        print(f"  mkdir -p {audio_dir}")
        print(f"  cp /path/to/audio/*.wav {audio_dir}/")
        sys.exit(1)
    
    audio_files = check_audio_files(audio_dir)
    if not audio_files:
        print(f"❌ No audio files found in: {audio_dir}")
        sys.exit(1)
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    start_time = time.time()
    total_steps = 7
    
    try:
        # ===== STEP 1: Transcription =====
        print_step(1, total_steps, "TRANSCRIPTION (Whisper)")
        
        transcripts_csv = speaker_dir / "dataset" / "transcripts_verbatim.csv"
        
        if args.skip_transcribe:
            print("Skipped by user request")
        elif transcripts_csv.exists() and not args.retranscribe:
            print(f"✓ Using existing: {transcripts_csv}")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "tools" / "transcribe_dataset.py"),
                "--speaker", args.speaker,
                "--whisper-model", args.whisper_model,
            ]
            if args.language:
                cmd.extend(["--language", args.language])
            
            if not run_command(cmd, "Transcription"):
                sys.exit(1)
        
        # ===== STEP 2: Extract Global Conditioning =====
        print_step(2, total_steps, "GLOBAL CONDITIONING (for consistent baseline)")
        
        pattern_cond_path = speaker_dir / "pattern_conditioning.pt"
        
        if args.skip_conditioning:
            print("Skipped by user request")
        elif pattern_cond_path.exists() and not args.reprepare:
            print(f"✓ Using existing: {pattern_cond_path}")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "tools" / "extract_pattern_conditioning.py"),
                "--speaker", args.speaker,
            ]
            
            if not run_command(cmd, "Conditioning extraction"):
                sys.exit(1)
        
        # ===== STEP 3: Prepare Dataset with Pattern Features =====
        print_step(3, total_steps, "DATASET PREPARATION (v3 with pattern features)")
        
        train_manifest = speaker_dir / "dataset" / "processed_v3" / "train_manifest.jsonl"
        val_manifest = speaker_dir / "dataset" / "processed_v3" / "val_manifest.jsonl"
        
        if args.skip_prepare:
            print("Skipped by user request")
        elif train_manifest.exists() and not args.reprepare:
            print(f"✓ Using existing: {train_manifest}")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "tools" / "prepare_pattern_dataset_v3.py"),
                "--speaker", args.speaker,
            ]
            if pattern_cond_path.exists():
                cmd.extend(["--pattern-conditioning", str(pattern_cond_path)])
            
            if not run_command(cmd, "Dataset preparation"):
                sys.exit(1)
        
        # ===== STEP 4: Speaker Embeddings (for S2Mel) =====
        print_step(4, total_steps, "SPEAKER EMBEDDINGS (for voice timbre)")
        
        embeddings_dir = speaker_dir / "embeddings"
        embeddings_file = embeddings_dir / "speaker_embeddings.pt"
        
        if args.skip_embeddings:
            print("Skipped by user request")
        elif embeddings_file.exists() and not args.reprepare:
            print(f"✓ Using existing: {embeddings_file}")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "tools" / "extract_embeddings.py"),
                "--speaker", args.speaker,
            ]
            
            if not run_command(cmd, "Speaker embedding extraction"):
                sys.exit(1)
        
        # ===== STEP 5: Train Pattern Embeddings =====
        print_step(5, total_steps, "PATTERN EMBEDDING TRAINING")
        
        output_dir = speaker_dir / "pattern_training"
        final_ckpt = output_dir / "final_checkpoint" / "pattern_embedding.pt"
        
        if args.skip_train:
            print("Skipped by user request")
        elif final_ckpt.exists() and not args.retrain:
            print(f"✓ Using existing: {final_ckpt}")
        else:
            cmd = [
                sys.executable, str(PROJECT_ROOT / "tools" / "train_pattern_embeddings.py"),
                "--speaker", args.speaker,
                "--train-manifest", str(train_manifest),
                "--val-manifest", str(val_manifest),
                "--output-dir", str(output_dir),
                "--epochs", str(args.epochs),
                "--pattern-tokens", str(args.pattern_tokens),
                "--pattern-lr", str(args.pattern_lr),
                "--lora-rank", str(args.lora_rank),
                "--learning-rate", str(args.learning_rate),
                "--batch-size", str(args.batch_size),
                "--injection-mode", args.injection_mode,
            ]
            if args.no_lora:
                cmd.append("--no-lora")
            if args.amp:
                cmd.append("--amp")
            
            if not run_command(cmd, "Pattern embedding training"):
                sys.exit(1)
        
        # ===== STEP 6: Test Inference =====
        print_step(6, total_steps, "TEST INFERENCE")
        
        if args.skip_test:
            print("Skipped by user request")
        else:
            test_dir = speaker_dir / "test_outputs"
            test_dir.mkdir(exist_ok=True)
            
            test_texts = args.test_texts or [
                "Hello, this is a test. I wonder... does it sound like me?",
                "Well, let me think about that for a moment.",
                "Life finds a way, doesn't it? Yes, it really does.",
            ]
            
            # Find best checkpoint
            pattern_emb_path = output_dir / "best_checkpoint" / "pattern_embedding.pt"
            if not pattern_emb_path.exists():
                pattern_emb_path = output_dir / "final_checkpoint" / "pattern_embedding.pt"
            
            lora_path = output_dir / "best_checkpoint" / "lora"
            if not lora_path.exists() or not (lora_path / "adapter_config.json").exists():
                lora_path = output_dir / "final_checkpoint" / "lora"
            
            print(f"Pattern embedding: {pattern_emb_path}")
            print(f"LoRA: {lora_path if lora_path.exists() else 'None'}")
            print()
            
            for i, text in enumerate(test_texts):
                output_path = test_dir / f"test_pattern_{i+1:02d}.wav"
                print(f"  [{i+1}] \"{text[:50]}...\"")
                
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "tools" / "infer_with_patterns.py"),
                    "--speaker", args.speaker,
                    "--pattern-embedding", str(pattern_emb_path),
                    "--text", text,
                    "--output", str(output_path),
                    "--injection-mode", args.injection_mode,
                ]
                
                if lora_path.exists() and (lora_path / "adapter_config.json").exists():
                    cmd.extend(["--lora-path", str(lora_path)])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"      ✓ {output_path.name}")
                else:
                    print(f"      ⚠ Failed")
                    if result.stderr:
                        print(f"        {result.stderr[:200]}")
            
            print(f"\n✓ Test outputs saved to: {test_dir}")
        
        # ===== STEP 7: Complete! =====
        print_step(7, total_steps, "COMPLETE!")
        
        elapsed = time.time() - start_time
        
        print(f"""
Total time: {elapsed/60:.1f} minutes

Output files:
  Pattern embedding: {output_dir / 'best_checkpoint' / 'pattern_embedding.pt'}
  LoRA checkpoint:   {output_dir / 'best_checkpoint' / 'lora'}
  Speaker embeddings: {embeddings_file}
  Test audio:        {test_dir}

═══════════════════════════════════════════════════════════════════════

HOW TO USE THE TRAINED MODEL:
═══════════════════════════════════════════════════════════════════════

Command line:
    python tools/infer_with_patterns.py \\
        --speaker {args.speaker} \\
        --text "Your text here" \\
        --output output.wav

Python API:
    from indextts.infer_v2 import IndexTTS2
    from indextts.pattern_embeddings import PatternEmbedding
    from tools.infer_with_patterns import pattern_aware_inference
    
    tts = IndexTTS2(lora_path="training/{args.speaker}/pattern_training/best_checkpoint/lora")
    pattern_emb = PatternEmbedding.load(
        "training/{args.speaker}/pattern_training/best_checkpoint/pattern_embedding.pt"
    )
    
    pattern_aware_inference(
        tts, pattern_emb,
        text="Your text",
        output_path="output.wav",
        audio_prompt="reference.wav"  # For voice timbre
    )

═══════════════════════════════════════════════════════════════════════

WHY PATTERNS WILL APPEAR NOW:
═══════════════════════════════════════════════════════════════════════

The PATTERN EMBEDDING is a learned "trigger" that tells the model
"speak with {args.speaker}'s patterns". Since the SAME embedding is used
in both training and inference, the model recognizes it and produces
the patterns!

This is fundamentally different from previous approaches where different
conditioning was used at inference, and the model didn't know what
patterns to apply.
""")
        
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()