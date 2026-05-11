#!/usr/bin/env python3
"""
Complete Prosody Training Pipeline for IndexTTS2

This pipeline trains BOTH:
1. GPT LoRA - learns text-to-semantic mapping (WHAT patterns to produce)
2. S2Mel LoRA - learns semantic-to-mel patterns (HOW patterns sound acoustically)

WHY TRAIN BOTH:
===============
- GPT alone: Learns semantic mapping but acoustic patterns may not transfer well
- S2Mel alone: Learns acoustic patterns but needs correct semantic input
- BOTH together: Complete prosodic pattern learning!

The Pipeline:
=============
1. Transcribe audio (with verbatim details using dual transcription)
2. Prepare GPT dataset (text -> semantic codes)
3. Prepare S2Mel dataset (semantic codes -> mel spectrograms)
4. Train GPT LoRA (learns stutter text -> stutter semantics)
5. Train S2Mel LoRA (learns stutter semantics -> stutter acoustics)
6. Extract speaker embeddings
7. Test combined inference

Usage:
======
    # Full pipeline
    python tools/train_prosody_pipeline.py --speaker ozzy
    
    # Skip transcription (if already done)
    python tools/train_prosody_pipeline.py --speaker ozzy --skip-transcribe
    
    # S2Mel only (if GPT already trained)
    python tools/train_prosody_pipeline.py --speaker ozzy --s2mel-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete prosody training pipeline (GPT + S2Mel LoRA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Speaker
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--audio-dir", type=Path, help="Custom audio directory")
    
    # Pipeline control
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription (use existing)")
    parser.add_argument("--skip-gpt", action="store_true",
                        help="Skip GPT LoRA training")
    parser.add_argument("--skip-s2mel", action="store_true",
                        help="Skip S2Mel LoRA training")
    parser.add_argument("--s2mel-only", action="store_true",
                        help="Only train S2Mel (requires existing GPT or skip)")
    parser.add_argument("--gpt-only", action="store_true",
                        help="Only train GPT LoRA")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test inference")
    
    # Training parameters
    parser.add_argument("--gpt-epochs", type=int, default=20)
    parser.add_argument("--s2mel-epochs", type=int, default=30)
    parser.add_argument("--gpt-lora-rank", type=int, default=16)
    parser.add_argument("--s2mel-lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    
    # Output
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    
    return parser.parse_args()


def run_command(cmd: list, description: str, cwd: Path = PROJECT_ROOT) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print()
    
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd),
            check=False,
        )
        
        if result.returncode != 0:
            print(f"\n WARNING: Command exited with code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"\n ERROR: {e}")
        return False


def check_transcripts(speaker_dir: Path) -> tuple[bool, Optional[Path]]:
    """Check if transcripts exist."""
    candidates = [
        speaker_dir / "dataset" / "transcripts_verbatim.csv",
        speaker_dir / "dataset" / "transcripts.csv",
    ]
    
    for c in candidates:
        if c.exists():
            return True, c
    return False, None


def check_gpt_dataset(speaker_dir: Path) -> bool:
    """Check if GPT dataset is prepared."""
    manifest = speaker_dir / "dataset" / "processed_verbatim" / "train_manifest.jsonl"
    return manifest.exists()


def check_s2mel_dataset(speaker_dir: Path) -> bool:
    """Check if S2Mel dataset is prepared."""
    manifest = speaker_dir / "dataset" / "processed_s2mel" / "train_manifest.jsonl"
    return manifest.exists()


def check_gpt_lora(speaker_dir: Path) -> Optional[Path]:
    """Check if GPT LoRA exists."""
    lora_dir = speaker_dir / "verbatim_training" / "best_checkpoint" / "lora"
    if lora_dir.exists():
        return lora_dir
    return None


def check_s2mel_lora(speaker_dir: Path) -> Optional[Path]:
    """Check if S2Mel LoRA exists."""
    lora_dir = speaker_dir / "s2mel_training" / "best_checkpoint"
    if lora_dir.exists():
        return lora_dir
    return None


def main():
    args = parse_args()
    
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    audio_dir = args.audio_dir or speaker_dir / "dataset" / "audio"
    
    print("=" * 70)
    print("COMPLETE PROSODY TRAINING PIPELINE")
    print("GPT LoRA + S2Mel LoRA for Speech Pattern Learning")
    print("=" * 70)
    print(f"""
Speaker: {args.speaker}
Audio directory: {audio_dir}

This pipeline trains BOTH models:
  1. GPT LoRA: Learns text -> semantic patterns
  2. S2Mel LoRA: Learns semantic -> acoustic patterns

Together they enable full prosodic pattern reproduction!
""")
    
    # Check prerequisites
    if not audio_dir.exists():
        print(f" Audio directory not found: {audio_dir}")
        print(f"\nPlease place audio files in: {audio_dir}")
        sys.exit(1)
    
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files")
    
    results = {
        "speaker": args.speaker,
        "timestamp": datetime.now().isoformat(),
        "steps": {},
    }
    
    # ================================================================
    # STEP 1: TRANSCRIPTION
    # ================================================================
    has_transcripts, transcript_path = check_transcripts(speaker_dir)
    
    if args.skip_transcribe and has_transcripts:
        print(f"\n[1/7] SKIPPING transcription (using: {transcript_path})")
        results["steps"]["transcribe"] = "skipped"
    elif args.skip_transcribe:
        print(f"\n[1/7] SKIPPING transcription but no transcripts found!")
        print("     Run without --skip-transcribe or provide transcripts")
        results["steps"]["transcribe"] = "skip_failed"
    else:
        # Try dual transcription first (best for verbatim)
        dual_script = PROJECT_ROOT / "tools" / "transcribe_dual.py"
        single_script = PROJECT_ROOT / "tools" / "transcribe_dataset.py"
        
        if dual_script.exists():
            success = run_command(
                ["python", str(dual_script), "--speaker", args.speaker],
                "Dual transcription (verbatim + clean)",
            )
        else:
            success = run_command(
                ["python", str(single_script), "--speaker", args.speaker, "--verbatim"],
                "Verbatim transcription",
            )
        
        results["steps"]["transcribe"] = "success" if success else "failed"
        
        if not success:
            print("\n Transcription failed!")
            print("Please ensure whisper/faster-whisper is installed")
    
    # ================================================================
    # STEP 2: PREPARE GPT DATASET
    # ================================================================
    if args.s2mel_only:
        print(f"\n[2/7] SKIPPING GPT dataset preparation (--s2mel-only)")
        results["steps"]["gpt_dataset"] = "skipped"
    elif check_gpt_dataset(speaker_dir):
        print(f"\n[2/7] GPT dataset already exists, skipping preparation")
        results["steps"]["gpt_dataset"] = "exists"
    else:
        success = run_command(
            ["python", "tools/prepare_verbatim_dataset.py", "--speaker", args.speaker],
            "Prepare GPT verbatim dataset",
        )
        results["steps"]["gpt_dataset"] = "success" if success else "failed"
    
    # ================================================================
    # STEP 3: PREPARE S2Mel DATASET
    # ================================================================
    if args.gpt_only:
        print(f"\n[3/7] SKIPPING S2Mel dataset preparation (--gpt-only)")
        results["steps"]["s2mel_dataset"] = "skipped"
    elif check_s2mel_dataset(speaker_dir):
        print(f"\n[3/7] S2Mel dataset already exists, skipping preparation")
        results["steps"]["s2mel_dataset"] = "exists"
    else:
        success = run_command(
            ["python", "tools/prepare_s2mel_dataset.py", "--speaker", args.speaker],
            "Prepare S2Mel dataset",
        )
        results["steps"]["s2mel_dataset"] = "success" if success else "failed"
    
    # ================================================================
    # STEP 4: TRAIN GPT LoRA
    # ================================================================
    gpt_lora_path = check_gpt_lora(speaker_dir)
    
    if args.skip_gpt or args.s2mel_only:
        print(f"\n[4/7] SKIPPING GPT LoRA training")
        results["steps"]["gpt_training"] = "skipped"
    elif gpt_lora_path:
        print(f"\n[4/7] GPT LoRA already exists: {gpt_lora_path}")
        results["steps"]["gpt_training"] = "exists"
    else:
        success = run_command(
            [
                "python", "tools/train_verbatim_lora.py",
                "--speaker", args.speaker,
                "--epochs", str(args.gpt_epochs),
                "--lora-rank", str(args.gpt_lora_rank),
                "--batch-size", str(args.batch_size),
            ],
            "Train GPT LoRA (text -> semantic patterns)",
        )
        results["steps"]["gpt_training"] = "success" if success else "failed"
        
        if success:
            gpt_lora_path = check_gpt_lora(speaker_dir)
    
    # ================================================================
    # STEP 5: TRAIN S2Mel LoRA
    # ================================================================
    s2mel_lora_path = check_s2mel_lora(speaker_dir)
    
    if args.skip_s2mel or args.gpt_only:
        print(f"\n[5/7] SKIPPING S2Mel LoRA training")
        results["steps"]["s2mel_training"] = "skipped"
    elif s2mel_lora_path:
        print(f"\n[5/7] S2Mel LoRA already exists: {s2mel_lora_path}")
        results["steps"]["s2mel_training"] = "exists"
    else:
        success = run_command(
            [
                "python", "tools/train_s2mel_lora.py",
                "--speaker", args.speaker,
                "--epochs", str(args.s2mel_epochs),
                "--lora-rank", str(args.s2mel_lora_rank),
                "--batch-size", str(args.batch_size),
            ],
            "Train S2Mel LoRA (semantic -> acoustic patterns)",
        )
        results["steps"]["s2mel_training"] = "success" if success else "failed"
        
        if success:
            s2mel_lora_path = check_s2mel_lora(speaker_dir)
    
    # ================================================================
    # STEP 6: EXTRACT SPEAKER EMBEDDINGS
    # ================================================================
    embeddings_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
    
    if embeddings_path.exists():
        print(f"\n[6/7] Speaker embeddings already exist: {embeddings_path}")
        results["steps"]["embeddings"] = "exists"
    else:
        extract_script = PROJECT_ROOT / "tools" / "extract_embeddings.py"
        if extract_script.exists():
            success = run_command(
                ["python", str(extract_script), "--speaker", args.speaker],
                "Extract speaker embeddings",
            )
            results["steps"]["embeddings"] = "success" if success else "failed"
        else:
            print(f"\n[6/7] SKIPPING embeddings extraction (script not found)")
            results["steps"]["embeddings"] = "skipped"
    
    # ================================================================
    # STEP 7: TEST INFERENCE
    # ================================================================
    if args.skip_test:
        print(f"\n[7/7] SKIPPING test inference")
        results["steps"]["test"] = "skipped"
    elif s2mel_lora_path:
        test_output = speaker_dir / "test_outputs"
        test_output.mkdir(parents=True, exist_ok=True)
        
        test_text = "I I I was going going to the store today."
        
        success = run_command(
            [
                "python", "tools/infer_with_s2mel_lora.py",
                "--speaker", args.speaker,
                "--text", test_text,
                "--output", str(test_output / "test_s2mel_lora.wav"),
                "--compare",
            ],
            "Test S2Mel LoRA inference",
        )
        results["steps"]["test"] = "success" if success else "failed"
    else:
        print(f"\n[7/7] SKIPPING test (no S2Mel LoRA trained)")
        results["steps"]["test"] = "skipped"
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    print("\nResults:")
    for step, status in results["steps"].items():
        emoji = "" if status in ["success", "exists"] else "" if status == "skipped" else ""
        print(f"  {emoji} {step}: {status}")
    
    # Save results
    output_dir = args.output_dir or speaker_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print usage instructions
    gpt_lora_path = check_gpt_lora(speaker_dir)
    s2mel_lora_path = check_s2mel_lora(speaker_dir)
    
    print(f"""
{'='*70}
HOW TO USE YOUR TRAINED MODELS
{'='*70}

Trained LoRA Models:
  GPT LoRA: {gpt_lora_path or 'Not trained'}
  S2Mel LoRA: {s2mel_lora_path or 'Not trained'}

INFERENCE OPTIONS:
==================

1. S2Mel LoRA Only (RECOMMENDED for prosodic patterns):
   python tools/infer_with_s2mel_lora.py \\
       --speaker {args.speaker} \\
       --text "I I I was going going to the store..." \\
       --output output.wav

2. GPT LoRA Only (for semantic patterns):
   python tools/infer_verbatim.py \\
       --speaker {args.speaker} \\
       --text "I I I was going going to the store..."

3. Combined (GPT + S2Mel LoRA):
   python tools/infer_with_s2mel_lora.py \\
       --speaker {args.speaker} \\
       --gpt-lora {gpt_lora_path} \\
       --text "I I I was going going to the store..."

WHY THIS WORKS FOR STUTTERS:
============================
- GPT LoRA: Ensures stuttered text maps to correct semantic codes
- S2Mel LoRA: Ensures semantic codes produce authentic stutter acoustics
- Together: Complete prosodic pattern reproduction!

The key insight is that S2Mel training captures the ACTUAL acoustic
patterns of stutters in the mel spectrogram domain. This is where
prosody, timing, and rhythm are encoded!
""")


if __name__ == "__main__":
    main()