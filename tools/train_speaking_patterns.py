#!/usr/bin/env python3
"""
Complete Speaking Pattern Training Pipeline for IndexTTS2

This is the ONE SCRIPT you need to train speaking patterns (stutters, pauses, etc.)!

THE KEY INSIGHT:
================
Previous training approaches failed because each sample used DIFFERENT conditioning.
At inference, you used yet another conditioning vector → no patterns transferred.

This script uses GLOBAL CONDITIONING: the SAME conditioning for ALL training samples
AND inference. This ensures patterns are learned and reproduced.

What this script does:
1. Transcribes audio with Whisper (detecting pauses, fillers)
2. Extracts GLOBAL conditioning from training audio
3. Prepares dataset with global conditioning (ALL samples use SAME conditioning)
4. Trains LoRA adapters
5. Extracts speaker embeddings (for S2Mel stage)
6. Tests the trained model

Usage:
    # Simple - just put audio files in training/ozzy/dataset/audio/
    python tools/train_speaking_patterns.py --speaker ozzy
    
    # With custom options
    python tools/train_speaking_patterns.py --speaker ozzy \
        --whisper-model large-v3 \
        --epochs 30 \
        --lora-rank 32

Directory structure (created automatically):
    training/
      ozzy/
        dataset/
          audio/                        # ← Put your audio files here
          transcripts_verbatim.csv      # Generated
          processed_v2/                 # Generated (with global conditioning)
            train_manifest.jsonl
            val_manifest.jsonl
            features/
              GLOBAL_condition.npy      # Same for ALL samples!
              GLOBAL_emo_vec.npy        # Same for ALL samples!
              sample_text_ids.npy
              sample_codes.npy
        pattern_conditioning.pt         # Global conditioning for inference
        embeddings/
          speaker_embeddings.pt         # For S2Mel stage
        lora/
          final_checkpoint/             # Trained LoRA
        test_outputs/                   # Test audio files
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_step(step_num: int, title: str, total_steps: int = 7):
    """Print a formatted step header."""
    print("\n" + "=" * 60)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 60)


def run_transcription(speaker_dir: Path, args) -> Path:
    """Run Whisper transcription."""
    from tools.transcribe_dataset import (
        WhisperTranscriber, get_audio_files, 
        save_transcripts_csv, save_stats
    )
    from tqdm import tqdm
    
    audio_dir = speaker_dir / "dataset" / "audio"
    output_dir = speaker_dir / "dataset"
    output_csv = output_dir / "transcripts_verbatim.csv"
    
    # Check for existing transcripts
    if output_csv.exists() and not args.retranscribe:
        print(f"✓ Using existing transcripts: {output_csv}")
        print("  (Use --retranscribe to regenerate)")
        return output_csv
    
    # Find audio files
    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {audio_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load Whisper
    print(f"Loading Whisper model: {args.whisper_model}...")
    transcriber = WhisperTranscriber(
        model_name=args.whisper_model,
        device="auto",
        use_faster_whisper=True,
    )
    
    # Transcribe
    print("Transcribing...")
    results = []
    for audio_path in tqdm(audio_files, desc="Transcribing"):
        try:
            result = transcriber.transcribe(
                str(audio_path),
                language=args.language,
                min_pause_duration=0.3,
                long_pause_duration=0.7,
                detect_fillers=True,
            )
            results.append(result)
        except Exception as e:
            print(f"  Warning: Failed to transcribe {audio_path.name}: {e}")
    
    if not results:
        raise ValueError("No files were successfully transcribed")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    save_transcripts_csv(results, output_csv)
    save_stats(results, output_dir / "transcription_stats.json")
    
    # Print examples
    print("\nSample transcriptions:")
    for result in results[:3]:
        print(f"  {Path(result.audio_path).name}:")
        print(f"    {result.verbatim_text[:80]}...")
    
    return output_csv


def run_pattern_conditioning_extraction(speaker_dir: Path, args) -> Path:
    """Extract GLOBAL pattern conditioning."""
    from indextts.infer_v2 import IndexTTS2
    from indextts.pattern_conditioning import PatternConditioningStore
    from tools.transcribe_dataset import get_audio_files
    
    audio_dir = speaker_dir / "dataset" / "audio"
    output_path = speaker_dir / "pattern_conditioning.pt"
    
    # Check for existing
    if output_path.exists() and not args.reextract_conditioning:
        print(f"✓ Using existing pattern conditioning: {output_path}")
        print("  (Use --reextract-conditioning to regenerate)")
        return output_path
    
    # Find audio files
    audio_files = get_audio_files(audio_dir)[:args.max_conditioning_files]
    
    print(f"Extracting global conditioning from {len(audio_files)} audio files...")
    print("\nThis is KEY: all training samples will use the SAME conditioning!")
    
    # Load model
    if not hasattr(run_pattern_conditioning_extraction, 'tts'):
        print("Loading IndexTTS2 model...")
        run_pattern_conditioning_extraction.tts = IndexTTS2(use_cuda_kernel=False)
    tts = run_pattern_conditioning_extraction.tts
    
    # Extract
    store = PatternConditioningStore(tts)
    store.extract_global_conditioning(
        [str(f) for f in audio_files],
        method="average",
        verbose=True
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)
    
    return output_path


def run_dataset_preparation(speaker_dir: Path, conditioning_path: Path, args) -> tuple:
    """Prepare dataset with GLOBAL conditioning."""
    import subprocess
    
    audio_dir = speaker_dir / "dataset" / "audio"
    transcripts_csv = speaker_dir / "dataset" / "transcripts_verbatim.csv"
    output_dir = speaker_dir / "dataset" / "processed_v2"
    train_manifest = output_dir / "train_manifest.jsonl"
    val_manifest = output_dir / "val_manifest.jsonl"
    
    # Check for existing
    if train_manifest.exists() and val_manifest.exists() and not args.reprepare:
        print(f"✓ Using existing manifests:")
        print(f"    {train_manifest}")
        print(f"    {val_manifest}")
        print("  (Use --reprepare to regenerate)")
        return train_manifest, val_manifest
    
    # Run preparation
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "prepare_pattern_dataset_v2.py"),
        "--pattern-conditioning", str(conditioning_path),
        "--audio-dir", str(audio_dir),
        "--transcripts", str(transcripts_csv),
        "--output-dir", str(output_dir),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError("Dataset preparation failed")
    
    return train_manifest, val_manifest


def run_training(speaker_dir: Path, train_manifest: Path, val_manifest: Path, args) -> Path:
    """Run LoRA training."""
    import subprocess
    
    output_dir = speaker_dir / "lora"
    checkpoint_dir = output_dir / "final_checkpoint"
    
    # Check for existing
    if (checkpoint_dir / "adapter_config.json").exists() and not args.retrain:
        print(f"✓ Using existing LoRA checkpoint: {checkpoint_dir}")
        print("  (Use --retrain to retrain)")
        return checkpoint_dir
    
    # Run training
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "train_gpt_lora.py"),
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--output-dir", str(output_dir),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation", str(args.grad_accumulation),
    ]
    
    if args.amp:
        cmd.append("--amp")
    
    print(f"Running training with:")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError("Training failed")
    
    return checkpoint_dir


def run_embedding_extraction(speaker_dir: Path, args) -> Path:
    """Extract speaker embeddings for S2Mel stage."""
    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    from tools.transcribe_dataset import get_audio_files
    
    embeddings_dir = speaker_dir / "embeddings"
    embeddings_file = embeddings_dir / "speaker_embeddings.pt"
    
    # Check for existing
    if embeddings_file.exists() and not args.reextract_embeddings:
        print(f"✓ Using existing embeddings: {embeddings_file}")
        print("  (Use --reextract-embeddings to regenerate)")
        return embeddings_file
    
    audio_dir = speaker_dir / "dataset" / "audio"
    audio_files = get_audio_files(audio_dir)[:args.max_embedding_files]
    
    print(f"Extracting speaker embeddings from {len(audio_files)} audio files...")
    
    # Load base model (no LoRA needed for S2Mel embeddings)
    if not hasattr(run_embedding_extraction, 'tts'):
        print("Loading IndexTTS2 model...")
        run_embedding_extraction.tts = IndexTTS2(use_cuda_kernel=False)
    tts = run_embedding_extraction.tts
    
    store = SpeakerEmbeddingStore(tts)
    
    # Extract averaged embeddings
    embeddings = store.extract_averaged_embeddings(
        [str(f) for f in audio_files],
        verbose=True
    )
    
    # Save
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    store.save_embeddings(embeddings, embeddings_file)
    
    return embeddings_file


def run_test_inference(speaker_dir: Path, lora_path: Path, conditioning_path: Path, embeddings_path: Path, args):
    """Run test inference with pattern conditioning."""
    import subprocess
    
    output_dir = speaker_dir / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test sentences
    test_texts = args.test_texts or [
        "Hello, this is a test of the trained voice model. I wonder... does it sound good?",
        "Life finds a way, doesn't it? Let's see if we are happy with the results.",
        "Well, let me tell you something interesting. It's quite remarkable really.",
    ]
    
    print("Generating test outputs with pattern conditioning...")
    for i, text in enumerate(test_texts):
        output_path = output_dir / f"test_{i+1:02d}.wav"
        print(f"\n  [{i+1}] \"{text}\"")
        print(f"      → {output_path}")
        
        cmd = [
            sys.executable, str(PROJECT_ROOT / "tools" / "infer_pattern.py"),
            "--lora-path", str(lora_path),
            "--pattern-conditioning", str(conditioning_path),
            "--embeddings", str(embeddings_path),
            "--text", text,
            "--output", str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"      ⚠ Failed: {result.stderr[:200]}")
        else:
            print(f"      ✓ Generated")
    
    print(f"\n✓ Test outputs saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Complete speaking pattern training pipeline (with global conditioning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--speaker", "-s", required=True,
                        help="Speaker name (creates training/{speaker}/ structure)")
    
    # Whisper options
    parser.add_argument("--whisper-model", "-w", default="medium",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (auto-detected if not specified)")
    
    # Pattern conditioning options
    parser.add_argument("--max-conditioning-files", type=int, default=20,
                        help="Max files for global conditioning extraction (default: 20)")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32 for patterns)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grad-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    
    # Embedding options
    parser.add_argument("--max-embedding-files", type=int, default=10,
                        help="Max files for speaker embedding extraction (default: 10)")
    
    # Control options
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription step")
    parser.add_argument("--skip-conditioning", action="store_true",
                        help="Skip conditioning extraction")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding extraction")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test inference")
    
    # Regeneration options
    parser.add_argument("--retranscribe", action="store_true",
                        help="Force re-transcription")
    parser.add_argument("--reextract-conditioning", action="store_true",
                        help="Force pattern conditioning re-extraction")
    parser.add_argument("--reprepare", action="store_true",
                        help="Force dataset re-preparation")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training")
    parser.add_argument("--reextract-embeddings", action="store_true",
                        help="Force speaker embedding re-extraction")
    
    # Test options
    parser.add_argument("--test-texts", nargs="+",
                        help="Custom test sentences")
    
    args = parser.parse_args()
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    audio_dir = speaker_dir / "dataset" / "audio"
    
    # Print header
    print("=" * 60)
    print("SPEAKING PATTERN TRAINING PIPELINE (v2)")
    print("=" * 60)
    print(f"""
This pipeline trains your model to reproduce speaking PATTERNS:
  - Pauses and hesitations (like "well... you know...")
  - Stutters and repetitions (like Ozzy's distinctive speech)
  - Filler words (uh, um, er)
  - Rhythm and pacing

THE KEY: Global conditioning ensures patterns transfer to inference!
""")
    print(f"Speaker: {args.speaker}")
    print(f"Audio directory: {audio_dir}")
    print(f"Training config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Check audio directory
    if not audio_dir.exists():
        print(f"\n❌ Audio directory not found!")
        print(f"\nPlease create and add audio files:")
        print(f"  mkdir -p {audio_dir}")
        print(f"  # Copy your audio files (wav, mp3, flac) to:")
        print(f"  {audio_dir}/")
        sys.exit(1)
    
    # Check for audio files
    from tools.transcribe_dataset import get_audio_files
    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        print(f"\n❌ No audio files found in: {audio_dir}")
        print("\nSupported formats: wav, mp3, flac, ogg, m4a, opus, webm")
        sys.exit(1)
    
    print(f"\n✓ Found {len(audio_files)} audio files")
    
    start_time = time.time()
    
    try:
        # Step 1: Transcription
        if not args.skip_transcribe:
            print_step(1, "TRANSCRIPTION (Whisper)", 7)
            transcripts_csv = run_transcription(speaker_dir, args)
        else:
            print_step(1, "TRANSCRIPTION (Skipped)", 7)
        
        # Step 2: Extract GLOBAL pattern conditioning
        if not args.skip_conditioning:
            print_step(2, "PATTERN CONDITIONING EXTRACTION", 7)
            conditioning_path = run_pattern_conditioning_extraction(speaker_dir, args)
        else:
            conditioning_path = speaker_dir / "pattern_conditioning.pt"
            print_step(2, "PATTERN CONDITIONING (Skipped)", 7)
        
        # Step 3: Dataset preparation with global conditioning
        if not args.skip_prepare:
            print_step(3, "DATASET PREPARATION (with Global Conditioning)", 7)
            train_manifest, val_manifest = run_dataset_preparation(
                speaker_dir, conditioning_path, args
            )
        else:
            train_manifest = speaker_dir / "dataset" / "processed_v2" / "train_manifest.jsonl"
            val_manifest = speaker_dir / "dataset" / "processed_v2" / "val_manifest.jsonl"
            print_step(3, "DATASET PREPARATION (Skipped)", 7)
        
        # Step 4: Training
        if not args.skip_train:
            print_step(4, "LORA TRAINING", 7)
            lora_path = run_training(speaker_dir, train_manifest, val_manifest, args)
        else:
            lora_path = speaker_dir / "lora" / "final_checkpoint"
            print_step(4, "LORA TRAINING (Skipped)", 7)
        
        # Step 5: Speaker embedding extraction
        if not args.skip_embeddings:
            print_step(5, "SPEAKER EMBEDDING EXTRACTION", 7)
            embeddings_path = run_embedding_extraction(speaker_dir, args)
        else:
            embeddings_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
            print_step(5, "SPEAKER EMBEDDINGS (Skipped)", 7)
        
        # Step 6: Test inference
        if not args.skip_test:
            print_step(6, "TEST INFERENCE", 7)
            test_output_dir = run_test_inference(
                speaker_dir, lora_path, conditioning_path, embeddings_path, args
            )
        else:
            print_step(6, "TEST INFERENCE (Skipped)", 7)
        
        # Step 7: Done!
        elapsed = time.time() - start_time
        
        print_step(7, "COMPLETE!", 7)
        print(f"""
Total time: {elapsed/60:.1f} minutes

Outputs:
  LoRA checkpoint:         {lora_path}
  Pattern conditioning:    {conditioning_path}
  Speaker embeddings:      {embeddings_path}
  Test audio:              {speaker_dir / 'test_outputs'}

To use the trained model with patterns:
=======================================

    from indextts.infer_v2 import IndexTTS2
    from indextts.pattern_conditioning import PatternConditioningStore
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    
    # Load model with LoRA
    tts = IndexTTS2(lora_path="{lora_path}")
    
    # Load pattern conditioning (for GPT stage)
    pattern_cond = PatternConditioningStore.load("{conditioning_path}")
    
    # Load speaker embeddings (for S2Mel stage)
    store = SpeakerEmbeddingStore(tts)
    speaker_emb = store.load_embeddings("{embeddings_path}")
    
    # Inference with patterns
    tts.infer(
        text="Your text here",
        speaker_embeddings=speaker_emb,
        output_path="output.wav"
    )

Or use the inference tool:

    python tools/infer_pattern.py --speaker {args.speaker} \\
        --text "Your text here"

WHY THIS WORKS:
===============
All training samples used the SAME global conditioning (from pattern_conditioning.pt).
At inference, we use the SAME conditioning → patterns transfer!
""")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()