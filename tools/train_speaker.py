#!/usr/bin/env python3
"""
All-in-One Speaker Training Pipeline for IndexTTS2

This is the EASIEST way to train a speaker's voice patterns!

Just provide audio files, and this script will:
1. Transcribe all audio with Whisper (detecting pauses & fillers)
2. Prepare the training dataset
3. Train LoRA adapters
4. Extract speaker embeddings
5. Test the trained model

Usage:
    # Simplest - just specify the speaker name
    python tools/train_speaker.py --speaker goldblum
    
    # Audio is expected in: training/goldblum/dataset/audio/
    
    # Custom settings
    python tools/train_speaker.py --speaker goldblum \
        --whisper-model large-v3 \
        --epochs 30 \
        --lora-rank 32

Directory Structure (created automatically):
    training/
      goldblum/
        dataset/
          audio/                    # ← Put your audio files here
          transcripts_verbatim.csv  # Generated
          processed/                # Generated
            train_manifest.jsonl
            val_manifest.jsonl
        lora/
          final_checkpoint/         # Trained LoRA
        embeddings/
          speaker_embeddings.pt     # For inference
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


def print_step(step_num: int, title: str, total_steps: int = 6):
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


def run_dataset_preparation(speaker_dir: Path, transcripts_csv: Path, args) -> tuple:
    """Prepare training dataset."""
    from tools.prepare_pattern_dataset import main as prepare_main
    import subprocess
    
    audio_dir = speaker_dir / "dataset" / "audio"
    output_dir = speaker_dir / "dataset" / "processed"
    train_manifest = output_dir / "train_manifest.jsonl"
    val_manifest = output_dir / "val_manifest.jsonl"
    
    # Check for existing manifests
    if train_manifest.exists() and val_manifest.exists() and not args.reprepare:
        print(f"✓ Using existing manifests:")
        print(f"    {train_manifest}")
        print(f"    {val_manifest}")
        print("  (Use --reprepare to regenerate)")
        return train_manifest, val_manifest
    
    # Run dataset preparation
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "prepare_pattern_dataset.py"),
        "--audio-dir", str(audio_dir),
        "--transcripts", str(transcripts_csv),
        "--output-dir", str(output_dir),
        "--detect-pauses",
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
    
    # Check for existing checkpoint
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
    """Extract speaker embeddings."""
    
    embeddings_dir = speaker_dir / "embeddings"
    embeddings_file = embeddings_dir / "speaker_embeddings.pt"
    
    # Check for existing embeddings
    if embeddings_file.exists() and not args.reextract:
        print(f"✓ Using existing embeddings: {embeddings_file}")
        print("  (Use --reextract to regenerate)")
        return embeddings_file
    
    # Import here to avoid slow startup
    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    from tools.transcribe_dataset import get_audio_files
    
    audio_dir = speaker_dir / "dataset" / "audio"
    audio_files = get_audio_files(audio_dir)[:args.max_embedding_files]
    
    print(f"Extracting embeddings from {len(audio_files)} audio files...")
    
    # Load base model (no LoRA needed for extraction)
    tts = IndexTTS2(use_cuda_kernel=False)
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


def run_test_inference(speaker_dir: Path, lora_path: Path, embeddings_path: Path, args):
    """Run test inference to verify training."""
    
    output_dir = speaker_dir / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid slow startup
    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    import torch
    
    # Test sentences
    test_texts = args.test_texts or [
        "Hello, this is a test of the trained voice model. I wonder... does it sound good?",
        "Life finds a way, doesn't it? Let's see if we are happy with the results or not",
        "Well, let me tell you something interesting about that. It's like something is completely changed, can you feel it!?",
    ]
    
    print("Loading trained model...")
    tts = IndexTTS2(lora_path=str(lora_path), use_cuda_kernel=False)
    
    print("Loading speaker embeddings...")
    store = SpeakerEmbeddingStore(tts)
    embeddings = store.load_embeddings(embeddings_path)
    
    print("\nGenerating test outputs...")
    for i, text in enumerate(test_texts):
        output_path = output_dir / f"test_{i+1:02d}.wav"
        print(f"\n  [{i+1}] \"{text}\"")
        print(f"      → {output_path}")
        
        tts.infer(
            text=text,
            speaker_embeddings=embeddings,
            output_path=str(output_path),
        )
    
    print(f"\n✓ Test outputs saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="All-in-one speaker training pipeline for IndexTTS2",
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
    
    # Training options
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16, use 32 for more pattern capacity)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32, usually 2x rank)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grad-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    
    # Embedding options
    parser.add_argument("--max-embedding-files", type=int, default=10,
                        help="Max files to use for embedding extraction (default: 10)")
    
    # Control options
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription step (use existing transcripts)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation step")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training step (use existing checkpoint)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding extraction")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip test inference")
    
    # Regeneration options
    parser.add_argument("--retranscribe", action="store_true",
                        help="Force re-transcription even if transcripts exist")
    parser.add_argument("--reprepare", action="store_true",
                        help="Force re-preparation even if manifests exist")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training even if checkpoint exists")
    parser.add_argument("--reextract", action="store_true",
                        help="Force re-extraction even if embeddings exist")
    
    # Test options
    parser.add_argument("--test-texts", nargs="+",
                        help="Custom test sentences for inference")
    
    args = parser.parse_args()
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    audio_dir = speaker_dir / "dataset" / "audio"
    
    # Print header
    print("=" * 60)
    print("INDEXTTS2 SPEAKER TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nSpeaker: {args.speaker}")
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
            print_step(1, "TRANSCRIPTION (Whisper)")
            transcripts_csv = run_transcription(speaker_dir, args)
        else:
            transcripts_csv = speaker_dir / "dataset" / "transcripts_verbatim.csv"
            print_step(1, "TRANSCRIPTION (Skipped)")
        
        # Step 2: Dataset preparation
        if not args.skip_prepare:
            print_step(2, "DATASET PREPARATION")
            train_manifest, val_manifest = run_dataset_preparation(
                speaker_dir, transcripts_csv, args
            )
        else:
            train_manifest = speaker_dir / "dataset" / "processed" / "train_manifest.jsonl"
            val_manifest = speaker_dir / "dataset" / "processed" / "val_manifest.jsonl"
            print_step(2, "DATASET PREPARATION (Skipped)")
        
        # Step 3: Training
        if not args.skip_train:
            print_step(3, "LORA TRAINING")
            lora_path = run_training(speaker_dir, train_manifest, val_manifest, args)
        else:
            lora_path = speaker_dir / "lora" / "final_checkpoint"
            print_step(3, "LORA TRAINING (Skipped)")
        
        # Step 4: Embedding extraction
        if not args.skip_embeddings:
            print_step(4, "EMBEDDING EXTRACTION")
            embeddings_path = run_embedding_extraction(speaker_dir, args)
        else:
            embeddings_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
            print_step(4, "EMBEDDING EXTRACTION (Skipped)")
        
        # Step 5: Test inference
        if not args.skip_test:
            print_step(5, "TEST INFERENCE")
            test_output_dir = run_test_inference(
                speaker_dir, lora_path, embeddings_path, args
            )
        else:
            print_step(5, "TEST INFERENCE (Skipped)")
        
        # Done!
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"""
Total time: {elapsed/60:.1f} minutes

Outputs:
  LoRA checkpoint:      {lora_path}
  Speaker embeddings:   {embeddings_path}
  Test audio:           {speaker_dir / 'test_outputs'}

To use the trained model:

    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    
    tts = IndexTTS2(lora_path="{lora_path}")
    store = SpeakerEmbeddingStore(tts)
    embeddings = store.load_embeddings("{embeddings_path}")
    
    tts.infer(
        text="Your text here",
        speaker_embeddings=embeddings,
        output_path="output.wav"
    )

Or use the inference tool:

    python tools/infer.py --speaker {args.speaker} \\
        --lora-path {lora_path} \\
        --text "Your text here"
""")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()