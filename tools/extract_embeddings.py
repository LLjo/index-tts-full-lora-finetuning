#!/usr/bin/env python3
"""
Speaker Embeddings Extraction Tool for IndexTTS2

Extract and store speaker embeddings from training data for use in inference.
This is REQUIRED for comparing finetuned models with the base model effectively.

PROJECT STRUCTURE:
==================
    training/
      {speaker_name}/
        dataset/
          audio/                 # Raw audio files
          processed/
            train_manifest.jsonl # Training manifest
        embeddings/
          speaker_embeddings.pt  # Output: stored embeddings

Usage:
    # Extract from a speaker (uses training audio files)
    python tools/extract_embeddings.py --speaker goldblum
    
    # Extract from specific audio file
    python tools/extract_embeddings.py --speaker goldblum --audio path/to/audio.wav
    
    # Extract and average from multiple files (more stable)
    python tools/extract_embeddings.py --speaker goldblum --max-files 20
    
    # List available speakers
    python tools/extract_embeddings.py --list-speakers
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths based on project structure
TRAINING_DIR = PROJECT_ROOT / "training"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def get_available_speakers():
    """List all available speakers in the training directory."""
    if not TRAINING_DIR.exists():
        return []
    
    speakers = []
    for d in TRAINING_DIR.iterdir():
        if d.is_dir():
            has_data = (d / "dataset").exists() or (d / "lora").exists() or (d / "finetune").exists()
            if has_data:
                speakers.append(d.name)
    return sorted(speakers)


def get_speaker_paths(speaker_name: str) -> dict:
    """Get all relevant paths for a speaker."""
    speaker_dir = TRAINING_DIR / speaker_name
    
    paths = {
        "speaker_dir": speaker_dir,
        "dataset_dir": speaker_dir / "dataset",
        "audio_dir": speaker_dir / "dataset" / "audio",
        "processed_dir": speaker_dir / "dataset" / "processed",
        "train_manifest": speaker_dir / "dataset" / "processed" / "train_manifest.jsonl",
        "embeddings_dir": speaker_dir / "embeddings",
        "embeddings_file": speaker_dir / "embeddings" / "speaker_embeddings.pt",
        "lora_checkpoint": speaker_dir / "lora" / "final_checkpoint",
        "finetune_checkpoint": speaker_dir / "finetune" / "best_model.pth",
    }
    
    return paths


def print_speaker_info(speaker_name: str, paths: dict):
    """Print information about a speaker."""
    print(f"\n📁 Speaker: {speaker_name}")
    print(f"   Location: {paths['speaker_dir']}")
    
    # Count audio files
    audio_files = []
    if paths["audio_dir"].exists():
        audio_files = list(paths["audio_dir"].glob("*.wav"))
    print(f"   Audio files: {len(audio_files)}")
    
    # Check existing embeddings
    if paths["embeddings_file"].exists():
        print(f"   Embeddings: ✅ Already extracted")
        print(f"               ({paths['embeddings_file']})")
    else:
        print(f"   Embeddings: ❌ Not yet extracted")


def extract_embeddings_for_speaker(args):
    """Extract and save speaker embeddings."""
    
    # Lazy imports
    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    
    paths = get_speaker_paths(args.speaker)
    print_speaker_info(args.speaker, paths)
    
    # Determine audio files to use
    audio_files = []
    
    if args.audio:
        # Use specified audio file
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"\n❌ Error: Audio file not found: {args.audio}")
            sys.exit(1)
        audio_files = [audio_path]
        print(f"\n>> Using specified audio file: {args.audio}")
    else:
        # Get audio files from training data
        if paths["audio_dir"].exists():
            audio_files = sorted(paths["audio_dir"].glob("*.wav"))
        
        if not audio_files:
            print(f"\n❌ Error: No audio files found in {paths['audio_dir']}")
            print("   Provide audio files or specify --audio path")
            sys.exit(1)
        
        # Limit number of files if specified
        if args.max_files and len(audio_files) > args.max_files:
            print(f"\n>> Limiting to {args.max_files} files (from {len(audio_files)} available)")
            # Sample evenly distributed files
            step = len(audio_files) // args.max_files
            audio_files = audio_files[::step][:args.max_files]
        
        print(f"\n>> Using {len(audio_files)} audio files from training data")
    
    # Initialize TTS model (base model is fine for extraction)
    print("\n>> Initializing IndexTTS2 model...")
    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=False,  # Not needed for embedding extraction
    )
    
    # Create embedding store
    store = SpeakerEmbeddingStore(tts)
    
    # Extract embeddings
    if len(audio_files) == 1:
        # Single file extraction
        print(f"\n>> Extracting embeddings from: {audio_files[0]}")
        embeddings = store.extract_embeddings(str(audio_files[0]))
    else:
        # Multiple file extraction with averaging
        print(f"\n>> Extracting and averaging embeddings from {len(audio_files)} files...")
        embeddings = store.extract_averaged_embeddings(
            [str(f) for f in audio_files],
            verbose=True
        )
    
    # Create output directory
    paths["embeddings_dir"].mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    output_path = paths["embeddings_file"]
    store.save_embeddings(embeddings, output_path)
    
    print("\n" + "=" * 60)
    print("✅ EMBEDDINGS EXTRACTED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nSaved to: {output_path}")
    print("\nEmbedding shapes:")
    for key, value in embeddings.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    print("\n📝 Next steps:")
    print(f"   1. Compare models:")
    print(f"      python tools/compare_models.py --speaker {args.speaker} --text \"Your text here\"")
    print(f"   2. Run inference:")
    print(f"      python tools/infer.py --speaker {args.speaker} --text \"Your text here\"")


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings for IndexTTS2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Speaker selection
    parser.add_argument("--speaker", "-s",
                       help="Speaker name from training/ directory")
    parser.add_argument("--list-speakers", action="store_true",
                       help="List all available speakers and exit")
    
    # Audio source
    parser.add_argument("--audio", "-a",
                       help="Specific audio file to extract from (optional)")
    parser.add_argument("--max-files", "-m", type=int, default=10,
                       help="Maximum number of files to use for averaging (default: 10)")
    
    # Model configuration
    parser.add_argument("--config", default="checkpoints/config.yaml",
                       help="Path to config file")
    parser.add_argument("--model-dir", default="checkpoints",
                       help="Path to model directory")
    
    # Output
    parser.add_argument("--output", "-o",
                       help="Custom output path (default: training/{speaker}/embeddings/speaker_embeddings.pt)")
    
    args = parser.parse_args()
    
    # List speakers
    if args.list_speakers:
        speakers = get_available_speakers()
        if speakers:
            print("\n📢 Available speakers in training/ directory:")
            for speaker in speakers:
                paths = get_speaker_paths(speaker)
                print_speaker_info(speaker, paths)
        else:
            print("\n❌ No speakers found in training/ directory")
            print("   Create a speaker directory: training/{speaker_name}/dataset/audio/")
        return
    
    # Validate
    if not args.speaker:
        speakers = get_available_speakers()
        if speakers:
            print("\n💡 Available speakers:")
            for s in speakers:
                print(f"   - {s}")
        parser.error("--speaker is required")
    
    # Check speaker exists
    paths = get_speaker_paths(args.speaker)
    if not paths["speaker_dir"].exists():
        print(f"\n❌ Error: Speaker directory not found: {paths['speaker_dir']}")
        print("\nAvailable speakers:")
        for s in get_available_speakers():
            print(f"   - {s}")
        sys.exit(1)
    
    extract_embeddings_for_speaker(args)


if __name__ == "__main__":
    main()