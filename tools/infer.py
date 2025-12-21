#!/usr/bin/env python3
"""
Simple Inference Tool for IndexTTS2

Run inference with trained models using the standard project structure.

PROJECT STRUCTURE:
==================
    training/
      {speaker_name}/
        embeddings/
          speaker_embeddings.pt  # Required for promptless inference
        lora/
          final_checkpoint/      # LoRA checkpoint (optional)
        finetune/
          best_model.pth         # Full finetune checkpoint (optional)

Usage:
    # Simple inference with trained speaker
    python tools/infer.py --speaker goldblum --text "Hello, how are you?"
    
    # With specific output path
    python tools/infer.py --speaker goldblum --text "Hello" --output output.wav
    
    # Use only LoRA (skip full finetune if both exist)
    python tools/infer.py --speaker goldblum --text "Hello" --model-type lora
    
    # Use base model only (for comparison)
    python tools/infer.py --speaker goldblum --text "Hello" --model-type base
    
    # List available speakers
    python tools/infer.py --list-speakers
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

# Default paths
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
    
    return {
        "speaker_dir": speaker_dir,
        "audio_dir": speaker_dir / "dataset" / "audio",
        "embeddings_file": speaker_dir / "embeddings" / "speaker_embeddings.pt",
        "lora_checkpoint": speaker_dir / "lora" / "final_checkpoint",
        "finetune_checkpoint": speaker_dir / "finetune" / "best_model.pth",
        "outputs_dir": speaker_dir / "outputs",
    }


def print_speaker_status(speaker_name: str):
    """Print the status of a speaker."""
    paths = get_speaker_paths(speaker_name)
    
    print(f"\n📁 Speaker: {speaker_name}")
    
    # Embeddings
    if paths["embeddings_file"].exists():
        print(f"   ✅ Embeddings: Ready")
    else:
        print(f"   ❌ Embeddings: Not found")
        print(f"      Run: python tools/extract_embeddings.py --speaker {speaker_name}")
    
    # Models
    has_lora = paths["lora_checkpoint"].exists()
    has_finetune = paths["finetune_checkpoint"].exists()
    
    if has_lora:
        print(f"   ✅ LoRA: {paths['lora_checkpoint'].name}")
    else:
        print(f"   ⚪ LoRA: Not trained")
    
    if has_finetune:
        print(f"   ✅ Finetune: {paths['finetune_checkpoint'].name}")
    else:
        print(f"   ⚪ Finetune: Not trained")


def run_inference(args):
    """Run TTS inference."""
    
    # Lazy imports
    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    
    paths = get_speaker_paths(args.speaker)
    
    print("=" * 60)
    print("IndexTTS2 Inference")
    print("=" * 60)
    
    print_speaker_status(args.speaker)
    
    # Check embeddings
    if not paths["embeddings_file"].exists():
        # Check for reference audio as fallback
        if args.reference_audio:
            print(f"\n>> Using reference audio: {args.reference_audio}")
        else:
            print(f"\n❌ Error: Speaker embeddings not found!")
            print(f"   Run: python tools/extract_embeddings.py --speaker {args.speaker}")
            sys.exit(1)
    
    # Determine which model to load
    lora_path = None
    gpt_checkpoint = None
    model_type_used = "base"
    
    if args.model_type == "base":
        pass  # Use base model only
    elif args.model_type == "lora":
        if paths["lora_checkpoint"].exists():
            lora_path = str(paths["lora_checkpoint"])
            model_type_used = "lora"
        else:
            print("\n⚠️  LoRA checkpoint not found, using base model")
    elif args.model_type == "finetune":
        if paths["finetune_checkpoint"].exists():
            gpt_checkpoint = str(paths["finetune_checkpoint"])
            model_type_used = "finetune"
        else:
            print("\n⚠️  Finetune checkpoint not found, using base model")
    else:  # auto
        # Prefer finetune > lora > base
        if paths["finetune_checkpoint"].exists():
            gpt_checkpoint = str(paths["finetune_checkpoint"])
            model_type_used = "finetune"
        elif paths["lora_checkpoint"].exists():
            lora_path = str(paths["lora_checkpoint"])
            model_type_used = "lora"
    
    print(f"\n>> Model type: {model_type_used}")
    if lora_path:
        print(f"   LoRA: {lora_path}")
    if gpt_checkpoint:
        print(f"   GPT: {gpt_checkpoint}")
    
    # Initialize TTS
    print("\n>> Initializing IndexTTS2...")
    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=args.cuda_kernel,
        lora_path=lora_path,
        gpt_checkpoint=gpt_checkpoint,
    )
    
    # Load speaker embeddings
    speaker_embeddings = None
    if not args.reference_audio and paths["embeddings_file"].exists():
        print(f">> Loading speaker embeddings...")
        store = SpeakerEmbeddingStore(tts)
        speaker_embeddings = store.load_embeddings(paths["embeddings_file"])
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output path
        paths["outputs_dir"].mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c if c.isalnum() or c in "- " else "_" for c in args.text[:30]).strip()
        safe_text = safe_text.replace(" ", "_")
        output_path = paths["outputs_dir"] / f"{timestamp}_{model_type_used}_{safe_text}.wav"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\n>> Synthesizing: \"{args.text}\"")
    print(f">> Output: {output_path}")
    
    start_time = time.time()
    
    if speaker_embeddings:
        tts.infer(
            spk_audio_prompt=None,
            text=args.text,
            output_path=str(output_path),
            speaker_embeddings=speaker_embeddings,
            verbose=args.verbose,
        )
    elif args.reference_audio:
        tts.infer(
            spk_audio_prompt=args.reference_audio,
            text=args.text,
            output_path=str(output_path),
            verbose=args.verbose,
        )
    else:
        print("❌ Error: No speaker embeddings or reference audio available")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ INFERENCE COMPLETE")
    print("=" * 60)
    print(f"   Output: {output_path}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Model: {model_type_used}")


def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Speaker selection
    parser.add_argument("--speaker", "-s",
                       help="Speaker name from training/ directory")
    parser.add_argument("--list-speakers", action="store_true",
                       help="List all available speakers and exit")
    
    # Required
    parser.add_argument("--text", "-t",
                       help="Text to synthesize")
    parser.add_argument("--output", "-o",
                       help="Output audio file path (auto-generated if not specified)")
    
    # Model selection
    parser.add_argument("--model-type", choices=["auto", "base", "lora", "finetune"],
                       default="auto",
                       help="Which model to use (default: auto = prefer finetune > lora > base)")
    
    # Fallback
    parser.add_argument("--reference-audio", "-r",
                       help="Reference audio file (fallback if no embeddings)")
    
    # Model configuration
    parser.add_argument("--config", default="checkpoints/config.yaml",
                       help="Path to config file")
    parser.add_argument("--model-dir", default="checkpoints",
                       help="Path to model directory")
    parser.add_argument("--cuda-kernel", action="store_true",
                       help="Use BigVGAN CUDA kernel")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print verbose output")
    
    args = parser.parse_args()
    
    # List speakers
    if args.list_speakers:
        speakers = get_available_speakers()
        if speakers:
            print("\n📢 Available speakers:")
            for speaker in speakers:
                print_speaker_status(speaker)
        else:
            print("\n❌ No speakers found in training/ directory")
        return
    
    # Validate
    if not args.speaker:
        speakers = get_available_speakers()
        if speakers:
            print("\n💡 Available speakers:")
            for s in speakers:
                print(f"   - {s}")
        parser.error("--speaker is required")
    
    if not args.text:
        parser.error("--text is required")
    
    # Check speaker exists
    paths = get_speaker_paths(args.speaker)
    if not paths["speaker_dir"].exists():
        print(f"\n❌ Error: Speaker not found: {args.speaker}")
        print("\nAvailable speakers:")
        for s in get_available_speakers():
            print(f"   - {s}")
        sys.exit(1)
    
    run_inference(args)


if __name__ == "__main__":
    main()