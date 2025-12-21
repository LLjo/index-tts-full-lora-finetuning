#!/usr/bin/env python3
"""
Model Comparison Tool for IndexTTS2

Easily compare outputs between:
- Base model (no fine-tuning)
- LoRA fine-tuned model
- Full fine-tuned model (custom GPT checkpoint)

This tool generates multiple audio files from the same text input
to help you evaluate the differences between models.

PROJECT STRUCTURE:
==================
The default structure assumed by this tool:
    training/
      {speaker_name}/
        dataset/
          audio/                 # Raw audio files
          transcripts.csv        # Transcriptions
          processed/             # Processed features & manifests
            train_manifest.jsonl
            val_manifest.jsonl
        embeddings/
          speaker_embeddings.pt  # Stored speaker embeddings
        lora/
          final_checkpoint/      # LoRA checkpoint
        finetune/
          best_model.pth         # Full finetune checkpoint

Usage:
    # Simple: Just specify the speaker name (uses default paths)
    python tools/compare_models.py --speaker goldblum --text "Hello, this is a test."

    # Compare LoRA with base model (explicit paths)
    python tools/compare_models.py \
        --text "Hello, this is a test of the voice synthesis." \
        --embeddings training/goldblum/embeddings/speaker_embeddings.pt \
        --lora training/goldblum/lora/final_checkpoint \
        --output-dir outputs/comparison

    # Compare full finetune with base model
    python tools/compare_models.py \
        --text "Hello, this is a test of the voice synthesis." \
        --embeddings training/goldblum/embeddings/speaker_embeddings.pt \
        --gpt-checkpoint training/goldblum/finetune/best_model.pth \
        --output-dir outputs/comparison

    # Compare with reference audio (traditional method)
    python tools/compare_models.py \
        --text "Hello, this is a test of the voice synthesis." \
        --reference-audio path/to/reference.wav \
        --lora training/goldblum/lora/final_checkpoint \
        --output-dir outputs/comparison
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

import torch

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
            # Check if it has dataset, lora, or finetune subdirectories
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
        "val_manifest": speaker_dir / "dataset" / "processed" / "val_manifest.jsonl",
        "embeddings_dir": speaker_dir / "embeddings",
        "embeddings_file": speaker_dir / "embeddings" / "speaker_embeddings.pt",
        "lora_dir": speaker_dir / "lora",
        "lora_checkpoint": speaker_dir / "lora" / "final_checkpoint",
        "finetune_dir": speaker_dir / "finetune",
        "finetune_checkpoint": speaker_dir / "finetune" / "best_model.pth",
        "outputs_dir": speaker_dir / "outputs",
    }
    
    return paths


def print_speaker_status(speaker_name: str):
    """Print the status of a speaker's training data and checkpoints."""
    paths = get_speaker_paths(speaker_name)
    
    print(f"\n📁 Speaker: {speaker_name}")
    print(f"   Location: {paths['speaker_dir']}")
    print()
    
    # Dataset status
    if paths["dataset_dir"].exists():
        audio_files = list(paths["audio_dir"].glob("*.wav")) if paths["audio_dir"].exists() else []
        print(f"   ✅ Dataset: {len(audio_files)} audio files")
        if paths["train_manifest"].exists():
            print(f"      ✅ Train manifest: {paths['train_manifest'].name}")
        else:
            print(f"      ❌ Train manifest: Not found")
    else:
        print(f"   ❌ Dataset: Not found")
    
    # Embeddings status
    if paths["embeddings_file"].exists():
        print(f"   ✅ Speaker embeddings: {paths['embeddings_file'].name}")
    else:
        print(f"   ❌ Speaker embeddings: Not extracted yet")
        print(f"      Run: python tools/extract_embeddings.py --speaker {speaker_name}")
    
    # LoRA status
    if paths["lora_checkpoint"].exists():
        print(f"   ✅ LoRA checkpoint: {paths['lora_checkpoint'].name}")
    else:
        print(f"   ⚪ LoRA checkpoint: Not trained")
    
    # Finetune status
    if paths["finetune_checkpoint"].exists():
        print(f"   ✅ Finetune checkpoint: {paths['finetune_checkpoint'].name}")
    else:
        print(f"   ⚪ Finetune checkpoint: Not trained")
    
    print()


def create_comparison(args):
    """Generate comparison audio files."""
    
    # Lazy imports to speed up --help
    from indextts.infer_v2 import IndexTTS2
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    
    print("=" * 70)
    print("IndexTTS2 Model Comparison Tool")
    print("=" * 70)
    
    # If speaker name provided, resolve paths automatically
    if args.speaker:
        paths = get_speaker_paths(args.speaker)
        print_speaker_status(args.speaker)
        
        # Auto-fill paths if not explicitly provided
        if not args.embeddings and paths["embeddings_file"].exists():
            args.embeddings = str(paths["embeddings_file"])
            print(f">> Using embeddings: {args.embeddings}")
        
        if not args.lora and paths["lora_checkpoint"].exists():
            args.lora = str(paths["lora_checkpoint"])
            print(f">> Using LoRA: {args.lora}")
        
        if not args.gpt_checkpoint and paths["finetune_checkpoint"].exists():
            args.gpt_checkpoint = str(paths["finetune_checkpoint"])
            print(f">> Using GPT checkpoint: {args.gpt_checkpoint}")
        
        if not args.reference_audio and paths["audio_dir"].exists():
            # Pick the first audio file as reference
            audio_files = list(paths["audio_dir"].glob("*.wav"))
            if audio_files:
                args.reference_audio = str(audio_files[0])
                print(f">> Using reference audio: {args.reference_audio}")
        
        # Use speaker-specific output directory
        if args.output_dir == "outputs/comparison":
            args.output_dir = str(paths["outputs_dir"] / "comparison")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize text for filename (first 30 chars)
    safe_text = "".join(c if c.isalnum() or c in "- " else "_" for c in args.text[:30]).strip()
    safe_text = safe_text.replace(" ", "_")
    
    results = []
    
    # ========== Load Speaker Embeddings (if provided) ==========
    speaker_embeddings = None
    if args.embeddings:
        print(f"\n>> Loading speaker embeddings from: {args.embeddings}")
        # We'll load these after initializing first TTS (need device info)
    
    # ========== 1. Base Model ==========
    print("\n" + "-" * 50)
    print("1. Generating with BASE MODEL")
    print("-" * 50)
    
    base_tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=args.cuda_kernel,
    )
    
    # Load embeddings now that we have device info
    if args.embeddings:
        store = SpeakerEmbeddingStore(base_tts)
        speaker_embeddings = store.load_embeddings(args.embeddings)
    
    base_output = output_dir / f"{timestamp}_1_base_{safe_text}.wav"
    
    start_time = time.time()
    if speaker_embeddings:
        base_tts.infer(
            spk_audio_prompt=None,
            text=args.text,
            output_path=str(base_output),
            speaker_embeddings=speaker_embeddings,
            verbose=args.verbose,
        )
    elif args.reference_audio:
        base_tts.infer(
            spk_audio_prompt=args.reference_audio,
            text=args.text,
            output_path=str(base_output),
            verbose=args.verbose,
        )
    else:
        print("ERROR: Either --embeddings or --reference-audio is required")
        sys.exit(1)
    
    base_time = time.time() - start_time
    results.append(("Base Model", str(base_output), base_time))
    print(f">> Saved: {base_output} ({base_time:.2f}s)")
    
    # Clean up base model to free memory
    del base_tts
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========== 2. LoRA Model (if specified) ==========
    if args.lora:
        print("\n" + "-" * 50)
        print("2. Generating with LORA MODEL")
        print("-" * 50)
        print(f"   LoRA path: {args.lora}")
        
        lora_tts = IndexTTS2(
            cfg_path=args.config,
            model_dir=args.model_dir,
            use_cuda_kernel=args.cuda_kernel,
            lora_path=args.lora,
        )
        
        lora_output = output_dir / f"{timestamp}_2_lora_{safe_text}.wav"
        
        start_time = time.time()
        if speaker_embeddings:
            lora_tts.infer(
                spk_audio_prompt=None,
                text=args.text,
                output_path=str(lora_output),
                speaker_embeddings=speaker_embeddings,
                verbose=args.verbose,
            )
        else:
            lora_tts.infer(
                spk_audio_prompt=args.reference_audio,
                text=args.text,
                output_path=str(lora_output),
                verbose=args.verbose,
            )
        
        lora_time = time.time() - start_time
        results.append(("LoRA Model", str(lora_output), lora_time))
        print(f">> Saved: {lora_output} ({lora_time:.2f}s)")
        
        del lora_tts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========== 3. Full Finetune Model (if specified) ==========
    if args.gpt_checkpoint:
        print("\n" + "-" * 50)
        print("3. Generating with FINETUNED GPT MODEL")
        print("-" * 50)
        print(f"   GPT checkpoint: {args.gpt_checkpoint}")
        
        ft_tts = IndexTTS2(
            cfg_path=args.config,
            model_dir=args.model_dir,
            use_cuda_kernel=args.cuda_kernel,
            gpt_checkpoint=args.gpt_checkpoint,
        )
        
        ft_output = output_dir / f"{timestamp}_3_finetuned_{safe_text}.wav"
        
        start_time = time.time()
        if speaker_embeddings:
            ft_tts.infer(
                spk_audio_prompt=None,
                text=args.text,
                output_path=str(ft_output),
                speaker_embeddings=speaker_embeddings,
                verbose=args.verbose,
            )
        else:
            ft_tts.infer(
                spk_audio_prompt=args.reference_audio,
                text=args.text,
                output_path=str(ft_output),
                verbose=args.verbose,
            )
        
        ft_time = time.time() - start_time
        results.append(("Finetuned GPT", str(ft_output), ft_time))
        print(f">> Saved: {ft_output} ({ft_time:.2f}s)")
        
        del ft_tts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========== 4. Generate with reference audio for comparison (if embeddings used) ==========
    if speaker_embeddings and args.reference_audio:
        print("\n" + "-" * 50)
        print("4. Generating with REFERENCE AUDIO (for comparison)")
        print("-" * 50)
        print(f"   Reference: {args.reference_audio}")
        
        ref_tts = IndexTTS2(
            cfg_path=args.config,
            model_dir=args.model_dir,
            use_cuda_kernel=args.cuda_kernel,
            lora_path=args.lora if args.lora else None,
            gpt_checkpoint=args.gpt_checkpoint if args.gpt_checkpoint else None,
        )
        
        ref_output = output_dir / f"{timestamp}_4_with_ref_audio_{safe_text}.wav"
        
        start_time = time.time()
        ref_tts.infer(
            spk_audio_prompt=args.reference_audio,
            text=args.text,
            output_path=str(ref_output),
            verbose=args.verbose,
        )
        
        ref_time = time.time() - start_time
        results.append(("With Ref Audio", str(ref_output), ref_time))
        print(f">> Saved: {ref_output} ({ref_time:.2f}s)")
        
        del ref_tts
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nText: \"{args.text}\"")
    print(f"\nGenerated files:")
    for i, (name, path, gen_time) in enumerate(results, 1):
        print(f"  {i}. {name:20s} -> {path} ({gen_time:.2f}s)")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create a summary file
    summary_path = output_dir / f"{timestamp}_comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"IndexTTS2 Model Comparison\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"\nText: {args.text}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  - Config: {args.config}\n")
        f.write(f"  - Model dir: {args.model_dir}\n")
        if args.embeddings:
            f.write(f"  - Speaker embeddings: {args.embeddings}\n")
        if args.reference_audio:
            f.write(f"  - Reference audio: {args.reference_audio}\n")
        if args.lora:
            f.write(f"  - LoRA checkpoint: {args.lora}\n")
        if args.gpt_checkpoint:
            f.write(f"  - GPT checkpoint: {args.gpt_checkpoint}\n")
        f.write(f"\nResults:\n")
        for name, path, gen_time in results:
            f.write(f"  - {name}: {path} ({gen_time:.2f}s)\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    if speaker_embeddings:
        print("\n💡 TIP: Listen to the files and compare:")
        print("   - Base Model uses the speaker embeddings without LoRA/finetuning")
        print("   - LoRA/Finetuned Model uses trained weights with the SAME embeddings")
        print("   - Any differences you hear are from the trained voice characteristics!")
    else:
        print("\n⚠️  WARNING: Using reference audio may mask training differences.")
        print("   For best comparison, use --embeddings with stored speaker embeddings.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare IndexTTS2 models (base vs fine-tuned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Speaker shortcut (auto-resolves paths)
    parser.add_argument("--speaker", "-s",
                       help="Speaker name from training/ directory (auto-resolves all paths)")
    parser.add_argument("--list-speakers", action="store_true",
                       help="List all available speakers and exit")
    
    # Required arguments
    parser.add_argument("--text", "-t",
                       help="Text to synthesize for comparison")
    parser.add_argument("--output-dir", "-o", default="outputs/comparison",
                       help="Directory to save comparison outputs")
    
    # Voice source (or use --speaker)
    voice_group = parser.add_argument_group("Voice Source (or use --speaker)")
    voice_group.add_argument("--embeddings", "-e",
                            help="Path to stored speaker embeddings (.pt file) - RECOMMENDED")
    voice_group.add_argument("--reference-audio", "-r",
                            help="Path to reference audio file (traditional method)")
    
    # Models to compare (or use --speaker)
    model_group = parser.add_argument_group("Models to Compare (or use --speaker)")
    model_group.add_argument("--lora",
                            help="Path to LoRA checkpoint directory")
    model_group.add_argument("--gpt-checkpoint",
                            help="Path to full fine-tuned GPT checkpoint")
    
    # Model configuration
    config_group = parser.add_argument_group("Model Configuration")
    config_group.add_argument("--config", default="checkpoints/config.yaml",
                             help="Path to config file")
    config_group.add_argument("--model-dir", default="checkpoints",
                             help="Path to model directory")
    config_group.add_argument("--cuda-kernel", action="store_true",
                             help="Use BigVGAN CUDA kernel (faster but requires compilation)")
    config_group.add_argument("--verbose", "-v", action="store_true",
                             help="Print verbose output")
    
    args = parser.parse_args()
    
    # List speakers and exit
    if args.list_speakers:
        speakers = get_available_speakers()
        if speakers:
            print("\n📢 Available speakers in training/ directory:")
            for speaker in speakers:
                print_speaker_status(speaker)
        else:
            print("\n❌ No speakers found in training/ directory")
            print("   Create a speaker directory with: training/{speaker_name}/dataset/audio/")
        return
    
    # Validate arguments
    if not args.text:
        parser.error("--text is required")
    
    if not args.speaker and not args.embeddings and not args.reference_audio:
        print("\n💡 TIP: Use --speaker to auto-resolve paths, or specify --embeddings or --reference-audio")
        speakers = get_available_speakers()
        if speakers:
            print("\nAvailable speakers:")
            for speaker in speakers:
                print(f"  - {speaker}")
        parser.error("At least one of --speaker, --embeddings, or --reference-audio is required")
    
    if not args.speaker and not args.lora and not args.gpt_checkpoint:
        print("NOTE: No --lora or --gpt-checkpoint specified.")
        print("      Will only generate base model output.\n")
    
    create_comparison(args)


if __name__ == "__main__":
    main()