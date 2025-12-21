#!/usr/bin/env python3
"""
Finetuned Model Inference Example for IndexTTS2

This script demonstrates the CORRECT way to use a finetuned model (LoRA or full finetune)
to synthesize speech that actually sounds like the trained voice.

WHY THIS IS NECESSARY:
=======================
IndexTTS2 uses a two-stage pipeline:
1. GPT Stage: Generates semantic tokens from text + speaker/emotion conditioning
2. S2Mel Stage: Converts semantic tokens to mel spectrogram using reference audio features

The problem: Even if you train the GPT stage with LoRA/finetuning, the S2Mel stage 
uses the reference audio's features (style, prompt_condition, ref_mel) at inference time.
This means the final voice timbre is determined by whatever audio you pass as reference,
NOT by what the model learned during training!

THE SOLUTION:
=============
Store speaker embeddings from your training data, then use those stored embeddings
during inference. This ensures the S2Mel stage uses the SAME voice characteristics
that the GPT was trained on.

WORKFLOW:
=========
1. After training, extract embeddings from ONE representative audio file from your training data
2. Save these embeddings to disk
3. At inference time, load the embeddings and pass them via `speaker_embeddings=` parameter
4. No reference audio file is needed anymore - the model uses learned voice characteristics

Usage:
    # Step 1: Extract and save embeddings (run once after training)
    python examples/finetuned_inference_example.py --extract \
        --audio training_data/speaker_sample.wav \
        --output checkpoints/my_speaker/embeddings.pt
    
    # Step 2: Run inference with stored embeddings
    python examples/finetuned_inference_example.py --infer \
        --embeddings checkpoints/my_speaker/embeddings.pt \
        --lora checkpoints/my_speaker/lora \
        --text "Hello, this is my finetuned voice!" \
        --output output.wav
"""

import argparse
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indextts.infer_v2 import IndexTTS2
from indextts.speaker_embeddings import SpeakerEmbeddingStore


def extract_embeddings(args):
    """Extract and save speaker embeddings from an audio file."""
    print("=" * 60)
    print("STEP 1: Extracting Speaker Embeddings")
    print("=" * 60)
    
    # Initialize TTS model (base model, no LoRA needed for extraction)
    print("\n>> Initializing IndexTTS2 model...")
    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=False,
    )
    
    # Create embedding store
    store = SpeakerEmbeddingStore(tts)
    
    # Extract embeddings from audio file
    print(f"\n>> Extracting embeddings from: {args.audio}")
    embeddings = store.extract_embeddings(args.audio)
    
    # Print embedding shapes for verification
    print("\n>> Extracted embeddings:")
    for key, value in embeddings.items():
        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Save embeddings
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    store.save_embeddings(embeddings, args.output)
    print(f"\n>> Embeddings saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! You can now use these embeddings for inference.")
    print("=" * 60)


def run_inference(args):
    """Run inference with stored speaker embeddings."""
    print("=" * 60)
    print("STEP 2: Running Inference with Stored Embeddings")
    print("=" * 60)
    
    # Initialize TTS model with LoRA (or full finetune checkpoint)
    print("\n>> Initializing IndexTTS2 model...")
    init_kwargs = {
        "cfg_path": args.config,
        "model_dir": args.model_dir,
        "use_cuda_kernel": False,
    }
    
    if args.lora:
        init_kwargs["lora_path"] = args.lora
        print(f">> Loading LoRA from: {args.lora}")
    elif args.gpt_checkpoint:
        init_kwargs["gpt_checkpoint"] = args.gpt_checkpoint
        print(f">> Loading custom GPT checkpoint from: {args.gpt_checkpoint}")
    
    tts = IndexTTS2(**init_kwargs)
    
    # Load speaker embeddings
    print(f"\n>> Loading speaker embeddings from: {args.embeddings}")
    store = SpeakerEmbeddingStore(tts)
    speaker_embeddings = store.load_embeddings(args.embeddings)
    
    # Print embedding shapes
    print("\n>> Loaded embeddings:")
    for key, value in speaker_embeddings.items():
        print(f"   {key}: shape={value.shape}")
    
    # Run inference
    print(f"\n>> Synthesizing text: '{args.text}'")
    print(f">> Output will be saved to: {args.output}")
    
    output = tts.infer(
        spk_audio_prompt=None,  # Not needed when using speaker_embeddings
        text=args.text,
        output_path=args.output,
        speaker_embeddings=speaker_embeddings,  # Use stored embeddings
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! Audio saved to: {args.output}")
    print("=" * 60)


def compare_outputs(args):
    """Compare base model vs finetuned model outputs."""
    print("=" * 60)
    print("Comparing Base Model vs Finetuned Model")
    print("=" * 60)
    
    # Initialize base model
    print("\n>> Initializing base model...")
    base_tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=False,
    )
    
    # Initialize finetuned model
    print("\n>> Initializing finetuned model...")
    ft_init_kwargs = {
        "cfg_path": args.config,
        "model_dir": args.model_dir,
        "use_cuda_kernel": False,
    }
    if args.lora:
        ft_init_kwargs["lora_path"] = args.lora
    elif args.gpt_checkpoint:
        ft_init_kwargs["gpt_checkpoint"] = args.gpt_checkpoint
    
    ft_tts = IndexTTS2(**ft_init_kwargs)
    
    # Load speaker embeddings
    store = SpeakerEmbeddingStore(ft_tts)
    speaker_embeddings = store.load_embeddings(args.embeddings)
    
    # Generate with base model (using reference audio - OLD WAY)
    print("\n>> Generating with BASE model + reference audio...")
    base_output = args.output.replace(".wav", "_base_with_ref.wav")
    base_tts.infer(
        spk_audio_prompt=args.audio,
        text=args.text,
        output_path=base_output,
    )
    print(f"   Saved to: {base_output}")
    
    # Generate with finetuned model (using reference audio - WRONG WAY)
    print("\n>> Generating with FINETUNED model + reference audio (WRONG WAY)...")
    wrong_output = args.output.replace(".wav", "_finetuned_with_ref_WRONG.wav")
    ft_tts.infer(
        spk_audio_prompt=args.audio,
        text=args.text,
        output_path=wrong_output,
    )
    print(f"   Saved to: {wrong_output}")
    print("   ^ This will sound similar to base model - reference audio dominates!")
    
    # Generate with finetuned model (using stored embeddings - CORRECT WAY)
    print("\n>> Generating with FINETUNED model + stored embeddings (CORRECT WAY)...")
    correct_output = args.output.replace(".wav", "_finetuned_with_embeddings_CORRECT.wav")
    ft_tts.infer(
        spk_audio_prompt=None,
        text=args.text,
        output_path=correct_output,
        speaker_embeddings=speaker_embeddings,
    )
    print(f"   Saved to: {correct_output}")
    print("   ^ This should sound like the trained voice!")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print("\nFiles generated:")
    print(f"  1. {base_output}")
    print(f"     - Base model with reference audio")
    print(f"  2. {wrong_output}")
    print(f"     - Finetuned model with reference audio (WRONG - will sound similar to #1)")
    print(f"  3. {correct_output}")
    print(f"     - Finetuned model with stored embeddings (CORRECT - trained voice)")
    print("\nListen to all three files and compare!")


def extract_from_training_dataset(args):
    """Extract averaged embeddings from multiple training audio files."""
    print("=" * 60)
    print("Extracting Averaged Embeddings from Training Dataset")
    print("=" * 60)
    
    # Find all audio files in the training directory
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    
    for root, dirs, files in os.walk(args.training_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"ERROR: No audio files found in {args.training_dir}")
        sys.exit(1)
    
    print(f"\n>> Found {len(audio_files)} audio files")
    
    # Limit files if too many
    max_files = args.max_files or 50
    if len(audio_files) > max_files:
        print(f">> Using first {max_files} files (use --max-files to change)")
        audio_files = audio_files[:max_files]
    
    # Initialize TTS model
    print("\n>> Initializing IndexTTS2 model...")
    tts = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=False,
    )
    
    # Create embedding store and extract averaged embeddings
    store = SpeakerEmbeddingStore(tts)
    
    print("\n>> Extracting and averaging embeddings from all files...")
    embeddings = store.extract_averaged_embeddings(audio_files, verbose=True)
    
    # Print embedding shapes
    print("\n>> Averaged embeddings:")
    for key, value in embeddings.items():
        print(f"   {key}: shape={value.shape}")
    
    # Save embeddings
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    store.save_embeddings(embeddings, args.output)
    print(f"\n>> Embeddings saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Averaged embeddings extracted from training data.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 Finetuned Model Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--extract", action="store_true",
                          help="Extract speaker embeddings from audio file")
    mode_group.add_argument("--extract-dataset", action="store_true",
                          help="Extract averaged embeddings from training dataset")
    mode_group.add_argument("--infer", action="store_true",
                          help="Run inference with stored embeddings")
    mode_group.add_argument("--compare", action="store_true",
                          help="Compare base vs finetuned model outputs")
    
    # Common arguments
    parser.add_argument("--config", default="checkpoints/config.yaml",
                       help="Path to config file")
    parser.add_argument("--model-dir", default="checkpoints",
                       help="Path to model directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Output path (embeddings.pt for extract, audio.wav for infer)")
    
    # Extract arguments
    parser.add_argument("--audio", "-a",
                       help="Path to audio file for embedding extraction")
    parser.add_argument("--training-dir",
                       help="Path to training data directory (for --extract-dataset)")
    parser.add_argument("--max-files", type=int,
                       help="Maximum number of files to use for averaging (default: 50)")
    
    # Inference arguments
    parser.add_argument("--embeddings", "-e",
                       help="Path to stored speaker embeddings")
    parser.add_argument("--lora",
                       help="Path to LoRA checkpoint directory")
    parser.add_argument("--gpt-checkpoint",
                       help="Path to full finetuned GPT checkpoint")
    parser.add_argument("--text", "-t",
                       help="Text to synthesize")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.extract:
        if not args.audio:
            parser.error("--extract requires --audio")
        extract_embeddings(args)
    
    elif args.extract_dataset:
        if not args.training_dir:
            parser.error("--extract-dataset requires --training-dir")
        extract_from_training_dataset(args)
    
    elif args.infer:
        if not args.embeddings:
            parser.error("--infer requires --embeddings")
        if not args.text:
            parser.error("--infer requires --text")
        if not args.lora and not args.gpt_checkpoint:
            print("WARNING: No --lora or --gpt-checkpoint specified, using base model")
        run_inference(args)
    
    elif args.compare:
        if not args.embeddings:
            parser.error("--compare requires --embeddings")
        if not args.audio:
            parser.error("--compare requires --audio")
        if not args.text:
            parser.error("--compare requires --text")
        if not args.lora and not args.gpt_checkpoint:
            parser.error("--compare requires --lora or --gpt-checkpoint")
        compare_outputs(args)


if __name__ == "__main__":
    main()