#!/usr/bin/env python3
"""
Inference with Verbatim-Trained LoRA for IndexTTS2

After training with verbatim transcriptions, use this script to generate
speech with stutters and imperfections by providing verbatim-style text.

Usage:
    # Basic usage
    python tools/infer_verbatim.py \
        --speaker ozzy \
        --text "I I I was going going to the store..."
    
    # With reference audio
    python tools/infer_verbatim.py \
        --speaker ozzy \
        --reference path/to/reference.wav \
        --text "I I I was going going to the store..." \
        --output output.wav
    
    # Compare clean vs stuttered
    python tools/infer_verbatim.py \
        --speaker ozzy \
        --text "I was going to the store" \
        --verbatim-text "I I I was going going to the store..." \
        --compare
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech with verbatim-trained LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize (can include stutters)")
    parser.add_argument("--reference", "-r", type=Path, help="Reference audio for voice cloning")
    parser.add_argument("--output", "-o", type=Path, default=Path("output.wav"), help="Output path")
    
    # Comparison mode
    parser.add_argument("--verbatim-text", help="Optional: verbatim version for comparison")
    parser.add_argument("--compare", action="store_true", 
                        help="Generate both clean and stuttered versions")
    
    # LoRA options
    parser.add_argument("--lora-dir", type=Path, help="Custom LoRA directory")
    parser.add_argument("--use-best", action="store_true", default=True,
                        help="Use best checkpoint (default)")
    parser.add_argument("--use-final", action="store_true",
                        help="Use final checkpoint instead of best")
    
    # Model options
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def find_reference_audio(speaker: str) -> Path:
    """Find a reference audio file for the speaker."""
    speaker_dir = PROJECT_ROOT / "training" / speaker
    
    # Try different locations
    candidates = [
        speaker_dir / "dataset" / "audio",
        speaker_dir / "reference",
        speaker_dir,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            audio_files = list(candidate.glob("*.wav")) + list(candidate.glob("*.mp3"))
            if audio_files:
                return audio_files[0]
    
    return None


def main():
    args = parse_args()
    
    # Find LoRA checkpoint
    if args.lora_dir:
        lora_path = args.lora_dir
    else:
        base_dir = PROJECT_ROOT / "training" / args.speaker / "verbatim_training"
        if args.use_final:
            lora_path = base_dir / "final_checkpoint"
        else:
            lora_path = base_dir / "best_checkpoint"
    
    if not lora_path.exists():
        print(f"❌ LoRA checkpoint not found: {lora_path}")
        print("\nTrain first with:")
        print(f"  python tools/train_verbatim_lora.py --speaker {args.speaker}")
        sys.exit(1)
    
    # Find reference audio
    reference = args.reference
    if reference is None:
        reference = find_reference_audio(args.speaker)
        if reference is None:
            print("❌ No reference audio found. Provide --reference path/to/audio.wav")
            sys.exit(1)
        print(f"Using reference: {reference}")
    
    # Import and load model
    print("\nLoading IndexTTS2 with LoRA...")
    from indextts.infer_v2 import IndexTTS2
    
    tts = IndexTTS2(
        cfg_path=str(args.config),
        model_dir=str(args.model_dir),
        lora_path=str(lora_path),
        device=args.device,
    )
    
    print(f"✓ LoRA loaded from: {lora_path}")
    
    # Generate
    if args.compare and args.verbatim_text:
        # Generate both clean and stuttered versions
        clean_output = args.output.with_stem(args.output.stem + "_clean")
        stutter_output = args.output.with_stem(args.output.stem + "_stuttered")
        
        print(f"\n[1/2] Generating CLEAN version...")
        print(f"  Text: {args.text}")
        tts.infer(
            spk_audio_prompt=str(reference),
            text=args.text,
            output_path=str(clean_output),
        )
        print(f"  ✓ Saved: {clean_output}")
        
        print(f"\n[2/2] Generating STUTTERED version...")
        print(f"  Text: {args.verbatim_text}")
        tts.infer(
            spk_audio_prompt=str(reference),
            text=args.verbatim_text,
            output_path=str(stutter_output),
        )
        print(f"  ✓ Saved: {stutter_output}")
        
        print("\n" + "=" * 50)
        print("COMPARISON COMPLETE")
        print("=" * 50)
        print(f"Clean:     {clean_output}")
        print(f"Stuttered: {stutter_output}")
        
    else:
        # Single generation
        print(f"\nGenerating speech...")
        print(f"  Text: {args.text}")
        
        tts.infer(
            spk_audio_prompt=str(reference),
            text=args.text,
            output_path=str(args.output),
        )
        
        print(f"\n✓ Generated: {args.output}")
    
    print("\n" + "=" * 50)
    print("TIPS FOR STUTTERED OUTPUT")
    print("=" * 50)
    print("""
To generate stuttered speech, include stutters in your text:

  Word repetitions: "I I I was", "going going"
  Fillers: "um", "uh", "er"  
  Hesitations: "I was... going to..."
  
Examples:
  - "I I I think um that we should go going now"
  - "What do you uh what do you mean..."
  - "So... I was um going to..."
""")


if __name__ == "__main__":
    main()