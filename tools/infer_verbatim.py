#!/usr/bin/env python3
"""
Inference with Verbatim-Trained LoRA for IndexTTS2

After training with verbatim transcriptions, use this script to generate
speech with stutters and imperfections by providing verbatim-style text.

SPEAKER EMBEDDINGS (KEY FEATURE):
=================================
Training extracts speaker embeddings from audio samples and uses them for
ALL training samples. At inference, we use the SAME embeddings to ensure
the LoRA adaptations are not overshadowed by different conditioning.

This prevents the "overshadowing" issue where different conditioning at
inference time would mask what the LoRA learned during training.

Usage:
    # Promptless inference (RECOMMENDED - no reference audio needed!)
    python tools/infer_verbatim.py \\
        --speaker ozzy \\
        --text "I I I was going going to the store..."
    
    # With reference audio (if you want to use a different voice)
    python tools/infer_verbatim.py \\
        --speaker ozzy \\
        --reference path/to/reference.wav \\
        --text "I I I was going going to the store..."
    
    # Compare clean vs stuttered
    python tools/infer_verbatim.py \\
        --speaker ozzy \\
        --text "I was going to the store" \\
        --verbatim-text "I I I was going going to the store..." \\
        --compare
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech with verbatim-trained LoRA (promptless!)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize (can include stutters)")
    parser.add_argument("--output", "-o", type=Path, default=Path("output.wav"), help="Output path")
    
    # Speaker embeddings (for promptless inference)
    parser.add_argument("--embeddings", "-e", type=Path,
                        help="Speaker embeddings file (auto-detected from training)")
    parser.add_argument("--reference", "-r", type=Path,
                        help="Reference audio (optional - uses embeddings by default)")
    parser.add_argument("--force-reference", action="store_true",
                        help="Force using reference audio instead of stored embeddings")
    
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


def find_speaker_embeddings(speaker: str) -> Optional[Path]:
    """Find speaker embeddings file from training output."""
    base_dir = PROJECT_ROOT / "training" / speaker / "verbatim_training"
    
    # Try different locations
    candidates = [
        base_dir / "best_checkpoint" / "speaker_embeddings.pt",
        base_dir / "speaker_embeddings.pt",
        base_dir / "final_checkpoint" / "speaker_embeddings.pt",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None




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
            lora_path = base_dir / "final_checkpoint/lora"
        else:
            lora_path = base_dir / "best_checkpoint/lora"
    
    if not lora_path.exists():
        print(f"❌ LoRA checkpoint not found: {lora_path}")
        print("\nTrain first with:")
        print(f"  python tools/train_verbatim_lora.py --speaker {args.speaker}")
        sys.exit(1)
    
    # ============================================================
    # SPEAKER EMBEDDINGS (CRITICAL FOR CONSISTENCY!)
    # ============================================================
    # Training extracts speaker embeddings and uses them for ALL samples.
    # Using the SAME embeddings at inference ensures the LoRA adaptations
    # aren't overshadowed by different conditioning.
    # ============================================================
    
    speaker_embeddings = None
    embeddings_path = args.embeddings
    
    if not args.force_reference:
        # Try to find and load speaker embeddings
        if embeddings_path is None:
            embeddings_path = find_speaker_embeddings(args.speaker)
        
        if embeddings_path and embeddings_path.exists():
            print(f"\n✓ Found speaker embeddings: {embeddings_path}")
            print("  (Using SAME embeddings that were used during training!)")
            print("  (PROMPTLESS inference - no reference audio needed!)")
            
            from indextts.speaker_embeddings import SpeakerEmbeddingStore
            
            # Load embeddings
            store = SpeakerEmbeddingStore()
            speaker_embeddings = store.load_embeddings(embeddings_path)
            
            # Show what's in the embeddings
            print(f"\n  Loaded embeddings:")
            for key, value in speaker_embeddings.items():
                print(f"    {key}: {value.shape}")
        else:
            print("\n⚠ No speaker embeddings found.")
            print("  Will use reference audio (if available)")
            print("  Note: This may not reproduce trained patterns as accurately!")
    
    # Only need reference audio if no embeddings
    reference = None
    if speaker_embeddings is None:
        reference = args.reference
        if reference is None:
            reference = find_reference_audio(args.speaker)
            if reference is None:
                print("❌ No reference audio found and no speaker embeddings available.")
                print("\nOptions:")
                print("  1. Provide reference audio: --reference path/to/audio.wav")
                print("  2. Re-train to generate embeddings:")
                print(f"     python tools/train_verbatim_lora.py --speaker {args.speaker}")
                sys.exit(1)
        print(f"Using reference audio: {reference}")
        print("  ⚠ Note: Reference audio conditioning may differ from training!")
    
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
    
    # Prepare inference kwargs
    infer_kwargs = {
        "text": args.text,
        "output_path": str(args.output),
        "repetition_penalty": 1.0,
        "temperature": 1.0
    }
    
    if speaker_embeddings is not None:
        # Promptless inference using stored embeddings
        # These are the SAME embeddings used during training!
        infer_kwargs["speaker_embeddings"] = speaker_embeddings
        infer_kwargs["spk_audio_prompt"] = None
    else:
        # Use reference audio (may not match training conditioning)
        infer_kwargs["spk_audio_prompt"] = str(reference)
    
    # Generate
    if args.compare and args.verbatim_text:
        # Generate both clean and stuttered versions
        clean_output = args.output.with_stem(args.output.stem + "_clean")
        stutter_output = args.output.with_stem(args.output.stem + "_stuttered")
        
        print(f"\n[1/2] Generating CLEAN version...")
        print(f"  Text: {args.text}")
        clean_kwargs = infer_kwargs.copy()
        clean_kwargs["text"] = args.text
        clean_kwargs["output_path"] = str(clean_output)
        tts.infer(**clean_kwargs)
        print(f"  ✓ Saved: {clean_output}")
        
        print(f"\n[2/2] Generating STUTTERED version...")
        print(f"  Text: {args.verbatim_text}")
        stutter_kwargs = infer_kwargs.copy()
        stutter_kwargs["text"] = args.verbatim_text
        stutter_kwargs["output_path"] = str(stutter_output)
        tts.infer(**stutter_kwargs)
        print(f"  ✓ Saved: {stutter_output}")
        
        print("\n" + "=" * 50)
        print("COMPARISON COMPLETE")
        print("=" * 50)
        print(f"Clean:     {clean_output}")
        print(f"Stuttered: {stutter_output}")
        
    else:
        # Single generation
        mode = "PROMPTLESS" if speaker_embeddings else "with reference"
        print(f"\nGenerating speech ({mode})...")
        print(f"  Text: {args.text}")
        
        tts.infer(**infer_kwargs)
        
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

PROMPTLESS MODE:
================
Your model was trained with stored speaker embeddings.
No reference audio is needed - the voice is baked in!

To use a different voice, pass --force-reference --reference audio.wav
""")


if __name__ == "__main__":
    main()