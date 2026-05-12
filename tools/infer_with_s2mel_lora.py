#!/usr/bin/env python3
"""
Inference with S2Mel LoRA for IndexTTS2

Generate speech using a trained S2Mel LoRA model that has learned
prosodic patterns like stutters, pauses, and speech rhythm.

WHY S2Mel LoRA Works for Stutters:
==================================
Unlike GPT LoRA (which affects semantic code generation), S2Mel LoRA
directly affects the mel spectrogram generation - where prosodic 
patterns actually live!

The S2Mel model has learned:
1. The acoustic patterns of your speaker's stutters
2. The timing and rhythm characteristics  
3. The prosodic features that make speech sound natural/stuttered

Usage:
======
    # Basic inference
    python tools/infer_with_s2mel_lora.py \\
        --speaker ozzy \\
        --text "Hello world" \\
        --output output.wav
    
    # With reference audio
    python tools/infer_with_s2mel_lora.py \\
        --speaker ozzy \\
        --reference examples/voice_01.wav \\
        --text "Testing the trained model"
    
    # Compare base vs S2Mel LoRA
    python tools/infer_with_s2mel_lora.py \\
        --speaker ozzy \\
        --text "I was going to the store" \\
        --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with S2Mel LoRA for prosodic patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", type=Path, default=Path("output.wav"),
                        help="Output audio path")
    
    # LoRA paths
    parser.add_argument("--s2mel-lora", type=Path,
                        help="Custom S2Mel LoRA directory")
    parser.add_argument("--gpt-lora", type=Path,
                        help="Optional GPT LoRA directory (for combined training)")
    parser.add_argument("--use-best", action="store_true", default=True,
                        help="Use best checkpoint (default)")
    parser.add_argument("--use-final", action="store_true",
                        help="Use final checkpoint instead of best")
    
    # Reference audio
    parser.add_argument("--reference", "-r", type=Path,
                        help="Reference audio for speaker conditioning")
    parser.add_argument("--speaker-embeddings", type=Path,
                        help="Pre-computed speaker embeddings")
    
    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Generate both base and LoRA versions for comparison")
    
    # Generation parameters
    parser.add_argument("--diffusion-steps", type=int, default=25,
                        help="Diffusion steps (default: 25)")
    parser.add_argument("--cfg-rate", type=float, default=0.7,
                        help="CFG rate for inference (default: 0.7)")
    
    # Model paths
    parser.add_argument("--config", type=Path, 
                        default=PROJECT_ROOT / "checkpoints" / "config.yaml")
    parser.add_argument("--model-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def find_s2mel_lora(speaker: str, use_best: bool = True) -> Optional[Path]:
    """Find S2Mel LoRA checkpoint for speaker."""
    base_dir = PROJECT_ROOT / "training" / speaker / "s2mel_training"
    
    if use_best:
        lora_dir = base_dir / "best_checkpoint"
    else:
        lora_dir = base_dir / "final_checkpoint"
    
    if lora_dir.exists():
        return lora_dir
    return None


def find_reference_audio(speaker: str) -> Optional[Path]:
    """Find reference audio for speaker."""
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


class S2MelLoRAInference:
    """
    Inference class with S2Mel LoRA support.
    
    This class loads the base IndexTTS2 model and applies S2Mel LoRA
    adapters for prosodic pattern generation.
    """
    
    def __init__(
        self,
        config_path: Path = PROJECT_ROOT / "checkpoints" / "config.yaml",
        model_dir: Path = PROJECT_ROOT / "checkpoints",
        s2mel_lora_path: Optional[Path] = None,
        gpt_lora_path: Optional[Path] = None,
        device: Optional[str] = None,
        diffusion_steps: int = 25,
        cfg_rate: float = 0.7,
    ):
        """
        Initialize inference with S2Mel LoRA.
        
        Args:
            config_path: Path to config.yaml
            model_dir: Path to model checkpoints
            s2mel_lora_path: Path to S2Mel LoRA checkpoint
            gpt_lora_path: Optional path to GPT LoRA checkpoint
            device: Device to use
            diffusion_steps: Number of diffusion steps
            cfg_rate: Classifier-free guidance rate
        """
        self.config_path = config_path
        self.model_dir = model_dir
        self.s2mel_lora_path = s2mel_lora_path
        self.gpt_lora_path = gpt_lora_path
        self.diffusion_steps = diffusion_steps
        self.cfg_rate = cfg_rate
        
        # Device setup
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        
        self._load_model()
        
        if s2mel_lora_path:
            self._load_s2mel_lora()
    
    def _load_model(self):
        """Load base IndexTTS2 model."""
        from indextts.infer_v2 import IndexTTS2
        
        print(f"Loading IndexTTS2 base model...")
        
        self.tts = IndexTTS2(
            cfg_path=str(self.config_path),
            model_dir=str(self.model_dir),
            lora_path=str(self.gpt_lora_path) if self.gpt_lora_path else None,
            device=self.device,
        )
        
        print("  Base model loaded!")
    
    def _load_s2mel_lora(self):
        """Load S2Mel LoRA adapters."""
        from indextts.utils.s2mel_lora_utils import load_s2mel_lora_checkpoint
        
        print(f"Loading S2Mel LoRA from: {self.s2mel_lora_path}")
        
        self.tts.s2mel = load_s2mel_lora_checkpoint(
            self.tts.s2mel,
            self.s2mel_lora_path,
            merge_weights=True,  # Merge for faster inference
            device=self.device,
        )
        
        # Load reference features if available
        ref_features_path = self.s2mel_lora_path / "reference_features.pt"
        if ref_features_path.exists():
            self.reference_features = torch.load(ref_features_path, map_location=self.device)
            print(f"  Loaded reference features from checkpoint")
        else:
            self.reference_features = None
        
        print("  S2Mel LoRA loaded and merged!")
    
    def generate(
        self,
        text: str,
        output_path: str,
        reference_audio: Optional[str] = None,
        speaker_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> str:
        """
        Generate speech with S2Mel LoRA patterns.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio
            reference_audio: Path to reference audio for conditioning
            speaker_embeddings: Pre-computed speaker embeddings
            **kwargs: Additional generation parameters
        
        Returns:
            Path to generated audio
        """
        # Prepare generation kwargs
        gen_kwargs = {
            "text": text,
            "output_path": output_path,
            **kwargs,
        }
        
        # Speaker conditioning
        if speaker_embeddings is not None:
            gen_kwargs["speaker_embeddings"] = speaker_embeddings
        elif reference_audio is not None:
            gen_kwargs["spk_audio_prompt"] = reference_audio
        elif self.reference_features is not None:
            # Use stored reference features (best for matching training)
            print("  Using stored reference features from training")
            # Note: We'd need to integrate these with the inference
            # For now, still need reference audio
            raise ValueError("Reference audio required - stored features not yet integrated")
        else:
            raise ValueError("Either reference_audio or speaker_embeddings required")
        
        # Generate
        self.tts.infer(**gen_kwargs)
        
        return output_path


def main():
    args = parse_args()
    
    # Find S2Mel LoRA
    if args.s2mel_lora:
        s2mel_lora_path = args.s2mel_lora
    else:
        s2mel_lora_path = find_s2mel_lora(args.speaker, use_best=not args.use_final)
    
    if not s2mel_lora_path or not s2mel_lora_path.exists():
        print(f"S2Mel LoRA not found for speaker: {args.speaker}")
        print("\nTrain S2Mel LoRA first:")
        print(f"  python tools/prepare_s2mel_dataset.py --speaker {args.speaker}")
        print(f"  python tools/train_s2mel_lora.py --speaker {args.speaker}")
        sys.exit(1)
    
    # Find reference audio
    reference = args.reference
    if reference is None:
        reference = find_reference_audio(args.speaker)
    
    # ----------------------------------------------------------------------
    # Updated logic: Prefer stored speaker embeddings whenever they are
    # available, regardless of whether a reference audio file is also
    # provided.  This ensures that GPT‑aligned training (which relies on
    # the stored speaker embeddings) is used correctly and avoids the
    # “drunk/mumbling” output caused by mismatched reference conditioning.
    # ----------------------------------------------------------------------
    speaker_embeddings_path = (
        Path(__file__).parent.parent
        / "training"
        / args.speaker
        / "embeddings"
        / "speaker_embeddings.pt"
    )
    
    use_speaker_embeddings = False
    # If stored speaker embeddings exist, use them regardless of reference audio.
    if speaker_embeddings_path.exists():
        use_speaker_embeddings = True
        print(f"  Using stored speaker embeddings from {speaker_embeddings_path}")
    
    if not use_speaker_embeddings and (reference is None or not reference.exists()):
        print("Reference audio not found!")
        print("\nProvide reference audio with --reference path/to/audio.wav")
        sys.exit(1)
    
    print("=" * 60)
    print("S2MEL LORA INFERENCE")
    print("=" * 60)
    print(f"""
Speaker: {args.speaker}
S2Mel LoRA: {s2mel_lora_path}
Reference: {reference}
Text: {args.text}
Output: {args.output}
""")
    
    if args.compare:
        # Generate both base and LoRA versions
        print("\n[COMPARISON MODE]")
        print("Generating both base and S2Mel LoRA versions...")
        
        # Base model
        print("\n[1/2] Generating with BASE model...")
        base_output = args.output.with_stem(args.output.stem + "_base")
        
        base_infer = S2MelLoRAInference(
            config_path=args.config,
            model_dir=args.model_dir,
            s2mel_lora_path=None,  # No LoRA
            device=args.device,
            diffusion_steps=args.diffusion_steps,
            cfg_rate=args.cfg_rate,
        )
        base_infer.generate(
            text=args.text,
            output_path=str(base_output),
            # Base model still requires reference audio for conditioning.
            # If the user prefers prompt‑less inference, they can run the
            # non‑compare mode which will automatically fall back to stored
            # speaker embeddings (see logic above).
            reference_audio=str(reference),
        )
        print(f"  Saved: {base_output}")
        
        # S2Mel LoRA
        print("\n[2/2] Generating with S2MEL LORA...")
        lora_output = args.output.with_stem(args.output.stem + "_s2mel_lora")
        
        lora_infer = S2MelLoRAInference(
            config_path=args.config,
            model_dir=args.model_dir,
            s2mel_lora_path=s2mel_lora_path,
            gpt_lora_path=args.gpt_lora,
            device=args.device,
            diffusion_steps=args.diffusion_steps,
            cfg_rate=args.cfg_rate,
        )
        # When generating with LoRA we can use either the reference audio
        # or the pre‑computed speaker embeddings.  The generate() method
        # will automatically pick the appropriate source based on the
        # arguments we pass.
        if use_speaker_embeddings:
            lora_infer.generate(
                text=args.text,
                output_path=str(lora_output),
                speaker_embeddings=torch.load(speaker_embeddings_path, map_location=lora_infer.device),
            )
        else:
            lora_infer.generate(
                text=args.text,
                output_path=str(lora_output),
                reference_audio=str(reference),
            )
        print(f"  Saved: {lora_output}")
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)
        print(f"""
Base model: {base_output}
S2Mel LoRA: {lora_output}

Listen to both and compare the prosodic patterns!
The S2Mel LoRA version should have learned your speaker's
unique speech rhythm, pauses, and stutter patterns.
""")
    
    else:
        # Single generation with LoRA
        print("\nInitializing S2Mel LoRA inference...")
        
        infer = S2MelLoRAInference(
            config_path=args.config,
            model_dir=args.model_dir,
            s2mel_lora_path=s2mel_lora_path,
            gpt_lora_path=args.gpt_lora,
            device=args.device,
            diffusion_steps=args.diffusion_steps,
            cfg_rate=args.cfg_rate,
        )
        
        print("\nGenerating speech with S2Mel LoRA...")
        # Single‑generation mode – use reference audio if available,
        # otherwise fall back to stored speaker embeddings.
        if use_speaker_embeddings:
            infer.generate(
                text=args.text,
                output_path=str(args.output),
                speaker_embeddings=torch.load(speaker_embeddings_path, map_location=infer.device),
            )
        else:
            infer.generate(
                text=args.text,
                output_path=str(args.output),
                reference_audio=str(reference),
            )
        
        print(f"\n Generated: {args.output}")
        
        print("""
PROSODIC PATTERNS:
==================
The S2Mel LoRA has learned your speaker's acoustic patterns.
The generated audio should reflect:
- Learned speech rhythm and timing
- Prosodic characteristics from training data
- If trained on stuttered speech: stutter-like patterns

To generate stuttered output, the TEXT should still contain
stutter indicators (e.g., "I I I was going going to").
The S2Mel LoRA ensures these are realized acoustically!
""")


if __name__ == "__main__":
    main()