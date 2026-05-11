#!/usr/bin/env python3
"""
Prepare Dataset for S2Mel LoRA Training

This script prepares data for training the S2Mel (Semantic-to-Mel) model,
which is the KEY component for learning prosodic patterns like stutters,
pauses, and speech rhythm.

What S2Mel Training Needs:
==========================
1. Target mel spectrogram (x1) - Ground truth mel from audio
2. Semantic codes (S_infer) - From semantic codec (quantized W2V-BERT features)
3. Style vector (style) - Global speaker style from CAMPPlus (192-dim)
4. Prompt condition (prompt_condition) - Reference mel features for conditioning
5. Reference mel (ref_mel) - Reference speaker mel spectrogram

Why This Works for Stutters:
============================
The S2Mel model learns to generate mel spectrograms from semantic codes.
When you train on verbatim audio (with stutters), the model learns:
- The ACTUAL mel patterns of stuttered speech
- The prosodic characteristics (timing, pauses, repetitions)
- These patterns become baked into the mel generation!

Usage:
    python tools/prepare_s2mel_dataset.py --speaker ozzy
    
    # With custom reference for style extraction
    python tools/prepare_s2mel_dataset.py --speaker ozzy --reference path/to/reference.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass 
class S2MelSample:
    """Represents a single S2Mel training sample."""
    id: str
    audio_path: str
    text: str
    duration: float
    has_stutters: bool = False
    stutter_count: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for S2Mel LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", required=True, help="Speaker name")
    parser.add_argument("--audio-dir", type=Path, help="Custom audio directory")
    parser.add_argument("--transcripts", type=Path, help="Transcripts CSV file")
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    
    # Reference audio for style extraction
    parser.add_argument("--reference", type=Path, 
                        help="Reference audio file for global style extraction")
    parser.add_argument("--reference-duration", type=float, default=15.0,
                        help="Max duration of reference audio (seconds)")
    
    # Processing options
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--use-verbatim", action="store_true", default=True,
                        help="Use verbatim transcriptions (for stutter detection)")
    
    # Model paths
    parser.add_argument("--config", type=Path, 
                        default=PROJECT_ROOT / "checkpoints" / "config.yaml")
    parser.add_argument("--model-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


def detect_stutters(text: str) -> Tuple[bool, int]:
    """
    Detect stutters and repetitions in verbatim text.
    
    Returns:
        (has_stutters, stutter_count)
    """
    import re
    
    words = text.lower().split()
    stutter_count = 0
    
    # Detect word repetitions (e.g., "I I I")
    for i in range(len(words) - 1):
        if words[i] == words[i+1]:
            stutter_count += 1
    
    # Detect filler words
    fillers = ["um", "uh", "er", "ah", "like", "you know"]
    for filler in fillers:
        stutter_count += text.lower().count(filler)
    
    # Detect hesitation patterns (...)
    stutter_count += text.count("...")
    stutter_count += text.count("…")
    
    has_stutters = stutter_count > 0
    
    return has_stutters, stutter_count


def load_transcripts(csv_path: Path, audio_dir: Path) -> List[S2MelSample]:
    """Load transcripts from CSV file."""
    import csv
    
    samples = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try different column names for audio path
            audio_file = row.get("audio_path") or row.get("file") or row.get("filename")
            if not audio_file:
                continue
            
            # Get text (prefer verbatim if available)
            text = row.get("parakeet") or row.get("fast_whisper") or row.get("transcript") or ""
            if not text:
                continue
            
            # Resolve audio path
            audio_path = Path(audio_file)
            if not audio_path.is_absolute():
                audio_path = audio_dir / audio_file
            
            if not audio_path.exists():
                # Try adding extension
                for ext in [".wav", ".mp3", ".flac"]:
                    test_path = audio_path.with_suffix(ext)
                    if test_path.exists():
                        audio_path = test_path
                        break
            
            if not audio_path.exists():
                continue
            
            # Get duration
            try:
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
            except:
                duration = 0.0
            
            # Detect stutters
            has_stutters, stutter_count = detect_stutters(text)
            
            sample = S2MelSample(
                id=audio_path.stem,
                audio_path=str(audio_path),
                text=text,
                duration=duration,
                has_stutters=has_stutters,
                stutter_count=stutter_count,
            )
            samples.append(sample)
    
    return samples


class S2MelFeatureExtractor:
    """
    Extract all features needed for S2Mel training.
    
    Extracts:
    - mel spectrogram (target)
    - semantic codes (from W2V-BERT + semantic codec)
    - style vector (from CAMPPlus)
    - prompt condition (from length regulator)
    """
    
    def __init__(self, config_path: Path, model_dir: Path, device: str = None):
        from omegaconf import OmegaConf
        
        self.config_path = config_path
        self.model_dir = model_dir
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.cfg = OmegaConf.load(config_path)
        self._load_models()
    
    def _load_models(self):
        """Load all required models for feature extraction."""
        import os
        os.environ['HF_HUB_CACHE'] = str(self.model_dir / 'hf_cache')
        
        from transformers import SeamlessM4TFeatureExtractor
        from huggingface_hub import hf_hub_download
        import safetensors
        
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.audio import mel_spectrogram
        
        print("Loading feature extraction models...")
        
        # W2V-BERT feature extractor
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        
        # Semantic model (W2V-BERT)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            str(self.model_dir / self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)
        print("  ✓ W2V-BERT loaded")
        
        # Semantic codec
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print("  ✓ Semantic codec loaded")
        
        # S2Mel model (for length regulator)
        s2mel_path = str(self.model_dir / self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel, None, s2mel_path,
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.eval()
        print("  ✓ S2Mel model loaded")
        
        # CAMPPlus for speaker style
        campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print("  ✓ CAMPPlus loaded")
        
        # Mel spectrogram function
        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        
        print("All models loaded!")
    
    @torch.no_grad()
    def get_semantic_embedding(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract semantic embedding from 16kHz audio."""
        inputs = self.extract_features(audio_16k.squeeze(0).numpy(), 
                                        sampling_rate=16000, 
                                        return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, 1024)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat
    
    @torch.no_grad()
    def get_semantic_codes(self, semantic_emb: torch.Tensor) -> torch.Tensor:
        """Quantize semantic embedding to codes."""
        _, codes = self.semantic_codec.quantize(semantic_emb)
        return codes  # (B, T)
    
    @torch.no_grad()
    def get_mel_spectrogram(self, audio_22k: torch.Tensor) -> torch.Tensor:
        """Get mel spectrogram from 22kHz audio."""
        mel = self.mel_fn(audio_22k.to(self.device))
        return mel  # (B, 80, T)
    
    @torch.no_grad()
    def get_style_vector(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Get global style vector from CAMPPlus."""
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))
        return style  # (1, 192)
    
    @torch.no_grad()
    def get_prompt_condition(self, semantic_codes: torch.Tensor, 
                             mel_length: int) -> torch.Tensor:
        """Get prompt condition from length regulator."""
        target_lengths = torch.LongTensor([mel_length]).to(self.device)
        
        condition = self.s2mel.models['length_regulator'](
            semantic_codes,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None
        )[0]
        return condition  # (B, T, 768)
    
    def extract_all_features(
        self, 
        audio_path: str,
        max_duration: float = 15.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features needed for S2Mel training from audio file.
        
        Returns:
            Dict with:
            - mel: Target mel spectrogram (B, 80, T)
            - semantic_codes: Quantized semantic codes (B, T_semantic)
            - style: Global style vector (B, 192)
            - prompt_condition: Length-regulated condition (B, T_mel, 768)
            - semantic_emb: Raw semantic embedding (B, T_semantic, 1024)
        """
        # Load and resample audio
        audio, sr = librosa.load(audio_path)
        audio = torch.tensor(audio).unsqueeze(0)
        
        # Truncate to max duration
        max_samples = int(max_duration * sr)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        # Resample to required sample rates
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        
        # Extract features
        semantic_emb = self.get_semantic_embedding(audio_16k)  # (1, T, 1024)
        semantic_codes = self.get_semantic_codes(semantic_emb)  # (1, T)
        mel = self.get_mel_spectrogram(audio_22k.to(self.device))  # (1, 80, T_mel)
        style = self.get_style_vector(audio_16k)  # (1, 192)
        
        # Get prompt condition (matches mel length)
        mel_length = mel.size(-1)
        prompt_condition = self.get_prompt_condition(semantic_codes, mel_length)
        
        return {
            "mel": mel.cpu(),
            "semantic_codes": semantic_codes.cpu(),
            "style": style.cpu(),
            "prompt_condition": prompt_condition.cpu(),
            "semantic_emb": semantic_emb.cpu(),
        }


def main():
    args = parse_args()
    
    # Setup paths
    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    
    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        audio_dir = speaker_dir / "dataset" / "audio"
    
    if args.transcripts:
        transcripts_file = args.transcripts
    else:
        # Try different transcript file names
        candidates = [
            speaker_dir / "dataset" / "transcripts_verbatim.csv",
            speaker_dir / "dataset" / "transcripts.csv",
        ]
        transcripts_file = None
        for c in candidates:
            if c.exists():
                transcripts_file = c
                break
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = speaker_dir / "dataset" / "processed_s2mel"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("S2Mel DATASET PREPARATION")
    print("=" * 60)
    print(f"""
Speaker: {args.speaker}
Audio directory: {audio_dir}
Transcripts: {transcripts_file}
Output: {output_dir}
""")
    
    # Check inputs
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    if not transcripts_file or not transcripts_file.exists():
        print(f"❌ Transcripts file not found!")
        print("\nPlease transcribe your audio first:")
        print(f"  python tools/transcribe_dataset.py --speaker {args.speaker}")
        sys.exit(1)
    
    # Load transcripts
    print("[1/4] Loading transcripts...")
    samples = load_transcripts(transcripts_file, audio_dir)
    
    # Filter by duration
    samples = [s for s in samples 
               if args.min_duration <= s.duration <= args.max_duration]
    
    print(f"  Loaded {len(samples)} samples")
    stutter_count = sum(1 for s in samples if s.has_stutters)
    print(f"  Samples with stutters: {stutter_count}/{len(samples)}")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        sys.exit(1)
    
    # Initialize feature extractor
    print("\n[2/4] Initializing feature extractor...")
    extractor = S2MelFeatureExtractor(args.config, args.model_dir)
    
    # Extract reference style (optional separate reference)
    reference_features = None
    if args.reference and args.reference.exists():
        print(f"\n[2.5/4] Extracting reference style from: {args.reference}")
        reference_features = extractor.extract_all_features(
            str(args.reference), 
            max_duration=args.reference_duration
        )
        print(f"  Reference style extracted: {reference_features['style'].shape}")
    
    # Process samples
    print("\n[3/4] Extracting features...")
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    train_manifest = []
    val_manifest = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        try:
            features = extractor.extract_all_features(
                sample.audio_path, 
                max_duration=args.max_duration
            )
            
            # Save features
            sample_dir = features_dir / sample.id
            sample_dir.mkdir(exist_ok=True)
            
            np.save(sample_dir / "mel.npy", features["mel"].numpy())
            np.save(sample_dir / "semantic_codes.npy", features["semantic_codes"].numpy())
            np.save(sample_dir / "style.npy", features["style"].numpy())
            np.save(sample_dir / "prompt_condition.npy", features["prompt_condition"].numpy())
            np.save(sample_dir / "semantic_emb.npy", features["semantic_emb"].numpy())
            
            # Create manifest entry
            entry = {
                "id": sample.id,
                "audio_path": sample.audio_path,
                "text": sample.text,
                "duration": sample.duration,
                "has_stutters": sample.has_stutters,
                "stutter_count": sample.stutter_count,
                "mel_path": str(sample_dir / "mel.npy"),
                "semantic_codes_path": str(sample_dir / "semantic_codes.npy"),
                "style_path": str(sample_dir / "style.npy"),
                "prompt_condition_path": str(sample_dir / "prompt_condition.npy"),
                "semantic_emb_path": str(sample_dir / "semantic_emb.npy"),
                "mel_length": features["mel"].shape[-1],
                "semantic_length": features["semantic_codes"].shape[-1],
            }
            
            # Train/val split
            if i < int(len(samples) * (1 - args.val_split)):
                train_manifest.append(entry)
            else:
                val_manifest.append(entry)
                
        except Exception as e:
            print(f"  Error processing {sample.id}: {e}")
            continue
    
    # Save reference style (for training)
    if reference_features:
        torch.save({
            "style": reference_features["style"],
            "prompt_condition": reference_features["prompt_condition"],
            "mel": reference_features["mel"],
        }, output_dir / "reference_features.pt")
        print(f"\n  Reference features saved to: {output_dir / 'reference_features.pt'}")
    
    # Save manifests
    print("\n[4/4] Saving manifests...")
    
    train_manifest_path = output_dir / "train_manifest.jsonl"
    with open(train_manifest_path, "w") as f:
        for entry in train_manifest:
            f.write(json.dumps(entry) + "\n")
    
    val_manifest_path = output_dir / "val_manifest.jsonl"
    with open(val_manifest_path, "w") as f:
        for entry in val_manifest:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\n✓ Train manifest: {train_manifest_path} ({len(train_manifest)} samples)")
    print(f"✓ Val manifest: {val_manifest_path} ({len(val_manifest)} samples)")
    
    # Summary
    total_stutters = sum(e["stutter_count"] for e in train_manifest + val_manifest)
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
Total samples: {len(train_manifest) + len(val_manifest)}
  Training: {len(train_manifest)}
  Validation: {len(val_manifest)}

Samples with stutters: {sum(1 for e in train_manifest + val_manifest if e["has_stutters"])}
Total stutter indicators: {total_stutters}

Output directory: {output_dir}

Next step - train S2Mel LoRA:
  python tools/train_s2mel_lora.py --speaker {args.speaker}

WHY S2MEL TRAINING WORKS FOR STUTTERS:
======================================
Unlike GPT training (which only affects semantic codes), S2Mel training
directly learns the MEL SPECTROGRAM patterns. This means:

1. The exact acoustic patterns of stutters are learned
2. Timing/rhythm of speech is captured in the mel domain
3. These patterns become embedded in the mel generation process
4. At inference, the trained S2Mel produces authentically stuttered mel specs!
""")


if __name__ == "__main__":
    main()