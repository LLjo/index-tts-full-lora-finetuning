#!/usr/bin/env python3
"""
Prepare GPT-Aligned Dataset for S2Mel LoRA Training

CRITICAL FIX: This version extracts semantic codes using the SAME process as inference:
Text → GPT → semantic_codes → vq2emb → S_infer

This ensures training and inference use the same representation!

The Problem with Standard Preparation:
======================================
Standard preparation: Audio → W2V-BERT → semantic_emb
Inference uses:       Text → GPT → codes → vq2emb + latent

These are DIFFERENT representations! The S2Mel learns from one domain 
but is asked to generate from another at inference time.

The Solution:
=============
1. Use GPT to generate semantic codes from verbatim text
2. Use the same code → embedding conversion as inference
3. Store the ACTUAL mel spectrograms from audio as targets
4. Train S2Mel to generate target mels from GPT-style conditioning

This way:
- Training: TextCodes → S2Mel → TargetMel (from real audio)
- Inference: TextCodes → S2Mel → GeneratedMel

Both use the same input representation!

Usage:
    python tools/prepare_gpt_aligned_dataset.py --speaker ozzy
    
    # Then train:
    python tools/train_s2mel_lora.py --speaker ozzy --use-gpt-aligned
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
class AlignedSample:
    """Represents a single GPT-aligned training sample."""
    id: str
    audio_path: str
    text: str
    duration: float
    has_stutters: bool = False
    stutter_count: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare GPT-aligned dataset for S2Mel training",
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
    
    # GPT generation options
    parser.add_argument("--skip-normalize", action="store_true", default=True,
                        help="Skip text normalization to preserve stutters (default: True)")
    parser.add_argument("--max-mel-tokens", type=int, default=1500,
                        help="Max mel tokens for GPT generation")
    
    # Model paths
    parser.add_argument("--config", type=Path, 
                        default=PROJECT_ROOT / "checkpoints" / "config.yaml")
    parser.add_argument("--model-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


def detect_stutters(text: str) -> Tuple[bool, int]:
    """Detect stutters and repetitions in verbatim text."""
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


def load_transcripts(csv_path: Path, audio_dir: Path) -> List[AlignedSample]:
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
            text = row.get("parakeet") or row.get("fastwhisper") or row.get("transcript") or ""
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
            
            sample = AlignedSample(
                id=audio_path.stem,
                audio_path=str(audio_path),
                text=text,
                duration=duration,
                has_stutters=has_stutters,
                stutter_count=stutter_count,
            )
            samples.append(sample)
    
    return samples


class GPTAlignedExtractor:
    """
    Extract features using GPT-aligned process.
    
    The KEY innovation: We run GPT inference to get semantic codes,
    then use those codes (via vq2emb + latent) as S2Mel input.
    This matches what happens at inference time!
    """
    
    def __init__(self, config_path: Path, model_dir: Path, device: str = None, skip_normalize: bool = True):
        import os
        os.environ['HF_HUB_CACHE'] = str(model_dir / 'hf_cache')
        from omegaconf import OmegaConf
        
        self.config_path = config_path
        self.model_dir = model_dir
        self.skip_normalize = skip_normalize
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.cfg = OmegaConf.load(config_path)
        self._load_models()
    
    def _load_models(self):
        """Load all required models."""
        from transformers import SeamlessM4TFeatureExtractor
        from huggingface_hub import hf_hub_download
        import safetensors
        
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.audio import mel_spectrogram
        from indextts.gpt.model_v2 import UnifiedVoice
        from indextts.utils.front import TextNormalizer, TextTokenizer
        from indextts.utils.checkpoint import load_checkpoint
        
        print("Loading models for GPT-aligned extraction...")
        
        # GPT model for code generation
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=False)
        gpt_path = str(self.model_dir / self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, gpt_path)
        self.gpt = self.gpt.to(self.device)
        self.gpt.eval()
        print(f"  ✓ GPT loaded from {gpt_path}")
        
        # Tokenizer
        bpe_path = str(self.model_dir / self.cfg.dataset["bpe_model"])
        if self.skip_normalize:
            # Create tokenizer WITHOUT normalizer to preserve stutters
            self.tokenizer = TextTokenizer(bpe_path, normalizer=None)
            print("  ✓ Tokenizer loaded (NO normalization - stutters preserved!)")
        else:
            normalizer = TextNormalizer(enable_glossary=True)
            normalizer.load()
            self.tokenizer = TextTokenizer(bpe_path, normalizer=normalizer)
            print("  ✓ Tokenizer loaded (with normalization)")
        
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
        
        # Semantic codec for quantization/dequantization
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print("  ✓ Semantic codec loaded")
        
        # S2Mel model (for length regulator and gpt_layer)
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
        
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
        
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
    def generate_gpt_codes(
        self, 
        text: str, 
        spk_cond_emb: torch.Tensor,
        max_mel_tokens: int = 1500,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate semantic codes from text using GPT.
        
        This is the KEY function that aligns training with inference:
        we use GPT to generate codes, just like inference does!
        
        Returns:
            codes: (1, T) - Semantic codes
            latent: (1, T, dim) - GPT latent for S2Mel
        """
        # Tokenize text
        text_tokens = self.tokenizer.encode(text, out_type=int)
        text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
        
        # Use speaker embedding for both spk and emo conditioning
        emo_cond_emb = spk_cond_emb
        
        # Generate emovec
        emovec = self.gpt.merge_emovec(
            spk_cond_emb,
            emo_cond_emb,
            torch.tensor([spk_cond_emb.shape[1]], device=self.device),
            torch.tensor([emo_cond_emb.shape[1]], device=self.device),
            alpha=1.0
        )
        
        # Generate codes
        codes, speech_conditioning_latent = self.gpt.inference_speech(
            spk_cond_emb,
            text_tokens,
            emo_cond_emb,
            cond_lengths=torch.tensor([spk_cond_emb.shape[1]], device=self.device),
            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[1]], device=self.device),
            emo_vec=emovec,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            num_return_sequences=1,
            length_penalty=0.0,
            num_beams=3,
            repetition_penalty=10.0,
            max_generate_length=max_mel_tokens,
        )
        
        # Get code length (before stop token)
        stop_mel_token = self.cfg.gpt.stop_mel_token
        if (codes == stop_mel_token).any():
            code_len = (codes[0] == stop_mel_token).nonzero(as_tuple=False)[0].item()
        else:
            code_len = codes.shape[1]
        
        codes = codes[:, :code_len]
        
        # Get GPT latent
        use_speed = torch.zeros(1).to(self.device).long()
        latent = self.gpt(
            speech_conditioning_latent,
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=self.device),
            codes,
            torch.tensor([codes.shape[-1]], device=self.device),
            emo_cond_emb,
            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=self.device),
            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[1]], device=self.device),
            emo_vec=emovec,
            use_speed=use_speed,
        )
        
        return codes, latent
    
    @torch.no_grad()
    def get_s2mel_condition(
        self,
        codes: torch.Tensor,
        latent: torch.Tensor,
        target_mel_length: int,
    ) -> torch.Tensor:
        """
        Convert GPT codes + latent to S2Mel conditioning.
        
        This mirrors the inference process exactly (from infer_v2.py lines 764-774):
        ```
        latent = self.s2mel.models['gpt_layer'](latent)
        S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)
        S_infer = S_infer + latent  # NO transpose on latent!
        ```
        """
        code_len = codes.shape[1]
        
        # Project latent through gpt_layer: (B, T, 1280) -> (B, T, 1024)
        latent_proj = self.s2mel.models['gpt_layer'](latent)
        
        # Ensure we use only the code token positions.
        # The GPT output includes hidden states for both text and code tokens;
        # the last `code_len` positions correspond to the generated codes.
        # Slice unconditionally to avoid accidental inclusion of text token latents.
        latent_proj = latent_proj[:, -code_len:, :]
        
        # Convert codes to embeddings (same as inference)
        # vq2emb: (B, 1, T_codes) -> (B, 1024, T_codes)
        S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        # transpose: (B, 1024, T_codes) -> (B, T_codes, 1024)
        S_infer = S_infer.transpose(1, 2)
        
        # Add latent (NO transpose - exactly like inference!)
        # Both tensors are now (B, T_codes, 1024)
        S_infer = S_infer + latent_proj
        
        # Through length regulator to get condition at target mel length
        # S_infer: (B, T_codes, 1024) -> cond: (B, T_mel, 768)
        target_lengths = torch.tensor([target_mel_length], device=self.device)
        
        cond = self.s2mel.models['length_regulator'](
            S_infer,  # (B, T_codes, 1024)
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]  # (B, T_mel, 768)
        
        return cond
    
    def extract_all_features(
        self, 
        audio_path: str,
        text: str,
        max_duration: float = 15.0,
        max_mel_tokens: int = 1500,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features using GPT-aligned process.
        
        Returns:
            Dict with:
            - mel: Target mel spectrogram (from audio)
            - gpt_codes: GPT-generated semantic codes
            - gpt_latent: GPT latent for S2Mel
            - s2mel_cond: GPT-aligned conditioning for S2Mel
            - style: Global style vector
            - spk_cond_emb: Speaker conditioning embedding
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
        
        # Extract speaker conditioning from audio
        spk_cond_emb = self.get_semantic_embedding(audio_16k)
        
        # Get target mel spectrogram (ground truth from audio)
        mel = self.get_mel_spectrogram(audio_22k.to(self.device))
        mel_length = mel.size(-1)
        
        # Get style vector
        style = self.get_style_vector(audio_16k)
        
        # Generate GPT codes from TEXT (this is the key alignment!)
        codes, latent = self.generate_gpt_codes(text, spk_cond_emb, max_mel_tokens)
        
        # Get S2Mel conditioning (using actual mel length for better alignment)
        s2mel_cond = self.get_s2mel_condition(codes, latent, mel_length)
        
        # Also store the duration ratio for inference adjustment
        code_len = codes.shape[1]
        duration_ratio = mel_length / (code_len + 1e-6)  # mel_frames per code
        
        return {
            "mel": mel.cpu(),
            "gpt_codes": codes.cpu(),
            "gpt_latent": latent.cpu(),
            "s2mel_cond": s2mel_cond.cpu(),
            "style": style.cpu(),
            "spk_cond_emb": spk_cond_emb.cpu(),
            "mel_length": mel_length,
            "code_length": code_len,
            "duration_ratio": duration_ratio,
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
        output_dir = speaker_dir / "dataset" / "processed_gpt_aligned"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GPT-ALIGNED DATASET PREPARATION")
    print("=" * 60)
    print(f"""
Speaker: {args.speaker}
Audio directory: {audio_dir}
Transcripts: {transcripts_file}
Output: {output_dir}
Skip normalization: {args.skip_normalize} (preserves stutters!)
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
    print("\n[2/4] Initializing GPT-aligned feature extractor...")
    extractor = GPTAlignedExtractor(
        args.config, 
        args.model_dir, 
        skip_normalize=args.skip_normalize
    )
    
    # Process samples
    print("\n[3/4] Extracting GPT-aligned features...")
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    train_manifest = []
    val_manifest = []
    
    # Track duration ratios for statistics
    duration_ratios = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        try:
            features = extractor.extract_all_features(
                sample.audio_path, 
                sample.text,
                max_duration=args.max_duration,
                max_mel_tokens=args.max_mel_tokens,
            )
            
            # Save features
            sample_dir = features_dir / sample.id
            sample_dir.mkdir(exist_ok=True)
            
            np.save(sample_dir / "mel.npy", features["mel"].numpy())
            np.save(sample_dir / "gpt_codes.npy", features["gpt_codes"].numpy())
            np.save(sample_dir / "gpt_latent.npy", features["gpt_latent"].numpy())
            np.save(sample_dir / "s2mel_cond.npy", features["s2mel_cond"].numpy())
            np.save(sample_dir / "style.npy", features["style"].numpy())
            np.save(sample_dir / "spk_cond_emb.npy", features["spk_cond_emb"].numpy())
            
            duration_ratios.append(features["duration_ratio"])
            
            # Create manifest entry
            entry = {
                "id": sample.id,
                "audio_path": sample.audio_path,
                "text": sample.text,
                "duration": sample.duration,
                "has_stutters": sample.has_stutters,
                "stutter_count": sample.stutter_count,
                "mel_path": str(sample_dir / "mel.npy"),
                "gpt_codes_path": str(sample_dir / "gpt_codes.npy"),
                "gpt_latent_path": str(sample_dir / "gpt_latent.npy"),
                "s2mel_cond_path": str(sample_dir / "s2mel_cond.npy"),
                "style_path": str(sample_dir / "style.npy"),
                "spk_cond_emb_path": str(sample_dir / "spk_cond_emb.npy"),
                "mel_length": features["mel_length"],
                "code_length": features["code_length"],
                "duration_ratio": features["duration_ratio"],
            }
            
            # Train/val split
            if i < int(len(samples) * (1 - args.val_split)):
                train_manifest.append(entry)
            else:
                val_manifest.append(entry)
                
        except Exception as e:
            print(f"\n  Error processing {sample.id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
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
    
    # Calculate statistics
    avg_duration_ratio = np.mean(duration_ratios) if duration_ratios else 1.72
    std_duration_ratio = np.std(duration_ratios) if duration_ratios else 0.0
    
    # Save metadata
    metadata = {
        "speaker": args.speaker,
        "avg_duration_ratio": float(avg_duration_ratio),
        "std_duration_ratio": float(std_duration_ratio),
        "skip_normalize": args.skip_normalize,
        "total_samples": len(train_manifest) + len(val_manifest),
        "train_samples": len(train_manifest),
        "val_samples": len(val_manifest),
        "stutter_samples": sum(1 for e in train_manifest + val_manifest if e["has_stutters"]),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Train manifest: {train_manifest_path} ({len(train_manifest)} samples)")
    print(f"✓ Val manifest: {val_manifest_path} ({len(val_manifest)} samples)")
    
    # Summary
    print("\n" + "=" * 60)
    print("GPT-ALIGNED PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
Total samples: {len(train_manifest) + len(val_manifest)}
  Training: {len(train_manifest)}
  Validation: {len(val_manifest)}

Samples with stutters: {metadata['stutter_samples']}

Duration ratio (mel_frames / code):
  Average: {avg_duration_ratio:.3f}
  Std dev: {std_duration_ratio:.3f}
  (Default inference uses 1.72)

Output directory: {output_dir}

WHY GPT-ALIGNED TRAINING WORKS:
===============================
The key insight: Training must use the SAME representations as inference!

Standard training uses:
  Audio → W2V-BERT → semantic_emb (continuous features)

Inference uses:
  Text → GPT → codes → vq2emb + latent (discrete codes)

These are DIFFERENT! The model learns from one domain but generates from another.

GPT-aligned training uses:
  Text → GPT → codes → vq2emb + latent (same as inference!)
  
Now both training and inference use the same input representation.
The S2Mel learns to generate mels from GPT-style codes, matching inference exactly.

Next step - train S2Mel LoRA with GPT-aligned data:
  python tools/train_s2mel_lora.py --speaker {args.speaker} --use-gpt-aligned
""")


if __name__ == "__main__":
    main()