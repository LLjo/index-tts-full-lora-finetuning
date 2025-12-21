#!/usr/bin/env python3
"""
Pattern Dataset Preparation v3 for IndexTTS2

This prepares datasets for PATTERN EMBEDDING training - the new approach
that actually makes speaking patterns work!

Key differences from v2:
1. Extracts PATTERN FEATURES from each audio (pauses, stutters, rate)
2. Saves pattern features alongside standard features
3. Pattern features enable pattern-aware loss during training

Usage:
    # Basic usage
    python tools/prepare_pattern_dataset_v3.py --speaker ozzy
    
    # With custom options  
    python tools/prepare_pattern_dataset_v3.py --speaker ozzy \
        --pause-threshold 0.3 \
        --detect-stutters
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExtractedPatternFeatures:
    """Pattern features extracted from audio and transcript."""
    pause_positions: List[int]
    pause_durations: List[float]
    filler_positions: List[int]
    stutter_positions: List[int]
    speech_rate: float
    rate_variations: List[float]
    total_duration: float
    num_pauses: int
    num_fillers: int
    num_stutters: int


class PatternAnalyzer:
    """Analyze audio and transcripts to extract speaking patterns."""
    
    SILENCE_CODES = {52}  # IndexTTS2's silence code
    PAUSE_THRESHOLD = 3  # Consecutive silence codes = pause
    
    FILLER_PATTERNS = [
        r'\b(uh+|uhh+)\b',
        r'\b(um+|umm+)\b',
        r'\b(er+|err+)\b',
        r'\b(ah+|ahh+)\b',
        r'\[UH\]', r'\[UM\]', r'\[ER\]', r'\[AH\]',
        r'\[PAUSE\]', r'\[LONG\]', r'\[BREATH\]',
    ]
    
    STUTTER_PATTERNS = [
        r'\b(\w+)-\1\b',  # Word-word (hyphen)
        r'\b(\w{1,3})-\1',  # Short repeated syllables
        r'\.{3,}',  # Long ellipsis (hesitation)
    ]
    
    def __init__(
        self,
        silence_codes: Optional[set] = None,
        pause_threshold: int = 3,
    ):
        self.silence_codes = silence_codes or self.SILENCE_CODES
        self.pause_threshold = pause_threshold
    
    def analyze_codes(
        self,
        codes: np.ndarray,
        text: str,
        audio_duration: float,
    ) -> ExtractedPatternFeatures:
        """
        Extract pattern features from semantic codes and transcript.
        
        Args:
            codes: Semantic codes (1D array)
            text: Transcript text
            audio_duration: Duration in seconds
            
        Returns:
            ExtractedPatternFeatures
        """
        codes = codes.flatten()
        
        # Analyze pauses from codes
        pause_positions, pause_durations = self._find_pauses_in_codes(
            codes, audio_duration
        )
        
        # Analyze text for fillers
        filler_positions = self._find_fillers_in_text(text)
        
        # Analyze text for stutters
        stutter_positions = self._find_stutters_in_text(text)
        
        # Calculate speech rate
        word_count = len([w for w in text.split() if not w.startswith('[')])
        speech_rate = word_count / max(audio_duration, 0.1)
        
        # Estimate rate variations
        rate_variations = self._estimate_rate_variations(codes)
        
        return ExtractedPatternFeatures(
            pause_positions=pause_positions,
            pause_durations=pause_durations,
            filler_positions=filler_positions,
            stutter_positions=stutter_positions,
            speech_rate=speech_rate,
            rate_variations=rate_variations,
            total_duration=audio_duration,
            num_pauses=len(pause_positions),
            num_fillers=len(filler_positions),
            num_stutters=len(stutter_positions),
        )
    
    def _find_pauses_in_codes(
        self,
        codes: np.ndarray,
        audio_duration: float,
    ) -> Tuple[List[int], List[float]]:
        """Find pause positions and durations from codes."""
        positions = []
        durations = []
        
        run_start = None
        run_length = 0
        
        for i, code in enumerate(codes):
            if code in self.silence_codes:
                if run_start is None:
                    run_start = i
                run_length += 1
            else:
                if run_length >= self.pause_threshold:
                    positions.append(run_start)
                    # Estimate duration
                    duration_per_code = audio_duration / max(len(codes), 1)
                    durations.append(run_length * duration_per_code * 1000)  # ms
                run_start = None
                run_length = 0
        
        # Check final run
        if run_length >= self.pause_threshold:
            positions.append(run_start)
            duration_per_code = audio_duration / max(len(codes), 1)
            durations.append(run_length * duration_per_code * 1000)
        
        return positions, durations
    
    def _find_fillers_in_text(self, text: str) -> List[int]:
        """Find filler word positions."""
        positions = []
        for pattern in self.FILLER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                positions.append(match.start())
        return sorted(set(positions))
    
    def _find_stutters_in_text(self, text: str) -> List[int]:
        """Find stutter/repetition positions."""
        positions = []
        for pattern in self.STUTTER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                positions.append(match.start())
        return sorted(set(positions))
    
    def _estimate_rate_variations(
        self,
        codes: np.ndarray,
        window_size: int = 50,
    ) -> List[float]:
        """Estimate local speech rate variations."""
        if len(codes) < window_size:
            return [1.0]
        
        variations = []
        for i in range(0, len(codes) - window_size, window_size // 2):
            window = codes[i:i + window_size]
            silence_ratio = sum(1 for c in window if c in self.silence_codes) / len(window)
            # Convert to rate multiplier
            rate = max(0.1, (1.0 - silence_ratio) * 2)
            variations.append(rate)
        
        return variations


def load_transcripts(path: Path) -> Dict[str, str]:
    """Load transcripts from CSV or JSON."""
    transcripts = {}
    
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename") or row.get("audio")
                text = row.get("verbatim") or row.get("text") or row.get("transcription")
                if filename and text:
                    transcripts[filename] = text
    
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    filename = item.get("filename") or item.get("audio")
                    text = item.get("verbatim") or item.get("text")
                    if filename and text:
                        transcripts[filename] = text
            else:
                transcripts = dict(data)
    
    return transcripts


def normalize_transcript(text: str) -> str:
    """Normalize transcript markers."""
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'\b(uh+|uhh+)\b', '[UH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(um+|umm+|hmm+)\b', '[UM]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ah+|ahh+)\b', '[AH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(er+|err+)\b', '[ER]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[pause\]', '[PAUSE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[long\s*pause\]', '[LONG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pattern dataset v3 with pattern feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", help="Speaker name")
    parser.add_argument("--audio-dir", type=Path, help="Audio directory")
    parser.add_argument("--transcripts", type=Path, help="Transcripts file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    # Pattern conditioning
    parser.add_argument("--pattern-conditioning", type=Path,
                        help="Global pattern conditioning file (from v2)")
    
    # Pattern analysis options
    parser.add_argument("--pause-threshold", type=int, default=3,
                        help="Consecutive silence codes for pause detection")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--train-split", type=float, default=0.9)
    
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker / "dataset"
        audio_dir = args.audio_dir or speaker_dir / "audio"
        transcripts_path = args.transcripts or speaker_dir / "transcripts_verbatim.csv"
        output_dir = args.output_dir or speaker_dir / "processed_v3"
        
        # Try to find pattern conditioning from v2
        if args.pattern_conditioning is None:
            pattern_cond_path = PROJECT_ROOT / "training" / args.speaker / "pattern_conditioning.pt"
            if pattern_cond_path.exists():
                args.pattern_conditioning = pattern_cond_path
    else:
        if not args.audio_dir or not args.transcripts or not args.output_dir:
            parser.error("--speaker or all of --audio-dir, --transcripts, --output-dir required")
        audio_dir = args.audio_dir
        transcripts_path = args.transcripts
        output_dir = args.output_dir
    
    print("=" * 60)
    print("PATTERN DATASET PREPARATION v3")
    print("=" * 60)
    print(f"\nAudio: {audio_dir}")
    print(f"Transcripts: {transcripts_path}")
    print(f"Output: {output_dir}")
    
    # Validate
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    if not transcripts_path.exists():
        print(f"❌ Transcripts not found: {transcripts_path}")
        sys.exit(1)
    
    # Load global conditioning if available
    global_condition = None
    global_emo_vec = None
    if args.pattern_conditioning and args.pattern_conditioning.exists():
        print(f"\n[*] Loading global conditioning from: {args.pattern_conditioning}")
        from indextts.pattern_conditioning import PatternConditioningStore
        cond_data = PatternConditioningStore.load(args.pattern_conditioning)
        global_condition = cond_data['gpt_conditioning'].squeeze(0).numpy().astype(np.float32)
        global_emo_vec = cond_data['emo_vec'].squeeze(0).numpy().astype(np.float32)
        print(f"  Condition shape: {global_condition.shape}")
    
    # Load models
    print("\n[1/5] Loading models...")
    
    from transformers import SeamlessM4TFeatureExtractor
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    import safetensors
    
    from indextts.utils.front import TextNormalizer, TextTokenizer
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    cfg = OmegaConf.load(args.config)
    
    # Tokenizer
    bpe_path = args.model_dir / cfg.dataset["bpe_model"]
    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(bpe_path), normalizer)
    
    # Semantic model
    extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        str(args.model_dir / cfg.w2v_stat)
    )
    semantic_model = semantic_model.to(device).eval()
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    
    # Semantic codec
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    codec_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, codec_ckpt)
    semantic_codec = semantic_codec.to(device).eval()
    
    # GPT for conditioning (if we need to extract)
    gpt = None
    if global_condition is None:
        print("  Loading GPT for conditioning extraction...")
        gpt_path = args.model_dir / cfg.gpt_checkpoint
        checkpoint = torch.load(gpt_path, map_location="cpu")
        raw_state = checkpoint.get("model", checkpoint)
        if "mel_pos_embedding.emb.weight" in raw_state:
            checkpoint_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
            if cfg.gpt.model_dim != checkpoint_dim:
                cfg.gpt.model_dim = checkpoint_dim
        gpt = UnifiedVoice(**cfg.gpt)
        load_checkpoint(gpt, str(gpt_path))
        gpt = gpt.to(device).eval()
    
    max_text_tokens = 120  # Safe default
    max_mel_tokens = 500
    
    # Pattern analyzer
    analyzer = PatternAnalyzer(pause_threshold=args.pause_threshold)
    
    # Load transcripts
    print("\n[2/5] Loading transcripts...")
    transcripts = load_transcripts(transcripts_path)
    print(f"  Loaded {len(transcripts)} transcriptions")
    
    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    audio_files = sorted(audio_files)
    print(f"  Found {len(audio_files)} audio files")
    
    # Match audio with transcripts
    matched = []
    for audio_path in audio_files:
        for candidate in [audio_path.name, audio_path.stem]:
            if candidate in transcripts:
                matched.append((audio_path, transcripts[candidate]))
                break
    print(f"  Matched {len(matched)} pairs")
    
    if not matched:
        print("❌ No matching audio-transcript pairs!")
        sys.exit(1)
    
    # Process
    print("\n[3/5] Processing audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Save global conditioning if we have it
    if global_condition is not None:
        np.save(features_dir / "GLOBAL_condition.npy", global_condition)
        np.save(features_dir / "GLOBAL_emo_vec.npy", global_emo_vec)
    
    manifest_entries = []
    pattern_stats = {
        "total_pauses": 0,
        "total_fillers": 0,
        "total_stutters": 0,
        "samples_with_pauses": 0,
        "samples_with_fillers": 0,
        "samples_with_stutters": 0,
    }
    
    with torch.no_grad():
        for audio_path, transcript in tqdm(matched, desc="Processing"):
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
                duration = len(audio) / sr
                
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                
                # Normalize transcript
                prepared_text = normalize_transcript(transcript)
                
                # Resample
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
                
                # Tokenize
                text_tokens = tokenizer.tokenize(prepared_text)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                text_ids_array = np.array(text_ids, dtype=np.int32)
                
                if len(text_ids) > max_text_tokens:
                    continue
                
                # Extract semantic codes
                inputs = extract_features(audio_16k_tensor, sampling_rate=16000, return_tensors="pt")
                input_features = inputs["input_features"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                vq_emb = semantic_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                feat = vq_emb.hidden_states[17]
                feat = (feat - semantic_mean) / semantic_std
                
                # Generate codes
                codes, _ = semantic_codec.quantize(feat)
                if codes.ndim == 2:
                    codes = codes[0]
                codes_np = codes.cpu().numpy().astype(np.int32)
                
                if codes_np.shape[0] > max_mel_tokens:
                    continue
                
                # === EXTRACT PATTERN FEATURES ===
                pattern_features = analyzer.analyze_codes(codes_np, prepared_text, duration)
                
                # Update stats
                pattern_stats["total_pauses"] += pattern_features.num_pauses
                pattern_stats["total_fillers"] += pattern_features.num_fillers
                pattern_stats["total_stutters"] += pattern_features.num_stutters
                if pattern_features.num_pauses > 0:
                    pattern_stats["samples_with_pauses"] += 1
                if pattern_features.num_fillers > 0:
                    pattern_stats["samples_with_fillers"] += 1
                if pattern_features.num_stutters > 0:
                    pattern_stats["samples_with_stutters"] += 1
                
                # Extract conditioning (if not using global)
                if global_condition is None:
                    cond_lengths = torch.tensor([feat.shape[1]], device=device)
                    gpt_cond = gpt.get_conditioning(feat.transpose(1, 2), cond_lengths)
                    emo_cond = gpt.get_emo_conditioning(feat.transpose(1, 2), cond_lengths)
                    emo_vec = gpt.emovec_layer(emo_cond)
                    emo_vec = gpt.emo_layer(emo_vec)
                    
                    condition = gpt_cond.squeeze(0).cpu().numpy().astype(np.float32)
                    emo_vec_np = emo_vec.squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    condition = global_condition
                    emo_vec_np = global_emo_vec
                
                # Save features
                sample_id = audio_path.stem
                np.save(features_dir / f"{sample_id}_text_ids.npy", text_ids_array)
                np.save(features_dir / f"{sample_id}_codes.npy", codes_np)
                
                # Save pattern features
                pf_dict = asdict(pattern_features)
                np.save(features_dir / f"{sample_id}_pattern_features.npy", pf_dict)
                
                # Determine condition path
                if global_condition is not None:
                    condition_path = "features/GLOBAL_condition.npy"
                    emo_vec_path = "features/GLOBAL_emo_vec.npy"
                else:
                    np.save(features_dir / f"{sample_id}_condition.npy", condition)
                    np.save(features_dir / f"{sample_id}_emo_vec.npy", emo_vec_np)
                    condition_path = f"features/{sample_id}_condition.npy"
                    emo_vec_path = f"features/{sample_id}_emo_vec.npy"
                
                manifest_entries.append({
                    "id": sample_id,
                    "text": prepared_text,
                    "audio_path": str(audio_path),
                    "text_ids_path": f"features/{sample_id}_text_ids.npy",
                    "codes_path": f"features/{sample_id}_codes.npy",
                    "condition_path": condition_path,
                    "emo_vec_path": emo_vec_path,
                    "pattern_features_path": f"features/{sample_id}_pattern_features.npy",
                    "text_len": len(text_ids),
                    "code_len": codes_np.shape[0],
                    "condition_len": condition.shape[0],
                    "duration": float(duration),
                    "sample_type": "pattern_v3",
                    # Pattern summary in manifest
                    "num_pauses": pattern_features.num_pauses,
                    "num_fillers": pattern_features.num_fillers,
                    "num_stutters": pattern_features.num_stutters,
                    "speech_rate": pattern_features.speech_rate,
                })
                
            except Exception as e:
                warnings.warn(f"Failed to process {audio_path.name}: {e}")
    
    print(f"  Processed {len(manifest_entries)} samples")
    
    # Pattern statistics
    print(f"\n[4/5] Pattern statistics:")
    print(f"  Total pauses detected: {pattern_stats['total_pauses']}")
    print(f"  Total fillers detected: {pattern_stats['total_fillers']}")
    print(f"  Total stutters detected: {pattern_stats['total_stutters']}")
    print(f"  Samples with pauses: {pattern_stats['samples_with_pauses']}/{len(manifest_entries)}")
    print(f"  Samples with fillers: {pattern_stats['samples_with_fillers']}/{len(manifest_entries)}")
    print(f"  Samples with stutters: {pattern_stats['samples_with_stutters']}/{len(manifest_entries)}")
    
    # Split and save
    print("\n[5/5] Saving manifests...")
    np.random.seed(42)
    indices = np.random.permutation(len(manifest_entries))
    split_idx = int(len(indices) * args.train_split)
    
    train_entries = [manifest_entries[i] for i in indices[:split_idx]]
    val_entries = [manifest_entries[i] for i in indices[split_idx:]]
    
    train_path = output_dir / "train_manifest.jsonl"
    val_path = output_dir / "val_manifest.jsonl"
    
    with open(train_path, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(val_path, "w") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Save dataset info
    info = {
        "version": "v3_pattern_features",
        "total_samples": len(manifest_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "uses_global_conditioning": global_condition is not None,
        "pattern_stats": pattern_stats,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n  Train: {train_path} ({len(train_entries)} samples)")
    print(f"  Val: {val_path} ({len(val_entries)} samples)")
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
This dataset includes PATTERN FEATURES for each sample, enabling:
  - Pattern-aware loss during training
  - Explicit pause/stutter detection
  - Speech rate analysis

NEXT STEPS:
===========
Train with pattern embeddings:

    python tools/train_pattern_embeddings.py \\
        --speaker {args.speaker or 'SPEAKER'} \\
        --epochs 40 \\
        --pattern-tokens 8

Then run inference with patterns:

    python tools/infer_with_patterns.py \\
        --speaker {args.speaker or 'SPEAKER'} \\
        --text "Your text here"
""")


if __name__ == "__main__":
    main()