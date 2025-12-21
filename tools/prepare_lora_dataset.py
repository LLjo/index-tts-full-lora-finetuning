#!/usr/bin/env python3
"""
Data preparation tool for LoRA fine-tuning of IndexTTS.

This script takes a directory of audio files and their transcriptions,
processes them through the IndexTTS pipeline, and generates the paired
manifest files needed for training.

Usage:
    python tools/prepare_lora_dataset.py \
        --audio-dir data/my_voice/audio \
        --transcripts data/my_voice/transcripts.csv \
        --output-dir data/my_voice_processed \
        --model-dir checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor

# Add parent directory to path to import indextts
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import safetensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare audio dataset for LoRA fine-tuning"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing audio files (wav, mp3, flac)",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        required=True,
        help="Path to transcriptions file (CSV or JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed features and manifests",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Path to model checkpoints directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (rest for validation)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps, xpu). Auto-detected if not specified",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def load_transcripts(transcript_path: Path) -> Dict[str, str]:
    """
    Load transcriptions from CSV or JSON file.
    
    CSV format: filename,text
    JSON format: [{"audio": "file.wav", "text": "..."}] or {"file.wav": "..."}
    
    Returns:
        Dictionary mapping audio filename to transcription text
    """
    transcripts = {}
    
    if transcript_path.suffix.lower() == ".csv":
        with open(transcript_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both 'filename' and 'audio' column names
                filename = row.get("filename") or row.get("audio")
                text = row.get("text") or row.get("transcription")
                if filename and text:
                    transcripts[filename] = text.strip()
    
    elif transcript_path.suffix.lower() == ".json":
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, list):
                # List format: [{"audio": "file.wav", "text": "..."}]
                for item in data:
                    filename = item.get("filename") or item.get("audio")
                    text = item.get("text") or item.get("transcription")
                    if filename and text:
                        transcripts[filename] = text.strip()
            
            elif isinstance(data, dict):
                # Dict format: {"file.wav": "text"}
                transcripts = {k: v.strip() for k, v in data.items()}
    
    else:
        raise ValueError(f"Unsupported transcript format: {transcript_path.suffix}")
    
    return transcripts


def find_audio_files(audio_dir: Path) -> List[Path]:
    """Find all supported audio files in directory."""
    supported_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    
    for ext in supported_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)


def process_audio_file(
    audio_path: Path,
    text: str,
    output_dir: Path,
    tokenizer: TextTokenizer,
    semantic_model,
    semantic_codec,
    semantic_mean,
    semantic_std,
    gpt_model: UnifiedVoice,
    extract_features,
    device: str,
    max_duration: float,
    min_duration: float,
) -> Dict | None:
    """
    Process a single audio file and extract all required features.
    
    Returns:
        Dictionary with feature paths and metadata, or None if processing failed
    """
    try:
        # Get max supported lengths from model's position embeddings
        # These are the actual limits from the checkpoint, which may be less than model_dim config
        max_text_tokens = gpt_model.text_pos_embedding.emb.num_embeddings
        max_mel_tokens = gpt_model.mel_pos_embedding.emb.num_embeddings
        
        # Account for start/stop tokens that will be added during training (+2 each)
        max_text_tokens_safe = max_text_tokens - 2
        max_mel_tokens_safe = max_mel_tokens - 2
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        duration = len(audio) / sr
        
        # Check duration
        if duration < min_duration or duration > max_duration:
            warnings.warn(
                f"Skipping {audio_path.name}: duration {duration:.2f}s "
                f"(min={min_duration}, max={max_duration})"
            )
            return None
        
        # Resample to required sample rates
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        
        audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
        audio_22k_tensor = torch.from_numpy(audio_22k).unsqueeze(0)
        
        # Tokenize text
        text_tokens = tokenizer.tokenize(text)
        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        text_ids_array = np.array(text_ids, dtype=np.int32)
        
        # Check if text is too long (accounting for +2 start/stop tokens)
        if len(text_ids) > max_text_tokens_safe:
            warnings.warn(
                f"Skipping {audio_path.name}: text too long "
                f"({len(text_ids)} tokens > max {max_text_tokens_safe}). "
                f"Consider splitting the text."
            )
            return None
        
        # Extract semantic features
        inputs = extract_features(audio_16k_tensor, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            vq_emb = semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17]
            feat = (feat - semantic_mean) / semantic_std
            
            # Generate semantic codes
            # quantize() returns (indices, embeddings) where:
            # - indices: (seq_len,) when single quantizer, or (num_quantizers, seq_len)
            # - embeddings: (batch, hidden_dim, seq_len)
            codes, _ = semantic_codec.quantize(feat)  # Get indices, discard embeddings
            
            # Handle multi-quantizer output (take first quantizer only)
            if codes.ndim == 2:
                codes = codes[0]  # Take first quantizer: (num_quantizers, seq_len) -> (seq_len,)
            
            codes = codes.cpu().numpy().astype(np.int32)
            
            # Ensure codes is 1D
            if codes.ndim != 1:
                warnings.warn(
                    f"Skipping {audio_path.name}: unexpected codes shape {codes.shape}. "
                    f"Expected 1D array of semantic indices."
                )
                return None
            
            # Check if codes are too long (accounting for +2 start/stop tokens)
            if codes.shape[0] > max_mel_tokens_safe:
                warnings.warn(
                    f"Skipping {audio_path.name}: semantic sequence too long "
                    f"({codes.shape[0]} codes > max {max_mel_tokens_safe}). "
                    f"Audio is {duration:.2f}s, try shorter clips (recommended max ~9s for this checkpoint)."
                )
                return None
            
            # Get conditioning latent
            cond_lengths = torch.tensor([feat.shape[1]], device=device)
            conditioning_latent = gpt_model.get_conditioning(
                feat.transpose(1, 2), cond_lengths
            )
            conditioning_latent = conditioning_latent.squeeze(0).cpu().numpy().astype(np.float32)
            
            # Get emotion conditioning
            emo_latent = gpt_model.get_emo_conditioning(
                feat.transpose(1, 2), cond_lengths
            )
            emo_vec = gpt_model.emovec_layer(emo_latent)
            emo_vec = gpt_model.emo_layer(emo_vec)
            emo_vec = emo_vec.squeeze(0).cpu().numpy().astype(np.float32)
        
        # Create output subdirectory for features
        features_dir = output_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique ID for this sample
        sample_id = audio_path.stem
        
        # Save features
        text_ids_path = features_dir / f"{sample_id}_text_ids.npy"
        codes_path = features_dir / f"{sample_id}_codes.npy"
        condition_path = features_dir / f"{sample_id}_condition.npy"
        emo_vec_path = features_dir / f"{sample_id}_emo_vec.npy"
        
        np.save(text_ids_path, text_ids_array)
        np.save(codes_path, codes)
        np.save(condition_path, conditioning_latent)
        np.save(emo_vec_path, emo_vec)
        
        # Return manifest entry
        return {
            "id": sample_id,
            "text": text,
            "audio_path": str(audio_path),
            "text_ids_path": str(text_ids_path.relative_to(output_dir)),
            "codes_path": str(codes_path.relative_to(output_dir)),
            "condition_path": str(condition_path.relative_to(output_dir)),
            "emo_vec_path": str(emo_vec_path.relative_to(output_dir)),
            "text_len": len(text_ids),
            "code_len": codes.shape[0],
            "condition_len": conditioning_latent.shape[0],
            "duration": float(duration),
            "sample_type": "single",
        }
    
    except Exception as e:
        warnings.warn(f"Failed to process {audio_path.name}: {e}")
        return None


def main():
    args = parse_args()
    
    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Load tokenizer
    bpe_path = args.model_dir / cfg.dataset["bpe_model"]
    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(bpe_path), normalizer)
    print(f"Loaded tokenizer from: {bpe_path}")
    
    # Load semantic model
    extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        str(args.model_dir / cfg.w2v_stat)
    )
    semantic_model = semantic_model.to(device).eval()
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    print("Loaded semantic model")
    
    # Load semantic codec
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device).eval()
    print("Loaded semantic codec")
    
    # Load GPT model (for conditioning extraction)
    gpt_path = args.model_dir / cfg.gpt_checkpoint
    
    # Detect model_dim from checkpoint to ensure compatibility
    checkpoint = torch.load(gpt_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint)
    if "mel_pos_embedding.emb.weight" in raw_state:
        checkpoint_model_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != checkpoint_model_dim:
            print(f"[Warn] Config specifies model_dim={cfg.gpt.model_dim}, but checkpoint uses {checkpoint_model_dim}")
            print(f"[Info] Using checkpoint's model_dim={checkpoint_model_dim} for compatibility")
            cfg.gpt.model_dim = checkpoint_model_dim
    
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, str(gpt_path))
    gpt = gpt.to(device).eval()
    print(f"Loaded GPT model from: {gpt_path}")
    
    # Load transcripts
    print(f"Loading transcripts from: {args.transcripts}")
    transcripts = load_transcripts(args.transcripts)
    print(f"Loaded {len(transcripts)} transcriptions")
    
    # Find audio files
    print(f"Searching for audio files in: {args.audio_dir}")
    audio_files = find_audio_files(args.audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    # Match audio files with transcripts
    matched_data = []
    for audio_path in audio_files:
        # Try different filename variations
        candidates = [
            audio_path.name,
            audio_path.stem,
            str(audio_path.relative_to(args.audio_dir)),
        ]
        
        text = None
        for candidate in candidates:
            if candidate in transcripts:
                text = transcripts[candidate]
                break
        
        if text is None:
            warnings.warn(f"No transcription found for: {audio_path.name}")
            continue
        
        matched_data.append((audio_path, text))
    
    print(f"Matched {len(matched_data)} audio files with transcriptions")
    
    if len(matched_data) == 0:
        print("ERROR: No matching audio-transcript pairs found!")
        print("\nPlease check:")
        print("  1. Audio files exist in the audio directory")
        print("  2. Transcript filenames match the audio filenames")
        print("  3. Transcript file format is correct (CSV or JSON)")
        sys.exit(1)
    
    # Process all audio files
    print("\nProcessing audio files...")
    manifest_entries = []
    
    with torch.no_grad():
        for audio_path, text in tqdm(matched_data, desc="Processing"):
            entry = process_audio_file(
                audio_path=audio_path,
                text=text,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                semantic_model=semantic_model,
                semantic_codec=semantic_codec,
                semantic_mean=semantic_mean,
                semantic_std=semantic_std,
                gpt_model=gpt,
                extract_features=extract_features,
                device=device,
                max_duration=args.max_duration,
                min_duration=args.min_duration,
            )
            
            if entry is not None:
                manifest_entries.append(entry)
    
    print(f"\nSuccessfully processed {len(manifest_entries)} samples")
    
    if len(manifest_entries) == 0:
        print("ERROR: No samples were successfully processed!")
        sys.exit(1)
    
    # Split into train/val
    np.random.seed(args.seed)
    indices = np.random.permutation(len(manifest_entries))
    split_idx = int(len(indices) * args.train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_entries = [manifest_entries[i] for i in train_indices]
    val_entries = [manifest_entries[i] for i in val_indices]
    
    print(f"\nTrain samples: {len(train_entries)}")
    print(f"Validation samples: {len(val_entries)}")
    
    # Save manifests
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train_manifest_path = args.output_dir / "train_manifest.jsonl"
    val_manifest_path = args.output_dir / "val_manifest.jsonl"
    
    with open(train_manifest_path, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(val_manifest_path, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\nManifests saved:")
    print(f"  Train: {train_manifest_path}")
    print(f"  Val: {val_manifest_path}")
    
    # Save dataset info
    info = {
        "num_samples": len(manifest_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "train_split": args.train_split,
        "max_duration": args.max_duration,
        "min_duration": args.min_duration,
        "audio_dir": str(args.audio_dir),
        "transcripts": str(args.transcripts),
    }
    
    info_path = args.output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"  Info: {info_path}")
    print("\nDataset preparation complete!")
    print(f"\nYou can now train with:")
    print(f"  python tools/train_gpt_lora.py \\")
    print(f"    --train-manifest {train_manifest_path} \\")
    print(f"    --val-manifest {val_manifest_path} \\")
    print(f"    --output-dir trained_lora/my_voice")


if __name__ == "__main__":
    main()