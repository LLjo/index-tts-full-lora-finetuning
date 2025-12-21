#!/usr/bin/env python3
"""
Pattern-Conditioned Dataset Preparation for IndexTTS2 (v2)

THIS IS THE KEY TO MAKING SPEAKING PATTERN TRAINING WORK!

The Critical Difference from v1:
===============================
v1 (prepare_pattern_dataset.py):
  - Each sample uses conditioning extracted from ITS OWN audio
  - At inference, you use different audio → different conditioning
  - Patterns don't transfer because they're tied to individual embeddings

v2 (THIS SCRIPT):
  - ALL samples use the SAME GLOBAL conditioning
  - At inference, you use the SAME global conditioning
  - Patterns transfer because the model sees consistent conditioning!

How it Works:
============
1. First run: extract_pattern_conditioning.py to get global conditioning
2. This script uses that global conditioning for ALL training samples
3. At inference, use the same conditioning → patterns appear!

Usage:
    # Step 1: Extract global conditioning
    python tools/extract_pattern_conditioning.py --speaker ozzy
    
    # Step 2: Prepare dataset with global conditioning
    python tools/prepare_pattern_dataset_v2.py \\
        --speaker ozzy \\
        --pattern-conditioning training/ozzy/pattern_conditioning.pt
    
    # Step 3: Train
    python tools/train_gpt_lora.py \\
        --train-manifest training/ozzy/dataset/processed_v2/train_manifest.jsonl \\
        --val-manifest training/ozzy/dataset/processed_v2/val_manifest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from dataclasses import dataclass

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def normalize_transcript_markers(text: str) -> str:
    """Normalize various pause/filler notations to standard format."""
    # Normalize ellipsis variations
    text = re.sub(r'\.{2,}', '...', text)
    
    # Normalize common filler spellings
    text = re.sub(r'\b(uh+|uhh+)\b', '[UH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(um+|umm+|hmm+)\b', '[UM]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ah+|ahh+)\b', '[AH]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(er+|err+)\b', '[ER]', text, flags=re.IGNORECASE)
    
    # Normalize explicit pause markers
    text = re.sub(r'\[pause\]', '[PAUSE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[long\s*pause\]', '[LONG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[breath\]', '[BREATH]', text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_transcripts(transcript_path: Path) -> Dict[str, str]:
    """Load transcriptions from CSV or JSON."""
    transcripts = {}
    
    if transcript_path.suffix.lower() == ".csv":
        with open(transcript_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename") or row.get("audio")
                # Support both 'text' and 'verbatim' columns
                text = row.get("verbatim") or row.get("text") or row.get("transcription")
                if filename and text:
                    transcripts[filename] = text
    
    elif transcript_path.suffix.lower() == ".json":
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    filename = item.get("filename") or item.get("audio")
                    text = item.get("verbatim") or item.get("text")
                    if filename and text:
                        transcripts[filename] = text
            elif isinstance(data, dict):
                transcripts = dict(data)
    
    return transcripts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pattern-conditioned dataset (v2) for speaking pattern training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required: Pattern conditioning
    parser.add_argument("--pattern-conditioning", type=Path, required=True,
                        help="Path to pattern_conditioning.pt (from extract_pattern_conditioning.py)")
    
    # Input options
    parser.add_argument("--speaker", "-s",
                        help="Speaker name (uses training/{speaker}/dataset/)")
    parser.add_argument("--audio-dir", type=Path,
                        help="Custom audio directory (overrides --speaker)")
    parser.add_argument("--transcripts", type=Path,
                        help="Transcripts CSV/JSON (default: transcripts_verbatim.csv)")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: processed_v2)")
    
    # Model options
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    
    # Processing options
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Determine paths
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker / "dataset"
        audio_dir = args.audio_dir or speaker_dir / "audio"
        transcripts_path = args.transcripts or speaker_dir / "transcripts_verbatim.csv"
        output_dir = args.output_dir or speaker_dir / "processed_v2"
    elif args.audio_dir:
        audio_dir = args.audio_dir
        transcripts_path = args.transcripts
        output_dir = args.output_dir or audio_dir.parent / "processed_v2"
    else:
        parser.error("Either --speaker or --audio-dir is required")
    
    if transcripts_path is None:
        parser.error("--transcripts is required when using --audio-dir")
    
    # Validate paths
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    if not transcripts_path.exists():
        print(f"❌ Transcripts not found: {transcripts_path}")
        sys.exit(1)
    
    if not args.pattern_conditioning.exists():
        print(f"❌ Pattern conditioning not found: {args.pattern_conditioning}")
        print("\nFirst run: python tools/extract_pattern_conditioning.py --speaker <name>")
        sys.exit(1)
    
    print("=" * 60)
    print("PATTERN-CONDITIONED DATASET PREPARATION (v2)")
    print("=" * 60)
    print(f"\nThis creates datasets using GLOBAL conditioning for ALL samples.")
    print(f"This is KEY to making speaking pattern training work!\n")
    print(f"Audio directory: {audio_dir}")
    print(f"Transcripts: {transcripts_path}")
    print(f"Pattern conditioning: {args.pattern_conditioning}")
    print(f"Output: {output_dir}")
    
    # Load pattern conditioning
    print("\n[1/5] Loading global pattern conditioning...")
    from indextts.pattern_conditioning import PatternConditioningStore
    
    global_conditioning = PatternConditioningStore.load(args.pattern_conditioning)
    
    # Extract the conditioning arrays we'll use for ALL samples
    global_condition = global_conditioning['gpt_conditioning'].squeeze(0).numpy().astype(np.float32)
    global_emo_vec = global_conditioning['emo_vec'].squeeze(0).numpy().astype(np.float32)
    
    print(f"  Global conditioning shape: {global_condition.shape}")
    print(f"  Global emotion vector shape: {global_emo_vec.shape}")
    
    # Import required modules
    print("\n[2/5] Loading models...")
    from transformers import SeamlessM4TFeatureExtractor
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    import safetensors
    
    from indextts.utils.front import TextNormalizer, TextTokenizer
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    
    # Setup device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"  Using device: {device}")
    
    # Load config and models
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
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device).eval()
    
    # GPT model (for max token limits)
    gpt_path = args.model_dir / cfg.gpt_checkpoint
    checkpoint = torch.load(gpt_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint)
    if "mel_pos_embedding.emb.weight" in raw_state:
        checkpoint_model_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != checkpoint_model_dim:
            cfg.gpt.model_dim = checkpoint_model_dim
    
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, str(gpt_path))
    gpt = gpt.to(device).eval()
    
    max_text_tokens = gpt.text_pos_embedding.emb.num_embeddings - 2
    max_mel_tokens = gpt.mel_pos_embedding.emb.num_embeddings - 2
    
    print("  Models loaded.")
    
    # Load transcripts
    print("\n[3/5] Loading transcripts...")
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
    
    print(f"  Matched {len(matched)} audio-transcript pairs")
    
    if len(matched) == 0:
        print("❌ No matches found!")
        sys.exit(1)
    
    # Process files
    print("\n[4/5] Processing audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Save global conditioning to features dir (used by ALL samples)
    np.save(features_dir / "GLOBAL_condition.npy", global_condition)
    np.save(features_dir / "GLOBAL_emo_vec.npy", global_emo_vec)
    
    manifest_entries = []
    
    with torch.no_grad():
        for audio_path, transcript in tqdm(matched, desc="Processing"):
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
                duration = len(audio) / sr
                
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                
                # Normalize transcript
                prepared_text = normalize_transcript_markers(transcript)
                
                # Resample
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
                
                # Tokenize
                text_tokens = tokenizer.tokenize(prepared_text)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                text_ids_array = np.array(text_ids, dtype=np.int32)
                
                if len(text_ids) > max_text_tokens:
                    warnings.warn(f"Skipping {audio_path.name}: text too long")
                    continue
                
                # Extract semantic codes (STILL per-sample, as this is what we're predicting)
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
                
                # Generate semantic codes
                codes, _ = semantic_codec.quantize(feat)
                if codes.ndim == 2:
                    codes = codes[0]
                codes = codes.cpu().numpy().astype(np.int32)
                
                if codes.shape[0] > max_mel_tokens:
                    warnings.warn(f"Skipping {audio_path.name}: codes too long")
                    continue
                
                # Save features
                # NOTE: We save PER-SAMPLE text_ids and codes, but use GLOBAL conditioning!
                sample_id = audio_path.stem
                np.save(features_dir / f"{sample_id}_text_ids.npy", text_ids_array)
                np.save(features_dir / f"{sample_id}_codes.npy", codes)
                
                manifest_entries.append({
                    "id": sample_id,
                    "text": prepared_text,
                    "audio_path": str(audio_path),
                    "text_ids_path": f"features/{sample_id}_text_ids.npy",
                    "codes_path": f"features/{sample_id}_codes.npy",
                    # KEY DIFFERENCE: All samples point to GLOBAL conditioning!
                    "condition_path": "features/GLOBAL_condition.npy",
                    "emo_vec_path": "features/GLOBAL_emo_vec.npy",
                    "text_len": len(text_ids),
                    "code_len": codes.shape[0],
                    "condition_len": global_condition.shape[0],
                    "duration": float(duration),
                    "sample_type": "pattern_v2",  # Mark as v2 dataset
                })
                
            except Exception as e:
                warnings.warn(f"Failed to process {audio_path.name}: {e}")
    
    print(f"  Processed {len(manifest_entries)} samples")
    
    # Split and save manifests
    print("\n[5/5] Saving manifests...")
    np.random.seed(42)
    indices = np.random.permutation(len(manifest_entries))
    split_idx = int(len(indices) * args.train_split)
    
    train_entries = [manifest_entries[i] for i in indices[:split_idx]]
    val_entries = [manifest_entries[i] for i in indices[split_idx:]]
    
    train_manifest = output_dir / "train_manifest.jsonl"
    val_manifest = output_dir / "val_manifest.jsonl"
    
    with open(train_manifest, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(val_manifest, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Save dataset info
    info = {
        "version": "v2_pattern_conditioned",
        "pattern_conditioning_source": str(args.pattern_conditioning),
        "global_condition_shape": list(global_condition.shape),
        "global_emo_vec_shape": list(global_emo_vec.shape),
        "num_samples": len(manifest_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n  Train: {train_manifest} ({len(train_entries)} samples)")
    print(f"  Val: {val_manifest} ({len(val_entries)} samples)")
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
KEY FEATURE: All samples use GLOBAL conditioning from:
  {args.pattern_conditioning}

This means the model learns:
  "When I see THIS conditioning + any text → add the speaker's patterns"

NEXT STEPS:
===========

1. Train with the prepared dataset:
   python tools/train_gpt_lora.py \\
       --train-manifest {train_manifest} \\
       --val-manifest {val_manifest} \\
       --lora-rank 32 \\
       --lora-alpha 64 \\
       --epochs 30 \\
       --learning-rate 5e-4 \\
       --output-dir training/{args.speaker or 'speaker'}/lora

2. Also extract speaker embeddings (for S2Mel stage):
   python tools/extract_embeddings.py --speaker {args.speaker or 'speaker'}

3. Inference using BOTH pattern conditioning AND speaker embeddings:
   python tools/infer_pattern.py \\
       --speaker {args.speaker or 'speaker'} \\
       --lora-path training/{args.speaker or 'speaker'}/lora/final_checkpoint \\
       --pattern-conditioning {args.pattern_conditioning} \\
       --text "Life finds a way"
""")


if __name__ == "__main__":
    main()