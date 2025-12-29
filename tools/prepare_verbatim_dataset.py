#!/usr/bin/env python3
"""
Verbatim Dataset Preparation for IndexTTS2 LoRA Training

This prepares datasets using VERBATIM transcriptions (from Parakeet/detailed transcribers)
to train the model to reproduce stutters, hesitations, and speech imperfections.

AUTOMATIC: If no transcripts.csv exists, this will run dual transcription:
- FastWhisper: "I was going to" (clean - removes stutters)
- Parakeet/Nvidia: "I I I was going going to" (verbatim - preserves stutters)

The Key Insight:
================
By training on verbatim text → audio codes, the model learns:
  "I I I was" → [codes that produce stuttered speech]

At inference, you can then input verbatim-style text to get stuttered output!

Usage:
    # Full automatic pipeline (transcribes + prepares)
    python tools/prepare_verbatim_dataset.py --speaker ozzy
    
    # Skip transcription if you have transcripts
    python tools/prepare_verbatim_dataset.py --speaker ozzy --transcripts transcripts.csv
    
    # Manual paths
    python tools/prepare_verbatim_dataset.py \
        --audio-dir training/ozzy/dataset/audio \
        --transcripts training/ozzy/dataset/transcripts.csv \
        --output-dir training/ozzy/dataset/processed_verbatim
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
class VerbatimFeatures:
    """Features extracted for verbatim training."""
    has_word_repetition: bool
    has_filler_words: bool
    has_hesitation: bool
    repetition_count: int
    filler_count: int
    stutter_positions: List[int]  # Token positions that are stutters
    

class VerbatimAnalyzer:
    """Analyze verbatim transcripts to identify stutter patterns."""
    
    # Patterns that indicate stutters/repetitions
    REPETITION_PATTERNS = [
        r'\b(\w+)\s+\1\b',           # "I I", "go go"
        r'\b(\w+)\s+\1\s+\1\b',      # "I I I"
        r'\b(\w{1,3})-\1',           # "s-s-so"
        r'\.{2,}',                   # Hesitation dots "..."
    ]
    
    # Filler words (both English and markers)
    FILLER_PATTERNS = [
        r'\b(uh+|uhh+)\b',
        r'\b(um+|umm+)\b',
        r'\b(er+|err+)\b',
        r'\b(ah+|ahh+)\b',
        r'\b(hmm+)\b',
        r'\b(like)\b(?=\s*(,|\.|\s+\w))',  # "like" as filler
        r'\b(you know)\b',
        r'\b(I mean)\b',
    ]
    
    def analyze(self, verbatim_text: str) -> VerbatimFeatures:
        """Analyze verbatim text for stutter features."""
        
        repetition_count = 0
        filler_count = 0
        stutter_positions = []
        
        # Check repetitions
        for pattern in self.REPETITION_PATTERNS:
            matches = list(re.finditer(pattern, verbatim_text, re.IGNORECASE))
            repetition_count += len(matches)
            for m in matches:
                stutter_positions.append(m.start())
        
        # Check fillers
        for pattern in self.FILLER_PATTERNS:
            matches = list(re.finditer(pattern, verbatim_text, re.IGNORECASE))
            filler_count += len(matches)
            for m in matches:
                stutter_positions.append(m.start())
        
        return VerbatimFeatures(
            has_word_repetition=repetition_count > 0,
            has_filler_words=filler_count > 0,
            has_hesitation='...' in verbatim_text or '..' in verbatim_text,
            repetition_count=repetition_count,
            filler_count=filler_count,
            stutter_positions=sorted(set(stutter_positions)),
        )


def load_dual_transcripts(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    """
    Load transcripts with both clean and verbatim versions.
    
    Returns:
        tuple: (transcripts dict, stats dict)
        - transcripts: filename -> {'clean': ..., 'verbatim': ...}
        - stats: counts of various conditions
    """
    transcripts = {}
    stats = {
        "total": 0,
        "missing_verbatim": 0,
        "missing_clean": 0,
        "identical_texts": 0,
    }
    
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            # Detect column names
            filename_col = None
            clean_col = None
            verbatim_col = None
            
            for col in columns:
                col_lower = col.lower()
                if col_lower in ('filename', 'file', 'audio', 'name'):
                    filename_col = col
                elif col_lower in ('fastwhisper', 'whisper', 'clean', 'text', 'transcription'):
                    clean_col = col
                elif col_lower in ('parakeet', 'verbatim', 'detailed', 'nvidia'):
                    verbatim_col = col
            
            if not filename_col:
                raise ValueError(f"No filename column found. Columns: {columns}")
            
            print(f"  Using columns: filename='{filename_col}', clean='{clean_col}', verbatim='{verbatim_col}'")
            
            for row in reader:
                filename = row.get(filename_col, "")
                if not filename:
                    continue
                
                stats["total"] += 1
                    
                clean = row.get(clean_col, "") if clean_col else ""
                verbatim = row.get(verbatim_col, "") if verbatim_col else ""
                
                original_verbatim = verbatim
                original_clean = clean
                
                # Fall back: if only one transcription available, use it for both
                if not verbatim and clean:
                    verbatim = clean
                    stats["missing_verbatim"] += 1
                elif not clean and verbatim:
                    clean = verbatim
                    stats["missing_clean"] += 1
                
                # Check if they're identical (indicates transcription issue)
                if verbatim and clean and verbatim.strip() == clean.strip():
                    # Only count as identical if BOTH were empty originally (fallback used)
                    # OR if they were genuinely identical from separate transcriptions
                    if not original_verbatim or not original_clean:
                        stats["identical_texts"] += 1
                
                if verbatim:  # Need at least verbatim for training
                    transcripts[filename] = {
                        'clean': clean.strip(),
                        'verbatim': verbatim.strip(),
                    }
    
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    filename = item.get("filename") or item.get("audio") or item.get("name")
                    clean = item.get("fastwhisper") or item.get("clean") or item.get("text") or ""
                    verbatim = item.get("parakeet") or item.get("verbatim") or item.get("detailed") or ""
                    
                    stats["total"] += 1
                    
                    original_verbatim = verbatim
                    
                    if not verbatim and clean:
                        verbatim = clean
                        stats["missing_verbatim"] += 1
                    elif not clean and verbatim:
                        clean = verbatim
                        stats["missing_clean"] += 1
                    
                    if verbatim and clean and verbatim.strip() == clean.strip():
                        if not original_verbatim:
                            stats["identical_texts"] += 1
                    
                    if filename and verbatim:
                        transcripts[filename] = {'clean': clean, 'verbatim': verbatim}
    
    return transcripts, stats


def normalize_verbatim_text(text: str) -> str:
    """
    Normalize verbatim text while PRESERVING stutter patterns.
    
    Unlike clean normalization, we keep repetitions and fillers!
    """
    # Normalize whitespace but keep structure
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Standardize filler spellings (optional - can be customized)
    # text = re.sub(r'\b(uhh+)\b', 'uh', text, flags=re.IGNORECASE)
    # text = re.sub(r'\b(umm+)\b', 'um', text, flags=re.IGNORECASE)
    
    # Keep punctuation that indicates hesitation
    # "..." represents natural pause/hesitation - keep it!
    
    return text


def run_dual_transcription(audio_dir: Path, output_csv: Path, device: str = "cuda") -> Path:
    """
    Run dual transcription (FastWhisper + Parakeet) if transcripts don't exist.
    Returns path to transcripts CSV.
    """
    print("\n" + "=" * 60)
    print("RUNNING AUTOMATIC TRANSCRIPTION")
    print("=" * 60)
    print("\nNo transcripts file found. Running dual transcription:")
    print("  - FastWhisper: For clean text")
    print("  - Parakeet: For verbatim text (with stutters)")
    
    try:
        from tools.transcribe_dual import DualTranscriber, find_audio_files
    except ImportError:
        from transcribe_dual import DualTranscriber, find_audio_files
    
    # Find audio files
    audio_files = find_audio_files(audio_dir)
    print(f"\nFound {len(audio_files)} audio files to transcribe")
    
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    # Initialize transcriber (normalize device for faster-whisper)
    transcriber_device = device
    if transcriber_device.startswith("cuda:"):
        transcriber_device = "cuda"  # ctranslate2 doesn't support device index
    transcriber = DualTranscriber(device=transcriber_device)
    
    # Transcribe
    results = []
    for audio_path in tqdm(audio_files, desc="Transcribing"):
        transcripts = transcriber.transcribe(str(audio_path))
        results.append({
            "filename": audio_path.name,
            "fastwhisper": transcripts["fastwhisper"],
            "parakeet": transcripts["parakeet"],
        })
    
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "fastwhisper", "parakeet"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Transcripts saved to: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Prepare verbatim dataset for stutter-aware training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", help="Speaker name (uses default paths)")
    parser.add_argument("--audio-dir", type=Path, help="Audio directory")
    parser.add_argument("--transcripts", type=Path, help="Transcripts CSV/JSON with clean + verbatim columns")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    # Transcription options
    parser.add_argument("--skip-transcription", action="store_true",
                        help="Skip automatic transcription even if no transcripts found")
    
    # Processing options
    parser.add_argument("--use-verbatim", action="store_true", default=True,
                        help="Use verbatim text for training (default: True)")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--train-split", type=float, default=0.9)
    
    # Model paths
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Resolve device early for transcription
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Resolve paths
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker / "dataset"
        audio_dir = args.audio_dir or speaker_dir / "audio"
        
        # Try multiple transcript filenames
        if args.transcripts:
            transcripts_path = args.transcripts
        else:
            candidates = [
                speaker_dir / "transcripts.csv",
                speaker_dir / "transcripts_verbatim.csv",
                speaker_dir / "transcripts_dual.csv",
            ]
            transcripts_path = None
            for c in candidates:
                if c.exists():
                    transcripts_path = c
                    break
            
            # No transcripts found - run automatic transcription!
            if transcripts_path is None:
                if args.skip_transcription:
                    print(f"❌ No transcript file found. Tried: {candidates}")
                    print("   Remove --skip-transcription to auto-transcribe")
                    sys.exit(1)
                else:
                    # Auto-transcribe
                    default_csv = speaker_dir / "transcripts.csv"
                    transcripts_path = run_dual_transcription(audio_dir, default_csv, device)
        
        output_dir = args.output_dir or speaker_dir / "processed_verbatim"
    else:
        if not args.audio_dir or not args.output_dir:
            parser.error("--speaker or both --audio-dir and --output-dir required")
        audio_dir = args.audio_dir
        output_dir = args.output_dir
        
        # Handle transcripts
        if args.transcripts:
            transcripts_path = args.transcripts
        else:
            # Try to find or auto-generate
            default_csv = audio_dir.parent / "transcripts.csv"
            if default_csv.exists():
                transcripts_path = default_csv
            elif not args.skip_transcription:
                transcripts_path = run_dual_transcription(audio_dir, default_csv, device)
            else:
                print(f"❌ No transcripts and --skip-transcription set")
                sys.exit(1)
    
    print("\n" + "=" * 60)
    print("VERBATIM DATASET PREPARATION")
    print("=" * 60)
    print(f"\nAudio: {audio_dir}")
    print(f"Transcripts: {transcripts_path}")
    print(f"Output: {output_dir}")
    print(f"Training text: {'VERBATIM' if args.use_verbatim else 'CLEAN'}")
    print(f"Device: {device}")
    
    # Validate
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    if not transcripts_path.exists():
        print(f"❌ Transcripts not found: {transcripts_path}")
        sys.exit(1)
    
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
    
    # GPT for conditioning extraction
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
    
    max_text_tokens = 120
    max_mel_tokens = 500
    
    # Verbatim analyzer
    analyzer = VerbatimAnalyzer()
    
    # Load transcripts
    print("\n[2/5] Loading transcripts...")
    transcripts, transcript_stats = load_dual_transcripts(transcripts_path)
    print(f"  Loaded {len(transcripts)} transcriptions")
    
    # Warn about transcription issues
    if transcript_stats["missing_verbatim"] > 0:
        pct = 100 * transcript_stats["missing_verbatim"] / max(1, transcript_stats["total"])
        print(f"\n  ⚠️  WARNING: {transcript_stats['missing_verbatim']}/{transcript_stats['total']} ({pct:.1f}%) samples missing verbatim text!")
        print(f"     This means Parakeet transcription likely failed.")
        print(f"     Using FastWhisper text as fallback (won't have stutters).")
        print(f"")
        print(f"     To fix: Install NeMo and re-run transcription:")
        print(f"       pip install nemo_toolkit[asr]")
        print(f"       python tools/transcribe_dual.py --speaker {args.speaker or 'SPEAKER'}")
        print(f"")
        
    if transcript_stats["identical_texts"] > 0:
        pct = 100 * transcript_stats["identical_texts"] / max(1, transcript_stats["total"])
        if pct > 50:  # More than half are identical = problem
            print(f"\n  ⚠️  WARNING: {transcript_stats['identical_texts']}/{transcript_stats['total']} ({pct:.1f}%) samples have IDENTICAL clean/verbatim text!")
            print(f"     This defeats the purpose of verbatim training.")
            print(f"     Your 'parakeet' column appears to be empty or identical to 'fastwhisper'.")
            print(f"")
    
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
        print("\nMake sure your transcript CSV has a 'filename' column matching your audio files.")
        sys.exit(1)
    
    # Process
    print("\n[3/5] Processing audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    manifest_entries = []
    verbatim_stats = {
        "samples_with_repetitions": 0,
        "samples_with_fillers": 0,
        "samples_with_hesitations": 0,
        "total_repetitions": 0,
        "total_fillers": 0,
    }
    
    with torch.no_grad():
        for audio_path, transcript_data in tqdm(matched, desc="Processing"):
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
                duration = len(audio) / sr
                
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                
                # Get text (verbatim or clean based on args)
                if args.use_verbatim:
                    text = transcript_data['verbatim']
                else:
                    text = transcript_data['clean']
                
                if not text:
                    continue
                
                # Normalize (preserves stutters)  
                prepared_text = normalize_verbatim_text(text)
                
                # Analyze verbatim features
                vf = analyzer.analyze(prepared_text)
                
                # Update stats
                if vf.has_word_repetition:
                    verbatim_stats["samples_with_repetitions"] += 1
                    verbatim_stats["total_repetitions"] += vf.repetition_count
                if vf.has_filler_words:
                    verbatim_stats["samples_with_fillers"] += 1
                    verbatim_stats["total_fillers"] += vf.filler_count
                if vf.has_hesitation:
                    verbatim_stats["samples_with_hesitations"] += 1
                
                # Resample audio
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
                
                # Tokenize verbatim text
                text_tokens = tokenizer.tokenize(prepared_text)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                text_ids_array = np.array(text_ids, dtype=np.int32)
                
                if len(text_ids) > max_text_tokens:
                    continue
                
                # Extract semantic features
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
                codes_np = codes.cpu().numpy().astype(np.int32)
                
                if codes_np.shape[0] > max_mel_tokens:
                    continue
                
                # Extract GPT conditioning
                cond_lengths = torch.tensor([feat.shape[1]], device=device)
                gpt_cond = gpt.get_conditioning(feat.transpose(1, 2), cond_lengths)
                emo_cond = gpt.get_emo_conditioning(feat.transpose(1, 2), cond_lengths)
                emo_vec = gpt.emovec_layer(emo_cond)
                emo_vec = gpt.emo_layer(emo_vec)
                
                condition = gpt_cond.squeeze(0).cpu().numpy().astype(np.float32)
                emo_vec_np = emo_vec.squeeze(0).cpu().numpy().astype(np.float32)
                
                # Save features
                sample_id = audio_path.stem
                np.save(features_dir / f"{sample_id}_text_ids.npy", text_ids_array)
                np.save(features_dir / f"{sample_id}_codes.npy", codes_np)
                np.save(features_dir / f"{sample_id}_condition.npy", condition)
                np.save(features_dir / f"{sample_id}_emo_vec.npy", emo_vec_np)
                
                # Save verbatim features
                vf_dict = asdict(vf)
                np.save(features_dir / f"{sample_id}_verbatim_features.npy", vf_dict)
                
                manifest_entries.append({
                    "id": sample_id,
                    "text_verbatim": prepared_text,
                    "text_clean": transcript_data['clean'],
                    "audio_path": str(audio_path),
                    "text_ids_path": f"features/{sample_id}_text_ids.npy",
                    "codes_path": f"features/{sample_id}_codes.npy",
                    "condition_path": f"features/{sample_id}_condition.npy",
                    "emo_vec_path": f"features/{sample_id}_emo_vec.npy",
                    "verbatim_features_path": f"features/{sample_id}_verbatim_features.npy",
                    "text_len": len(text_ids),
                    "code_len": codes_np.shape[0],
                    "condition_len": condition.shape[0],
                    "duration": float(duration),
                    "sample_type": "verbatim",
                    # Verbatim summary
                    "has_repetitions": vf.has_word_repetition,
                    "has_fillers": vf.has_filler_words,
                    "has_hesitations": vf.has_hesitation,
                    "repetition_count": vf.repetition_count,
                    "filler_count": vf.filler_count,
                })
                
            except Exception as e:
                warnings.warn(f"Failed to process {audio_path.name}: {e}")
    
    print(f"  Processed {len(manifest_entries)} samples")
    
    # Verbatim statistics
    print(f"\n[4/5] Verbatim statistics:")
    print(f"  Samples with word repetitions: {verbatim_stats['samples_with_repetitions']}/{len(manifest_entries)}")
    print(f"  Samples with filler words: {verbatim_stats['samples_with_fillers']}/{len(manifest_entries)}")
    print(f"  Samples with hesitations: {verbatim_stats['samples_with_hesitations']}/{len(manifest_entries)}")
    print(f"  Total repetitions: {verbatim_stats['total_repetitions']}")
    print(f"  Total fillers: {verbatim_stats['total_fillers']}")
    
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
        "version": "verbatim_v1",
        "total_samples": len(manifest_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "uses_verbatim_text": args.use_verbatim,
        "verbatim_stats": verbatim_stats,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n  Train: {train_path} ({len(train_entries)} samples)")
    print(f"  Val: {val_path} ({len(val_entries)} samples)")
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
This dataset uses VERBATIM transcriptions for training!

KEY INSIGHT:
============
Your audio contains stutters like "I I I was going going to"
Now your text input ALSO contains those stutters
The model will learn: stuttered text → stuttered speech

NEXT STEPS:
===========
1. Train with verbatim LoRA:

    python tools/train_verbatim_lora.py \\
        --speaker {args.speaker or 'SPEAKER'} \\
        --epochs 20 \\
        --lora-rank 16

2. Inference with stutters:

    # Input text WITH stutters → get stuttered speech!
    python tools/infer_with_verbatim.py \\
        --speaker {args.speaker or 'SPEAKER'} \\
        --text "I I I was going going to..."

The model learns the DIRECT mapping from stuttered text to stuttered audio!
""")


if __name__ == "__main__":
    main()