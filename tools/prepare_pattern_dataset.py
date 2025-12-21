#!/usr/bin/env python3
"""
Dataset Preparation for Speaking Pattern Training (e.g., Goldblum-style hesitations)

This script prepares training data specifically designed to capture SPEAKING PATTERNS:
- Pauses and hesitations
- "Uh", "um", "ah" fillers
- Repetitions and restarts  
- Rhythm and pacing

KEY INSIGHT:
============
To train the model to reproduce speaking patterns, we need:

1. VERBATIM transcriptions that include fillers, pauses, etc.
   - Original: "Hello world"
   - Verbatim: "Hello... uh... world"
   
2. OR: Clean transcriptions + pattern markers inferred from audio timing
   - The model learns: for this speaker, add pauses even if not in text

3. Consistent speaker embeddings at inference (from training audio)

TRANSCRIPTION FORMAT:
====================
Support special markers in transcriptions:

  [PAUSE]     - Short pause (100-300ms)
  [LONG]      - Long pause (300-1000ms)  
  [UH]        - "Uh" hesitation filler
  [UM]        - "Um" hesitation filler
  [BREATH]    - Audible breath
  ...         - Ellipsis = natural trailing pause
  
Example verbatim transcription:
  "Well [PAUSE] you know [UH] life... [LONG] life finds a way"

Usage:
    python tools/prepare_pattern_dataset.py \
        --audio-dir training/goldblum/dataset/audio \
        --transcripts training/goldblum/dataset/transcripts_verbatim.csv \
        --output-dir training/goldblum/dataset/processed \
        --detect-pauses  # Auto-detect pauses from audio timing
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
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass  
class PauseMarker:
    """Represents a detected pause in audio."""
    start_time: float
    end_time: float
    duration: float
    pause_type: str  # SHORT, MEDIUM, LONG


def detect_pauses_in_audio(
    audio: np.ndarray, 
    sr: int,
    silence_threshold_db: float = -40,
    min_pause_duration: float = 0.15,
    short_pause_max: float = 0.3,
    medium_pause_max: float = 0.7,
) -> List[PauseMarker]:
    """
    Detect pauses (silence) in audio using energy-based analysis.
    
    Returns list of PauseMarker with timing and classification.
    """
    # Convert to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    
    # Calculate frame-wise energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # RMS energy per frame
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    
    # Find silent frames
    is_silent = rms_db < silence_threshold_db
    
    # Find pause regions (consecutive silent frames)
    pauses = []
    in_pause = False
    pause_start = 0
    
    for i, silent in enumerate(is_silent):
        time = i * hop_length / sr
        
        if silent and not in_pause:
            # Start of pause
            in_pause = True
            pause_start = time
        elif not silent and in_pause:
            # End of pause
            in_pause = False
            pause_end = time
            duration = pause_end - pause_start
            
            if duration >= min_pause_duration:
                # Classify pause type
                if duration < short_pause_max:
                    pause_type = "SHORT"
                elif duration < medium_pause_max:
                    pause_type = "MEDIUM"
                else:
                    pause_type = "LONG"
                
                pauses.append(PauseMarker(
                    start_time=pause_start,
                    end_time=pause_end,
                    duration=duration,
                    pause_type=pause_type
                ))
    
    return pauses


def detect_fillers_in_audio(
    audio: np.ndarray,
    sr: int,
    transcript: str,
) -> List[Tuple[float, str]]:
    """
    Attempt to detect filler words (uh, um) using simple heuristics.
    
    NOTE: This is a simplified detection. For production use, consider:
    - Whisper with word-level timestamps
    - Forced alignment tools
    - Manual annotation
    
    Returns list of (timestamp, filler_type) tuples.
    """
    # This is a placeholder - actual filler detection requires ASR with timestamps
    # For now, return empty list (user should provide verbatim transcripts)
    return []


def insert_pauses_into_transcript(
    transcript: str,
    pauses: List[PauseMarker],
    audio_duration: float,
    words_per_second: float = 2.5,  # Approximate speaking rate
) -> str:
    """
    Insert pause markers into transcript at estimated positions.
    
    This is an approximation - for best results, use verbatim transcriptions.
    """
    words = transcript.split()
    if not words:
        return transcript
    
    # Estimate word timing (very rough)
    word_duration = 1.0 / words_per_second
    
    result_words = []
    current_time = 0.0
    word_idx = 0
    
    # Sort pauses by start time
    sorted_pauses = sorted(pauses, key=lambda p: p.start_time)
    pause_idx = 0
    
    for word in words:
        # Check if there's a pause before this word
        while pause_idx < len(sorted_pauses):
            pause = sorted_pauses[pause_idx]
            if pause.start_time < current_time + word_duration:
                # Insert pause marker
                if pause.pause_type == "LONG":
                    result_words.append("[LONG]")
                elif pause.pause_type == "MEDIUM":
                    result_words.append("[PAUSE]")
                # SHORT pauses might just be natural word boundaries
                pause_idx += 1
            else:
                break
        
        result_words.append(word)
        current_time += word_duration
    
    return " ".join(result_words)


def normalize_transcript_markers(text: str) -> str:
    """
    Normalize various pause/filler notations to standard format.
    """
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


def prepare_training_text(
    transcript: str,
    audio: np.ndarray,
    sr: int,
    detect_pauses: bool = True,
    audio_duration: float = None,
) -> str:
    """
    Prepare transcript for training, optionally detecting and inserting pauses.
    """
    # First normalize any existing markers
    transcript = normalize_transcript_markers(transcript)
    
    # If transcript already has markers, use as-is
    has_markers = any(marker in transcript for marker in ['[PAUSE]', '[LONG]', '[UH]', '[UM]', '...'])
    
    if has_markers:
        return transcript
    
    # Optionally detect and insert pauses
    if detect_pauses:
        pauses = detect_pauses_in_audio(audio, sr)
        if pauses:
            audio_duration = audio_duration or len(audio) / sr
            transcript = insert_pauses_into_transcript(transcript, pauses, audio_duration)
    
    return transcript


def load_transcripts_verbatim(transcript_path: Path) -> Dict[str, str]:
    """
    Load transcriptions, preserving verbatim content including fillers/pauses.
    """
    transcripts = {}
    
    if transcript_path.suffix.lower() == ".csv":
        with open(transcript_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename") or row.get("audio")
                # Support both 'text' and 'verbatim' columns
                text = row.get("verbatim") or row.get("text") or row.get("transcription")
                if filename and text:
                    transcripts[filename] = text  # Don't strip - preserve spacing
    
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
        description="Prepare dataset for speaking pattern training"
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--detect-pauses", action="store_true",
                        help="Auto-detect pauses from audio and insert markers")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*60)
    print("SPEAKING PATTERN DATASET PREPARATION")
    print("="*60)
    print("\nThis tool prepares data to train speaking PATTERNS:")
    print("  - Pauses and hesitations")
    print("  - Filler words (uh, um)")
    print("  - Rhythm and pacing")
    print("\nFor best results, provide VERBATIM transcriptions!")
    print("="*60)
    
    # Import required modules
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
    print(f"\nUsing device: {device}")
    
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
    
    # GPT model
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
    
    print("Models loaded.\n")
    
    # Load transcripts
    transcripts = load_transcripts_verbatim(args.transcripts)
    print(f"Loaded {len(transcripts)} transcriptions")
    
    # Check for verbatim markers
    verbatim_count = sum(1 for t in transcripts.values() if any(m in t for m in ['[', '...', 'uh', 'um', 'Uh', 'Um']))
    if verbatim_count == 0 and not args.detect_pauses:
        print("\n⚠️  WARNING: No verbatim markers found in transcriptions!")
        print("   For best pattern training, provide transcriptions like:")
        print('   "Well [PAUSE] you know [UH] life... life finds a way"')
        print("   Or use --detect-pauses to auto-detect from audio")
    else:
        print(f"   {verbatim_count} transcripts have verbatim markers")
    
    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(args.audio_dir.glob(f"*{ext}"))
        audio_files.extend(args.audio_dir.glob(f"*{ext.upper()}"))
    audio_files = sorted(audio_files)
    print(f"Found {len(audio_files)} audio files")
    
    # Match audio with transcripts
    matched = []
    for audio_path in audio_files:
        for candidate in [audio_path.name, audio_path.stem]:
            if candidate in transcripts:
                matched.append((audio_path, transcripts[candidate]))
                break
    
    print(f"Matched {len(matched)} audio-transcript pairs")
    
    if len(matched) == 0:
        print("ERROR: No matches found!")
        sys.exit(1)
    
    # Process files
    print("\nProcessing audio files...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = args.output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    manifest_entries = []
    pattern_stats = {"with_markers": 0, "pauses_detected": 0, "total_pauses": 0}
    
    max_text_tokens = gpt.text_pos_embedding.emb.num_embeddings - 2
    max_mel_tokens = gpt.mel_pos_embedding.emb.num_embeddings - 2
    
    with torch.no_grad():
        for audio_path, original_transcript in tqdm(matched, desc="Processing"):
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
                duration = len(audio) / sr
                
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                
                # Prepare transcript with pattern markers
                prepared_text = prepare_training_text(
                    original_transcript,
                    audio,
                    sr,
                    detect_pauses=args.detect_pauses,
                    audio_duration=duration,
                )
                
                # Track stats
                if any(m in prepared_text for m in ['[PAUSE]', '[LONG]', '[UH]', '[UM]', '...']):
                    pattern_stats["with_markers"] += 1
                
                if args.detect_pauses:
                    pauses = detect_pauses_in_audio(audio, sr)
                    if pauses:
                        pattern_stats["pauses_detected"] += 1
                        pattern_stats["total_pauses"] += len(pauses)
                
                # Resample
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_16k_tensor = torch.from_numpy(audio_16k).unsqueeze(0)
                
                # Tokenize with pattern markers
                text_tokens = tokenizer.tokenize(prepared_text)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                text_ids_array = np.array(text_ids, dtype=np.int32)
                
                if len(text_ids) > max_text_tokens:
                    warnings.warn(f"Skipping {audio_path.name}: text too long")
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
                codes = codes.cpu().numpy().astype(np.int32)
                
                if codes.shape[0] > max_mel_tokens:
                    warnings.warn(f"Skipping {audio_path.name}: codes too long")
                    continue
                
                # Get conditioning
                cond_lengths = torch.tensor([feat.shape[1]], device=device)
                conditioning_latent = gpt.get_conditioning(feat.transpose(1, 2), cond_lengths)
                conditioning_latent = conditioning_latent.squeeze(0).cpu().numpy().astype(np.float32)
                
                emo_latent = gpt.get_emo_conditioning(feat.transpose(1, 2), cond_lengths)
                emo_vec = gpt.emovec_layer(emo_latent)
                emo_vec = gpt.emo_layer(emo_vec)
                emo_vec = emo_vec.squeeze(0).cpu().numpy().astype(np.float32)
                
                # Save features
                sample_id = audio_path.stem
                np.save(features_dir / f"{sample_id}_text_ids.npy", text_ids_array)
                np.save(features_dir / f"{sample_id}_codes.npy", codes)
                np.save(features_dir / f"{sample_id}_condition.npy", conditioning_latent)
                np.save(features_dir / f"{sample_id}_emo_vec.npy", emo_vec)
                
                manifest_entries.append({
                    "id": sample_id,
                    "text": prepared_text,
                    "original_text": original_transcript,
                    "audio_path": str(audio_path),
                    "text_ids_path": f"features/{sample_id}_text_ids.npy",
                    "codes_path": f"features/{sample_id}_codes.npy",
                    "condition_path": f"features/{sample_id}_condition.npy",
                    "emo_vec_path": f"features/{sample_id}_emo_vec.npy",
                    "text_len": len(text_ids),
                    "code_len": codes.shape[0],
                    "condition_len": conditioning_latent.shape[0],
                    "duration": float(duration),
                    "sample_type": "pattern",
                })
                
            except Exception as e:
                warnings.warn(f"Failed to process {audio_path.name}: {e}")
    
    print(f"\nProcessed {len(manifest_entries)} samples")
    print(f"\nPattern Statistics:")
    print(f"  Samples with markers: {pattern_stats['with_markers']}")
    if args.detect_pauses:
        print(f"  Samples with detected pauses: {pattern_stats['pauses_detected']}")
        print(f"  Total pauses detected: {pattern_stats['total_pauses']}")
    
    # Split and save manifests
    np.random.seed(42)
    indices = np.random.permutation(len(manifest_entries))
    split_idx = int(len(indices) * args.train_split)
    
    train_entries = [manifest_entries[i] for i in indices[:split_idx]]
    val_entries = [manifest_entries[i] for i in indices[split_idx:]]
    
    train_manifest = args.output_dir / "train_manifest.jsonl"
    val_manifest = args.output_dir / "val_manifest.jsonl"
    
    with open(train_manifest, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(val_manifest, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\nSaved manifests:")
    print(f"  Train: {train_manifest} ({len(train_entries)} samples)")
    print(f"  Val: {val_manifest} ({len(val_entries)} samples)")
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR PATTERN TRAINING")
    print("="*60)
    print("""
1. Train with higher LoRA rank for pattern capacity:
   python tools/train_gpt_lora.py \\
       --train-manifest {train} \\
       --val-manifest {val} \\
       --lora-rank 32 \\
       --lora-alpha 64 \\
       --epochs 30 \\
       --learning-rate 5e-4 \\
       --output-dir training/goldblum/lora

2. Extract speaker embeddings from training audio:
   python tools/extract_embeddings.py --speaker goldblum

3. Test with the trained speaker's embeddings:
   python tools/infer.py --speaker goldblum \\
       --lora-path training/goldblum/lora/final_checkpoint \\
       --text "Life finds a way"

IMPORTANT: The model learns patterns from the conditioning!
Using different reference audio at inference will lose the patterns.
Always use embeddings from the training audio for pattern reproduction.
""".format(train=train_manifest, val=val_manifest))


if __name__ == "__main__":
    main()