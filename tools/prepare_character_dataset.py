#!/usr/bin/env python3
"""Prepare a character-LoRA training manifest.

Outputs per-sample tensors that the new character LoRA trainer consumes:
    {id, text_ids, codes, condition, emo_vec, stutter_mask}

Design:
  - Input text = CLEAN (Whisper / fastwhisper). What the user / LLM would type
    at inference.
  - Target codes = mel tokens extracted from the actual audio. The audio has
    all the speaker's stutters/pauses/fillers baked in.
  - stutter_mask = boolean tensor over mel-token positions; True where the
    audio is in a "stutter / filler / repetition" region. The trainer uses
    this to weight the loss aggressively in those regions.

Stutter detection:
  - We diff clean (Whisper) vs verbatim (Parakeet) word-lists. Verbatim
    tokens that have no clean counterpart (insertions) are stutter / filler
    candidates: filler words ("uh", "um", "hmm", "er"), single-letter
    fragments ("b", "h"), and repetitions of the prior word.
  - We map detected stutter spans to mel-token positions either via Whisper
    word-level timestamps (when re-transcribing with timestamps) or via
    uniform spacing as a fallback. The mask is padded ±N frames so the loss
    has some surrounding context.

The output directory layout:
    training/<speaker>/character_dataset/
        manifest.jsonl
        features/<id>_text_ids.npy
        features/<id>_codes.npy
        features/<id>_condition.npy
        features/<id>_emo_vec.npy
        features/<id>_stutter_mask.npy

Usage:
    python tools/prepare_character_dataset.py --speaker ozzyv5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tokens that count as fillers regardless of position
FILLER_WORDS = {
    "uh", "um", "umm", "uhh", "uhm", "hmm", "hm", "er", "erm",
    "ah", "eh", "oh", "mm", "mhm", "huh",
}
# Punctuation we strip when comparing words
_WORD_STRIP = re.compile(r"[^\w']+", re.UNICODE)


def _norm_word(w: str) -> str:
    return _WORD_STRIP.sub("", w).lower().strip()


def _words(text: str) -> List[str]:
    return [_norm_word(t) for t in text.split() if _norm_word(t)]


@dataclass
class StutterSpan:
    """A region in the verbatim transcript that should be loss-upweighted."""
    word_start: int  # index in verbatim words
    word_end: int    # exclusive
    reason: str      # "filler" | "repetition" | "fragment"


def detect_stutter_spans(clean: str, verbatim: str) -> List[StutterSpan]:
    """Find stutter / filler / fragment spans in verbatim text.

    Strategy:
      1. Align clean and verbatim word-lists with SequenceMatcher.
      2. Any verbatim words that don't match clean are stutter candidates.
      3. Also flag filler words and obvious fragments anywhere.
    """
    clean_w = _words(clean)
    verb_w = _words(verbatim)
    if not verb_w:
        return []

    spans: List[StutterSpan] = []

    # 1. Sequence-align to find insertions in verbatim that aren't in clean.
    sm = SequenceMatcher(a=clean_w, b=verb_w, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace") and j2 > j1:
            # Inspect the inserted run to figure out why.
            run = verb_w[j1:j2]
            reason = "filler" if any(w in FILLER_WORDS for w in run) else "repetition"
            if any(len(w) <= 1 for w in run):
                reason = "fragment"
            spans.append(StutterSpan(j1, j2, reason))

    # 2. Standalone filler words even when they "match" (rare but possible
    #    if both transcripts kept them). Catches "uh" that survived Whisper.
    for i, w in enumerate(verb_w):
        if w in FILLER_WORDS:
            spans.append(StutterSpan(i, i + 1, "filler"))

    # 3. Adjacent repetitions in verbatim itself ("I I I"). Even when clean
    #    has none of them, sequence-align catches this — but we also catch
    #    longer streaks reliably here.
    i = 0
    while i < len(verb_w) - 1:
        run_end = i + 1
        while run_end < len(verb_w) and verb_w[run_end] == verb_w[i]:
            run_end += 1
        if run_end - i >= 2:
            spans.append(StutterSpan(i, run_end, "repetition"))
        i = run_end

    # Merge overlapping / adjacent spans
    spans.sort(key=lambda s: (s.word_start, s.word_end))
    merged: List[StutterSpan] = []
    for s in spans:
        if merged and s.word_start <= merged[-1].word_end:
            merged[-1] = StutterSpan(
                merged[-1].word_start,
                max(merged[-1].word_end, s.word_end),
                merged[-1].reason,
            )
        else:
            merged.append(s)
    return merged


def word_spans_to_token_mask(
    spans: List[StutterSpan],
    n_words: int,
    n_codes: int,
    word_timestamps: Optional[List[Tuple[float, float]]],
    audio_duration: float,
    pad_tokens: int = 8,
) -> np.ndarray:
    """Project verbatim-word spans to a boolean mask over mel-token positions.

    Uses Whisper word timestamps when available (`word_timestamps`); otherwise
    falls back to uniform spacing across the utterance. Either way the mask
    is padded by ±`pad_tokens` so the loss landscape is smoother.
    """
    mask = np.zeros(n_codes, dtype=np.bool_)
    if n_words <= 0 or n_codes <= 0:
        return mask

    code_rate = n_codes / max(audio_duration, 1e-6)  # tokens / second

    for span in spans:
        if word_timestamps and span.word_start < len(word_timestamps):
            ws_idx = min(span.word_start, len(word_timestamps) - 1)
            we_idx = min(span.word_end - 1, len(word_timestamps) - 1)
            t0 = word_timestamps[ws_idx][0]
            t1 = word_timestamps[we_idx][1]
        else:
            # Uniform spacing across the verbatim word stream
            t0 = span.word_start / n_words * audio_duration
            t1 = span.word_end / n_words * audio_duration

        i0 = max(0, int(t0 * code_rate) - pad_tokens)
        i1 = min(n_codes, int(t1 * code_rate) + pad_tokens)
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speaker", "-s", required=True)
    p.add_argument("--audio-dir", type=Path, default=None)
    p.add_argument("--clean-csv", type=Path, default=None,
                   help="CSV with clean text per audio. Defaults to dataset/transcripts.csv "
                        "or dataset/transcripts_dual.csv (fastwhisper column).")
    p.add_argument("--verbatim-csv", type=Path, default=None,
                   help="CSV with verbatim text per audio. Defaults to dataset/transcripts_verbatim.csv "
                        "or dataset/transcripts_dual.csv (parakeet column).")
    p.add_argument("--word-timestamps-jsonl", type=Path, default=None,
                   help="Optional JSONL: {filename, words:[{w,t0,t1}]}. Improves stutter alignment.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "checkpoints" / "config.yaml")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=15.0)
    p.add_argument("--max-text-tokens", type=int, default=200)
    p.add_argument("--max-mel-tokens", type=int, default=600)
    p.add_argument("--pad-tokens", type=int, default=8,
                   help="Mel-token padding around each stutter span (smooths loss).")
    p.add_argument("--reference-audio", type=Path, default=None,
                   help="If set, extract global condition/emo from this single clip and "
                        "reuse for every sample (recommended for single-speaker training).")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: cap N samples.")
    return p.parse_args()


def load_transcript_csv(path: Path, text_column: Optional[str] = None) -> Dict[str, str]:
    """Load filename → text from a CSV. Auto-detects the text column when not given."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename") or row.get("file") or row.get("audio")
            if not fn:
                continue
            if text_column and text_column in row:
                text = row[text_column]
            else:
                # auto-detect preferred order
                for col in ("verbatim", "parakeet", "fastwhisper", "fast_whisper", "text", "transcript"):
                    if col in row and row[col]:
                        text = row[col]
                        break
                else:
                    text = ""
            if text and text.strip():
                # store by basename and stem so both lookup styles work
                out[fn] = text.strip()
                out[Path(fn).stem] = text.strip()
                out[Path(fn).name] = text.strip()
    return out


def load_word_timestamps(path: Optional[Path]) -> Dict[str, List[Tuple[float, float]]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, List[Tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fn = row.get("filename") or row.get("file")
            if not fn:
                continue
            spans = [(float(w["t0"]), float(w["t1"])) for w in row.get("words", []) if "t0" in w and "t1" in w]
            for key in (fn, Path(fn).stem, Path(fn).name):
                out[key] = spans
    return out


def main():
    args = parse_args()

    # Lazy imports — they pull heavy deps
    from huggingface_hub import hf_hub_download
    from transformers import SeamlessM4TFeatureExtractor
    import safetensors.torch
    from indextts.utils.front import TextTokenizer, TextNormalizer
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    if not speaker_dir.exists():
        print(f"❌ speaker not found: {speaker_dir}", file=sys.stderr)
        sys.exit(2)

    audio_dir = args.audio_dir or (speaker_dir / "dataset" / "audio")
    if not audio_dir.exists():
        print(f"❌ audio dir not found: {audio_dir}", file=sys.stderr)
        sys.exit(2)

    output_dir = args.output_dir or (speaker_dir / "character_dataset")
    features_dir = output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    # Resolve transcript sources. Fall back chain:
    # 1. explicit --clean-csv / --verbatim-csv
    # 2. transcripts.csv (clean) + transcripts_verbatim.csv (verbatim)
    # 3. transcripts_dual.csv (single file with fastwhisper + parakeet columns)
    ds_root = speaker_dir / "dataset"
    clean_csv = args.clean_csv or next(
        (p for p in [ds_root / "transcripts.csv", ds_root / "transcripts_dual.csv"] if p.exists()),
        None,
    )
    verbatim_csv = args.verbatim_csv or next(
        (p for p in [ds_root / "transcripts_verbatim.csv", ds_root / "transcripts_dual.csv"] if p.exists()),
        None,
    )
    if clean_csv is None or verbatim_csv is None:
        print(f"❌ need clean and verbatim transcripts under {ds_root}/", file=sys.stderr)
        print(f"   expected: transcripts.csv + transcripts_verbatim.csv  OR  transcripts_dual.csv", file=sys.stderr)
        sys.exit(2)

    clean_text_by_file = load_transcript_csv(clean_csv, text_column="fastwhisper" if "dual" in clean_csv.name else None)
    if not clean_text_by_file and "dual" not in clean_csv.name:
        # Try auto-detect text column
        clean_text_by_file = load_transcript_csv(clean_csv)
    verbatim_text_by_file = load_transcript_csv(verbatim_csv, text_column="parakeet" if "dual" in verbatim_csv.name else None)
    if not verbatim_text_by_file and "dual" not in verbatim_csv.name:
        verbatim_text_by_file = load_transcript_csv(verbatim_csv)

    word_ts_by_file = load_word_timestamps(args.word_timestamps_jsonl)

    print(f"[prep] clean={clean_csv.name}  verbatim={verbatim_csv.name}  "
          f"word_ts={'yes' if word_ts_by_file else 'no'}")
    print(f"[prep] clean rows={len(clean_text_by_file)//3}  verbatim rows={len(verbatim_text_by_file)//3}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    bpe_path = args.model_dir / cfg.dataset["bpe_model"]
    tokenizer = TextTokenizer(str(bpe_path), TextNormalizer())

    print(f"[prep] loading W2V-BERT + semantic codec...")
    extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        str(args.model_dir / cfg.w2v_stat)
    )
    semantic_model = semantic_model.to(device).eval()
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)

    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    codec_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, codec_ckpt)
    semantic_codec = semantic_codec.to(device).eval()

    print(f"[prep] loading GPT for conditioning extraction...")
    gpt_path = args.model_dir / cfg.gpt_checkpoint
    raw_state = torch.load(gpt_path, map_location="cpu").get("model", {})
    if "mel_pos_embedding.emb.weight" in raw_state:
        ckpt_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != ckpt_dim:
            cfg.gpt.model_dim = ckpt_dim
    del raw_state
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, str(gpt_path))
    gpt = gpt.to(device).eval()

    # Pre-compute a single global conditioning so the LoRA trains against a
    # stable speaker prompt — same vector every batch. This is what the
    # existing verbatim trainer recommends and it dramatically reduces
    # learning instability on small datasets.
    global_condition: Optional[np.ndarray] = None
    global_emo_vec: Optional[np.ndarray] = None
    if args.reference_audio is not None:
        ref_path = args.reference_audio
        if not ref_path.is_absolute():
            ref_path = audio_dir / ref_path
        if not ref_path.exists():
            print(f"❌ reference audio not found: {ref_path}", file=sys.stderr)
            sys.exit(2)
        print(f"[prep] extracting global condition from {ref_path.name}")
        a, sr = librosa.load(str(ref_path), sr=16000, mono=True)
        ai = extract_features(torch.from_numpy(a).unsqueeze(0), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            vq = semantic_model(
                input_features=ai["input_features"].to(device),
                attention_mask=ai["attention_mask"].to(device),
                output_hidden_states=True,
            )
            feat = (vq.hidden_states[17] - semantic_mean) / semantic_std
            cond_lens = torch.tensor([feat.shape[1]], device=device)
            gpt_cond = gpt.get_conditioning(feat.transpose(1, 2), cond_lens)
            emo_cond = gpt.get_emo_conditioning(feat.transpose(1, 2), cond_lens)
            emo_vec = gpt.emo_layer(gpt.emovec_layer(emo_cond))
        global_condition = gpt_cond.squeeze(0).cpu().numpy().astype(np.float32)
        global_emo_vec = emo_vec.squeeze(0).cpu().numpy().astype(np.float32)
        np.save(features_dir / "GLOBAL_condition.npy", global_condition)
        np.save(features_dir / "GLOBAL_emo_vec.npy", global_emo_vec)

    # Walk audio dir
    audio_files = sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    )
    if args.limit:
        audio_files = audio_files[:args.limit]

    n_written = 0
    n_skipped = 0
    n_stutter_samples = 0
    total_stutter_tokens = 0
    total_tokens = 0

    manifest_path.unlink(missing_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as mf, torch.no_grad():
        for audio_path in tqdm(audio_files, desc="prep"):
            sid = audio_path.stem
            clean = clean_text_by_file.get(sid) or clean_text_by_file.get(audio_path.name)
            verbatim = verbatim_text_by_file.get(sid) or verbatim_text_by_file.get(audio_path.name)
            if not clean or not verbatim:
                n_skipped += 1
                continue

            try:
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
                duration = len(audio) / sr
                if duration < args.min_duration or duration > args.max_duration:
                    n_skipped += 1
                    continue

                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_16k_t = torch.from_numpy(audio_16k).unsqueeze(0)

                # Tokenize CLEAN text — this is the GPT input at inference
                text_tokens = tokenizer.tokenize(clean)
                text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                if len(text_ids) == 0 or len(text_ids) > args.max_text_tokens:
                    n_skipped += 1
                    continue
                text_ids_np = np.array(text_ids, dtype=np.int32)

                # Extract semantic codes — what the model has to PREDICT
                inputs = extract_features(audio_16k_t, sampling_rate=16000, return_tensors="pt")
                vq = semantic_model(
                    input_features=inputs["input_features"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    output_hidden_states=True,
                )
                feat = (vq.hidden_states[17] - semantic_mean) / semantic_std
                codes, _ = semantic_codec.quantize(feat)
                if codes.ndim == 2:
                    codes = codes[0]
                codes_np = codes.cpu().numpy().astype(np.int32)
                if codes_np.shape[0] == 0 or codes_np.shape[0] > args.max_mel_tokens:
                    n_skipped += 1
                    continue

                # Per-sample conditioning if no global is set
                if global_condition is None:
                    cond_lens = torch.tensor([feat.shape[1]], device=device)
                    gpt_cond = gpt.get_conditioning(feat.transpose(1, 2), cond_lens)
                    emo_cond = gpt.get_emo_conditioning(feat.transpose(1, 2), cond_lens)
                    emo_vec = gpt.emo_layer(gpt.emovec_layer(emo_cond))
                    cond_np = gpt_cond.squeeze(0).cpu().numpy().astype(np.float32)
                    emo_np = emo_vec.squeeze(0).cpu().numpy().astype(np.float32)
                    np.save(features_dir / f"{sid}_condition.npy", cond_np)
                    np.save(features_dir / f"{sid}_emo_vec.npy", emo_np)
                    cond_rel = f"features/{sid}_condition.npy"
                    emo_rel = f"features/{sid}_emo_vec.npy"
                else:
                    cond_rel = "features/GLOBAL_condition.npy"
                    emo_rel = "features/GLOBAL_emo_vec.npy"

                # Stutter detection + alignment → mel-token mask
                spans = detect_stutter_spans(clean, verbatim)
                verb_w = _words(verbatim)
                word_ts = word_ts_by_file.get(sid) or word_ts_by_file.get(audio_path.name)
                mask = word_spans_to_token_mask(
                    spans,
                    n_words=max(len(verb_w), 1),
                    n_codes=codes_np.shape[0],
                    word_timestamps=word_ts,
                    audio_duration=duration,
                    pad_tokens=args.pad_tokens,
                )
                total_tokens += codes_np.shape[0]
                total_stutter_tokens += int(mask.sum())
                if mask.any():
                    n_stutter_samples += 1

                # Save tensors
                np.save(features_dir / f"{sid}_text_ids.npy", text_ids_np)
                np.save(features_dir / f"{sid}_codes.npy", codes_np)
                np.save(features_dir / f"{sid}_stutter_mask.npy", mask.astype(np.bool_))

                record = {
                    "id": sid,
                    "text_ids_path": f"features/{sid}_text_ids.npy",
                    "codes_path": f"features/{sid}_codes.npy",
                    "condition_path": cond_rel,
                    "emo_vec_path": emo_rel,
                    "stutter_mask_path": f"features/{sid}_stutter_mask.npy",
                    "text_len": int(text_ids_np.shape[0]),
                    "code_len": int(codes_np.shape[0]),
                    "audio_duration": float(duration),
                    "n_stutter_spans": len(spans),
                    "stutter_token_ratio": float(mask.sum() / max(mask.size, 1)),
                    "clean_text": clean,
                    "verbatim_text": verbatim,
                }
                mf.write(json.dumps(record) + "\n")
                n_written += 1
            except Exception as e:
                print(f"  ! {audio_path.name}: {e}", file=sys.stderr)
                n_skipped += 1

    ratio = (total_stutter_tokens / total_tokens) if total_tokens else 0.0
    print(f"\n[prep] wrote {n_written} samples, skipped {n_skipped}")
    print(f"[prep] {n_stutter_samples} samples contain stutter regions")
    print(f"[prep] stutter token coverage: {total_stutter_tokens}/{total_tokens} = {ratio*100:.1f}%")
    print(f"[prep] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
