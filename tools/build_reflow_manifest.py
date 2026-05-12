#!/usr/bin/env python3
"""
Build a reflow manifest from a speaker's existing dataset.

Inputs: training/<speaker>/dataset/transcripts_verbatim.csv (or any compatible
CSV / JSONL with audio paths + verbatim text).

Output: a JSONL manifest consumable by tools/generate_reflow_pairs.py:
    {"id": "audio_001", "audio_prompt": "/abs/path/.wav", "text": "..."}

Why we keep this separate from generate_reflow_pairs.py:
- Manifest construction is fast (no GPU) and easy to iterate on/inspect.
- Lets the UI preview the manifest before kicking off the slow data-gen step.
- Different speakers may have different CSV columns or different naming schemes
  — having an explicit step makes those quirks obvious.

CSV format (auto-detected; flexible):
    filename, text, [verbatim], [duration], [...]
If a `verbatim` column exists, it's preferred over `text` because reflow learns
better when the trajectory matches the trained patterns (stutters and all).

Reference audio:
    Each manifest record needs an audio_prompt for speaker conditioning. Two
    strategies:
      --reference-audio path/to/single.wav   → all records share one reference.
                                              Recommended — fixes the speaker
                                              characteristics across the dataset.
      --reference-from-row                   → use each row's audio as its own
                                              reference. Higher variance, more
                                              prompt diversity.

Usage:
    python tools/build_reflow_manifest.py \\
        --speaker ozzyv5 \\
        --output training/ozzyv5/reflow_manifest.jsonl \\
        --reference-audio training/ozzyv5/dataset/audio/audio_000.wav
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speaker", "-s", required=True, help="Speaker folder name under training/.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Custom CSV. Defaults to training/<speaker>/dataset/transcripts_verbatim.csv.")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Where the per-row audio files live. Defaults to training/<speaker>/dataset/audio/.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSONL path. Defaults to training/<speaker>/reflow_manifest.jsonl.")
    parser.add_argument("--reference-audio", type=Path, default=None,
                        help="Single reference audio used for ALL records (recommended).")
    parser.add_argument("--reference-from-row", action="store_true",
                        help="Use each row's own audio as the reference. Mutually exclusive with --reference-audio.")
    parser.add_argument("--text-column", default=None,
                        help="Which CSV column to use for text. Default: auto-detect "
                             "(prefers 'verbatim', falls back to 'text').")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="How many z-draws to request per record at pair-gen time.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only write the first N records (useful for smoke testing).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def autodetect_text_column(fieldnames: List[str]) -> Optional[str]:
    """Pick the best text column from a CSV row, in order of preference."""
    preference = ["verbatim", "text", "parakeet", "fastwhisper", "transcript", "transcription"]
    fset = {f.lower(): f for f in fieldnames}
    for candidate in preference:
        if candidate in fset:
            return fset[candidate]
    return None


def autodetect_audio_column(fieldnames: List[str]) -> Optional[str]:
    """Pick the best audio-path column."""
    preference = ["filename", "audio_path", "file", "path", "audio"]
    fset = {f.lower(): f for f in fieldnames}
    for candidate in preference:
        if candidate in fset:
            return fset[candidate]
    return None


def iter_csv(csv_path: Path) -> Iterable[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            yield row


def main() -> int:
    args = parse_args()

    if args.reference_audio and args.reference_from_row:
        print("ERROR: pass either --reference-audio OR --reference-from-row, not both.", file=sys.stderr)
        return 1

    speaker_root = PROJECT_ROOT / "training" / args.speaker
    if not speaker_root.exists():
        print(f"ERROR: speaker folder not found: {speaker_root}", file=sys.stderr)
        return 1

    csv_path = args.csv or (speaker_root / "dataset" / "transcripts_verbatim.csv")
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    audio_dir = args.audio_dir or (speaker_root / "dataset" / "audio")
    output = args.output or (speaker_root / "reflow_manifest.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.reference_audio and not args.reference_audio.exists():
        print(f"ERROR: reference audio not found: {args.reference_audio}", file=sys.stderr)
        return 1

    # Sniff CSV to pick columns
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
    text_col = args.text_column or autodetect_text_column(fieldnames)
    audio_col = autodetect_audio_column(fieldnames)
    if text_col is None:
        print(f"ERROR: couldn't find a text column in {csv_path}. Columns: {fieldnames}", file=sys.stderr)
        return 1
    if audio_col is None:
        print(f"ERROR: couldn't find an audio-path column in {csv_path}. Columns: {fieldnames}", file=sys.stderr)
        return 1
    print(f">> Using text column: {text_col!r}, audio column: {audio_col!r}")

    # Optional duration filter — only applied when the row has a 'duration' column
    duration_col = None
    for cand in ("duration", "duration_s", "dur"):
        if cand in fieldnames:
            duration_col = cand
            break

    n_written = 0
    n_skipped_missing = 0
    n_skipped_short = 0
    n_skipped_long = 0
    n_skipped_empty = 0

    with open(output, "w", encoding="utf-8") as outfp:
        for i, row in enumerate(iter_csv(csv_path)):
            if args.limit and n_written >= args.limit:
                break

            text = (row.get(text_col) or "").strip()
            if not text:
                n_skipped_empty += 1
                continue

            audio_field = (row.get(audio_col) or "").strip()
            if not audio_field:
                n_skipped_missing += 1
                continue

            audio_path = Path(audio_field)
            if not audio_path.is_absolute():
                audio_path = audio_dir / audio_field
            if not audio_path.exists():
                # Try common extensions if the filename was given without one
                for ext in (".wav", ".mp3", ".flac"):
                    alt = audio_path.with_suffix(ext)
                    if alt.exists():
                        audio_path = alt
                        break
                else:
                    n_skipped_missing += 1
                    if args.verbose:
                        print(f"   skip (missing audio): {audio_field}", file=sys.stderr)
                    continue

            if duration_col is not None:
                try:
                    dur = float(row[duration_col])
                except (TypeError, ValueError):
                    dur = None
                if dur is not None:
                    if dur < args.min_duration:
                        n_skipped_short += 1
                        continue
                    if dur > args.max_duration:
                        n_skipped_long += 1
                        continue

            utt_id = audio_path.stem

            if args.reference_audio:
                ref = str(args.reference_audio.resolve())
            elif args.reference_from_row:
                ref = str(audio_path.resolve())
            else:
                # Default: use the row's own audio as both reference and (unused) target.
                # The reflow trainer only cares about the synthesized x_final, not the
                # original ground truth audio.
                ref = str(audio_path.resolve())

            record = {
                "id": utt_id,
                "audio_prompt": ref,
                "text": text,
                "n_samples": args.n_samples,
            }
            outfp.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\n>> Wrote {n_written} manifest records → {output}")
    if n_skipped_empty:
        print(f"   Skipped empty text: {n_skipped_empty}")
    if n_skipped_missing:
        print(f"   Skipped missing audio: {n_skipped_missing}")
    if n_skipped_short:
        print(f"   Skipped <{args.min_duration}s duration: {n_skipped_short}")
    if n_skipped_long:
        print(f"   Skipped >{args.max_duration}s duration: {n_skipped_long}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
