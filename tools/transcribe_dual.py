#!/usr/bin/env python3
"""
Dual Transcription for Verbatim Training

Transcribes audio files using TWO different transcribers:
1. FastWhisper - produces CLEAN text (removes stutters, normalizes)
2. NVIDIA Parakeet - produces VERBATIM text (keeps stutters, hesitations)

Output: CSV file with columns (filename, fastwhisper, parakeet)

Requirements:
    pip install faster-whisper
    pip install nemo_toolkit[asr]  # for Parakeet
    
Usage:
    python tools/transcribe_dual.py --speaker ozzy
    
    # Or with explicit paths
    python tools/transcribe_dual.py \
        --audio-dir training/ozzy/dataset/audio \
        --output-csv training/ozzy/dataset/transcripts.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# numpy 2.0 removed `np.sctypes`, but NeMo's preprocessing.segment still uses it
# to detect integer dtypes. Restore the dict NeMo expects BEFORE any nemo import.
# Purely additive — harmless if NeMo is later updated to drop the reference.
import numpy as _np
if not hasattr(_np, "sctypes"):
    _np.sctypes = {
        "int":     [_np.int8, _np.int16, _np.int32, _np.int64],
        "uint":    [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
        "float":   [_np.float16, _np.float32, _np.float64],
        "complex": [_np.complex64, _np.complex128],
        "others":  [bool, object, bytes, str, _np.void],
    }

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DualTranscriber:
    """Transcribes audio using both FastWhisper and Parakeet."""
    
    def __init__(
        self,
        whisper_model: str = "large-v3",
        parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        strict_parakeet: bool = False,
    ):
        self.device = device
        self.compute_type = compute_type
        # When True, Parakeet load/transcribe failures raise instead of silently
        # producing empty cells. Use this from CI / the API so we never silently
        # ship an all-empty verbatim column again.
        self.strict_parakeet = strict_parakeet

        print(f"\n[Transcriber] Initializing on device: {device}  strict_parakeet={strict_parakeet}")

        print(f"  Loading FastWhisper ({whisper_model})...")
        self._init_whisper(whisper_model)

        print(f"  Loading Parakeet ({parakeet_model})...")
        self._init_parakeet(parakeet_model)

        if self.parakeet is None and strict_parakeet:
            raise RuntimeError("PARAKEET_LOAD_FAILED: Parakeet did not load (see logs above).")
        print(f"  ✓ Transcribers loaded  (parakeet={'yes' if self.parakeet is not None else 'NO — verbatim will be empty'})")
    
    def _init_whisper(self, model_name: str):
        """Initialize FastWhisper model."""
        # try:
        #     from faster_whisper import WhisperModel
            
        #     # faster-whisper/ctranslate2 expects "cuda" not "cuda:0"
        #     whisper_device = self.device
        #     if whisper_device.startswith("cuda:"):
        #         whisper_device = "cuda"
            
        #     self.whisper = WhisperModel(
        #         model_name,
        #         device='cpu',
        #         # compute_type=self.compute_type,
        #     )
        # except ImportError:
        #     print("  ⚠ FastWhisper not installed. Install with: pip install faster-whisper")
        #     self.whisper = None

        self.whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo", torch_dtype=self.compute_type, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.whisper.to(device)
    
    def _init_parakeet(self, model_name: str):
        """Initialize NVIDIA Parakeet model."""
        try:
            import nemo.collections.asr as nemo_asr

            print(f"    Downloading/loading Parakeet model...")
            self.parakeet = nemo_asr.models.ASRModel.from_pretrained(model_name)

            if "cuda" in self.device:
                self.parakeet = self.parakeet.cuda()
            else:
                self.parakeet = self.parakeet.cpu()
            self.parakeet.eval()
            print(f"    ✓ Parakeet loaded successfully")
        except ImportError as e:
            msg = (
                f"NeMo / one of its deps failed to import: {e}\n"
                f"   Typical fixes: pip install nemo_toolkit[asr] lilcom\n"
                f"   If you saw `pyarrow.PyExtensionType`: upgrade `datasets` to 3.x"
            )
            if getattr(self, "strict_parakeet", False):
                raise RuntimeError(f"PARAKEET_LOAD_FAILED: {msg}") from e
            print(f"  ⚠ {msg}")
            print("    Continuing without Parakeet — verbatim column will be empty.")
            self.parakeet = None
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            if getattr(self, "strict_parakeet", False):
                raise RuntimeError(f"PARAKEET_LOAD_FAILED: {e}\n{tb}") from e
            print(f"  ⚠ Failed to load Parakeet: {e}")
            print(tb)
            self.parakeet = None
    
    def transcribe_whisper(self, audio_path: str) -> str:
        """Transcribe with FastWhisper (clean output)."""
        if self.whisper is None:
            return ""
        
        

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.compute_type,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        result = pipe(audio_path)
        print(result["text"])
        return result["text"]
    
    def transcribe_parakeet(self, audio_path: str) -> str:
        """Transcribe with Parakeet (verbatim). Returns "" on failure unless
        strict_parakeet is set, in which case it raises."""
        if self.parakeet is None:
            return ""

        import os
        import tempfile
        import soundfile as sf
        import numpy as np
        import torch

        tmp_path = None
        try:
            audio, sample_rate = sf.read(audio_path)
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, sample_rate)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results = self.parakeet.transcribe(
                [tmp_path],
                batch_size=1,
                timestamps=True,
                return_hypotheses=True,
            )

            if not results:
                return ""
            hyp = results[0]
            if isinstance(hyp, list):
                if not hyp:
                    return ""
                hyp = hyp[0]
            text = getattr(hyp, 'text', '') or ''
            return text.strip()

        except Exception as e:
            if getattr(self, "strict_parakeet", False):
                raise
            import traceback
            print(f"  ! parakeet on {Path(audio_path).name}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
            return ""
        finally:
            if tmp_path:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def transcribe(self, audio_path: str, use_whisper_fallback: bool = True) -> Dict[str, str]:
        """
        Transcribe audio file with both models.
        
        Args:
            audio_path: Path to audio file
            use_whisper_fallback: If True, use WhisperLongForm for parakeet when Parakeet is unavailable
        
        Returns:
            dict with 'fastwhisper' and 'parakeet' keys
        """
        fastwhisper_text = self.transcribe_whisper(audio_path)
        parakeet_text = self.transcribe_parakeet(audio_path)
        
        return {
            "fastwhisper": fastwhisper_text,
            "parakeet": parakeet_text,
        }

def find_audio_files(audio_dir: Path) -> List[Path]:
    """Find all audio files in directory."""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = []
    for ext in audio_extensions:
        files.extend(audio_dir.glob(f"*{ext}"))
        files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with both FastWhisper and Parakeet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--speaker", "-s", help="Speaker name (uses default paths)")
    parser.add_argument("--audio-dir", type=Path, help="Audio directory")
    parser.add_argument("--output-csv", type=Path, help="Output CSV path")
    
    # Model options
    parser.add_argument("--whisper-model", default="medium",
                        help="FastWhisper model size (default: large-v3)")
    parser.add_argument("--parakeet-model", default="nvidia/parakeet-tdt-0.6b-v3",
                        help="Parakeet model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-type", default="float16",
                        help="Compute type for FastWhisper (float16, int8)")
    
    # Processing options
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already in output CSV")
    parser.add_argument("--strict-parakeet", action="store_true",
                        help="Fail loudly (non-zero exit) if Parakeet can't load or transcribe. "
                             "Use this from the API so we never silently produce an empty verbatim column.")

    args = parser.parse_args()
    
    # Resolve paths
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker / "dataset"
        audio_dir = args.audio_dir or speaker_dir / "audio"
        output_csv = args.output_csv or speaker_dir / "transcripts.csv"
    else:
        if not args.audio_dir or not args.output_csv:
            parser.error("--speaker or both --audio-dir and --output-csv required")
        audio_dir = args.audio_dir
        output_csv = args.output_csv
    
    print("=" * 60)
    print("DUAL TRANSCRIPTION")
    print("=" * 60)
    print(f"\nAudio dir: {audio_dir}")
    print(f"Output CSV: {output_csv}")
    
    # Validate
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    # Find audio files
    audio_files = find_audio_files(audio_dir)
    print(f"\nFound {len(audio_files)} audio files")
    
    if not audio_files:
        print("❌ No audio files found!")
        sys.exit(1)
    
    # Load existing transcripts if skip_existing
    existing = set()
    if args.skip_existing and output_csv.exists():
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row.get("filename", ""))
        print(f"  Skip existing: {len(existing)} files already transcribed")
    
    # Filter files
    files_to_process = [f for f in audio_files if f.name not in existing]
    print(f"  Files to transcribe: {len(files_to_process)}")
    
    # Initialize transcriber
    print("\n" + "-" * 40)
    transcriber = DualTranscriber(
        whisper_model=args.whisper_model,
        parakeet_model=args.parakeet_model,
        device=args.device,
        compute_type=args.compute_type,
        strict_parakeet=getattr(args, "strict_parakeet", False),
    )
    print("-" * 40)
    
    # Process files
    print("\n[Transcribing]")
    results = []
    
    for audio_path in tqdm(files_to_process, desc="Processing"):
        transcripts = transcriber.transcribe(str(audio_path))
        results.append({
            "filename": audio_path.name,
            "fastwhisper": transcripts["fastwhisper"],
            "parakeet": transcripts["parakeet"],
        })
    
    # Merge with existing if any
    if args.skip_existing and output_csv.exists():
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_results = list(reader)
        results = existing_results + results
    
    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "fastwhisper", "parakeet"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Saved transcripts to: {output_csv}")

    # Column coverage report — catches "Parakeet silently None'd" and similar.
    n_total = len(results)
    n_fw_empty = sum(1 for r in results if not (r.get("fastwhisper") or "").strip())
    n_pk_empty = sum(1 for r in results if not (r.get("parakeet") or "").strip())
    print(f"\n[Coverage]  fastwhisper: {n_total - n_fw_empty}/{n_total} non-empty"
          f"  parakeet: {n_total - n_pk_empty}/{n_total} non-empty")
    if n_pk_empty == n_total and n_total > 0:
        print("⚠ Parakeet column is ENTIRELY EMPTY. The downstream character-LoRA dataset "
              "prep relies on this column to detect stutters — fix Parakeet before "
              "proceeding (see install hints above) and re-run with --strict-parakeet "
              "to fail loudly next time.")
    if n_fw_empty == n_total and n_total > 0:
        print("⚠ FastWhisper column is ENTIRELY EMPTY. Something went wrong with Whisper.")

    # Show sample
    print("\n[Sample Results]")
    for i, row in enumerate(results[:3]):
        print(f"\n{row['filename']}:")
        print(f"  FastWhisper: {row['fastwhisper'][:80]}...")
        print(f"  Parakeet:    {row['parakeet'][:80]}...")
    
    print("\n" + "=" * 60)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 60)
    print(f"""
Total files: {len(results)}
Output: {output_csv}

FastWhisper output: Clean, normalized text
Parakeet output: Verbatim with stutters/hesitations

NEXT STEPS:
===========
1. Review the CSV and manually correct any transcription errors

2. Prepare the dataset:
    python tools/prepare_verbatim_dataset.py --speaker {args.speaker or 'SPEAKER'}

3. Train the model:
    python tools/train_verbatim_lora.py --speaker {args.speaker or 'SPEAKER'}
""")


if __name__ == "__main__":
    main()