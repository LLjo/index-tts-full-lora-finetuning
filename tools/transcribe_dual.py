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

import torch
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
    ):
        self.device = device
        self.compute_type = compute_type
        
        print(f"\n[Transcriber] Initializing on device: {device}")
        
        # Load FastWhisper
        print(f"  Loading FastWhisper ({whisper_model})...")
        self._init_whisper(whisper_model)
        
        # Load Parakeet
        print(f"  Loading Parakeet ({parakeet_model})...")
        self._init_parakeet(parakeet_model)
        
        print("  ✓ Both transcribers loaded")
    
    def _init_whisper(self, model_name: str):
        """Initialize FastWhisper model."""
        try:
            from faster_whisper import WhisperModel
            
            # faster-whisper/ctranslate2 expects "cuda" not "cuda:0"
            whisper_device = self.device
            if whisper_device.startswith("cuda:"):
                whisper_device = "cuda"
            
            self.whisper = WhisperModel(
                model_name,
                device=whisper_device,
                compute_type=self.compute_type,
            )
        except ImportError:
            print("  ⚠ FastWhisper not installed. Install with: pip install faster-whisper")
            self.whisper = None
    
    def _init_parakeet(self, model_name: str):
        """Initialize NVIDIA Parakeet model."""
        try:
            import nemo.collections.asr as nemo_asr
            
            # Parakeet models are CTC-based and transcribe verbatim
            print(f"    Downloading/loading Parakeet model...")
            self.parakeet = nemo_asr.models.ASRModel.from_pretrained(model_name)
            
            # Handle device - NeMo models use .cuda() or .cpu()
            if "cuda" in self.device:
                self.parakeet = self.parakeet.cuda()
            else:
                self.parakeet = self.parakeet.cpu()
            self.parakeet.eval()
            print(f"    ✓ Parakeet loaded successfully")
        except ImportError as e:
            print(f"  ⚠ NeMo not installed: {e}")
            print("    Install with: pip install nemo_toolkit[asr]")
            print("    Or use --skip-parakeet to use FastWhisper for both columns")
            self.parakeet = None
        except Exception as e:
            print(f"  ⚠ Failed to load Parakeet: {e}")
            print("    Using FastWhisper fallback for verbatim column")
            self.parakeet = None
            import traceback
            traceback.print_exc()
    
    def transcribe_whisper(self, audio_path: str) -> str:
        """Transcribe with FastWhisper (clean output)."""
        if self.whisper is None:
            return ""
        
        try:
            segments, _ = self.whisper.transcribe(
                audio_path,
                beam_size=5,
                language="en",  # Adjust if needed
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            warnings.warn(f"FastWhisper failed for {audio_path}: {e}")
            return ""
    
    def transcribe_parakeet(self, audio_path: str) -> str:
        """Transcribe with Parakeet (verbatim output)."""
        if self.parakeet is None:
            return ""
        
        import tempfile
        import soundfile as sf
        import numpy as np
        import torch
        
        try:
            # Load and prepare audio
            audio, sample_rate = sf.read(audio_path)
            
            # Convert stereo to mono if necessary
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            
            # Ensure correct sample rate (16kHz for Parakeet)
            expected_sr = 16000
            if sample_rate != expected_sr:
                # Resample if needed
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=expected_sr)
                sample_rate = expected_sr
            
            # Create temporary mono audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, sample_rate)
            
            try:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Transcribe with timestamps - use batch_size=1
                results = self.parakeet.transcribe(
                    [tmp_path],
                    batch_size=1,
                    timestamps=True,
                    return_hypotheses=True,
                )
                
                # Extract results
                if not results or len(results) == 0:
                    print(f"Warning: No results for {audio_path}")
                    return "", [], "en"
                
                # Get first hypothesis
                hyp = results[0]
                if isinstance(hyp, list):
                    if len(hyp) == 0:
                        return "", [], "en"
                    hyp = hyp[0]
                
                # Extract text
                full_text = getattr(hyp, 'text', '')
                if not full_text:
                    return "", [], "en"
                
                # # Extract word timestamps
                # words = []
                # ts_data = getattr(hyp, 'timestep', None) or getattr(hyp, 'timestamp', None)
                
                # if ts_data and isinstance(ts_data, dict) and 'word' in ts_data:
                #     for word_info in ts_data['word']:
                #         try:
                #             word_text = word_info.get('word', '').strip()
                #             if word_text:  # Skip empty words
                #                 words.append(WordTiming(
                #                     word=word_text,
                #                     start=float(word_info.get('start', 0)),
                #                     end=float(word_info.get('end', 0)),
                #                     probability=1.0,
                #                 ))
                #         except (KeyError, ValueError, TypeError) as e:
                #             continue
                
                return full_text.strip()#, words, "en"
            
            finally:
                # Cleanup
                import os
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error transcribing {audio_path}: {type(e).__name__}: {str(e)[:100]}")
            return "", [], "en"
        
        try:
            # Parakeet transcribes including stutters and hesitations
            outputs = self.parakeet.transcribe([audio_path])
            if outputs and len(outputs) > 0:
                return outputs[0].strip()
            return ""
        except Exception as e:
            warnings.warn(f"Parakeet failed for {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
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
        
        # If Parakeet failed but we have whisper, create a "verbatim-like" version
        # by using whisper with different settings
        if not parakeet_text and fastwhisper_text and use_whisper_fallback:
            # Use whisper but with more verbatim-like settings
            parakeet_text = self._transcribe_whisper_verbatim(audio_path)
        
        return {
            "fastwhisper": fastwhisper_text,
            "parakeet": parakeet_text,
        }
    
    def _transcribe_whisper_verbatim(self, audio_path: str) -> str:
        """
        Transcribe with WhisperModel using verbatim-like settings.
        This is a fallback when Parakeet is not available.
        
        Uses settings that preserve more natural speech patterns:
        - Lower beam size for less smoothing
        - No VAD filter (preserve all pauses)
        - Hallucination suppression disabled
        """
        if self.whisper is None:
            return ""
        
        try:
            segments, _ = self.whisper.transcribe(
                audio_path,
                beam_size=1,  # Less smoothing
                language="en",
                vad_filter=False,  # Don't filter silence
                word_timestamps=True,  # Get word timings
                condition_on_previous_text=True,  # More coherent
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            warnings.warn(f"Whisper verbatim fallback failed for {audio_path}: {e}")
            return ""


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
    parser.add_argument("--whisper-model", default="large-v3",
                        help="FastWhisper model size (default: large-v3)")
    parser.add_argument("--parakeet-model", default="nvidia/parakeet-ctc-1.1b",
                        help="Parakeet model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-type", default="float16",
                        help="Compute type for FastWhisper (float16, int8)")
    
    # Processing options
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already in output CSV")
    
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