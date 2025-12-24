#!/usr/bin/env python3
"""
Automated Dataset Transcription Tool for IndexTTS2

Uses Whisper to automatically transcribe audio files with verbatim content,
capturing speaking patterns like pauses, fillers, and hesitations.

This creates EVERYTHING needed for training from just audio files!

Features:
- Automatic transcription with Whisper
- Detects and marks pauses from word timing gaps
- Captures filler words (uh, um, etc.)
- Generates training-ready CSV and manifest files
- Supports multiple Whisper model sizes

Usage:
    # Simplest - just provide audio directory
    python tools/transcribe_dataset.py --speaker goldblum
    
    # With custom options
    python tools/transcribe_dataset.py \
        --speaker goldblum \
        --whisper-model large-v3 \
        --language en \
        --detect-fillers

Requires:
    pip install openai-whisper
    # OR for faster inference:
    pip install faster-whisper
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import os
import sys
import torch



# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class WordTiming:
    """Word with timestamp information."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptionResult:
    """Result from transcribing an audio file."""
    audio_path: str
    text: str
    verbatim_text: str
    words: List[WordTiming] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    
    # Pattern statistics
    num_pauses: int = 0
    num_long_pauses: int = 0
    num_fillers: int = 0


class WhisperTranscriber:
    """Wrapper for Whisper transcription with pattern detection."""
    
    # Common filler words to detect
    FILLER_PATTERNS = {
        'uh', 'uhh', 'uhhh', 'um', 'umm', 'ummm', 'hmm', 'hmmm',
        'ah', 'ahh', 'er', 'err', 'eh', 'like', 'you know',
        'i mean', 'well', 'so', 'basically', 'actually',
    }
    
    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
        use_faster_whisper: bool = True,
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (auto, cuda, cpu)
            compute_type: Compute type for faster-whisper (auto, float16, int8, etc.)
            use_faster_whisper: Use faster-whisper if available (recommended)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.use_faster_whisper = use_faster_whisper
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        if self.use_faster_whisper:
            try:
                import nemo.collections.asr as nemo_asr
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name="nvidia/parakeet-tdt-0.6b-v3"
                )
                self._transcribe_fn = self._transcribe_nemo
                print('✓ NeMo Parakeet loaded successfully')
                return
                
            except Exception as e:
                print(f"Failed to load NeMo Parakeet: {e}")
                import traceback
                traceback.print_exc()


    
    def _transcribe_nemo(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> Tuple[str, List[WordTiming], str]:
        """Transcribe using NVIDIA Parakeet."""
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
                results = self.model.transcribe(
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
                
                # Extract word timestamps
                words = []
                ts_data = getattr(hyp, 'timestep', None) or getattr(hyp, 'timestamp', None)
                
                if ts_data and isinstance(ts_data, dict) and 'word' in ts_data:
                    for word_info in ts_data['word']:
                        try:
                            word_text = word_info.get('word', '').strip()
                            if word_text:  # Skip empty words
                                words.append(WordTiming(
                                    word=word_text,
                                    start=float(word_info.get('start', 0)),
                                    end=float(word_info.get('end', 0)),
                                    probability=1.0,
                                ))
                        except (KeyError, ValueError, TypeError) as e:
                            continue
                
                return full_text.strip(), words, "en"
            
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
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_pause_duration: float = 0.3,
        long_pause_duration: float = 0.7,
        detect_fillers: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio file with pattern detection.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en') or None for auto-detection
            min_pause_duration: Minimum gap (seconds) to consider a pause
            long_pause_duration: Duration threshold for [LONG] vs [PAUSE]
            detect_fillers: Whether to mark filler words
            
        Returns:
            TranscriptionResult with verbatim text including pattern markers
        """
        # Get transcription with word timestamps
        text, words, detected_language = self._transcribe_fn(audio_path, language)
        if not words:
            # No word timestamps available
            return TranscriptionResult(
                audio_path=audio_path,
                text=text,
                verbatim_text=text,
                language=detected_language,
            )
        
        # Build verbatim text with pause markers and filler detection
        verbatim_parts = []
        num_pauses = 0
        num_long_pauses = 0
        num_fillers = 0
        
        for i, word in enumerate(words):
            # Check for pause before this word
            if i > 0:
                gap = word.start - words[i-1].end
                if gap >= long_pause_duration:
                    verbatim_parts.append("[LONG]")
                    num_long_pauses += 1
                elif gap >= min_pause_duration:
                    verbatim_parts.append("[PAUSE]")
                    num_pauses += 1
            
            # Check if this word is a filler
            word_lower = word.word.lower().strip()
            if detect_fillers and word_lower in self.FILLER_PATTERNS:
                if word_lower in ('uh', 'uhh', 'uhhh'):
                    verbatim_parts.append("[UH]")
                    num_fillers += 1
                elif word_lower in ('um', 'umm', 'ummm'):
                    verbatim_parts.append("[UM]")
                    num_fillers += 1
                elif word_lower in ('hmm', 'hmmm'):
                    verbatim_parts.append("[HMM]")
                    num_fillers += 1
                elif word_lower in ('ah', 'ahh'):
                    verbatim_parts.append("[AH]")
                    num_fillers += 1
                elif word_lower in ('er', 'err'):
                    verbatim_parts.append("[ER]")
                    num_fillers += 1
                else:
                    # Keep other fillers as-is (like, you know, etc.)
                    verbatim_parts.append(word.word)
            else:
                verbatim_parts.append(word.word)
        
        verbatim_text = " ".join(verbatim_parts)
        # Clean up spacing
        verbatim_text = verbatim_text.replace("  ", " ").strip()
        
        # Calculate duration
        duration = words[-1].end if words else 0.0
        
        return TranscriptionResult(
            audio_path=audio_path,
            text=text,
            verbatim_text=verbatim_text,
            words=words,
            language=detected_language,
            duration=duration,
            num_pauses=num_pauses,
            num_long_pauses=num_long_pauses,
            num_fillers=num_fillers,
        )


def get_audio_files(audio_dir: Path) -> List[Path]:
    """Find all audio files in directory."""
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.webm'}
    files = []
    for ext in extensions:
        files.extend(audio_dir.glob(f"*{ext}"))
        files.extend(audio_dir.glob(f"*{ext.upper()}"))
    return sorted(files)


def save_transcripts_csv(
    results: List[TranscriptionResult],
    output_path: Path,
    verbatim: bool = True,
):
    """Save transcriptions to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'text', 'verbatim', 'duration', 'num_pauses', 'num_fillers']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'filename': Path(result.audio_path).name,
                'text': result.text,
                'verbatim': result.verbatim_text,
                'duration': f"{result.duration:.2f}",
                'num_pauses': result.num_pauses + result.num_long_pauses,
                'num_fillers': result.num_fillers,
            })
    
    print(f"✓ Saved transcripts to: {output_path}")


def save_stats(results: List[TranscriptionResult], output_path: Path):
    """Save transcription statistics."""
    total_duration = sum(r.duration for r in results)
    total_pauses = sum(r.num_pauses for r in results)
    total_long_pauses = sum(r.num_long_pauses for r in results)
    total_fillers = sum(r.num_fillers for r in results)
    
    stats = {
        'num_files': len(results),
        'total_duration_seconds': round(total_duration, 2),
        'total_duration_minutes': round(total_duration / 60, 2),
        'total_pauses': total_pauses,
        'total_long_pauses': total_long_pauses,
        'total_fillers': total_fillers,
        'avg_pauses_per_file': round(total_pauses / len(results), 2) if results else 0,
        'avg_fillers_per_file': round(total_fillers / len(results), 2) if results else 0,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to: {output_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Automatically transcribe audio files for IndexTTS2 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Transcribe all audio in a speaker's directory
    python tools/transcribe_dataset.py --speaker goldblum
    
    # Use a larger model for better accuracy
    python tools/transcribe_dataset.py --speaker goldblum --whisper-model large-v3
    
    # Specify language explicitly
    python tools/transcribe_dataset.py --speaker goldblum --language en
    
    # Custom audio directory
    python tools/transcribe_dataset.py --audio-dir my_audio/ --output-dir my_output/
"""
    )
    
    # Input options
    parser.add_argument("--speaker", "-s",
                        help="Speaker name (uses training/{speaker}/dataset/audio)")
    parser.add_argument("--audio-dir", type=Path,
                        help="Custom audio directory (overrides --speaker)")
    parser.add_argument("--output-dir", type=Path,
                        help="Custom output directory (default: alongside audio)")
    
    # Whisper options
    parser.add_argument("--whisper-model", "-m", default="medium",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (e.g., 'en'). Auto-detected if not specified")
    parser.add_argument("--no-faster-whisper", action="store_true",
                        help="Use openai-whisper instead of faster-whisper")
    
    # Pattern detection options
    parser.add_argument("--min-pause", type=float, default=0.3,
                        help="Minimum pause duration in seconds (default: 0.3)")
    parser.add_argument("--long-pause", type=float, default=0.7,
                        help="Long pause threshold in seconds (default: 0.7)")
    parser.add_argument("--no-fillers", action="store_true",
                        help="Don't mark filler words")
    
    # Processing options
    parser.add_argument("--device", default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if transcripts CSV already exists")
    
    args = parser.parse_args()
    
    # Determine paths
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker / "dataset"
        audio_dir = speaker_dir / "audio"
        output_dir = args.output_dir or speaker_dir
    elif args.audio_dir:
        audio_dir = args.audio_dir
        output_dir = args.output_dir or audio_dir.parent
    else:
        parser.error("Either --speaker or --audio-dir is required")
    
    # Validate
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        print(f"\nFor speaker '{args.speaker}', create:")
        print(f"  {audio_dir}/")
        print(f"  └── your_audio_files.wav")
        sys.exit(1)
    
    # Find audio files
    audio_files = get_audio_files(audio_dir)
    if not audio_files:
        print(f"❌ No audio files found in: {audio_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("AUTOMATED TRANSCRIPTION FOR INDEXTTS2")
    print("=" * 60)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Found {len(audio_files)} audio files")
    print(f"Transcribe model: {args.whisper_model}")
    print(f"Language: {args.language or 'auto-detect'}")
    print()
    
    # Check for existing transcripts
    output_csv = output_dir / "transcripts_verbatim.csv"
    if args.skip_existing and output_csv.exists():
        print(f"✓ Transcripts already exist: {output_csv}")
        print("  Use --no-skip-existing to regenerate")
        return
    
    # Initialize transcriber
    print("Loading transcribe model...")
    transcriber = WhisperTranscriber(
        model_name=args.whisper_model,
        device=args.device,
        use_faster_whisper=True,
    )
    
    # Transcribe all files
    print("\nTranscribing audio files...")
    results = []
    
    from tqdm import tqdm
    for audio_path in tqdm(audio_files, desc="Transcribing"):
        try:
            result = transcriber.transcribe(
                str(audio_path),
                language=args.language,
                min_pause_duration=args.min_pause,
                long_pause_duration=args.long_pause,
                detect_fillers=not args.no_fillers,
            )
            results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to transcribe {audio_path.name}: {e}")
    
    if not results:
        print("❌ No files were successfully transcribed")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transcripts
    save_transcripts_csv(results, output_csv)
    
    # Save statistics
    stats = save_stats(results, output_dir / "transcription_stats.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 60)
    print(f"""
Files transcribed:     {stats['num_files']}
Total duration:        {stats['total_duration_minutes']:.1f} minutes
Total pauses detected: {stats['total_pauses'] + stats['total_long_pauses']}
Total fillers found:   {stats['total_fillers']}

Output files:
  {output_csv}
  {output_dir / 'transcription_stats.json'}
""")
    
    # Print example
    if results:
        example = results[0]
        print("Example transcription:")
        print(f"  File: {Path(example.audio_path).name}")
        print(f"  Original:  {example.text[:100]}...")
        print(f"  Verbatim:  {example.verbatim_text[:100]}...")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. Review and edit transcripts if needed:
   {output_csv}

2. Prepare training dataset:
   python tools/prepare_pattern_dataset.py \\
       --audio-dir {audio_dir} \\
       --transcripts {output_csv} \\
       --output-dir {output_dir / 'processed'}

3. Train the model:
   python tools/train_gpt_lora.py \\
       --train-manifest {output_dir / 'processed/train_manifest.jsonl'} \\
       --val-manifest {output_dir / 'processed/val_manifest.jsonl'} \\
       --output-dir training/{args.speaker or 'custom'}/lora

4. Extract embeddings:
   python tools/extract_embeddings.py {'--speaker ' + args.speaker if args.speaker else '--audio ' + str(audio_files[0])}

5. Run inference with patterns:
   python tools/infer.py {'--speaker ' + args.speaker if args.speaker else ''} \\
       --text "Your text here"
""")


if __name__ == "__main__":
    main()