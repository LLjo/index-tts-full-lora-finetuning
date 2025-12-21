"""
Speaker Embeddings Storage for Promptless Inference

This module allows you to:
1. Extract and store speaker embeddings from reference audio
2. Load stored embeddings for inference without needing audio prompt

CRITICAL INSIGHT:
=================
IndexTTS2 has a two-stage architecture:

Stage 1 (GPT): Text + Conditioning → Semantic Tokens
  - LoRA/finetuning trains this stage
  - Learns prosody, rhythm, speaking patterns
  
Stage 2 (S2Mel + BigVGAN): Semantic Tokens + Reference Audio → Waveform
  - Uses ref_mel, style, prompt_condition from reference audio
  - Determines the ACTUAL voice timbre/quality
  
THE PROBLEM:
When you provide a reference audio at inference, Stage 2 uses THAT audio's
characteristics, potentially overwriting what the GPT learned.

THE SOLUTION:
Store embeddings extracted from your training voice samples and use them
at inference time. This ensures Stage 2 uses the same voice characteristics
that the GPT was trained on.

The embeddings include all components needed for full voice reproduction:
- spk_cond_emb: Speaker conditioning for GPT (W2V-BERT features)
- style: CAMPPlus speaker style vector
- prompt_condition: Conditioning for S2Mel (acoustic detail)
- ref_mel: Reference mel spectrogram (voice reconstruction target)
- emo_cond_emb: Emotion conditioning (optional, defaults to speaker)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
import torchaudio
import librosa
import numpy as np


class SpeakerEmbeddingStore:
    """
    Store and load speaker embeddings for promptless inference.
    
    This ensures your finetuned/LoRA model uses consistent voice characteristics
    by storing all embeddings from representative audio samples.
    
    Usage:
        # SETUP: Extract and save embeddings from your voice samples
        store = SpeakerEmbeddingStore("./speaker_embeddings")
        store.extract_and_save(tts_model, "my_voice_sample.wav", "my_voice")
        
        # Or extract from multiple samples for more stable embeddings:
        embeddings = extract_multiple_utterances(tts_model, audio_paths)
        store.save(embeddings, "my_voice_averaged")
        
        # INFERENCE: Load and use stored embeddings
        embeddings = store.load("my_voice", device="cuda")
        tts.infer(
            text="Hello world",
            output_path="output.wav",
            speaker_embeddings=embeddings  # Use stored embeddings!
        )
    """
    
    def __init__(self, tts_model: 'IndexTTS2' = None, storage_dir: Union[str, Path] = "./speaker_embeddings"):
        """
        Initialize the embedding store.
        
        Args:
            tts_model: Optional IndexTTS2 model instance (required for extraction)
            storage_dir: Directory to store embeddings (for directory-based storage)
        """
        self.tts = tts_model
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_embeddings(
        self,
        audio_path: Union[str, Path],
        tts: 'IndexTTS2' = None,  # Optional, uses self.tts if not provided
        max_audio_length: float = 15.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all speaker embeddings from a reference audio file.
        
        This extracts ALL components needed for inference:
        - spk_cond_emb: W2V-BERT semantic features for GPT conditioning
        - style: CAMPPlus style vector for voice characteristics
        - prompt_condition: S2Mel prompt for acoustic reconstruction
        - ref_mel: Reference mel spectrogram
        - emo_cond_emb: Emotion conditioning (same as speaker for neutral)
        
        Args:
            tts: Initialized IndexTTS2 model
            audio_path: Path to speaker's reference audio
            max_audio_length: Maximum audio length in seconds
            
        Returns:
            Dictionary containing all speaker embeddings
        """
        tts = tts or self.tts
        if tts is None:
            raise ValueError("TTS model must be provided either in constructor or as argument")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and preprocess audio
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        
        max_audio_samples = int(max_audio_length * sr)
        if len(audio) > max_audio_samples:
            warnings.warn(f"Audio truncated from {len(audio)/sr:.1f}s to {max_audio_length}s")
            audio = audio[:max_audio_samples]
        
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        
        # Resample to required rates
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio_tensor)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)
        
        device = tts.device
        
        with torch.no_grad():
            # === Extract W2V-BERT semantic features ===
            # These are used for GPT speaker conditioning
            inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            spk_cond_emb = tts.get_emb(input_features, attention_mask)
            # Shape: (1, seq_len, 1024)
            
            # === Extract semantic codes for S2Mel prompt ===
            _, S_ref = tts.semantic_codec.quantize(spk_cond_emb)
            
            # === Extract mel spectrogram ===
            ref_mel = tts.mel_fn(audio_22k.to(device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)
            # Shape: (1, n_mels, time)
            
            # === Extract CAMPPlus style vector ===
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = tts.campplus_model(feat.unsqueeze(0))
            # Shape: (1, 192)
            
            # === Extract S2Mel prompt conditioning ===
            prompt_condition = tts.s2mel.models['length_regulator'](
                S_ref,
                ylens=ref_target_lengths,
                n_quantizers=3,
                f0=None
            )[0]
            # Shape: (1, mel_time, dim)
            
            # === Emotion conditioning (same as speaker for neutral emotion) ===
            emo_cond_emb = spk_cond_emb.clone()
        
        embeddings = {
            'spk_cond_emb': spk_cond_emb.cpu(),
            'style': style.cpu(),
            'prompt_condition': prompt_condition.cpu(),
            'ref_mel': ref_mel.cpu(),
            'emo_cond_emb': emo_cond_emb.cpu(),
        }
        
        # Store auxiliary info for debugging
        embeddings['_metadata'] = {
            'audio_path': str(audio_path),
            'audio_duration': len(audio) / sr,
            'shapes': {k: list(v.shape) for k, v in embeddings.items() if isinstance(v, torch.Tensor)},
        }
        
        return embeddings
    
    def save(
        self,
        embeddings: Dict[str, torch.Tensor],
        speaker_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save speaker embeddings to disk.
        
        Args:
            embeddings: Dictionary of embedding tensors
            speaker_name: Name identifier for the speaker
            metadata: Optional metadata (source audio, date, etc.)
            
        Returns:
            Path to saved embedding directory
        """
        speaker_dir = self.storage_dir / speaker_name
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each embedding as separate file
        embedding_keys = []
        shapes = {}
        for name, tensor in embeddings.items():
            if name.startswith('_'):
                continue  # Skip metadata
            if isinstance(tensor, torch.Tensor):
                torch.save(tensor, speaker_dir / f"{name}.pt")
                embedding_keys.append(name)
                shapes[name] = list(tensor.shape)
        
        # Save metadata
        meta = metadata or {}
        meta['embedding_keys'] = embedding_keys
        meta['shapes'] = shapes
        
        # Include internal metadata if present
        if '_metadata' in embeddings:
            meta.update(embeddings['_metadata'])
        
        with open(speaker_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"Speaker embeddings saved to: {speaker_dir}")
        print(f"  Keys: {embedding_keys}")
        print(f"  Shapes: {shapes}")
        return speaker_dir
    
    def save_embeddings(
        self,
        embeddings: Dict[str, torch.Tensor],
        path: Union[str, Path],
    ) -> Path:
        """
        Save embeddings to a single .pt file (simpler format for distribution).
        
        Args:
            embeddings: Dictionary of embedding tensors
            path: Path to save the .pt file
            
        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove internal metadata before saving
        to_save = {k: v for k, v in embeddings.items() if not k.startswith('_')}
        torch.save(to_save, path)
        
        print(f"Speaker embeddings saved to: {path}")
        for key, tensor in to_save.items():
            print(f"  {key}: {tensor.shape}")
        
        return path
    
    def load_embeddings(
        self,
        path: Union[str, Path],
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load embeddings from a single .pt file.
        
        Args:
            path: Path to the .pt file
            device: Device to load tensors to
            
        Returns:
            Dictionary of embedding tensors
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        
        embeddings = torch.load(path, map_location=device or 'cpu')
        
        print(f"Loaded speaker embeddings from: {path}")
        for key, tensor in embeddings.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {key}: {tensor.shape}")
        
        return embeddings
    
    def extract_and_save(
        self,
        tts: 'IndexTTS2' = None,
        audio_path: Union[str, Path] = None,
        speaker_name: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Extract embeddings from audio and save them.
        
        Args:
            tts: Initialized IndexTTS2 model (optional if set in constructor)
            audio_path: Path to speaker's reference audio
            speaker_name: Name identifier for the speaker
            metadata: Optional metadata
            
        Returns:
            Path to saved embedding directory
        """
        print(f"Extracting embeddings from: {audio_path}")
        embeddings = self.extract_embeddings(audio_path, tts=tts)
        
        meta = metadata or {}
        meta['source_audio'] = str(audio_path)
        
        return self.save(embeddings, speaker_name, meta)
    
    def load(
        self,
        speaker_name: str,
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load speaker embeddings from disk.
        
        Args:
            speaker_name: Name identifier for the speaker
            device: Device to load tensors to (default: CPU)
            
        Returns:
            Dictionary of embedding tensors ready for inference
        """
        speaker_dir = self.storage_dir / speaker_name
        
        if not speaker_dir.exists():
            available = self.list_speakers()
            raise FileNotFoundError(
                f"Speaker embeddings not found: {speaker_name}\n"
                f"Available speakers: {available}"
            )
        
        # Load metadata to get embedding keys
        meta_path = speaker_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
            
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        embeddings = {}
        for key in metadata['embedding_keys']:
            tensor_path = speaker_dir / f"{key}.pt"
            if tensor_path.exists():
                tensor = torch.load(tensor_path, map_location=device or 'cpu')
                embeddings[key] = tensor
            else:
                warnings.warn(f"Missing embedding file: {tensor_path}")
        
        print(f"Loaded speaker embeddings: {speaker_name}")
        print(f"  Device: {device or 'cpu'}")
        print(f"  Keys: {list(embeddings.keys())}")
        
        return embeddings
    
    def list_speakers(self) -> List[str]:
        """List all available speaker embeddings."""
        speakers = []
        for d in self.storage_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                speakers.append(d.name)
        return sorted(speakers)
    
    def delete(self, speaker_name: str) -> bool:
        """Delete a speaker's embeddings."""
        speaker_dir = self.storage_dir / speaker_name
        if speaker_dir.exists():
            import shutil
            shutil.rmtree(speaker_dir)
            print(f"Deleted speaker: {speaker_name}")
            return True
        return False
    
    def extract_averaged_embeddings(
        self,
        audio_paths: List[Union[str, Path]],
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract and average embeddings from multiple audio files.
        
        This provides more stable speaker embeddings by averaging across
        multiple samples. Recommended for production use.
        
        Args:
            audio_paths: List of paths to speaker's audio files
            verbose: Print progress for each file
            
        Returns:
            Averaged embeddings dictionary
        """
        if self.tts is None:
            raise ValueError("TTS model must be provided in constructor for this method")
        
        all_embeddings = []
        for i, path in enumerate(audio_paths):
            try:
                emb = self.extract_embeddings(path)
                all_embeddings.append(emb)
                if verbose:
                    print(f"  [{i+1}/{len(audio_paths)}] Extracted from: {path}")
            except Exception as e:
                warnings.warn(f"Failed to extract from {path}: {e}")
        
        if not all_embeddings:
            raise ValueError("No embeddings could be extracted from the provided audio files")
        
        print(f"Combining embeddings from {len(all_embeddings)} utterances...")
        
        # Keys that can be safely averaged (fixed-size or can be meaningfully combined)
        averageable_keys = {'spk_cond_emb', 'style', 'emo_cond_emb'}
        
        # Keys with variable sequence lengths - use the longest one for best quality
        variable_keys = {'ref_mel', 'prompt_condition'}
        
        combined = {}
        
        # Average the averageable keys
        for key in averageable_keys:
            tensors = [e[key] for e in all_embeddings if key in e]
            if tensors:
                # For sequence-based tensors, handle potential length differences
                if tensors[0].dim() == 3 and key in {'spk_cond_emb', 'emo_cond_emb'}:
                    # (1, seq_len, dim) - truncate to minimum length then average
                    min_seq_len = min(t.shape[1] for t in tensors)
                    truncated = [t[:, :min_seq_len, :] for t in tensors]
                    combined[key] = torch.stack(truncated).mean(dim=0)
                else:
                    # Fixed-size tensors like style (1, 192)
                    combined[key] = torch.stack(tensors).mean(dim=0)
                print(f"  Averaged {key}: shape {combined[key].shape}")
        
        # For variable-length keys, use the longest sample (better acoustic detail)
        for key in variable_keys:
            tensors = [e[key] for e in all_embeddings if key in e]
            if tensors:
                # Pick the longest
                if key == 'ref_mel':
                    # Shape (1, n_mels, time) - sort by time dimension
                    longest = max(tensors, key=lambda x: x.shape[-1])
                else:
                    # Shape (1, time, dim) - sort by time dimension
                    longest = max(tensors, key=lambda x: x.shape[1] if x.dim() > 2 else x.shape[0])
                combined[key] = longest
                print(f"  Selected longest {key}: shape {combined[key].shape}")
        
        return combined


def extract_multiple_utterances(
    tts: 'IndexTTS2',
    audio_paths: List[Union[str, Path]],
    storage_dir: str = "./speaker_embeddings"
) -> Dict[str, torch.Tensor]:
    """
    Extract and average embeddings from multiple audio files for better speaker representation.
    
    NOTE: Only certain embeddings can be safely averaged:
    - spk_cond_emb: Can be averaged (semantic features)
    - style: Can be averaged (speaker style vector)
    - emo_cond_emb: Can be averaged (emotion features)
    
    For variable-length embeddings (ref_mel, prompt_condition), we use the LONGEST
    sample as these provide more acoustic detail for reconstruction.
    
    Args:
        tts: Initialized IndexTTS2 model
        audio_paths: List of paths to speaker's audio files
        storage_dir: Directory for the embedding store
        
    Returns:
        Combined/averaged embeddings dictionary
    """
    store = SpeakerEmbeddingStore(storage_dir)
    
    all_embeddings = []
    for path in audio_paths:
        try:
            emb = store.extract_embeddings(tts, path)
            all_embeddings.append(emb)
            print(f"  Extracted from: {path}")
        except Exception as e:
            warnings.warn(f"Failed to extract from {path}: {e}")
    
    if not all_embeddings:
        raise ValueError("No embeddings could be extracted from the provided audio files")
    
    print(f"Combining embeddings from {len(all_embeddings)} utterances...")
    
    # Keys that can be safely averaged (fixed-size or can be meaningfully combined)
    averageable_keys = {'spk_cond_emb', 'style', 'emo_cond_emb'}
    
    # Keys with variable sequence lengths - use the longest one for best quality
    variable_keys = {'ref_mel', 'prompt_condition'}
    
    combined = {}
    
    # Average the averageable keys
    for key in averageable_keys:
        tensors = [e[key] for e in all_embeddings if key in e]
        if tensors:
            # For sequence-based tensors, we need to handle potential length differences
            if tensors[0].dim() == 3 and key in {'spk_cond_emb', 'emo_cond_emb'}:
                # (1, seq_len, dim) - average along batch, handle seq_len differences
                # Find the minimum sequence length to safely average
                min_seq_len = min(t.shape[1] for t in tensors)
                truncated = [t[:, :min_seq_len, :] for t in tensors]
                combined[key] = torch.stack(truncated).mean(dim=0)
            else:
                # Fixed-size tensors like style (1, 192)
                combined[key] = torch.stack(tensors).mean(dim=0)
            print(f"  Averaged {key}: shape {combined[key].shape}")
    
    # For variable-length keys, use the longest sample (better acoustic detail)
    for key in variable_keys:
        tensors = [(e[key], e.get('_metadata', {}).get('audio_duration', 0))
                   for e in all_embeddings if key in e]
        if tensors:
            # Sort by duration/length and pick the longest
            if key == 'ref_mel':
                # Shape (1, n_mels, time) - sort by time dimension
                longest = max(tensors, key=lambda x: x[0].shape[-1])
            else:
                # Shape (1, time, dim) - sort by time dimension
                longest = max(tensors, key=lambda x: x[0].shape[1] if x[0].dim() > 2 else x[0].shape[0])
            combined[key] = longest[0]
            print(f"  Selected longest {key}: shape {combined[key].shape}")
    
    return combined


def extract_from_training_manifest(
    tts: 'IndexTTS2',
    manifest_path: Union[str, Path],
    speaker_name: str,
    storage_dir: str = "./speaker_embeddings",
    max_samples: int = 10,
) -> Path:
    """
    Extract speaker embeddings from training data manifest.
    
    This is the RECOMMENDED way to create speaker embeddings for a finetuned model,
    as it uses the same audio files the model was trained on.
    
    Args:
        tts: Initialized IndexTTS2 model (with LoRA or finetuned weights loaded)
        manifest_path: Path to training manifest JSONL
        speaker_name: Name to save embeddings under
        storage_dir: Directory to store embeddings
        max_samples: Maximum number of samples to use
        
    Returns:
        Path to saved embeddings
    """
    import random
    
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    # Read audio paths from manifest
    audio_paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            audio_path = entry.get("audio_path")
            if audio_path and Path(audio_path).exists():
                audio_paths.append(audio_path)
    
    if not audio_paths:
        raise ValueError(f"No valid audio paths found in manifest: {manifest_path}")
    
    print(f"Found {len(audio_paths)} audio files in manifest")
    
    # Sample if we have too many
    if len(audio_paths) > max_samples:
        audio_paths = random.sample(audio_paths, max_samples)
        print(f"Sampled {max_samples} files for embedding extraction")
    
    # Extract and combine embeddings
    embeddings = extract_multiple_utterances(tts, audio_paths, storage_dir)
    
    # Save
    store = SpeakerEmbeddingStore(storage_dir)
    return store.save(
        embeddings,
        speaker_name,
        metadata={
            "source": "training_manifest",
            "manifest": str(manifest_path),
            "num_samples": len(audio_paths),
        }
    )