"""
Pattern-Conditioned Training Support for IndexTTS2

This module provides the key insight for making speaking pattern training work:

THE PROBLEM:
============
IndexTTS2's architecture extracts conditioning from each individual audio file.
When you train on 100 audio samples, you get 100 DIFFERENT conditioning vectors.
At inference, when you use a reference audio, you get YET ANOTHER conditioning
vector that the model has never seen.

The semantic patterns (pauses, stutters, hesitations) ARE encoded in the GPT output,
but they're CONDITIONED on specific embeddings the model saw during training.
Using different conditioning at inference = no patterns.

THE SOLUTION:
=============
Use CONSISTENT conditioning:
1. Extract conditioning from your training audio (averaged across samples)
2. Use this SAME conditioning for ALL training samples
3. Use this SAME conditioning at inference

This way, the model learns:
  "When I see THIS conditioning + any text → add these speaking patterns"

Instead of:
  "When I see conditioning_X + text_X → patterns (but only for X)"
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

import torch
import torchaudio
import librosa


class PatternConditioningStore:
    """
    Stores and manages speaker conditioning for pattern training.
    
    This is the KEY to making speaking pattern training work!
    
    Unlike regular speaker embeddings (for voice timbre), this stores the
    GPT-level conditioning that determines speaking PATTERNS.
    
    Usage:
        # DURING DATASET PREPARATION:
        store = PatternConditioningStore()
        
        # Extract from all training audio files
        store.extract_global_conditioning(tts_model, audio_paths)
        
        # Save for later use
        store.save("training/speaker/pattern_conditioning.pt")
        
        # DURING TRAINING:
        # The prepare_pattern_dataset.py will use this global conditioning
        # for ALL samples instead of per-sample conditioning
        
        # DURING INFERENCE:
        conditioning = PatternConditioningStore.load("training/speaker/pattern_conditioning.pt")
        tts.infer(text="...", pattern_conditioning=conditioning)
    """
    
    def __init__(self, tts_model: 'IndexTTS2' = None):
        """
        Initialize the conditioning store.
        
        Args:
            tts_model: Optional IndexTTS2 model (required for extraction)
        """
        self.tts = tts_model
        self.conditioning = None
        self.metadata = {}
    
    def extract_conditioning_from_audio(
        self,
        audio_path: Union[str, Path],
        max_audio_length: float = 15.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract GPT conditioning from a single audio file.
        
        This extracts the SAME conditioning used during training:
        - spk_cond_emb: Raw W2V-BERT features (before perceiver)
        - gpt_conditioning: Perceiver-compressed conditioning (what GPT sees)
        - emo_cond: Emotion conditioning vector
        
        Args:
            audio_path: Path to audio file
            max_audio_length: Maximum audio length in seconds
            
        Returns:
            Dictionary with conditioning tensors
        """
        if self.tts is None:
            raise ValueError("TTS model required for extraction")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        max_samples = int(max_audio_length * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)
        
        device = self.tts.device
        
        with torch.no_grad():
            # Extract W2V-BERT features (same as training)
            inputs = self.tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Get raw semantic embeddings
            spk_cond_emb = self.tts.get_emb(input_features, attention_mask)
            # Shape: (1, seq_len, 1024)
            
            # Get GPT conditioning (what the GPT model actually sees)
            cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
            gpt_conditioning = self.tts.gpt.get_conditioning(
                spk_cond_emb.transpose(1, 2), 
                cond_lengths
            )
            # Shape: (1, 32, model_dim) - the perceiver-compressed conditioning
            
            # Get emotion conditioning
            emo_cond = self.tts.gpt.get_emo_conditioning(
                spk_cond_emb.transpose(1, 2),
                cond_lengths
            )
            emo_vec = self.tts.gpt.emovec_layer(emo_cond)
            emo_vec = self.tts.gpt.emo_layer(emo_vec)
            # Shape: (1, model_dim)
        
        return {
            'spk_cond_emb': spk_cond_emb.cpu(),  # Raw features
            'gpt_conditioning': gpt_conditioning.cpu(),  # Perceiver output
            'emo_vec': emo_vec.cpu(),  # Emotion vector
            'audio_duration': len(audio) / sr,
            'audio_path': str(audio_path),
        }
    
    def extract_global_conditioning(
        self,
        audio_paths: List[Union[str, Path]],
        method: str = "average",
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract GLOBAL conditioning from multiple audio files.
        
        This is the key to pattern training - all samples should use
        the SAME conditioning so patterns aren't tied to specific embeddings.
        
        Args:
            audio_paths: List of audio files to extract from
            method: How to combine multiple extractions:
                    - "average": Average all embeddings (recommended)
                    - "first": Use first audio only
                    - "longest": Use the longest audio
            verbose: Print progress
            
        Returns:
            Consolidated conditioning dictionary
        """
        if not audio_paths:
            raise ValueError("At least one audio path required")
        
        if verbose:
            print(f"Extracting global conditioning from {len(audio_paths)} audio files...")
        
        all_conditioning = []
        for i, path in enumerate(audio_paths):
            try:
                cond = self.extract_conditioning_from_audio(path)
                all_conditioning.append(cond)
                if verbose:
                    print(f"  [{i+1}/{len(audio_paths)}] Extracted from: {path}")
            except Exception as e:
                warnings.warn(f"Failed to extract from {path}: {e}")
        
        if not all_conditioning:
            raise ValueError("No conditioning could be extracted")
        
        if method == "first":
            combined = all_conditioning[0]
        elif method == "longest":
            combined = max(all_conditioning, key=lambda x: x['audio_duration'])
        elif method == "average":
            combined = self._average_conditioning(all_conditioning)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store metadata
        combined['_metadata'] = {
            'num_samples': len(all_conditioning),
            'method': method,
            'audio_paths': [str(p) for p in audio_paths],
            'total_duration': sum(c['audio_duration'] for c in all_conditioning),
        }
        
        self.conditioning = combined
        
        if verbose:
            print(f"Global conditioning extracted:")
            print(f"  - Method: {method}")
            print(f"  - Samples used: {len(all_conditioning)}")
            print(f"  - GPT conditioning shape: {combined['gpt_conditioning'].shape}")
            print(f"  - Emotion vector shape: {combined['emo_vec'].shape}")
        
        return combined
    
    def _average_conditioning(
        self, 
        all_conditioning: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Average multiple conditioning extractions."""
        
        # For gpt_conditioning (1, 32, dim) - safe to average
        gpt_conds = torch.stack([c['gpt_conditioning'] for c in all_conditioning])
        avg_gpt = gpt_conds.mean(dim=0)
        
        # For emo_vec (1, dim) - safe to average
        emo_vecs = torch.stack([c['emo_vec'] for c in all_conditioning])
        avg_emo = emo_vecs.mean(dim=0)
        
        # For spk_cond_emb - variable length, truncate to minimum
        min_len = min(c['spk_cond_emb'].shape[1] for c in all_conditioning)
        spk_conds = torch.stack([c['spk_cond_emb'][:, :min_len, :] for c in all_conditioning])
        avg_spk = spk_conds.mean(dim=0)
        
        return {
            'spk_cond_emb': avg_spk,
            'gpt_conditioning': avg_gpt,
            'emo_vec': avg_emo,
            'audio_duration': sum(c['audio_duration'] for c in all_conditioning) / len(all_conditioning),
        }
    
    def save(self, path: Union[str, Path]) -> Path:
        """
        Save conditioning to file.
        
        Args:
            path: Path to save the .pt file
            
        Returns:
            Path to saved file
        """
        if self.conditioning is None:
            raise ValueError("No conditioning extracted yet")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove non-tensor items for saving
        to_save = {
            k: v for k, v in self.conditioning.items() 
            if isinstance(v, torch.Tensor)
        }
        to_save['_metadata'] = self.conditioning.get('_metadata', {})
        
        torch.save(to_save, path)
        print(f"Pattern conditioning saved to: {path}")
        
        return path
    
    @staticmethod
    def load(
        path: Union[str, Path],
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load conditioning from file.
        
        Args:
            path: Path to the .pt file
            device: Device to load tensors to
            
        Returns:
            Conditioning dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Conditioning not found: {path}")
        
        data = torch.load(path, map_location=device or 'cpu')
        
        print(f"Loaded pattern conditioning from: {path}")
        if '_metadata' in data:
            meta = data['_metadata']
            print(f"  - Samples: {meta.get('num_samples', 'N/A')}")
            print(f"  - Method: {meta.get('method', 'N/A')}")
        
        return data


def prepare_training_conditioning(
    audio_dir: Union[str, Path],
    output_path: Union[str, Path],
    tts_model: 'IndexTTS2' = None,
    max_files: int = 20,
    method: str = "average",
) -> Path:
    """
    Convenience function to extract and save pattern conditioning.
    
    Args:
        audio_dir: Directory containing training audio
        output_path: Where to save the conditioning file
        tts_model: IndexTTS2 model (loaded if not provided)
        max_files: Maximum number of audio files to use
        method: Combination method (average, first, longest)
        
    Returns:
        Path to saved conditioning file
    """
    audio_dir = Path(audio_dir)
    
    # Find audio files
    extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    audio_files = sorted(audio_files)[:max_files]
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files in {audio_dir}")
    
    print(f"Found {len(audio_files)} audio files for conditioning extraction")
    
    # Load model if needed
    if tts_model is None:
        from indextts.infer_v2 import IndexTTS2
        print("Loading IndexTTS2 model...")
        tts_model = IndexTTS2(use_cuda_kernel=False)
    
    # Extract and save
    store = PatternConditioningStore(tts_model)
    store.extract_global_conditioning(
        [str(f) for f in audio_files],
        method=method,
        verbose=True
    )
    
    return store.save(output_path)


def export_conditioning_for_training(
    conditioning_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Export conditioning to numpy format for training scripts.
    
    This converts the pattern conditioning to .npy files that can be
    used by prepare_pattern_dataset.py.
    
    Args:
        conditioning_path: Path to pattern_conditioning.pt
        output_dir: Directory to save .npy files
        
    Returns:
        Tuple of (condition_path, emo_vec_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conditioning = PatternConditioningStore.load(conditioning_path)
    
    # Export GPT conditioning
    gpt_cond = conditioning['gpt_conditioning'].squeeze(0).numpy().astype(np.float32)
    cond_path = output_dir / "global_condition.npy"
    np.save(cond_path, gpt_cond)
    
    # Export emotion vector
    emo_vec = conditioning['emo_vec'].squeeze(0).numpy().astype(np.float32)
    emo_path = output_dir / "global_emo_vec.npy"
    np.save(emo_path, emo_vec)
    
    print(f"Exported conditioning to:")
    print(f"  - {cond_path} (shape: {gpt_cond.shape})")
    print(f"  - {emo_path} (shape: {emo_vec.shape})")
    
    return cond_path, emo_path