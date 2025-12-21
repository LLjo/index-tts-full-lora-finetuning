"""
Pattern Embeddings for IndexTTS2 Speaking Pattern Training

THE CORE INSIGHT:
================
Previous training approaches failed because:
1. Training: codes contain patterns, but they're implicit targets
2. Loss: treats pause codes and non-pause codes equally  
3. Result: LoRA learns generic code prediction, not "add patterns"

THE SOLUTION:
=============
Explicit Pattern Embeddings - learnable vectors that:
1. Get injected into the GPT conditioning alongside speaker embeddings
2. Are trained with a pattern-aware loss that rewards pattern reproduction
3. Persist to inference - the model learns "when I see this embedding, add patterns"

This is similar to textual inversion or LoRA triggers in image generation.

Usage:
    # Training
    pattern_emb = PatternEmbedding(model_dim=1024, num_tokens=4)
    trainer = PatternAwareTrainer(model, pattern_emb)
    trainer.train(dataloader)
    
    # Inference  
    pattern_emb = PatternEmbedding.load("path/to/pattern_emb.pt")
    tts.infer(text="...", pattern_embedding=pattern_emb)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class PatternFeatures:
    """Extracted pattern features from audio."""
    pause_positions: List[int]  # Token positions where pauses occur
    pause_durations: List[float]  # Duration of each pause in ms
    filler_positions: List[int]  # Positions of filler words (uh, um)
    speech_rate: float  # Average speech rate
    rate_variations: List[float]  # Local speech rate variations
    stutter_positions: List[int]  # Positions of stutters/repetitions
    total_duration: float
    # Summary statistics (optional, for convenience)
    num_pauses: int = 0
    num_fillers: int = 0
    num_stutters: int = 0


class PatternEmbedding(nn.Module):
    """
    Learnable pattern embedding that conditions GPT to produce speaking patterns.
    
    This is the key to making pattern training work:
    - Instead of hoping the model implicitly learns patterns from codes
    - We provide an explicit, learnable "pattern trigger" embedding
    - This embedding is injected into the conditioning
    - During training, gradient flows through to this embedding
    - The embedding learns to encode "speak with these patterns"
    
    Architecture:
        [pattern_tokens] -> Linear -> conditioned embedding
        
    The pattern tokens are learnable parameters (like textual inversion).
    """
    
    def __init__(
        self,
        model_dim: int = 1024,
        num_pattern_tokens: int = 4,
        hidden_dim: Optional[int] = None,
        init_std: float = 0.02,
    ):
        """
        Args:
            model_dim: Dimension of GPT model (must match UnifiedVoice.model_dim)
            num_pattern_tokens: Number of learnable pattern tokens
            hidden_dim: Hidden dimension for pattern projection
            init_std: Standard deviation for initialization
        """
        super().__init__()
        
        self.model_dim = model_dim
        self.num_pattern_tokens = num_pattern_tokens
        hidden_dim = hidden_dim or model_dim
        
        # Learnable pattern tokens - THE CORE of pattern learning
        # These learn to encode "speak with this person's patterns"
        self.pattern_tokens = nn.Parameter(
            torch.randn(num_pattern_tokens, hidden_dim) * init_std
        )
        
        # Project to model dimension
        self.pattern_proj = nn.Sequential(
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        
        # Pattern intensity control (learnable scale)
        self.pattern_scale = nn.Parameter(torch.ones(1))
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.pattern_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate pattern conditioning for injection into GPT.
        
        Args:
            batch_size: Batch size for expanding
            
        Returns:
            Pattern embedding of shape (batch_size, num_pattern_tokens, model_dim)
        """
        # Project pattern tokens
        pattern_emb = self.pattern_proj(self.pattern_tokens)  # (num_tokens, model_dim)
        
        # Apply learnable scale
        pattern_emb = pattern_emb * self.pattern_scale
        
        # Expand for batch
        pattern_emb = pattern_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pattern_emb
    
    def get_injection_embedding(
        self,
        speaker_conditioning: torch.Tensor,
        injection_mode: str = "prepend",
    ) -> torch.Tensor:
        """
        Get the combined embedding for injection into GPT conditioning.
        
        Args:
            speaker_conditioning: Original speaker conditioning (B, cond_len, dim)
            injection_mode: How to combine with speaker conditioning:
                - "prepend": Add pattern tokens before speaker conditioning
                - "add": Add pattern embedding to speaker conditioning  
                - "replace_first": Replace first N tokens with pattern tokens
                
        Returns:
            Combined conditioning tensor
        """
        batch_size = speaker_conditioning.size(0)
        pattern_emb = self.forward(batch_size)
        device = speaker_conditioning.device
        pattern_emb = pattern_emb.to(device)
        
        if injection_mode == "prepend":
            # Prepend pattern tokens to conditioning
            return torch.cat([pattern_emb, speaker_conditioning], dim=1)
        
        elif injection_mode == "add":
            # Add pattern embedding to first N tokens of conditioning
            n_tokens = min(self.num_pattern_tokens, speaker_conditioning.size(1))
            result = speaker_conditioning.clone()
            result[:, :n_tokens, :] = result[:, :n_tokens, :] + pattern_emb[:, :n_tokens, :]
            return result
        
        elif injection_mode == "replace_first":
            # Replace first N tokens with pattern tokens
            n_tokens = min(self.num_pattern_tokens, speaker_conditioning.size(1))
            result = speaker_conditioning.clone()
            result[:, :n_tokens, :] = pattern_emb[:, :n_tokens, :]
            return result
        
        else:
            raise ValueError(f"Unknown injection mode: {injection_mode}")
    
    def save(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        """Save pattern embedding to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'state_dict': self.state_dict(),
            'config': {
                'model_dim': self.model_dim,
                'num_pattern_tokens': self.num_pattern_tokens,
            },
            'metadata': metadata or {},
        }
        torch.save(save_dict, path)
        print(f"Pattern embedding saved to: {path}")
    
    @classmethod
    def load(
        cls, 
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> 'PatternEmbedding':
        """Load pattern embedding from file."""
        path = Path(path)
        data = torch.load(path, map_location=device or 'cpu')
        
        config = data['config']
        embedding = cls(
            model_dim=config['model_dim'],
            num_pattern_tokens=config['num_pattern_tokens'],
        )
        embedding.load_state_dict(data['state_dict'])
        
        print(f"Loaded pattern embedding from: {path}")
        if 'metadata' in data:
            meta = data['metadata']
            print(f"  Trained for {meta.get('epochs', 'N/A')} epochs")
            print(f"  Final loss: {meta.get('final_loss', 'N/A')}")
        
        return embedding


class PatternExtractor:
    """
    Extract speaking pattern features from audio and transcripts.
    
    This analyzes audio to find:
    - Pause locations and durations
    - Filler words (uh, um)
    - Stutters and repetitions
    - Speech rate variations
    """
    
    # Semantic codes that typically correspond to silence/pauses
    # These are learned from IndexTTS2's semantic codec
    SILENCE_CODES = {52}  # Main silence token
    PAUSE_THRESHOLD_CODES = 3  # Consecutive silence codes = pause
    
    def __init__(self, silence_codes: Optional[set] = None):
        self.silence_codes = silence_codes or self.SILENCE_CODES
    
    def extract_from_codes(
        self,
        codes: np.ndarray,
        text: str,
        audio_duration: float,
    ) -> PatternFeatures:
        """
        Extract pattern features from semantic codes.
        
        Args:
            codes: Semantic codes from audio (1D array)
            text: Original transcript
            audio_duration: Duration of audio in seconds
            
        Returns:
            PatternFeatures with extracted information
        """
        codes = codes.flatten()
        
        # Find pause positions
        pause_positions = []
        pause_durations = []
        
        # Detect consecutive silence codes
        silence_run_start = None
        silence_run_length = 0
        
        for i, code in enumerate(codes):
            if code in self.silence_codes:
                if silence_run_start is None:
                    silence_run_start = i
                silence_run_length += 1
            else:
                if silence_run_length >= self.PAUSE_THRESHOLD_CODES:
                    pause_positions.append(silence_run_start)
                    # Estimate duration based on code position
                    duration_per_code = audio_duration / len(codes)
                    pause_durations.append(silence_run_length * duration_per_code * 1000)  # ms
                silence_run_start = None
                silence_run_length = 0
        
        # Check final run
        if silence_run_length >= self.PAUSE_THRESHOLD_CODES:
            pause_positions.append(silence_run_start)
            duration_per_code = audio_duration / len(codes)
            pause_durations.append(silence_run_length * duration_per_code * 1000)
        
        # Detect fillers from text
        filler_positions = self._find_fillers_in_text(text)
        
        # Detect stutters from text
        stutter_positions = self._find_stutters_in_text(text)
        
        # Calculate speech rate
        word_count = len(text.split())
        speech_rate = word_count / audio_duration if audio_duration > 0 else 0
        
        # Estimate rate variations (simple: based on code density)
        rate_variations = self._estimate_rate_variations(codes, window_size=50)
        
        return PatternFeatures(
            pause_positions=pause_positions,
            pause_durations=pause_durations,
            filler_positions=filler_positions,
            speech_rate=speech_rate,
            rate_variations=rate_variations,
            stutter_positions=stutter_positions,
            total_duration=audio_duration,
        )
    
    def _find_fillers_in_text(self, text: str) -> List[int]:
        """Find filler word positions in text."""
        import re
        
        filler_patterns = [
            r'\b(uh+|uhh+)\b',
            r'\b(um+|umm+)\b',
            r'\b(er+|err+)\b',
            r'\b(ah+|ahh+)\b',
            r'\[UH\]', r'\[UM\]', r'\[ER\]', r'\[AH\]',
        ]
        
        positions = []
        for pattern in filler_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                positions.append(match.start())
        
        return sorted(positions)
    
    def _find_stutters_in_text(self, text: str) -> List[int]:
        """Find stutter/repetition positions in text."""
        import re
        
        # Pattern: repeated word or syllable (like "I-I" or "the the")
        stutter_patterns = [
            r'\b(\w+)-\1\b',  # Word-word
            r'\b(\w+)\s+\1\b',  # Word word
            r'\b([A-Za-z])-\1-',  # Letter repetition
        ]
        
        positions = []
        for pattern in stutter_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                positions.append(match.start())
        
        return sorted(positions)
    
    def _estimate_rate_variations(
        self, 
        codes: np.ndarray,
        window_size: int = 50,
    ) -> List[float]:
        """Estimate speech rate variations from code density."""
        if len(codes) < window_size:
            return [1.0]
        
        variations = []
        for i in range(0, len(codes) - window_size, window_size // 2):
            window = codes[i:i + window_size]
            silence_ratio = sum(1 for c in window if c in self.silence_codes) / len(window)
            # High silence ratio = slower speech
            # Convert to rate multiplier (1.0 = average)
            rate = 1.0 - silence_ratio
            variations.append(max(0.1, rate * 2))  # Scale to ~0.1-2.0
        
        return variations


class PatternAwareLoss(nn.Module):
    """
    Pattern-aware loss function that explicitly rewards pattern reproduction.
    
    Standard CE loss treats all code predictions equally.
    This loss adds explicit terms for:
    1. Pause prediction accuracy at known pause positions
    2. Speech rate matching
    3. Pattern token reconstruction
    """
    
    def __init__(
        self,
        pause_weight: float = 2.0,
        rate_weight: float = 0.5,
        ce_weight: float = 1.0,
        silence_codes: Optional[set] = None,
    ):
        super().__init__()
        self.pause_weight = pause_weight
        self.rate_weight = rate_weight
        self.ce_weight = ce_weight
        self.silence_codes = silence_codes or {52}
    
    def forward(
        self,
        logits: torch.Tensor,  # (B, vocab, seq_len)
        targets: torch.Tensor,  # (B, seq_len)
        mask: torch.Tensor,  # (B, seq_len)
        pattern_features: Optional[List[PatternFeatures]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute pattern-aware loss.
        
        Args:
            logits: Model predictions
            targets: Target codes
            mask: Valid position mask
            pattern_features: Optional pattern features for pattern-aware weighting
            
        Returns:
            Total loss and metrics dict
        """
        metrics = {}
        
        # Standard CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        ce_loss = (ce_loss * mask).sum() / mask.sum().clamp_min(1)
        metrics['ce_loss'] = ce_loss.item()
        
        total_loss = self.ce_weight * ce_loss
        
        # Pattern-aware components
        if pattern_features is not None and len(pattern_features) > 0:
            # Pause prediction bonus
            pause_loss = self._compute_pause_loss(logits, targets, mask, pattern_features)
            if pause_loss is not None:
                total_loss = total_loss + self.pause_weight * pause_loss
                metrics['pause_loss'] = pause_loss.item()
            
            # Rate variation matching
            rate_loss = self._compute_rate_loss(logits, targets, mask, pattern_features)
            if rate_loss is not None:
                total_loss = total_loss + self.rate_weight * rate_loss
                metrics['rate_loss'] = rate_loss.item()
        
        metrics['total_loss'] = total_loss.item()
        
        return total_loss, metrics
    
    def _compute_pause_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        pattern_features: List[PatternFeatures],
    ) -> Optional[torch.Tensor]:
        """Compute loss that emphasizes pause prediction."""
        batch_size = logits.size(0)
        
        # Create mask for pause positions
        pause_mask = torch.zeros_like(mask, dtype=torch.float)
        
        for b in range(batch_size):
            if b < len(pattern_features) and pattern_features[b] is not None:
                for pos in pattern_features[b].pause_positions:
                    if pos < mask.size(1):
                        # Weight nearby positions too
                        for offset in range(-2, 3):
                            p = pos + offset
                            if 0 <= p < mask.size(1):
                                weight = 1.0 if offset == 0 else 0.5
                                pause_mask[b, p] = max(pause_mask[b, p], weight)
        
        if pause_mask.sum() == 0:
            return None
        
        # Compute weighted CE at pause positions
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pause_ce = (ce_loss * pause_mask * mask).sum() / pause_mask.sum().clamp_min(1)
        
        return pause_ce
    
    def _compute_rate_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        pattern_features: List[PatternFeatures],
    ) -> Optional[torch.Tensor]:
        """Compute loss that encourages matching speech rate patterns."""
        # This is more subtle - we want the model's silence predictions
        # to match the target's silence density
        
        batch_size = logits.size(0)
        device = logits.device
        
        # Get predicted silence probability
        silence_code = list(self.silence_codes)[0]  # Use first silence code
        pred_probs = F.softmax(logits, dim=1)  # (B, vocab, seq_len)
        
        if silence_code >= pred_probs.size(1):
            return None
        
        pred_silence_prob = pred_probs[:, silence_code, :]  # (B, seq_len)
        
        # Target silence indicator
        target_silence = (targets == silence_code).float()  # (B, seq_len)
        
        # Compute BCE between predicted silence probability and target
        rate_loss = F.binary_cross_entropy(
            pred_silence_prob * mask,
            target_silence * mask,
            reduction='sum'
        ) / mask.sum().clamp_min(1)
        
        return rate_loss


class PatternAwareTrainer:
    """
    Trainer that combines LoRA fine-tuning with pattern embedding learning.
    
    This trainer:
    1. Trains LoRA adapters on the GPT model (for voice adaptation)
    2. Simultaneously trains PatternEmbedding (for pattern injection)
    3. Uses pattern-aware loss to explicitly reward pattern reproduction
    """
    
    def __init__(
        self,
        model: nn.Module,
        pattern_embedding: PatternEmbedding,
        pattern_extractor: Optional[PatternExtractor] = None,
        learning_rate: float = 5e-4,
        pattern_lr: float = 1e-3,  # Higher LR for pattern embedding
        weight_decay: float = 0.01,
        injection_mode: str = "add",
    ):
        """
        Args:
            model: UnifiedVoice model (with or without LoRA)
            pattern_embedding: PatternEmbedding module to train
            pattern_extractor: Optional extractor for pattern features
            learning_rate: LR for model parameters
            pattern_lr: LR for pattern embedding (often higher)
            weight_decay: Weight decay
            injection_mode: How to inject pattern embedding
        """
        self.model = model
        self.pattern_embedding = pattern_embedding
        self.pattern_extractor = pattern_extractor or PatternExtractor()
        self.injection_mode = injection_mode
        
        # Separate optimizers for different components
        model_params = [p for p in model.parameters() if p.requires_grad]
        pattern_params = list(pattern_embedding.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': model_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': pattern_params, 'lr': pattern_lr, 'weight_decay': 0.0},  # No decay for embedding
        ])
        
        self.loss_fn = PatternAwareLoss()
        self.device = next(model.parameters()).device
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        pattern_features: Optional[List[PatternFeatures]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a batch with pattern embedding injection.
        
        This is the key difference from standard training:
        - We inject the pattern embedding into the conditioning
        - We use pattern-aware loss
        """
        # Get base model for PEFT compatibility
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        
        condition = batch["condition"].to(self.device)  # (B, cond_len, dim)
        text_ids = batch["text_ids"].to(self.device)
        codes = batch["codes"].to(self.device)
        emo_vec = batch["emo_vec"].to(self.device)
        text_lengths = batch["text_lengths"].to(self.device)
        code_lengths = batch["code_lengths"].to(self.device)
        
        batch_size = text_ids.size(0)
        
        # INJECT PATTERN EMBEDDING into conditioning
        # This is the key to making patterns learnable!
        pattern_conditioned = self.pattern_embedding.get_injection_embedding(
            condition,
            injection_mode=self.injection_mode,
        )
        
        # Add emotion vector
        emo_expanded = emo_vec.unsqueeze(1)
        
        # Build conditioning (same as original but with pattern injection)
        use_speed = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        duration_ctrl = base_model.speed_emb(torch.ones_like(use_speed))
        duration_free = base_model.speed_emb(torch.zeros_like(use_speed))
        
        # Original method uses: condition + emo_vec.unsqueeze(1)
        # We use: pattern_conditioned + emo_vec.unsqueeze(1)
        conds = torch.cat(
            (pattern_conditioned + emo_expanded, duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
            dim=1,
        )
        
        # Process text and mel inputs (same as original training)
        text_inputs = base_model.set_text_padding(text_ids.clone(), text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=base_model.stop_text_token)
        text_inputs, text_targets = base_model.build_aligned_inputs_and_targets(
            text_inputs, base_model.start_text_token, base_model.stop_text_token
        )
        
        mel_inputs = base_model.set_mel_padding(codes.clone(), code_lengths)
        mel_inputs = F.pad(mel_inputs, (0, 1), value=base_model.stop_mel_token)
        mel_inputs, mel_targets = base_model.build_aligned_inputs_and_targets(
            mel_inputs, base_model.start_mel_token, base_model.stop_mel_token
        )
        
        # Embed
        text_emb = base_model.text_embedding(text_inputs) + base_model.text_pos_embedding(text_inputs)
        mel_emb = base_model.mel_embedding(mel_inputs) + base_model.mel_pos_embedding(mel_inputs)
        
        # Get logits
        text_logits, mel_logits = base_model.get_logits(
            conds, text_emb, base_model.text_head, mel_emb, base_model.mel_head
        )
        
        # Masks
        text_mask = (
            torch.arange(text_targets.size(1), device=self.device).unsqueeze(0)
            < (text_lengths + 1).unsqueeze(1)
        )
        mel_mask = (
            torch.arange(mel_targets.size(1), device=self.device).unsqueeze(0)
            < (code_lengths + 1).unsqueeze(1)
        )
        
        # Pattern-aware mel loss
        mel_loss, mel_metrics = self.loss_fn(
            mel_logits, mel_targets, mel_mask,
            pattern_features=pattern_features,
        )
        
        # Standard text loss
        text_ce = F.cross_entropy(text_logits, text_targets, reduction='none')
        text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
        
        # Combined loss
        total_loss = 0.2 * text_loss + 0.8 * mel_loss
        
        metrics = {
            'text_loss': text_loss.item(),
            'mel_loss': mel_metrics['total_loss'],
            **{f'mel_{k}': v for k, v in mel_metrics.items()},
            'pattern_scale': self.pattern_embedding.pattern_scale.item(),
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        pattern_features: Optional[List[PatternFeatures]] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.pattern_embedding.train()
        
        self.optimizer.zero_grad()
        
        loss, metrics = self.compute_loss(batch, pattern_features)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.pattern_embedding.parameters(), 1.0)
        
        self.optimizer.step()
        
        return metrics


def inject_pattern_for_inference(
    conditioning: torch.Tensor,
    pattern_embedding: PatternEmbedding,
    injection_mode: str = "add",
) -> torch.Tensor:
    """
    Inject pattern embedding into conditioning for inference.
    
    This is what makes patterns appear at inference time:
    - Load trained pattern embedding
    - Inject it into the speaker conditioning
    - The model now "knows" to produce patterns
    
    Args:
        conditioning: Original GPT conditioning (B, cond_len, dim)
        pattern_embedding: Trained PatternEmbedding
        injection_mode: How to inject
        
    Returns:
        Modified conditioning with pattern embedding
    """
    pattern_embedding.eval()
    with torch.no_grad():
        return pattern_embedding.get_injection_embedding(conditioning, injection_mode)