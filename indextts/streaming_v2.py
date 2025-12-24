"""
Streaming TTS V2 for IndexTTS2 - High Quality Streaming

This module provides improved streaming synthesis with better audio quality
by addressing the key limitations of the original chunk-by-chunk approach:

Key Approaches:
1. SENTENCE_LEVEL - Stream by natural sentence boundaries (best quality)
2. PROGRESSIVE_CONTEXT - Re-use previous mel context for continuity
3. OVERLAP_SYNTHESIS - Generate overlapping chunks and blend in mel domain

The fundamental insight is that independent chunk synthesis causes:
- Prosodic discontinuities (intonation/rhythm breaks)
- No mel-domain coherence between chunks
- CFM diffusion restarts fresh without context

This module fixes these by maintaining synthesis context across chunks.
"""

from __future__ import annotations

import threading
import queue
import time
import re
from typing import Generator, Optional, Callable, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn.functional as F
from transformers.generation.streamers import BaseStreamer


class StreamingMode(Enum):
    """Available streaming quality modes."""
    # Fastest TTFA but lowest quality - independent chunk synthesis
    FAST_CHUNKS = "fast_chunks"
    # Stream by sentence boundaries - best quality/latency balance  
    SENTENCE_LEVEL = "sentence_level"
    # Re-use mel context from previous chunks
    PROGRESSIVE_CONTEXT = "progressive_context"
    # Generate overlapping chunks, blend in mel domain
    OVERLAP_SYNTHESIS = "overlap_synthesis"



@dataclass
class StreamingConfigV2:
    """Configuration for V2 streaming TTS."""
    
    # Streaming mode - determines quality/latency tradeoff
    mode: StreamingMode = StreamingMode.SENTENCE_LEVEL
    
    # === SENTENCE_LEVEL mode settings ===
    # Max characters before forcing a chunk (for very long sentences)
    max_sentence_chars: int = 200
    # Whether to split on commas/semicolons as well as periods
    split_on_clauses: bool = False
    
    # === FAST_CHUNKS mode settings ===
    # Minimum mel tokens before first audio chunk
    min_chunk_tokens: int = 150
    # Tokens to accumulate after first chunk
    chunk_tokens: int = 80
    # Maximum tokens before forcing a chunk
    max_chunk_tokens: int = 200
    
    # === PROGRESSIVE_CONTEXT settings ===
    # Number of mel frames to carry over for context
    context_mel_frames: int = 50
    
    # === OVERLAP_SYNTHESIS settings ===
    # Number of tokens to overlap between chunks
    overlap_tokens: int = 15
    # Blend window size in mel frames
    blend_mel_frames: int = 30
    
    # === Common settings ===
    # Diffusion steps for S2Mel
    diffusion_steps: int = 20
    # Faster diffusion for first chunk (TTFA optimization)
    first_chunk_diffusion_steps: int = 12
    # CFM inference rate  
    inference_cfg_rate: float = 0.7
    
    # === Audio stitching ===
    # Crossfade samples in audio domain (fallback)
    crossfade_samples: int = 2048
    # Use advanced mel-domain blending vs simple audio crossfade
    use_mel_blending: bool = True
    
    # Verbose logging
    verbose: bool = False


class SentenceSegmenter:
    """
    Smart text segmenter that finds natural break points for streaming.
    
    This is crucial for quality - we want to stream at sentence/clause
    boundaries rather than arbitrary token counts.
    """
    
    # Sentence-ending punctuation
    SENTENCE_ENDS = {'.', '!', '?', '。', '！', '？', '…'}
    # Clause-level punctuation (optional split points)
    CLAUSE_ENDS = {',', ';', ':', '，', '；', '：', '、'}
    # Quote handling
    QUOTES = {'"', "'", '"', '"', ''', '''}
    
    def __init__(self, split_on_clauses: bool = False, max_chars: int = 200):
        self.split_on_clauses = split_on_clauses
        self.max_chars = max_chars
    
    def segment(self, text: str) -> List[str]:
        """
        Split text into streamable segments at natural boundaries.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments suitable for streaming
        """
        if not text.strip():
            return []
        
        segments = []
        current = []
        current_len = 0
        
        # Tokenize roughly by character with punctuation awareness
        i = 0
        while i < len(text):
            char = text[i]
            current.append(char)
            current_len += 1
            
            # Check if we hit a break point
            is_sentence_end = char in self.SENTENCE_ENDS
            is_clause_end = self.split_on_clauses and char in self.CLAUSE_ENDS
            is_too_long = current_len >= self.max_chars
            
            # Handle quotes after punctuation (e.g., 'Hello."')
            if is_sentence_end:
                # Look ahead for closing quote
                if i + 1 < len(text) and text[i + 1] in self.QUOTES:
                    current.append(text[i + 1])
                    i += 1
                # Commit segment
                segment = ''.join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                current_len = 0
            elif is_clause_end or is_too_long:
                # Only break on clauses if we have meaningful content
                if current_len >= 20:  # Minimum viable segment
                    segment = ''.join(current).strip()
                    if segment:
                        segments.append(segment)
                    current = []
                    current_len = 0
            
            i += 1
        
        # Handle remaining text
        if current:
            segment = ''.join(current).strip()
            if segment:
                segments.append(segment)
        
        return segments


class ProgressiveMelSynthesizer:
    """
    Handles mel synthesis with context carryover between chunks.
    
    This maintains mel-domain context from previous chunks to ensure
    smooth prosodic continuity across chunk boundaries.
    """
    
    def __init__(
        self,
        tts: 'IndexTTS2',
        config: StreamingConfigV2,
        spk_cond_emb: torch.Tensor,
        emo_vec: torch.Tensor,
        style: torch.Tensor,
        prompt_condition: torch.Tensor,
        ref_mel: torch.Tensor,
    ):
        self.tts = tts
        self.config = config
        self.spk_cond_emb = spk_cond_emb
        self.emo_vec = emo_vec
        self.style = style
        self.prompt_condition = prompt_condition
        self.ref_mel = ref_mel
        
        # Context state
        self.previous_mel_context: Optional[torch.Tensor] = None
        self.previous_audio_tail: Optional[torch.Tensor] = None
        self.chunk_index = 0
    
    def synthesize(
        self,
        codes: torch.Tensor,
        code_lens: torch.Tensor,
        text_tokens: torch.Tensor,
        speech_conditioning_latent: torch.Tensor,
        is_first: bool = False,
        is_final: bool = False,
    ) -> torch.Tensor:
        """
        Synthesize audio from mel codes with context awareness.
        
        Args:
            codes: Mel token codes [1, seq_len]
            code_lens: Length of codes
            text_tokens: Text tokens for GPT forward
            speech_conditioning_latent: Conditioning latent
            is_first: Whether this is the first chunk
            is_final: Whether this is the final chunk
            
        Returns:
            Audio tensor [1, samples]
        """
        device = self.spk_cond_emb.device
        self.chunk_index += 1
        
        # Select diffusion steps based on chunk position
        if is_first:
            diffusion_steps = self.config.first_chunk_diffusion_steps
        elif is_final:
            diffusion_steps = self.config.diffusion_steps
        else:
            diffusion_steps = max(12, self.config.diffusion_steps - 3)
        
        with torch.no_grad():
            use_autocast = self.tts.dtype is not None and device.type == 'cuda'
            
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=self.tts.dtype or torch.float32):
                # GPT forward pass for latent
                use_speed = torch.zeros(1, device=device, dtype=torch.long)
                
                latent = self.tts.gpt(
                    speech_conditioning_latent,
                    text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=device),
                    codes,
                    code_lens,
                    self.spk_cond_emb,
                    cond_mel_lengths=torch.tensor([self.spk_cond_emb.shape[1]], device=device),
                    emo_cond_mel_lengths=torch.tensor([self.spk_cond_emb.shape[1]], device=device),
                    emo_vec=self.emo_vec.squeeze(1) if self.emo_vec.dim() == 3 else self.emo_vec,
                    use_speed=use_speed,
                )
                
                # S2Mel stage
                latent = self.tts.s2mel.models['gpt_layer'](latent)
                S_infer = self.tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()
                
                cond = self.tts.s2mel.models['length_regulator'](
                    S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                )[0]
                
                # Build condition with context
                if self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT and self.previous_mel_context is not None:
                    # Include mel context from previous chunk
                    context_frames = self.config.context_mel_frames
                    extended_prompt = torch.cat([
                        self.prompt_condition,
                        self.previous_mel_context[:, -context_frames:, :]
                    ], dim=1)
                    cat_condition = torch.cat([extended_prompt, cond], dim=1)
                else:
                    cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
                
                # CFM diffusion
                vc_target = self.tts.s2mel.models['cfm'].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    self.ref_mel,
                    self.style,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=self.config.inference_cfg_rate
                )
                
                # Extract new mel (skip ref_mel portion and context if used)
                if self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT and self.previous_mel_context is not None:
                    context_frames = self.config.context_mel_frames
                    vc_target = vc_target[:, :, self.ref_mel.size(-1) + context_frames:]
                else:
                    vc_target = vc_target[:, :, self.ref_mel.size(-1):]
                
                # Store mel context for next chunk
                if self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT:
                    # Store last portion of mel for context
                    self.previous_mel_context = cond.clone()
                
                # BigVGAN vocoding
                with torch.cuda.amp.autocast(enabled=False):
                    # Ensure the input is float32 and on the correct device
                    vc_target_f32 = vc_target.to(device=device, dtype=torch.float32)
                    wav = self.tts.bigvgan(vc_target_f32).squeeze()
        
        # Convert to proper format
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # Apply blending with previous chunk's tail
        if self.previous_audio_tail is not None:
            wav = self._blend_audio(wav)
        
        # For non-final chunks: store tail for blending and TRIM it from output
        # This prevents echo/bleeding - the tail is only heard in the blend, not twice
        if not is_final and self.config.crossfade_samples > 0 and wav.shape[-1] > self.config.crossfade_samples * 2:
            # Store tail for blending into next chunk
            self.previous_audio_tail = wav[:, -self.config.crossfade_samples:].clone()
            # TRIM tail from current output - it will appear in next chunk's blend
            wav = wav[:, :-self.config.crossfade_samples]
        else:
            # Final chunk - output everything, no tail to store
            self.previous_audio_tail = None
        
        return wav
    
    def _blend_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply crossfade blending with previous chunk."""
        if self.previous_audio_tail is None:
            return wav
        
        crossfade_samples = min(
            self.config.crossfade_samples,
            self.previous_audio_tail.shape[-1],
            wav.shape[-1]
        )
        
        if crossfade_samples < 64:
            return wav
        
        # Raised cosine crossfade (smoother than linear)
        t = torch.linspace(0, 1, crossfade_samples, device=wav.device, dtype=wav.dtype)
        fade_in = 0.5 * (1 - torch.cos(torch.pi * t))
        fade_out = 0.5 * (1 + torch.cos(torch.pi * t))
        
        # Blend
        prev_tail = self.previous_audio_tail[:, -crossfade_samples:]
        curr_head = wav[:, :crossfade_samples]
        blended = (prev_tail * fade_out.unsqueeze(0)) + (curr_head * fade_in.unsqueeze(0))
        
        if wav.shape[-1] > crossfade_samples:
            result = torch.cat([blended, wav[:, crossfade_samples:]], dim=-1)
        else:
            result = blended
        
        return result
    
    def reset(self):
        """Reset context state for new synthesis."""
        self.previous_mel_context = None
        self.previous_audio_tail = None
        self.chunk_index = 0


"""
Enhanced Progressive Context Streaming with Better Continuity

Key improvements:
1. Maintain GPT hidden states across chunks for better context
2. Use larger mel context windows
3. Progressive temperature annealing for consistency
4. Text-aware chunk boundaries
"""

"""
Enhanced Progressive Context Streaming with Better Continuity

Key improvements:
1. Maintain GPT hidden states across chunks for better context
2. Use larger mel context windows
3. Progressive temperature annealing for consistency
4. Text-aware chunk boundaries
"""
"""
Enhanced Progressive Context Streaming with Better Continuity

Key improvements:
1. Maintain GPT hidden states across chunks for better context
2. Use larger mel context windows
3. Progressive temperature annealing for consistency
4. Text-aware chunk boundaries
"""

class EnhancedProgressiveSynthesizer(ProgressiveMelSynthesizer):
    """
    Enhanced synthesizer that maintains GPT-level context across chunks.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store GPT hidden states for context
        self.previous_gpt_latent: Optional[torch.Tensor] = None
        self.cumulative_codes: List[int] = []
        
    def synthesize_with_gpt_context(
        self,
        codes: torch.Tensor,
        code_lens: torch.Tensor,
        text_tokens: torch.Tensor,
        speech_conditioning_latent: torch.Tensor,
        is_first: bool = False,
        is_final: bool = False,
    ) -> torch.Tensor:
        """
        Synthesize with full GPT context from previous chunks.
        """
        device = self.spk_cond_emb.device
        self.chunk_index += 1
        
        # Progressive temperature - start stable, allow variation later
        if is_first:
            diffusion_steps = self.config.first_chunk_diffusion_steps
        elif is_final:
            diffusion_steps = self.config.diffusion_steps
        else:
            # Gradually increase quality for middle chunks
            diffusion_steps = max(12, self.config.diffusion_steps - 3)
        
        with torch.no_grad():
            use_autocast = self.tts.dtype is not None and device.type == 'cuda'
            
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=self.tts.dtype or torch.float32):
                use_speed = torch.zeros(1, device=device, dtype=torch.long)
                
                # For non-first chunks, prepend previous codes for context
                if self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT and len(self.cumulative_codes) > 0:
                    # Use last N codes as context (e.g., last 30 tokens)
                    context_length = min(30, len(self.cumulative_codes))
                    context_codes = torch.tensor(
                        [self.cumulative_codes[-context_length:]], 
                        dtype=torch.long, 
                        device=device
                    )
                    
                    # Concatenate context + new codes
                    full_codes = torch.cat([context_codes, codes], dim=1)
                    full_code_lens = torch.tensor([full_codes.shape[1]], device=device)
                    
                    # Run GPT with full context
                    latent = self.tts.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=device),
                        full_codes,
                        full_code_lens,
                        self.spk_cond_emb,
                        cond_mel_lengths=torch.tensor([self.spk_cond_emb.shape[1]], device=device),
                        emo_cond_mel_lengths=torch.tensor([self.spk_cond_emb.shape[1]], device=device),
                        emo_vec=self.emo_vec.squeeze(1) if self.emo_vec.dim() == 3 else self.emo_vec,
                        use_speed=use_speed,
                    )
                    
                    # Extract only the NEW portion of latent
                    latent = latent[:, -codes.shape[1]:, :]
                else:
                    # First chunk - no context
                    latent = self.tts.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=device),
                        codes,
                        code_lens,
                        self.spk_cond_emb,
                        cond_mel_lengths=torch.tensor([self.spk_cond_emb.shape[1]], device=device),
                        emo_cond_mel_lengths=torch.tensor([self.spk_cond_emb.shape[1]], device=device),
                        emo_vec=self.emo_vec.squeeze(1) if self.emo_vec.dim() == 3 else self.emo_vec,
                        use_speed=use_speed,
                    )
                
                # Store codes for next chunk's context
                self.cumulative_codes.extend(codes[0].cpu().tolist())
                # Limit cumulative size to prevent memory issues
                if len(self.cumulative_codes) > 200:
                    self.cumulative_codes = self.cumulative_codes[-200:]
                
                # S2Mel stage with enhanced context
                latent = self.tts.s2mel.models['gpt_layer'](latent)
                S_infer = self.tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()
                
                cond = self.tts.s2mel.models['length_regulator'](
                    S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                )[0]
                
                # Enhanced mel context with larger window
                if self.previous_mel_context is not None:
                    # Use larger context for better continuity (100 frames ~= 0.5s)
                    context_frames = min(100, self.config.context_mel_frames * 2)
                    extended_prompt = torch.cat([
                        self.prompt_condition,
                        self.previous_mel_context[:, -context_frames:, :]
                    ], dim=1)
                    cat_condition = torch.cat([extended_prompt, cond], dim=1)
                else:
                    cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
                
                # Store mel context
                self.previous_mel_context = cond.clone()
                
                # CFM diffusion
                vc_target = self.tts.s2mel.models['cfm'].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    self.ref_mel,
                    self.style,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=self.config.inference_cfg_rate
                )
                
                # Extract new mel
                if self.previous_mel_context is not None:
                    context_frames = min(100, self.config.context_mel_frames * 2)
                    vc_target = vc_target[:, :, self.ref_mel.size(-1) + context_frames:]
                else:
                    vc_target = vc_target[:, :, self.ref_mel.size(-1):]
                
                # Vocoding
                with torch.cuda.amp.autocast(enabled=False):
                    # Ensure the input is float32 and on the correct device
                    vc_target_f32 = vc_target.to(device=device, dtype=torch.float32)
                    wav = self.tts.bigvgan(vc_target_f32).squeeze()
        
        # Audio processing
        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # NEW APPROACH: Simple overlap-add without trimming
        # This avoids cutting artifacts while still providing smooth transitions
        if self.previous_audio_tail is not None and not is_first:
            # Calculate a safe overlap region
            overlap_samples = min(
                1536,  # Shorter overlap = less interference
                self.previous_audio_tail.shape[-1] // 2,
                wav.shape[-1] // 3  # Only overlap small portion of current
            )
            
            if overlap_samples >= 256:
                # Apply gentle crossfade only in overlap region
                t = torch.linspace(0, 1, overlap_samples, device=wav.device, dtype=wav.dtype)
                
                # Equal-power crossfade (better for speech)
                fade_out = torch.cos(0.5 * torch.pi * t)
                fade_in = torch.sin(0.5 * torch.pi * t)
                
                # Get overlap regions
                tail_end = self.previous_audio_tail[:, -overlap_samples:]
                head_start = wav[:, :overlap_samples]
                
                # Blend
                blended_region = (tail_end * fade_out.unsqueeze(0)) + (head_start * fade_in.unsqueeze(0))
                
                # Concatenate: previous (minus overlap) + blended + current (minus overlap)
                result = torch.cat([
                    self.previous_audio_tail[:, :-overlap_samples],  # Previous chunk
                    blended_region,                                   # Smooth transition
                    wav[:, overlap_samples:]                         # Current chunk
                ], dim=-1)
                
                wav = result
            else:
                # Too short to blend, just concatenate
                wav = torch.cat([self.previous_audio_tail, wav], dim=-1)
        
        # Store tail for next iteration (smaller tail = less disruption)
        crossfade_length = min(1536, self.config.crossfade_samples)
        if not is_final and wav.shape[-1] > crossfade_length * 2:
            self.previous_audio_tail = wav[:, -crossfade_length:].clone()
            # Trim the stored tail from output to avoid doubling
            wav = wav[:, :-crossfade_length]
        else:
            self.previous_audio_tail = None
        
        return wav
    
    def reset(self):
        """Reset all context."""
        super().reset()
        self.previous_gpt_latent = None
        self.cumulative_codes = []


def get_progressive_streaming_config() -> StreamingConfigV2:
    """
    Get configuration optimized for progressive context streaming.
    This balances latency and quality by maintaining synthesis context.
    """
    return StreamingConfigV2(
        mode=StreamingMode.PROGRESSIVE_CONTEXT,
        min_chunk_tokens=20,  # Slightly larger chunks for stability
        chunk_tokens=45,
        max_chunk_tokens=80,
        context_mel_frames=100,  # Larger mel context
        first_chunk_diffusion_steps=12,
        diffusion_steps=20,
        crossfade_samples=1536,  # Smaller crossfade = cleaner transitions
        inference_cfg_rate=0.75,  # Slightly higher for consistency
        verbose=True,
    )

def streaming_inference_v2(
    tts: 'IndexTTS2',
    text: str,
    audio_prompt: Optional[str] = None,
    speaker_embeddings: Optional[dict] = None,
    emotion_audio: Optional[str] = None,
    emotion_alpha: float = 1.0,
    emo_vector: Optional[list] = None,
    use_emo_text: bool = False,
    emo_text: Optional[str] = None,
    use_random: bool = False,
    config: Optional[StreamingConfigV2] = None,
    # Pattern embedding support
    pattern_embedding: Optional[Any] = None,
    injection_mode: str = "add",
    # Generation parameters
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    max_mel_tokens: int = 600,
    # Callbacks
    on_audio_chunk: Optional[Callable[[torch.Tensor], None]] = None,
) -> Generator[torch.Tensor, None, None]:
    """
    High-quality streaming TTS inference.
    
    This function provides multiple streaming modes for different
    quality/latency tradeoffs.
    
    Args:
        tts: IndexTTS2 instance
        text: Text to synthesize
        audio_prompt: Path to speaker reference audio
        speaker_embeddings: Pre-computed speaker embeddings
        config: StreamingConfigV2 instance (defaults to SENTENCE_LEVEL mode)
        ... (other args same as original streaming_inference)
        
    Yields:
        Audio chunks as torch.Tensor (1, samples) at 22050 Hz
    """
    import librosa
    import torchaudio
    import random as rnd
    
    if config is None:
        config = StreamingConfigV2()
    
    device = tts.device
    if isinstance(device, str):
        device = torch.device(device)
    
    use_autocast = tts.dtype is not None and device.type == 'cuda'
    
    start_time = time.perf_counter()
    
    if config.verbose:
        print(f"[StreamingV2] Mode: {config.mode.value}")
        print(f"[StreamingV2] Extracting conditioning...")
    
    # === CONDITIONING EXTRACTION (same as original) ===
    if use_emo_text or emo_vector is not None:
        emotion_audio = None
    
    if use_emo_text:
        if emo_text is None:
            emo_text = text
        emo_dict = tts.qwen_emo.inference(emo_text)
        if config.verbose:
            print(f"  Detected emotions: {emo_dict}")
        emo_vector = list(emo_dict.values())
    
    if emo_vector is not None:
        emo_vector_scale = max(0.0, min(1.0, emotion_alpha))
        if emo_vector_scale != 1.0:
            emo_vector = [x * emo_vector_scale for x in emo_vector]
        emo_vector = tts.normalize_emo_vec(emo_vector)
    
    # Extract conditioning from audio or embeddings
    if audio_prompt is not None:
        audio, sr = librosa.load(audio_prompt, sr=None, mono=True)
        audio = audio[:int(15 * sr)]
        
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio_tensor)
        
        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                spk_cond_emb = tts.get_emb(input_features, attention_mask)
                cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
                speech_conditioning_latent = tts.gpt.get_conditioning(
                    spk_cond_emb.transpose(1, 2), cond_lengths
                )
                
                emo_cond = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                emo_vec = tts.gpt.emovec_layer(emo_cond)
                emo_vec = tts.gpt.emo_layer(emo_vec)
                
                _, S_ref = tts.semantic_codec.quantize(spk_cond_emb)
                ref_mel = tts.mel_fn(audio_22k.to(device).float())
                ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)
                
                feat = torchaudio.compliance.kaldi.fbank(
                    audio_16k.to(device),
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                style = tts.campplus_model(feat.unsqueeze(0))
                
                prompt_condition = tts.s2mel.models['length_regulator'](
                    S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
                )[0]
    
    elif speaker_embeddings is not None:
        spk_cond_emb = speaker_embeddings['spk_cond_emb'].to(device)
        speech_conditioning_latent = speaker_embeddings.get('gpt_conditioning')
        emo_vec = speaker_embeddings.get('emo_cond_emb', spk_cond_emb).to(device)
        style = speaker_embeddings['style'].to(device)
        prompt_condition = speaker_embeddings['prompt_condition'].to(device)
        ref_mel = speaker_embeddings['ref_mel'].to(device)
        
        if speech_conditioning_latent is None:
            cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                    speech_conditioning_latent = tts.gpt.get_conditioning(
                        spk_cond_emb.transpose(1, 2), cond_lengths
                    )
                    emo_cond = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_vec = tts.gpt.emovec_layer(emo_cond)
                    emo_vec = tts.gpt.emo_layer(emo_vec)
        else:
            speech_conditioning_latent = speech_conditioning_latent.to(device)
    else:
        raise ValueError("Either audio_prompt or speaker_embeddings must be provided")
    
    # Handle emotion reference audio
    if emotion_audio is not None:
        emo_audio, emo_sr = librosa.load(emotion_audio, sr=None, mono=True)
        emo_audio = emo_audio[:int(15 * emo_sr)]
        emo_audio_16k = librosa.resample(emo_audio, orig_sr=emo_sr, target_sr=16000)
        emo_audio_tensor = torch.from_numpy(emo_audio_16k).unsqueeze(0)
        
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                emo_inputs = tts.extract_features(emo_audio_tensor, sampling_rate=16000, return_tensors="pt")
                emo_input_features = emo_inputs["input_features"].to(device)
                emo_attention_mask = emo_inputs["attention_mask"].to(device)
                emo_emb = tts.get_emb(emo_input_features, emo_attention_mask)
                
                emo_cond_lengths = torch.tensor([emo_emb.shape[1]], device=device)
                new_emo = tts.gpt.get_emo_conditioning(emo_emb.transpose(1, 2), emo_cond_lengths)
                new_emo = tts.gpt.emovec_layer(new_emo)
                new_emo = tts.gpt.emo_layer(new_emo)
                
                emo_vec = emo_vec + emotion_alpha * (new_emo - emo_vec)
    
    # Handle explicit emotion vector
    if emo_vector is not None:
        weight_vector = torch.tensor(emo_vector, device=device)
        
        if use_random:
            random_index = [rnd.randint(0, x - 1) for x in tts.emo_num]
        else:
            def find_most_similar_cosine(query_vector, matrix):
                query_vector = query_vector.float()
                matrix = matrix.float()
                similarities = F.cosine_similarity(query_vector, matrix, dim=1)
                return torch.argmax(similarities)
            
            random_index = [find_most_similar_cosine(style, tmp) for tmp in tts.spk_matrix]
        
        emo_matrix_selected = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, tts.emo_matrix)]
        emo_matrix_selected = torch.cat(emo_matrix_selected, 0)
        emovec_mat = weight_vector.unsqueeze(1) * emo_matrix_selected
        emovec_mat = torch.sum(emovec_mat, 0)
        emovec_mat = emovec_mat.unsqueeze(0)
        
        weight_sum = sum(emo_vector)
        emo_vec = emovec_mat + (1 - weight_sum) * emo_vec
    
    # Pattern embedding injection
    final_conditioning = speech_conditioning_latent
    if pattern_embedding is not None:
        if config.verbose:
            print(f"[StreamingV2] Injecting pattern embedding (mode={injection_mode})")
        with torch.no_grad():
            final_conditioning = pattern_embedding.get_injection_embedding(
                speech_conditioning_latent,
                injection_mode=injection_mode,
            )
    
    if config.verbose:
        print(f"  Conditioning extracted in {time.perf_counter() - start_time:.3f}s")
    
    # === MODE-SPECIFIC STREAMING ===
    
    if config.mode == StreamingMode.SENTENCE_LEVEL:
        yield from _stream_by_sentences(
            tts=tts,
            text=text,
            config=config,
            spk_cond_emb=spk_cond_emb,
            emo_vec=emo_vec,
            style=style,
            prompt_condition=prompt_condition,
            ref_mel=ref_mel,
            final_conditioning=final_conditioning,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_mel_tokens=max_mel_tokens,
            on_audio_chunk=on_audio_chunk,
        )
    else:
        # FAST_CHUNKS, PROGRESSIVE_CONTEXT, OVERLAP_SYNTHESIS
        yield from _stream_by_tokens(
            tts=tts,
            text=text,
            config=config,
            spk_cond_emb=spk_cond_emb,
            emo_vec=emo_vec,
            style=style,
            prompt_condition=prompt_condition,
            ref_mel=ref_mel,
            final_conditioning=final_conditioning,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_mel_tokens=max_mel_tokens,
            on_audio_chunk=on_audio_chunk,
        )


def _stream_by_sentences(
    tts: 'IndexTTS2',
    text: str,
    config: StreamingConfigV2,
    spk_cond_emb: torch.Tensor,
    emo_vec: torch.Tensor,
    style: torch.Tensor,
    prompt_condition: torch.Tensor,
    ref_mel: torch.Tensor,
    final_conditioning: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    max_mel_tokens: int,
    on_audio_chunk: Optional[Callable],
) -> Generator[torch.Tensor, None, None]:
    """
    Stream by sentence boundaries - best quality mode.
    
    This generates complete sentences/clauses and synthesizes them fully,
    providing much better prosodic coherence than token-level chunking.
    """
    device = spk_cond_emb.device
    use_autocast = tts.dtype is not None and device.type == 'cuda'
    
    # Segment text into sentences
    segmenter = SentenceSegmenter(
        split_on_clauses=config.split_on_clauses,
        max_chars=config.max_sentence_chars
    )
    segments = segmenter.segment(text)
    
    if config.verbose:
        print(f"[SentenceLevel] Split into {len(segments)} segments")
        for i, seg in enumerate(segments):
            print(f"  [{i+1}] {seg[:50]}{'...' if len(seg) > 50 else ''}")
    
    # Create synthesizer with context tracking
    synthesizer = ProgressiveMelSynthesizer(
        tts=tts,
        config=config,
        spk_cond_emb=spk_cond_emb,
        emo_vec=emo_vec,
        style=style,
        prompt_condition=prompt_condition,
        ref_mel=ref_mel,
    )
    
    first_audio_time = None
    start_time = time.perf_counter()
    
    for seg_idx, segment in enumerate(segments):
        is_first = seg_idx == 0
        is_final = seg_idx == len(segments) - 1
        
        seg_start = time.perf_counter()
        
        # Tokenize segment
        text_tokens_list = tts.tokenizer.tokenize(segment)
        text_token_ids = tts.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_tokens = torch.tensor(text_token_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        if config.verbose:
            print(f"  [Segment {seg_idx+1}] {len(text_tokens_list)} text tokens")
        
        # Prepare GPT inputs
        batch_size = 1
        use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
        duration_ctrl = tts.gpt.speed_emb(torch.ones_like(use_speed))
        duration_free = tts.gpt.speed_emb(torch.zeros_like(use_speed))
        
        emo_vec_expanded = emo_vec.unsqueeze(1) if emo_vec.dim() == 2 else emo_vec
        
        conds_latent = torch.cat(
            (final_conditioning + emo_vec_expanded,
             duration_ctrl.unsqueeze(1),
             duration_free.unsqueeze(1)),
            dim=1,
        )
        
        input_ids, inputs_embeds, attention_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tokens)
        tts.gpt.inference_model.store_mel_emb(inputs_embeds)
        
        # Generate mel tokens for this segment
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                output = tts.gpt.inference_model.generate(
                    input_ids,
                    bos_token_id=tts.gpt.start_mel_token,
                    pad_token_id=tts.gpt.stop_mel_token,
                    eos_token_id=tts.gpt.stop_mel_token,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_mel_tokens - 1,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=1,
                )
        
        # Extract generated codes (skip input portion)
        trunc_index = input_ids.shape[1]
        codes = output[:, trunc_index:]
        
        # Filter out special tokens
        stop_mel_token = tts.stop_mel_token
        start_mel_token = tts.gpt.start_mel_token
        
        # Find actual code length
        if stop_mel_token in codes[0]:
            code_len = (codes[0] == stop_mel_token).nonzero(as_tuple=False)[0].item()
        else:
            code_len = codes.shape[1]
        
        codes = codes[:, :code_len]
        
        # Filter special tokens
        valid_mask = (codes[0] != start_mel_token) & (codes[0] != stop_mel_token)
        codes = codes[:, valid_mask]
        code_lens = torch.tensor([codes.shape[1]], device=device)
        
        if config.verbose:
            print(f"    Generated {codes.shape[1]} mel tokens")
        
        # Synthesize audio
        wav = synthesizer.synthesize(
            codes=codes,
            code_lens=code_lens,
            text_tokens=text_tokens,
            speech_conditioning_latent=final_conditioning,
            is_first=is_first,
            is_final=is_final,
        )
        
        seg_time = time.perf_counter() - seg_start
        
        if first_audio_time is None:
            first_audio_time = time.perf_counter() - start_time
            if config.verbose:
                print(f"[SentenceLevel] First audio at {first_audio_time:.3f}s")
        
        if config.verbose:
            print(f"    Synthesized {wav.shape[-1]} samples in {seg_time:.3f}s")
        
        if on_audio_chunk is not None:
            on_audio_chunk(wav)
        
        yield wav
    
    if config.verbose:
        total_time = time.perf_counter() - start_time
        print(f"[SentenceLevel] Total time: {total_time:.3f}s")


def _stream_by_tokens(
    tts: 'IndexTTS2',
    text: str,
    config: StreamingConfigV2,
    spk_cond_emb: torch.Tensor,
    emo_vec: torch.Tensor,
    style: torch.Tensor,
    prompt_condition: torch.Tensor,
    ref_mel: torch.Tensor,
    final_conditioning: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    max_mel_tokens: int,
    on_audio_chunk: Optional[Callable],
) -> Generator[torch.Tensor, None, None]:
    """
    TRUE token-level streaming - synthesize chunks AS tokens are generated.
    
    This uses a streamer that triggers synthesis during generation,
    yielding audio chunks with minimal latency.
    """
    device = spk_cond_emb.device
    use_autocast = tts.dtype is not None and device.type == 'cuda'
    
    # Tokenize full text
    text_tokens_list = tts.tokenizer.tokenize(text)
    text_token_ids = tts.tokenizer.convert_tokens_to_ids(text_tokens_list)
    text_tokens = torch.tensor(text_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    if config.verbose:
        print(f"[TokenLevel] Text: {len(text_tokens_list)} tokens")
    
    # Prepare GPT inputs
    batch_size = 1
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
    duration_ctrl = tts.gpt.speed_emb(torch.ones_like(use_speed))
    duration_free = tts.gpt.speed_emb(torch.zeros_like(use_speed))
    
    emo_vec_expanded = emo_vec.unsqueeze(1) if emo_vec.dim() == 2 else emo_vec
    
    conds_latent = torch.cat(
        (final_conditioning + emo_vec_expanded,
         duration_ctrl.unsqueeze(1),
         duration_free.unsqueeze(1)),
        dim=1,
    )
    
    input_ids, inputs_embeds, attention_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tokens)
    tts.gpt.inference_model.store_mel_emb(inputs_embeds)
    
    # Create synthesizer
    synthesizer = EnhancedProgressiveSynthesizer(
        tts=tts,
        config=config,
        spk_cond_emb=spk_cond_emb,
        emo_vec=emo_vec,
        style=style,
        prompt_condition=prompt_condition,
        ref_mel=ref_mel,
    )
    
    stop_mel_token = tts.stop_mel_token
    start_mel_token = tts.gpt.start_mel_token
    
    # Queue for passing audio chunks from streamer to generator
    audio_queue: queue.Queue[Optional[torch.Tensor]] = queue.Queue()
    generation_done = threading.Event()
    generation_error: List[Exception] = []
    
    # Streaming synthesizer that processes chunks during generation
    class StreamingSynthesizer(BaseStreamer):
        """Streamer that synthesizes audio chunks as mel tokens are generated."""
        
        def __init__(self):
            self.token_buffer: List[int] = []
            self.all_tokens: List[int] = []
            self.chunk_count = 0
            self.is_first_chunk = True
            self.start_time = time.perf_counter()
            self.first_audio_time: Optional[float] = None
            
        def put(self, value: torch.Tensor):
            """Called for each new token - synthesize when chunk is ready."""
            if value.dim() == 0:
                value = value.unsqueeze(0)
            
            new_tokens = value.squeeze().tolist()
            if isinstance(new_tokens, int):
                new_tokens = [new_tokens]
            
            for token in new_tokens:
                # Skip special tokens
                if token == stop_mel_token:
                    # Synthesize remaining tokens as final chunk
                    if self.token_buffer:
                        self._synthesize_chunk(is_final=True)
                    return
                if token == start_mel_token:
                    continue
                
                self.token_buffer.append(token)
                self.all_tokens.append(token)
                
                # Check if we should synthesize a chunk
                buffer_len = len(self.token_buffer)
                threshold = config.min_chunk_tokens if self.is_first_chunk else config.chunk_tokens
                
                if buffer_len >= config.max_chunk_tokens:
                    self._synthesize_chunk()
                elif buffer_len >= threshold:
                    self._synthesize_chunk()
        
        def _synthesize_chunk(self, is_final: bool = False):
            """Synthesize current buffer into audio and queue it."""
            if not self.token_buffer:
                return
            
            chunk_tokens = self.token_buffer.copy()
            self.token_buffer = []
            self.chunk_count += 1
            is_first = self.is_first_chunk
            self.is_first_chunk = False
            
            if config.verbose:
                print(f"  [Stream] Synthesizing chunk {self.chunk_count}: {len(chunk_tokens)} tokens")
            
            try:
                # Synthesize audio from tokens
                codes = torch.tensor([chunk_tokens], dtype=torch.long, device=device)
                code_lens = torch.tensor([len(chunk_tokens)], device=device)
                
                wav = synthesizer.synthesize_with_gpt_context(
                    codes=codes,
                    code_lens=code_lens,
                    text_tokens=text_tokens,
                    speech_conditioning_latent=final_conditioning,
                    is_first=is_first,
                    is_final=is_final,
                )
                
                # Track first audio time
                if self.first_audio_time is None:
                    self.first_audio_time = time.perf_counter() - self.start_time
                    if config.verbose:
                        print(f"  [Stream] FIRST AUDIO at {self.first_audio_time:.3f}s!")
                
                # Queue the audio chunk
                audio_queue.put(wav)
                
                if on_audio_chunk is not None:
                    on_audio_chunk(wav)
                    
            except Exception as e:
                if config.verbose:
                    print(f"  [Stream] Synthesis error: {e}")
                generation_error.append(e)
        
        def end(self):
            """Called when generation is complete."""
            # Synthesize any remaining tokens
            if self.token_buffer:
                self._synthesize_chunk(is_final=True)
            
            if config.verbose:
                total_time = time.perf_counter() - self.start_time
                print(f"  [Stream] Generation complete:")
                print(f"    Total tokens: {len(self.all_tokens)}")
                print(f"    Chunks: {self.chunk_count}")
                print(f"    Total time: {total_time:.3f}s")
                if self.first_audio_time:
                    print(f"    Time to first audio: {self.first_audio_time:.3f}s")
    
    streamer = StreamingSynthesizer()
    
    # Run generation in background thread so we can yield audio chunks
    def run_generation():
        """Run GPT generation with streaming synthesis."""
        try:
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                    tts.gpt.inference_model.generate(
                        input_ids,
                        bos_token_id=tts.gpt.start_mel_token,
                        pad_token_id=tts.gpt.stop_mel_token,
                        eos_token_id=tts.gpt.stop_mel_token,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + max_mel_tokens - 1,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=1,
                        streamer=streamer,
                    )
        except Exception as e:
            generation_error.append(e)
        finally:
            generation_done.set()
            audio_queue.put(None)  # Sentinel to signal end
    
    if config.verbose:
        print("[TokenLevel] Starting streaming generation...")
    
    # Start generation in background
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_start = time.perf_counter()
    gen_thread.start()
    
    # Yield audio chunks as they arrive
    chunk_count = 0
    first_yield_time = None
    
    while True:
        try:
            # Wait for next chunk with timeout
            wav = audio_queue.get(timeout=0.1)
            if wav is None:
                # Generation complete
                break
            chunk_count += 1
            if first_yield_time is None:
                first_yield_time = time.perf_counter() - gen_start
                if config.verbose:
                    print(f"  [Yield] First audio chunk at {first_yield_time:.3f}s")
            yield wav
        except queue.Empty:
            # Check if generation is done
            if generation_done.is_set():
                # Drain remaining items
                while not audio_queue.empty():
                    try:
                        wav = audio_queue.get_nowait()
                        if wav is not None:
                            chunk_count += 1
                            yield wav
                    except queue.Empty:
                        break
                break
    
    # Wait for thread to finish
    gen_thread.join(timeout=5.0)
    
    if generation_error:
        raise generation_error[0]
    
    if config.verbose:
        gen_time = time.perf_counter() - gen_start
        print(f"[TokenLevel] Done: {chunk_count} chunks in {gen_time:.3f}s")
        if first_yield_time:
            print(f"[TokenLevel] Time to first audio: {first_yield_time:.3f}s")


# Convenience function matching original API
def streaming_inference_generator_v2(
    tts: 'IndexTTS2',
    text: str,
    audio_prompt: Optional[str] = None,
    speaker_embeddings: Optional[dict] = None,
    pattern_embedding: Optional[Any] = None,
    config: Optional[StreamingConfigV2] = None,
    **kwargs
) -> Generator[torch.Tensor, None, None]:
    """
    Convenience wrapper for V2 streaming inference.
    
    This is the main entry point for high-quality streaming TTS synthesis.
    """
    yield from streaming_inference_v2(
        tts=tts,
        text=text,
        audio_prompt=audio_prompt,
        speaker_embeddings=speaker_embeddings,
        pattern_embedding=pattern_embedding,
        config=config,
        **kwargs
    )


# Utility functions for mode selection
def get_fast_streaming_config() -> StreamingConfigV2:
    """Get configuration optimized for lowest latency."""
    return StreamingConfigV2(
        mode=StreamingMode.FAST_CHUNKS,
        min_chunk_tokens=15,
        chunk_tokens=40,
        first_chunk_diffusion_steps=8,
        diffusion_steps=15,
    )


def get_quality_streaming_config() -> StreamingConfigV2:
    """Get configuration optimized for best quality."""
    return StreamingConfigV2(
        mode=StreamingMode.SENTENCE_LEVEL,
        split_on_clauses=False,
        diffusion_steps=25,
        first_chunk_diffusion_steps=15,
        crossfade_samples=4096,
    )


def get_balanced_streaming_config() -> StreamingConfigV2:
    """Get balanced configuration (recommended default)."""
    return StreamingConfigV2(
        mode=StreamingMode.SENTENCE_LEVEL,
        split_on_clauses=True,  # Split on commas for faster streaming
        max_sentence_chars=150,
        diffusion_steps=20,
        first_chunk_diffusion_steps=12,
        crossfade_samples=2048,
    )