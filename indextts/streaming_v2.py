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

import contextlib
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
    # CFG rate for the first chunk only. CFG doubles the per-step DiT compute (it batches
    # the conditional + unconditional pass), so dropping it to 0 on the first chunk roughly
    # halves first-chunk diffusion latency. Subsequent chunks still get `inference_cfg_rate`.
    first_chunk_cfg_rate: float = 0.0
    # ODE solver for CFM: "euler" (1st-order, 1 estimator call/step) or "heun" (2nd-order,
    # 2 calls/step but ~half the steps for equivalent quality).
    solver: str = "heun"

    # Pause GPT decode while synth (CFM+BigVGAN) is running for that chunk. Default on:
    # without this, GPT's autoregressive decode kernels share SMs with CFM diffusion,
    # which makes CFM run 3-4× slower (per per-stage timing data). Serializing them
    # actually *improves* both TTFA and steady-state cadence because GPT has way more
    # slack than it needs to feed the synth queue.
    serialize_synth_with_gpt: bool = True

    # When True, every non-first chunk gets exactly `context_mel_frames_target` frames
    # of mel context concatenated before the new cond, zero-padded on the left if the
    # previous chunk produced fewer frames. This keeps the CFM input shape constant
    # across chunks 2+ so torch.compile's dynamic-shape recompile only happens once.
    # Without padding, chunk 2 has fewer context frames than chunks 3+ and pays a
    # partial recompile cost (~140ms first time it hits the chunk-2 shape).
    pad_mel_context: bool = True
    # The fixed-size mel-context window to pad/truncate to. Pick a value at least as
    # large as the expected previous-chunk cond size to avoid lossy truncation. The
    # ultra_fast preset's chunk_tokens=40 → ~69 frames of cond, so 100 is comfortable.
    context_mel_frames_target: int = 100

    # When True, call torch.cuda.empty_cache() after each stream completes. PyTorch's
    # caching allocator otherwise holds onto freed blocks, which inflates the GPU
    # footprint in nvidia-smi monotonically as you serve more requests. Negligible
    # cost (a handful of ms once, plus a slightly cold allocator on the next call).
    release_cuda_cache_on_done: bool = True

    # When True (and the accel engine is active), reuse the decode-time hidden states
    # the accel engine already produced for each generated token instead of re-running
    # a full GPT forward in the synth worker to extract S2Mel's latent. Saves ~10-30ms
    # per chunk.
    #
    # OFF BY DEFAULT — Quick Win #3 in docs/STREAMING_LATENCY_ROADMAP.md. The accel
    # engine's hidden states come from a fused-attention + CUDA-graph path and may not
    # be numerically bit-identical to the eager `tts.gpt(...)` forward this replaces.
    # A/B audio output (especially for the trained LoRA stutter voice) before turning on.
    #
    # When enabled, the chunk-dispatch logic lags by one token (because the latest
    # token's hidden state isn't captured until the next decode iteration). Effective
    # chunk sizes stay the same — the streamer just waits one extra token before
    # flushing the previous chunk.
    use_decoded_hidden_states: bool = False
    
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
        
        # Select diffusion steps based on chunk position. Middle/final chunks
        # use the configured `diffusion_steps`; only the first chunk gets the
        # cheaper `first_chunk_diffusion_steps` budget. The legacy `max(12, …)`
        # floor was overriding the config (e.g. ultra_fast asks for 10 steps but
        # got 12) — config wins now.
        if is_first:
            diffusion_steps = self.config.first_chunk_diffusion_steps
        else:
            diffusion_steps = self.config.diffusion_steps

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
                
                # Build condition with context. Track the exact frames prepended so the
                # post-CFM slice matches (previous code used a fixed config offset that
                # could exceed the actual previous mel length, leaving an empty tensor).
                if self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT and self.previous_mel_context is not None:
                    context_frames_used = min(
                        self.config.context_mel_frames,
                        self.previous_mel_context.size(1),
                    )
                    extended_prompt = torch.cat([
                        self.prompt_condition,
                        self.previous_mel_context[:, -context_frames_used:, :]
                    ], dim=1)
                    cat_condition = torch.cat([extended_prompt, cond], dim=1)
                else:
                    context_frames_used = 0
                    cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
                
                # CFM diffusion
                cfg_rate = self.config.first_chunk_cfg_rate if is_first else self.config.inference_cfg_rate
                vc_target = self.tts.s2mel.models['cfm'].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    self.ref_mel,
                    self.style,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=cfg_rate,
                    solver_type=self.config.solver,
                )

                # Extract new mel: drop ref_mel echo plus the exact context we prepended.
                vc_target = vc_target[:, :, self.ref_mel.size(-1) + context_frames_used:]
                
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
        log_event: Optional[Callable[..., None]] = None,
        precomputed_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesize with full GPT context from previous chunks.

        precomputed_latent: optional [1, len(codes), hidden_dim] tensor that
        replaces the redundant tts.gpt(...) forward — used by Quick Win #3
        when the accel engine has already produced the per-token hidden states.
        """
        device = self.spk_cond_emb.device
        self.chunk_index += 1
        chunk_idx = self.chunk_index

        # Local no-op when no logger is provided. Each substage forces a CUDA sync
        # before logging so we measure GPU completion time, not kernel launch time.
        if log_event is None:
            def _emit(event: str, **extra):
                pass
        else:
            def _emit(event: str, **extra):
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()
                log_event(event, chunk_idx=chunk_idx, **extra)

        # Select diffusion steps based on chunk position. Middle/final chunks
        # use the configured `diffusion_steps`; only the first chunk gets the
        # cheaper `first_chunk_diffusion_steps` budget. The legacy `max(12, …)`
        # floor was silently overriding the config (e.g. ultra_fast asks for 10
        # but got 12 — ~70 ms wasted per chunk). Config wins now; raise
        # `diffusion_steps` in the preset if you actually want more.
        if is_first:
            diffusion_steps = self.config.first_chunk_diffusion_steps
        else:
            diffusion_steps = self.config.diffusion_steps
        
        with torch.no_grad():
            use_autocast = self.tts.dtype is not None and device.type == 'cuda'

            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=self.tts.dtype or torch.float32):
                use_speed = torch.zeros(1, device=device, dtype=torch.long)

                if precomputed_latent is not None:
                    # Quick Win #3: reuse the decode-time hidden states the accel
                    # engine already computed for these codes. Saves a full GPT
                    # forward (~10-30ms per chunk).
                    latent = precomputed_latent
                # For non-first chunks, prepend previous codes for context
                elif self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT and len(self.cumulative_codes) > 0:
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

                _emit("synth_gpt_latent_done")

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

                _emit("synth_length_reg_done")
                
                # Enhanced mel context with larger window. Track the actual number of
                # context frames prepended so the post-CFM slice matches exactly. The
                # previous version sliced with a hardcoded 100 even when no context was
                # added (chunk 1) or when the previous chunk was shorter than 100 frames,
                # leaving an empty mel and crashing BigVGAN.
                #
                # When `pad_mel_context` is on, every non-first chunk gets a fixed
                # `context_mel_frames_target`-frame context (left-zero-padded if the
                # actual previous mel is shorter). This keeps the CFM input shape
                # constant across chunks 2+, so torch.compile's dynamic-shape branch
                # only specializes once.
                use_mel_context = (
                    self.config.mode == StreamingMode.PROGRESSIVE_CONTEXT
                    and self.previous_mel_context is not None
                )
                if use_mel_context:
                    if self.config.pad_mel_context:
                        context_target = self.config.context_mel_frames_target
                        available = self.previous_mel_context.size(1)
                        if available >= context_target:
                            actual_ctx = self.previous_mel_context[:, -context_target:, :]
                        else:
                            # Left-pad with zeros so the real (most recent) context
                            # stays adjacent to the new cond — pad first, then real.
                            pad = torch.zeros(
                                self.previous_mel_context.size(0),
                                context_target - available,
                                self.previous_mel_context.size(2),
                                device=self.previous_mel_context.device,
                                dtype=self.previous_mel_context.dtype,
                            )
                            actual_ctx = torch.cat([pad, self.previous_mel_context], dim=1)
                        context_frames_used = context_target
                    else:
                        requested_ctx = min(100, self.config.context_mel_frames * 2)
                        context_frames_used = min(requested_ctx, self.previous_mel_context.size(1))
                        actual_ctx = self.previous_mel_context[:, -context_frames_used:, :]
                    extended_prompt = torch.cat([self.prompt_condition, actual_ctx], dim=1)
                    cat_condition = torch.cat([extended_prompt, cond], dim=1)
                else:
                    context_frames_used = 0
                    cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

                # Save mel context for the NEXT call. Must happen after context_frames_used
                # is computed for this call.
                self.previous_mel_context = cond.clone()
                
                # CFM diffusion
                cfg_rate = self.config.first_chunk_cfg_rate if is_first else self.config.inference_cfg_rate
                vc_target = self.tts.s2mel.models['cfm'].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    self.ref_mel,
                    self.style,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=cfg_rate,
                    solver_type=self.config.solver,
                )

                _emit("synth_cfm_done", steps=diffusion_steps)

                # Extract new mel: skip the ref_mel echo and however many context frames
                # we actually prepended this call.
                vc_target = vc_target[:, :, self.ref_mel.size(-1) + context_frames_used:]

                # Vocoding
                with torch.cuda.amp.autocast(enabled=False):
                    # Ensure the input is float32 and on the correct device
                    vc_target_f32 = vc_target.to(device=device, dtype=torch.float32)
                    wav = self.tts.bigvgan(vc_target_f32).squeeze()

                _emit("synth_bigvgan_done")

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
    # Diagnostics: caller-provided list that gets `{event, t_ms, ...}` entries
    # for every major pipeline stage. Lets the API surface per-stage timing.
    timing_log: Optional[List[dict]] = None,
    # When set, reuses a previously-computed (gpt_conditioning, emo_vec) tuple
    # under this key from the model's `_cond_latent_cache`. The api passes the
    # speaker embedding file path here so repeated requests for the same speaker
    # skip the conditioning encoder forward.
    cond_cache_key: Optional[str] = None,
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

    def log_event(event: str, **extra) -> None:
        if timing_log is not None:
            timing_log.append({
                "event": event,
                "t_ms": round((time.perf_counter() - start_time) * 1000, 2),
                **extra,
            })

    log_event("request_start")

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

        # If a cache key is supplied, see if we've already paid the conditioning
        # encoder cost for this speaker. Cache stores fully device-resident tensors.
        cond_cache = getattr(tts, "_cond_latent_cache", None)
        cache_entry = None
        if speech_conditioning_latent is None and cond_cache is not None and cond_cache_key:
            cache_entry = cond_cache.get(cond_cache_key)

        if cache_entry is not None:
            speech_conditioning_latent = cache_entry["gpt_conditioning"]
            emo_vec = cache_entry["emo_vec"]
        elif speech_conditioning_latent is None:
            cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                    speech_conditioning_latent = tts.gpt.get_conditioning(
                        spk_cond_emb.transpose(1, 2), cond_lengths
                    )
                    emo_cond = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_vec = tts.gpt.emovec_layer(emo_cond)
                    emo_vec = tts.gpt.emo_layer(emo_vec)
            if cond_cache is not None and cond_cache_key:
                cond_cache[cond_cache_key] = {
                    "gpt_conditioning": speech_conditioning_latent,
                    "emo_vec": emo_vec,
                }
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
        log_event("conditioning_done")
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
            log_event=log_event,
            stream_start_anchor=start_time,
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
    log_event: Optional[Callable[..., None]] = None,
    stream_start_anchor: Optional[float] = None,
) -> Generator[torch.Tensor, None, None]:
    """
    TRUE token-level streaming - synthesize chunks AS tokens are generated.
    
    This uses a streamer that triggers synthesis during generation,
    yielding audio chunks with minimal latency.
    """
    device = spk_cond_emb.device
    use_autocast = tts.dtype is not None and device.type == 'cuda'

    # Route through the accel engine when available — it captures CUDA graphs around the
    # per-token decode loop, which is otherwise bottlenecked by kernel-launch overhead.
    # `use_accel=True` on IndexTTS2 construction is what populates `tts.gpt.accel_engine`.
    # Resolved up front so the streamer can decide whether to lag-by-1 dispatch (Quick
    # Win #3 only makes sense when the accel engine is actually populating hidden states).
    accel_engine = getattr(tts.gpt, "accel_engine", None)
    use_hidden_state_reuse = (
        config.use_decoded_hidden_states and accel_engine is not None
    )

    # No-op fallback when caller didn't supply a logger.
    if log_event is None:
        def log_event(event, **extra):
            pass

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
    
    # Synthesis runs on a dedicated worker thread (NOT inline in streamer.put), so that
    # the GPT decode keeps producing tokens for chunk N+1 while CFM+BigVGAN render
    # chunk N. The streamer's job shrinks to: buffer tokens, push a SynthJob to the
    # synth worker when a chunk threshold is hit.
    audio_queue: queue.Queue[Optional[torch.Tensor]] = queue.Queue()
    # Queue tuple: (tokens, is_first, is_final, chunk_idx, token_offset)
    # `token_offset` is the position of `tokens[0]` in the cumulative generated
    # sequence — used by Quick Win #3 to index accel_engine.last_decoded_hidden_states.
    synth_queue: queue.Queue[Optional[tuple]] = queue.Queue()
    generation_done = threading.Event()
    # Fatal errors (GPT generation itself blew up) — re-raised to the caller.
    generation_error: List[Exception] = []
    # Per-chunk synthesis failures — logged but not fatal; we just skip the chunk.
    # Only escalated if zero chunks ever succeeded.
    chunk_errors: List[Exception] = []
    chunk_success_count = [0]  # boxed so the inner class can mutate it
    first_audio_time_ref: List[Optional[float]] = [None]
    stream_start_time = stream_start_anchor if stream_start_anchor is not None else time.perf_counter()

    # Serialization gate: when synth is running CFM/BigVGAN, clear this so streamer.put
    # (and therefore the accel-engine decode loop) blocks. Per-stage timing showed CFM
    # runs 3-4× slower when GPT is concurrently decoding because they share the same
    # SMs — even with separate CUDA streams. Pausing GPT for the ~250ms of synth gives
    # CFM the GPU alone, which makes the chunk finish faster than the parallel version.
    gpt_can_proceed = threading.Event()
    gpt_can_proceed.set()  # initially: nothing in synth, GPT free to run

    # Streaming synthesizer that buffers tokens and dispatches synth jobs to the worker.
    class StreamingSynthesizer(BaseStreamer):
        """Streamer that buffers mel tokens and queues synth jobs to a worker thread."""

        def __init__(self):
            self.token_buffer: List[int] = []
            self.all_tokens: List[int] = []
            self.chunk_count = 0
            self.is_first_chunk = True
            self.first_token_logged = False
            # Position in the cumulative generated sequence at which the *next*
            # dispatched chunk begins. Always equals (tokens already dispatched).
            # Used by Quick Win #3 to index accel_engine.last_decoded_hidden_states.
            self.dispatched_offset = 0

        def put(self, value: torch.Tensor):
            """Called for each new token — buffer + queue a synth job at chunk thresholds.

            If `serialize_synth_with_gpt` is on, block here while a previous chunk is
            still synthesizing so GPT doesn't compete with CFM/BigVGAN for the GPU.
            """
            if config.serialize_synth_with_gpt:
                gpt_can_proceed.wait()
            if not self.first_token_logged:
                log_event("gpt_first_token")
                self.first_token_logged = True
            if value.dim() == 0:
                value = value.unsqueeze(0)

            new_tokens = value.squeeze().tolist()
            if isinstance(new_tokens, int):
                new_tokens = [new_tokens]

            for token in new_tokens:
                # Skip special tokens
                if token == stop_mel_token:
                    # Flush remaining tokens as final chunk (no lag needed: all
                    # captured hidden states are guaranteed available by stop time).
                    if self.token_buffer:
                        self._queue_chunk(is_final=True, lag=0)
                    return
                if token == start_mel_token:
                    continue

                self.token_buffer.append(token)
                self.all_tokens.append(token)

                # Check if we should dispatch a chunk. When using decoded hidden
                # states, we lag dispatch by one token because the hidden state
                # for the latest token isn't captured until the next decode
                # iteration runs. So we require threshold + 1 in the buffer and
                # ship out the first `threshold` tokens.
                buffer_len = len(self.token_buffer)
                base_threshold = (
                    config.min_chunk_tokens if self.is_first_chunk else config.chunk_tokens
                )
                base_max = config.max_chunk_tokens
                lag = 1 if use_hidden_state_reuse else 0
                threshold = base_threshold + lag
                max_chunk = base_max + lag

                if buffer_len >= max_chunk or buffer_len >= threshold:
                    self._queue_chunk(lag=lag)

        def _queue_chunk(self, is_final: bool = False, lag: int = 0):
            """Hand off the current buffer to the synth worker. Non-blocking.

            When `lag > 0`, dispatch buffer[:-lag] and keep the last `lag` token(s)
            in the buffer for the next chunk. Used by Quick Win #3 to ensure the
            accel engine has time to capture the hidden state for every token we
            ship out.
            """
            if not self.token_buffer:
                return

            if lag > 0 and len(self.token_buffer) > lag:
                chunk_tokens = self.token_buffer[:-lag]
                self.token_buffer = self.token_buffer[-lag:]
            else:
                chunk_tokens = self.token_buffer
                self.token_buffer = []

            if not chunk_tokens:
                return

            self.chunk_count += 1
            is_first = self.is_first_chunk
            self.is_first_chunk = False
            token_offset = self.dispatched_offset
            self.dispatched_offset += len(chunk_tokens)

            log_event("chunk_dispatched", chunk_idx=self.chunk_count, tokens=len(chunk_tokens))
            synth_queue.put((chunk_tokens, is_first, is_final, self.chunk_count, token_offset))

        def end(self):
            """Called when GPT generation completes."""
            # Flush any leftover tokens as the final chunk (no lag — all captured
            # hidden states are guaranteed available at this point).
            if self.token_buffer:
                self._queue_chunk(is_final=True, lag=0)
            # Signal the synth worker that no more jobs are coming
            synth_queue.put(None)

            if config.verbose:
                total_time = time.perf_counter() - stream_start_time
                print(f"  [Stream] GPT generation complete:")
                print(f"    Total tokens: {len(self.all_tokens)}")
                print(f"    Chunks queued: {self.chunk_count}")
                print(f"    GPT time: {total_time:.3f}s")

    streamer = StreamingSynthesizer()

    # Dedicated CUDA stream for the synth worker. The default stream serializes all
    # kernels — without this, the synth worker's `.cpu()` at the end of a chunk would
    # implicitly sync the *entire* default stream, blocking GPT decode that's queued
    # on it. Giving synth its own stream lets `.cpu()` sync only synth's work and
    # leaves GPT decode running concurrently. This is the actual CFM↔GPT overlap.
    synth_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def synth_worker():
        """Drain synth_queue, run CFM+BigVGAN per chunk, push audio to audio_queue.

        Decoupling synthesis from GPT decode is the main throughput win: while this
        worker is running CFM diffusion (300-400ms) and BigVGAN (30-80ms) for chunk N,
        the GPT decode thread keeps generating tokens for chunk N+1. Without this split
        GPT would block on every chunk boundary.

        Stream isolation: when running on CUDA, all synth ops are issued on
        `synth_stream`. Conditioning tensors (spk_cond_emb, prompt_condition, ref_mel,
        ...) were produced on the default stream before this worker started, so we
        wait on the default stream once at startup to make them visible.
        """
        if synth_stream is not None:
            synth_stream.wait_stream(torch.cuda.default_stream(device))

        try:
            while True:
                item = synth_queue.get()
                if item is None:  # sentinel from streamer.end() (or GPT error path)
                    return
                chunk_tokens, is_first, is_final, chunk_idx, token_offset = item

                # Hold GPT off the GPU while we run this chunk. See the comment on
                # `gpt_can_proceed` above for why this is a strict serialization.
                if config.serialize_synth_with_gpt:
                    gpt_can_proceed.clear()

                log_event("synth_start", chunk_idx=chunk_idx, tokens=len(chunk_tokens))

                if config.verbose:
                    print(f"  [Synth] Chunk {chunk_idx}: {len(chunk_tokens)} tokens (is_first={is_first}, is_final={is_final})")

                try:
                    # Run the full chunk (GPT-latent forward, S2Mel length reg, CFM
                    # diffusion, BigVGAN, .cpu()) on the synth stream so its sync
                    # at .cpu() doesn't block GPT decode on the default stream.
                    if synth_stream is not None:
                        stream_ctx = torch.cuda.stream(synth_stream)
                    else:
                        stream_ctx = contextlib.nullcontext()

                    # Optionally splice the decoded hidden states the accel engine
                    # captured during decode into a [1, L, H] latent — saves a full
                    # redundant GPT forward (~10-30ms). Only attempted when the user
                    # opted in AND the accel engine actually populated the buffer.
                    precomputed_latent = None
                    if (
                        use_hidden_state_reuse
                        and getattr(accel_engine, "last_decoded_hidden_states", None)
                    ):
                        captured = accel_engine.last_decoded_hidden_states
                        end_offset = token_offset + len(chunk_tokens)
                        if end_offset <= len(captured):
                            # Each entry is [1, hidden_dim] on the default stream.
                            # Stack along dim=1 to get [1, L, hidden_dim], matching
                            # what tts.gpt(...) returns.
                            slice_tensors = captured[token_offset:end_offset]
                            # Cross-stream sync: clones were queued on the default
                            # stream; synth runs on synth_stream. Insert a wait so
                            # synth_stream observes the writes before consuming.
                            if synth_stream is not None:
                                synth_stream.wait_stream(torch.cuda.default_stream(device))
                            with stream_ctx:
                                precomputed_latent = torch.stack(slice_tensors, dim=1)

                    with stream_ctx:
                        codes = torch.tensor([chunk_tokens], dtype=torch.long, device=device)
                        code_lens = torch.tensor([len(chunk_tokens)], device=device)

                        # Only collect substage timings for the first couple of chunks —
                        # the per-substage cuda.synchronize() adds latency, so we want it
                        # off in steady state once we know what we're looking at.
                        substage_log = log_event if chunk_idx <= 2 else None

                        wav = synthesizer.synthesize_with_gpt_context(
                            codes=codes,
                            code_lens=code_lens,
                            text_tokens=text_tokens,
                            speech_conditioning_latent=final_conditioning,
                            is_first=is_first,
                            is_final=is_final,
                            log_event=substage_log,
                            precomputed_latent=precomputed_latent,
                        )

                    log_event("synth_done", chunk_idx=chunk_idx, samples=int(wav.shape[-1]))

                    if first_audio_time_ref[0] is None:
                        first_audio_time_ref[0] = time.perf_counter() - stream_start_time
                        if config.verbose:
                            print(f"  [Synth] FIRST AUDIO at {first_audio_time_ref[0]:.3f}s!")

                    audio_queue.put(wav)
                    chunk_success_count[0] += 1

                    if on_audio_chunk is not None:
                        on_audio_chunk(wav)

                except Exception as e:
                    if config.verbose:
                        print(f"  [Synth] Synthesis error on chunk {chunk_idx} (skipping): {e}")
                    chunk_errors.append(e)
                finally:
                    # Whether the chunk succeeded or failed, let GPT run again. Otherwise
                    # a single bad chunk would deadlock the whole pipeline.
                    if config.serialize_synth_with_gpt:
                        gpt_can_proceed.set()
        finally:
            # Always signal the yield loop that no more audio is coming, even if the
            # worker itself crashed for some unexpected reason.
            audio_queue.put(None)
            # Final safety: ensure GPT thread isn't stuck waiting on the gate.
            gpt_can_proceed.set()
    
    def run_generation():
        """Run GPT generation. Tokens flow into `streamer`, which enqueues synth jobs."""
        try:
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                    if accel_engine is not None:
                        # Accel engine path: returns generated tokens but emits them
                        # via `streamer.put()` along the way.
                        max_new_tokens = max_mel_tokens - 1
                        accel_engine.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            attention_mask=attention_mask,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            stop_tokens=[tts.gpt.stop_mel_token],
                            tts_embeddings=inputs_embeds,
                            tts_mel_embedding=tts.gpt.inference_model.embeddings,
                            tts_text_pos_embedding=tts.gpt.inference_model.text_pos_embedding,
                            streamer=streamer,
                        )
                    else:
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
            # If generate() raised before streamer.end() was called, the synth worker
            # would deadlock on synth_queue.get(). Send the sentinel ourselves.
            synth_queue.put(None)
        finally:
            generation_done.set()

    if config.verbose:
        print("[TokenLevel] Starting streaming generation...")

    # Start GPT generation and synth worker in parallel.
    gen_thread = threading.Thread(target=run_generation, daemon=True, name="gpt-gen")
    synth_thread = threading.Thread(target=synth_worker, daemon=True, name="synth-worker")
    gen_start = time.perf_counter()
    log_event("threads_starting")
    synth_thread.start()
    gen_thread.start()

    # Yield audio chunks as they arrive. The synth worker is the sole producer of
    # audio_queue and always puts a final `None` sentinel when it exits.
    chunk_count = 0
    first_yield_time = None

    try:
        while True:
            wav = audio_queue.get()  # blocks until a chunk (or sentinel) is ready
            if wav is None:
                break
            chunk_count += 1
            log_event("chunk_yielded", chunk_idx=chunk_count, samples=int(wav.shape[-1]))
            if first_yield_time is None:
                first_yield_time = time.perf_counter() - gen_start
                if config.verbose:
                    print(f"  [Yield] First audio chunk at {first_yield_time:.3f}s")
            yield wav
    finally:
        # Make sure we tear down even if the consumer disconnected mid-stream.
        # Otherwise the GPT thread can stay blocked on the gate / synth queue.
        gpt_can_proceed.set()
        try:
            synth_queue.put_nowait(None)
        except Exception:
            pass

        # Wait for threads to finish cleanly
        gen_thread.join(timeout=5.0)
        synth_thread.join(timeout=5.0)

        # Drop large per-stream tensors before any cache release so the allocator
        # actually has freeable blocks to reclaim. Setting to None is enough —
        # the synthesizer goes out of scope when the function returns.
        try:
            synthesizer.reset()
        except Exception:
            pass

        # Release PyTorch's CUDA caching-allocator blocks back to the OS so the
        # GPU footprint shown in nvidia-smi doesn't keep growing across requests.
        # The accel engine's CUDA-graph captures are NOT affected: they live in
        # their own pool and persist across calls.
        if config.release_cuda_cache_on_done and device.type == "cuda":
            try:
                if accel_engine is not None:
                    accel_engine.last_decoded_hidden_states = []
            except Exception:
                pass
            torch.cuda.empty_cache()

    # Fatal: GPT generation itself failed. Always raise.
    if generation_error:
        raise generation_error[0]

    # Per-chunk failures: only escalate if literally nothing was produced. Otherwise the
    # client got partial audio and we'd be raising after-the-fact, which corrupts the
    # response body.
    if chunk_errors and chunk_success_count[0] == 0:
        raise chunk_errors[0]

    if config.verbose:
        gen_time = time.perf_counter() - gen_start
        print(f"[TokenLevel] Done: {chunk_count} chunks in {gen_time:.3f}s")
        if chunk_errors:
            print(f"[TokenLevel] {len(chunk_errors)} chunk(s) skipped due to synthesis errors")
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
    """Lowest-latency preset that still bridges to chunk 2 without a player gap.

    Sizing notes:
        - 22 first-chunk tokens ≈ 38 mel frames ≈ 440ms of audio. After the 512-sample
          (~23ms) tail trim the player gets ~417ms — comfortably longer than the
          ~300-400ms synthesis time of chunk 2, so the playback buffer never drains.
        - 5 Heun steps (= 10 DiT calls) with CFG=0 keeps first-chunk diffusion under
          ~150ms while still denoising enough to produce clear audio.
        - 512-sample crossfade is enough for speech to mask the boundary without
          eating significant audio out of each chunk.

    Mode is PROGRESSIVE_CONTEXT so chunks 2+ carry forward (a) last 30 GPT codes
    into the per-chunk latent forward and (b) last 150 mel-cond frames into the
    CFM. Chunk-1 TTFA is unaffected (no previous context to carry) but the
    speaking arc / tone holds across boundaries instead of resetting per chunk.
    """
    return StreamingConfigV2(
        mode=StreamingMode.PROGRESSIVE_CONTEXT,
        min_chunk_tokens=22,
        chunk_tokens=40,
        max_chunk_tokens=100,
        first_chunk_diffusion_steps=5,
        diffusion_steps=10,
        first_chunk_cfg_rate=0.0,
        inference_cfg_rate=0.7,
        solver="heun",
        crossfade_samples=512,
        context_mel_frames_target=150,
    )


def get_fast_quality_streaming_config() -> StreamingConfigV2:
    """`fast`-style TTFA with larger steady-state chunks for higher quality.

    First chunk stays small (40 tokens ≈ 800ms audio) so TTFA matches `fast`,
    then chunks 2+ grow to 100 tokens (≈ 2.0s audio each). Bigger chunks mean:
        - More mel frames per CFM call → closer to training distribution
          (CFM was trained on full-utterance pairs, so the bigger the chunk the
          less it has to extrapolate).
        - Fewer chunk boundaries across an utterance → fewer places for tone /
          prosody to drift.
        - Plenty of audio cushion for the player while the next chunk synthesizes
          (a 100-token chunk synthesizes in ~600-900ms, plays for ~2.0s — buffer
          never drains).

    Use this when you want quality closer to `balanced` without paying the
    sentence-level TTFA penalty.
    """
    return StreamingConfigV2(
        mode=StreamingMode.PROGRESSIVE_CONTEXT,
        min_chunk_tokens=40,
        chunk_tokens=100,
        max_chunk_tokens=200,
        first_chunk_diffusion_steps=5,
        diffusion_steps=10,
        first_chunk_cfg_rate=0.0,
        inference_cfg_rate=0.7,
        solver="heun",
        crossfade_samples=512,
        context_mel_frames_target=150,
    )


def get_ultra_fast_streaming_config() -> StreamingConfigV2:
    """Most aggressive TTFA preset that still plays without audible gaps.

    Sizing math (with serialized GPT/synth, steady cadence is ~590ms per chunk):
        - First chunk = 60 mel tokens → 60 × 1.72 frames × 256/22050s ≈ 1.2s of audio.
          That covers the ~590ms wait for chunk 2 with comfortable margin and pulls
          the first CFM call closer to the student's training distribution (see
          docs/STREAMING_LATENCY_RESULTS.md §5).
        - 5 Heun steps (= 10 DiT calls) with CFG=0 keeps first-chunk synth around
          ~200ms warm. TTFA lands ~400-450ms on a warm pipeline (teacher CFM).

    For the distilled student use `get_ultra_fast_distilled_streaming_config()` —
    same chunking but solver=single_step / 1 step / CFG=0.0.

    Mode is PROGRESSIVE_CONTEXT so chunks 2+ carry the last 30 GPT codes and last
    150 mel-cond frames of the previous chunk into the next CFM call. Holds the
    prosody arc across chunk boundaries; chunk-1 TTFA is unaffected.
    """
    return StreamingConfigV2(
        mode=StreamingMode.PROGRESSIVE_CONTEXT,
        min_chunk_tokens=60,
        chunk_tokens=40,
        max_chunk_tokens=100,
        first_chunk_diffusion_steps=5,
        diffusion_steps=10,
        first_chunk_cfg_rate=0.0,
        inference_cfg_rate=0.7,
        solver="heun",
        crossfade_samples=512,
        context_mel_frames_target=150,
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


def get_ultra_fast_distilled_streaming_config() -> StreamingConfigV2:
    """Ultra-fast preset baked for a distilled CFM student.

    Pairs the bumped first-chunk size from `get_ultra_fast_streaming_config()` with
    the single-step distilled solver (1 DiT call total per chunk, CFG=0). Targets
    sub-200ms TTFA on a warm pipeline when `s2mel_distilled.pth` is active.

    Only meaningful with a distilled student loaded — on the base teacher CFM this
    will sound terrible because 1 estimator call at t=0 isn't enough to denoise.

    Mode is PROGRESSIVE_CONTEXT (carries last 30 GPT codes + last 150 mel-cond
    frames per chunk boundary) — keeps the speaking style continuous instead of
    resetting per chunk.
    """
    return StreamingConfigV2(
        mode=StreamingMode.PROGRESSIVE_CONTEXT,
        min_chunk_tokens=60,
        chunk_tokens=40,
        max_chunk_tokens=100,
        first_chunk_diffusion_steps=1,
        diffusion_steps=1,
        first_chunk_cfg_rate=0.0,
        inference_cfg_rate=0.0,
        solver="single_step",
        crossfade_samples=512,
        context_mel_frames_target=150,
    )


def get_balanced_distilled_streaming_config() -> StreamingConfigV2:
    """Production-recommended distilled config: sentence-level + single_step.

    Mirrors `get_balanced_streaming_config()` but bakes the distilled-student
    overrides (solver=single_step, 1 step, CFG=0.0). Sentence-sized chunks stay
    inside the student's training distribution, so quality matches the teacher
    while TTFA lands ~250-300ms. See docs/STREAMING_LATENCY_RESULTS.md §6.

    Only meaningful with a distilled student loaded.
    """
    return StreamingConfigV2(
        mode=StreamingMode.SENTENCE_LEVEL,
        split_on_clauses=True,
        max_sentence_chars=150,
        diffusion_steps=1,
        first_chunk_diffusion_steps=1,
        first_chunk_cfg_rate=0.0,
        inference_cfg_rate=0.0,
        solver="single_step",
        crossfade_samples=2048,
    )