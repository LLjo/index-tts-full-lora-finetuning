#!/usr/bin/env python3
"""
True Streaming TTS Inference with Pattern Embeddings

This script demonstrates the optimized streaming inference that yields
audio chunks as mel tokens are generated, achieving much faster
time-to-first-audio (TTFA).

Key features:
1. Token-level streaming from GPT generation
2. Chunked audio synthesis (~15-50 tokens per chunk)
3. Threaded pipeline for concurrent generation and synthesis
4. ~0.3s time to first audio (vs 3-15s for full generation)
5. Full support for pattern embeddings, LoRA, and emotion control

Usage:
    # Basic streaming with pattern embeddings
    python tools/infer_with_patterns_streaming.py --speaker ozzy \
        --text "Hello world, this is a streaming test." \
        --output output_streaming.wav
    
    # With custom audio prompt
    python tools/infer_with_patterns_streaming.py --speaker ozzy \
        --text "Life finds a way" \
        --audio-prompt path/to/voice.wav
    
    # With emotion control
    python tools/infer_with_patterns_streaming.py --speaker ozzy \
        --text "I am so happy!" \
        --emo-vector 0.8 0.0 0.0 0.0 0.0 0.0 0.0 0.2
    
    # With text-based emotion
    python tools/infer_with_patterns_streaming.py --speaker ozzy \
        --text "I am furious!" \
        --use-emo-text
    
    # Ultra-fast first audio (lower quality first chunk)
    python tools/infer_with_patterns_streaming.py --speaker ozzy \
        --text "Hello world!" \
        --output output.wav \
        --min-chunk-tokens 10 \
        --first-chunk-diffusion-steps 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Generator, Union, List
import os

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="True streaming TTS with pattern embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required
    parser.add_argument("--text", "-t", required=True, help="Text to synthesize")
    
    # Speaker (provides pattern embedding + default voice)
    parser.add_argument("--speaker", "-s", help="Speaker name (uses training/{speaker}/)")
    
    # Custom paths (override speaker defaults)
    parser.add_argument("--pattern-embedding", type=Path,
                        help="Path to pattern_embedding.pt")
    parser.add_argument("--lora-path", type=Path,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--speaker-embeddings", type=Path,
                        help="Path to speaker_embeddings.pt (for voice timbre)")
    
    # Voice control
    parser.add_argument("--audio-prompt", type=Path,
                        help="Reference audio for voice timbre (overrides speaker embeddings)")
    parser.add_argument("--emotion-audio", type=Path,
                        help="Reference audio for emotion")
    parser.add_argument("--emotion-alpha", type=float, default=1.0,
                        help="Emotion mixing weight (0-1)")
    
    # Emotion vector control (explicit emotions)
    parser.add_argument("--emo-vector", type=float, nargs=8, 
                        metavar=('HAPPY', 'ANGRY', 'SAD', 'AFRAID', 'DISGUSTED', 'MELANCHOLIC', 'SURPRISED', 'CALM'),
                        help="Explicit emotion vector: 8 floats for [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
    
    # Text-based emotion extraction
    parser.add_argument("--use-emo-text", action="store_true",
                        help="Extract emotion from text (uses synthesis text or --emo-text)")
    parser.add_argument("--emo-text", type=str,
                        help="Custom text for emotion extraction (requires --use-emo-text)")
    parser.add_argument("--use-random", action="store_true",
                        help="Use random emotion sampling (for emotion vector mode)")
    
    # Pattern control
    parser.add_argument("--pattern-scale", type=float, default=1.0,
                        help="Scale pattern embedding strength (default: 1.0)")
    parser.add_argument("--injection-mode", choices=["add", "prepend", "replace_first"],
                        default="add", help="How to inject pattern embedding")
    
    # Output
    parser.add_argument("--output", "-o", type=Path, default=Path("output_streaming.wav"),
                        help="Output audio path")
    
    # Generation params
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-mel-tokens", type=int, default=600,
                        help="Maximum mel tokens to generate")
    
    # Model configuration
    parser.add_argument("--use-fp16", action="store_true",
                        help="Use FP16 precision for faster inference (CUDA only)")
    parser.add_argument("--use-accel", action="store_true",
                        help="Use acceleration engine for GPT2")
    parser.add_argument("--use-torch-compile", action="store_true",
                        help="Use torch.compile for optimization")
    parser.add_argument("--use-cuda-kernel", action="store_true",
                        help="Use BigVGAN custom CUDA kernel (CUDA only)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified.")
    
    # Streaming configuration
    parser.add_argument("--min-chunk-tokens", type=int, default=15,
                        help="Minimum tokens before first audio chunk (lower = faster TTFA)")
    parser.add_argument("--chunk-tokens", type=int, default=50,
                        help="Tokens per chunk after first chunk")
    parser.add_argument("--diffusion-steps", type=int, default=12,
                        help="Diffusion steps for S2Mel (lower = faster but lower quality)")
    parser.add_argument("--first-chunk-diffusion-steps", type=int, default=6,
                        help="Diffusion steps for first chunk only (lower = faster TTFA)")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TRUE STREAMING PATTERN-AWARE INFERENCE")
    print("=" * 60)
    
    # Resolve paths from speaker
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker
        
        # Find pattern embedding
        pattern_emb_path = args.pattern_embedding
        if pattern_emb_path is None:
            candidates = [
                speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt",
                speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt",
            ]
            for c in candidates:
                if c.exists():
                    pattern_emb_path = c
                    break
        
        # Find LoRA
        lora_path = args.lora_path
        if lora_path is None:
            candidates = [
                speaker_dir / "pattern_training" / "best_checkpoint" / "lora",
                speaker_dir / "pattern_training" / "final_checkpoint" / "lora",
                speaker_dir / "lora" / "final_checkpoint",
            ]
            for c in candidates:
                if c.exists() and (c / "adapter_config.json").exists():
                    lora_path = c
                    break
        
        # Find speaker embeddings
        speaker_emb_path = args.speaker_embeddings
        if speaker_emb_path is None:
            speaker_emb_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
            if not speaker_emb_path.exists():
                speaker_emb_path = None
    else:
        pattern_emb_path = args.pattern_embedding
        lora_path = args.lora_path
        speaker_emb_path = args.speaker_embeddings
    
    # Validate
    if pattern_emb_path is None or not pattern_emb_path.exists():
        print("⚠️ Pattern embedding not found - using base model")
        pattern_emb_path = None
    
    if args.audio_prompt is None and speaker_emb_path is None:
        print("❌ Voice reference required!")
        print("\nProvide either:")
        print("  --audio-prompt path/to/voice.wav")
        print("  --speaker-embeddings path/to/embeddings.pt")
        print("  --speaker <name> (with extracted embeddings)")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Text: \"{args.text[:50]}{'...' if len(args.text) > 50 else ''}\"")
    if pattern_emb_path:
        print(f"  Pattern embedding: {pattern_emb_path}")
    print(f"  LoRA: {lora_path or 'None'}")
    print(f"  Voice source: {args.audio_prompt or speaker_emb_path}")
    
    # Streaming config
    print(f"\nStreaming config:")
    print(f"  Min chunk tokens: {args.min_chunk_tokens}")
    print(f"  Chunk tokens: {args.chunk_tokens}")
    print(f"  First chunk diffusion: {args.first_chunk_diffusion_steps} steps")
    print(f"  Regular diffusion: {args.diffusion_steps} steps")
    
    # Show emotion configuration
    if args.emo_vector:
        print(f"  Emotion vector: {args.emo_vector}")
    elif args.use_emo_text:
        emo_source = args.emo_text if args.emo_text else args.text
        print(f"  Emotion from text: \"{emo_source[:50]}...\"" if len(emo_source) > 50 else f"  Emotion from text: \"{emo_source}\"")
    elif args.emotion_audio:
        print(f"  Emotion audio: {args.emotion_audio}")
    
    # Load model
    print("\n[1/3] Loading IndexTTS2 model...")
    start_time = time.perf_counter()
    
    from indextts.infer_v2 import IndexTTS2
    
    tts = IndexTTS2(
        lora_path=str(lora_path) if lora_path else None,
        use_fp16=args.use_fp16,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
        use_cuda_kernel=args.use_cuda_kernel,
        device=args.device,
    )
    
    load_time = time.perf_counter() - start_time
    print(f"  Model loaded in {load_time:.2f}s")
    
    # Load pattern embedding if available
    pattern_embedding = None
    if pattern_emb_path:
        print(f"\n[2/3] Loading pattern embedding...")
        from indextts.pattern_embeddings import PatternEmbedding
        
        pattern_embedding = PatternEmbedding.load(pattern_emb_path, device=tts.device)
        pattern_embedding.eval()
        
        if args.pattern_scale != 1.0:
            pattern_embedding.pattern_scale.data *= args.pattern_scale
            print(f"  Adjusted pattern scale: {pattern_embedding.pattern_scale.item():.3f}")
    else:
        print("\n[2/3] No pattern embedding (using base model)")
    
    # Prepare voice reference
    print("\n[3/3] Starting streaming synthesis...")
    
    # Import streaming module
    from indextts.streaming import streaming_inference, StreamingConfig
    
    # Configure streaming
    config = StreamingConfig(
        min_chunk_tokens=args.min_chunk_tokens,
        chunk_tokens=args.chunk_tokens,
        diffusion_steps=args.diffusion_steps,
        first_chunk_diffusion_steps=args.first_chunk_diffusion_steps,
        inference_cfg_rate=0.7,
        synthesize_during_generation=True,
        verbose=args.verbose,
    )
    
    # Collect audio chunks
    audio_chunks = []
    gen_start_time = time.perf_counter()
    first_chunk_time = None
    
    # Prepare speaker embeddings if using pre-computed
    speaker_embeddings = None
    if args.audio_prompt is None and speaker_emb_path:
        from indextts.speaker_embeddings import SpeakerEmbeddingStore
        store = SpeakerEmbeddingStore(tts)
        speaker_embeddings = store.load_embeddings(speaker_emb_path)
    
    # Run streaming inference with pattern embedding support
    for wav_chunk in streaming_inference(
        tts=tts,
        text=args.text,
        audio_prompt=str(args.audio_prompt) if args.audio_prompt else None,
        speaker_embeddings=speaker_embeddings,
        emotion_audio=str(args.emotion_audio) if args.emotion_audio else None,
        emotion_alpha=args.emotion_alpha,
        emo_vector=args.emo_vector,
        use_emo_text=args.use_emo_text,
        emo_text=args.emo_text,
        use_random=args.use_random,
        config=config,
        pattern_embedding=pattern_embedding,  # Pass pattern embedding
        injection_mode=args.injection_mode,    # Pass injection mode
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_mel_tokens=args.max_mel_tokens,
    ):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - gen_start_time
            print(f"  ✓ First audio chunk at {first_chunk_time:.3f}s (TTFA)")
        
        chunk_samples = wav_chunk.shape[-1]
        duration = chunk_samples / 22050
        print(f"  → Chunk {len(audio_chunks)+1}: {duration:.2f}s audio ({chunk_samples} samples)")
        audio_chunks.append(wav_chunk)
    
    gen_end_time = time.perf_counter()
    total_time = gen_end_time - gen_start_time
    
    # Combine and save audio
    if audio_chunks:
        print(f"\n[Result] Combining {len(audio_chunks)} audio chunks...")
        
        # Ensure all chunks have same dimensions
        processed_chunks = []
        for chunk in audio_chunks:
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            processed_chunks.append(chunk)
        
        # Concatenate all chunks
        combined_audio = torch.cat(processed_chunks, dim=1)
        
        # Save to file
        output_dir = args.output.parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        torchaudio.save(
            str(args.output),
            combined_audio.type(torch.int16),
            22050
        )
        
        audio_duration = combined_audio.shape[1] / 22050
        
        print(f"\n" + "=" * 60)
        print("STREAMING RESULTS")
        print("=" * 60)
        print(f"  Output file: {args.output}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Total generation time: {total_time:.2f}s")
        print(f"  TIME TO FIRST AUDIO: {first_chunk_time:.3f}s")
        print(f"  Number of chunks: {len(audio_chunks)}")
        print(f"  Real-time factor (RTF): {total_time / audio_duration:.2f}x")
        print("=" * 60)
    else:
        print("\n[ERROR] No audio was generated!")
        return 1
    
    return 0


def pattern_aware_inference_streaming(
    tts: 'IndexTTS2',
    pattern_embedding: 'PatternEmbedding',
    text: str,
    audio_prompt: Optional[str] = None,
    speaker_embeddings: Optional[dict] = None,
    emotion_audio: Optional[str] = None,
    emotion_alpha: float = 1.0,
    emo_vector: Optional[list] = None,
    use_emo_text: bool = False,
    emo_text: Optional[str] = None,
    use_random: bool = False,
    injection_mode: str = "add",
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    max_mel_tokens: int = 600,
    # Streaming config
    min_chunk_tokens: int = 15,
    chunk_tokens: int = 50,
    diffusion_steps: int = 12,
    first_chunk_diffusion_steps: int = 6,
    verbose: bool = False,
) -> Generator[torch.Tensor, None, None]:
    """
    True streaming inference with pattern embedding injection.
    
    Yields audio chunks as they are generated, with ~0.3s time to first audio.
    The pattern embedding is injected into GPT conditioning to produce trained
    speaking patterns in the output.
    
    This is called by pattern_aware_inference when stream_return=True.
    
    Args:
        tts: IndexTTS2 instance
        pattern_embedding: PatternEmbedding instance with trained patterns
        text: Text to synthesize
        audio_prompt: Path to speaker reference audio
        speaker_embeddings: Pre-computed speaker embeddings
        emotion_audio: Path to emotion reference audio
        emotion_alpha: Emotion mixing weight (0-1)
        emo_vector: Explicit emotion vector
        use_emo_text: Extract emotion from text
        emo_text: Custom text for emotion extraction
        use_random: Use random emotion sampling
        injection_mode: How to inject pattern ("add", "prepend", "replace_first")
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        top_k: Top-k sampling threshold
        max_mel_tokens: Maximum mel tokens to generate
        min_chunk_tokens: Tokens for first chunk
        chunk_tokens: Tokens per subsequent chunk
        diffusion_steps: Diffusion steps for synthesis
        first_chunk_diffusion_steps: Diffusion steps for first chunk
        verbose: Print timing information
        
    Yields:
        Audio chunks as torch.Tensor (1, samples) at 22050 Hz
    """
    from indextts.streaming import streaming_inference, StreamingConfig
    from indextts.streaming_v2 import streaming_inference_v2, StreamingConfigV2, StreamingMode, get_fast_streaming_config, get_balanced_streaming_config
    
    # StreamingConfigV2(
    #     # min_chunk_tokens=min_chunk_tokens,
    #     # chunk_tokens=chunk_tokens,
    #     # diffusion_steps=diffusion_steps,
    #     # first_chunk_diffusion_steps=first_chunk_diffusion_steps,
    #     inference_cfg_rate=0.7,
    #     split_on_clauses=True,
    #     verbose=True
    # )
    config = StreamingConfigV2(
        mode=StreamingMode.PROGRESSIVE_CONTEXT,
        diffusion_steps=20,
        first_chunk_diffusion_steps=6,
        crossfade_samples=2048,
        min_chunk_tokens=130,
        verbose=True,
        chunk_tokens=80
    )
    
    # Stream with pattern embedding injection
    yield from streaming_inference_v2(
        tts=tts,
        text=text,
        audio_prompt=audio_prompt,
        speaker_embeddings=speaker_embeddings,
        emotion_audio=emotion_audio,
        emotion_alpha=emotion_alpha,
        emo_vector=emo_vector,
        use_emo_text=use_emo_text,
        emo_text=emo_text,
        use_random=use_random,
        config=config,
        pattern_embedding=pattern_embedding,
        injection_mode=injection_mode,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_mel_tokens=max_mel_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())