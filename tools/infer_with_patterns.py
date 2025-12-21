#!/usr/bin/env python3
"""
Inference with Pattern Embeddings for IndexTTS2

This script performs inference using TRAINED PATTERN EMBEDDINGS.
This is what makes the speaking patterns (stutters, pauses, etc.) 
actually appear in the output!

THE KEY INSIGHT:
================
Previous inference approaches used different conditioning at inference
than what was used during training. The patterns were "learned" but
tied to training-specific embeddings.

This script:
1. Loads the SAME pattern embedding used during training
2. Injects it into the GPT conditioning
3. Patterns appear because the model recognizes the "trigger"!

Usage:
    # Basic usage (uses speaker's trained embeddings)
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "Life finds a way"
    
    # With custom audio prompt for voice timbre (patterns still come from embedding)
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "Life finds a way" \
        --audio-prompt path/to/voice.wav
    
    # With emotion control via reference audio
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "Life finds a way" \
        --emotion-audio sad_reference.wav
    
    # With explicit emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "I am so happy today!" \
        --emo-vector 0.8 0.0 0.0 0.0 0.0 0.0 0.0 0.2
    
    # With text-based emotion extraction (auto-detect from text)
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "I am furious about this!" \
        --use-emo-text
    
    # With custom emotion text (different from synthesis text)
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "Life finds a way" \
        --use-emo-text --emo-text "I am extremely angry and upset"
    
    # With FP16 precision and torch.compile optimization
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "Life finds a way" \
        --use-fp16 --use-torch-compile
    
    # With streaming output (yields audio chunks as they're generated)
    python tools/infer_with_patterns.py --speaker ozzy \
        --text "Life finds a way" \
        --stream-return
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with trained pattern embeddings",
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
    parser.add_argument("--emo-vector", type=float, nargs=8, metavar=('HAPPY', 'ANGRY', 'SAD', 'AFRAID', 'DISGUSTED', 'MELANCHOLIC', 'SURPRISED', 'CALM'),
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
    parser.add_argument("--output", "-o", type=Path, default=Path("output_with_patterns.wav"),
                        help="Output audio path")
    
    # Generation params
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    
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
    
    # Streaming
    parser.add_argument("--stream-return", action="store_true",
                        help="Return audio as streaming generator (for real-time playback)")
    
    parser.add_argument("--verbose", "-v", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PATTERN-AWARE INFERENCE")
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
        print("❌ Pattern embedding not found!")
        if args.speaker:
            print(f"\nTrain patterns first:")
            print(f"  python tools/train_pattern_embeddings.py --speaker {args.speaker}")
        else:
            print("\nProvide --pattern-embedding path")
        sys.exit(1)
    
    if args.audio_prompt is None and speaker_emb_path is None:
        print("❌ Voice reference required!")
        print("\nProvide either:")
        print("  --audio-prompt path/to/voice.wav")
        print("  --speaker-embeddings path/to/embeddings.pt")
        print("  --speaker <name> (with extracted embeddings)")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Text: \"{args.text}\"")
    print(f"  Pattern embedding: {pattern_emb_path}")
    print(f"  LoRA: {lora_path or 'None'}")
    print(f"  Voice source: {args.audio_prompt or speaker_emb_path}")
    print(f"  Injection mode: {args.injection_mode}")
    print(f"  Pattern scale: {args.pattern_scale}")
    
    # Show emotion configuration
    if args.emo_vector:
        print(f"  Emotion vector: {args.emo_vector}")
    elif args.use_emo_text:
        emo_source = args.emo_text if args.emo_text else args.text
        print(f"  Emotion from text: \"{emo_source[:50]}...\"" if len(emo_source) > 50 else f"  Emotion from text: \"{emo_source}\"")
    elif args.emotion_audio:
        print(f"  Emotion audio: {args.emotion_audio}")
    
    # Show model configuration
    if args.use_fp16:
        print(f"  FP16: enabled")
    if args.use_accel:
        print(f"  Acceleration: enabled")
    if args.use_torch_compile:
        print(f"  torch.compile: enabled")
    if args.use_cuda_kernel:
        print(f"  CUDA kernel: enabled")
    if args.stream_return:
        print(f"  Streaming: enabled")
    
    # Load model
    print("\n[1/3] Loading model...")
    from indextts.infer_v2 import IndexTTS2
    
    tts = IndexTTS2(
        lora_path=str(lora_path) if lora_path else None,
        use_fp16=args.use_fp16,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
        use_cuda_kernel=args.use_cuda_kernel,
        device=args.device,
    )
    
    # Load pattern embedding
    print("\n[2/3] Loading pattern embedding...")
    from indextts.pattern_embeddings import PatternEmbedding
    
    pattern_embedding = PatternEmbedding.load(pattern_emb_path, device=tts.device)
    pattern_embedding.eval()
    
    # Apply pattern scale
    if args.pattern_scale != 1.0:
        pattern_embedding.pattern_scale.data *= args.pattern_scale
        print(f"  Adjusted pattern scale: {pattern_embedding.pattern_scale.item():.3f}")
    
    # Prepare reference audio / embeddings
    print("\n[3/3] Preparing voice reference...")
    
    if args.audio_prompt:
        # Use audio file for voice
        audio_prompt = str(args.audio_prompt)
        speaker_embeddings = None
        print(f"  Using audio prompt: {audio_prompt}")
    else:
        # Use pre-computed embeddings
        from indextts.speaker_embeddings import SpeakerEmbeddingStore
        store = SpeakerEmbeddingStore(tts)
        speaker_embeddings = store.load_embeddings(speaker_emb_path)
        audio_prompt = None
        print(f"  Using speaker embeddings from: {speaker_emb_path}")
    
    # === THE KEY: Pattern-aware inference ===
    print("\n" + "=" * 60)
    print("GENERATING WITH PATTERN EMBEDDING")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    # Custom inference with pattern embedding injection
    if args.stream_return:
        # Streaming mode - returns generator
        print("\n[Streaming mode] Generating audio chunks...")
        audio_generator = pattern_aware_inference(
            tts=tts,
            pattern_embedding=pattern_embedding,
            text=args.text,
            output_path=args.output,
            audio_prompt=audio_prompt,
            speaker_embeddings=speaker_embeddings,
            emotion_audio=str(args.emotion_audio) if args.emotion_audio else None,
            emotion_alpha=args.emotion_alpha,
            emo_vector=args.emo_vector,
            use_emo_text=args.use_emo_text,
            emo_text=args.emo_text,
            use_random=args.use_random,
            injection_mode=args.injection_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            verbose=args.verbose,
            stream_return=True,
        )
        
        # Collect all chunks for final output
        all_chunks = []
        for i, chunk in enumerate(audio_generator):
            if chunk is not None:
                all_chunks.append(chunk)
                print(f"  Received chunk {i+1}: {chunk.shape}")
        
        # Save combined audio
        if all_chunks:
            wav = torch.cat(all_chunks, dim=1)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), wav.type(torch.int16), 22050)
        else:
            output_path = args.output
    else:
        # Non-streaming mode
        output_path = pattern_aware_inference(
            tts=tts,
            pattern_embedding=pattern_embedding,
            text=args.text,
            output_path=args.output,
            audio_prompt=audio_prompt,
            speaker_embeddings=speaker_embeddings,
            emotion_audio=str(args.emotion_audio) if args.emotion_audio else None,
            emotion_alpha=args.emotion_alpha,
            emo_vector=args.emo_vector,
            use_emo_text=args.use_emo_text,
            emo_text=args.emo_text,
            use_random=args.use_random,
            injection_mode=args.injection_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            verbose=args.verbose,
            stream_return=False,
        )
    
    elapsed = time.perf_counter() - start_time
    
    print(f"\n✓ Generated in {elapsed:.2f}s")
    print(f"✓ Output saved to: {output_path}")
    print(f"""
The patterns from training should now appear in the output!
This works because:
  1. Pattern embedding was trained to encode "{args.speaker or 'this speaker'}'s" patterns
  2. Same embedding is injected during inference
  3. Model recognizes the "trigger" and produces patterns
""")


from typing import Generator, Union

def pattern_aware_inference(
    tts: 'IndexTTS2',
    pattern_embedding: 'PatternEmbedding',
    text: str,
    output_path: Path,
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
    verbose: bool = False,
    stream_return: bool = False,
) -> Union[Path, Generator[torch.Tensor, None, None]]:
    """
    Perform inference with pattern embedding injection.
    
    This is the KEY function that makes patterns appear:
    1. Extract conditioning from reference audio
    2. INJECT pattern embedding into conditioning
    3. Run GPT generation with modified conditioning
    4. S2Mel stage uses reference audio features (unchanged)
    
    Emotion control options (mutually exclusive, in priority order):
    - emo_vector: Explicit emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    - use_emo_text: Extract emotion from text using QwenEmotion model
    - emotion_audio: Reference audio for emotion transfer
    
    If stream_return=True, returns a generator yielding audio chunks.
    Otherwise, returns the output path after saving the complete audio.
    """
    if stream_return:
        return _pattern_aware_inference_streaming(
            tts=tts,
            pattern_embedding=pattern_embedding,
            text=text,
            audio_prompt=audio_prompt,
            speaker_embeddings=speaker_embeddings,
            emotion_audio=emotion_audio,
            emotion_alpha=emotion_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            injection_mode=injection_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            verbose=verbose,
        )
    
    import random
    import torch
    import torch.nn.functional as F
    import librosa
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    
    # Handle device - could be string or torch.device
    device = tts.device
    if isinstance(device, str):
        device = torch.device(device)
    
    # === STEP 0: Process emotion settings ===
    # Priority: emo_vector > use_emo_text > emotion_audio
    if use_emo_text or emo_vector is not None:
        # Using text or explicit vector; disable emotion audio
        emotion_audio = None
    
    if use_emo_text:
        # Extract emotion from text using QwenEmotion
        if emo_text is None:
            emo_text = text  # Use synthesis text
        emo_dict = tts.qwen_emo.inference(emo_text)
        if verbose:
            print(f"  Detected emotions from text: {emo_dict}")
        emo_vector = list(emo_dict.values())
    
    if emo_vector is not None:
        # Scale emotion vector by alpha
        emo_vector_scale = max(0.0, min(1.0, emotion_alpha))
        if emo_vector_scale != 1.0:
            emo_vector = [x * emo_vector_scale for x in emo_vector]
            if verbose:
                print(f"  Scaled emotion vector to {emo_vector_scale}x: {emo_vector}")
        # Normalize emotion vector
        emo_vector = tts.normalize_emo_vec(emo_vector)
        if verbose:
            print(f"  Normalized emotion vector: {emo_vector}")
    
    # === STEP 1: Get base conditioning ===
    if audio_prompt is not None and audio_prompt != "":
        # Extract from audio file
        audio, sr = librosa.load(audio_prompt, sr=None, mono=True)
        audio = audio[:int(15 * sr)]  # Max 15s
        
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio_tensor)
        
        # W2V-BERT features
        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Determine autocast settings for conditioning extraction
        use_autocast = tts.dtype is not None and device.type == 'cuda'
        
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                spk_cond_emb = tts.get_emb(input_features, attention_mask)
                
                cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
                gpt_conditioning = tts.gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                
                # Emotion conditioning
                emo_cond = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                emo_vec = tts.gpt.emovec_layer(emo_cond)
                emo_vec = tts.gpt.emo_layer(emo_vec)
                
                # S2Mel features
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
    else:
        # Use pre-computed embeddings
        if speaker_embeddings is None:
            raise ValueError("Either audio_prompt or speaker_embeddings must be provided")
        
        spk_cond_emb = speaker_embeddings['spk_cond_emb'].to(device)
        gpt_conditioning = speaker_embeddings.get('gpt_conditioning')
        emo_vec = speaker_embeddings.get('emo_cond_emb', spk_cond_emb).to(device)
        style = speaker_embeddings['style'].to(device)
        prompt_condition = speaker_embeddings['prompt_condition'].to(device)
        ref_mel = speaker_embeddings['ref_mel'].to(device)
        
        # Determine autocast settings
        use_autocast = tts.dtype is not None and device.type == 'cuda'
        
        # Extract GPT conditioning if not cached
        if gpt_conditioning is None:
            cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                    gpt_conditioning = tts.gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_cond = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_vec = tts.gpt.emovec_layer(emo_cond)
                    emo_vec = tts.gpt.emo_layer(emo_vec)
        else:
            gpt_conditioning = gpt_conditioning.to(device)
    
    # Determine autocast settings (may not be set if we came from pre-computed embeddings path)
    use_autocast = tts.dtype is not None and device.type == 'cuda'
    
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
                
                # Blend emotions
                emo_vec = emo_vec + emotion_alpha * (new_emo - emo_vec)
    
    # Handle explicit emotion vector (from emo_vector or use_emo_text)
    emovec_mat = None
    if emo_vector is not None:
        weight_vector = torch.tensor(emo_vector, device=device)
        
        # Note: `style` is already defined by this point:
        # - From audio_prompt path: computed via campplus_model (line 397)
        # - From speaker_embeddings path: loaded from embeddings (line 407)
        
        if use_random:
            random_index = [random.randint(0, x - 1) for x in tts.emo_num]
        else:
            # Find most similar emotion samples
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
        
        if verbose:
            print(f"  Emotion matrix shape: {emovec_mat.shape}")
    
    # Merge emotion vectors
    if emovec_mat is not None:
        weight_sum = sum(emo_vector)
        emo_vec = emovec_mat + (1 - weight_sum) * emo_vec
        if verbose:
            print(f"  Merged emotion vector with weight sum: {weight_sum:.3f}")
    
    # === STEP 2: INJECT PATTERN EMBEDDING ===
    # THIS IS THE KEY TO MAKING PATTERNS APPEAR!
    if verbose:
        print(f"  GPT conditioning shape before injection: {gpt_conditioning.shape}")
    
    with torch.no_grad():
        pattern_conditioned = pattern_embedding.get_injection_embedding(
            gpt_conditioning,
            injection_mode=injection_mode,
        )
    
    if verbose:
        print(f"  GPT conditioning shape after injection: {pattern_conditioned.shape}")
    
    # Build final conditioning
    batch_size = 1
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
    duration_ctrl = tts.gpt.speed_emb(torch.ones_like(use_speed))
    duration_free = tts.gpt.speed_emb(torch.zeros_like(use_speed))
    
    # Combine pattern-injected conditioning with emotion
    # Ensure emo_vec has correct shape for broadcasting
    if emo_vec.dim() == 2:
        emo_vec_expanded = emo_vec.unsqueeze(1)
    else:
        emo_vec_expanded = emo_vec
    
    conds_latent = torch.cat(
        (pattern_conditioned + emo_vec_expanded,
         duration_ctrl.unsqueeze(1),
         duration_free.unsqueeze(1)),
        dim=1,
    )
    
    # === STEP 3: Tokenize text ===
    text_tokens_list = tts.tokenizer.tokenize(text)
    text_token_ids = tts.tokenizer.convert_tokens_to_ids(text_tokens_list)
    text_tokens = torch.tensor(text_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # === STEP 4: GPT Generation with pattern-injected conditioning ===
    if verbose:
        print(f"  Text tokens: {len(text_token_ids)}")
    
    input_ids, inputs_embeds, attention_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tokens)
    tts.gpt.inference_model.store_mel_emb(inputs_embeds)
    
    # Note: use_autocast was already determined above
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
            output = tts.gpt.inference_model.generate(
                input_ids,
                bos_token_id=tts.gpt.start_mel_token,
                pad_token_id=tts.gpt.stop_mel_token,
                eos_token_id=tts.gpt.stop_mel_token,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + tts.gpt.max_mel_tokens - 1,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=1,
            )
    
    trunc_index = input_ids.shape[1]
    codes = output[:, trunc_index:]
    
    # Trim to stop token
    if (codes == tts.stop_mel_token).any():
        stop_idx = (codes[0] == tts.stop_mel_token).nonzero(as_tuple=False)
        if len(stop_idx) > 0:
            codes = codes[:, :stop_idx[0].item()]
    
    if verbose:
        print(f"  Generated codes: {codes.shape[1]}")
    
    code_lens = torch.tensor([codes.shape[1]], device=device)
    
    # === STEP 5: GPT Forward for latents ===
    # Ensure emo_vec has correct 2D shape for GPT forward
    if emo_vec.dim() == 3:
        emo_vec_2d = emo_vec.squeeze(1)
    else:
        emo_vec_2d = emo_vec
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
            # We need to use the original conditioning method for forward pass
            # But still with our pattern-injected conditioning
            latent = tts.gpt(
                pattern_conditioned,  # Use pattern-injected conditioning
                text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=device),
                codes,
                code_lens,
                spk_cond_emb,  # For emotion computation
                cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device),
                emo_cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device),
                emo_vec=emo_vec_2d,
                use_speed=use_speed,
            )
    
    # === STEP 6: S2Mel stage (unchanged - uses reference features) ===
    with torch.no_grad():
        diffusion_steps = 25
        inference_cfg_rate = 0.7
        
        latent = tts.s2mel.models['gpt_layer'](latent)
        S_infer = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)
        S_infer = S_infer + latent
        
        target_lengths = (code_lens * 1.72).long()
        
        cond = tts.s2mel.models['length_regulator'](
            S_infer, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]
        
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        
        vc_target = tts.s2mel.models['cfm'].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(device),
            ref_mel, style, None, diffusion_steps,
            inference_cfg_rate=inference_cfg_rate
        )
        vc_target = vc_target[:, :, ref_mel.size(-1):]
        
        wav = tts.bigvgan(vc_target.float()).squeeze()
    
    # Save
    wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
    wav = wav.cpu().unsqueeze(0)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), wav.type(torch.int16), 22050)
    
    return output_path


def _pattern_aware_inference_streaming(
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
    verbose: bool = False,
) -> Generator[torch.Tensor, None, None]:
    """
    Streaming version of pattern_aware_inference.
    Yields audio chunks as they are generated.
    
    Note: Due to the architecture, this currently generates the full audio
    and yields it as a single chunk. True token-by-token streaming would
    require deeper integration with the GPT generation loop.
    """
    import random
    import torch
    import torch.nn.functional as F
    import librosa
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    
    # Handle device
    device = tts.device
    if isinstance(device, str):
        device = torch.device(device)
    
    # === Process emotion settings ===
    if use_emo_text or emo_vector is not None:
        emotion_audio = None
    
    if use_emo_text:
        if emo_text is None:
            emo_text = text
        emo_dict = tts.qwen_emo.inference(emo_text)
        if verbose:
            print(f"  Detected emotions from text: {emo_dict}")
        emo_vector = list(emo_dict.values())
    
    if emo_vector is not None:
        emo_vector_scale = max(0.0, min(1.0, emotion_alpha))
        if emo_vector_scale != 1.0:
            emo_vector = [x * emo_vector_scale for x in emo_vector]
        emo_vector = tts.normalize_emo_vec(emo_vector)
    
    # === Get base conditioning ===
    if audio_prompt is not None and audio_prompt != "":
        audio, sr = librosa.load(audio_prompt, sr=None, mono=True)
        audio = audio[:int(15 * sr)]
        
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio_tensor)
        
        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Determine autocast settings
        use_autocast = tts.dtype is not None and device.type == 'cuda'
        
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                spk_cond_emb = tts.get_emb(input_features, attention_mask)
                cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
                gpt_conditioning = tts.gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                
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
    else:
        if speaker_embeddings is None:
            raise ValueError("Either audio_prompt or speaker_embeddings must be provided")
        
        spk_cond_emb = speaker_embeddings['spk_cond_emb'].to(device)
        gpt_conditioning = speaker_embeddings.get('gpt_conditioning')
        emo_vec = speaker_embeddings.get('emo_cond_emb', spk_cond_emb).to(device)
        style = speaker_embeddings['style'].to(device)
        prompt_condition = speaker_embeddings['prompt_condition'].to(device)
        ref_mel = speaker_embeddings['ref_mel'].to(device)
        
        # Determine autocast settings
        use_autocast = tts.dtype is not None and device.type == 'cuda'
        
        if gpt_conditioning is None:
            cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
            with torch.no_grad():
                with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
                    gpt_conditioning = tts.gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_cond = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_vec = tts.gpt.emovec_layer(emo_cond)
                    emo_vec = tts.gpt.emo_layer(emo_vec)
        else:
            gpt_conditioning = gpt_conditioning.to(device)
    
    # Determine autocast settings (may not be set if we came from pre-computed embeddings path)
    use_autocast = tts.dtype is not None and device.type == 'cuda'
    
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
    emovec_mat = None
    if emo_vector is not None:
        weight_vector = torch.tensor(emo_vector, device=device)
        
        if use_random:
            random_index = [random.randint(0, x - 1) for x in tts.emo_num]
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
    
    if emovec_mat is not None:
        weight_sum = sum(emo_vector)
        emo_vec = emovec_mat + (1 - weight_sum) * emo_vec
    
    # === INJECT PATTERN EMBEDDING ===
    with torch.no_grad():
        pattern_conditioned = pattern_embedding.get_injection_embedding(
            gpt_conditioning,
            injection_mode=injection_mode,
        )
    
    # Build final conditioning
    batch_size = 1
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
    duration_ctrl = tts.gpt.speed_emb(torch.ones_like(use_speed))
    duration_free = tts.gpt.speed_emb(torch.zeros_like(use_speed))
    
    if emo_vec.dim() == 2:
        emo_vec_expanded = emo_vec.unsqueeze(1)
    else:
        emo_vec_expanded = emo_vec
    
    conds_latent = torch.cat(
        (pattern_conditioned + emo_vec_expanded,
         duration_ctrl.unsqueeze(1),
         duration_free.unsqueeze(1)),
        dim=1,
    )
    
    # Tokenize text
    text_tokens_list = tts.tokenizer.tokenize(text)
    text_token_ids = tts.tokenizer.convert_tokens_to_ids(text_tokens_list)
    text_tokens = torch.tensor(text_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # GPT Generation
    input_ids, inputs_embeds, attention_mask = tts.gpt.prepare_gpt_inputs(conds_latent, text_tokens)
    tts.gpt.inference_model.store_mel_emb(inputs_embeds)
    
    # Note: use_autocast was already set above
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
            output = tts.gpt.inference_model.generate(
                input_ids,
                bos_token_id=tts.gpt.start_mel_token,
                pad_token_id=tts.gpt.stop_mel_token,
                eos_token_id=tts.gpt.stop_mel_token,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + tts.gpt.max_mel_tokens - 1,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=1,
            )
    
    trunc_index = input_ids.shape[1]
    codes = output[:, trunc_index:]
    
    if (codes == tts.stop_mel_token).any():
        stop_idx = (codes[0] == tts.stop_mel_token).nonzero(as_tuple=False)
        if len(stop_idx) > 0:
            codes = codes[:, :stop_idx[0].item()]
    
    code_lens = torch.tensor([codes.shape[1]], device=device)
    
    # GPT Forward for latents
    if emo_vec.dim() == 3:
        emo_vec_2d = emo_vec.squeeze(1)
    else:
        emo_vec_2d = emo_vec
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=use_autocast, dtype=tts.dtype or torch.float32):
            latent = tts.gpt(
                pattern_conditioned,
                text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=device),
                codes,
                code_lens,
                spk_cond_emb,
                cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device),
                emo_cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device),
                emo_vec=emo_vec_2d,
                use_speed=use_speed,
            )
    
    # S2Mel stage
    with torch.no_grad():
        diffusion_steps = 25
        inference_cfg_rate = 0.7
        
        latent = tts.s2mel.models['gpt_layer'](latent)
        S_infer = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)
        S_infer = S_infer + latent
        
        target_lengths = (code_lens * 1.72).long()
        
        cond = tts.s2mel.models['length_regulator'](
            S_infer, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]
        
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        
        vc_target = tts.s2mel.models['cfm'].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(device),
            ref_mel, style, None, diffusion_steps,
            inference_cfg_rate=inference_cfg_rate
        )
        vc_target = vc_target[:, :, ref_mel.size(-1):]
        
        wav = tts.bigvgan(vc_target.float()).squeeze()
    
    # Yield the audio chunk
    print('YIELDD')
    wav = wav.cpu().unsqueeze(0)
    yield wav


if __name__ == "__main__":
    main()