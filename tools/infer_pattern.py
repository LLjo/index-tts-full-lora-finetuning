#!/usr/bin/env python3
"""
Pattern-Conditioned Inference for IndexTTS2

This script uses pattern conditioning to reproduce learned speaking patterns.

The Key Insight:
================
For pattern training to work, you MUST use the SAME conditioning at inference
that was used during training. This script handles that automatically.

What you need:
1. Trained LoRA: training/speaker/lora/final_checkpoint/
2. Pattern conditioning: training/speaker/pattern_conditioning.pt
3. Speaker embeddings: training/speaker/embeddings/speaker_embeddings.pt

Usage:
    # Full pattern-conditioned inference
    python tools/infer_pattern.py \
        --speaker ozzy \
        --text "Life finds a way" \
        --output output.wav
    
    # With custom paths
    python tools/infer_pattern.py \
        --lora-path path/to/lora \
        --pattern-conditioning path/to/conditioning.pt \
        --embeddings path/to/embeddings.pt \
        --text "Your text here"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Pattern-conditioned inference for trained speaking patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Speaker shortcut
    parser.add_argument("--speaker", "-s",
                        help="Speaker name (auto-finds LoRA, conditioning, embeddings)")
    
    # Manual paths (override speaker auto-detection)
    parser.add_argument("--lora-path", type=Path,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--pattern-conditioning", type=Path,
                        help="Path to pattern_conditioning.pt")
    parser.add_argument("--embeddings", type=Path,
                        help="Path to speaker_embeddings.pt")
    
    # Inference options
    parser.add_argument("--text", "-t", required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", "-o", type=Path, default=Path("output_pattern.wav"),
                        help="Output audio path")
    
    # Generation options
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-p", type=float, default=0.8,
                        help="Top-p sampling (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Top-k sampling (default: 30)")
    parser.add_argument("--repetition-penalty", type=float, default=10.0,
                        help="Repetition penalty (default: 10.0)")
    
    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Generate both pattern-conditioned and base outputs for comparison")
    parser.add_argument("--reference-audio", type=Path,
                        help="Reference audio for comparison (without pattern conditioning)")
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.speaker:
        speaker_dir = PROJECT_ROOT / "training" / args.speaker
        lora_path = args.lora_path or speaker_dir / "lora" / "final_checkpoint"
        pattern_cond_path = args.pattern_conditioning or speaker_dir / "pattern_conditioning.pt"
        embeddings_path = args.embeddings or speaker_dir / "embeddings" / "speaker_embeddings.pt"
    else:
        if not args.lora_path or not args.pattern_conditioning or not args.embeddings:
            parser.error("Either --speaker or all of (--lora-path, --pattern-conditioning, --embeddings) required")
        lora_path = args.lora_path
        pattern_cond_path = args.pattern_conditioning
        embeddings_path = args.embeddings
    
    # Validate paths
    missing = []
    if not lora_path.exists():
        missing.append(f"LoRA checkpoint: {lora_path}")
    if not pattern_cond_path.exists():
        missing.append(f"Pattern conditioning: {pattern_cond_path}")
    if not embeddings_path.exists():
        missing.append(f"Speaker embeddings: {embeddings_path}")
    
    if missing:
        print("❌ Missing required files:")
        for m in missing:
            print(f"   - {m}")
        if args.speaker:
            print(f"\nSetup for speaker '{args.speaker}':")
            print(f"  1. python tools/extract_pattern_conditioning.py --speaker {args.speaker}")
            print(f"  2. python tools/prepare_pattern_dataset_v2.py --speaker {args.speaker} --pattern-conditioning {pattern_cond_path}")
            print(f"  3. python tools/train_gpt_lora.py --train-manifest ... --output-dir {lora_path.parent}")
            print(f"  4. python tools/extract_embeddings.py --speaker {args.speaker}")
        sys.exit(1)
    
    print("=" * 60)
    print("PATTERN-CONDITIONED INFERENCE")
    print("=" * 60)
    print(f"\nLoRA: {lora_path}")
    print(f"Pattern conditioning: {pattern_cond_path}")
    print(f"Speaker embeddings: {embeddings_path}")
    print(f"\nText: {args.text}")
    print(f"Output: {args.output}")
    
    # Load model
    print("\n[1/4] Loading model with LoRA...")
    from indextts.infer_v2 import IndexTTS2
    
    tts = IndexTTS2(
        lora_path=str(lora_path),
        use_cuda_kernel=False,
    )
    
    # Load pattern conditioning
    print("[2/4] Loading pattern conditioning...")
    from indextts.pattern_conditioning import PatternConditioningStore
    
    pattern_conditioning = PatternConditioningStore.load(pattern_cond_path, device=tts.device)
    
    # Load speaker embeddings
    print("[3/4] Loading speaker embeddings...")
    from indextts.speaker_embeddings import SpeakerEmbeddingStore
    
    store = SpeakerEmbeddingStore(tts)
    speaker_embeddings = store.load_embeddings(embeddings_path)
    
    # Inject pattern conditioning into speaker embeddings
    # The pattern conditioning overrides the GPT-level conditioning
    # while speaker embeddings handle the S2Mel stage
    print("[4/4] Combining pattern conditioning with speaker embeddings...")
    
    # Get GPT conditioning from pattern conditioning
    gpt_conditioning = pattern_conditioning['gpt_conditioning'].to(tts.device)  # (1, 32, dim)
    pattern_emo_vec = pattern_conditioning['emo_vec'].to(tts.device)  # (1, dim)
    
    # Create combined embeddings with pattern conditioning
    # We override the GPT-level conditioning while keeping S2Mel embeddings
    pattern_embeddings = {
        'spk_cond_emb': pattern_conditioning.get('spk_cond_emb', speaker_embeddings['spk_cond_emb']).to(tts.device),
        'style': speaker_embeddings['style'].to(tts.device),
        'prompt_condition': speaker_embeddings['prompt_condition'].to(tts.device),
        'ref_mel': speaker_embeddings['ref_mel'].to(tts.device),
        'emo_cond_emb': pattern_conditioning.get('spk_cond_emb', speaker_embeddings.get('emo_cond_emb', speaker_embeddings['spk_cond_emb'])).to(tts.device),
        # Pattern-specific overrides
        '_pattern_gpt_conditioning': gpt_conditioning,
        '_pattern_emo_vec': pattern_emo_vec,
    }
    
    # Generate with pattern conditioning
    print("\n>> Generating with pattern conditioning...")
    generate_with_pattern_conditioning(
        tts=tts,
        text=args.text,
        output_path=args.output,
        pattern_embeddings=pattern_embeddings,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    
    print(f"\n✓ Pattern-conditioned output saved: {args.output}")
    
    # Comparison mode
    if args.compare or args.reference_audio:
        print("\n>> Generating comparison outputs...")
        
        base_output = args.output.with_stem(args.output.stem + "_base")
        
        if args.reference_audio:
            # Use reference audio directly
            tts.infer(
                spk_audio_prompt=str(args.reference_audio),
                text=args.text,
                output_path=str(base_output),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            # Use speaker embeddings without pattern conditioning
            tts.infer(
                text=args.text,
                speaker_embeddings={k: v for k, v in speaker_embeddings.items() if not k.startswith('_')},
                output_path=str(base_output),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        
        print(f"✓ Base output saved: {base_output}")
        print(f"\nCompare the outputs to hear the difference in patterns!")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


def generate_with_pattern_conditioning(
    tts: 'IndexTTS2',
    text: str,
    output_path: Path,
    pattern_embeddings: dict,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    repetition_penalty: float = 10.0,
):
    """
    Generate audio using pattern conditioning.
    
    This is a custom implementation that injects pattern conditioning
    into the GPT stage while using speaker embeddings for S2Mel.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    
    device = tts.device
    
    # Get pattern conditioning
    pattern_gpt_cond = pattern_embeddings.get('_pattern_gpt_conditioning')
    pattern_emo = pattern_embeddings.get('_pattern_emo_vec')
    
    # Get speaker embeddings for S2Mel
    spk_cond_emb = pattern_embeddings['spk_cond_emb']
    style = pattern_embeddings['style']
    prompt_condition = pattern_embeddings['prompt_condition']
    ref_mel = pattern_embeddings['ref_mel']
    emo_cond_emb = pattern_embeddings['emo_cond_emb']
    
    # Tokenize text
    text_tokens_list = tts.tokenizer.tokenize(text)
    segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=120)
    
    wavs = []
    
    for segment in segments:
        text_tokens = tts.tokenizer.convert_tokens_to_ids(segment)
        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.int32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            dtype = torch.float16 if tts.use_fp16 else None
            with torch.amp.autocast(device, enabled=dtype is not None, dtype=dtype):
                # Get base model
                gpt = tts.gpt
                
                # Use pattern conditioning if available, otherwise compute from embeddings
                if pattern_gpt_cond is not None:
                    # Use pre-computed pattern conditioning directly
                    speech_conditioning_latent = pattern_gpt_cond
                    
                    if pattern_emo is not None:
                        emo_vec = pattern_emo
                    else:
                        # Compute emotion from embeddings
                        emo_vec = gpt.get_emovec(emo_cond_emb, torch.tensor([emo_cond_emb.shape[1]], device=device))
                else:
                    # Fall back to computing from embeddings
                    cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)
                    speech_conditioning_latent = gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
                    emo_vec = gpt.get_emovec(emo_cond_emb, cond_lengths)
                
                # Merge emotion
                emovec = emo_vec
                
                # Build conditioning tensor
                tmp = torch.zeros(text_tokens_tensor.size(0)).to(device)
                duration_emb = gpt.speed_emb(torch.zeros_like(tmp).long())
                duration_emb_half = gpt.speed_emb(torch.ones_like(tmp).long())
                conds_latent = torch.cat((
                    speech_conditioning_latent + emovec.unsqueeze(1),
                    duration_emb_half.unsqueeze(1),
                    duration_emb.unsqueeze(1)
                ), 1)
                
                # Use pattern-conditioned inference
                input_ids, inputs_embeds, attention_mask = gpt.prepare_gpt_inputs(conds_latent, text_tokens_tensor)
                gpt.inference_model.store_mel_emb(inputs_embeds)
                
                trunc_index = input_ids.shape[1]
                max_length = trunc_index + gpt.max_mel_tokens - 1
                
                # Generate codes
                output = gpt.inference_model.generate(
                    input_ids,
                    bos_token_id=gpt.start_mel_token,
                    pad_token_id=gpt.stop_mel_token,
                    eos_token_id=gpt.stop_mel_token,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=1,
                )
                
                codes = output[:, trunc_index:]
                
                # Get code lengths
                code_lens = []
                for code in codes:
                    if gpt.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == gpt.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                max_code_len = max(code_lens)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens).to(device)
                
                # Get GPT latent
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(device).long()
                latent = gpt(
                    speech_conditioning_latent,
                    text_tokens_tensor,
                    torch.tensor([text_tokens_tensor.shape[-1]], device=device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=device),
                    emo_cond_emb,
                    cond_mel_lengths=torch.tensor([spk_cond_emb.shape[1]], device=device),
                    emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[1]], device=device),
                    emo_vec=emovec,
                    use_speed=use_speed,
                )
                
                # S2Mel stage
                diffusion_steps = 25
                inference_cfg_rate = 0.7
                latent = tts.s2mel.models['gpt_layer'](latent)
                S_infer = tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()
                
                cond = tts.s2mel.models['length_regulator'](
                    S_infer,
                    ylens=target_lengths,
                    n_quantizers=3,
                    f0=None
                )[0]
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = tts.s2mel.models['cfm'].inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    ref_mel, style, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate
                )
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                
                # Vocoder
                wav = tts.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                wav = wav.squeeze(1)
                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav.cpu())
    
    # Concatenate and save
    import torchaudio
    
    wav = torch.cat(wavs, dim=1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), wav.type(torch.int16), 22050)


if __name__ == "__main__":
    main()