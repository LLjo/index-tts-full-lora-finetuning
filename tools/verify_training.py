#!/usr/bin/env python3
"""
Training Verification Tool for IndexTTS2

This tool helps verify that your LoRA or full fine-tuning is actually working
by comparing the SEMANTIC TOKEN outputs of the trained vs base model.

KEY INSIGHT:
============
IndexTTS2 has TWO stages:
1. GPT: Generates semantic tokens (THIS IS WHAT YOU TRAIN)
2. S2Mel: Converts tokens to audio using REFERENCE AUDIO features

If you can't hear differences, it's because S2Mel uses the reference audio's
voice characteristics regardless of what the GPT produces. What the training
ACTUALLY changes is the token sequence (prosody, rhythm, phrasing).

This tool shows you whether your GPT is producing different tokens,
which confirms the training is working even if you can't hear it due to S2Mel.

Usage:
    python tools/verify_training.py --speaker goldblum --text "Hello world"
    python tools/verify_training.py --lora-path trained_lora/final_checkpoint --audio ref.wav --text "Test"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def compare_token_outputs(
    base_model,
    trained_model,
    reference_audio: str,
    text: str,
    verbose: bool = True
):
    """
    Compare semantic token outputs between base and trained models.
    
    Returns:
        dict with comparison statistics
    """
    
    # Get text tokens
    text_tokens_list = base_model.tokenizer.tokenize(text)
    text_token_ids = base_model.tokenizer.convert_tokens_to_ids(text_tokens_list)
    text_tokens = torch.tensor(text_token_ids, dtype=torch.int32, device=base_model.device).unsqueeze(0)
    
    # Load reference audio for conditioning
    import librosa
    import torchaudio
    
    audio, sr = librosa.load(reference_audio, sr=None, mono=True)
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_tensor)
    
    # Get speaker conditioning (same for both models)
    inputs = base_model.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(base_model.device)
    attention_mask = inputs["attention_mask"].to(base_model.device)
    spk_cond_emb = base_model.get_emb(input_features, attention_mask)
    
    # Get emotion conditioning
    emo_cond_emb = spk_cond_emb.clone()
    
    # Compute emotion vector
    emovec = base_model.gpt.merge_emovec(
        spk_cond_emb,
        emo_cond_emb,
        torch.tensor([spk_cond_emb.shape[1]], device=base_model.device),
        torch.tensor([emo_cond_emb.shape[1]], device=base_model.device),
        alpha=1.0
    )
    
    # Generate with base model
    if verbose:
        print("\n>> Generating with BASE model...")
    with torch.no_grad():
        base_codes, _ = base_model.gpt.inference_speech(
            spk_cond_emb,
            text_tokens,
            emo_cond_emb,
            cond_lengths=torch.tensor([spk_cond_emb.shape[1]], device=base_model.device),
            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[1]], device=base_model.device),
            emo_vec=emovec,
            do_sample=False,  # Deterministic for comparison
            temperature=0.1,
            top_k=1,
            max_generate_length=500,
        )
    
    # Generate with trained model
    if verbose:
        print(">> Generating with TRAINED model...")
    with torch.no_grad():
        # Transfer conditioning to trained model's device
        trained_device = trained_model.device
        spk_cond_emb_t = spk_cond_emb.to(trained_device)
        emo_cond_emb_t = emo_cond_emb.to(trained_device)
        text_tokens_t = text_tokens.to(trained_device)
        
        emovec_t = trained_model.gpt.merge_emovec(
            spk_cond_emb_t,
            emo_cond_emb_t,
            torch.tensor([spk_cond_emb_t.shape[1]], device=trained_device),
            torch.tensor([emo_cond_emb_t.shape[1]], device=trained_device),
            alpha=1.0
        )
        
        trained_codes, _ = trained_model.gpt.inference_speech(
            spk_cond_emb_t,
            text_tokens_t,
            emo_cond_emb_t,
            cond_lengths=torch.tensor([spk_cond_emb_t.shape[1]], device=trained_device),
            emo_cond_lengths=torch.tensor([emo_cond_emb_t.shape[1]], device=trained_device),
            emo_vec=emovec_t,
            do_sample=False,
            temperature=0.1,
            top_k=1,
            max_generate_length=500,
        )
    
    # Compare tokens
    base_codes = base_codes.cpu().numpy().flatten()
    trained_codes = trained_codes.cpu().numpy().flatten()
    
    # Truncate to same length
    min_len = min(len(base_codes), len(trained_codes))
    base_codes = base_codes[:min_len]
    trained_codes = trained_codes[:min_len]
    
    # Calculate differences
    different_positions = np.sum(base_codes != trained_codes)
    total_positions = len(base_codes)
    diff_percentage = (different_positions / total_positions) * 100 if total_positions > 0 else 0
    
    results = {
        "base_codes": base_codes,
        "trained_codes": trained_codes,
        "base_length": len(base_codes),
        "trained_length": len(trained_codes),
        "different_tokens": int(different_positions),
        "total_tokens": total_positions,
        "difference_percentage": diff_percentage,
        "codes_identical": diff_percentage == 0,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("TOKEN COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Base model tokens:    {results['base_length']}")
        print(f"Trained model tokens: {results['trained_length']}")
        print(f"Different positions:  {results['different_tokens']} / {results['total_tokens']}")
        print(f"Difference:           {results['difference_percentage']:.2f}%")
        
        if results['codes_identical']:
            print("\n⚠️  WARNING: Token outputs are IDENTICAL!")
            print("   This means the training had NO EFFECT on generation.")
            print("   Possible causes:")
            print("   1. LoRA weights are all zero (training didn't converge)")
            print("   2. LoRA wasn't properly applied to the model")
            print("   3. Model was loaded without LoRA adapters")
        elif results['difference_percentage'] < 10:
            print("\n⚡ Token outputs differ slightly.")
            print("   Training is working but changes are subtle.")
            print("   This is NORMAL - training affects prosody/rhythm,")
            print("   but S2Mel determines the voice timbre.")
        else:
            print("\n✅ Significant difference in token outputs!")
            print(f"   {results['difference_percentage']:.1f}% of tokens are different.")
            print("   Your training IS working!")
            print("\n   If you still can't hear a difference, it's because")
            print("   the S2Mel stage uses reference audio features.")
        
        # Show first few different tokens
        if different_positions > 0:
            print(f"\nFirst 10 positions with different tokens:")
            count = 0
            for i in range(min_len):
                if base_codes[i] != trained_codes[i]:
                    print(f"  Position {i}: base={base_codes[i]}, trained={trained_codes[i]}")
                    count += 1
                    if count >= 10:
                        break
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify training effectiveness by comparing token outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument("--speaker", "-s",
                        help="Speaker name (loads from training/ directory)")
    parser.add_argument("--lora-path",
                        help="Path to LoRA checkpoint")
    parser.add_argument("--gpt-checkpoint",
                        help="Path to fine-tuned GPT checkpoint")
    parser.add_argument("--audio", "-a",
                        help="Reference audio for conditioning")
    parser.add_argument("--text", "-t", default="Hello, this is a test of the training verification tool.",
                        help="Text to generate")
    
    # Model options
    parser.add_argument("--config", default="checkpoints/config.yaml")
    parser.add_argument("--model-dir", default="checkpoints")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.speaker and not (args.lora_path or args.gpt_checkpoint):
        parser.error("Either --speaker or --lora-path/--gpt-checkpoint is required")
    
    # Import here to avoid slow startup
    from indextts.infer_v2 import IndexTTS2
    
    PROJECT_ROOT = Path(__file__).parent.parent
    TRAINING_DIR = PROJECT_ROOT / "training"
    
    # Determine paths
    lora_path = None
    gpt_checkpoint = None
    audio_path = None
    
    if args.speaker:
        speaker_dir = TRAINING_DIR / args.speaker
        
        # Try to find LoRA checkpoint
        lora_checkpoint = speaker_dir / "lora" / "final_checkpoint"
        if lora_checkpoint.exists():
            lora_path = str(lora_checkpoint)
            print(f">> Found LoRA checkpoint: {lora_path}")
        
        # Try to find fine-tuned checkpoint
        ft_checkpoint = speaker_dir / "finetune" / "best_model.pth"
        if not lora_path and ft_checkpoint.exists():
            gpt_checkpoint = str(ft_checkpoint)
            print(f">> Found fine-tuned checkpoint: {gpt_checkpoint}")
        
        # Get reference audio
        audio_dir = speaker_dir / "dataset" / "audio"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav"))
            if audio_files:
                audio_path = str(audio_files[0])
                print(f">> Using reference audio: {audio_path}")
        
        if not lora_path and not gpt_checkpoint:
            print(f"\n❌ No trained checkpoints found for speaker: {args.speaker}")
            print(f"   Looked in: {speaker_dir}")
            sys.exit(1)
    else:
        lora_path = args.lora_path
        gpt_checkpoint = args.gpt_checkpoint
        audio_path = args.audio
    
    if args.audio:
        audio_path = args.audio
    
    if not audio_path:
        print("\n❌ Reference audio is required")
        print("   Use --audio or ensure speaker has training audio")
        sys.exit(1)
    
    # Load base model
    print("\n>> Loading BASE model (no training)...")
    base_model = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_cuda_kernel=False,
    )
    
    # Load trained model
    print("\n>> Loading TRAINED model...")
    trained_model = IndexTTS2(
        cfg_path=args.config,
        model_dir=args.model_dir,
        lora_path=lora_path,
        gpt_checkpoint=gpt_checkpoint,
        use_cuda_kernel=False,
    )
    
    # Compare outputs
    print(f"\n>> Comparing token outputs for: '{args.text}'")
    results = compare_token_outputs(
        base_model,
        trained_model,
        audio_path,
        args.text,
        verbose=True
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if results['codes_identical']:
        print("\n❌ TRAINING VERIFICATION FAILED")
        print("   The trained model produces IDENTICAL outputs to the base model.")
        print("\n   Suggestions:")
        print("   1. Run: python tools/diagnose_training.py --lora-path <path>")
        print("      to check if LoRA weights are non-zero")
        print("   2. Check training logs - did the loss decrease?")
        print("   3. Ensure you trained for enough epochs")
    else:
        print("\n✅ TRAINING VERIFICATION PASSED")
        print(f"   Token difference: {results['difference_percentage']:.1f}%")
        print("\n   Your training IS working! If you can't hear differences:")
        print("   - This is EXPECTED behavior due to the S2Mel architecture")
        print("   - S2Mel uses reference audio to determine voice timbre")
        print("   - Training affects prosody, rhythm, and token selection")
        print("\n   To hear maximal differences:")
        print("   1. Extract embeddings from your training audio:")
        print(f"      python tools/extract_embeddings.py --speaker {args.speaker or 'your_speaker'}")
        print("   2. Use different reference audio from the same speaker")
        print("   3. Compare long passages where rhythm/prosody matter more")


if __name__ == "__main__":
    main()