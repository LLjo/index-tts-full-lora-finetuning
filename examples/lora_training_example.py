#!/usr/bin/env python3
"""
Complete example of LoRA fine-tuning workflow for IndexTTS.

This script demonstrates:
1. Data preparation
2. Training configuration
3. Model training (conceptual)
4. Inference with LoRA

Usage:
    python examples/lora_training_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_data_structure():
    """
    Example 1: Recommended data structure for LoRA training.
    """
    print("=" * 80)
    print("EXAMPLE 1: Data Structure")
    print("=" * 80)
    
    structure = """
    Your project structure should look like this:
    
    my_voice_project/
    ├── audio/                          # Your audio files
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   ├── sample_003.wav
    │   └── ...
    ├── transcripts.csv                 # Transcriptions
    └── processed/                      # Will be created by prepare script
        ├── train_manifest.jsonl
        ├── val_manifest.jsonl
        ├── dataset_info.json
        └── features/
            ├── sample_001_text_ids.npy
            ├── sample_001_codes.npy
            └── ...
    
    transcripts.csv format:
    ┌──────────────────┬─────────────────────────────────┐
    │ filename         │ text                            │
    ├──────────────────┼─────────────────────────────────┤
    │ sample_001.wav   │ 这是第一个示例句子               │
    │ sample_002.wav   │ This is an English example      │
    │ sample_003.wav   │ Mixed languages work too        │
    └──────────────────┴─────────────────────────────────┘
    """
    print(structure)


def example_data_preparation():
    """
    Example 2: Data preparation command.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Data Preparation")
    print("=" * 80)
    
    command = """
    Step 1: Prepare your training data
    
    python tools/prepare_lora_dataset.py \\
      --audio-dir my_voice_project/audio \\
      --transcripts my_voice_project/transcripts.csv \\
      --output-dir my_voice_project/processed \\
      --model-dir checkpoints \\
      --config checkpoints/config.yaml \\
      --train-split 0.9 \\
      --max-duration 15.0 \\
      --min-duration 1.0 \\
      --device cuda
    
    What this does:
    - Loads your audio files and transcriptions
    - Extracts semantic features using W2V-BERT
    - Generates conditioning embeddings
    - Creates train/validation split (90/10)
    - Saves processed features and manifests
    
    Expected output:
    >> Using device: cuda
    >> Loaded tokenizer from: checkpoints/bpe.model
    >> Loaded semantic model
    >> Loaded semantic codec
    >> Loaded GPT model from: checkpoints/gpt.pth
    >> Loading transcripts from: my_voice_project/transcripts.csv
    >> Loaded 250 transcriptions
    >> Found 250 audio files
    >> Matched 250 audio files with transcriptions
    >> Processing audio files...
    >> Successfully processed 245 samples
    >> Train samples: 220
    >> Validation samples: 25
    >> Manifests saved:
       Train: my_voice_project/processed/train_manifest.jsonl
       Val: my_voice_project/processed/val_manifest.jsonl
    """
    print(command)


def example_training():
    """
    Example 3: Training command with explanations.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: LoRA Training")
    print("=" * 80)
    
    command = """
    Step 2: Train LoRA adapters
    
    # Basic training (recommended for first attempt)
    python tools/train_gpt_lora.py \\
      --train-manifest my_voice_project/processed/train_manifest.jsonl \\
      --val-manifest my_voice_project/processed/val_manifest.jsonl \\
      --output-dir trained_lora/my_voice \\
      --batch-size 8 \\
      --epochs 20 \\
      --learning-rate 3e-4 \\
      --lora-rank 8 \\
      --lora-alpha 16 \\
      --amp
    
    # Advanced training (with all options)
    python tools/train_gpt_lora.py \\
      --train-manifest my_voice_project/processed/train_manifest.jsonl \\
      --val-manifest my_voice_project/processed/val_manifest.jsonl \\
      --tokenizer checkpoints/bpe.model \\
      --config checkpoints/config.yaml \\
      --base-checkpoint checkpoints/gpt.pth \\
      --output-dir trained_lora/my_voice \\
      --lora-rank 8 \\
      --lora-alpha 16 \\
      --lora-dropout 0.05 \\
      --lora-include-heads \\
      --batch-size 8 \\
      --grad-accumulation 2 \\
      --epochs 20 \\
      --learning-rate 3e-4 \\
      --weight-decay 0.01 \\
      --warmup-steps 100 \\
      --log-interval 50 \\
      --val-interval 0 \\
      --save-interval 500 \\
      --grad-clip 1.0 \\
      --text-loss-weight 0.2 \\
      --mel-loss-weight 0.8 \\
      --amp \\
      --seed 1234
    
    Expected training output:
    >> Using device: cuda
    >> [Step 1/4] Loading base model...
    >> Base model loaded from: checkpoints/gpt.pth
    >> [Step 2/4] Applying LoRA adapters...
    >> trainable params: 2,359,296 || all params: 482,154,496 || trainable%: 0.4893
    >> [Step 3/4] Loading datasets...
    >> [Info] Loaded 220 samples from train_manifest.jsonl
    >> [Info] Loaded 25 samples from val_manifest.jsonl
    >> [Step 4/4] Setting up training...
    >> AMP enabled for faster training
    >> Starting LoRA training:
       Epochs: 20
       Batch size: 8
       Learning rate: 3e-4
       LoRA rank: 8
    >> [Train] epoch=1 step=50 text_loss=2.1234 mel_loss=1.5432 ...
    >> [Train] epoch=1 step=100 text_loss=1.8765 mel_loss=1.2345 ...
    >> Checkpoint saved at step 500
    >> [Val] epoch=5 text_loss=1.5432 mel_loss=1.0234 mel_top1=0.6543
    >> ...
    >> Training complete!
    >> Final LoRA checkpoint saved to: trained_lora/my_voice/final_checkpoint
    
    Training will create:
    trained_lora/my_voice/
    ├── checkpoint_step500/
    │   ├── adapter_config.json       # LoRA configuration
    │   ├── adapter_model.bin         # LoRA weights (~3-8 MB)
    │   └── training_metadata.json    # Training info
    ├── checkpoint_step1000/
    ├── final_checkpoint/              # Best checkpoint
    └── logs/
        └── lora_run_*/                # TensorBoard logs
    """
    print(command)


def example_inference():
    """
    Example 4: Using LoRA for inference.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Inference with LoRA")
    print("=" * 80)
    
    code = '''
    Step 3: Generate speech with your fine-tuned voice
    
    from indextts.infer_v2 import IndexTTS2
    
    # Initialize TTS with LoRA
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        lora_path="trained_lora/my_voice/final_checkpoint",
        use_fp16=True,
        device="cuda"
    )
    
    # Basic inference
    tts.infer(
        spk_audio_prompt="reference_voice.wav",  # Any sample of the target voice
        text="Hello! This is my custom fine-tuned voice speaking.",
        output_path="output.wav"
    )
    
    # Advanced inference with emotion control
    tts.infer(
        spk_audio_prompt="reference_voice.wav",
        text="I'm so excited about this new feature!",
        output_path="output_excited.wav",
        emo_vector=[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # Happy emotion
        # emo_vector format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    )
    
    # Using text-based emotion
    tts.infer(
        spk_audio_prompt="reference_voice.wav",
        text="This is a sad message about something unfortunate.",
        output_path="output_sad.wav",
        use_emo_text=True,  # Automatically detect emotion from text
        emo_alpha=0.7,      # Emotion strength (0.0-1.0)
    )
    '''
    print(code)


def example_comparison():
    """
    Example 5: Comparing different LoRA checkpoints.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comparing Checkpoints")
    print("=" * 80)
    
    code = '''
    Compare different training checkpoints to find the best one:
    
    from indextts.infer_v2 import IndexTTS2
    
    test_text = "This is a test sentence to compare checkpoint quality."
    reference_audio = "reference.wav"
    
    checkpoints = [
        "trained_lora/my_voice/checkpoint_step500",
        "trained_lora/my_voice/checkpoint_step1000",
        "trained_lora/my_voice/checkpoint_step1500",
        "trained_lora/my_voice/final_checkpoint",
    ]
    
    for i, ckpt_path in enumerate(checkpoints):
        print(f"Testing checkpoint: {ckpt_path}")
        
        # Load TTS with this checkpoint
        tts = IndexTTS2(
            model_dir="checkpoints",
            lora_path=ckpt_path,
            use_fp16=True
        )
        
        # Generate sample
        output_path = f"comparison_step{i*500}.wav"
        tts.infer(
            spk_audio_prompt=reference_audio,
            text=test_text,
            output_path=output_path
        )
        
        print(f"  Generated: {output_path}")
    
    print("\\nListen to the outputs and choose the best sounding checkpoint!")
    '''
    print(code)


def example_multiple_voices():
    """
    Example 6: Training and using multiple voices.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Multiple Voice LoRAs")
    print("=" * 80)
    
    code = '''
    Train separate LoRAs for different voices and switch between them:
    
    # Train voice 1
    python tools/train_gpt_lora.py \\
      --train-manifest data/voice1/train.jsonl \\
      --val-manifest data/voice1/val.jsonl \\
      --output-dir trained_lora/voice1
    
    # Train voice 2
    python tools/train_gpt_lora.py \\
      --train-manifest data/voice2/train.jsonl \\
      --val-manifest data/voice2/val.jsonl \\
      --output-dir trained_lora/voice2
    
    # Use in Python
    from indextts.infer_v2 import IndexTTS2
    
    # Voice 1
    tts1 = IndexTTS2(lora_path="trained_lora/voice1/final_checkpoint")
    tts1.infer(
        spk_audio_prompt="ref_voice1.wav",
        text="Speaking with voice 1",
        output_path="voice1_output.wav"
    )
    
    # Voice 2
    tts2 = IndexTTS2(lora_path="trained_lora/voice2/final_checkpoint")
    tts2.infer(
        spk_audio_prompt="ref_voice2.wav",
        text="Speaking with voice 2",
        output_path="voice2_output.wav"
    )
    
    # Base model (no LoRA)
    tts_base = IndexTTS2()  # No lora_path
    tts_base.infer(
        spk_audio_prompt="ref_any.wav",
        text="Speaking with base model",
        output_path="base_output.wav"
    )
    '''
    print(code)


def example_hyperparameter_tuning():
    """
    Example 7: Hyperparameter tuning guide.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Hyperparameter Tuning")
    print("=" * 80)
    
    guide = """
    Tune hyperparameters based on your observations:
    
    Problem: Training loss not decreasing
    Solutions:
    1. Increase learning rate:
       python tools/train_gpt_lora.py ... --learning-rate 5e-4
    
    2. Increase LoRA rank:
       python tools/train_gpt_lora.py ... --lora-rank 12
    
    3. Train longer:
       python tools/train_gpt_lora.py ... --epochs 30
    
    Problem: Out of GPU memory
    Solutions:
    1. Reduce batch size + increase gradient accumulation:
       python tools/train_gpt_lora.py ... --batch-size 4 --grad-accumulation 4
    
    2. Reduce LoRA rank:
       python tools/train_gpt_lora.py ... --lora-rank 6
    
    3. Disable AMP (if on older GPU):
       python tools/train_gpt_lora.py ... (remove --amp flag)
    
    Problem: Overfitting (val loss increases while train loss decreases)
    Solutions:
    1. Reduce LoRA rank:
       python tools/train_gpt_lora.py ... --lora-rank 6
    
    2. Increase dropout:
       python tools/train_gpt_lora.py ... --lora-dropout 0.1
    
    3. Reduce epochs:
       python tools/train_gpt_lora.py ... --epochs 15
    
    4. Add more training data
    
    Problem: Generated voice doesn't match target
    Solutions:
    1. Train longer:
       python tools/train_gpt_lora.py ... --epochs 30
    
    2. Use more training data (record more samples)
    
    3. Increase LoRA rank:
       python tools/train_gpt_lora.py ... --lora-rank 12
    
    4. Include embedding layers:
       python tools/train_gpt_lora.py ... --lora-include-embeddings
    """
    print(guide)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "IndexTTS LoRA Training Examples" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run all examples
    example_data_structure()
    example_data_preparation()
    example_training()
    example_inference()
    example_comparison()
    example_multiple_voices()
    example_hyperparameter_tuning()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary = """
    Complete LoRA Training Workflow:
    
    1. Prepare data structure (audio + transcripts.csv)
    2. Run prepare_lora_dataset.py to extract features
    3. Run train_gpt_lora.py to train LoRA adapters
    4. Test different checkpoints for best quality
    5. Use IndexTTS2(lora_path=...) for inference
    
    Key Files Created:
    - data/*/processed/train_manifest.jsonl  # Training data
    - trained_lora/*/final_checkpoint/       # LoRA weights
    - trained_lora/*/logs/                   # TensorBoard logs
    
    Documentation:
    - Quick Start: docs/LORA_QUICKSTART.md
    - Full Guide: docs/lora_training_guide.md
    - Main README: README_LORA.md
    
    Tips:
    - Start with 100-300 samples for testing
    - Use default hyperparameters first
    - Monitor training with TensorBoard
    - Compare multiple checkpoints
    - Quality audio > quantity
    
    Happy training! 🎙️
    """
    print(summary)


if __name__ == "__main__":
    main()