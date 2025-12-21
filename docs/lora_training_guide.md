# LoRA Fine-Tuning Guide for IndexTTS

This guide explains how to fine-tune IndexTTS for custom voices using LoRA (Low-Rank Adaptation), enabling efficient voice adaptation with minimal data and computational resources.

## Table of Contents

- [Overview](#overview)
- [What is LoRA?](#what-is-lora)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Overview

LoRA fine-tuning allows you to adapt IndexTTS to a specific voice with:

- **Minimal data**: 50-500 audio samples (2-20 minutes of audio)
- **Fast training**: 1-4 hours on consumer GPUs
- **Small checkpoints**: ~1-10 MB (vs ~1 GB for full model)
- **High quality**: Preserves base model quality while adapting speaker characteristics

---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:

1. **Freezes** the base model weights (no changes)
2. **Injects** small trainable adapter layers
3. **Trains** only the adapters (~0.1-1% of parameters)
4. **Merges** adapters with base model for inference (optional)

**Benefits:**
- 100x faster training than full fine-tuning
- 100x smaller checkpoints
- Can switch between multiple voices easily
- Prevents catastrophic forgetting of base model

---

## Requirements

### Hardware

**Minimum:**
- GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060)
- CPU: Modern multi-core processor
- RAM: 16GB+
- Disk: 10GB free space

**Recommended:**
- GPU: NVIDIA GPU with 12GB+ VRAM (e.g., RTX 3080, RTX 4070)
- CPU: 8+ cores
- RAM: 32GB+
- Disk: 50GB free space (for larger datasets)

**Also supported:**
- Apple Silicon (M1/M2/M3) via MPS backend
- Intel GPUs via XPU backend
- CPU-only (much slower, not recommended)

### Software

```bash
# Install IndexTTS with LoRA support
pip install peft>=0.7.0

# Or update dependencies
uv sync  # if using uv
```

### Data Requirements

**Audio Files:**
- Format: WAV, MP3, FLAC (will be converted to 16kHz/22kHz)
- Quality: Clean, minimal background noise
- Duration per clip: 2-10 seconds ideal
- Total clips:
  - Minimum: 50-100 clips (2-5 minutes)
  - Recommended: 200-500 clips (10-20 minutes)
  - Optimal: 1000+ clips (30+ minutes)

**Transcriptions:**
- Accurate text matching the audio
- Proper punctuation and formatting
- Languages: Supports multiple languages (Chinese, English, etc.)

---

## Quick Start

### 1. Prepare Your Data

Organize your files:

```
my_voice_data/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcripts.csv
```

**transcripts.csv:**
```csv
filename,text
sample_001.wav,这是第一个示例句子
sample_002.wav,This is an English example
sample_003.wav,混合语言也可以使用
```

Or **transcripts.json:**
```json
[
  {"filename": "sample_001.wav", "text": "这是第一个示例句子"},
  {"filename": "sample_002.wav", "text": "This is an English example"}
]
```

### 2. Preprocess Data

```bash
python tools/prepare_lora_dataset.py \
  --audio-dir my_voice_data/audio \
  --transcripts my_voice_data/transcripts.csv \
  --output-dir data/my_voice_processed
```

This will:
- Extract semantic features from audio
- Tokenize text
- Generate conditioning embeddings
- Create train/val manifests
- Save to `data/my_voice_processed/`

### 3. Train LoRA

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/my_voice_processed/train_manifest.jsonl \
  --val-manifest data/my_voice_processed/val_manifest.jsonl \
  --output-dir trained_lora/my_voice \
  --batch-size 8 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --lora-rank 8 \
  --amp
```

Training will take 1-4 hours depending on dataset size and hardware.

### 4. Use for Inference

**Python API:**
```python
from indextts.infer_v2 import IndexTTS2

# Load model with LoRA
tts = IndexTTS2(
    model_dir="checkpoints",
    lora_path="trained_lora/my_voice/final_checkpoint"
)

# Generate speech
tts.infer(
    spk_audio_prompt="reference.wav",
    text="Hello, this is my custom voice!",
    output_path="output.wav"
)
```

**Command Line (coming soon):**
```bash
indextts "Hello world" \
  --voice reference.wav \
  --lora trained_lora/my_voice/final_checkpoint \
  --output output.wav
```

---

## Step-by-Step Guide

### Step 1: Data Collection

**Guidelines for recording:**

1. **Environment:**
   - Quiet room with minimal echo
   - Consistent recording setup
   - Same microphone/equipment

2. **Speaking style:**
   - Natural, conversational tone
   - Consistent volume and pace
   - Clear pronunciation

3. **Content diversity:**
   - Cover various phonemes
   - Different sentence structures
   - Mix of questions, statements, exclamations
   - Emotional variety (if desired)

4. **Technical quality:**
   - Sample rate: 16kHz or higher
   - Bit depth: 16-bit or 24-bit
   - Format: Lossless (WAV) preferred

**Example recording script:**

```python
# Record audio samples
import sounddevice as sd
import soundfile as sf

# List of sentences to record
sentences = [
    "这是第一个示例句子。",
    "This is an English example.",
    # ... more sentences
]

for i, text in enumerate(sentences):
    print(f"Record: {text}")
    input("Press Enter when ready...")
    
    duration = 5  # seconds
    fs = 22050
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    sf.write(f"audio/sample_{i:03d}.wav", recording, fs)
    print(f"Saved: sample_{i:03d}.wav\n")
```

### Step 2: Data Preparation

**Full command with all options:**

```bash
python tools/prepare_lora_dataset.py \
  --audio-dir my_voice_data/audio \
  --transcripts my_voice_data/transcripts.csv \
  --output-dir data/my_voice_processed \
  --model-dir checkpoints \
  --config checkpoints/config.yaml \
  --train-split 0.9 \
  --max-duration 15.0 \
  --min-duration 1.0 \
  --device cuda \
  --seed 42
```

**Parameters:**
- `--audio-dir`: Directory containing audio files
- `--transcripts`: CSV or JSON with transcriptions
- `--output-dir`: Where to save processed features
- `--model-dir`: IndexTTS checkpoint directory
- `--train-split`: Fraction for training (0.9 = 90% train, 10% val)
- `--max-duration`: Skip audio longer than this (seconds)
- `--min-duration`: Skip audio shorter than this (seconds)
- `--device`: cuda/cpu/mps/xpu (auto-detected if omitted)

**Output:**
```
data/my_voice_processed/
├── train_manifest.jsonl      # Training samples
├── val_manifest.jsonl        # Validation samples
├── dataset_info.json         # Dataset statistics
└── features/
    ├── sample_001_text_ids.npy
    ├── sample_001_codes.npy
    ├── sample_001_condition.npy
    ├── sample_001_emo_vec.npy
    └── ...
```

### Step 3: Training

**Basic training command:**

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/my_voice_processed/train_manifest.jsonl \
  --val-manifest data/my_voice_processed/val_manifest.jsonl \
  --output-dir trained_lora/my_voice \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --lora-rank 8 \
  --amp
```

**Advanced training command:**

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/my_voice_processed/train_manifest.jsonl \
  --val-manifest data/my_voice_processed/val_manifest.jsonl \
  --tokenizer checkpoints/bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt.pth \
  --output-dir trained_lora/my_voice \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-include-heads \
  --batch-size 8 \
  --grad-accumulation 2 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --warmup-steps 100 \
  --log-interval 50 \
  --val-interval 0 \
  --save-interval 500 \
  --grad-clip 1.0 \
  --text-loss-weight 0.2 \
  --mel-loss-weight 0.8 \
  --amp \
  --seed 1234
```

**Monitoring training:**

```bash
# View logs in real-time
tail -f trained_lora/my_voice/logs/*/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir trained_lora/my_voice/logs
```

**Training output:**
```
trained_lora/my_voice/
├── checkpoint_step500/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_metadata.json
├── checkpoint_step1000/
├── final_checkpoint/
└── logs/
    └── lora_run_20231219_143000/
```

### Step 4: Inference

**Load and use LoRA:**

```python
from indextts.infer_v2 import IndexTTS2

# Initialize with LoRA
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    lora_path="trained_lora/my_voice/final_checkpoint",
    use_fp16=True,
    device="cuda"
)

# Basic inference
output = tts.infer(
    spk_audio_prompt="reference_voice.wav",
    text="Hello, this is my fine-tuned voice speaking!",
    output_path="output.wav"
)

# Advanced inference with emotion control
output = tts.infer(
    spk_audio_prompt="reference_voice.wav",
    text="I'm so excited to test this!",
    output_path="output_excited.wav",
    emo_vector=[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # Happy emotion
    use_emo_text=False
)
```

**Switching between voices:**

```python
# Base voice (no LoRA)
tts_base = IndexTTS2(model_dir="checkpoints")

# Voice 1
tts_voice1 = IndexTTS2(
    model_dir="checkpoints",
    lora_path="trained_lora/voice1/final_checkpoint"
)

# Voice 2
tts_voice2 = IndexTTS2(
    model_dir="checkpoints",
    lora_path="trained_lora/voice2/final_checkpoint"
)
```

---

## Hyperparameter Tuning

### LoRA Rank

**Effect:** Controls adapter capacity

| Rank | Use Case | Pros | Cons |
|------|----------|------|------|
| 4 | Small datasets (<100 samples) | Fast, prevents overfitting | Limited capacity |
| 8 | Medium datasets (100-500) | **Recommended**, good balance | - |
| 12-16 | Large datasets (500+) | High capacity | Slower, larger checkpoints |

**When to increase:**
- Training loss isn't decreasing
- Large, diverse dataset
- Complex voice characteristics

**When to decrease:**
- Overfitting (val loss increasing)
- Small dataset
- Limited GPU memory

### Learning Rate

**Effect:** Controls training speed

| LR | Use Case | Behavior |
|----|----------|----------|
| 1e-4 | Conservative | Slow but stable |
| 3e-4 | **Recommended** | Good balance |
| 5e-4 | Aggressive | Fast but may be unstable |

**Signs LR is too high:**
- Loss oscillates or diverges
- NaN values appear
- Unstable training

**Signs LR is too low:**
- Loss decreases very slowly
- Training takes too long
- Plateaus early

### Batch Size

**Effect:** Training stability and speed

| Batch Size | GPU Memory | Training |
|------------|------------|----------|
| 4 | 8GB | Stable but slow |
| 8 | 12GB | **Recommended** |
| 16 | 24GB | Fast |

**Effective batch size = batch_size × grad_accumulation**

Example: `batch_size=4 grad_accumulation=4` = effective batch of 16

### Epochs

**Effect:** Training duration

| Dataset Size | Recommended Epochs |
|--------------|-------------------|
| <100 samples | 15-25 |
| 100-500 samples | 10-20 |
| 500+ samples | 5-15 |

**Early stopping criteria:**
- Validation loss stops improving for 3-5 epochs
- Training loss << validation loss (overfitting)

---

## Troubleshooting

### Training Issues

**Problem: Training loss not decreasing**

Solutions:
- Increase `learning-rate` (try 5e-4 or 1e-3)
- Increase `lora-rank` (try 12 or 16)
- Check data quality (corrupted audio, mismatched transcripts)
- Reduce `batch-size` and increase `grad-accumulation`

**Problem: Out of memory (OOM)**

Solutions:
- Reduce `batch-size` (try 4 or 2)
- Increase `grad-accumulation` to maintain effective batch size
- Enable `--amp` for mixed precision
- Use smaller `lora-rank`
- Close other applications

**Problem: Overfitting (val loss increasing)**

Solutions:
- Decrease `lora-rank` (try 4 or 6)
- Increase `lora-dropout` (try 0.1 or 0.2)
- Reduce `epochs`
- Add more training data
- Increase `weight-decay`

**Problem: NaN losses**

Solutions:
- Decrease `learning-rate` (try 1e-4)
- Enable gradient clipping (`--grad-clip 1.0`)
- Check for corrupted audio files
- Reduce batch size

### Data Preparation Issues

**Problem: No samples processed**

Solutions:
- Check filename matching between audio and transcripts
- Verify audio file formats
- Check for corrupted audio files
- Review duration constraints (`--min-duration`, `--max-duration`)

**Problem: Unknown tokens in text**

Solutions:
- Check BPE model path
- Ensure text is properly normalized
- Update BPE model with new vocabulary
- Check for special characters

### Inference Issues

**Problem: Generated audio sounds like base model**

Solutions:
- Verify LoRA loaded correctly (check console output)
- Try training longer (more epochs)
- Increase `lora-rank`
- Check reference audio quality

**Problem: Poor audio quality**

Solutions:
- Use higher quality reference audio
- Check LoRA checkpoint (try different checkpoints)
- Adjust inference parameters (temperature, top-p)
- Retrain with more/better data

---

## Advanced Usage

### Merging LoRA into Base Model

For deployment, merge LoRA into a single checkpoint:

```python
from indextts.utils.lora_utils import load_lora_checkpoint, merge_lora_weights
from indextts.gpt.model_v2 import UnifiedVoice
import torch

# Load base model
base_model = UnifiedVoice(**config.gpt)
base_model.load_state_dict(torch.load("checkpoints/gpt.pth")["model"])

# Load LoRA
lora_model = load_lora_checkpoint(
    base_model,
    "trained_lora/my_voice/final_checkpoint",
    merge_weights=False
)

# Merge and save
merged_model = merge_lora_weights(
    lora_model,
    output_path="checkpoints/my_voice_merged.pth"
)
```

### Training on Multiple Datasets

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/voice1/train_manifest.jsonl \
  --train-manifest data/voice2/train_manifest.jsonl \
  --val-manifest data/voice1/val_manifest.jsonl \
  --val-manifest data/voice2/val_manifest.jsonl \
  --output-dir trained_lora/multi_voice
```

### Custom Target Modules

Target specific layers:

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/my_voice/train_manifest.jsonl \
  --val-manifest data/my_voice/val_manifest.jsonl \
  --lora-target-modules "gpt.h.*.attn.c_attn" "gpt.h.*.mlp.c_fc" \
  --output-dir trained_lora/custom_target
```

### Resuming Training

(Future feature - save optimizer state)

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/my_voice/train_manifest.jsonl \
  --val-manifest data/my_voice/val_manifest.jsonl \
  --resume trained_lora/my_voice/checkpoint_step500 \
  --output-dir trained_lora/my_voice_continued
```

---

## Best Practices

### Data Collection

1. **Quality over quantity**: 200 high-quality samples > 1000 noisy samples
2. **Consistency**: Same recording environment and equipment
3. **Diversity**: Cover various phonemes, emotions, speaking styles
4. **Accuracy**: Double-check transcriptions

### Training

1. **Start conservative**: Use default hyperparameters first
2. **Monitor closely**: Watch training/validation losses
3. **Save often**: Keep checkpoint_step500, checkpoint_step1000, etc.
4. **Validate early**: Run inference tests after 5-10 epochs
5. **Experiment systematically**: Change one parameter at a time

### Deployment

1. **Test thoroughly**: Generate diverse samples
2. **Compare checkpoints**: Try different training steps
3. **Document settings**: Record hyperparameters that worked
4. **Version control**: Tag successful models

---

## FAQ

**Q: How much data do I need?**
A: Minimum 50 samples, recommended 200-500 for best results.

**Q: How long does training take?**
A: 1-4 hours on RTX 3080, depending on dataset size.

**Q: Can I fine-tune for multiple speakers?**
A: Yes, but train separate LoRAs. Or combine datasets for multi-speaker adaptation.

**Q: Does LoRA affect base model quality?**
A: No, base model stays frozen. You can switch back anytime.

**Q: Can I use LoRA with other languages?**
A: Yes, IndexTTS supports multiple languages.

**Q: What if I have limited GPU memory?**
A: Use `batch-size=2`, `grad-accumulation=8`, and `--amp`.

**Q: How do I choose the best checkpoint?**
A: Listen to samples from different checkpoints. Usually final or lowest validation loss.

---

## Resources

- **IndexTTS Repository**: https://github.com/index-tts/index-tts
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/index-tts/index-tts/issues
- Discussions: https://github.com/index-tts/index-tts/discussions

---

**Happy training! 🎙️**