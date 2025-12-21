# LoRA Fine-Tuning Quick Start

Quick guide to fine-tune IndexTTS on your custom voice in 4 simple steps!

## Prerequisites

```bash
# Install LoRA support
pip install peft>=0.7.0
```

## Step 1: Prepare Your Data

Create this structure:
```
my_voice/
├── audio/           # Your WAV/MP3 files
│   ├── clip1.wav
│   ├── clip2.wav
│   └── ...
└── transcripts.csv  # Transcriptions
```

**transcripts.csv:**
```csv
filename,text
clip1.wav,这是第一句话
clip2.wav,This is the second sentence
```

## Step 2: Process Dataset

```bash
python tools/prepare_lora_dataset.py \
  --audio-dir my_voice/audio \
  --transcripts my_voice/transcripts.csv \
  --output-dir data/my_voice_processed
```

Takes 5-15 minutes depending on dataset size.

## Step 3: Train LoRA

```bash
python tools/train_gpt_lora.py \
  --train-manifest data/my_voice_processed/train_manifest.jsonl \
  --val-manifest data/my_voice_processed/val_manifest.jsonl \
  --output-dir trained_lora/my_voice \
  --batch-size 8 \
  --epochs 20 \
  --lora-rank 8 \
  --amp
```

Takes 1-4 hours on RTX 3080.

## Step 4: Use Your Voice

```python
from indextts.infer_v2 import IndexTTS2

# Load with LoRA
tts = IndexTTS2(
    lora_path="trained_lora/my_voice/final_checkpoint"
)

# Generate speech
tts.infer(
    spk_audio_prompt="reference.wav",  # Any sample of the voice
    text="Hello! This is my custom voice.",
    output_path="output.wav"
)
```

## Data Requirements

| Dataset Size | Samples | Duration | Quality |
|--------------|---------|----------|---------|
| Minimal | 50-100 | 2-5 min | Testing |
| Recommended | 200-500 | 10-20 min | Good |
| Optimal | 1000+ | 30+ min | Best |

## Common Issues

**Out of memory?**
```bash
# Use smaller batch size
--batch-size 4 --grad-accumulation 4
```

**Training not working?**
```bash
# Check your data was processed correctly
ls data/my_voice_processed/features/
# Should see .npy files
```

**Generated audio doesn't sound like target voice?**
- Train longer (try 30 epochs)
- Use more/better quality training data
- Increase LoRA rank to 12

## Next Steps

- Read full guide: [`docs/lora_training_guide.md`](lora_training_guide.md:1)
- See example config: [`configs/lora_finetune_example.yaml`](../configs/lora_finetune_example.yaml:1)
- Experiment with hyperparameters

## Tips

1. **Start small**: Test with 50 samples first
2. **Quality matters**: Clean audio > quantity
3. **Monitor training**: Check losses decrease
4. **Test early**: Generate samples after 10 epochs
5. **Compare checkpoints**: Try step500, step1000, final

---

For detailed documentation, see [`lora_training_guide.md`](lora_training_guide.md:1)