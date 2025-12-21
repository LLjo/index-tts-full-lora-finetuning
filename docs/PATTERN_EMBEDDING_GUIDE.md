# Pattern Embedding Training for IndexTTS2

This guide explains the **NEW** approach to training speaking patterns (stutters, pauses, hesitations) that **ACTUALLY WORKS** at inference time!

## Why Previous Approaches Failed

### The Core Problem

Previous training approaches failed because:

1. **Training**: Each sample used conditioning extracted from its own audio
2. **Inference**: You use DIFFERENT audio → DIFFERENT conditioning
3. **Result**: Patterns were learned but tied to training-specific embeddings

```
TRAINING (Old Approach):
  Sample 1: conditioning_A + text_A → codes_A (with patterns)
  Sample 2: conditioning_B + text_B → codes_B (with patterns)
  ...

INFERENCE:
  New audio: conditioning_X (never seen!) + new_text → codes WITHOUT patterns!
```

### The Real Issue

The GPT model learns: "When I see THIS conditioning + THIS text → produce THESE codes"

But at inference, you provide a completely different conditioning vector. The model doesn't know what patterns to apply!

## The Solution: Pattern Embeddings

Instead of hoping the model implicitly learns patterns from codes, we provide an **explicit, learnable "pattern trigger"** - a set of learned tokens that encode "speak with THIS person's patterns".

### How It Works

```
TRAINING (New Approach):
  Sample 1: [PATTERN_EMB] + conditioning + text_A → codes_A (with patterns)
  Sample 2: [PATTERN_EMB] + conditioning + text_B → codes_B (with patterns)
  ...
  
INFERENCE:
  New audio: [PATTERN_EMB] + conditioning + new_text → codes WITH patterns!
```

The **same pattern embedding** is used in training AND inference, so the model recognizes the "trigger" and produces patterns!

## Quick Start

### Step 1: Prepare Audio Files

```bash
mkdir -p training/ozzy/dataset/audio/
# Copy audio files with distinctive speaking patterns
cp /path/to/ozzy_recordings/*.wav training/ozzy/dataset/audio/
```

**Audio Guidelines:**
- Include clips that showcase the patterns you want (stutters, pauses, etc.)
- Duration: 1-15 seconds per clip
- Quality: Clean recordings, minimal background noise
- Quantity: 30-100 clips recommended

### Step 2: Transcribe Audio

```bash
python tools/transcribe_dataset.py --speaker ozzy --whisper-model large-v3
```

This creates verbatim transcripts that include:
- Filler words: `uh`, `um`, `er`
- Pauses: `[PAUSE]`, `...`
- Stutters/repetitions

### Step 3: Extract Pattern Conditioning (Optional)

If you want global conditioning (from v2 approach):

```bash
python tools/extract_pattern_conditioning.py --speaker ozzy
```

### Step 4: Prepare Dataset with Pattern Features

```bash
python tools/prepare_pattern_dataset_v3.py --speaker ozzy
```

This extracts **pattern features** from each audio:
- Pause positions and durations
- Filler word locations
- Stutter/repetition positions
- Speech rate variations

### Step 5: Train Pattern Embeddings

```bash
python tools/train_pattern_embeddings.py \
    --speaker ozzy \
    --epochs 40 \
    --pattern-tokens 8 \
    --lora-rank 32
```

This trains:
- **Pattern Embedding**: Learnable tokens that encode patterns
- **LoRA adapters** (optional): For voice-specific adaptation
- Uses **pattern-aware loss** that explicitly rewards pause reproduction

### Step 6: Generate with Patterns

```bash
python tools/infer_with_patterns.py \
    --speaker ozzy \
    --text "Life finds a way... you know what I mean?" \
    --output output.wav
```

The patterns will appear because the trained pattern embedding is injected!

## Using the Complete Pipeline Script

For convenience, use the all-in-one pipeline:

```bash
python tools/train_patterns_pipeline.py --speaker ozzy
```

This handles everything: transcription, preparation, training, and testing.

## Technical Details

### Pattern Embedding Architecture

```python
PatternEmbedding:
  pattern_tokens: Parameter(num_tokens, hidden_dim)  # Learnable!
  pattern_proj: Linear(hidden_dim → model_dim)
  pattern_scale: Parameter(1)  # Learnable intensity
```

The pattern tokens are learned during training to encode "speak with this person's patterns".

### Injection Modes

**add** (default): Add pattern embedding to first N tokens of conditioning
```
conditioning = conditioning + pattern_embedding  # at positions 0:N
```

**prepend**: Prepend pattern tokens before conditioning
```
conditioning = [pattern_embedding, conditioning]
```

**replace_first**: Replace first N tokens with pattern tokens
```
conditioning[:N] = pattern_embedding
```

### Pattern-Aware Loss

Standard CE loss treats all code predictions equally. Pattern-aware loss adds:

1. **Pause Loss**: Higher weight at known pause positions
2. **Rate Loss**: Encourages matching silence density patterns
3. **CE Loss**: Standard cross-entropy (weighted down relatively)

```python
total_loss = ce_weight * ce_loss + pause_weight * pause_loss + rate_weight * rate_loss
```

## Configuration Options

### Pattern Embedding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pattern-tokens` | 8 | Number of learnable pattern tokens |
| `--pattern-lr` | 1e-3 | Learning rate for pattern embedding |
| `--injection-mode` | add | How to inject (add/prepend/replace_first) |

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 40 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--lora-rank` | 32 | LoRA rank (higher = more capacity) |
| `--pause-weight` | 2.0 | Weight for pause loss |
| `--no-lora` | false | Train only pattern embedding |

### Inference Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pattern-scale` | 1.0 | Scale pattern embedding strength |
| `--injection-mode` | add | Must match training mode |

## Troubleshooting

### "Patterns still don't appear"

1. **Check pattern embedding was used**: Verify inference uses `infer_with_patterns.py`
2. **Check injection mode**: Must match between training and inference
3. **Increase pattern tokens**: Try `--pattern-tokens 16` or `32`
4. **Train longer**: Pattern learning takes more epochs than voice quality
5. **Check pattern scale**: Try `--pattern-scale 1.5` at inference

### "Audio quality degraded"

1. **Lower LoRA rank**: Try `--lora-rank 16` instead of 32
2. **Use only pattern embedding**: Try `--no-lora` to train without LoRA
3. **Check reference audio**: Use high-quality reference for voice timbre

### "Training loss doesn't decrease"

1. **Lower learning rate**: Try `--learning-rate 1e-4`
2. **Check dataset**: Ensure pattern features were extracted correctly
3. **Verify transcripts**: Bad transcripts = bad pattern detection

## Comparison: Old vs New Approach

| Aspect | Old (v2) | New (Pattern Embedding) |
|--------|----------|------------------------|
| Conditioning | Global (same for all) | Injected pattern tokens |
| Pattern transfer | Inconsistent | Reliable |
| What's trained | LoRA only | Pattern embedding + LoRA |
| Loss function | Standard CE | Pattern-aware |
| Inference | Regular + conditioning | Pattern embedding injection |

## API Usage

### Python API

```python
from indextts.infer_v2 import IndexTTS2
from indextts.pattern_embeddings import PatternEmbedding
from tools.infer_with_patterns import pattern_aware_inference

# Load model with LoRA
tts = IndexTTS2(lora_path="training/ozzy/pattern_training/best_checkpoint/lora")

# Load pattern embedding
pattern_emb = PatternEmbedding.load(
    "training/ozzy/pattern_training/best_checkpoint/pattern_embedding.pt",
    device=tts.device
)

# Inference with patterns
pattern_aware_inference(
    tts=tts,
    pattern_embedding=pattern_emb,
    text="Your text here",
    output_path="output.wav",
    audio_prompt="reference.wav",  # For voice timbre
)
```

### Integration with Existing Code

The pattern embedding can be integrated into your existing inference pipeline:

```python
# Get conditioning from reference audio
gpt_conditioning = extract_conditioning(reference_audio)

# INJECT PATTERN EMBEDDING (the key step!)
pattern_conditioned = pattern_emb.get_injection_embedding(
    gpt_conditioning,
    injection_mode="add"
)

# Use pattern_conditioned instead of gpt_conditioning for generation
```

## Directory Structure

```
training/
  ozzy/
    dataset/
      audio/                          # Your audio files
      transcripts_verbatim.csv        # Whisper transcripts
      processed_v3/                   # Pattern dataset
        train_manifest.jsonl
        val_manifest.jsonl
        features/
          sample_text_ids.npy
          sample_codes.npy
          sample_pattern_features.npy  # Pattern analysis!
          GLOBAL_condition.npy         # (optional)
    pattern_training/                 # Training output
      best_checkpoint/
        pattern_embedding.pt          # THE KEY FILE!
        lora/                         # LoRA adapters
      final_checkpoint/
      logs/
    embeddings/
      speaker_embeddings.pt           # For S2Mel voice timbre
```

## How It All Fits Together

1. **Pattern Embedding**: Learned tokens that trigger pattern generation
2. **LoRA Adapters**: Fine-tune GPT for this speaker's voice
3. **Pattern-Aware Loss**: Explicitly rewards pause/pattern reproduction
4. **Injection at Inference**: Same embedding triggers patterns in new text

The magic is that the pattern embedding is:
- **Trained** alongside the codes (so it learns to trigger patterns)
- **Used at inference** (so the model recognizes it and produces patterns)

This is similar to how textual inversion works in image generation - a learned trigger for specific behaviors!