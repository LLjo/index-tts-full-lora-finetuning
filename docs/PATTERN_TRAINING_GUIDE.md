# Training Speaking Patterns in IndexTTS2 (v2 - Global Conditioning)

This guide explains how to train models that capture unique speaking patterns like:
- **Pauses and hesitations** ("well... let me think...")
- **Filler words** ("uh", "um", "you know")
- **Stutters and repetitions** (like Ozzy's distinctive speech)
- **Rhythm and pacing** (fast/slow, choppy/smooth)

## The Key Insight: Why Previous Training Failed

**The Problem:**
Previous approaches extracted conditioning from each audio file individually. 
At inference, you used DIFFERENT audio → DIFFERENT conditioning.
The patterns were learned but tied to specific conditioning vectors the model never sees at inference!

**The Solution: Global Conditioning**
Use the SAME conditioning for ALL training samples AND inference. This ensures patterns transfer because the model sees consistent conditioning.

## Quick Start (Recommended)

```bash
# 1. Put your audio files in:
#    training/ozzy/dataset/audio/

# 2. Run the complete pipeline:
python tools/train_speaking_patterns.py --speaker ozzy

# That's it! The script handles everything:
# - Transcription with Whisper
# - Global conditioning extraction
# - Dataset preparation
# - Training
# - Testing
```

## Understanding the Architecture

IndexTTS2 has a **two-stage pipeline**:

```
                    STAGE 1: GPT (Trainable)
Text ──────────────────────────────────────────► Semantic Tokens
       + Speaker Conditioning                      (prosody, timing,
         (from reference audio)                     pauses encoded here)
                                                          │
                                                          ▼
                    STAGE 2: S2Mel (Not Trained)
Semantic Tokens ──────────────────────────────────► Audio Waveform
       + Reference Audio Features                   (voice timbre from
         (ref_mel, style, prompt_condition)          reference audio)
```

### The Critical Flow

1. **Training (OLD - didn't work):**
   - Sample 1: conditioning_1 + text_1 → codes_1
   - Sample 2: conditioning_2 + text_2 → codes_2
   - ...
   - At inference: conditioning_X (new!) + text → patterns DON'T appear

2. **Training (NEW - works!):**
   - Sample 1: GLOBAL_conditioning + text_1 → codes_1
   - Sample 2: GLOBAL_conditioning + text_2 → codes_2
   - ...
   - At inference: GLOBAL_conditioning (same!) + text → patterns APPEAR!

## Step-by-Step Guide

### Step 1: Prepare Audio Files

Place your audio files in the expected location:

```bash
mkdir -p training/ozzy/dataset/audio/
# Copy your audio files (wav, mp3, flac, etc.)
cp /path/to/audio/*.wav training/ozzy/dataset/audio/
```

**Audio guidelines:**
- Duration: 1-15 seconds per clip (shorter is OK, longer will be trimmed)
- Quality: Clean recordings, minimal background noise
- Content: Include examples of the patterns you want to capture!
  - If Ozzy stutters, include clips with stutters
  - If they have distinctive pauses, include those

### Step 2: Run Complete Training Pipeline

The simplest way:

```bash
python tools/train_speaking_patterns.py --speaker ozzy
```

This automatically:
1. Transcribes audio with Whisper (detecting pauses/fillers)
2. Extracts GLOBAL conditioning from all audio
3. Prepares dataset with global conditioning
4. Trains LoRA adapters
5. Extracts speaker embeddings
6. Generates test audio

### Step 3: Test the Results

```bash
# Generate with pattern conditioning
python tools/infer_pattern.py \
    --speaker ozzy \
    --text "Life finds a way" \
    --output output.wav
```

## Manual Step-by-Step (Advanced)

If you want more control:

### 1. Transcribe Audio

```bash
python tools/transcribe_dataset.py --speaker ozzy --whisper-model large-v3
```

### 2. Extract Global Conditioning

```bash
python tools/extract_pattern_conditioning.py --speaker ozzy
```

This creates: `training/ozzy/pattern_conditioning.pt`

### 3. Prepare Dataset with Global Conditioning

```bash
python tools/prepare_pattern_dataset_v2.py \
    --speaker ozzy \
    --pattern-conditioning training/ozzy/pattern_conditioning.pt
```

**Key difference from v1:** ALL samples now use the SAME conditioning!

### 4. Train LoRA Adapters

```bash
python tools/train_gpt_lora.py \
    --train-manifest training/ozzy/dataset/processed_v2/train_manifest.jsonl \
    --val-manifest training/ozzy/dataset/processed_v2/val_manifest.jsonl \
    --output-dir training/ozzy/lora \
    --lora-rank 32 \
    --lora-alpha 64 \
    --epochs 30 \
    --learning-rate 5e-4
```

### 5. Extract Speaker Embeddings

```bash
python tools/extract_embeddings.py --speaker ozzy
```

### 6. Test with Pattern Conditioning

```bash
python tools/infer_pattern.py \
    --speaker ozzy \
    --text "Your test text here"
```

## Using in Code

```python
from indextts.infer_v2 import IndexTTS2
from indextts.pattern_conditioning import PatternConditioningStore
from indextts.speaker_embeddings import SpeakerEmbeddingStore

# Load model with LoRA
tts = IndexTTS2(lora_path="training/ozzy/lora/final_checkpoint")

# Load pattern conditioning (for GPT stage)
pattern_cond = PatternConditioningStore.load("training/ozzy/pattern_conditioning.pt")

# Load speaker embeddings (for S2Mel stage)
store = SpeakerEmbeddingStore(tts)
speaker_emb = store.load_embeddings("training/ozzy/embeddings/speaker_embeddings.pt")

# Inference with patterns
tts.infer(
    text="Life finds a way",
    speaker_embeddings=speaker_emb,
    output_path="output.wav"
)
```

## Why Global Conditioning Works

### The Technical Explanation

The GPT model learns a mapping:
```
conditioning + text_tokens → semantic_codes
```

When each training sample has DIFFERENT conditioning, the model learns:
```
"When I see conditioning_A, generate codes_A"
"When I see conditioning_B, generate codes_B"
...
```

But at inference, you provide conditioning_X (from your reference audio), which the model has NEVER seen. It doesn't know what patterns to apply!

With GLOBAL conditioning:
```
"When I see GLOBAL_conditioning, generate varied patterns based on text"
```

At inference, you use the SAME GLOBAL_conditioning → the model knows exactly what to do!

### The Analogy

Think of it like teaching someone to speak in a specific style:

**Old approach:** "Here's 100 different examples, each with a different prompt"
- Student learns each example independently
- When given a NEW prompt, they don't know what style to use

**New approach:** "Here's 100 examples, all with the SAME prompt: 'speak like Ozzy'"
- Student learns: when told to "speak like Ozzy", add stutters/pauses
- When given the same prompt at test time, they know the style!

## Training Tips

### For Better Pattern Learning

1. **Higher LoRA rank:** Use `--lora-rank 32` or `64` (default is 16)
   - More capacity to learn pattern nuances
   
2. **More epochs:** Use `--epochs 30` or more
   - Patterns take longer to learn than voice quality
   
3. **Higher learning rate:** Use `--learning-rate 5e-4`
   - LoRA can handle higher LR than full fine-tuning

4. **Verbatim transcriptions:** Include markers for patterns
   ```
   "Well [PAUSE] you know [UH] life... life finds a way"
   ```

5. **More diverse training data:**
   - Include examples with varying amounts of patterns
   - This helps the model learn when to add patterns

### Dataset Size Guidelines

- **Minimum:** 10-20 clips (basic pattern learning)
- **Recommended:** 50-100 clips (good pattern capture)
- **Ideal:** 200+ clips (nuanced pattern reproduction)

### Troubleshooting

#### "I don't hear any patterns"

1. **Check you're using pattern-conditioned inference:**
   ```bash
   python tools/infer_pattern.py --speaker ozzy --text "test"
   ```
   NOT the regular `infer.py` or just using reference audio!

2. **Verify global conditioning was used in training:**
   ```bash
   # Check if dataset has global conditioning
   cat training/ozzy/dataset/processed_v2/dataset_info.json
   # Should show "version": "v2_pattern_conditioned"
   ```

3. **Check training was successful:**
   ```bash
   python tools/diagnose_training.py --lora-path training/ozzy/lora/final_checkpoint
   ```

#### "Patterns are inconsistent"

1. **Train longer:** Try 40-50 epochs
2. **Use higher rank:** Try `--lora-rank 64`
3. **More training data:** Add more clips with patterns

#### "Model outputs garbage"

1. **Lower learning rate:** Try `1e-4` instead of `5e-4`
2. **Check audio quality:** Ensure clean recordings
3. **Verify transcriptions:** Bad transcripts = bad training

## Directory Structure

```
training/
  ozzy/
    dataset/
      audio/                        # Your audio files
      transcripts_verbatim.csv      # Auto-generated transcriptions
      transcription_stats.json      # Stats about detected patterns
      processed_v2/                 # Pattern-conditioned dataset
        train_manifest.jsonl
        val_manifest.jsonl
        dataset_info.json           # Shows v2 with global conditioning
        features/
          GLOBAL_condition.npy      # Same for ALL samples!
          GLOBAL_emo_vec.npy        # Same for ALL samples!
          sample1_text_ids.npy
          sample1_codes.npy
          sample2_text_ids.npy
          ...
    pattern_conditioning.pt         # Global conditioning for inference
    embeddings/
      speaker_embeddings.pt         # For S2Mel stage
    lora/
      final_checkpoint/             # Trained LoRA weights
        adapter_config.json
        adapter_model.bin
    test_outputs/                   # Test audio files
```

## Comparison: v1 vs v2 Training

| Aspect | v1 (Old) | v2 (New - Global Conditioning) |
|--------|----------|-------------------------------|
| Conditioning | Per-sample | Global (same for all) |
| Pattern transfer | ❌ Doesn't work | ✅ Works! |
| Why | Different embedding at inference | Same embedding at inference |
| Script | `prepare_pattern_dataset.py` | `prepare_pattern_dataset_v2.py` |
| Inference | Regular inference | `infer_pattern.py` |

## Technical Details

### What Gets Stored in pattern_conditioning.pt

```python
{
    'spk_cond_emb': tensor,       # Raw W2V-BERT features (1, seq_len, 1024)
    'gpt_conditioning': tensor,   # Perceiver output (1, 32, model_dim) 
    'emo_vec': tensor,            # Emotion vector (1, model_dim)
    '_metadata': {
        'num_samples': int,
        'method': str,            # 'average', 'longest', 'first'
        'audio_paths': list,
    }
}
```

### What Gets Stored in speaker_embeddings.pt

```python
{
    'spk_cond_emb': tensor,       # For GPT conditioning
    'style': tensor,              # CAMPPlus style (for S2Mel)
    'prompt_condition': tensor,   # For S2Mel
    'ref_mel': tensor,            # Reference mel (for S2Mel)
    'emo_cond_emb': tensor,       # Emotion conditioning
}
```

### The Two Embeddings Serve Different Purposes

1. **pattern_conditioning.pt:** Controls GPT generation (semantic tokens with patterns)
2. **speaker_embeddings.pt:** Controls S2Mel (voice quality/timbre)

Both are needed for full pattern+voice reproduction!