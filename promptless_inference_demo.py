"""
Promptless Inference Demo for IndexTTS2

This script demonstrates:
"""

# Fix for missing temp directory - must be done BEFORE any other imports
import os
import tempfile

# Create temp directory if it doesn't exist
temp_dirs = ['/tmp', '/var/tmp', os.path.join(os.getcwd(), '.tmp')]
for temp_dir in temp_dirs:
    try:
        os.makedirs(temp_dir, exist_ok=True)
        # Test if we can write to it
        test_file = os.path.join(temp_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        os.environ['TMPDIR'] = temp_dir
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir
        print(f"Using temp directory: {temp_dir}")
        break
    except (OSError, PermissionError):
        continue

"""
1. WHY LoRA/finetuning might not show visible differences
2. HOW to use speaker embeddings for promptless inference

THE CORE ISSUE WITH LORA/FINETUNING:
=====================================
IndexTTS2 has a two-stage architecture:

Stage 1: GPT generates semantic tokens from text + speaker conditioning
  - This is what LoRA/finetuning modifies
  - Controls: prosody, rhythm, pacing, semantic content
  
Stage 2: S2Mel + BigVGAN converts semantic tokens to audio
  - Uses reference audio's mel spectrogram, style vector, and prompt condition
  - This determines the ACTUAL VOICE QUALITY/TIMBRE
  - NOT affected by LoRA/finetuning!

When you provide a reference audio (spk_audio_prompt), Stage 2 uses that audio's
characteristics to reconstruct the voice. This OVERWRITES any voice characteristics
the GPT might have learned.

SOLUTION: Store speaker embeddings from your target voice and use them directly.
This bypasses the need for a reference audio file during inference.
"""

from indextts.infer_v2 import IndexTTS2
from indextts.speaker_embeddings import SpeakerEmbeddingStore, extract_multiple_utterances

# ============================================================================
# STEP 1: Extract and store speaker embeddings from your voice samples
# ============================================================================

def setup_speaker_embeddings():
    """One-time setup to extract and store speaker embeddings."""
    
    print("\n" + "="*60)
    print("STEP 1: Extract Speaker Embeddings")
    print("="*60)
    
    # Initialize the TTS model (with or without LoRA/finetuning)
    # If you finetuned the GPT, load it here:
    tts = IndexTTS2(gpt_checkpoint='trained_checkpoints/my_voice/best_model.pth')
    # Or for LoRA:
    # tts = IndexTTS2(lora_path='trained_lora/my_voice_v2/final_checkpoint')
    # Or for base model:
    # tts = IndexTTS2()
    
    # Create embedding store
    store = SpeakerEmbeddingStore("./speaker_embeddings")
    
    # Option A: Extract from a single audio file
    audio_path = "/var/www/pingpong-project/index-tts/my_voice_project/audio/2_ossy_mixdown_Track 3_(Vocals)_slice_0.wav"
    store.extract_and_save(
        tts, 
        audio_path, 
        speaker_name="my_voice",
        metadata={"description": "My voice from training data"}
    )
    
    # Option B: Extract and average from multiple files for better representation
    # audio_files = [
    #     "my_voice_project/audio/sample1.wav",
    #     "my_voice_project/audio/sample2.wav",
    #     "my_voice_project/audio/sample3.wav",
    # ]
    # avg_embeddings = extract_multiple_utterances(tts, audio_files)
    # store.save(avg_embeddings, "my_voice_averaged")
    
    print("\nSpeaker embeddings saved!")
    print(f"Available speakers: {store.list_speakers()}")
    
    return tts

# ============================================================================
# STEP 2: Use stored embeddings for promptless inference
# ============================================================================

def promptless_inference(tts=None):
    """Generate speech without needing a reference audio file."""
    
    print("\n" + "="*60)
    print("STEP 2: Promptless Inference")
    print("="*60)
    
    if tts is None:
        # Load model (same as you trained with finetuning/LoRA)
        tts = IndexTTS2(gpt_checkpoint='trained_checkpoints/my_voice/best_model.pth')
    
    # Load stored speaker embeddings
    store = SpeakerEmbeddingStore("./speaker_embeddings")
    embeddings = store.load("my_voice", device=tts.device)
    
    # Generate speech WITHOUT audio prompt!
    text = "is the fact that Black Sabbath wasn't a band that was created by some big boss mogul guy, that was four guys that were going, let's have a go, have a dream, and it came."
    
    tts.infer(
        text=text,
        output_path="output_promptless.wav",
        speaker_embeddings=embeddings  # Use stored embeddings instead of audio!
    )
    
    print("\nGenerated: output_promptless.wav")
    print("This used stored speaker embeddings - no audio file needed!")

# ============================================================================
# STEP 3: Compare the approaches
# ============================================================================

def compare_approaches():
    """Compare audio-prompted vs promptless inference."""
    
    print("\n" + "="*60)
    print("COMPARISON: Audio Prompt vs Stored Embeddings")
    print("="*60)
    
    tts = IndexTTS2(gpt_checkpoint='trained_checkpoints/my_voice/best_model.pth')
    
    text = "Hello, this is a test of the text to speech system."
    reference_audio = "/var/www/pingpong-project/index-tts/my_voice_project/audio/2_ossy_mixdown_Track 3_(Vocals)_slice_0.wav"
    
    # Method 1: Traditional with audio prompt
    print("\n1. Generating with audio prompt...")
    tts.infer(
        spk_audio_prompt=reference_audio,
        text=text,
        output_path="output_with_prompt.wav"
    )
    
    # Method 2: Promptless with stored embeddings
    print("\n2. Generating with stored embeddings...")
    store = SpeakerEmbeddingStore("./speaker_embeddings")
    embeddings = store.load("my_voice", device=tts.device)
    
    tts.infer(
        text=text,
        output_path="output_with_embeddings.wav",
        speaker_embeddings=embeddings
    )
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print("1. output_with_prompt.wav - Uses audio file for voice characteristics")
    print("2. output_with_embeddings.wav - Uses stored embeddings")
    print("\nBoth should sound very similar if embeddings were extracted from the same audio!")
    print("\nThe advantage of embeddings:")
    print("  - No audio file needed at runtime")
    print("  - Faster (skips audio processing)")
    print("  - Can average multiple samples for more stable voice")

# ============================================================================
# WHY LORA/FINETUNING SHOWS MINIMAL DIFFERENCE
# ============================================================================

def explain_the_issue():
    """Explain why LoRA/finetuning appears to have no effect."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  WHY YOUR LORA/FINETUNING ISN'T SHOWING CHANGES              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  IndexTTS2 Pipeline:                                                          ║
║                                                                              ║
║  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                  ║
║  │    Text      │ --> │     GPT      │ --> │   S2Mel +    │ --> Audio        ║
║  │  + Speaker   │     │  (Semantic   │     │   BigVGAN    │                  ║
║  │    Cond      │     │   Tokens)    │     │  (Voice!)    │                  ║
║  └──────────────┘     └──────────────┘     └──────────────┘                  ║
║        ^                     ^                    ^                          ║
║        |                     |                    |                          ║
║    Reference            LoRA/Finetune         Uses ref_mel,                 ║
║    Audio                affects THIS          style, prompt                  ║
║                                               from Reference!                ║
║                                                                              ║
║  THE PROBLEM:                                                                ║
║  Even if GPT learns your voice patterns through LoRA/finetuning,            ║
║  the S2Mel stage uses the reference audio's mel spectrogram and              ║
║  style vector to reconstruct the final audio waveform.                       ║
║                                                                              ║
║  This means the reference audio "overwrites" learned voice characteristics!  ║
║                                                                              ║
║  WHAT LORA/FINETUNING ACTUALLY AFFECTS:                                      ║
║  - Prosody and rhythm patterns                                               ║
║  - Speaking style and pacing                                                 ║
║  - Word emphasis patterns                                                    ║
║  - Semantic token selection                                                  ║
║                                                                              ║
║  WHAT IT DOESN'T AFFECT:                                                     ║
║  - Voice timbre/quality (determined by reference audio)                      ║
║  - Pitch characteristics (from reference)                                    ║
║  - Speaker identity (from reference)                                         ║
║                                                                              ║
║  SOLUTION FOR VOICE CLONING:                                                 ║
║  1. Use speaker_embeddings from your target voice                            ║
║  2. Store embeddings from good quality samples of your voice                 ║
║  3. Use those embeddings for inference (promptless mode)                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys
    
    explain_the_issue()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "setup":
            setup_speaker_embeddings()
        elif sys.argv[1] == "infer":
            promptless_inference()
        elif sys.argv[1] == "compare":
            compare_approaches()
    else:
        print("\nUsage:")
        print("  python promptless_inference_demo.py setup   - Extract and save speaker embeddings")
        print("  python promptless_inference_demo.py infer   - Generate audio with stored embeddings")
        print("  python promptless_inference_demo.py compare - Compare both methods")