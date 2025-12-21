"""
LoRA Comparison Test Script

This script generates audio with and without LoRA to help you hear the difference.
Use a DIFFERENT speaker's voice as reference to see how much the LoRA adapts the output.
"""
from indextts.infer_v2 import IndexTTS2

# Test text
text = "is the fact that Black Sabbath wasn't a band that was created by some big boss mogul guy, that was four guys that were going, let's have a go, have a dream, and it came."

# Use a DIFFERENT speaker's voice as reference (not from your training set)
# This helps demonstrate that the LoRA adapts the output toward your trained voice
reference_audio = "/var/www/pingpong-project/index-tts/my_voice_project/audio/1_ossy_mixdown_Track 2_(Vocals)_slice_0.wav"  # Change this to any reference audio

# Generate WITHOUT LoRA (base model)
print("\n=== Generating with BASE MODEL (no LoRA) ===")
tts_base = IndexTTS2()
tts_base.infer(
    spk_audio_prompt=reference_audio,
    text=text,
    output_path="output_base_model.wav"
)
print("Saved: output_base_model.wav")

# Clean up base model to free GPU memory
del tts_base
import torch
torch.cuda.empty_cache()

# Generate WITH LoRA
print("\n=== Generating with LoRA MODEL ===")
tts_lora = IndexTTS2(lora_path='trained_lora/my_voice_v2/final_checkpoint')
tts_lora.infer(
    spk_audio_prompt=reference_audio,
    text=text,
    output_path="output_lora_model.wav"
)
print("Saved: output_lora_model.wav")

print("\n=== COMPARISON COMPLETE ===")
print("Compare these two files to hear the LoRA effect:")
print("  1. output_base_model.wav - Base model output")
print("  2. output_lora_model.wav - LoRA fine-tuned output")
print("\nIf the LoRA is working, output_lora_model.wav should sound more")
print("like your trained voice, even though we used a different reference audio!")