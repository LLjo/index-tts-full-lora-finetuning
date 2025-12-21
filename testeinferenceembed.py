from indextts.speaker_embeddings import SpeakerEmbeddingStore
from indextts.infer_v2 import IndexTTS2
audio_path = "/var/www/pingpong-project/index-tts/my_voice_project/audio/1_ossy_mixdown_Track 2_(Vocals)_slice_0.wav"

audio_path = "/var/www/pingpong-project/index-tts/dataset/goldblum/audio/goldblum_mixdown_Track 1_slice_4.wav"
text = "is the fact that Black Sabbath wasn't a band that was created by some big boss mogul guy, that was four guys that were going, let's have a go, have a dream, and it came."
text = "Haha what are you saying? Isn't that some kind of trick question? I will need to think about this for a while..."



# tts = IndexTTS2(gpt_checkpoint='trained_checkpoints/goldblum/best_model.pth')
# # Or for LoRA:
# # tts = IndexTTS2(lora_path='trained_lora/my_voice_v2/final_checkpoint')
# # Or for base model:
# # tts = IndexTTS2()

# # Create embedding store
# store = SpeakerEmbeddingStore("./speaker_embeddings")

# # Option A: Extract from a single audio file
# store.extract_and_save(
#     tts, 
#     audio_path, 
#     speaker_name="goldblum",
#     metadata={"description": "My voice from training data"}
# )


tts = IndexTTS2(
    gpt_checkpoint='trained_checkpoints/goldblum/best_model.pth',
    use_accel=True, 
)

# Load stored speaker embeddings
store = SpeakerEmbeddingStore("./speaker_embeddings")
embeddings = store.load("goldblum", device=tts.device)

# Generate speech WITHOUT audio prompt!

tts.infer(
    text=text,
    output_path="output_promptless_goldblum_1.wav",
    speaker_embeddings=embeddings,  # Use stored embeddings instead of audio!
    # spk_audio_prompt=audio_path,
    # use_emo_text=True,
    # emo_text="really giggly happy"
)
tts.infer(
    text=text,
    output_path="output_promptless_goldblum_2.wav",
    speaker_embeddings=embeddings,  # Use stored embeddings instead of audio!
    # spk_audio_prompt=audio_path,
    # use_emo_text=True,
    # emo_text="really giggly happy"
)
tts.infer(
    text=text,
    output_path="output_promptless_goldblum_3.wav",
    speaker_embeddings=embeddings,  # Use stored embeddings instead of audio!
    # spk_audio_prompt=audio_path,
    # use_emo_text=True,
    # emo_text="really giggly happy"
)
