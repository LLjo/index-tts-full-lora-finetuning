from indextts.infer_v2 import IndexTTS2
from indextts.speaker_embeddings import SpeakerEmbeddingStore

tts = IndexTTS2(lora_path="trained_lora/my_voice/final_checkpoint")
store = SpeakerEmbeddingStore(tts)
embeddings = store.load_embeddings("trained_lora/my_voice/speaker_embeddings.pt")

# Use stored embeddings - NO reference audio!
tts.infer(
    spk_audio_prompt=None,
    text="Hello world",
    output_path="output.wav",
    speaker_embeddings=embeddings
)