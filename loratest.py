# from indextts.infer_v2 import IndexTTS2

# tts = IndexTTS2(gpt_checkpoint='trained_checkpoints/my_voice/best_model.pth')
# tts.infer(
#     spk_audio_prompt="/var/www/pingpong-project/index-tts/my_voice_project/audio/2_ossy_mixdown_Track 3_(Vocals)_slice_0.wav",
#     text="is the fact that Black Sabbath wasn't a band that was created by some big boss mogul guy, that was four guys that were going, let's have a go, have a dream, and it came.",
#     output_path="output3.wav"
# )

from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2()
tts.infer(
    spk_audio_prompt="/var/www/index-tts-full-lora-finetuning/training/ozzy/dataset/audio/1_ossy_mixdown_Track 2_(Vocals)_slice_1.wav",
    text="God let me be an actor. I would write on it. And then before I left, I would wipe it off because I still didn't want anybody to know. And then right after high school...",
    output_path="output3-normal.wav"
)