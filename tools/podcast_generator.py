#!/usr/bin/env python3
"""
Podcast Generator for IndexTTS2

This script generates a podcast dialogue between two trained voices (goldblum and ozzy)
with optimized model loading for fast switching and streaming support.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Union
import threading
import queue

import torch
import torchaudio
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indextts.infer_v2 import IndexTTS2
from indextts.pattern_embeddings import PatternEmbedding
from indextts.speaker_embeddings import SpeakerEmbeddingStore


class PodcastGenerator:
    """
    A reusable class for generating podcast dialogues between multiple trained voices.
    
    Features:
    - Optimized model loading with fast switching between speakers
    - Streaming support for real-time generation
    - Audio stitching for complete podcast creation
    """
    
    def __init__(
        self,
        use_fp16: bool = False,
        use_accel: bool = False,
        use_torch_compile: bool = False,
        use_cuda_kernel: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the podcast generator.
        
        Args:
            use_fp16: Use FP16 precision for faster inference
            use_accel: Use acceleration engine for GPT2
            use_torch_compile: Use torch.compile for optimization
            use_cuda_kernel: Use BigVGAN custom CUDA kernel
            device: Device to use (auto-detected if not specified)
        """
        self.use_fp16 = use_fp16
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile
        self.use_cuda_kernel = use_cuda_kernel
        self.device = device
        
        # Initialize shared model (loaded once)
        self.tts = None
        self.speaker_configs = {}
        self.loaded_speakers = {}
        self.character_defaults = {}  # Store default generation settings per character
        
        # Audio generation queue for streaming
        self.audio_queue = queue.Queue()
        self.generation_thread = None
        
    def load_shared_model(self):
        """Load the shared TTS model once for all speakers."""
        if self.tts is not None:
            return  # Already loaded
            
        print("Loading shared TTS model...")
        self.tts = IndexTTS2(
            lora_path=None,  # We'll load LoRA per speaker
            use_fp16=self.use_fp16,
            use_accel=self.use_accel,
            use_torch_compile=self.use_torch_compile,
            use_cuda_kernel=self.use_cuda_kernel,
            device=self.device,
        )
        
        # Store base model state for clean LoRA switching
        print("Storing base GPT model state...")
        self.base_gpt_state = {k: v.clone().cpu() for k, v in self.tts.gpt.named_parameters()}
        
        print("✓ Shared model loaded")
        
    def load_characters(self, characters_path: Path):
        """
        Load character configurations from a JSON file.
        
        Args:
            characters_path: Path to JSON file with character configurations
        """
        with open(characters_path, 'r') as f:
            characters = json.load(f)
        
        for char_name, char_config in characters.items():
            # Register the speaker
            self.register_speaker(
                name=char_name,
                pattern_embedding_path=Path(char_config["pattern_embedding_path"]) if "pattern_embedding_path" in char_config else None,
                lora_path=Path(char_config["lora_path"]) if "lora_path" in char_config else None,
                speaker_embeddings_path=Path(char_config["speaker_embeddings_path"]) if "speaker_embeddings_path" in char_config else None,
                audio_prompt_path=Path(char_config["audio_prompt_path"]) if "audio_prompt_path" in char_config else None,
            )
            
            # Store default generation settings for this character
            self.character_defaults[char_name] = {
                "pattern_scale": char_config.get("pattern_scale", 0.02),
                "temperature": char_config.get("temperature", 0.8),
                "top_p": char_config.get("top_p", 0.8),
                "top_k": char_config.get("top_k", 30),
                "emotion_alpha": char_config.get("emotion_alpha", 1.0),
                "use_emo_text": char_config.get("use_emo_text", False),
                "emo_text": char_config.get("emo_text", None),
                "emotion_vector": char_config.get("emotion_vector", None),
                "injection_mode": char_config.get("injection_mode", "add"),
            }
            
        print(f"✓ Loaded {len(characters)} characters from {characters_path}")
    
    def register_speaker(
        self,
        name: str,
        pattern_embedding_path: Optional[Path] = None,
        lora_path: Optional[Path] = None,
        speaker_embeddings_path: Optional[Path] = None,
        audio_prompt_path: Optional[Path] = None,
    ):
        """
        Register a speaker configuration.
        
        Args:
            name: Speaker name (e.g., "goldblum", "ozzy")
            pattern_embedding_path: Path to pattern embedding
            lora_path: Path to LoRA checkpoint
            speaker_embeddings_path: Path to speaker embeddings
            audio_prompt_path: Path to reference audio (alternative to embeddings)
        """
        # Auto-detect paths if not provided
        speaker_dir = PROJECT_ROOT / "training" / name
        
        if pattern_embedding_path is None:
            candidates = [
                speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt",
                speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt",
            ]
            for c in candidates:
                if c.exists():
                    pattern_embedding_path = c
                    break
                    
        if lora_path is None:
            candidates = [
                speaker_dir / "pattern_training" / "final_checkpoint" / "lora",
                speaker_dir / "pattern_training" / "best_checkpoint" / "lora",
                speaker_dir / "lora" / "final_checkpoint",
            ]
            for c in candidates:
                if c.exists() and (c / "adapter_config.json").exists():
                    lora_path = c
                    break
                    
        if speaker_embeddings_path is None:
            speaker_embeddings_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
            if not speaker_embeddings_path.exists():
                speaker_embeddings_path = None
                
        # Store configuration
        self.speaker_configs[name] = {
            "pattern_embedding_path": pattern_embedding_path,
            "lora_path": lora_path,
            "speaker_embeddings_path": speaker_embeddings_path,
            "audio_prompt_path": audio_prompt_path,
        }
        
        print(f"✓ Registered speaker '{name}'")
        
    def load_speaker(self, name: str):
        """
        Load a speaker's models and embeddings.
        
        Args:
            name: Speaker name to load
        """
        if name in self.loaded_speakers:
            return  # Already loaded
            
        if name not in self.speaker_configs:
            raise ValueError(f"Speaker '{name}' not registered. Call register_speaker() first.")
            
        config = self.speaker_configs[name]
        
        # Ensure shared model is loaded
        self.load_shared_model()
        
        print(f"Loading speaker '{name}'...")
        
        # Load pattern embedding
        pattern_embedding = PatternEmbedding.load(
            config["pattern_embedding_path"], 
            device=self.tts.device
        )
        pattern_embedding.eval()
        
        # Load speaker embeddings or prepare audio prompt
        speaker_embeddings = None
        audio_prompt = None
        
        if config["audio_prompt_path"]:
            audio_prompt = str(config["audio_prompt_path"])
        elif config["speaker_embeddings_path"]:
            store = SpeakerEmbeddingStore(self.tts)
            speaker_embeddings = store.load_embeddings(config["speaker_embeddings_path"])
        else:
            raise ValueError(f"No voice reference found for speaker '{name}'")
            
        # Store loaded speaker data
        self.loaded_speakers[name] = {
            "pattern_embedding": pattern_embedding,
            "speaker_embeddings": speaker_embeddings,
            "audio_prompt": audio_prompt,
            "lora_path": config["lora_path"],
        }
        
        print(f"✓ Speaker '{name}' loaded")
        
    def switch_speaker(self, name: str):
        """
        Switch to a different speaker by reloading the model with their LoRA weights.
        
        Args:
            name: Speaker name to switch to
        """
        if name not in self.loaded_speakers:
            self.load_speaker(name)
            
        speaker_data = self.loaded_speakers[name]
        lora_path = speaker_data["lora_path"]
        
        if lora_path:
            print(f"Switching to speaker '{name}' LoRA...")
            
            # CRITICAL FIX: Restore base model weights before loading new LoRA
            # This prevents accumulated corruption from multiple LoRA merges
            print("  Restoring base GPT model state...")
            with torch.no_grad():
                for name_param, param in self.tts.gpt.named_parameters():
                    if name_param in self.base_gpt_state:
                        param.copy_(self.base_gpt_state[name_param].to(self.tts.device))
            
            # Clear CUDA cache to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load LoRA weights into the restored base model
            from indextts.utils.lora_utils import load_lora_checkpoint
            
            self.tts.gpt = load_lora_checkpoint(
                self.tts.gpt,
                str(lora_path),
                merge_weights=True,
                device=self.tts.device
            )
            
            # Ensure proper device and dtype
            self.tts.gpt = self.tts.gpt.to(self.tts.device)
            if self.use_fp16:
                self.tts.gpt.eval().half()
            else:
                self.tts.gpt.eval()
            
            print(f"✓ Switched to speaker '{name}'")
            
    def generate_speech(
        self,
        speaker: str,
        text: str,
        emotion_vector: Optional[List[float]] = None,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        emotion_audio: Optional[Path] = None,
        emotion_alpha: float = 1.0,
        pattern_scale: float = 0.02,
        injection_mode: str = "add",
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        stream: bool = False,
        verbose: bool = False,
    ) -> Union[Path, Generator[torch.Tensor, None, None]]:
        """
        Generate speech for a given speaker.
        
        Args:
            speaker: Speaker name
            text: Text to synthesize
            emotion_vector: Explicit emotion vector
            use_emo_text: Extract emotion from text
            emo_text: Custom text for emotion extraction
            emotion_audio: Reference audio for emotion
            emotion_alpha: Emotion mixing weight
            pattern_scale: Scale pattern embedding strength
            injection_mode: How to inject pattern embedding
            temperature: Generation temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            stream: Return streaming generator
            verbose: Print verbose output
            
        Returns:
            Path to saved audio file or generator of audio chunks
        """
        # Ensure speaker is loaded
        if speaker not in self.loaded_speakers:
            self.load_speaker(speaker)
            
        # Switch to speaker's LoRA
        self.switch_speaker(speaker)
        
        # Get speaker data
        speaker_data = self.loaded_speakers[speaker]
        pattern_embedding = speaker_data["pattern_embedding"]
        
        # Apply pattern scale
        if pattern_scale != 1.0:
            pattern_embedding.pattern_scale.data *= pattern_scale
            
        # Generate output path
        output_path = PROJECT_ROOT / "output" / f"podcast_{speaker}_{int(time.time())}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Import the pattern-aware inference function
        from tools.infer_with_patterns import pattern_aware_inference
        
        # Clear CUDA cache before generation to prevent degradation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate speech
        return pattern_aware_inference(
            tts=self.tts,
            pattern_embedding=pattern_embedding,
            text=text,
            output_path=output_path,
            audio_prompt=speaker_data["audio_prompt"],
            speaker_embeddings=speaker_data["speaker_embeddings"],
            emotion_audio=str(emotion_audio) if emotion_audio else None,
            emotion_alpha=emotion_alpha,
            emo_vector=emotion_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            injection_mode=injection_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            verbose=verbose,
            stream_return=stream,
        )
        
    def generate_dialogue(
        self,
        dialogue: List[Dict[str, str]],
        output_path: Optional[Path] = None,
        pause_duration: float = 0.0,
        stream: bool = False,
        verbose: bool = False,
    ) -> Path:
        """
        Generate a complete dialogue between multiple speakers.
        
        Args:
            dialogue: List of dictionaries with required 'speaker' and 'text' keys,
                     plus optional per-turn overrides:
                     - pattern_scale: Scale pattern embedding strength
                     - temperature: Generation temperature
                     - top_p: Top-p sampling
                     - top_k: Top-k sampling
                     - emotion_alpha: Emotion mixing weight
                     - use_emo_text: Extract emotion from text
                     - emo_text: Custom text for emotion extraction
                     - emotion_vector: Explicit emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
                     - emotion_audio: Path to emotion reference audio
                     - injection_mode: How to inject pattern embedding
            output_path: Path to save final podcast (auto-generated if None)
            pause_duration: Duration of pause between turns (seconds)
            stream: Use streaming generation
            verbose: Print verbose output
            
        Returns:
            Path to final podcast audio file
        """
        if output_path is None:
            output_path = PROJECT_ROOT / "output" / f"podcast_{int(time.time())}.wav"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating podcast with {len(dialogue)} turns...")
        
        # Generate audio for each turn
        audio_segments = []
        
        for i, turn in enumerate(dialogue):
            speaker = turn["speaker"]
            text = turn["text"]
            
            # Get character defaults
            char_defaults = self.character_defaults.get(speaker, {})
            
            # Merge character defaults with per-turn overrides
            generation_params = {
                "pattern_scale": turn.get("pattern_scale", char_defaults.get("pattern_scale", 0.02)),
                "temperature": turn.get("temperature", char_defaults.get("temperature", 0.8)),
                "top_p": turn.get("top_p", char_defaults.get("top_p", 0.8)),
                "top_k": turn.get("top_k", char_defaults.get("top_k", 30)),
                "emotion_alpha": turn.get("emotion_alpha", char_defaults.get("emotion_alpha", 1.0)),
                "use_emo_text": turn.get("use_emo_text", char_defaults.get("use_emo_text", False)),
                "emo_text": turn.get("emo_text", char_defaults.get("emo_text", None)),
                "emotion_vector": turn.get("emotion_vector", char_defaults.get("emotion_vector", None)),
                "emotion_audio": Path(turn["emotion_audio"]) if "emotion_audio" in turn else None,
                "injection_mode": turn.get("injection_mode", char_defaults.get("injection_mode", "add")),
            }
            
            # Build display info
            display_info = f"[{i+1}/{len(dialogue)}] {speaker}: {text[:50]}..."
            if generation_params["use_emo_text"]:
                display_info += " [emotion from text]"
            elif generation_params["emotion_vector"]:
                display_info += f" [emotion: {generation_params['emotion_vector']}]"
            print(f"\n{display_info}")
            
            # Clear cache more aggressively to prevent degradation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate speech with all parameters
            if stream:
                # Collect all chunks from streaming generator
                audio_chunks = []
                audio_gen = self.generate_speech(
                    speaker=speaker,
                    text=text,
                    stream=True,
                    verbose=verbose,
                    **generation_params
                )
                
                for chunk in audio_gen:
                    if chunk is not None:
                        audio_chunks.append(chunk)
                        
                # Combine chunks
                if audio_chunks:
                    audio = torch.cat(audio_chunks, dim=1)
                else:
                    continue
            else:
                # Non-streaming generation
                audio_path = self.generate_speech(
                    speaker=speaker,
                    text=text,
                    stream=False,
                    verbose=verbose,
                    **generation_params
                )
                
                # Load generated audio
                audio, sr = torchaudio.load(str(audio_path))
                
            # Add to segments
            audio_segments.append(audio)
            
            # Add pause between turns (except after last turn)
            if i < len(dialogue) - 1:
                pause_samples = int(pause_duration * 22050)  # 22.05kHz sample rate
                pause = torch.zeros(1, pause_samples)
                audio_segments.append(pause)
                
        # Concatenate all segments
        if audio_segments:
            final_audio = torch.cat(audio_segments, dim=1)
        else:
            raise ValueError("No audio segments generated")
            
        # Save final podcast
        final_audio = torch.clamp(32767 * final_audio, -32767.0, 32767.0)
        torchaudio.save(str(output_path), final_audio.type(torch.int16), 22050)
        
        print(f"\n✓ Podcast saved to: {output_path}")
        return output_path
        
    def generate_dialogue_streaming(
        self,
        dialogue: List[Dict[str, str]],
        pause_duration: float = 0.0,
        verbose: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate a dialogue with streaming output.
        
        Args:
            dialogue: List of dictionaries with 'speaker' and 'text' keys
            pause_duration: Duration of pause between turns (seconds)
            verbose: Print verbose output
            
        Yields:
            Audio chunks as they are generated
        """
        for i, turn in enumerate(dialogue):
            speaker = turn["speaker"]
            text = turn["text"]
            
            if verbose:
                print(f"\n[{i+1}/{len(dialogue)}] {speaker}: {text[:50]}...")
                
            # Generate speech with streaming
            audio_gen = self.generate_speech(
                speaker=speaker,
                text=text,
                stream=True,
                verbose=verbose,
            )
            
            # Yield chunks from this turn
            for chunk in audio_gen:
                if chunk is not None:
                    yield chunk
                    
            # Add pause between turns (except after last turn)
            if i < len(dialogue) - 1:
                pause_samples = int(pause_duration * 22050)
                pause = torch.zeros(1, pause_samples)
                yield pause


def create_sample_dialogue() -> List[Dict[str, str]]:
    """Create a sample dialogue between goldblum and ozzy."""
    return [
        {
            "speaker": "goldblum",
            "text": "You know, I've been thinking about the nature of reality lately. It's quite fascinating how our perceptions shape everything."
        },
        {
            "speaker": "ozzy",
            "text": "Reality? Mate, I'm just trying to figure out what's real after all these years on the road!"
        },
        {
            "speaker": "goldblum",
            "text": "Exactly! That's the beauty of it. The uncertainty, the chaos... it's where the magic happens. Life, uh, finds a way."
        },
        {
            "speaker": "ozzy",
            "text": "Haha! You're right about that. After all the crazy stuff I've seen, magic is the only explanation!"
        },
        {
            "speaker": "goldblum",
            "text": "And isn't that wonderful? The unpredictability, the sheer improbability of it all. We're all just... stumbling through the cosmos."
        },
        {
            "speaker": "ozzy",
            "text": "Stumbling is right! But hey, at least we're stumbling together. Cheers to that, mate!"
        },
    ]


def main():
    """Example usage of the podcast generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a podcast between trained voices")
    parser.add_argument("--output", "-o", type=Path, help="Output path for podcast")
    parser.add_argument("--characters", "-c", type=Path, help="JSON file with character configurations")
    parser.add_argument("--dialogue", type=Path, help="JSON file with dialogue")
    parser.add_argument("--pause", type=float, default=0.1, help="Pause duration between turns")
    parser.add_argument("--stream", action="store_true", help="Use streaming generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create podcast generator
    generator = PodcastGenerator(
        use_fp16=True,
        use_accel=True,
        use_torch_compile=True,
        use_cuda_kernel=True,
    )
    
    # Load characters or register default speakers
    if args.characters:
        generator.load_characters(args.characters)
    else:
        # Register default speakers for backward compatibility
        generator.register_speaker("goldblum")
        generator.register_speaker("ozzy")
    
    # Load dialogue
    if args.dialogue:
        with open(args.dialogue, 'r') as f:
            dialogue = json.load(f)
    else:
        dialogue = create_sample_dialogue()
        
    # Generate podcast
    if args.stream:
        print("Generating streaming podcast...")
        output_path = PROJECT_ROOT / "output" / f"podcast_stream_{int(time.time())}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect streaming chunks
        audio_chunks = []
        for chunk in generator.generate_dialogue_streaming(
            dialogue=dialogue,
            pause_duration=args.pause,
            verbose=args.verbose,
        ):
            audio_chunks.append(chunk)
            
        # Save final audio
        if audio_chunks:
            final_audio = torch.cat(audio_chunks, dim=1)
            final_audio = torch.clamp(32767 * final_audio, -32767.0, 32767.0)
            torchaudio.save(str(output_path), final_audio.type(torch.int16), 22050)
            print(f"\n✓ Streaming podcast saved to: {output_path}")
    else:
        generator.generate_dialogue(
            dialogue=dialogue,
            output_path=args.output,
            pause_duration=args.pause,
            stream=False,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()