"""
IndexTTS FastAPI Server
Provides REST API endpoints for inference, streaming, training, and model management

Streaming endpoints use the optimized streaming module for ~0.3s time-to-first-audio.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import shutil
import tempfile
from datetime import datetime
import json
import queue

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indextts.infer_v2 import IndexTTS2
from indextts.pattern_embeddings import PatternEmbedding
from indextts.streaming_v2 import (
    streaming_inference_v2,
    StreamingConfigV2,
    StreamingMode,
    get_fast_streaming_config,
    get_fast_quality_streaming_config,
    get_ultra_fast_streaming_config,
    get_ultra_fast_distilled_streaming_config,
    get_balanced_streaming_config,
    get_balanced_distilled_streaming_config,
    get_quality_streaming_config,
    get_progressive_streaming_config,
)
from tools.infer_with_patterns import pattern_aware_inference, pattern_aware_inference_streaming

# Global state
app = FastAPI(
    title="IndexTTS API",
    description="API for IndexTTS2 text-to-speech synthesis with pattern embeddings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for WebUI
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global TTS model
tts_model: Optional[IndexTTS2] = None
loaded_models: Dict[str, Dict[str, Any]] = {}
training_tasks: Dict[str, Dict[str, Any]] = {}


# ============= Pydantic Models =============

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    speaker: Optional[str] = Field(None, description="Speaker name (for pattern embeddings)")
    use_patterns: bool = Field(False, description="Use pattern embeddings")
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    top_k: int = Field(30, ge=0, le=100)
    emo_vector: Optional[List[float]] = Field(None, description="Emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
    use_emo_text: bool = Field(False, description="Extract emotion from text")
    emo_text: Optional[str] = Field(None, description="Custom emotion text")
    use_fp16: bool = Field(False, description="Use FP16 precision")
    use_torch_compile: bool = Field(False, description="Use torch.compile optimization")


class StreamTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    speaker: Optional[str] = Field(None, description="Speaker name")
    use_patterns: bool = Field(False, description="Use pattern embeddings")
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.8, ge=0.0, le=1.0)
    top_k: int = Field(30, ge=0, le=100)
    emo_vector: Optional[List[float]] = Field(None, description="Emotion vector")
    use_emo_text: bool = Field(False, description="Extract emotion from text")
    emo_text: Optional[str] = Field(None, description="Custom emotion text")
    # Streaming-specific options
    min_chunk_tokens: int = Field(15, ge=5, le=50, description="Tokens for first chunk (lower = faster TTFA)")
    chunk_tokens: int = Field(50, ge=20, le=100, description="Tokens per subsequent chunk")
    diffusion_steps: int = Field(12, ge=4, le=30, description="Diffusion steps (lower = faster)")
    first_chunk_diffusion_steps: int = Field(6, ge=2, le=15, description="Diffusion steps for first chunk")
    # Latency/quality preset. "ultra_fast" / "fast" pick Phase 1 optimized configs (Heun
    # solver + CFG=0 first chunk + small min_chunk_tokens). When set to anything other
    # than "custom", the scalar knobs above are ignored.
    streaming_preset: str = Field(
        "ultra_fast",
        description="One of: ultra_fast, ultra_fast_distilled, fast, fast_quality, balanced, balanced_distilled, quality, progressive, custom",
    )
    verbose: bool = Field(True, description="Log per-stage timing on the server (for TTFA debugging)")
    # Overrides applied AFTER the preset is built. Use these from the Inference tab to
    # force a specific solver (e.g. "single_step" when a distilled student is active)
    # without having to wire up a separate "custom" preset.
    solver_override: Optional[str] = Field(None, description="If set, replaces the preset's solver. One of: euler, heun, single_step.")
    diffusion_steps_override: Optional[int] = Field(None, ge=1, le=50, description="If set, replaces the preset's diffusion_steps AND first_chunk_diffusion_steps (use 1 for distilled student).")
    inference_cfg_override: Optional[float] = Field(None, ge=0.0, le=2.0, description="If set, replaces the preset's CFG rate (use 0.0 for distilled student).")


class TrainingRequest(BaseModel):
    speaker_name: str = Field(..., description="Name for the new speaker")
    epochs: int = Field(40, ge=1, le=200)
    pattern_tokens: int = Field(8, ge=1, le=32)
    lora_rank: int = Field(32, ge=4, le=128)
    learning_rate: float = Field(5e-4, ge=1e-6, le=1e-2)
    batch_size: int = Field(4, ge=1, le=32)
    whisper_model: str = Field("medium", description="Whisper model size")


class ModelInfo(BaseModel):
    name: str
    type: str
    loaded: bool
    has_lora: bool
    has_patterns: bool
    path: Optional[str] = None


class TrainingStatus(BaseModel):
    task_id: str
    speaker_name: str
    status: str
    progress: float
    message: str
    started_at: str
    completed_at: Optional[str] = None


# ============= Startup/Shutdown =============

@app.on_event("startup")
async def startup_event():
    """Server startup - model loading disabled, use manual loading instead"""
    global tts_model
    
    model_dir = PROJECT_ROOT / "checkpoints"
    if not model_dir.exists():
        print("⚠️ Warning: checkpoints directory not found.")
    
    print("✅ IndexTTS API server started")
    print("💡 Model not loaded automatically - use the WebUI or API to load models")
    tts_model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global tts_model
    if tts_model is not None:
        del tts_model
        torch.cuda.empty_cache()
    print("👋 Server shutdown complete")


# ============= Health Check =============

@app.get("/")
async def root():
    """Serve WebUI"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "name": "IndexTTS API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": tts_model is not None,
        "message": "WebUI not found. Please ensure static files are present."
    }


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": "IndexTTS API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": tts_model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "device": str(tts_model.device) if tts_model else None,
        "cuda_available": torch.cuda.is_available()
    }


# ============= Model Management =============

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models and their status"""
    models = []
    training_dir = PROJECT_ROOT / "training"
    
    # Add base model
    models.append(ModelInfo(
        name="base",
        type="base",
        loaded=tts_model is not None,
        has_lora=False,
        has_patterns=False
    ))
    
    # Check for trained speakers
    if training_dir.exists():
        for speaker_dir in training_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            has_lora = False
            has_patterns = False
            
            # Check for pattern embeddings
            pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
            if pattern_path.exists():
                has_patterns = True
            
            # Check for a GPT LoRA in any of the supported training dirs.
            if _find_gpt_lora_for_speaker(speaker_dir.name) is not None:
                has_lora = True
            
            if has_lora or has_patterns:
                models.append(ModelInfo(
                    name=speaker_dir.name,
                    type="speaker",
                    loaded=speaker_dir.name in loaded_models,
                    has_lora=has_lora,
                    has_patterns=has_patterns,
                    path=str(speaker_dir)
                ))
    
    return models


@app.post("/models/load/{speaker_name}")
async def load_model(speaker_name: str):
    """Load a specific speaker model or base model"""
    global tts_model, loaded_models
    
    if speaker_name == "base":
        # Load base model
        model_dir = PROJECT_ROOT / "checkpoints"
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Checkpoints directory not found")

        # Phase 3: optionally overlay a distilled CFM. Two ways to wire it in:
        #   1. env var S2MEL_DISTILLED_CHECKPOINT=/abs/path/to/checkpoint.pth
        #   2. drop a file at checkpoints/s2mel_distilled.pth — auto-detected
        # The student is loaded on top of the base s2mel; only keys it contains
        # (typically the "cfm" submodule) are replaced.
        distilled_ckpt: Optional[str] = os.environ.get("S2MEL_DISTILLED_CHECKPOINT") or None
        if distilled_ckpt is None:
            auto_path = model_dir / "s2mel_distilled.pth"
            if auto_path.exists():
                distilled_ckpt = str(auto_path)

        try:
            print("🚀 Loading IndexTTS2 base model...")
            if distilled_ckpt:
                print(f"   with distilled CFM: {distilled_ckpt}")
            tts_model = IndexTTS2(
                model_dir=str(model_dir),
                use_fp16=torch.cuda.is_available(),
                use_cuda_kernel=torch.cuda.is_available(),
                use_accel=True,
                use_deepspeed=True,
                use_torch_compile=True,
                s2mel_distilled_checkpoint=distilled_ckpt,
            )
            loaded_models.clear()
            print("✅ IndexTTS2 base model loaded successfully")
            return {
                "status": "success",
                "message": "Base model loaded successfully",
                "distilled_cfm": distilled_ckpt,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load base model: {str(e)}")
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Base TTS model not loaded. Load base model first.")
    
    speaker_dir = PROJECT_ROOT / "training" / speaker_name
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_name}' not found")
    
    # Find a GPT LoRA — check character_lora (new trainer) first, then verbatim_training,
    # then the legacy pattern_training paths. First hit wins.
    lora_path = _find_gpt_lora_for_speaker(speaker_name)

    if lora_path is not None:
        try:
            tts_model.load_lora(str(lora_path))
            loaded_models[speaker_name] = {"lora_path": str(lora_path)}
            return {"status": "success", "message": f"Loaded LoRA for {speaker_name}", "lora_path": str(lora_path)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load LoRA: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail=f"No LoRA found for speaker '{speaker_name}'")


# ============= Inference Endpoints =============

@app.post("/inference/generate")
async def generate_speech(
    audio_file: Optional[UploadFile] = File(None, description="Speaker reference audio (optional when using patterns)"),
    request_json: str = Form(..., description="JSON request parameters")
):
    """Generate speech with optional reference audio (required unless using patterns)"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    # Parse request
    try:
        request = TTSRequest.parse_raw(request_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Validate inputs
    if not request.use_patterns and audio_file is None:
        raise HTTPException(status_code=400, detail="audio_file is required when not using patterns")
    
    # Save uploaded audio temporarily (only if provided)
    tmp_audio_path = None
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            shutil.copyfileobj(audio_file.file, tmp_audio)
            tmp_audio_path = tmp_audio.name
    
    try:
        # Prepare output path
        output_dir = PROJECT_ROOT / "outputs" / "api"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        # Generate speech
        if request.use_patterns and request.speaker:
            # Use pattern-aware inference
            result = await generate_with_patterns(
                speaker=request.speaker,
                text=request.text,
                audio_prompt=tmp_audio_path,
                output_path=output_path,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                emo_vector=request.emo_vector,
                use_emo_text=request.use_emo_text,
                emo_text=request.emo_text
            )
        else:
            # Standard inference
            result = tts_model.infer(
                spk_audio_prompt=tmp_audio_path,
                text=request.text,
                output_path=str(output_path),
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                emo_vector=request.emo_vector,
                use_emo_text=request.use_emo_text,
                emo_text=request.emo_text
            )
        
        # Return generated audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
    
    finally:
        # Cleanup temp file
        if tmp_audio_path is not None:
            try:
                os.unlink(tmp_audio_path)
            except:
                pass


def _build_streaming_config(request: "StreamTTSRequest") -> StreamingConfigV2:
    """Map a StreamTTSRequest to a StreamingConfigV2.

    Presets pull from indextts.streaming_v2 helpers. "custom" honors the scalar fields
    on the request so the WebUI's manual sliders still work.
    """
    preset = (request.streaming_preset or "ultra_fast").lower()
    if preset == "ultra_fast":
        config = get_ultra_fast_streaming_config()
    elif preset == "ultra_fast_distilled":
        config = get_ultra_fast_distilled_streaming_config()
    elif preset == "fast":
        config = get_fast_streaming_config()
    elif preset == "fast_quality":
        config = get_fast_quality_streaming_config()
    elif preset == "balanced":
        config = get_balanced_streaming_config()
    elif preset == "balanced_distilled":
        config = get_balanced_distilled_streaming_config()
    elif preset == "quality":
        config = get_quality_streaming_config()
    elif preset == "progressive":
        config = get_progressive_streaming_config()
    elif preset == "custom":
        config = StreamingConfigV2(
            mode=StreamingMode.FAST_CHUNKS,
            min_chunk_tokens=request.min_chunk_tokens,
            chunk_tokens=request.chunk_tokens,
            diffusion_steps=request.diffusion_steps,
            first_chunk_diffusion_steps=request.first_chunk_diffusion_steps,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown streaming_preset: {preset}")
    # `verbose` is only on StreamTTSRequest, not WarmupRequest — getattr keeps the
    # helper usable for both. Defaults to True so server logs are populated.
    config.verbose = bool(getattr(request, "verbose", True))

    # Apply optional overrides (so e.g. the Inference tab can force single_step + 1 step
    # when a distilled student is loaded, regardless of the chosen preset).
    solver_override = getattr(request, "solver_override", None)
    if solver_override:
        config.solver = solver_override
    steps_override = getattr(request, "diffusion_steps_override", None)
    if steps_override is not None:
        config.diffusion_steps = int(steps_override)
        config.first_chunk_diffusion_steps = int(steps_override)
    cfg_override = getattr(request, "inference_cfg_override", None)
    if cfg_override is not None:
        config.inference_cfg_rate = float(cfg_override)
        config.first_chunk_cfg_rate = float(cfg_override)
    return config


@app.post("/inference/stream")
async def stream_speech(
    audio_file: Optional[UploadFile] = File(None, description="Speaker reference audio (optional when using patterns)"),
    request_json: str = Form(..., description="JSON request parameters")
):
    """
    Stream speech generation with optimized streaming for fast time-to-first-audio.
    
    Uses the optimized streaming module that synthesizes audio chunks as mel tokens
    are generated, achieving ~0.3s time-to-first-audio vs 3-15s for standard inference.
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    # Parse request
    try:
        request = StreamTTSRequest.parse_raw(request_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Validate inputs
    if not request.use_patterns and audio_file is None:
        raise HTTPException(status_code=400, detail="audio_file is required when not using patterns")
    
    # Save uploaded audio temporarily (only if provided)
    tmp_audio_path = None
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            shutil.copyfileobj(audio_file.file, tmp_audio)
            tmp_audio_path = tmp_audio.name
    
    async def generate_chunks():
        """
        OPTIMIZED streaming generator using the new streaming module.
        
        Achieves ~0.3s time-to-first-audio by synthesizing chunks during
        GPT token generation.
        """
        import io
        import wave
        
        try:
            # Load speaker embeddings and pattern embedding if using patterns
            speaker_embeddings = None
            pattern_embedding = None
            cond_cache_key: Optional[str] = None

            if request.use_patterns and request.speaker:
                speaker_dir = PROJECT_ROOT / "training" / request.speaker

                # Load pattern embedding
                pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
                if not pattern_path.exists():
                    pattern_path = speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt"

                if pattern_path.exists():
                    pattern_embedding = PatternEmbedding.load(pattern_path, device=tts_model.device)
                    pattern_embedding.eval()
                # Load speaker embeddings if no audio file
                if tmp_audio_path is None:
                    from indextts.speaker_embeddings import SpeakerEmbeddingStore

                    speaker_emb_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"

                    if speaker_emb_path.exists():
                        store = SpeakerEmbeddingStore(tts_model)
                        speaker_embeddings = store.load_embeddings(speaker_emb_path)
                        cond_cache_key = str(speaker_emb_path)
                    else:
                        yield b"Error: Speaker embeddings not found"
                        return
            
            # Build streaming config from the requested preset
            stream_config = _build_streaming_config(request)
            print(
                f"[stream] preset={request.streaming_preset} "
                f"mode={stream_config.mode.value} "
                f"min_chunk_tokens={stream_config.min_chunk_tokens} "
                f"first_steps={stream_config.first_chunk_diffusion_steps} "
                f"solver={stream_config.solver} "
                f"first_cfg={stream_config.first_chunk_cfg_rate}",
                flush=True,
            )

            chunk_idx = 0
            header_sent = False

            for wav_chunk in pattern_aware_inference_streaming(
                tts=tts_model,
                text=request.text,
                audio_prompt=tmp_audio_path,
                speaker_embeddings=speaker_embeddings,
                emotion_audio=None,
                emo_vector=request.emo_vector,
                use_emo_text=request.use_emo_text,
                emo_text=request.emo_text,
                config=stream_config,
                pattern_embedding=pattern_embedding,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                cond_cache_key=cond_cache_key,
            ):
                # Ensure chunk has correct shape
                if wav_chunk.dim() == 1:
                    wav_chunk = wav_chunk.unsqueeze(0)
                
                # Convert to int16 bytes
                chunk_int16 = wav_chunk.type(torch.int16)
                
                # Send WAV header with first chunk
                if not header_sent:
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(22050)
                        wav_file.writeframes(b'\x00\x00')
                    
                    wav_io.seek(0)
                    header = wav_io.read(44)
                    yield header
                    header_sent = True
                
                # Stream raw PCM data
                yield chunk_int16.cpu().numpy().tobytes()
                chunk_idx += 1
        
        finally:
            # Cleanup temp file
            if tmp_audio_path is not None:
                try:
                    os.unlink(tmp_audio_path)
                except:
                    pass
    
    return StreamingResponse(generate_chunks(), media_type="audio/wav")


# ============= Phase 2 testing helpers =============


class WarmupRequest(BaseModel):
    """Warm up the streaming pipeline (CUDA graph capture, torch.compile JIT, BigVGAN
    kernel init). Run once after model load before measuring TTFA.

    Use realistic text — the warmup needs to hit chunk_tokens-sized synthesis (~40
    tokens, ~70 mel frames) so CUDA graphs and torch.compile specialize for the
    shapes the real benchmark will use. A 3-token toy warmup misses this.
    """
    speaker: Optional[str] = Field(None, description="Speaker name (uses stored embeddings)")
    use_patterns: bool = Field(False, description="Use pattern embeddings if available")
    text: str = Field(
        "Hello, this is a warmup pass to capture CUDA graphs and just-in-time compile "
        "the diffusion model against realistic chunk shapes before benchmarking.",
        description="Warmup text. Should be long enough to trigger multiple full-size chunks.",
    )
    streaming_preset: str = Field("ultra_fast", description="Single preset to warm up with (used if `presets` is empty)")
    presets: List[str] = Field(
        default_factory=list,
        description="Optional list of presets to warm in sequence. If non-empty, overrides streaming_preset.",
    )


@app.post("/inference/warmup")
async def warmup(
    audio_file: Optional[UploadFile] = File(None),
    request_json: str = Form(...),
):
    """Run a one-shot streaming synthesis to warm up the pipeline.

    The very first request after model load pays the cost of:
      - AccelInferenceEngine CUDA graph capture (~1-2s)
      - torch.compile JIT for CFM (~5-10s if use_torch_compile=True)
      - BigVGAN CUDA kernel init (~100-500ms)

    This endpoint eats those costs upfront so the first real request hits warm caches.
    Returns timing info; the audio itself is discarded.
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    try:
        request = WarmupRequest.parse_raw(request_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    if not request.use_patterns and audio_file is None and not request.speaker:
        raise HTTPException(
            status_code=400,
            detail="Provide audio_file, speaker (with use_patterns), or both",
        )

    tmp_audio_path = None
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            tmp_audio_path = tmp.name

    speaker_embeddings = None
    pattern_embedding = None
    cond_cache_key: Optional[str] = None
    try:
        if request.use_patterns and request.speaker:
            speaker_dir = PROJECT_ROOT / "training" / request.speaker
            pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
            if not pattern_path.exists():
                pattern_path = speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt"
            if pattern_path.exists():
                pattern_embedding = PatternEmbedding.load(pattern_path, device=tts_model.device)
                pattern_embedding.eval()
            if tmp_audio_path is None:
                from indextts.speaker_embeddings import SpeakerEmbeddingStore
                speaker_emb_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
                if speaker_emb_path.exists():
                    store = SpeakerEmbeddingStore(tts_model)
                    speaker_embeddings = store.load_embeddings(speaker_emb_path)
                    cond_cache_key = str(speaker_emb_path)
                else:
                    raise HTTPException(status_code=404, detail="Speaker embeddings not found")

        # Build the list of presets to warm. CUDA graphs are captured per-shape, and
        # CFM's torch.compile may re-JIT for different chunk sizes, so warming each
        # preset separately gives the cleanest benchmarking baseline.
        presets_to_warm = request.presets if request.presets else [request.streaming_preset]

        import time as _time
        per_preset = []
        overall_start = _time.perf_counter()

        for preset_name in presets_to_warm:
            # Mutate `request` so _build_streaming_config picks up this preset.
            request.streaming_preset = preset_name
            stream_config = _build_streaming_config(request)
            stream_config.verbose = True

            t_start = _time.perf_counter()
            t_first_chunk = None
            chunk_count = 0
            total_audio_samples = 0

            for wav_chunk in pattern_aware_inference_streaming(
                tts=tts_model,
                text=request.text,
                audio_prompt=tmp_audio_path,
                speaker_embeddings=speaker_embeddings,
                config=stream_config,
                pattern_embedding=pattern_embedding,
                cond_cache_key=cond_cache_key,
            ):
                if t_first_chunk is None:
                    t_first_chunk = _time.perf_counter() - t_start
                chunk_count += 1
                total_audio_samples += wav_chunk.shape[-1]

            per_preset.append({
                "preset": preset_name,
                "ttfa_ms": round((t_first_chunk or 0) * 1000, 1),
                "total_time_ms": round((_time.perf_counter() - t_start) * 1000, 1),
                "chunks": chunk_count,
                "audio_seconds": round(total_audio_samples / 22050.0, 3),
            })

        total_elapsed = _time.perf_counter() - overall_start

        # Backwards-compatible top-level fields for the single-preset case + a per-preset breakdown.
        first = per_preset[0] if per_preset else {}
        return {
            "status": "warmed_up",
            "preset": first.get("preset"),
            "ttfa_ms": first.get("ttfa_ms"),
            "total_time_ms": first.get("total_time_ms"),
            "chunks": first.get("chunks"),
            "audio_seconds": first.get("audio_seconds"),
            "presets_warmed": per_preset,
            "overall_time_ms": round(total_elapsed * 1000, 1),
            "message": (
                "Pipeline is warm. CUDA graphs captured, kernels initialized."
                if len(per_preset) == 1
                else f"Warmed {len(per_preset)} presets in {total_elapsed:.1f}s."
            ),
        }
    finally:
        if tmp_audio_path is not None:
            try:
                os.unlink(tmp_audio_path)
            except OSError:
                pass


@app.post("/inference/stream/diagnostics")
async def stream_diagnostics(
    audio_file: Optional[UploadFile] = File(None),
    request_json: str = Form(...),
):
    """Run a streaming synthesis and return JSON timing breakdown instead of audio.

    Same request format as /inference/stream — accepts the full StreamTTSRequest.
    Useful for benchmarking TTFA + steady-state cadence without parsing WAV bytes.
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    try:
        request = StreamTTSRequest.parse_raw(request_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    if not request.use_patterns and audio_file is None:
        raise HTTPException(status_code=400, detail="audio_file is required when not using patterns")

    tmp_audio_path = None
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            tmp_audio_path = tmp.name

    speaker_embeddings = None
    pattern_embedding = None
    cond_cache_key: Optional[str] = None
    try:
        if request.use_patterns and request.speaker:
            speaker_dir = PROJECT_ROOT / "training" / request.speaker
            pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
            if not pattern_path.exists():
                pattern_path = speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt"
            if pattern_path.exists():
                pattern_embedding = PatternEmbedding.load(pattern_path, device=tts_model.device)
                pattern_embedding.eval()
            if tmp_audio_path is None:
                from indextts.speaker_embeddings import SpeakerEmbeddingStore
                speaker_emb_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
                if speaker_emb_path.exists():
                    store = SpeakerEmbeddingStore(tts_model)
                    speaker_embeddings = store.load_embeddings(speaker_emb_path)
                    cond_cache_key = str(speaker_emb_path)
                else:
                    raise HTTPException(status_code=404, detail="Speaker embeddings not found")

        stream_config = _build_streaming_config(request)

        import time as _time
        t_start = _time.perf_counter()
        chunk_timings = []
        total_audio_samples = 0
        last_t = t_start
        timing_events: list = []

        for wav_chunk in pattern_aware_inference_streaming(
            tts=tts_model,
            text=request.text,
            audio_prompt=tmp_audio_path,
            speaker_embeddings=speaker_embeddings,
            emotion_audio=None,
            emo_vector=request.emo_vector,
            use_emo_text=request.use_emo_text,
            emo_text=request.emo_text,
            config=stream_config,
            pattern_embedding=pattern_embedding,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            timing_log=timing_events,
            cond_cache_key=cond_cache_key,
        ):
            now = _time.perf_counter()
            samples = int(wav_chunk.shape[-1])
            chunk_timings.append({
                "elapsed_ms": round((now - t_start) * 1000, 1),
                "since_prev_ms": round((now - last_t) * 1000, 1),
                "samples": samples,
                "audio_ms": round(samples / 22.050, 1),
            })
            total_audio_samples += samples
            last_t = now

        t_total = _time.perf_counter() - t_start
        audio_seconds = total_audio_samples / 22050.0

        # Derive a per-stage summary from the raw event log. This is what tells us
        # where the first-chunk time actually goes (setup vs prefill vs synth vs ...).
        def _t(event_name, chunk_idx=None):
            for ev in timing_events:
                if ev["event"] == event_name and (chunk_idx is None or ev.get("chunk_idx") == chunk_idx):
                    return ev["t_ms"]
            return None

        stages = {
            "request_start_ms":            _t("request_start"),
            "conditioning_done_ms":        _t("conditioning_done"),
            "threads_starting_ms":         _t("threads_starting"),
            "gpt_first_token_ms":          _t("gpt_first_token"),
            "chunk1_dispatched_ms":        _t("chunk_dispatched", 1),
            "chunk1_synth_start_ms":       _t("synth_start", 1),
            "chunk1_gpt_latent_done_ms":   _t("synth_gpt_latent_done", 1),
            "chunk1_length_reg_done_ms":   _t("synth_length_reg_done", 1),
            "chunk1_cfm_done_ms":          _t("synth_cfm_done", 1),
            "chunk1_bigvgan_done_ms":      _t("synth_bigvgan_done", 1),
            "chunk1_synth_done_ms":        _t("synth_done", 1),
            "chunk1_yielded_ms":           _t("chunk_yielded", 1),
            "chunk2_dispatched_ms":        _t("chunk_dispatched", 2),
            "chunk2_synth_start_ms":       _t("synth_start", 2),
            "chunk2_gpt_latent_done_ms":   _t("synth_gpt_latent_done", 2),
            "chunk2_length_reg_done_ms":   _t("synth_length_reg_done", 2),
            "chunk2_cfm_done_ms":          _t("synth_cfm_done", 2),
            "chunk2_bigvgan_done_ms":      _t("synth_bigvgan_done", 2),
            "chunk2_synth_done_ms":        _t("synth_done", 2),
            "chunk2_yielded_ms":           _t("chunk_yielded", 2),
        }

        return {
            "preset": request.streaming_preset,
            "solver": stream_config.solver,
            "first_chunk_cfg_rate": stream_config.first_chunk_cfg_rate,
            "min_chunk_tokens": stream_config.min_chunk_tokens,
            "first_chunk_diffusion_steps": stream_config.first_chunk_diffusion_steps,
            "ttfa_ms": chunk_timings[0]["elapsed_ms"] if chunk_timings else None,
            "total_time_ms": round(t_total * 1000, 1),
            "audio_seconds": round(audio_seconds, 3),
            "rtf": round(t_total / audio_seconds, 3) if audio_seconds > 0 else None,
            "chunk_count": len(chunk_timings),
            "accel_engine_active": getattr(tts_model.gpt, "accel_engine", None) is not None,
            "chunks": chunk_timings,
            "stages": stages,
            "events": timing_events,
        }
    finally:
        if tmp_audio_path is not None:
            try:
                os.unlink(tmp_audio_path)
            except OSError:
                pass


async def generate_with_patterns(
    speaker: str,
    text: str,
    audio_prompt: Optional[str],
    output_path: Path,
    **kwargs
):
    """Helper function for pattern-aware inference"""
    speaker_dir = PROJECT_ROOT / "training" / speaker
    
    # Find pattern embedding
    pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
    if not pattern_path.exists():
        pattern_path = speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt"
    
    if not pattern_path.exists():
        raise HTTPException(status_code=404, detail=f"Pattern embedding not found for speaker '{speaker}'")
    
    # Load pattern embedding
    pattern_embedding = PatternEmbedding.load(pattern_path, device=tts_model.device)
    pattern_embedding.eval()
    if hasattr(pattern_embedding, 'pattern_scale'):
        original_scale = pattern_embedding.pattern_scale.item()
    # Load speaker embeddings if no audio prompt
    speaker_embeddings = None
    if audio_prompt is None:
        from indextts.speaker_embeddings import SpeakerEmbeddingStore
        
        speaker_emb_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
        if not speaker_emb_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Speaker embeddings not found for '{speaker}'. Either provide audio_file or train embeddings first."
            )
        
        store = SpeakerEmbeddingStore(tts_model)
        speaker_embeddings = store.load_embeddings(speaker_emb_path)
    
    # Generate
    result = pattern_aware_inference(
        tts=tts_model,
        pattern_embedding=pattern_embedding,
        text=text,
        output_path=output_path,
        audio_prompt=audio_prompt,
        speaker_embeddings=speaker_embeddings,
        stream_return=False,
        **kwargs
    )
    
    return result


# ============= Training Endpoints =============

@app.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    audio_files: List[UploadFile] = File(..., description="Training audio files"),
    request_json: str = Form(..., description="JSON training parameters")
):
    """Start training a new speaker model"""
    
    # Parse request
    try:
        request = TrainingRequest.parse_raw(request_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Validate files
    if len(audio_files) < 5:
        raise HTTPException(status_code=400, detail="At least 5 audio files required for training")
    
    # Create speaker directory
    speaker_dir = PROJECT_ROOT / "training" / request.speaker_name
    if speaker_dir.exists():
        raise HTTPException(status_code=400, detail=f"Speaker '{request.speaker_name}' already exists")
    
    audio_dir = speaker_dir / "dataset" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for i, audio_file in enumerate(audio_files):
        file_ext = Path(audio_file.filename).suffix or ".wav"
        save_path = audio_dir / f"audio_{i:03d}{file_ext}"
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        saved_files.append(str(save_path))
    
    # Create task ID
    task_id = f"{request.speaker_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Store task info
    training_tasks[task_id] = {
        "speaker_name": request.speaker_name,
        "status": "queued",
        "progress": 0.0,
        "message": "Training queued",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "audio_files": saved_files
    }
    
    # Start training in background
    background_tasks.add_task(
        run_training_pipeline,
        task_id=task_id,
        speaker_name=request.speaker_name,
        epochs=request.epochs,
        pattern_tokens=request.pattern_tokens,
        lora_rank=request.lora_rank,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size,
        whisper_model=request.whisper_model
    )
    
    return {
        "task_id": task_id,
        "speaker_name": request.speaker_name,
        "status": "queued",
        "message": f"Training started for {request.speaker_name}",
        "audio_files_count": len(saved_files)
    }


@app.get("/training/status/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str):
    """Get training task status"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"Training task '{task_id}' not found")
    
    task = training_tasks[task_id]
    return TrainingStatus(
        task_id=task_id,
        speaker_name=task["speaker_name"],
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        started_at=task["started_at"],
        completed_at=task.get("completed_at")
    )


@app.get("/training/tasks")
async def list_training_tasks():
    """List all training tasks"""
    return [
        {
            "task_id": task_id,
            **task_info
        }
        for task_id, task_info in training_tasks.items()
    ]


async def run_training_pipeline(
    task_id: str,
    speaker_name: str,
    epochs: int,
    pattern_tokens: int,
    lora_rank: int,
    learning_rate: float,
    batch_size: int,
    whisper_model: str
):
    """Run the training pipeline in background"""
    import subprocess
    
    training_tasks[task_id]["status"] = "running"
    training_tasks[task_id]["message"] = "Training started"
    training_tasks[task_id]["progress"] = 0.1
    
    try:
        # Run training script
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "train_patterns_pipeline.py"),
            "--speaker", speaker_name,
            "--epochs", str(epochs),
            "--pattern-tokens", str(pattern_tokens),
            "--lora-rank", str(lora_rank),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--whisper-model", whisper_model
        ]
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Monitor progress
        for line in process.stdout:
            print(f"[{task_id}] {line.rstrip()}")
            
            # Update progress based on output
            if "STEP 1/" in line:
                training_tasks[task_id]["progress"] = 0.2
                training_tasks[task_id]["message"] = "Transcribing audio..."
            elif "STEP 2/" in line:
                training_tasks[task_id]["progress"] = 0.3
                training_tasks[task_id]["message"] = "Extracting conditioning..."
            elif "STEP 3/" in line:
                training_tasks[task_id]["progress"] = 0.4
                training_tasks[task_id]["message"] = "Preparing dataset..."
            elif "STEP 4/" in line:
                training_tasks[task_id]["progress"] = 0.5
                training_tasks[task_id]["message"] = "Extracting embeddings..."
            elif "STEP 5/" in line:
                training_tasks[task_id]["progress"] = 0.6
                training_tasks[task_id]["message"] = "Training pattern embeddings..."
            elif "Epoch" in line:
                # Try to extract epoch number
                import re
                match = re.search(r'Epoch (\d+)/(\d+)', line)
                if match:
                    current_epoch = int(match.group(1))
                    total_epochs = int(match.group(2))
                    progress = 0.6 + (0.3 * current_epoch / total_epochs)
                    training_tasks[task_id]["progress"] = progress
                    training_tasks[task_id]["message"] = f"Training epoch {current_epoch}/{total_epochs}..."
        
        process.wait()
        
        if process.returncode == 0:
            training_tasks[task_id]["status"] = "completed"
            training_tasks[task_id]["progress"] = 1.0
            training_tasks[task_id]["message"] = "Training completed successfully"
            training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        else:
            training_tasks[task_id]["status"] = "failed"
            training_tasks[task_id]["message"] = f"Training failed with exit code {process.returncode}"
            training_tasks[task_id]["completed_at"] = datetime.now().isoformat()
    
    except Exception as e:
        training_tasks[task_id]["status"] = "failed"
        training_tasks[task_id]["message"] = f"Training failed: {str(e)}"
        training_tasks[task_id]["completed_at"] = datetime.now().isoformat()


# ============= Speaker Endpoints =============

@app.get("/speakers")
async def list_speakers():
    """List all speakers with everything the UI needs to show useful state.

    A "trainable" speaker is one with audio + at least one of: a character LoRA,
    pattern embeddings, or stored speaker embeddings. The Inference tab's
    dropdown shows all of these; the WebUI no longer filters by has_patterns
    (which used to hide new character-LoRA speakers that didn't go through the
    legacy pattern-embedding pipeline).
    """
    speakers = []
    training_dir = PROJECT_ROOT / "training"
    if not training_dir.exists():
        return speakers

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    for speaker_dir in sorted(p for p in training_dir.iterdir() if p.is_dir()):
        name = speaker_dir.name

        embeddings_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
        pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
        audio_dir = speaker_dir / "dataset" / "audio"
        n_audio = 0
        if audio_dir.exists():
            n_audio = sum(1 for p in audio_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in audio_exts)

        lora_info = _classify_gpt_lora(name)
        lora_kind = lora_info["kind"] if lora_info else None
        lora_path_value = str(lora_info["path"]) if lora_info else None
        student = _student_path(name)

        speakers.append({
            "name": name,
            "has_embeddings": embeddings_path.exists(),
            "has_patterns": pattern_path.exists(),
            # Strict: only the new clean-text trainer's output counts as a
            # "real" character LoRA. Legacy verbatim/pattern adapters load fine
            # at inference but don't behave the same on clean text — they're
            # surfaced via lora_kind so the UI can warn.
            "has_character_lora": lora_kind == "character",
            "gpt_lora_kind": lora_kind,
            "gpt_lora_path": lora_path_value,
            "character_lora_path": lora_path_value if lora_kind == "character" else None,
            "n_audio_files": n_audio,
            "has_distilled_student": student.exists(),
            "is_active_student": _is_active_student_for_speaker(name),
            # A speaker is "loadable" — i.e. worth showing in the Inference dropdown
            # — if it has any inference-time artifact (any LoRA kind, patterns,
            # or stored embeddings).
            "is_loadable": bool(lora_info) or pattern_path.exists() or embeddings_path.exists(),
            "path": str(speaker_dir),
        })

    return speakers


# ============= Phase 3: Reflow distillation =============
#
# Each distillation stage (snapshot teacher / build manifest / generate pairs /
# train student / A/B eval / activate) is exposed as an endpoint. Long-running
# stages spawn subprocesses of the corresponding tools/*.py script and stream
# progress back via the shared `distillation_tasks` registry.

distillation_tasks: Dict[str, Dict[str, Any]] = {}


def _teacher_path(speaker: str) -> Path:
    """Per-voice teacher snapshot path. The plain default is reserved for the base."""
    return PROJECT_ROOT / "checkpoints" / f"s2mel_teacher_{speaker}.pth"


def _pairs_dir(speaker: str) -> Path:
    return PROJECT_ROOT / "training" / speaker / "reflow_pairs"


def _manifest_path(speaker: str) -> Path:
    return PROJECT_ROOT / "training" / speaker / "reflow_manifest.jsonl"


def _student_path(speaker: str) -> Path:
    return PROJECT_ROOT / "training" / speaker / "cfm_reflow_student" / "best.pth"


def _active_student_path() -> Path:
    return PROJECT_ROOT / "checkpoints" / "s2mel_distilled.pth"


def _active_student_meta_path() -> Path:
    """Sidecar JSON next to s2mel_distilled.pth recording which speaker activated
    it. Written by /distill/activate; read by status endpoints so we don't have
    to guess from byte-size equality (which is unreliable — all speakers' student
    checkpoints share the same architecture and therefore the same byte size)."""
    return PROJECT_ROOT / "checkpoints" / "s2mel_distilled.meta.json"


# Cache for file-head hashes keyed by (resolved_path, mtime_ns, size). Cleared
# implicitly whenever a file changes (mtime / size change → different key).
_FILE_HEAD_HASH_CACHE: Dict[tuple, str] = {}


def _file_head_sha256(path: Path, n_bytes: int = 1 << 20) -> Optional[str]:
    """Return a hex digest of the first n_bytes of a file, or None if unreadable.
    1 MB is plenty to distinguish per-speaker LoRA-overlaid student checkpoints
    — verified empirically that distinct speakers' best.pth diverge well within
    the first MB despite sharing architecture and total byte size."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    cached = _FILE_HEAD_HASH_CACHE.get(key)
    if cached:
        return cached
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(n_bytes))
        digest = h.hexdigest()
    except OSError:
        return None
    _FILE_HEAD_HASH_CACHE[key] = digest
    return digest


def _resolve_active_student_speaker() -> Optional[str]:
    """Return the speaker whose best.pth is the currently-active distilled
    checkpoint. Prefers the sidecar manifest; falls back to head-hash
    comparison so already-activated checkpoints from before the sidecar
    feature still resolve correctly."""
    active = _active_student_path()
    if not active.exists():
        return None

    # 1) Sidecar wins if its claimed speaker still points at a matching student.
    meta_path = _active_student_meta_path()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            sp = meta.get("speaker")
            if sp:
                cand = _student_path(sp)
                if cand.exists():
                    expected = meta.get("sha256_head")
                    if expected:
                        # Validate the active checkpoint still matches what the
                        # sidecar describes (user could have manually replaced
                        # the file without re-running activate).
                        if _file_head_sha256(active) == expected:
                            return sp
                    else:
                        return sp
        except (OSError, json.JSONDecodeError):
            pass

    # 2) Hash-based fallback: hash the active file and every speaker's best.pth
    #    head; first equal hash wins. Speakers iterated alphabetically, but
    #    since hashes uniquely identify the content this is deterministic.
    active_hash = _file_head_sha256(active)
    if active_hash is None:
        return None
    training_dir = PROJECT_ROOT / "training"
    if not training_dir.exists():
        return None
    for sp_dir in sorted(p for p in training_dir.iterdir() if p.is_dir()):
        cand = sp_dir / "cfm_reflow_student" / "best.pth"
        if cand.exists() and _file_head_sha256(cand) == active_hash:
            return sp_dir.name
    return None


def _is_active_student_for_speaker(speaker: str) -> bool:
    """Per-speaker version of the active-student check used by /distill/speakers
    and /character/status. Uses the same hash-based identity as the resolver."""
    active = _active_student_path()
    if not active.exists():
        return False
    cand = _student_path(speaker)
    if not cand.exists():
        return False
    # Sidecar shortcut
    meta_path = _active_student_meta_path()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("speaker") == speaker:
                return True
            # If the sidecar claims a DIFFERENT speaker and the hash agrees,
            # trust the sidecar.
            if meta.get("speaker") and meta.get("sha256_head"):
                if _file_head_sha256(active) == meta["sha256_head"]:
                    return False
        except (OSError, json.JSONDecodeError):
            pass
    return _file_head_sha256(active) == _file_head_sha256(cand)


def _count_pair_files(speaker: str) -> int:
    d = _pairs_dir(speaker)
    if not d.exists():
        return 0
    return sum(1 for _ in d.glob("*.pt"))


def _lora_path_for_speaker(speaker: str) -> Optional[Path]:
    """Find an S2Mel LoRA for the speaker, if any. Used by the snapshot-teacher
    flow to bake the trained patterns into the teacher before distillation."""
    base = PROJECT_ROOT / "training" / speaker
    for candidate in (
        base / "s2mel_lora" / "best_checkpoint",
        base / "s2mel_lora" / "final_checkpoint",
        base / "lora",
    ):
        if candidate.exists() and (candidate / "adapter_config.json").exists():
            return candidate
    return None


def _find_gpt_lora_for_speaker(speaker: str) -> Optional[Path]:
    """Locate any GPT-side LoRA for back-compat loading. Use _classify_gpt_lora
    when you need to know whether it's the real (new) character LoRA versus a
    legacy one trained on verbatim text or with the older pattern-combo trainer."""
    info = _classify_gpt_lora(speaker)
    return info["path"] if info else None


def _classify_gpt_lora(speaker: str) -> Optional[Dict[str, Any]]:
    """Classify which kind of GPT LoRA (if any) we have for this speaker.

    Returns {path, kind} where kind is one of:
      * "character"      — trained by tools/train_character_lora.py on CLEAN
                           text with the masked stutter-weighted loss. This is
                           what the new voice-to-voice pipeline expects.
      * "verbatim"       — trained by the older tools/train_verbatim_lora.py on
                           VERBATIM text. Loads fine but expects verbatim input
                           at inference; will under-stutter on clean text.
      * "legacy_pattern" — output of the old pattern+LoRA combo trainer
                           (pattern_training/). Similar caveats to verbatim.

    Order: prefer the strongest "character" hit; otherwise fall through.
    """
    base = PROJECT_ROOT / "training" / speaker
    candidates = [
        ("character",      base / "character_lora" / "best_checkpoint" / "lora"),
        ("character",      base / "character_lora" / "final_checkpoint" / "lora"),
        ("verbatim",       base / "verbatim_training" / "best_checkpoint" / "lora"),
        ("verbatim",       base / "verbatim_training" / "final_checkpoint" / "lora"),
        ("legacy_pattern", base / "pattern_training" / "best_checkpoint" / "lora"),
        ("legacy_pattern", base / "pattern_training" / "final_checkpoint" / "lora"),
    ]
    for kind, cand in candidates:
        if cand.exists() and (cand / "adapter_config.json").exists():
            return {"path": cand, "kind": kind}
    return None


def _character_dirs(speaker: str) -> Dict[str, Path]:
    base = PROJECT_ROOT / "training" / speaker
    return {
        "base": base,
        "audio": base / "dataset" / "audio",
        "transcripts": base / "dataset" / "transcripts_dual.csv",
        "transcripts_verbatim": base / "dataset" / "transcripts_verbatim.csv",
        "transcripts_clean": base / "dataset" / "transcripts.csv",
        "dataset_manifest": base / "character_dataset" / "manifest.jsonl",
        "lora_best": base / "character_lora" / "best_checkpoint" / "lora",
        "lora_final": base / "character_lora" / "final_checkpoint" / "lora",
        "logit_diff": base / "character_lora" / "logit_diff.json",
    }


def _new_distill_task(task_type: str, speaker: str) -> str:
    task_id = f"distill_{task_type}_{speaker}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    distillation_tasks[task_id] = {
        "type": task_type,
        "speaker_name": speaker,
        "status": "queued",
        "progress": 0.0,
        "message": "Queued",
        "log": [],
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "pid": None,
    }
    return task_id


def _run_distill_then(task_id: str, cmd: List[str], on_complete) -> None:
    """Run _run_distill_subprocess and then fire a post-completion callback.

    The callback runs even if the subprocess failed; it can read the task dict
    via `distillation_tasks[task_id]` and update status/log as needed (for
    example to downgrade a 0-exit run to "failed" after a quality check).
    """
    try:
        _run_distill_subprocess(task_id, cmd)
    finally:
        try:
            on_complete()
        except Exception as e:
            print(f"[{task_id}] on_complete crashed: {e}")


def _run_distill_subprocess(
    task_id: str,
    cmd: List[str],
    log_max_lines: int = 500,
    progress_parser=None,
) -> None:
    """Run a tools/*.py subprocess and stream stdout into the task log buffer.

    progress_parser is an optional callable(line: str, task: dict) -> None that
    inspects each log line and may set task['progress'] and task['message'].
    Keeps things flexible per-stage without hardcoding regexes here.
    """
    import subprocess

    task = distillation_tasks[task_id]
    task["status"] = "running"
    task["message"] = "Started"
    task["progress"] = 0.01

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        task["pid"] = process.pid

        for line in process.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            print(f"[{task_id}] {line}", flush=True)
            log = task["log"]
            log.append(line)
            if len(log) > log_max_lines:
                # Drop the oldest entries; keep tail.
                del log[: len(log) - log_max_lines]
            if progress_parser is not None:
                try:
                    progress_parser(line, task)
                except Exception:
                    pass

        process.wait()
        task["pid"] = None

        if process.returncode == 0:
            task["status"] = "completed"
            task["progress"] = 1.0
            task["message"] = "Done"
        else:
            task["status"] = "failed"
            task["message"] = f"Exited with code {process.returncode}"
        task["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        task["status"] = "failed"
        task["message"] = f"Crashed: {e}"
        task["completed_at"] = datetime.now().isoformat()
        task["pid"] = None


# ---- Pydantic models for the endpoints ----

class DistillSnapshotRequest(BaseModel):
    speaker: str
    merge_lora: bool = Field(True, description="Bake the S2Mel LoRA into the teacher (recommended for per-voice student).")
    force: bool = Field(False, description="Overwrite an existing teacher snapshot.")


class DistillManifestRequest(BaseModel):
    speaker: str
    reference_audio: Optional[str] = Field(None, description="Optional path under training/<speaker>/dataset/audio (or absolute).")
    reference_from_row: bool = Field(False, description="If true, each row uses its own audio as reference. Mutually exclusive with reference_audio.")
    n_samples: int = Field(4, ge=1, le=16, description="How many z draws per record to request at pair-gen time.")
    min_duration: float = Field(1.0, ge=0.0)
    max_duration: float = Field(15.0, ge=1.0, le=60.0)
    limit: Optional[int] = Field(None, description="Cap manifest size (smoke test).")


class DistillSyntheticManifestRequest(BaseModel):
    speaker: str
    reference_audio: str = Field(..., description="Path under training/<speaker>/dataset/audio or absolute. Required for synthetic records.")
    style_prompt: str = Field(..., description="Free-text description of how the speaker talks (used as system prompt for Ollama).")
    num_records: int = Field(50, ge=1, le=2000, description="How many text records to generate and append.")
    n_samples: int = Field(4, ge=1, le=16, description="Per-record z draws for downstream pair generation.")
    ollama_url: str = Field("http://localhost:11434", description="Base URL of the Ollama HTTP API.")
    ollama_model: str = Field("llama3.2", description="Ollama model name (must be pulled on the Ollama host).")
    batch_size: int = Field(10, ge=1, le=30, description="How many records to ask Ollama for per call.")
    min_words: int = Field(5, ge=1)
    max_words: int = Field(25, ge=5, le=100)
    extra_instructions: Optional[str] = Field(None, description="Optional appended user instruction (e.g. topic constraints).")


class DistillPairsRequest(BaseModel):
    speaker: str
    teacher_steps: int = Field(50, ge=10, le=200)
    teacher_cfg: float = Field(0.7, ge=0.0, le=2.0)
    n_samples: int = Field(4, ge=1, le=16)
    limit: Optional[int] = Field(None, description="Process only first N manifest records.")


class DistillTrainRequest(BaseModel):
    speaker: str
    epochs: int = Field(40, ge=1, le=400)
    batch_size: int = Field(4, ge=1, le=64)
    grad_accumulation: int = Field(4, ge=1, le=64)
    learning_rate: float = Field(5e-5, gt=0)
    val_split: float = Field(0.02, ge=0.0, le=0.5)
    save_every_epochs: int = Field(2, ge=1, le=50)
    resume: bool = Field(False, description="Resume from this speaker's best.pth if present.")


class DistillAbRequest(BaseModel):
    speaker: str
    text: str
    student_steps: int = Field(1, ge=1, le=20)
    student_solver: str = Field("single_step")
    teacher_steps: int = Field(10, ge=1, le=50)
    teacher_solver: str = Field("heun")
    reference_audio: Optional[str] = None


class DistillActivateRequest(BaseModel):
    speaker: str


# ============= Character LoRA pipeline (new) =============

class CharacterCreateSpeakerRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64,
                      description="Folder-safe speaker name (letters/digits/underscore).")


class CharacterTranscribeRequest(BaseModel):
    speaker: str
    skip_parakeet: bool = Field(False, description="Skip the verbatim ASR — only emit clean Whisper.")
    word_timestamps: bool = Field(True, description="Also run Whisper word-level timestamps and emit a JSONL.")
    force: bool = Field(False, description="Re-transcribe even if outputs already exist.")


class CharacterPrepareRequest(BaseModel):
    speaker: str
    reference_audio: Optional[str] = Field(None,
        description="Optional clip under dataset/audio (or abs path). Enables global conditioning.")
    min_duration: float = Field(1.0, ge=0.0)
    max_duration: float = Field(15.0, ge=1.0, le=60.0)
    pad_tokens: int = Field(8, ge=0, le=64,
        description="Mel-token padding around each detected stutter span.")
    limit: Optional[int] = Field(None, description="Cap N samples for smoke test.")


class CharacterTrainRequest(BaseModel):
    speaker: str
    epochs: int = Field(40, ge=1, le=400)
    batch_size: int = Field(2, ge=1, le=32)
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-2)
    lora_rank: int = Field(32, ge=4, le=128)
    lora_alpha: int = Field(64, ge=4, le=256)
    stutter_weight: float = Field(15.0, ge=1.0, le=100.0,
        description="Per-mel-token weight on stutter-mask positions.")
    overfit_test: bool = Field(False,
        description="Sanity-check mode: 2 samples, 200 epochs. ~3 min on GPU.")
    logit_diff: bool = Field(True,
        description="Dump pre/post-LoRA top-k mel-token probabilities after training.")


def _safe_speaker_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in name).strip("_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="speaker name must contain alphanumeric characters")
    return cleaned


@app.get("/character/status")
async def character_status(speaker: str):
    """One-stop status used by the Speakers WebUI tab."""
    d = _character_dirs(speaker)
    if not d["base"].exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {speaker}")

    audio_files = []
    if d["audio"].exists():
        audio_files = [p.name for p in d["audio"].iterdir()
                       if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
    n_audio = len(audio_files)

    has_clean = d["transcripts_clean"].exists() or d["transcripts"].exists()
    has_verbatim = d["transcripts_verbatim"].exists() or d["transcripts"].exists()
    n_manifest = _count_manifest_entries(d["dataset_manifest"]) if d["dataset_manifest"].exists() else 0

    lora_path = _find_gpt_lora_for_speaker(speaker)
    has_lora = lora_path is not None

    logit_diff = None
    if d["logit_diff"].exists():
        try:
            logit_diff = json.loads(d["logit_diff"].read_text())
        except Exception:
            logit_diff = None

    # Distillation cross-check (matches distill_list_speakers logic)
    teacher = _teacher_path(speaker)
    student = _student_path(speaker)
    is_active_student = _is_active_student_for_speaker(speaker)

    return {
        "speaker": speaker,
        "n_audio_files": n_audio,
        "audio_files_sample": audio_files[:8],
        "has_clean_transcripts": has_clean,
        "has_verbatim_transcripts": has_verbatim,
        "n_manifest_entries": n_manifest,
        "has_character_lora": has_lora,
        "character_lora_path": str(lora_path) if lora_path else None,
        "logit_diff": logit_diff,
        "has_teacher_snapshot": teacher.exists(),
        "has_distilled_student": student.exists(),
        "is_active_student": is_active_student,
    }


@app.post("/character/create-speaker")
async def character_create_speaker(req: CharacterCreateSpeakerRequest):
    name = _safe_speaker_name(req.name)
    d = _character_dirs(name)
    if d["base"].exists():
        raise HTTPException(status_code=400, detail=f"speaker '{name}' already exists")
    d["audio"].mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "speaker": name, "audio_dir": str(d["audio"])}


@app.post("/character/upload-audio")
async def character_upload_audio(
    speaker: str = Form(...),
    audio_files: List[UploadFile] = File(...),
):
    """Persist uploaded audio files into training/<speaker>/dataset/audio/."""
    d = _character_dirs(speaker)
    if not d["base"].exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {speaker}")
    d["audio"].mkdir(parents=True, exist_ok=True)

    saved = []
    for up in audio_files:
        # Strip path components for safety, keep just the basename
        safe = Path(up.filename or "audio.wav").name
        if not safe:
            continue
        # Avoid clobbering: if name exists, suffix with timestamp
        dest = d["audio"] / safe
        if dest.exists():
            stem, suf = dest.stem, dest.suffix
            dest = d["audio"] / f"{stem}_{int(time.time())}{suf}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append(dest.name)
    return {"status": "ok", "saved": saved, "total_in_dir": len(list(d["audio"].iterdir()))}


@app.post("/character/transcribe")
async def character_transcribe(req: CharacterTranscribeRequest, background_tasks: BackgroundTasks):
    """Background-run tools/transcribe_dual.py to produce clean + verbatim transcripts.

    Word-level timestamps are emitted as a sidecar JSONL when requested. The
    new pipeline expects transcripts at dataset/transcripts.csv (clean) and
    dataset/transcripts_verbatim.csv (verbatim). transcribe_dual emits a
    combined CSV with `fastwhisper` and `parakeet` columns, which the prep
    step reads directly — no normalization needed.
    """
    d = _character_dirs(req.speaker)
    if not d["base"].exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")
    if not d["audio"].exists() or not any(d["audio"].iterdir()):
        raise HTTPException(status_code=400, detail="no audio files — upload some first")

    transcripts_dual = d["base"] / "dataset" / "transcripts_dual.csv"
    if transcripts_dual.exists() and not req.force:
        return {"status": "skipped", "reason": "transcripts already exist; pass force=true to re-run",
                "path": str(transcripts_dual)}

    task_id = _new_distill_task("character_transcribe", req.speaker)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "transcribe_dual.py"),
        "--audio-dir", str(d["audio"]),
        "--output-csv", str(transcripts_dual),
        # Fail loudly if Parakeet can't load, so we never silently ship an empty
        # verbatim column again (this is the bug that hit ozzyv6).
        "--strict-parakeet",
    ]

    def _verify_columns(line: str, task: Dict[str, Any]):
        # The subprocess's own [Coverage] report carries the truth — but the
        # background_tasks runner already streams it into the log. We re-scan
        # the CSV on completion below; nothing to do here.
        pass

    def _on_complete():
        # Re-scan the output CSV. If a column is entirely empty, downgrade the
        # task to "failed" with a clear, UI-visible message even if the
        # subprocess returned 0.
        try:
            if not transcripts_dual.exists():
                return
            import csv as _csv
            with open(transcripts_dual, encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            n = len(rows)
            if n == 0:
                return
            empty = {
                col: sum(1 for r in rows if not (r.get(col) or "").strip())
                for col in ("fastwhisper", "parakeet")
            }
            t = distillation_tasks.get(task_id)
            if t is None:
                return
            problems = [col for col, k in empty.items() if k == n]
            if problems:
                t["status"] = "failed"
                t["message"] = (
                    f"transcribe completed but column(s) {problems} are entirely empty. "
                    f"Check the log tail for load errors."
                )
                t.setdefault("log", []).append(
                    f"[POST-CHECK] all {n} rows have empty {problems} — task downgraded to failed."
                )
            else:
                # success summary into the log so the UI can show useful stats
                t.setdefault("log", []).append(
                    f"[POST-CHECK] {n} rows  fastwhisper-empty={empty['fastwhisper']}  "
                    f"parakeet-empty={empty['parakeet']}"
                )
        except Exception as e:
            print(f"[{task_id}] post-check error: {e}")

    background_tasks.add_task(_run_distill_then, task_id, cmd, _on_complete)
    return {"task_id": task_id, "status": "queued", "output": str(transcripts_dual)}


@app.post("/character/prepare-dataset")
async def character_prepare_dataset(req: CharacterPrepareRequest, background_tasks: BackgroundTasks):
    d = _character_dirs(req.speaker)
    if not d["base"].exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "prepare_character_dataset.py"),
        "--speaker", req.speaker,
        "--min-duration", str(req.min_duration),
        "--max-duration", str(req.max_duration),
        "--pad-tokens", str(req.pad_tokens),
    ]
    if req.reference_audio:
        ref = Path(req.reference_audio)
        if not ref.is_absolute():
            ref = d["audio"] / req.reference_audio
        if not ref.exists():
            raise HTTPException(status_code=404, detail=f"reference audio not found: {ref}")
        cmd.extend(["--reference-audio", str(ref)])
    if req.limit is not None:
        cmd.extend(["--limit", str(req.limit)])

    task_id = _new_distill_task("character_prepare", req.speaker)
    background_tasks.add_task(_run_distill_subprocess, task_id, cmd)
    return {"task_id": task_id, "status": "queued"}


@app.post("/character/train")
async def character_train(req: CharacterTrainRequest, background_tasks: BackgroundTasks):
    d = _character_dirs(req.speaker)
    if not d["dataset_manifest"].exists():
        raise HTTPException(status_code=400, detail="no character dataset manifest — run prepare-dataset first")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "train_character_lora.py"),
        "--speaker", req.speaker,
        "--epochs", str(req.epochs),
        "--batch-size", str(req.batch_size),
        "--learning-rate", str(req.learning_rate),
        "--lora-rank", str(req.lora_rank),
        "--lora-alpha", str(req.lora_alpha),
        "--stutter-weight", str(req.stutter_weight),
    ]
    if req.overfit_test:
        cmd.append("--overfit-test")
    if req.logit_diff:
        cmd.append("--logit-diff")

    task_type = "character_overfit" if req.overfit_test else "character_train"
    task_id = _new_distill_task(task_type, req.speaker)

    def _parse_progress(line: str, task: Dict[str, Any]):
        # match "epoch  NN/MM  loss=..." style emitted by the trainer
        import re as _re
        m = _re.search(r"epoch\s+(\d+)/(\d+)", line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            task["progress"] = min(0.99, cur / max(total, 1))
            task["message"] = f"Epoch {cur}/{total}"

    background_tasks.add_task(_run_distill_subprocess, task_id, cmd, 1000, _parse_progress)
    return {"task_id": task_id, "status": "queued"}


@app.get("/character/diagnose")
async def character_diagnose(speaker: str):
    """Return the logit_diff.json the trainer emits (if --logit-diff was used)."""
    d = _character_dirs(speaker)
    if not d["logit_diff"].exists():
        raise HTTPException(status_code=404, detail="no logit_diff.json — train with --logit-diff first")
    return json.loads(d["logit_diff"].read_text())


# ---- Endpoints ----


@app.get("/distill/speakers")
async def distill_list_speakers():
    """List speakers + pipeline state for each stage. Drives the UI."""
    training_dir = PROJECT_ROOT / "training"
    out: List[Dict[str, Any]] = []
    active_student = _active_student_path()
    active_exists = active_student.exists()
    active_size = active_student.stat().st_size if active_exists else 0

    if training_dir.exists():
        for sp in sorted(p for p in training_dir.iterdir() if p.is_dir()):
            name = sp.name
            teacher = _teacher_path(name)
            manifest = _manifest_path(name)
            student = _student_path(name)
            # Two different LoRAs — both matter to the Distill flow but for
            # different reasons:
            #   * Character LoRA (GPT-side) — auto-loaded at pair-gen so the
            #     teacher emits stuttered tokens. The thing the user trains via
            #     the Speakers tab.
            #   * S2Mel LoRA (CFM-side) — optional "Merge LoRA" toggle on the
            #     Snapshot card. Rarely used in this repo.
            gpt_lora_info = _classify_gpt_lora(name)
            gpt_lora_kind = gpt_lora_info["kind"] if gpt_lora_info else None
            gpt_lora_path_value = str(gpt_lora_info["path"]) if gpt_lora_info else None
            s2mel_lora_path = _lora_path_for_speaker(name)

            is_active = _is_active_student_for_speaker(name)

            out.append({
                "name": name,
                # Strict: only the new clean-text trainer's output is a real
                # character LoRA. Legacy verbatim/pattern dirs still load, but
                # they expect a different input distribution at inference.
                "has_character_lora": gpt_lora_kind == "character",
                "character_lora_path": gpt_lora_path_value if gpt_lora_kind == "character" else None,
                "gpt_lora_kind": gpt_lora_kind,
                "gpt_lora_path": gpt_lora_path_value,
                "has_s2mel_lora": s2mel_lora_path is not None,
                "s2mel_lora_path": str(s2mel_lora_path) if s2mel_lora_path else None,
                # Back-compat: `has_lora` used to mean "any LoRA". Keep it
                # mirroring has_character_lora so old UI code reads correctly.
                "has_lora": gpt_lora_kind == "character",
                "has_csv": (sp / "dataset" / "transcripts_verbatim.csv").exists(),
                "has_teacher": teacher.exists(),
                "teacher_path": str(teacher) if teacher.exists() else None,
                "has_manifest": manifest.exists(),
                "manifest_entries": _count_manifest_entries(manifest) if manifest.exists() else 0,
                "pair_count": _count_pair_files(name),
                "has_student": student.exists(),
                "student_path": str(student) if student.exists() else None,
                "is_active_student": is_active,
            })
    return {
        "speakers": out,
        "active_student_path": str(active_student) if active_exists else None,
        "active_student_size_mb": round(active_size / 1024 / 1024, 1) if active_exists else 0,
    }


def _count_manifest_entries(path: Path) -> int:
    try:
        with open(path) as fp:
            return sum(1 for line in fp if line.strip())
    except Exception:
        return 0


@app.get("/distill/list-audio")
async def distill_list_audio(speaker: str):
    """List audio files under training/<speaker>/dataset/audio/ so the UI can
    offer a clickable dropdown of available reference clips."""
    audio_dir = PROJECT_ROOT / "training" / speaker / "dataset" / "audio"
    if not audio_dir.exists():
        return {"speaker": speaker, "audio_dir": str(audio_dir), "files": []}
    files = sorted(
        p.name for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
    )
    return {"speaker": speaker, "audio_dir": str(audio_dir), "files": files}


@app.post("/distill/snapshot-teacher")
async def distill_snapshot_teacher(req: DistillSnapshotRequest, background_tasks: BackgroundTasks):
    speaker_dir = PROJECT_ROOT / "training" / req.speaker
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")

    dest = _teacher_path(req.speaker)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "snapshot_cfm_teacher.py"),
        "--source", str(PROJECT_ROOT / "checkpoints" / "s2mel.pth"),
        "--dest", str(dest),
    ]
    if req.force:
        cmd.append("--force")
    if req.merge_lora:
        lora_path = _lora_path_for_speaker(req.speaker)
        if lora_path is None:
            # Distinguish "they have a character LoRA but not an S2Mel one" from
            # "they have nothing at all" — the user shouldn't be told to train
            # something they don't need.
            has_gpt_lora = _find_gpt_lora_for_speaker(req.speaker) is not None
            if has_gpt_lora:
                detail = (
                    f"merge_lora=True but no S2Mel (CFM-side) LoRA found for {req.speaker}. "
                    f"Note: this is a different LoRA from the character LoRA you trained on "
                    f"the Speakers tab. The character LoRA targets the GPT (which mel tokens "
                    f"to emit) — it's loaded automatically at pair-generation time and does "
                    f"NOT need to be merged into the teacher snapshot. "
                    f"For most workflows: leave 'Merge LoRA' OFF and let the pair-gen step "
                    f"auto-load your character LoRA."
                )
            else:
                detail = (
                    f"merge_lora=True but no S2Mel LoRA found for {req.speaker}. "
                    f"S2Mel LoRA training is rarely needed — leave 'Merge LoRA' OFF unless "
                    f"you've specifically trained one with tools/train_s2mel_lora.py."
                )
            raise HTTPException(status_code=400, detail=detail)
        cmd.extend(["--merge-lora", str(lora_path)])

    task_id = _new_distill_task("snapshot", req.speaker)
    background_tasks.add_task(_run_distill_subprocess, task_id, cmd)
    return {"task_id": task_id, "status": "queued", "command": cmd}


@app.post("/distill/build-manifest")
async def distill_build_manifest(req: DistillManifestRequest):
    """Build the JSONL manifest. Synchronous — this is fast (CPU-bound, no model)."""
    speaker_dir = PROJECT_ROOT / "training" / req.speaker
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "build_reflow_manifest.py"),
        "--speaker", req.speaker,
        "--output", str(_manifest_path(req.speaker)),
        "--n-samples", str(req.n_samples),
        "--min-duration", str(req.min_duration),
        "--max-duration", str(req.max_duration),
    ]
    if req.reference_audio and req.reference_from_row:
        raise HTTPException(status_code=400, detail="reference_audio and reference_from_row are mutually exclusive")
    if req.reference_audio:
        ref = Path(req.reference_audio)
        if not ref.is_absolute():
            ref = speaker_dir / "dataset" / "audio" / req.reference_audio
        if not ref.exists():
            raise HTTPException(status_code=404, detail=f"reference audio not found: {ref}")
        cmd.extend(["--reference-audio", str(ref)])
    if req.reference_from_row:
        cmd.append("--reference-from-row")
    if req.limit:
        cmd.extend(["--limit", str(req.limit)])

    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"build-manifest failed:\n{result.stdout}\n{result.stderr}")
    return {
        "status": "ok",
        "manifest": str(_manifest_path(req.speaker)),
        "entries": _count_manifest_entries(_manifest_path(req.speaker)),
        "stdout": result.stdout,
    }


def _ollama_generate_batch(
    url: str,
    model: str,
    style_prompt: str,
    n: int,
    min_words: int,
    max_words: int,
    extra: Optional[str],
    timeout: float = 120.0,
) -> List[str]:
    """Call Ollama's /api/generate with format=json and parse a JSON array of strings.

    We request structured output (Ollama's `format: "json"` forces valid JSON) and
    explicitly ask the model for `{"utterances": [...]}` because that shape is more
    reliable than a top-level array across model families. We then extract the array.
    Any parsing weirdness raises so the caller can fall back / log it.
    """
    import urllib.request
    import urllib.error

    schema_hint = (
        'Return ONLY a JSON object of the form '
        '{"utterances": ["…", "…"]}.\n'
        'No prose, no markdown, no code fences. The array MUST contain exactly '
        f'{n} strings.'
    )
    user_prompt = (
        f"Speaker style: {style_prompt}\n\n"
        f"Write {n} different short utterances (each {min_words}-{max_words} words) "
        f"that sound like something this speaker would naturally say. "
        f"Vary the topic and emotional register across the {n} entries — do not "
        f"repeat themes. Do NOT include speaker labels, quote marks, or stage "
        f"directions. Just the spoken text.\n"
    )
    if extra:
        user_prompt += f"\nAdditional constraints: {extra}\n"
    user_prompt += "\n" + schema_hint

    body = json.dumps({
        "model": model,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.9, "top_p": 0.95},
    }).encode("utf-8")

    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}") from e

    outer = json.loads(raw)
    response_text = outer.get("response", "").strip()
    if not response_text:
        raise RuntimeError(f"Ollama returned empty response. Full reply: {outer}")

    # Strip code fences if the model added them despite format=json
    if response_text.startswith("```"):
        response_text = response_text.strip("`")
        if response_text.lower().startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    parsed = json.loads(response_text)
    # Accept either a top-level array or {"utterances": [...]}
    if isinstance(parsed, dict):
        for key in ("utterances", "items", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Last-ditch: take the first list value
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
    if not isinstance(parsed, list):
        raise RuntimeError(f"Ollama response wasn't a list of strings: {response_text[:200]}")

    out = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip().strip('"').strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            # Some models wrap each string in {"text": "..."} — accept that.
            for k in ("text", "utterance", "content"):
                if k in item and isinstance(item[k], str):
                    out.append(item[k].strip())
                    break
    return out


def _run_synthetic_manifest(
    task_id: str,
    speaker: str,
    reference_audio: str,
    style_prompt: str,
    num_records: int,
    n_samples: int,
    ollama_url: str,
    ollama_model: str,
    batch_size: int,
    min_words: int,
    max_words: int,
    extra_instructions: Optional[str],
):
    """Generate `num_records` synthetic text records via Ollama and append to the manifest.

    Resilient: any failed Ollama batch is logged and skipped; we keep trying until
    we either hit the target count or run out of attempts (max 3× the target).
    """
    task = distillation_tasks[task_id]
    task["status"] = "running"
    task["message"] = "Calling Ollama…"
    task["progress"] = 0.01

    manifest_path = _manifest_path(speaker)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Stable unique-ID prefix so re-running doesn't collide with previous calls.
    id_prefix = f"syn_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    log = task["log"]

    written = 0
    attempts = 0
    max_attempts = max(3, (num_records // max(1, batch_size)) * 3)

    with open(manifest_path, "a", encoding="utf-8") as fp:
        while written < num_records and attempts < max_attempts:
            attempts += 1
            n_this = min(batch_size, num_records - written)
            try:
                lines = _ollama_generate_batch(
                    ollama_url, ollama_model, style_prompt,
                    n_this, min_words, max_words, extra_instructions,
                )
            except Exception as e:
                msg = f"[attempt {attempts}] Ollama batch failed: {e}"
                log.append(msg)
                print(f"[{task_id}] {msg}", flush=True)
                continue

            n_added = 0
            for utterance in lines:
                if not utterance:
                    continue
                rec = {
                    "id": f"{id_prefix}_{written + n_added:05d}",
                    "audio_prompt": reference_audio,
                    "text": utterance,
                    "n_samples": n_samples,
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_added += 1
                if written + n_added >= num_records:
                    break
            fp.flush()
            written += n_added

            task["progress"] = min(0.99, written / max(1, num_records))
            task["message"] = f"Generated {written}/{num_records} ({attempts} batches)"
            log_line = f"[attempt {attempts}] +{n_added} → total {written}/{num_records}"
            log.append(log_line)
            print(f"[{task_id}] {log_line}", flush=True)
            if len(log) > 500:
                del log[: len(log) - 500]

    if written >= num_records:
        task["status"] = "completed"
        task["progress"] = 1.0
        task["message"] = f"Appended {written} synthetic records to manifest"
    else:
        task["status"] = "failed"
        task["message"] = f"Stopped after {attempts} attempts — wrote {written}/{num_records}"
    task["completed_at"] = datetime.now().isoformat()


@app.post("/distill/append-synthetic-manifest")
async def distill_append_synthetic_manifest(
    req: DistillSyntheticManifestRequest,
    background_tasks: BackgroundTasks,
):
    """Use Ollama to generate synthetic text records and append them to the speaker's
    reflow manifest, all pointing at one fixed reference audio. The speaker's voice
    is already baked into the teacher (LoRA merged), so what we need from the manifest
    is *text diversity* — exactly what Ollama provides."""
    speaker_dir = PROJECT_ROOT / "training" / req.speaker
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")

    # Resolve reference audio to an absolute path now so the manifest is portable.
    ref = Path(req.reference_audio)
    if not ref.is_absolute():
        ref = speaker_dir / "dataset" / "audio" / req.reference_audio
    if not ref.exists():
        raise HTTPException(status_code=404, detail=f"reference audio not found: {ref}")

    task_id = _new_distill_task("synthetic_manifest", req.speaker)
    background_tasks.add_task(
        _run_synthetic_manifest,
        task_id,
        req.speaker,
        str(ref.resolve()),
        req.style_prompt,
        req.num_records,
        req.n_samples,
        req.ollama_url,
        req.ollama_model,
        req.batch_size,
        req.min_words,
        req.max_words,
        req.extra_instructions,
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/distill/ollama-models")
async def distill_list_ollama_models(url: str = "http://localhost:11434"):
    """Probe an Ollama instance for installed models. Used by the UI to populate
    the model dropdown without forcing the user to type model names by hand."""
    import urllib.request
    try:
        req = urllib.request.Request(url.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        return {"ok": True, "url": url, "models": models}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e), "models": []}


@app.post("/distill/generate-pairs")
async def distill_generate_pairs(req: DistillPairsRequest, background_tasks: BackgroundTasks):
    speaker_dir = PROJECT_ROOT / "training" / req.speaker
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")

    teacher = _teacher_path(req.speaker)
    if not teacher.exists():
        # Fall back to the plain default; UI is encouraged to make the per-voice one explicit.
        teacher = PROJECT_ROOT / "checkpoints" / "s2mel_teacher.pth"
    if not teacher.exists():
        raise HTTPException(status_code=400, detail="no teacher snapshot — run snapshot first.")

    manifest = _manifest_path(req.speaker)
    if not manifest.exists():
        raise HTTPException(status_code=400, detail="no manifest — run build-manifest first.")

    out_dir = _pairs_dir(req.speaker)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "generate_reflow_pairs.py"),
        "--manifest", str(manifest),
        "--teacher", str(teacher),
        "--output-dir", str(out_dir),
        "--n-samples", str(req.n_samples),
        "--teacher-steps", str(req.teacher_steps),
        "--teacher-cfg", str(req.teacher_cfg),
    ]
    # Auto-load only the real (new) character LoRA. Legacy verbatim/pattern
    # adapters were trained on a different input distribution (verbatim text)
    # and would inject wrong behavior into the teacher during pair-gen on
    # clean manifest text — better to run without a LoRA than with the wrong one.
    gpt_lora_info = _classify_gpt_lora(req.speaker)
    if gpt_lora_info and gpt_lora_info["kind"] == "character":
        cmd.extend(["--gpt-lora", str(gpt_lora_info["path"])])
    if req.limit:
        cmd.extend(["--limit", str(req.limit)])

    def _parser(line: str, task: Dict[str, Any]):
        # Lines look like: "[3/120] utt_001: pairs=10 skipped=0 errors=0 rate=2.1/s eta=..."
        import re
        m = re.match(r"^\[(\d+)/(\d+)\]", line)
        if m:
            done = int(m.group(1))
            total = int(m.group(2))
            task["progress"] = min(0.99, done / max(1, total))
            task["message"] = f"Records {done}/{total}"

    task_id = _new_distill_task("pairs", req.speaker)
    background_tasks.add_task(_run_distill_subprocess, task_id, cmd, 800, _parser)
    return {"task_id": task_id, "status": "queued"}


@app.post("/distill/train")
async def distill_train(req: DistillTrainRequest, background_tasks: BackgroundTasks):
    speaker_dir = PROJECT_ROOT / "training" / req.speaker
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"speaker not found: {req.speaker}")

    pairs_dir = _pairs_dir(req.speaker)
    if not pairs_dir.exists() or _count_pair_files(req.speaker) == 0:
        raise HTTPException(status_code=400, detail="no paired data — run generate-pairs first.")

    out_dir = speaker_dir / "cfm_reflow_student"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "train_cfm_reflow.py"),
        "--pairs-dir", str(pairs_dir),
        "--output-dir", str(out_dir),
        "--epochs", str(req.epochs),
        "--batch-size", str(req.batch_size),
        "--grad-accumulation", str(req.grad_accumulation),
        "--learning-rate", str(req.learning_rate),
        "--val-split", str(req.val_split),
        "--save-every-epochs", str(req.save_every_epochs),
    ]
    if req.resume and (out_dir / "best.pth").exists():
        cmd.extend(["--resume", str(out_dir / "best.pth")])

    def _parser(line: str, task: Dict[str, Any]):
        import re
        m = re.match(r"^== epoch (\d+)/(\d+)\s+train=([\d.]+)\s+val=([\d.]+)", line)
        if m:
            ep = int(m.group(1))
            total = int(m.group(2))
            task["progress"] = min(0.99, ep / max(1, total))
            task["message"] = f"Epoch {ep}/{total}  train={m.group(3)} val={m.group(4)}"

    task_id = _new_distill_task("train", req.speaker)
    background_tasks.add_task(_run_distill_subprocess, task_id, cmd, 800, _parser)
    return {"task_id": task_id, "status": "queued"}


@app.post("/distill/ab-eval")
async def distill_ab_eval(req: DistillAbRequest, background_tasks: BackgroundTasks):
    student = _student_path(req.speaker)
    if not student.exists():
        raise HTTPException(status_code=400, detail=f"no student checkpoint for {req.speaker}")

    speaker_dir = PROJECT_ROOT / "training" / req.speaker
    if req.reference_audio:
        ref = Path(req.reference_audio)
        if not ref.is_absolute():
            ref = speaker_dir / "dataset" / "audio" / req.reference_audio
    else:
        # Use the first audio file in the dataset as a reasonable default.
        audio_dir = speaker_dir / "dataset" / "audio"
        wavs = sorted(audio_dir.glob("*.wav")) if audio_dir.exists() else []
        if not wavs:
            raise HTTPException(status_code=400, detail="no reference audio found and none provided")
        ref = wavs[0]
    if not ref.exists():
        raise HTTPException(status_code=404, detail=f"reference audio not found: {ref}")

    out_dir = speaker_dir / "ab_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "ab_eval_cfm.py"),
        "--audio-prompt", str(ref),
        "--text", req.text,
        "--student-checkpoint", str(student),
        "--output-dir", str(out_dir),
        "--student-steps", str(req.student_steps),
        "--student-solver", req.student_solver,
        "--teacher-steps", str(req.teacher_steps),
        "--teacher-solver", req.teacher_solver,
    ]

    task_id = _new_distill_task("ab_eval", req.speaker)

    def _ab_runner():
        _run_distill_subprocess(task_id, cmd)
        # On success, expose the wav paths so the UI can fetch them.
        if distillation_tasks[task_id]["status"] == "completed":
            distillation_tasks[task_id]["result"] = {
                "teacher_wav": f"/distill/ab-audio/{req.speaker}/{out_dir.name}/teacher.wav",
                "student_wav": f"/distill/ab-audio/{req.speaker}/{out_dir.name}/student.wav",
            }

    background_tasks.add_task(_ab_runner)
    return {"task_id": task_id, "status": "queued"}


@app.get("/distill/ab-audio/{speaker}/{run_id}/{filename}")
async def distill_ab_audio(speaker: str, run_id: str, filename: str):
    if filename not in ("teacher.wav", "student.wav"):
        raise HTTPException(status_code=400, detail="filename must be teacher.wav or student.wav")
    path = PROJECT_ROOT / "training" / speaker / "ab_results" / run_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=str(path))
    return FileResponse(str(path), media_type="audio/wav")


@app.post("/distill/activate")
async def distill_activate(req: DistillActivateRequest):
    student = _student_path(req.speaker)
    if not student.exists():
        raise HTTPException(status_code=400, detail=f"no student checkpoint for {req.speaker}")

    active = _active_student_path()
    # Use copy (not symlink) — IndexTTS2 reads via torch.load on a path; a symlink
    # would work too, but a real file is more obvious in the checkpoints dir.
    shutil.copy2(student, active)

    # Sidecar so we don't have to guess "which speaker is active?" later. All
    # students share the same architecture and therefore byte size, so size
    # equality is unreliable; the head-hash here is the authoritative tag.
    meta_path = _active_student_meta_path()
    meta_path.write_text(json.dumps({
        "speaker": req.speaker,
        "source_path": str(student),
        "activated_at": datetime.now().isoformat(),
        "sha256_head": _file_head_sha256(active),
    }, indent=2))
    return {
        "status": "ok",
        "active_path": str(active),
        "speaker": req.speaker,
        "note": "Reload the base model (/models/load/base) for the new student to take effect.",
    }


@app.get("/distill/active-status")
async def distill_active_status():
    """Lightweight status for the Inference tab. Tells the UI:
      - whether a distilled student is on disk (would-be-active after model reload)
      - whether the currently-loaded model is using it (i.e. was loaded after activation)
      - which speaker's checkpoint it matches, if any
    The Inference tab uses this to surface a banner and to suggest the right solver/steps.
    """
    active_path = _active_student_path()
    on_disk = active_path.exists()
    on_disk_size_mb = round(active_path.stat().st_size / 1024 / 1024, 1) if on_disk else 0

    # Resolve which speaker owns this active checkpoint via the sidecar manifest
    # (preferred) with a head-hash fallback. The old byte-size heuristic returned
    # the alphabetically-first speaker since CFM students share architecture and
    # therefore byte size.
    speaker_match = _resolve_active_student_speaker() if on_disk else None

    # Is the LOADED model actually using the distilled checkpoint? IndexTTS2 stores the
    # path it was constructed with on self.s2mel_distilled_checkpoint.
    model_loaded = tts_model is not None
    loaded_distilled_path = None
    if model_loaded:
        loaded_distilled_path = getattr(tts_model, "s2mel_distilled_checkpoint", None)
    in_use = model_loaded and bool(loaded_distilled_path)
    needs_reload = on_disk and (
        not in_use or
        (loaded_distilled_path and Path(loaded_distilled_path).resolve() != active_path.resolve())
    )

    return {
        "active_on_disk": on_disk,
        "active_path": str(active_path) if on_disk else None,
        "active_size_mb": on_disk_size_mb,
        "speaker_match": speaker_match,
        "model_loaded": model_loaded,
        "in_use": in_use,
        "loaded_distilled_path": loaded_distilled_path,
        "needs_reload": needs_reload,
        # Suggested solver / steps when distilled is in use — the Inference tab
        # surfaces these as a one-click "set defaults" button.
        "suggested": {
            "solver": "single_step" if in_use else "heun",
            "diffusion_steps": 1 if in_use else None,  # null = leave preset alone
            "inference_cfg": 0.0 if in_use else None,
        },
    }


@app.post("/distill/deactivate")
async def distill_deactivate():
    active = _active_student_path()
    meta = _active_student_meta_path()
    removed = []
    if active.exists():
        active.unlink()
        removed.append(str(active))
    if meta.exists():
        meta.unlink()
        removed.append(str(meta))
    if removed:
        return {"status": "ok", "removed": removed}
    return {"status": "ok", "removed": None, "note": "no active student to remove"}


@app.get("/distill/tasks/{task_id}")
async def distill_get_task(task_id: str, log_lines: int = 50):
    if task_id not in distillation_tasks:
        raise HTTPException(status_code=404, detail="task not found")
    task = distillation_tasks[task_id]
    log = task.get("log") or []
    return {
        "task_id": task_id,
        "type": task["type"],
        "speaker_name": task["speaker_name"],
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "started_at": task["started_at"],
        "completed_at": task.get("completed_at"),
        "result": task.get("result"),
        "log_tail": log[-log_lines:] if log_lines > 0 else [],
        "log_total_lines": len(log),
    }


@app.get("/distill/tasks")
async def distill_list_tasks():
    return [
        {
            "task_id": tid,
            "type": t["type"],
            "speaker_name": t["speaker_name"],
            "status": t["status"],
            "progress": t["progress"],
            "message": t["message"],
            "started_at": t["started_at"],
            "completed_at": t.get("completed_at"),
        }
        for tid, t in distillation_tasks.items()
    ]


@app.post("/distill/tasks/{task_id}/cancel")
async def distill_cancel_task(task_id: str):
    if task_id not in distillation_tasks:
        raise HTTPException(status_code=404, detail="task not found")
    task = distillation_tasks[task_id]
    pid = task.get("pid")
    if not pid or task["status"] not in ("running", "queued"):
        raise HTTPException(status_code=400, detail=f"task not running (status={task['status']})")
    try:
        os.kill(int(pid), 15)  # SIGTERM
        task["status"] = "cancelled"
        task["message"] = "Cancelled by user"
        task["completed_at"] = datetime.now().isoformat()
    except ProcessLookupError:
        task["status"] = "cancelled"
        task["message"] = "Process already gone"
    return {"status": "ok"}


# ============= GPU memory management =============


@app.get("/system/gpu-stats")
async def gpu_stats():
    """Return PyTorch + driver-level GPU memory usage. Driver-level numbers come from
    nvidia-smi-style counters (allocated vs reserved); the gap is the caching
    allocator's free-block pool — that's what /system/gpu-cleanup releases."""
    if not torch.cuda.is_available():
        return {"available": False}
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "available": True,
        "device": dev,
        "device_name": props.name,
        "total_mb": round(total / 1024 / 1024, 1),
        "free_mb": round(free / 1024 / 1024, 1),
        "used_by_other_mb": round((total - free - reserved) / 1024 / 1024, 1),
        "torch_allocated_mb": round(allocated / 1024 / 1024, 1),
        "torch_reserved_mb": round(reserved / 1024 / 1024, 1),
        "torch_cache_mb": round((reserved - allocated) / 1024 / 1024, 1),
    }


@app.post("/system/gpu-cleanup")
async def gpu_cleanup():
    """Force PyTorch to release its CUDA caching-allocator blocks back to the OS.
    Cheap to call; the next allocation pays a small one-time cost. Useful when the
    GPU footprint has crept up and you want to see how much was actually live."""
    if not torch.cuda.is_available():
        return {"ok": False, "reason": "CUDA not available"}
    import gc
    before_reserved = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_reserved = torch.cuda.memory_reserved()
    return {
        "ok": True,
        "freed_mb": round((before_reserved - after_reserved) / 1024 / 1024, 1),
        "reserved_before_mb": round(before_reserved / 1024 / 1024, 1),
        "reserved_after_mb": round(after_reserved / 1024 / 1024, 1),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)