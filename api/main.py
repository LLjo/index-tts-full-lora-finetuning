"""
IndexTTS FastAPI Server
Provides REST API endpoints for inference, streaming, training, and model management
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
from tools.infer_with_patterns import pattern_aware_inference

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
            
            # Check for LoRA
            lora_path = speaker_dir / "pattern_training" / "best_checkpoint" / "lora"
            if lora_path.exists() and (lora_path / "adapter_config.json").exists():
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
        
        try:
            print("🚀 Loading IndexTTS2 base model...")
            tts_model = IndexTTS2(
                model_dir=str(model_dir),
                use_fp16=torch.cuda.is_available(),
                use_cuda_kernel=torch.cuda.is_available(),
            )
            loaded_models.clear()
            print("✅ IndexTTS2 base model loaded successfully")
            return {"status": "success", "message": "Base model loaded successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load base model: {str(e)}")
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Base TTS model not loaded. Load base model first.")
    
    speaker_dir = PROJECT_ROOT / "training" / speaker_name
    if not speaker_dir.exists():
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_name}' not found")
    
    # Find LoRA path
    lora_path = speaker_dir / "pattern_training" / "best_checkpoint" / "lora"
    if not lora_path.exists():
        lora_path = speaker_dir / "pattern_training" / "final_checkpoint" / "lora"
    
    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        try:
            tts_model.load_lora(str(lora_path))
            loaded_models[speaker_name] = {"lora_path": str(lora_path)}
            return {"status": "success", "message": f"Loaded LoRA for {speaker_name}"}
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


@app.post("/inference/stream")
async def stream_speech(
    audio_file: Optional[UploadFile] = File(None, description="Speaker reference audio (optional when using patterns)"),
    request_json: str = Form(..., description="JSON request parameters")
):
    """Stream speech generation"""
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
        """Generator for streaming audio chunks"""
        try:
            if request.use_patterns and request.speaker:
                # Pattern-aware streaming
                speaker_dir = PROJECT_ROOT / "training" / request.speaker
                pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
                
                if not pattern_path.exists():
                    pattern_path = speaker_dir / "pattern_training" / "final_checkpoint" / "pattern_embedding.pt"
                
                if not pattern_path.exists():
                    yield b"Error: Pattern embedding not found"
                    return
                
                # Load pattern embedding
                pattern_embedding = PatternEmbedding.load(pattern_path, device=tts_model.device)
                
                # Load speaker embeddings if no audio prompt
                speaker_embeddings = None
                if tmp_audio_path is None:
                    from indextts.speaker_embeddings import SpeakerEmbeddingStore
                    
                    speaker_emb_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
                    if not speaker_emb_path.exists():
                        yield b"Error: Speaker embeddings not found. Either provide audio_file or train embeddings first."
                        return
                    
                    store = SpeakerEmbeddingStore(tts_model)
                    speaker_embeddings = store.load_embeddings(speaker_emb_path)
                
                # Stream generation
                audio_gen = pattern_aware_inference(
                    tts=tts_model,
                    pattern_embedding=pattern_embedding,
                    text=request.text,
                    output_path=None,
                    audio_prompt=tmp_audio_path,
                    speaker_embeddings=speaker_embeddings,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stream_return=True
                )
                
                for chunk in audio_gen:
                    if chunk is not None:
                        # Convert tensor to bytes
                        chunk_bytes = chunk.numpy().tobytes()
                        yield chunk_bytes
            else:
                # Standard streaming
                audio_gen = tts_model.infer(
                    spk_audio_prompt=tmp_audio_path,
                    text=request.text,
                    output_path=None,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stream_return=True
                )
                
                for chunk in audio_gen:
                    if chunk is not None:
                        chunk_bytes = chunk.numpy().tobytes()
                        yield chunk_bytes
        
        finally:
            # Cleanup
            if tmp_audio_path is not None:
                try:
                    os.unlink(tmp_audio_path)
                except:
                    pass
    
    return StreamingResponse(generate_chunks(), media_type="audio/wav")


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
    """List all available speakers"""
    speakers = []
    training_dir = PROJECT_ROOT / "training"
    
    if training_dir.exists():
        for speaker_dir in training_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            # Check for embeddings
            embeddings_path = speaker_dir / "embeddings" / "speaker_embeddings.pt"
            pattern_path = speaker_dir / "pattern_training" / "best_checkpoint" / "pattern_embedding.pt"
            
            speakers.append({
                "name": speaker_dir.name,
                "has_embeddings": embeddings_path.exists(),
                "has_patterns": pattern_path.exists(),
                "path": str(speaker_dir)
            })
    
    return speakers


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)