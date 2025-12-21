# IndexTTS API & WebUI - Quick Start Guide

🎉 **Your complete API and WebUI system is ready!**

## 📁 What's Been Created

### API Backend (`api/`)
- **`main.py`** - FastAPI server with all endpoints
- **`requirements.txt`** - API dependencies
- **`README.md`** - Complete API documentation

### WebUI (`api/static/`)
- **`index.html`** - Modern, responsive web interface
- **`styles.css`** - Beautiful styling
- **`app.js`** - Full-featured JavaScript client

### Launch Scripts
- **`start_api.sh`** - Linux/Mac launcher
- **`start_api.bat`** - Windows launcher

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r api/requirements.txt
```

### 2. Start the Server
**Linux/Mac:**
```bash
chmod +x start_api.sh
./start_api.sh
```

**Windows:**
```cmd
start_api.bat
```

**Or manually:**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open Your Browser
Navigate to: **http://localhost:8000**

## 🎯 Key Features

### ✅ Inference
- Standard TTS generation
- Pattern-aware inference (with trained speakers)
- Emotion control (vector or text-based)
- Real-time streaming (experimental)

### ✅ Training
- Upload 5+ audio files
- Automated pipeline (transcription → conditioning → pattern training)
- Real-time progress monitoring
- Automatic LoRA + pattern embedding creation

### ✅ Model Management
- List all available models
- Dynamic LoRA loading
- Switch between speakers
- View pattern embedding status

### ✅ Modern WebUI
- Clean, responsive design
- Tabbed interface (Inference, Training, Models)
- Advanced parameter controls
- File upload with preview
- Real-time status updates

## 📡 API Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve WebUI |
| `/health` | GET | Health check |
| `/api` | GET | API info |
| `/models` | GET | List models |
| `/models/load/{name}` | POST | Load model |
| `/speakers` | GET | List speakers |
| `/inference/generate` | POST | Generate speech |
| `/inference/stream` | POST | Stream speech |
| `/training/start` | POST | Start training |
| `/training/status/{id}` | GET | Check status |
| `/training/tasks` | GET | List tasks |

**Full API docs:** http://localhost:8000/docs

## 💡 Usage Examples

### Generate Speech (WebUI)
1. Go to "Inference" tab
2. Enter text to synthesize
3. Upload reference audio OR select trained speaker
4. Enable "Use Pattern Embeddings" if using trained speaker
5. Click "Generate Speech"
6. Download the result

### Train New Speaker (WebUI)
1. Go to "Training" tab
2. Enter speaker name (e.g., "john_doe")
3. Upload 5+ audio files of the speaker
4. Adjust parameters (defaults are good)
5. Click "Start Training"
6. Monitor progress in real-time
7. When complete, use in Inference tab

### Generate Speech (API - Python)
```python
import requests
import json

files = {'audio_file': open('reference.wav', 'rb')}
data = {
    'request_json': json.dumps({
        'text': 'Hello, this is a test with patterns.',
        'speaker': 'john_doe',  # trained speaker
        'use_patterns': True,
        'temperature': 0.8,
        'top_p': 0.8,
        'top_k': 30
    })
}

response = requests.post(
    'http://localhost:8000/inference/generate',
    files=files,
    data=data
)

with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Train New Speaker (API - Python)
```python
import requests
import json

# Prepare files
files = [
    ('audio_files', open('audio1.wav', 'rb')),
    ('audio_files', open('audio2.wav', 'rb')),
    ('audio_files', open('audio3.wav', 'rb')),
    ('audio_files', open('audio4.wav', 'rb')),
    ('audio_files', open('audio5.wav', 'rb'))
]

data = {
    'request_json': json.dumps({
        'speaker_name': 'john_doe',
        'epochs': 40,
        'pattern_tokens': 8,
        'lora_rank': 32,
        'learning_rate': 0.0005,
        'batch_size': 4,
        'whisper_model': 'medium'
    })
}

response = requests.post(
    'http://localhost:8000/training/start',
    files=files,
    data=data
)

result = response.json()
print(f"Training started! Task ID: {result['task_id']}")

# Check status
status_response = requests.get(
    f"http://localhost:8000/training/status/{result['task_id']}"
)
print(status_response.json())
```

## 🎨 Emotion Control

Three methods available:

### 1. Manual Emotion Vector
Set 8 sliders (0.0-1.0):
- 😊 Happy, 😠 Angry, 😢 Sad, 😨 Afraid
- 🤢 Disgusted, 😔 Melancholic, 😲 Surprised, 😌 Calm

### 2. Text-Based Emotion
Enable "Extract Emotion from Text" and provide text like:
- "very happy and excited"
- "sad and disappointed"
- "calm and peaceful"

### 3. Reference Audio
Provide emotion reference audio via API.

## 📊 Training Pipeline Details

Automated steps:
1. **Transcription** - Whisper-based audio to text
2. **Conditioning** - Extract global features
3. **Dataset Prep** - Process with pattern features
4. **Embeddings** - Extract speaker embeddings
5. **Training** - Pattern embeddings + LoRA
6. **Validation** - Generate test outputs

**Time estimates:**
- 5-10 files: 30-60 min
- 20+ files: 1-3 hours
- GPU speeds up 2-4x

## 🔧 Troubleshooting

### Server won't start
- Install dependencies: `pip install -r api/requirements.txt`
- Check if port 8000 is free: `lsof -i :8000` (Mac/Linux)
- Try different port: `uvicorn api.main:app --port 8080`

### Model not loading
- Ensure `checkpoints/` directory exists
- Download model files if missing
- Check console for specific errors

### Training fails
- Need minimum 5 audio files
- Audio should be clear speech
- Check disk space
- Verify speaker name is unique

### WebUI not loading
- Ensure `api/static/` folder exists
- Check browser console for errors
- Try hard refresh (Ctrl+Shift+R)

## 📚 Additional Resources

- **Full API Docs**: [`api/README.md`](api/README.md)
- **Interactive API**: http://localhost:8000/docs (when running)
- **Main Project**: Your existing IndexTTS scripts still work!
- **Training Script**: [`tools/train_patterns_pipeline.py`](tools/train_patterns_pipeline.py)
- **Inference Script**: [`tools/infer_with_patterns.py`](tools/infer_with_patterns.py)

## 🎯 Next Steps

1. **Start the server**: `./start_api.sh` or `start_api.bat`
2. **Open WebUI**: http://localhost:8000
3. **Try inference** with an example audio file
4. **Train your first speaker** with 5+ audio samples
5. **Experiment** with emotion controls and parameters

## 💻 Integration with Existing Scripts

The API uses your existing best scripts:
- **Inference**: [`tools/infer_with_patterns.py`](tools/infer_with_patterns.py) pattern-aware inference
- **Training**: [`tools/train_patterns_pipeline.py`](tools/train_patterns_pipeline.py) full training pipeline
- **Model**: [`indextts/infer_v2.py`](indextts/infer_v2.py) IndexTTS2 inference class

Everything is backwards compatible - your existing CLI tools still work!

## 🌟 Features Summary

✅ REST API with all major endpoints  
✅ Modern, responsive WebUI  
✅ Pattern embedding support  
✅ Dynamic LoRA loading  
✅ Automated training pipeline  
✅ Real-time progress monitoring  
✅ Emotion control (3 methods)  
✅ Streaming inference (experimental)  
✅ Model management  
✅ File upload handling  
✅ Comprehensive documentation

---

**Enjoy your new IndexTTS API & WebUI! 🎉**

For questions, check [`api/README.md`](api/README.md) or the main project documentation.