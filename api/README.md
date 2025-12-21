# IndexTTS API & WebUI

Complete REST API and modern web interface for IndexTTS2 text-to-speech synthesis with pattern embeddings, LoRA fine-tuning, and training capabilities.

## Features

✨ **Comprehensive API**
- 🎵 **Inference**: Generate speech with pattern embeddings
- 📡 **Streaming**: Real-time audio streaming
- 🎓 **Training**: Automated training pipeline with file upload
- 📦 **Model Management**: Dynamic LoRA loading and model switching
- 🎙️ **Pattern-Aware**: Full support for pattern embeddings

🌐 **Modern WebUI**
- Clean, responsive design
- Real-time training progress monitoring
- Emotion control (vector-based or text-based)
- Advanced parameter tuning
- File management for training

## Quick Start

### 1. Installation

```bash
# Install API dependencies
cd api
pip install -r requirements.txt

# Ensure main project dependencies are installed
cd ..
pip install -r requirements.txt  # or follow main installation guide
```

### 2. Start the Server

```bash
# From project root
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the WebUI

Open your browser and navigate to:
```
http://localhost:8000
```

## API Endpoints

### Health & Status

#### `GET /health`
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "cuda_available": true
}
```

### Inference

#### `POST /inference/generate`
Generate speech from text with reference audio.

**Parameters:**
- `audio_file`: Speaker reference audio file (multipart/form-data)
- `request_json`: JSON string with request parameters

**Request JSON:**
```json
{
  "text": "Hello, this is a test.",
  "speaker": "john_doe",
  "use_patterns": true,
  "temperature": 0.8,
  "top_p": 0.8,
  "top_k": 30,
  "emo_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
  "use_emo_text": false,
  "emo_text": null
}
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/inference/generate" \
  -F "audio_file=@reference.wav" \
  -F 'request_json={"text":"Hello world","speaker":"john_doe","use_patterns":true}'
```

**Example (Python):**
```python
import requests

files = {'audio_file': open('reference.wav', 'rb')}
data = {
    'request_json': json.dumps({
        'text': 'Hello, this is a test.',
        'speaker': 'john_doe',
        'use_patterns': True,
        'temperature': 0.8,
        'top_p': 0.8,
        'top_k': 30
    })
}

response = requests.post('http://localhost:8000/inference/generate', files=files, data=data)
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

#### `POST /inference/stream`
Stream speech generation in real-time (experimental).

### Training

#### `POST /training/start`
Start training a new speaker model.

**Parameters:**
- `audio_files`: Multiple audio files (minimum 5) (multipart/form-data)
- `request_json`: JSON string with training parameters

**Request JSON:**
```json
{
  "speaker_name": "john_doe",
  "epochs": 40,
  "pattern_tokens": 8,
  "lora_rank": 32,
  "learning_rate": 0.0005,
  "batch_size": 4,
  "whisper_model": "medium"
}
```

**Example (Python):**
```python
import requests

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

response = requests.post('http://localhost:8000/training/start', files=files, data=data)
print(response.json())
```

#### `GET /training/status/{task_id}`
Get training task status.

**Response:**
```json
{
  "task_id": "john_doe_20231221_120000",
  "speaker_name": "john_doe",
  "status": "running",
  "progress": 0.65,
  "message": "Training epoch 26/40...",
  "started_at": "2023-12-21T12:00:00",
  "completed_at": null
}
```

#### `GET /training/tasks`
List all training tasks.

### Model Management

#### `GET /models`
List all available models and their status.

**Response:**
```json
[
  {
    "name": "base",
    "type": "base",
    "loaded": true,
    "has_lora": false,
    "has_patterns": false,
    "path": null
  },
  {
    "name": "john_doe",
    "type": "speaker",
    "loaded": false,
    "has_lora": true,
    "has_patterns": true,
    "path": "/path/to/training/john_doe"
  }
]
```

#### `POST /models/load/{speaker_name}`
Load a specific speaker's LoRA model.

**Example:**
```bash
curl -X POST "http://localhost:8000/models/load/john_doe"
```

#### `GET /speakers`
List all available trained speakers.

## WebUI Usage

### Inference Tab

1. **Enter Text**: Type or paste the text you want to synthesize
2. **Reference Audio**: 
   - Upload a reference audio file, OR
   - Select a trained speaker from the dropdown
3. **Pattern Embeddings**: Enable if using a trained speaker with patterns
4. **Advanced Options**:
   - Adjust temperature, top-p, top-k for generation control
   - Set emotion vectors (8 sliders for different emotions)
   - Or use text-based emotion extraction
5. **Generate**: Click "Generate Speech" or "Stream Speech"
6. **Download**: Save the generated audio

### Training Tab

1. **Speaker Name**: Enter a unique name (letters, numbers, underscores only)
2. **Upload Files**: Add at least 5 audio files of the speaker
3. **Training Parameters**:
   - **Epochs**: 40 recommended (more for better quality, slower training)
   - **Pattern Tokens**: 8 recommended (how many pattern features to learn)
   - **LoRA Rank**: 32 recommended (model capacity)
   - **Batch Size**: 4 recommended (adjust based on GPU memory)
   - **Whisper Model**: "medium" recommended (for transcription)
4. **Start Training**: Click to begin
5. **Monitor**: Watch real-time progress in the training tasks panel

### Models Tab

- View all available models (base + trained speakers)
- Load/unload LoRA models dynamically
- See which models have pattern embeddings
- Refresh to update model list

## Emotion Control

IndexTTS supports three methods of emotion control:

### 1. Emotion Vector (Manual)
Set 8 emotion sliders (values 0.0-1.0):
- 😊 Happy
- 😠 Angry
- 😢 Sad
- 😨 Afraid
- 🤢 Disgusted
- 😔 Melancholic
- 😲 Surprised
- 😌 Calm

### 2. Text-Based Emotion (Automatic)
Enable "Extract Emotion from Text" and optionally provide custom emotion text like:
- "very happy and excited"
- "sad and melancholic"
- "calm and peaceful"

### 3. Reference Audio Emotion
Use a reference audio file with the desired emotion (via emotion_audio parameter in API).

## Training Pipeline

The automated training pipeline performs:

1. **Transcription**: Whisper-based audio transcription
2. **Conditioning**: Extract global conditioning features
3. **Dataset Preparation**: Process audio with pattern features
4. **Speaker Embeddings**: Extract voice embeddings
5. **Pattern Training**: Train pattern embeddings + LoRA
6. **Validation**: Generate test outputs

Training typically takes:
- **5-10 audio files**: ~30-60 minutes (CPU/GPU dependent)
- **20+ audio files**: 1-3 hours
- With GPU acceleration: 2-4x faster

## Advanced Configuration

### Custom Model Paths

Modify [`api/main.py`](api/main.py:116) to use custom checkpoint directories:

```python
model_dir = Path("/custom/path/to/checkpoints")
```

### Port Configuration

Change the default port (8000):

```bash
uvicorn api.main:app --port 8080
```

### Production Deployment

For production, use proper ASGI server configuration:

```bash
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --limit-concurrency 100 \
  --timeout-keep-alive 30
```

## API Response Formats

### Success Response
```json
{
  "status": "success",
  "message": "Operation completed",
  "data": {...}
}
```

### Error Response
```json
{
  "detail": "Error message description"
}
```

## Troubleshooting

### Model Not Loading
- Ensure `checkpoints/` directory exists in project root
- Verify all required model files are present
- Check console output for specific errors

### Training Fails
- Minimum 5 audio files required
- Audio files should be clear speech (WAV/MP3/FLAC)
- Check disk space for dataset processing
- Verify speaker name doesn't already exist

### Inference Errors
- Ensure reference audio is valid
- For pattern inference, verify speaker is trained
- Check temperature/top-p/top-k values are in valid ranges

### WebUI Not Loading
- Verify static files exist in `api/static/`
- Check browser console for JavaScript errors
- Ensure correct URL: `http://localhost:8000/`

## File Structure

```
api/
├── __init__.py                 # Package init
├── main.py                     # FastAPI application
├── requirements.txt            # API dependencies
├── README.md                   # This file
└── static/                     # WebUI files
    ├── index.html             # Main HTML
    ├── styles.css             # Styles
    └── app.js                 # JavaScript logic
```

## Contributing

Contributions welcome! Please:
1. Test your changes thoroughly
2. Update documentation as needed
3. Follow existing code style
4. Add examples for new features

## License

Same license as main IndexTTS project.

## Support

For issues or questions:
- Check existing issues in main IndexTTS repository
- Review documentation above
- Check console logs for detailed error messages

---

**Built with ❤️ for the IndexTTS community**