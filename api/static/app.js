// IndexTTS Web UI JavaScript
const API_BASE = window.location.origin;

// State
let currentAudioUrl = null;
let trainingTasksInterval = null;
let emotionVector = [0, 0, 0, 0, 0, 0, 0, 0];
let audioContext = null;
let audioQueue = [];
let isPlaying = false;
let nextStartTime = 0;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    checkHealth();
    loadSpeakers();
    loadModels();
    setupEventListeners();
    setupDragAndDrop();
});

// ============= Initialization =============

function initializeUI() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });

    // Slider value displays
    updateSliderDisplays();
}

function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    // Load data for specific tabs
    if (tabName === 'models') {
        loadModels();
        loadSpeakers();
    } else if (tabName === 'training') {
        loadTrainingTasks();
    }
}

function updateSliderDisplays() {
    // Temperature
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    // Top P
    const topPSlider = document.getElementById('topP');
    const topPValue = document.getElementById('topPValue');
    topPSlider.addEventListener('input', () => {
        topPValue.textContent = topPSlider.value;
    });

    // Top K
    const topKSlider = document.getElementById('topK');
    const topKValue = document.getElementById('topKValue');
    topKSlider.addEventListener('input', () => {
        topKValue.textContent = topKSlider.value;
    });

    // Training sliders
    const slidersConfig = [
        { id: 'epochs', displayId: 'epochsValue' },
        { id: 'patternTokens', displayId: 'patternTokensValue' },
        { id: 'loraRank', displayId: 'loraRankValue' },
        { id: 'batchSize', displayId: 'batchSizeValue' }
    ];

    slidersConfig.forEach(({ id, displayId }) => {
        const slider = document.getElementById(id);
        const display = document.getElementById(displayId);
        slider.addEventListener('input', () => {
            display.textContent = slider.value;
        });
    });

    // Emotion sliders
    const emoLabels = ['Happy', 'Angry', 'Sad', 'Afraid', 'Disgusted', 'Melancholic', 'Surprised', 'Calm'];
    document.querySelectorAll('.emo-slider').forEach((slider, index) => {
        const displayId = `emo${emoLabels[index]}`;
        const display = document.getElementById(displayId);
        slider.addEventListener('input', () => {
            const value = parseFloat(slider.value);
            emotionVector[index] = value;
            display.textContent = value.toFixed(2);
        });
    });
}

// ============= Drag and Drop Setup =============

function setupDragAndDrop() {
    // Reference audio drag and drop (inference)
    const referenceDropZone = document.querySelector('[data-drop-zone="reference"]');
    const referenceInput = document.getElementById('referenceAudio');
    
    if (referenceDropZone) {
        setupDropZone(referenceDropZone, referenceInput, handleReferenceAudioDrop);
    }

    // Training files drag and drop
    const trainingDropZone = document.querySelector('[data-drop-zone="training"]');
    const trainingInput = document.getElementById('trainingFiles');
    
    if (trainingDropZone) {
        setupDropZone(trainingDropZone, trainingInput, handleTrainingFilesDrop);
    }
}

function setupDropZone(dropZone, fileInput, dropHandler) {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        dropHandler(files, fileInput);
    }, false);

    // Click to browse
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleReferenceAudioDrop(files, fileInput) {
    if (files.length === 0) return;

    // Only accept first audio file
    const audioFile = Array.from(files).find(file => file.type.startsWith('audio/'));
    
    if (!audioFile) {
        showNotification('Please drop an audio file', 'error');
        return;
    }

    // Create a new FileList-like object
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(audioFile);
    fileInput.files = dataTransfer.files;

    // Update display
    document.getElementById('referenceAudioName').textContent = audioFile.name;
    showNotification(`Added: ${audioFile.name}`, 'success');
}

function handleTrainingFilesDrop(files, fileInput) {
    if (files.length === 0) return;

    // Filter for audio files only
    const audioFiles = Array.from(files).filter(file => file.type.startsWith('audio/'));
    
    if (audioFiles.length === 0) {
        showNotification('Please drop audio files', 'error');
        return;
    }

    // Create a new FileList-like object
    const dataTransfer = new DataTransfer();
    audioFiles.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;

    // Update display
    updateFileList(dataTransfer.files);
    showNotification(`Added ${audioFiles.length} audio file(s)`, 'success');
}

// ============= Event Listeners =============

function setupEventListeners() {
    // Load base model button
    document.getElementById('loadBaseModelBtn').addEventListener('click', loadBaseModel);
    
    // File inputs
    document.getElementById('referenceAudio').addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || 'No file selected';
        document.getElementById('referenceAudioName').textContent = fileName;
    });

    document.getElementById('trainingFiles').addEventListener('change', (e) => {
        updateFileList(e.target.files);
    });

    // Emotion text checkbox
    document.getElementById('useEmoText').addEventListener('change', (e) => {
        document.getElementById('emoTextGroup').style.display = e.target.checked ? 'block' : 'none';
    });

    // Generate buttons
    document.getElementById('generateBtn').addEventListener('click', generateSpeech);
    document.getElementById('streamBtn').addEventListener('click', streamSpeech);

    // Training
    document.getElementById('startTrainingBtn').addEventListener('click', startTraining);

    // Models
    document.getElementById('refreshModelsBtn').addEventListener('click', () => {
        loadModels();
        loadSpeakers();
    });

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', downloadAudio);

    const clearAudioBtn = document.getElementById('clearReferenceAudioBtn');
    if (clearAudioBtn) {
        clearAudioBtn.addEventListener('click', clearReferenceAudio);
    }
}

// ============= API Calls =============

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const modelLoadingCard = document.getElementById('modelLoadingCard');
        
        if (data.status === 'healthy' && data.model_loaded) {
            statusDot.classList.add('connected');
            statusText.textContent = `Connected (${data.device || 'cpu'})`;
            modelLoadingCard.style.display = 'none';
        } else if (data.status === 'healthy') {
            statusDot.classList.remove('error');
            statusText.textContent = 'Connected (model not loaded)';
            modelLoadingCard.style.display = 'block';
        } else {
            statusDot.classList.add('error');
            statusText.textContent = 'Error';
            modelLoadingCard.style.display = 'none';
        }
    } catch (error) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        statusDot.classList.add('error');
        statusText.textContent = 'Disconnected';
        console.error('Health check failed:', error);
    }
}

async function loadBaseModel() {
    const btn = document.getElementById('loadBaseModelBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span> Loading...';
    
    try {
        const response = await fetch(`${API_BASE}/models/load/base`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load base model');
        }
        
        const data = await response.json();
        showNotification(data.message, 'success');
        
        // Refresh health check to update UI
        await checkHealth();
        
    } catch (error) {
        console.error('Failed to load base model:', error);
        showNotification(error.message || 'Failed to load base model', 'error');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">📦</span> Load Base Model';
    }
}

async function loadSpeakers() {
    try {
        const response = await fetch(`${API_BASE}/speakers`);
        const speakers = await response.json();
        
        const speakerSelect = document.getElementById('speakerSelect');
        speakerSelect.innerHTML = '<option value="">None (use audio file)</option>';
        
        speakers.forEach(speaker => {
            if (speaker.has_patterns) {
                const option = document.createElement('option');
                option.value = speaker.name;
                option.textContent = `${speaker.name} ${speaker.has_patterns ? '(with patterns)' : ''}`;
                speakerSelect.appendChild(option);
            }
        });

        // Update speakers list in Models tab
        const speakersList = document.getElementById('speakersList');
        if (speakers.length === 0) {
            speakersList.innerHTML = '<p class="text-muted">No trained speakers found</p>';
        } else {
            speakersList.innerHTML = speakers.map(speaker => `
                <div class="speaker-item">
                    <div class="model-info">
                        <div class="model-name">${speaker.name}</div>
                        <div class="model-badges">
                            ${speaker.has_embeddings ? '<span class="badge badge-info">Embeddings</span>' : ''}
                            ${speaker.has_patterns ? '<span class="badge badge-success">Patterns</span>' : ''}
                        </div>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load speakers:', error);
        showNotification('Failed to load speakers', 'error');
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const models = await response.json();
        
        const modelsList = document.getElementById('modelsList');
        if (models.length === 0) {
            modelsList.innerHTML = '<p class="text-muted">No models found</p>';
        } else {
            modelsList.innerHTML = models.map(model => `
                <div class="model-item ${model.loaded ? 'loaded' : ''}">
                    <div class="model-info">
                        <div class="model-name">${model.name}</div>
                        <div class="model-type">${model.type}</div>
                        <div class="model-badges">
                            ${model.loaded ? '<span class="badge badge-success">Loaded</span>' : ''}
                            ${model.has_lora ? '<span class="badge badge-info">LoRA</span>' : ''}
                            ${model.has_patterns ? '<span class="badge badge-info">Patterns</span>' : ''}
                        </div>
                    </div>
                    ${!model.loaded && model.name !== 'base' ? `
                        <button class="btn btn-small btn-secondary" onclick="loadModel('${model.name}')">
                            Load Model
                        </button>
                    ` : ''}
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        showNotification('Failed to load models', 'error');
    }
}

async function loadModel(modelName) {
    try {
        const response = await fetch(`${API_BASE}/models/load/${modelName}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (response.ok) {
            showNotification(data.message, 'success');
            loadModels();
        } else {
            showNotification(data.detail || 'Failed to load model', 'error');
        }
    } catch (error) {
        console.error('Failed to load model:', error);
        showNotification('Failed to load model', 'error');
    }
}

async function generateSpeech() {
    const text = document.getElementById('inferenceText').value.trim();
    const audioFile = document.getElementById('referenceAudio').files[0];
    const speaker = document.getElementById('speakerSelect').value;
    const usePatterns = document.getElementById('usePatternsCheckbox').checked;

    if (!text) {
        showNotification('Please enter text to synthesize', 'error');
        return;
    }

    // Validate: need either audio file or speaker with patterns
    if (!audioFile && !speaker) {
        showNotification('Please upload reference audio or select a speaker', 'error');
        return;
    }

    if (usePatterns && !speaker) {
        showNotification('Pattern embeddings require a trained speaker', 'error');
        return;
    }

    // If using patterns with speaker but no audio, that's OK (will use embeddings)
    // If NOT using patterns, we must have an audio file
    if (!usePatterns && !audioFile) {
        showNotification('Please upload reference audio when not using patterns', 'error');
        return;
    }

    const generateBtn = document.getElementById('generateBtn');
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';

    showProgress('Generating speech...', 0);

    try {
        const formData = new FormData();
        
        // Only append audio file if it exists
        if (audioFile) {
            formData.append('audio_file', audioFile);
        }
        // When using patterns with speaker but no audio file, don't send any audio
        // The backend will use speaker embeddings

        // Get emotion vector (only non-zero values)
        const hasEmotionVector = emotionVector.some(v => v > 0);
        
        const requestData = {
            text: text,
            speaker: speaker || null,
            use_patterns: usePatterns,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('topP').value),
            top_k: parseInt(document.getElementById('topK').value),
            emo_vector: hasEmotionVector ? emotionVector : null,
            use_emo_text: document.getElementById('useEmoText').checked,
            emo_text: document.getElementById('emotionText').value || null
        };

        formData.append('request_json', JSON.stringify(requestData));

        const response = await fetch(`${API_BASE}/inference/generate`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        if (currentAudioUrl) {
            URL.revokeObjectURL(currentAudioUrl);
        }
        currentAudioUrl = audioUrl;

        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = audioUrl;
        
        document.getElementById('audioOutput').style.display = 'block';
        hideProgress();
        showNotification('Speech generated successfully!', 'success');

    } catch (error) {
        console.error('Generation failed:', error);
        hideProgress();
        showNotification(error.message || 'Failed to generate speech', 'error');
    } finally {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<span class="btn-icon">🎵</span> Generate Speech';
    }
}

async function streamSpeech() {
    const text = document.getElementById('inferenceText').value.trim();
    const audioFile = document.getElementById('referenceAudio').files[0];
    const speaker = document.getElementById('speakerSelect').value;
    const usePatterns = document.getElementById('usePatternsCheckbox').checked;

    // --- Validation ---
    if (!text) { showNotification('Please enter text to synthesize', 'error'); return; }
    if (!audioFile && !speaker) { showNotification('Please upload reference audio or select a speaker', 'error'); return; }
    if (usePatterns && !speaker) { showNotification('Pattern embeddings require a trained speaker', 'error'); return; }
    if (!usePatterns && !audioFile) { showNotification('Please upload reference audio when not using patterns', 'error'); return; }

    const streamBtn = document.getElementById('streamBtn');
    streamBtn.disabled = true;
    streamBtn.textContent = 'Streaming...';
    showProgress('Initializing stream...', 0);

    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        const formData = new FormData();
        if (audioFile) formData.append('audio_file', audioFile);

        // Check if we have any emotion values set
        const hasEmotionVector = emotionVector.some(v => v > 0);

        const requestData = {
            text: text,
            speaker: speaker || null,
            use_patterns: usePatterns,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('topP').value),
            top_k: parseInt(document.getElementById('topK').value),
            emo_vector: hasEmotionVector ? emotionVector : null,
            use_emo_text: document.getElementById('useEmoText').checked,
            emo_text: document.getElementById('emotionText').value || null
        };
        
        formData.append('request_json', JSON.stringify(requestData));

        const response = await fetch(`${API_BASE}/inference/stream`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(await response.text() || 'Streaming failed');

        const reader = response.body.getReader();
        
        let nextStartTime = audioContext.currentTime;
        let chunkCount = 0;
        let isFirstChunk = true;
        let sampleRate = 24000; 
        let channels = 1;
        let leftoverChunk = new Uint8Array(0);

        console.log('Stream connection established. Playing...');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const combinedData = new Uint8Array(leftoverChunk.length + value.length);
            combinedData.set(leftoverChunk);
            combinedData.set(value, leftoverChunk.length);

            let processData = combinedData;

            if (isFirstChunk) {
                if (combinedData.length >= 44) {
                    const view = new DataView(combinedData.buffer, combinedData.byteOffset, combinedData.byteLength);
                    channels = view.getUint16(22, true);
                    sampleRate = view.getUint32(24, true);
                    processData = combinedData.slice(44);
                    isFirstChunk = false;
                } else {
                    leftoverChunk = combinedData;
                    continue; 
                }
            }

            const remainder = processData.length % 2;
            if (remainder !== 0) {
                leftoverChunk = processData.slice(processData.length - 1);
                processData = processData.slice(0, processData.length - 1);
            } else {
                leftoverChunk = new Uint8Array(0);
            }

            if (processData.length > 0) {
                const audioFloat32 = convertInt16ToFloat32(processData);
                const audioBuffer = audioContext.createBuffer(channels, audioFloat32.length, sampleRate);
                audioBuffer.getChannelData(0).set(audioFloat32);

                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);

                if (nextStartTime < audioContext.currentTime) {
                    nextStartTime = audioContext.currentTime + 0.05;
                }

                source.start(nextStartTime);
                nextStartTime += audioBuffer.duration;

                chunkCount++;
                showProgress(`Streaming... Chunk ${chunkCount}`, 50);
            }
        }

        hideProgress();
        showNotification('Stream finished', 'success');

    } catch (error) {
        console.error('Streaming failed:', error);
        hideProgress();
        showNotification(error.message || 'Failed to stream speech', 'error');
    } finally {
        streamBtn.disabled = false;
        streamBtn.innerHTML = '<span class="btn-icon">🎙️</span> Stream Speech';
    }
}

function convertInt16ToFloat32(inputArray) {
    // Determine the length of the float32 array
    // Int16 uses 2 bytes per sample, so length is half
    const output = new Float32Array(inputArray.length / 2);
    const view = new DataView(inputArray.buffer, inputArray.byteOffset, inputArray.byteLength);
    
    for (let i = 0; i < output.length; i++) {
        // Convert PCM 16-bit Int to Float32 (-1.0 to 1.0)
        // Little Endian is standard for WAV
        const int16 = view.getInt16(i * 2, true);
        output[i] = int16 / 32768; 
    }
    return output;
}

function downloadAudio() {
    if (!currentAudioUrl) return;
    
    const a = document.createElement('a');
    a.href = currentAudioUrl;
    a.download = `indextts_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showNotification('Audio downloaded', 'success');
}

// ============= Training =============

function updateFileList(files) {
    const fileList = document.getElementById('fileList');
    const fileCount = files.length;
    
    document.getElementById('trainingFilesName').textContent = 
        `${fileCount} file${fileCount !== 1 ? 's' : ''} selected`;

    if (fileCount === 0) {
        fileList.innerHTML = '';
        return;
    }

    const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);
    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    fileList.innerHTML = `
        <div style="margin-bottom: 15px;">
            <strong>Total: ${fileCount} files (${formatSize(totalSize)})</strong>
        </div>
        ${Array.from(files).map((file, i) => `
            <div class="file-item">
                <span class="file-item-name">${file.name}</span>
                <span class="file-item-size">${formatSize(file.size)}</span>
            </div>
        `).join('')}
    `;
}

function clearReferenceAudio() {
    const fileInput = document.getElementById('referenceAudio');
    const fileNameDisplay = document.getElementById('referenceAudioName');
    
    // Reset the input value
    fileInput.value = '';
    
    // Update the UI text
    fileNameDisplay.textContent = 'No file selected';
    
    // Optional: Hide the audio output if it was related to this file
    // document.getElementById('audioOutput').style.display = 'none';
    
    showNotification('Reference audio removed', 'info');
}

async function startTraining() {
    const speakerName = document.getElementById('speakerName').value.trim();
    const files = document.getElementById('trainingFiles').files;

    if (!speakerName) {
        showNotification('Please enter a speaker name', 'error');
        return;
    }

    if (!/^[a-zA-Z0-9_]+$/.test(speakerName)) {
        showNotification('Speaker name can only contain letters, numbers, and underscores', 'error');
        return;
    }

    if (files.length < 5) {
        showNotification('Please upload at least 5 audio files', 'error');
        return;
    }

    const startBtn = document.getElementById('startTrainingBtn');
    startBtn.disabled = true;
    startBtn.textContent = 'Starting training...';

    try {
        const formData = new FormData();
        
        Array.from(files).forEach(file => {
            formData.append('audio_files', file);
        });

        const requestData = {
            speaker_name: speakerName,
            epochs: parseInt(document.getElementById('epochs').value),
            pattern_tokens: parseInt(document.getElementById('patternTokens').value),
            lora_rank: parseInt(document.getElementById('loraRank').value),
            learning_rate: 5e-4,
            batch_size: parseInt(document.getElementById('batchSize').value),
            whisper_model: document.getElementById('whisperModel').value
        };

        formData.append('request_json', JSON.stringify(requestData));

        const response = await fetch(`${API_BASE}/training/start`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start training');
        }

        const data = await response.json();
        showNotification(`Training started for ${speakerName}!`, 'success');
        
        // Switch to monitoring
        document.getElementById('trainingTasksCard').style.display = 'block';
        loadTrainingTasks();
        
        // Start polling for updates
        if (trainingTasksInterval) {
            clearInterval(trainingTasksInterval);
        }
        trainingTasksInterval = setInterval(loadTrainingTasks, 5000);

    } catch (error) {
        console.error('Failed to start training:', error);
        showNotification(error.message || 'Failed to start training', 'error');
    } finally {
        startBtn.disabled = false;
        startBtn.innerHTML = '<span class="btn-icon">🚀</span> Start Training';
    }
}

async function loadTrainingTasks() {
    try {
        const response = await fetch(`${API_BASE}/training/tasks`);
        const tasks = await response.json();
        
        const tasksList = document.getElementById('trainingTasksList');
        
        if (tasks.length === 0) {
            tasksList.innerHTML = '<p class="text-muted">No training tasks</p>';
            document.getElementById('trainingTasksCard').style.display = 'none';
            if (trainingTasksInterval) {
                clearInterval(trainingTasksInterval);
                trainingTasksInterval = null;
            }
            return;
        }

        document.getElementById('trainingTasksCard').style.display = 'block';

        tasksList.innerHTML = tasks.map(task => `
            <div class="task-item ${task.status}">
                <div class="task-header">
                    <div class="task-name">${task.speaker_name}</div>
                    <div class="task-status ${task.status}">${task.status}</div>
                </div>
                <div class="task-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${task.progress * 100}%"></div>
                    </div>
                </div>
                <div class="task-message">${task.message}</div>
                <small class="text-muted">Started: ${new Date(task.started_at).toLocaleString()}</small>
            </div>
        `).join('');

        // Stop polling if all tasks are completed or failed
        const hasRunning = tasks.some(t => t.status === 'running' || t.status === 'queued');
        if (!hasRunning && trainingTasksInterval) {
            clearInterval(trainingTasksInterval);
            trainingTasksInterval = null;
        }

    } catch (error) {
        console.error('Failed to load training tasks:', error);
    }
}


// ============= UI Helpers =============

function showProgress(message, progress) {
    const container = document.getElementById('inferenceProgress');
    const fill = document.getElementById('inferenceProgressFill');
    const text = document.getElementById('inferenceProgressText');
    
    container.style.display = 'block';
    fill.style.width = `${progress}%`;
    text.textContent = message;
}

function hideProgress() {
    document.getElementById('inferenceProgress').style.display = 'none';
}

function showNotification(message, type = 'info') {
    // 1. Ensure a container exists on the page
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        document.body.appendChild(container);
    }

    const emojis = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    };

    // 2. Create the toast element
    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.innerHTML = `<span>${emojis[type] || 'ℹ️'}</span> <span>${message}</span>`;

    // 3. Add to container
    container.appendChild(toast);

    // 4. Automatically remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.5s ease';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
    }
    if (trainingTasksInterval) {
        clearInterval(trainingTasksInterval);
    }
});

// Auto-refresh health check every 30 seconds
setInterval(checkHealth, 30000);