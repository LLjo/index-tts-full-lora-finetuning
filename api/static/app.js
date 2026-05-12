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

    // Inference tab is the default landing tab — refresh distill status now so
    // the banner reflects current state before any user interaction.
    refreshInferDistillStatus();
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
    } else if (tabName === 'distill') {
        distillInitOnce();
        distillRefresh();
    } else if (tabName === 'inference') {
        refreshInferDistillStatus();
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

    // Test Lab
    const labAudioInput = document.getElementById('lab-audio');
    if (labAudioInput) {
        labAudioInput.addEventListener('change', (e) => {
            const name = e.target.files[0]?.name || 'No file selected';
            document.getElementById('lab-audio-name').textContent = name;
        });
    }
    const labWarmupBtn = document.getElementById('lab-warmup-btn');
    if (labWarmupBtn) labWarmupBtn.addEventListener('click', () => labWarmup(false));
    const labWarmupAllBtn = document.getElementById('lab-warmup-all-btn');
    if (labWarmupAllBtn) labWarmupAllBtn.addEventListener('click', () => labWarmup(true));
    const labRunBtn = document.getElementById('lab-run-btn');
    if (labRunBtn) labRunBtn.addEventListener('click', labRunSingle);
    const labCompareBtn = document.getElementById('lab-compare-btn');
    if (labCompareBtn) labCompareBtn.addEventListener('click', labCompareAll);
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

        // Also populate the Test Lab speaker dropdown (mirrors trained-speakers list).
        const labSpeakerSelect = document.getElementById('lab-speaker');
        if (labSpeakerSelect) {
            labSpeakerSelect.innerHTML = '<option value="">None</option>';
        }

        speakers.forEach(speaker => {
            if (speaker.has_patterns) {
                const option = document.createElement('option');
                option.value = speaker.name;
                option.textContent = `${speaker.name} ${speaker.has_patterns ? '(with patterns)' : ''}`;
                speakerSelect.appendChild(option);

                if (labSpeakerSelect) {
                    const labOption = option.cloneNode(true);
                    labSpeakerSelect.appendChild(labOption);
                }
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

        // Streaming preset chosen on the Inference tab. Defaults to ultra_fast.
        const presetEl = document.getElementById('streamingPreset');
        const streamingPreset = presetEl ? presetEl.value : 'ultra_fast';

        const requestData = {
            text: text,
            speaker: speaker || null,
            use_patterns: usePatterns,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('topP').value),
            top_k: parseInt(document.getElementById('topK').value),
            emo_vector: hasEmotionVector ? emotionVector : null,
            use_emo_text: document.getElementById('useEmoText').checked,
            emo_text: document.getElementById('emotionText').value || null,
            streaming_preset: streamingPreset,
            ...inferOverridePayload(),
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

// ============= Test Lab =============

const LAB_ALL_PRESETS = ["ultra_fast", "ultra_fast_distilled", "fast", "fast_quality", "balanced", "balanced_distilled", "quality", "progressive"];

function labReadInputs() {
    return {
        text: document.getElementById('lab-text').value.trim(),
        audio: document.getElementById('lab-audio').files[0] || null,
        speaker: document.getElementById('lab-speaker').value || null,
        usePatterns: document.getElementById('lab-use-patterns').checked,
        preset: document.getElementById('lab-preset').value,
        runs: parseInt(document.getElementById('lab-runs').value, 10) || 1,
    };
}

function labSetStatus(text, kind) {
    const el = document.getElementById('lab-status');
    if (!el) return;
    const color = kind === 'error' ? '#d63031'
        : kind === 'success' ? '#00b894'
        : '#2d3436';
    el.style.color = color;
    el.style.whiteSpace = 'pre-wrap';
    el.style.fontFamily = text.includes('\n') ? 'monospace' : 'inherit';
    el.style.fontSize = text.includes('\n') ? '12px' : 'inherit';
    el.textContent = text;
}

function labBuildFormData(inputs, preset, textOverride) {
    const fd = new FormData();
    if (inputs.audio) fd.append('audio_file', inputs.audio);
    const payload = {
        text: textOverride || inputs.text,
        speaker: inputs.speaker,
        use_patterns: inputs.usePatterns,
        streaming_preset: preset,
        verbose: true,
    };
    fd.append('request_json', JSON.stringify(payload));
    return fd;
}

function labValidate(inputs) {
    if (!inputs.text) {
        labSetStatus('Enter some text first.', 'error');
        return false;
    }
    if (!inputs.audio && !inputs.speaker) {
        labSetStatus('Provide reference audio or pick a trained speaker.', 'error');
        return false;
    }
    if (inputs.usePatterns && !inputs.speaker) {
        labSetStatus('Pattern embeddings require a trained speaker.', 'error');
        return false;
    }
    return true;
}

// Realistic-length warmup text. Must exceed several chunk_tokens worth of synthesis
// so torch.compile / CUDA graphs specialize for the shapes the benchmark will use.
const LAB_WARMUP_TEXT = (
    "Hello, this is a warmup pass to capture CUDA graphs and just-in-time compile " +
    "the diffusion model against realistic chunk shapes before benchmarking."
);

async function labWarmup(allPresets) {
    const inputs = labReadInputs();
    if (!labValidate(inputs)) return;

    const warmupBtn = document.getElementById('lab-warmup-btn');
    const warmupAllBtn = document.getElementById('lab-warmup-all-btn');
    warmupBtn.disabled = true;
    if (warmupAllBtn) warmupAllBtn.disabled = true;

    const presets = allPresets ? LAB_ALL_PRESETS : [inputs.preset];
    labSetStatus(
        allPresets
            ? `Warming all presets (${presets.length}). First-time CUDA graph capture can take 30-60s per preset...`
            : `Warming pipeline for "${inputs.preset}" (first call can take 20-40s)...`
    );

    try {
        const fd = labBuildFormData(inputs, inputs.preset, LAB_WARMUP_TEXT);
        // Override request_json to include the presets list.
        const payload = JSON.parse(fd.get('request_json'));
        payload.text = LAB_WARMUP_TEXT;
        payload.presets = presets;
        fd.set('request_json', JSON.stringify(payload));

        const r = await fetch(`${API_BASE}/inference/warmup`, { method: 'POST', body: fd });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'warmup failed');

        if (data.presets_warmed && data.presets_warmed.length > 1) {
            const lines = data.presets_warmed.map(p =>
                `${p.preset.padEnd(12)} TTFA=${p.ttfa_ms}ms total=${p.total_time_ms}ms chunks=${p.chunks}`
            );
            labSetStatus(
                `Warmed ${data.presets_warmed.length} presets in ${(data.overall_time_ms / 1000).toFixed(1)}s. ` +
                `Latest TTFA per preset:\n${lines.join('\n')}`,
                'success'
            );
        } else {
            labSetStatus(
                `Warmed up: TTFA=${data.ttfa_ms}ms, total=${data.total_time_ms}ms, ` +
                `chunks=${data.chunks}, audio=${data.audio_seconds}s`,
                'success'
            );
        }
    } catch (e) {
        labSetStatus(`Warmup error: ${e.message}`, 'error');
    } finally {
        warmupBtn.disabled = false;
        if (warmupAllBtn) warmupAllBtn.disabled = false;
    }
}

async function labRunDiagnosticsOnce(inputs, preset) {
    const fd = labBuildFormData(inputs, preset);
    const r = await fetch(`${API_BASE}/inference/stream/diagnostics`, { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `diagnostics ${preset} failed`);
    return data;
}

function labClearResults() {
    document.getElementById('lab-results-body').innerHTML = '';
    document.getElementById('lab-summary').innerHTML = '';
    document.getElementById('lab-chunks-body').innerHTML = '';
}

function labAppendResultRow(data, label) {
    const tbody = document.getElementById('lab-results-body');
    const tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid #dfe6e9';
    const cells = [
        ['left',   label || data.preset],
        ['right',  data.ttfa_ms ?? '—'],
        ['right',  data.total_time_ms ?? '—'],
        ['right',  data.audio_seconds ?? '—'],
        ['right',  data.rtf ?? '—'],
        ['right',  data.chunk_count ?? '—'],
        ['center', data.solver || '—'],
        ['center', data.accel_engine_active ? '✓' : '✗'],
    ];
    cells.forEach(([align, val]) => {
        const td = document.createElement('td');
        td.style.padding = '8px';
        td.style.textAlign = align;
        td.textContent = String(val);
        tr.appendChild(td);
    });
    tbody.appendChild(tr);
    document.getElementById('lab-results-card').style.display = 'block';
}

function labRenderStages(data) {
    const tbody = document.getElementById('lab-stages-body');
    if (!tbody) return;
    tbody.innerHTML = '';
    const stages = data.stages || {};
    // Define the stage display order so the table reads top-to-bottom in time.
    const order = [
        ['request_start_ms',           'request start'],
        ['conditioning_done_ms',       'conditioning extracted'],
        ['threads_starting_ms',        'gpt+synth threads started'],
        ['gpt_first_token_ms',         'gpt: first token sampled'],
        ['chunk1_dispatched_ms',       'chunk 1: dispatched to synth queue'],
        ['chunk1_synth_start_ms',      'chunk 1: synth start'],
        ['chunk1_gpt_latent_done_ms',  'chunk 1: gpt-latent forward done'],
        ['chunk1_length_reg_done_ms',  'chunk 1: length regulator done'],
        ['chunk1_cfm_done_ms',         'chunk 1: CFM diffusion done'],
        ['chunk1_bigvgan_done_ms',     'chunk 1: BigVGAN done'],
        ['chunk1_synth_done_ms',       'chunk 1: synth done (.cpu())'],
        ['chunk1_yielded_ms',          'chunk 1: yielded to client'],
        ['chunk2_dispatched_ms',       'chunk 2: dispatched'],
        ['chunk2_synth_start_ms',      'chunk 2: synth start'],
        ['chunk2_gpt_latent_done_ms',  'chunk 2: gpt-latent forward done'],
        ['chunk2_length_reg_done_ms',  'chunk 2: length regulator done'],
        ['chunk2_cfm_done_ms',         'chunk 2: CFM diffusion done'],
        ['chunk2_bigvgan_done_ms',     'chunk 2: BigVGAN done'],
        ['chunk2_synth_done_ms',       'chunk 2: synth done'],
        ['chunk2_yielded_ms',          'chunk 2: yielded'],
    ];
    let prevT = null;
    order.forEach(([key, label]) => {
        const t = stages[key];
        if (t == null) return;
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #f1f3f5';
        const delta = prevT == null ? '' : (t - prevT).toFixed(1);
        const cells = [
            ['left',  label],
            ['right', t.toFixed(1)],
            ['right', delta],
        ];
        cells.forEach(([align, val]) => {
            const td = document.createElement('td');
            td.style.padding = '6px';
            td.style.textAlign = align;
            td.textContent = String(val);
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
        prevT = t;
    });
    document.getElementById('lab-stages-card').style.display = 'block';
}

function labRenderChunks(data) {
    const tbody = document.getElementById('lab-chunks-body');
    tbody.innerHTML = '';
    (data.chunks || []).forEach((c, i) => {
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #f1f3f5';
        // Highlight chunks where the player buffer would underrun.
        const underrun = (c.since_prev_ms || 0) > (c.audio_ms || 0) && i > 0;
        if (underrun) tr.style.background = '#ffeaa7';
        const cells = [
            ['left',  i + 1],
            ['right', c.elapsed_ms],
            ['right', c.since_prev_ms],
            ['right', c.audio_ms],
            ['right', c.samples],
        ];
        cells.forEach(([align, val]) => {
            const td = document.createElement('td');
            td.style.padding = '6px';
            td.style.textAlign = align;
            td.textContent = String(val);
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    document.getElementById('lab-chunks-card').style.display = 'block';
}

async function labRunSingle() {
    const inputs = labReadInputs();
    if (!labValidate(inputs)) return;

    const btn = document.getElementById('lab-run-btn');
    btn.disabled = true;
    labClearResults();
    labSetStatus(`Running ${inputs.runs} run(s) of "${inputs.preset}"...`);

    try {
        let lastData = null;
        for (let i = 0; i < inputs.runs; i++) {
            const data = await labRunDiagnosticsOnce(inputs, inputs.preset);
            const label = inputs.runs > 1 ? `${inputs.preset} #${i + 1}` : inputs.preset;
            labAppendResultRow(data, label);
            lastData = data;
        }
        if (lastData) {
            labRenderStages(lastData);
            labRenderChunks(lastData);
        }
        labSetStatus(`Done. TTFA on last run: ${lastData?.ttfa_ms} ms.`, 'success');
    } catch (e) {
        labSetStatus(`Error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}

async function labCompareAll() {
    const inputs = labReadInputs();
    if (!labValidate(inputs)) return;

    const btn = document.getElementById('lab-compare-btn');
    btn.disabled = true;
    labClearResults();

    const perPresetTimings = {};
    let lastDataForChunks = null;

    try {
        for (const preset of LAB_ALL_PRESETS) {
            const ttfas = [];
            const totals = [];
            for (let i = 0; i < inputs.runs; i++) {
                labSetStatus(`Running ${preset} (${i + 1}/${inputs.runs})...`);
                const data = await labRunDiagnosticsOnce(inputs, preset);
                ttfas.push(data.ttfa_ms);
                totals.push(data.total_time_ms);
                lastDataForChunks = data;
            }
            const median = arr => {
                const s = [...arr].sort((a, b) => a - b);
                const m = Math.floor(s.length / 2);
                return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
            };
            perPresetTimings[preset] = {
                ttfa_median: median(ttfas),
                total_median: median(totals),
                last: lastDataForChunks,
            };
            // Render a summary row for this preset using the last run + median TTFA.
            const row = { ...lastDataForChunks, ttfa_ms: Math.round(median(ttfas) * 10) / 10 };
            labAppendResultRow(row, `${preset} (median of ${inputs.runs})`);
        }
        if (lastDataForChunks) {
            labRenderStages(lastDataForChunks);
            labRenderChunks(lastDataForChunks);
        }

        const summaryLines = Object.entries(perPresetTimings).map(
            ([p, t]) => `${p.padEnd(12)} TTFA=${Math.round(t.ttfa_median)} ms, total=${Math.round(t.total_median)} ms`
        );
        document.getElementById('lab-summary').innerHTML =
            `<pre style="background: #f8f9fa; padding: 12px; border-radius: 4px;">${summaryLines.join('\n')}</pre>`;
        labSetStatus(`Compared ${LAB_ALL_PRESETS.length} presets, ${inputs.runs} run(s) each.`, 'success');
    } catch (e) {
        labSetStatus(`Error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}


// ============= Distillation tab =============

const DISTILL_POLL_MS = 2000;
let distillInited = false;
let distillSpeakers = [];
let distillActivePolls = {}; // taskId -> intervalId

function distillInitOnce() {
    if (distillInited) return;
    distillInited = true;

    // Range sliders that mirror their values
    const sliders = [
        ['distill-manifest-nsamples', 'distill-manifest-nsamples-value'],
        ['distill-manifest-min-dur', 'distill-manifest-min-dur-value'],
        ['distill-manifest-max-dur', 'distill-manifest-max-dur-value'],
        ['distill-pairs-steps', 'distill-pairs-steps-value'],
        ['distill-pairs-cfg', 'distill-pairs-cfg-value'],
        ['distill-pairs-nsamples', 'distill-pairs-nsamples-value'],
        ['distill-train-epochs', 'distill-train-epochs-value'],
        ['distill-train-batch', 'distill-train-batch-value'],
        ['distill-train-grad-accum', 'distill-train-grad-accum-value'],
        ['distill-ab-student-steps', 'distill-ab-student-steps-value'],
        ['distill-ab-teacher-steps', 'distill-ab-teacher-steps-value'],
    ];
    for (const [inp, lab] of sliders) {
        const el = document.getElementById(inp);
        const out = document.getElementById(lab);
        if (el && out) {
            const refresh = () => { out.textContent = el.value; };
            el.addEventListener('input', refresh);
            refresh();
        }
    }

    document.getElementById('distill-refresh-btn').addEventListener('click', distillRefresh);
    document.getElementById('distill-speaker').addEventListener('change', () => {
        distillRenderStatus();
        distillPopulateAudioList();
    });

    // Persist Ollama URL across reloads (localStorage, not cookies — same
    // intent, cheaper than sending it on every HTTP request).
    const ollamaUrlEl = document.getElementById('distill-ollama-url');
    if (ollamaUrlEl) {
        const saved = localStorage.getItem('distill_ollama_url');
        if (saved) ollamaUrlEl.value = saved;
        ollamaUrlEl.addEventListener('change', () => {
            localStorage.setItem('distill_ollama_url', ollamaUrlEl.value.trim());
        });
    }

    document.getElementById('distill-snapshot-btn').addEventListener('click', distillSnapshot);
    document.getElementById('distill-manifest-btn').addEventListener('click', distillBuildManifest);
    document.getElementById('distill-pairs-btn').addEventListener('click', distillGeneratePairs);
    document.getElementById('distill-train-btn').addEventListener('click', distillTrain);
    document.getElementById('distill-ab-btn').addEventListener('click', distillAbEval);
    document.getElementById('distill-activate-btn').addEventListener('click', distillActivate);
    document.getElementById('distill-deactivate-btn').addEventListener('click', distillDeactivate);
    document.getElementById('distill-ollama-probe-btn').addEventListener('click', distillProbeOllama);
    document.getElementById('distill-syn-btn').addEventListener('click', distillSyntheticManifest);

    // Slider value mirror for synthetic n_samples
    const synN = document.getElementById('distill-syn-nsamples');
    const synNVal = document.getElementById('distill-syn-nsamples-value');
    if (synN && synNVal) {
        const refresh = () => { synNVal.textContent = synN.value; };
        synN.addEventListener('input', refresh);
        refresh();
    }
}

async function distillProbeOllama() {
    const url = document.getElementById('distill-ollama-url').value.trim() || 'http://localhost:11434';
    const select = document.getElementById('distill-ollama-model');
    const btn = document.getElementById('distill-ollama-probe-btn');
    btn.disabled = true; btn.textContent = '⏳ Probing…';
    try {
        const r = await fetchJSON(`/distill/ollama-models?url=${encodeURIComponent(url)}`);
        if (!r.ok) {
            select.innerHTML = '<option value="">(probe failed)</option>';
            showNotification(`Ollama probe failed: ${r.error}`, 'error');
            return;
        }
        if (!r.models.length) {
            select.innerHTML = '<option value="">(no models installed)</option>';
            showNotification('No models found. Run `ollama pull llama3.2` first.', 'warning');
            return;
        }
        select.innerHTML = r.models.map(m => `<option value="${m}">${m}</option>`).join('');
        showNotification(`Found ${r.models.length} model(s)`, 'success');
    } catch (e) {
        showNotification(`Probe error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false; btn.textContent = '🔍 Probe';
    }
}

async function distillSyntheticManifest() {
    const speaker = getSpeakerName(); if (!speaker) return;

    const stylePrompt = document.getElementById('distill-style-prompt').value.trim();
    if (!stylePrompt) {
        showNotification('Speaker style description is required.', 'error');
        return;
    }
    let refAudio = document.getElementById('distill-syn-ref').value.trim();
    if (!refAudio) {
        // Fall back to the Stage 2 reference field
        refAudio = document.getElementById('distill-manifest-ref').value.trim();
    }
    if (!refAudio) {
        showNotification('Reference audio is required (either in this section or Stage 2 above).', 'error');
        return;
    }
    let model = document.getElementById('distill-ollama-model').value.trim();
    if (!model) {
        showNotification('Pick an Ollama model first (click Probe).', 'error');
        return;
    }

    const body = {
        speaker,
        reference_audio: refAudio,
        style_prompt: stylePrompt,
        num_records: parseInt(document.getElementById('distill-syn-num').value, 10),
        n_samples: parseInt(document.getElementById('distill-syn-nsamples').value, 10),
        ollama_url: document.getElementById('distill-ollama-url').value.trim() || 'http://localhost:11434',
        ollama_model: model,
        batch_size: parseInt(document.getElementById('distill-syn-batch').value, 10),
        min_words: parseInt(document.getElementById('distill-syn-min-words').value, 10),
        max_words: parseInt(document.getElementById('distill-syn-max-words').value, 10),
        extra_instructions: document.getElementById('distill-syn-extra').value.trim() || null,
    };
    try {
        const r = await postJSON('/distill/append-synthetic-manifest', body);
        startTaskPoll(r.task_id, 'distill-syn-progress');
    } catch (e) {
        showNotification(e.message, 'error');
    }
}

async function distillRefresh() {
    try {
        const data = await fetchJSON('/distill/speakers');
        distillSpeakers = data.speakers || [];

        // Populate the dropdown — preserve selection if possible
        const select = document.getElementById('distill-speaker');
        const prev = select.value;
        select.innerHTML = distillSpeakers
            .map(s => `<option value="${s.name}">${s.name}${s.has_lora ? '' : ' (no LoRA)'}${s.is_active_student ? ' ⚡' : ''}</option>`)
            .join('');
        if (prev && distillSpeakers.find(s => s.name === prev)) select.value = prev;

        // Active student banner
        const banner = document.getElementById('distill-active-banner');
        if (data.active_student_path) {
            const activeSp = distillSpeakers.find(s => s.is_active_student);
            banner.className = 'distill-active-banner has-active';
            banner.innerHTML = `⚡ <strong>Active student:</strong> ${data.active_student_path} (${data.active_student_size_mb} MB)`
                + (activeSp ? ` — matches <strong>${activeSp.name}</strong>` : ' — no matching speaker found');
        } else {
            banner.className = 'distill-active-banner none-active';
            banner.textContent = 'No distilled student active. Streaming uses the full-step teacher.';
        }

        distillRenderStatus();
        distillPopulateAudioList();
    } catch (e) {
        showNotification(`Failed to load distillation status: ${e.message}`, 'error');
    }
}

async function distillPopulateAudioList() {
    // Refresh the shared <datalist> behind both reference-audio inputs so users
    // get clickable autocomplete of the speaker's actual dataset clips.
    const speaker = document.getElementById('distill-speaker').value;
    const dl = document.getElementById('distill-audio-list');
    if (!speaker || !dl) return;
    try {
        const r = await fetchJSON(`/distill/list-audio?speaker=${encodeURIComponent(speaker)}`);
        const files = r.files || [];
        dl.innerHTML = files.map(f => `<option value="${f}"></option>`).join('');
    } catch (e) {
        // Non-fatal — text input still works.
        dl.innerHTML = '';
    }
}

function distillCurrentSpeaker() {
    const s = document.getElementById('distill-speaker').value;
    return distillSpeakers.find(x => x.name === s);
}

function distillRenderStatus() {
    const sp = distillCurrentSpeaker();
    const target = document.getElementById('distill-status');
    if (!sp) {
        target.innerHTML = '<div class="distill-status-item missing">No speaker selected</div>';
        return;
    }
    const items = [
        ['has_lora', 'S2Mel LoRA', sp.has_lora ? 'yes' : 'missing'],
        ['has_csv', 'Verbatim CSV', sp.has_csv ? 'yes' : 'missing'],
        ['has_teacher', 'Teacher snapshot', sp.has_teacher ? 'yes' : 'missing'],
        ['has_manifest', 'Reflow manifest', sp.has_manifest ? `${sp.manifest_entries} entries` : 'missing'],
        ['pairs', 'Paired data', sp.pair_count > 0 ? `${sp.pair_count} files` : 'none'],
        ['has_student', 'Student checkpoint', sp.has_student ? 'yes' : 'missing'],
        ['is_active', 'Active right now', sp.is_active_student ? '⚡ YES' : 'no'],
    ];
    target.innerHTML = items.map(([k, label, val]) => {
        let cls = 'distill-status-item';
        if (k === 'is_active' && sp.is_active_student) cls += ' active';
        else if (val === 'missing' || val === 'none') cls += ' missing';
        else cls += ' ok';
        return `<div class="${cls}"><div class="label">${label}</div><div class="value">${val}</div></div>`;
    }).join('');

    // Stage badges
    setBadge('snapshot', sp.has_teacher);
    setBadge('manifest', sp.has_manifest);
    setBadge('pairs', sp.pair_count > 0);
    setBadge('train', sp.has_student);
    setBadge('ab', sp.has_student);  // can A/B when student exists
    setBadge('activate', sp.is_active_student);
}

function setBadge(name, done) {
    document.querySelectorAll(`.stage-badge[data-stage="${name}"]`).forEach(b => {
        b.classList.toggle('done', !!done);
    });
}

function getSpeakerName() {
    const sp = distillCurrentSpeaker();
    if (!sp) { showNotification('Select a speaker first', 'error'); return null; }
    return sp.name;
}

async function distillSnapshot() {
    const speaker = getSpeakerName(); if (!speaker) return;
    const merge_lora = document.getElementById('distill-merge-lora').checked;
    const force = document.getElementById('distill-snapshot-force').checked;
    try {
        const r = await postJSON('/distill/snapshot-teacher', { speaker, merge_lora, force });
        startTaskPoll(r.task_id, 'distill-snapshot-progress');
    } catch (e) { showNotification(e.message, 'error'); }
}

async function distillBuildManifest() {
    const speaker = getSpeakerName(); if (!speaker) return;
    const ref = document.getElementById('distill-manifest-ref').value.trim();
    const limitStr = document.getElementById('distill-manifest-limit').value.trim();
    const body = {
        speaker,
        reference_audio: ref || null,
        reference_from_row: !ref,
        n_samples: parseInt(document.getElementById('distill-manifest-nsamples').value, 10),
        min_duration: parseFloat(document.getElementById('distill-manifest-min-dur').value),
        max_duration: parseFloat(document.getElementById('distill-manifest-max-dur').value),
        limit: limitStr ? parseInt(limitStr, 10) : null,
    };
    const wrap = document.getElementById('distill-manifest-progress');
    wrap.className = 'task-progress active';
    wrap.innerHTML = `<div>Building manifest…</div>`;
    try {
        const r = await postJSON('/distill/build-manifest', body);
        wrap.innerHTML = `<div>✅ Wrote ${r.entries} records to ${r.manifest}</div>`;
        distillRefresh();
    } catch (e) {
        wrap.innerHTML = `<div style="color:#d63031;">❌ ${e.message}</div>`;
    }
}

async function distillGeneratePairs() {
    const speaker = getSpeakerName(); if (!speaker) return;
    const limitStr = document.getElementById('distill-pairs-limit').value.trim();
    const body = {
        speaker,
        teacher_steps: parseInt(document.getElementById('distill-pairs-steps').value, 10),
        teacher_cfg: parseFloat(document.getElementById('distill-pairs-cfg').value),
        n_samples: parseInt(document.getElementById('distill-pairs-nsamples').value, 10),
        limit: limitStr ? parseInt(limitStr, 10) : null,
    };
    try {
        const r = await postJSON('/distill/generate-pairs', body);
        startTaskPoll(r.task_id, 'distill-pairs-progress');
    } catch (e) { showNotification(e.message, 'error'); }
}

async function distillTrain() {
    const speaker = getSpeakerName(); if (!speaker) return;
    const body = {
        speaker,
        epochs: parseInt(document.getElementById('distill-train-epochs').value, 10),
        batch_size: parseInt(document.getElementById('distill-train-batch').value, 10),
        grad_accumulation: parseInt(document.getElementById('distill-train-grad-accum').value, 10),
        learning_rate: parseFloat(document.getElementById('distill-train-lr').value),
        resume: document.getElementById('distill-train-resume').checked,
    };
    try {
        const r = await postJSON('/distill/train', body);
        startTaskPoll(r.task_id, 'distill-train-progress');
    } catch (e) { showNotification(e.message, 'error'); }
}

async function distillAbEval() {
    const speaker = getSpeakerName(); if (!speaker) return;
    const body = {
        speaker,
        text: document.getElementById('distill-ab-text').value,
        student_steps: parseInt(document.getElementById('distill-ab-student-steps').value, 10),
        student_solver: document.getElementById('distill-ab-student-solver').value,
        teacher_steps: parseInt(document.getElementById('distill-ab-teacher-steps').value, 10),
        teacher_solver: 'heun',
    };
    try {
        const r = await postJSON('/distill/ab-eval', body);
        startTaskPoll(r.task_id, 'distill-ab-progress', (task) => {
            if (task.status === 'completed' && task.result) {
                const r = task.result;
                const target = document.getElementById('distill-ab-result');
                target.className = 'distill-ab-result has-result';
                target.innerHTML = `
                    <div class="ab-pane">
                        <h4>Teacher</h4>
                        <audio controls src="${r.teacher_wav}"></audio>
                    </div>
                    <div class="ab-pane">
                        <h4>Student</h4>
                        <audio controls src="${r.student_wav}"></audio>
                    </div>`;
            }
        });
    } catch (e) { showNotification(e.message, 'error'); }
}

async function distillActivate() {
    const speaker = getSpeakerName(); if (!speaker) return;
    const wrap = document.getElementById('distill-activate-progress');
    wrap.className = 'task-progress active';
    wrap.innerHTML = `<div>Copying student to checkpoints/s2mel_distilled.pth…</div>`;
    try {
        const r = await postJSON('/distill/activate', { speaker });
        wrap.innerHTML = `<div>✅ Active. ${r.note || ''} <button class="btn btn-secondary" id="distill-reload-base-btn">Reload base model now</button></div>`;
        document.getElementById('distill-reload-base-btn').addEventListener('click', async () => {
            wrap.innerHTML = `<div>Reloading base model…</div>`;
            try {
                await postJSON('/models/load/base', {});
                wrap.innerHTML = `<div>✅ Base model reloaded with distilled CFM</div>`;
                distillRefresh();
            } catch (e) {
                wrap.innerHTML = `<div style="color:#d63031;">❌ Reload failed: ${e.message}</div>`;
            }
        });
        distillRefresh();
    } catch (e) {
        wrap.innerHTML = `<div style="color:#d63031;">❌ ${e.message}</div>`;
    }
}

async function distillDeactivate() {
    const wrap = document.getElementById('distill-activate-progress');
    wrap.className = 'task-progress active';
    try {
        const r = await postJSON('/distill/deactivate', {});
        wrap.innerHTML = `<div>✅ Deactivated. ${r.note || 'Reload the base model to revert to teacher.'}</div>`;
        distillRefresh();
    } catch (e) {
        wrap.innerHTML = `<div style="color:#d63031;">❌ ${e.message}</div>`;
    }
}

// Background-task polling. `extraOnComplete` (optional) is invoked once when the task completes.
function startTaskPoll(taskId, wrapId, extraOnComplete) {
    const wrap = document.getElementById(wrapId);
    wrap.className = 'task-progress active';
    wrap.innerHTML = `
        <div><strong>${taskId}</strong> queued…</div>
        <div class="bar"><div class="bar-fill" style="width:0%"></div></div>
        <div class="meta"><span class="status-text">starting</span><span class="msg-text"></span></div>
    `;
    document.getElementById('distill-log-card').style.display = 'block';
    document.getElementById('distill-log-task').textContent = ` (${taskId})`;

    if (distillActivePolls[taskId]) clearInterval(distillActivePolls[taskId]);

    distillActivePolls[taskId] = setInterval(async () => {
        try {
            const task = await fetchJSON(`/distill/tasks/${taskId}?log_lines=300`);
            const pct = Math.round((task.progress || 0) * 100);
            wrap.querySelector('.bar-fill').style.width = pct + '%';
            wrap.querySelector('.status-text').textContent = `${task.status} (${pct}%)`;
            wrap.querySelector('.msg-text').textContent = task.message || '';

            // Stream log tail
            const log = document.getElementById('distill-log');
            if (log && task.log_tail) {
                log.textContent = task.log_tail.join('\n');
                log.scrollTop = log.scrollHeight;
            }

            if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
                clearInterval(distillActivePolls[taskId]);
                delete distillActivePolls[taskId];
                if (extraOnComplete) try { extraOnComplete(task); } catch (e) { console.error(e); }
                distillRefresh();
            }
        } catch (e) {
            console.error(`poll ${taskId}:`, e);
        }
    }, DISTILL_POLL_MS);
}

// Small JSON helpers if not already present.
async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
}
async function postJSON(url, body) {
    const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!r.ok) {
        let err;
        try { err = (await r.json()).detail || r.statusText; } catch { err = r.statusText; }
        throw new Error(`${r.status} ${err}`);
    }
    return await r.json();
}


// ============= Inference tab — distilled-CFM status =============
//
// The Inference tab pulls /distill/active-status on entry and renders a banner
// describing whether a distilled student is loaded. The "Use distilled defaults"
// button populates the solver/steps/CFG override fields with values the backend
// suggests; those fields are then included in the streaming request body.

let inferDistillStatusInited = false;

function initInferDistillHandlersOnce() {
    if (inferDistillStatusInited) return;
    inferDistillStatusInited = true;
    const def = document.getElementById('inferDistillDefaultsBtn');
    const clr = document.getElementById('inferClearOverridesBtn');
    if (def) def.addEventListener('click', applyInferDistillDefaults);
    if (clr) clr.addEventListener('click', clearInferOverrides);
}

async function refreshInferDistillStatus() {
    initInferDistillHandlersOnce();
    const banner = document.getElementById('inferDistillStatus');
    if (!banner) return;
    try {
        const r = await fetchJSON('/distill/active-status');
        // Persist for the "Use distilled defaults" button.
        window.__inferDistillStatus = r;

        if (!r.active_on_disk) {
            banner.className = 'distill-active-banner none-active';
            banner.style.display = 'block';
            banner.innerHTML = `🔹 <strong>No distilled student active.</strong> Inference is using the teacher CFM (full diffusion steps).`;
            return;
        }

        if (r.in_use) {
            const who = r.speaker_match ? ` (matches <strong>${r.speaker_match}</strong>)` : '';
            banner.className = 'distill-active-banner has-active';
            banner.style.display = 'block';
            banner.innerHTML = `
                ⚡ <strong>Distilled student is loaded</strong>${who}.
                Path: <code>${r.loaded_distilled_path}</code> (${r.active_size_mb} MB).
                Use solver <code>${r.suggested.solver}</code> with ${r.suggested.diffusion_steps} step(s) for the speed-up —
                <button type="button" class="btn btn-small" id="inferDistillQuickApply">Apply now</button>
            `;
            const qa = document.getElementById('inferDistillQuickApply');
            if (qa) qa.addEventListener('click', applyInferDistillDefaults);
        } else if (r.needs_reload) {
            const who = r.speaker_match ? ` (matches <strong>${r.speaker_match}</strong>)` : '';
            banner.className = 'distill-active-banner none-active';
            banner.style.display = 'block';
            banner.innerHTML = `
                ⚠️ Distilled student is <strong>on disk but not loaded</strong>${who}.
                <button type="button" class="btn btn-small" id="inferReloadBaseBtn">Reload base model now</button>
                to pick it up.
            `;
            const rb = document.getElementById('inferReloadBaseBtn');
            if (rb) rb.addEventListener('click', async () => {
                rb.disabled = true; rb.textContent = '⏳ reloading…';
                try {
                    await postJSON('/models/load/base', {});
                    showNotification('Base model reloaded with distilled CFM.', 'success');
                    refreshInferDistillStatus();
                } catch (e) {
                    showNotification(`Reload failed: ${e.message}`, 'error');
                    rb.disabled = false; rb.textContent = 'Reload base model now';
                }
            });
        } else {
            banner.className = 'distill-active-banner none-active';
            banner.style.display = 'block';
            banner.innerHTML = `🔹 No active student — using teacher CFM.`;
        }
    } catch (e) {
        // Endpoint not reachable — hide banner silently, this is non-critical
        if (banner) banner.style.display = 'none';
    }
}

function applyInferDistillDefaults() {
    const status = window.__inferDistillStatus;
    if (!status || !status.suggested) {
        showNotification('Active-status not loaded yet.', 'warning');
        return;
    }
    const sug = status.suggested;
    const solverEl = document.getElementById('inferSolverOverride');
    const stepsEl = document.getElementById('inferStepsOverride');
    const cfgEl = document.getElementById('inferCfgOverride');
    if (solverEl) solverEl.value = sug.solver || '';
    if (stepsEl) stepsEl.value = sug.diffusion_steps != null ? sug.diffusion_steps : '';
    if (cfgEl) cfgEl.value = sug.inference_cfg != null ? sug.inference_cfg : '';
    showNotification(`Set solver=${sug.solver}, steps=${sug.diffusion_steps}, cfg=${sug.inference_cfg}`, 'success');
}

function clearInferOverrides() {
    ['inferSolverOverride', 'inferStepsOverride', 'inferCfgOverride'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    showNotification('Overrides cleared — preset values will be used.', 'info');
}

function inferOverridePayload() {
    const payload = {};
    const solver = document.getElementById('inferSolverOverride')?.value.trim();
    if (solver) payload.solver_override = solver;
    const steps = document.getElementById('inferStepsOverride')?.value.trim();
    if (steps) payload.diffusion_steps_override = parseInt(steps, 10);
    const cfg = document.getElementById('inferCfgOverride')?.value.trim();
    if (cfg !== '' && cfg != null) payload.inference_cfg_override = parseFloat(cfg);
    return payload;
}


// ============= Header GPU stats =============
//
// Polls /system/gpu-stats every 5s and shows "reserved / total MB". The 🧹 button
// triggers /system/gpu-cleanup. Streaming auto-cleans on completion (see
// release_cuda_cache_on_done in StreamingConfigV2) so the readout drops naturally
// after each request — the manual button is for cases where you want to see the
// real live footprint vs the cached pool.

let gpuStatsTimer = null;

async function refreshGpuStats() {
    const el = document.getElementById('gpuStatsInline');
    if (!el) return;
    try {
        const s = await fetchJSON('/system/gpu-stats');
        if (!s.available) {
            el.textContent = 'no cuda';
            return;
        }
        const reserved = s.torch_reserved_mb;
        const total = s.total_mb;
        const pct = reserved / total;
        el.textContent = `${Math.round(reserved)} / ${Math.round(total)} MB`;
        el.title = `PyTorch allocated: ${s.torch_allocated_mb} MB
PyTorch reserved (cache pool): ${s.torch_reserved_mb} MB
Cache (reusable, releasable): ${s.torch_cache_mb} MB
Free system-wide: ${s.free_mb} MB
Used by other processes: ${s.used_by_other_mb} MB
Total GPU: ${s.total_mb} MB`;
        el.classList.remove('warn', 'crit');
        if (pct > 0.85) el.classList.add('crit');
        else if (pct > 0.65) el.classList.add('warn');
    } catch (e) {
        el.textContent = 'gpu?';
    }
}

function startGpuStatsPolling() {
    refreshGpuStats();
    if (gpuStatsTimer) clearInterval(gpuStatsTimer);
    gpuStatsTimer = setInterval(refreshGpuStats, 5000);
    const btn = document.getElementById('gpuCleanupBtn');
    if (btn) {
        btn.addEventListener('click', async () => {
            btn.disabled = true;
            const orig = btn.textContent;
            btn.textContent = '⏳ Freeing…';
            try {
                const r = await postJSON('/system/gpu-cleanup', {});
                if (r.ok) {
                    showNotification(`Freed ${r.freed_mb} MB (reserved: ${r.reserved_before_mb} → ${r.reserved_after_mb} MB)`, 'success');
                } else {
                    showNotification(`Cleanup skipped: ${r.reason || 'unknown'}`, 'warning');
                }
                refreshGpuStats();
            } catch (e) {
                showNotification(`Cleanup failed: ${e.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = orig;
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', startGpuStatsPolling);