/**
 * Dia2 English Conversation App
 * Real-time English conversation practice with AI
 */

class ConversationApp {
    constructor() {
        // State
        this.conversationId = this.generateId();
        this.isRecording = false;
        this.isConnected = false;
        this.ws = null;
        this.recognition = null;
        this.audioContext = null;
        this.currentAudio = null;

        // Settings
        this.settings = {
            voiceSpeed: 1.0,
            autoPlayVoice: true,
            showTranscription: true,
            speechLanguage: 'en-US'
        };

        // Load saved settings
        this.loadSettings();

        // DOM Elements
        this.elements = {
            welcomeScreen: document.getElementById('welcomeScreen'),
            chatContainer: document.getElementById('chatContainer'),
            messagesContainer: document.getElementById('messagesContainer'),
            inputArea: document.getElementById('inputArea'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            voiceBtn: document.getElementById('voiceBtn'),
            typingIndicator: document.getElementById('typingIndicator'),
            recordingIndicator: document.getElementById('recordingIndicator'),
            connectionStatus: document.getElementById('connectionStatus'),
            settingsModal: document.getElementById('settingsModal'),
            settingsBtn: document.getElementById('settingsBtn'),
            startConversationBtn: document.getElementById('startConversationBtn'),
            newConversationBtn: document.getElementById('newConversationBtn')
        };

        // Initialize
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupSpeechRecognition();
        this.connect();
    }

    generateId() {
        return 'conv_' + Math.random().toString(36).substring(2, 15);
    }

    // Settings Management
    loadSettings() {
        const saved = localStorage.getItem('conversationSettings');
        if (saved) {
            try {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        }
    }

    saveSettings() {
        localStorage.setItem('conversationSettings', JSON.stringify(this.settings));
    }

    // WebSocket Connection
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.conversationId}`;

        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus('connected');
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('disconnected');

                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connect(), 3000);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
        } catch (error) {
            console.error('Failed to connect:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    updateConnectionStatus(status) {
        const statusElement = this.elements.connectionStatus;
        const textElement = statusElement.querySelector('.status-text');

        statusElement.className = 'connection-status ' + status;

        switch (status) {
            case 'connected':
                textElement.textContent = 'Connected';
                break;
            case 'connecting':
                textElement.textContent = 'Connecting...';
                break;
            case 'disconnected':
                textElement.textContent = 'Disconnected';
                break;
        }
    }

    // Message Handling
    handleMessage(data) {
        switch (data.type) {
            case 'ack':
                console.log('Message acknowledged:', data.user_text);
                this.showTypingIndicator();
                break;

            case 'response':
                this.hideTypingIndicator();
                this.addMessage('assistant', data.text);
                break;

            case 'audio':
                this.playAudio(data.audio);
                break;

            case 'cleared':
                this.clearMessages();
                break;

            case 'pong':
                console.log('Pong received');
                break;
        }
    }

    sendMessage(text) {
        if (!text.trim() || !this.isConnected) return;

        // Add user message to chat
        this.addMessage('user', text);

        // Clear input
        this.elements.messageInput.value = '';
        this.autoResizeTextarea();

        // Send via WebSocket
        this.ws.send(JSON.stringify({
            type: 'message',
            text: text
        }));
    }

    addMessage(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;

        const avatar = role === 'user' ? '👤' : '🤖';
        const audioControls = role === 'assistant' ? this.createAudioControls() : '';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${this.escapeHtml(text)}</div>
                ${audioControls}
            </div>
        `;

        this.elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    createAudioControls() {
        return `
            <div class="message-actions">
                <button class="message-action replay-audio" title="Replay audio">
                    <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                </button>
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearMessages() {
        this.elements.messagesContainer.innerHTML = '';
    }

    scrollToBottom() {
        const container = this.elements.chatContainer;
        container.scrollTop = container.scrollHeight;
    }

    showTypingIndicator() {
        this.elements.typingIndicator.classList.remove('hidden');
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.elements.typingIndicator.classList.add('hidden');
    }

    // Audio Playback
    async playAudio(base64Audio) {
        if (!this.settings.autoPlayVoice) return;

        try {
            // Convert base64 to blob
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const blob = new Blob([bytes], { type: 'audio/wav' });

            // Create audio URL
            const audioUrl = URL.createObjectURL(blob);

            // Stop any currently playing audio
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }

            // Play audio
            this.currentAudio = new Audio(audioUrl);
            this.currentAudio.playbackRate = this.settings.voiceSpeed;

            await this.currentAudio.play();

            // Clean up URL when done
            this.currentAudio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
        } catch (error) {
            console.error('Failed to play audio:', error);
        }
    }

    // Speech Recognition
    setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported');
            this.elements.voiceBtn.style.display = 'none';
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = this.settings.speechLanguage;

        this.recognition.onstart = () => {
            this.isRecording = true;
            this.elements.voiceBtn.classList.add('recording');
            this.elements.recordingIndicator.classList.remove('hidden');
            this.elements.voiceBtn.querySelector('.voice-status').textContent = 'Listening...';
        };

        this.recognition.onend = () => {
            this.isRecording = false;
            this.elements.voiceBtn.classList.remove('recording');
            this.elements.recordingIndicator.classList.add('hidden');
            this.elements.voiceBtn.querySelector('.voice-status').textContent = 'Click to speak';
        };

        this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            // Show interim results in input
            if (interimTranscript && this.settings.showTranscription) {
                this.elements.messageInput.value = interimTranscript;
                this.autoResizeTextarea();
            }

            // Send final transcript
            if (finalTranscript) {
                this.elements.messageInput.value = '';
                this.sendMessage(finalTranscript);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isRecording = false;
            this.elements.voiceBtn.classList.remove('recording');
            this.elements.recordingIndicator.classList.add('hidden');
            this.elements.voiceBtn.querySelector('.voice-status').textContent = 'Click to speak';

            if (event.error === 'not-allowed') {
                alert('Microphone access denied. Please allow microphone access to use voice input.');
            }
        };
    }

    toggleRecording() {
        if (this.isRecording) {
            this.recognition.stop();
        } else {
            // Stop any playing audio
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }

            try {
                this.recognition.lang = this.settings.speechLanguage;
                this.recognition.start();
            } catch (error) {
                console.error('Failed to start recognition:', error);
            }
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Start conversation
        this.elements.startConversationBtn.addEventListener('click', () => {
            this.startConversation();
        });

        // New conversation
        this.elements.newConversationBtn.addEventListener('click', () => {
            this.newConversation();
        });

        // Send message
        this.elements.sendBtn.addEventListener('click', () => {
            this.sendMessage(this.elements.messageInput.value);
        });

        // Enter to send
        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage(this.elements.messageInput.value);
            }
        });

        // Auto-resize textarea
        this.elements.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });

        // Voice button
        this.elements.voiceBtn.addEventListener('click', () => {
            this.toggleRecording();
        });

        // Settings
        this.elements.settingsBtn.addEventListener('click', () => {
            this.openSettings();
        });

        // Close settings modal
        this.elements.settingsModal.querySelector('.modal-close').addEventListener('click', () => {
            this.closeSettings();
        });

        this.elements.settingsModal.querySelector('.modal-overlay').addEventListener('click', () => {
            this.closeSettings();
        });

        // Settings controls
        document.getElementById('voiceSpeed').addEventListener('input', (e) => {
            this.settings.voiceSpeed = parseFloat(e.target.value);
            document.getElementById('voiceSpeedValue').textContent = this.settings.voiceSpeed.toFixed(1) + 'x';
            this.saveSettings();
        });

        document.getElementById('autoPlayVoice').addEventListener('change', (e) => {
            this.settings.autoPlayVoice = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('showTranscription').addEventListener('change', (e) => {
            this.settings.showTranscription = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('speechLanguage').addEventListener('change', (e) => {
            this.settings.speechLanguage = e.target.value;
            if (this.recognition) {
                this.recognition.lang = this.settings.speechLanguage;
            }
            this.saveSettings();
        });

        // Replay audio buttons (event delegation)
        this.elements.messagesContainer.addEventListener('click', (e) => {
            const replayBtn = e.target.closest('.replay-audio');
            if (replayBtn && this.currentAudio) {
                this.currentAudio.currentTime = 0;
                this.currentAudio.play();
            }
        });

        // Keep-alive ping
        setInterval(() => {
            if (this.isConnected && this.ws) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    autoResizeTextarea() {
        const textarea = this.elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    // UI State Management
    startConversation() {
        this.elements.welcomeScreen.classList.add('hidden');
        this.elements.chatContainer.classList.remove('hidden');
        this.elements.inputArea.classList.remove('hidden');

        // Add initial greeting
        this.addMessage('assistant', "Hello! I'm your English conversation partner. I'm here to help you practice speaking English. Feel free to talk about anything - your day, hobbies, interests, or any topic you'd like to discuss. How can I help you today?");
    }

    newConversation() {
        // Clear messages
        this.clearMessages();

        // Generate new conversation ID
        this.conversationId = this.generateId();

        // Reconnect with new ID
        if (this.ws) {
            this.ws.close();
        }
        this.connect();

        // Show welcome or add greeting
        if (this.elements.welcomeScreen.classList.contains('hidden')) {
            this.addMessage('assistant', "Let's start fresh! What would you like to talk about?");
        }
    }

    openSettings() {
        // Update UI with current settings
        document.getElementById('voiceSpeed').value = this.settings.voiceSpeed;
        document.getElementById('voiceSpeedValue').textContent = this.settings.voiceSpeed.toFixed(1) + 'x';
        document.getElementById('autoPlayVoice').checked = this.settings.autoPlayVoice;
        document.getElementById('showTranscription').checked = this.settings.showTranscription;
        document.getElementById('speechLanguage').value = this.settings.speechLanguage;

        this.elements.settingsModal.classList.remove('hidden');
    }

    closeSettings() {
        this.elements.settingsModal.classList.add('hidden');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ConversationApp();
});
