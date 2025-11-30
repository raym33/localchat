# How I Built LocalChat: A Step-by-Step Guide

This guide documents the complete process of building LocalChat, a real-time English conversation practice app with local AI and text-to-speech. Perfect for developers who want to understand the architecture or build something similar.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Step 1: Project Setup](#step-1-project-setup)
4. [Step 2: Backend Architecture](#step-2-backend-architecture)
5. [Step 3: LM Studio Integration](#step-3-lm-studio-integration)
6. [Step 4: Text-to-Speech with Supertonic](#step-4-text-to-speech-with-supertonic)
7. [Step 5: WebSocket Real-time Communication](#step-5-websocket-real-time-communication)
8. [Step 6: Frontend Development](#step-6-frontend-development)
9. [Step 7: Speech Recognition](#step-7-speech-recognition)
10. [Challenges & Solutions](#challenges--solutions)
11. [Lessons Learned](#lessons-learned)

---

## Project Overview

**Goal**: Create a web app where users can practice English conversations with an AI that speaks back, all running locally without cloud APIs.

**Key Requirements**:
- Real-time conversation with AI
- Text-to-speech for AI responses
- Voice input from user's microphone
- 100% local/offline capability
- Cross-platform (Mac, Windows, Linux)

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Backend | FastAPI (Python) | Async support, WebSocket, easy to use |
| LLM | LM Studio / Ollama | Local inference, OpenAI-compatible API |
| TTS | Supertonic (ONNX) | High quality, works on CPU, no CUDA needed |
| Frontend | Vanilla JS + HTML/CSS | Simple, no build step required |
| Speech Input | Web Speech API | Browser-native, no dependencies |
| Communication | WebSocket | Low latency, real-time bidirectional |

---

## Step 1: Project Setup

### 1.1 Initialize the Project

```bash
mkdir localchat
cd localchat
```

### 1.2 Create Project Structure

```
localchat/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── conversation.py
│   ├── tts_service.py
│   └── main.py
├── frontend/
│   └── index.html
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
├── assets/              # TTS models (downloaded separately)
├── pyproject.toml
├── run.py
└── README.md
```

### 1.3 Configure Dependencies (pyproject.toml)

```toml
[project]
name = "localchat"
version = "1.0.0"
description = "English conversation practice with local AI"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "websockets>=12.0",
    "python-multipart>=0.0.6",
    "httpx>=0.26.0",
    "numpy>=1.26.0",
    "onnxruntime>=1.17.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["backend"]
```

Key insight: The `[tool.hatch.build.targets.wheel]` section is essential - without it, hatchling won't know which packages to include.

### 1.4 Install with uv

```bash
uv sync
```

---

## Step 2: Backend Architecture

### 2.1 Configuration (backend/config.py)

```python
from dataclasses import dataclass
import os

@dataclass
class Config:
    lm_studio_url: str = "http://localhost:1234"
    lm_studio_model: str = "qwen3-4b"
    host: str = "0.0.0.0"
    port: int = 8000

    def __post_init__(self):
        # Allow environment variable overrides
        self.lm_studio_url = os.getenv("LM_STUDIO_URL", self.lm_studio_url)
        self.lm_studio_model = os.getenv("LM_STUDIO_MODEL", self.lm_studio_model)

config = Config()
```

### 2.2 FastAPI Application (backend/main.py)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize TTS
    tts_service = get_tts_service()
    await tts_service.initialize()
    yield
    # Shutdown: Cleanup

app = FastAPI(
    title="LocalChat",
    lifespan=lifespan
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
```

---

## Step 3: LM Studio Integration

### 3.1 Why LM Studio?

- Free, local LLM inference
- OpenAI-compatible API (easy to integrate)
- Supports many models (Llama, Qwen, Mistral, etc.)
- Cross-platform GUI

### 3.2 Conversation Manager (backend/conversation.py)

```python
import httpx
from dataclasses import dataclass, field
from datetime import datetime

SYSTEM_PROMPT = """You are a friendly English conversation partner.
Keep responses concise (1-3 sentences). Be encouraging and supportive.
Do NOT use markdown formatting - just plain conversational text."""

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

class ConversationManager:
    def __init__(self, lm_studio_url: str, model: str):
        self.lm_studio_url = lm_studio_url
        self.model = model
        self.conversations: dict[str, list[Message]] = {}
        self._client = httpx.AsyncClient(timeout=60.0)

    async def generate_response(self, conversation_id: str, user_message: str) -> str:
        # Add user message to history
        self.add_message(conversation_id, "user", user_message)

        # Build messages for API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in self.conversations[conversation_id][-20:]:  # Last 20 messages
            messages.append({"role": msg.role, "content": msg.content})

        # Call LM Studio API
        response = await self._client.post(
            f"{self.lm_studio_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.8,
            }
        )

        raw_response = response.json()["choices"][0]["message"]["content"]
        cleaned = self._clean_response(raw_response)

        self.add_message(conversation_id, "assistant", cleaned)
        return cleaned
```

### 3.3 Handling "Thinking" Models

Some models (like Qwen thinking variants) output their reasoning in `<think>...</think>` tags:

```python
def _clean_response(self, text: str) -> str:
    """Extract only the final response, removing thinking tags."""

    # Extract content AFTER </think>
    if '</think>' in text:
        text = text.split('</think>')[-1]
    elif '<think>' in text:
        # Model is still thinking, no response yet
        text = ""

    # Remove any remaining XML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*+', '', text)

    return text.strip()
```

This was a critical bug fix - without it, users would see the raw reasoning output!

---

## Step 4: Text-to-Speech with Supertonic

### 4.1 Why Supertonic?

Initially, I tried [Dia2](https://github.com/nari-labs/dia2) but it requires CUDA (GPU). Supertonic uses ONNX Runtime which works on CPU across all platforms.

### 4.2 Download TTS Assets

```bash
git clone https://huggingface.co/Supertone/supertonic assets
```

This downloads ~250MB of model files:
- `onnx/` - ONNX model files
- `voice_styles/` - Voice style embeddings (F1, F2, M1, M2)

### 4.3 TTS Service Architecture

```python
class SupertonicTTS:
    def __init__(self, assets_path: str):
        self.assets_path = Path(assets_path)
        self.session = None
        self.tokenizer = None
        self.style = None

    async def initialize(self):
        """Load ONNX models (runs in background on startup)."""
        # Load ONNX session
        self.session = ort.InferenceSession(
            str(self.assets_path / "onnx" / "model.onnx"),
            providers=['CPUExecutionProvider']
        )

        # Load tokenizer
        with open(self.assets_path / "onnx" / "indexer.json") as f:
            self.tokenizer = Tokenizer(json.load(f))

        # Load voice style (F1 = female voice 1)
        self.style = Style.load(self.assets_path / "voice_styles" / "F1")

    async def generate_speech(self, text: str) -> bytes:
        """Convert text to WAV audio bytes."""
        # Tokenize
        text_ids, text_mask = self.tokenizer([text])

        # Run inference
        outputs = self.session.run(None, {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "style": self.style.embedding
        })

        # Convert to WAV
        audio = outputs[0]
        return self._to_wav(audio, sample_rate=24000)
```

### 4.4 Bug Fix: Tokenizer Indexer

The original code assumed `indexer` was a dict, but it's actually a list:

```python
# Wrong (throws error)
token_id = self.indexer.get(str(unicode_val), 0)

# Correct
if unicode_val < len(self.indexer):
    token_id = self.indexer[unicode_val]
else:
    token_id = 0
```

---

## Step 5: WebSocket Real-time Communication

### 5.1 Why WebSocket?

- **Low latency**: No HTTP overhead per message
- **Bidirectional**: Server can push audio as soon as it's ready
- **Persistent**: Single connection for entire conversation

### 5.2 WebSocket Endpoint

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket

    async def send_json(self, client_id: str, data: dict):
        if client_id in self.connections:
            await self.connections[client_id].send_json(data)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "message":
                # 1. Acknowledge receipt
                await manager.send_json(client_id, {"type": "ack"})

                # 2. Generate AI response
                response = await conversation_manager.generate_response(
                    client_id, data["text"]
                )
                await manager.send_json(client_id, {
                    "type": "response",
                    "text": response
                })

                # 3. Generate and send audio
                audio = await tts_service.generate_speech(response)
                await manager.send_json(client_id, {
                    "type": "audio",
                    "audio": base64.b64encode(audio).decode()
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
```

### 5.3 Message Flow

```
User types "Hello"
    ↓
[WebSocket] → Backend receives message
    ↓
Backend → LM Studio API (generate response)
    ↓
[WebSocket] ← Send text response to frontend
    ↓
Backend → TTS Service (generate audio)
    ↓
[WebSocket] ← Send audio (base64) to frontend
    ↓
Frontend plays audio
```

---

## Step 6: Frontend Development

### 6.1 HTML Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>LocalChat</title>
    <link rel="stylesheet" href="/static/css/styles.css">
</head>
<body>
    <div class="chat-container">
        <header>
            <h1>English Conversation</h1>
            <button id="newChat">+ New Chat</button>
        </header>

        <div id="messages" class="messages"></div>

        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message...">
            <button id="micButton">🎤</button>
            <button id="sendButton">➤</button>
        </div>
    </div>

    <script src="/static/js/app.js"></script>
</body>
</html>
```

### 6.2 WebSocket Client

```javascript
class ChatApp {
    constructor() {
        this.clientId = crypto.randomUUID();
        this.ws = null;
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(`ws://${location.host}/ws/${this.clientId}`);

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            switch(data.type) {
                case 'response':
                    this.addMessage('assistant', data.text);
                    break;
                case 'audio':
                    this.playAudio(data.audio);
                    break;
            }
        };
    }

    sendMessage(text) {
        this.addMessage('user', text);
        this.ws.send(JSON.stringify({
            type: 'message',
            text: text
        }));
    }

    playAudio(base64Audio) {
        const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
        audio.play();
    }
}
```

---

## Step 7: Speech Recognition

### 7.1 Web Speech API

```javascript
setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        console.warn('Speech recognition not supported');
        return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';  // or 'en-GB', 'en-AU'

    this.recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');

        if (event.results[0].isFinal) {
            this.sendMessage(transcript);
        }
    };
}

startListening() {
    this.recognition.start();
    this.micButton.classList.add('listening');
}
```

### 7.2 Important: Localhost Requirement

Microphone access requires a "secure context". Browsers allow this for:
- `https://` URLs
- `http://localhost:*` (but NOT `http://127.0.0.1:*` in some browsers!)

Always use `http://localhost:8000` instead of `http://127.0.0.1:8000`.

---

## Challenges & Solutions

### Challenge 1: Dia2 TTS Required CUDA

**Problem**: Initial TTS choice (Dia2) only works with NVIDIA GPUs.

**Solution**: Switched to Supertonic which uses ONNX Runtime and works on CPU.

### Challenge 2: Thinking Model Output

**Problem**: Qwen thinking models output `<think>reasoning</think>response`, showing raw reasoning to users.

**Solution**: Parse response to extract only content after `</think>` tag.

### Challenge 3: hatchling Build Error

**Problem**: `Unable to determine which files to ship inside the wheel`

**Solution**: Add `[tool.hatch.build.targets.wheel] packages = ["backend"]` to pyproject.toml.

### Challenge 4: Tokenizer Type Mismatch

**Problem**: `'list' object has no attribute 'get'`

**Solution**: The indexer is a list (index = unicode value, value = token id), not a dict. Changed from `.get()` to index access.

### Challenge 5: max_tokens Too Low

**Problem**: Thinking models need tokens for both `<think>` content AND the response.

**Solution**: Increased `max_tokens` from 200 to 1024.

---

## Lessons Learned

1. **Start simple**: Vanilla JS works fine for this use case. No need for React/Vue.

2. **Local-first is possible**: With LM Studio + ONNX, you can build fully local AI apps.

3. **Test with real models**: "Thinking" models behave differently - test with the actual model users will use.

4. **WebSocket > REST for real-time**: The persistent connection makes a huge UX difference.

5. **Read error messages carefully**: Most bugs were solved by understanding the actual error.

6. **ONNX is underrated**: It enables running ML models anywhere without GPU dependencies.

---

## What's Next?

Check the [Roadmap](../README.md#roadmap) for planned features:
- Voice selection UI
- Conversation export
- Docker support
- Ollama integration

---

## Resources

- [FastAPI WebSocket Docs](https://fastapi.tiangolo.com/advanced/websockets/)
- [LM Studio](https://lmstudio.ai/)
- [Supertonic TTS](https://github.com/supertone-inc/supertonic)
- [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- [ONNX Runtime](https://onnxruntime.ai/)

---

*Built with Claude Code assistance. MIT License.*
