# LocalChat - English Conversation Practice

Real-time English conversation practice web application with local AI and TTS. No cloud APIs required - everything runs on your machine for complete privacy.

## Features

- **Real-time Voice Conversation**: Practice English speaking with AI-powered responses
- **Supertonic TTS**: High-quality text-to-speech using ONNX (works on Mac/Windows/Linux)
- **LM Studio Integration**: Local AI conversation using any compatible model
- **Speech Recognition**: Speak directly using your microphone (Web Speech API)
- **WebSocket Communication**: Low-latency real-time messaging
- **Responsive UI**: Works on desktop and mobile devices
- **Conversation History**: Context-aware responses that remember the conversation
- **100% Local**: No cloud APIs, complete privacy

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- [LM Studio](https://lmstudio.ai/) running with a loaded model
- Git LFS (for downloading TTS models)

### Installation by Platform

#### macOS

```bash
# Install uv and git-lfs
brew install uv git-lfs
git lfs install

# Clone and setup
git clone https://github.com/raym33/localchat.git
cd localchat
git clone https://huggingface.co/Supertone/supertonic assets
uv sync
```

#### Windows (PowerShell)

```powershell
# Install uv (run as Administrator or use winget)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Git LFS (download from https://git-lfs.com or use winget)
winget install GitHub.GitLFS
git lfs install

# Clone and setup
git clone https://github.com/raym33/localchat.git
cd localchat
git clone https://huggingface.co/Supertone/supertonic assets
uv sync
```

#### Linux (Ubuntu/Debian)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install git-lfs
sudo apt-get install git-lfs
git lfs install

# Clone and setup
git clone https://github.com/raym33/localchat.git
cd localchat
git clone https://huggingface.co/Supertone/supertonic assets
uv sync
```

#### Linux (Fedora/RHEL)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install git-lfs
sudo dnf install git-lfs
git lfs install

# Clone and setup
git clone https://github.com/raym33/localchat.git
cd localchat
git clone https://huggingface.co/Supertone/supertonic assets
uv sync
```

### Configuration (Optional)

```bash
cp .env.example .env
# Edit .env with your LM Studio settings
```

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a model (e.g., `qwen3-4b` or any chat model)
3. Start the local server in LM Studio (default port: 1234)
4. Update `LM_STUDIO_URL` in config if needed

### Running the App

```bash
uv run python run.py
```

Or directly with uvicorn:
```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Usage

1. Click "Start Conversation" to begin
2. Type your message or click the microphone to speak
3. The AI will respond with text and voice
4. Practice your English conversation skills!

### Voice Input

- Click the microphone button to start speaking
- Speak clearly in English
- The app will automatically transcribe and send your message

> **Note**: For microphone access on localhost, use `http://localhost:8000` (not 127.0.0.1)

### Available Voices

The TTS system includes 4 voices:
- **F1, F2**: Female voices
- **M1, M2**: Male voices

Default is F1. You can change it in the code.

### Settings

- **Voice Speed**: Adjust playback speed (0.5x - 2.0x)
- **Auto-play Voice**: Toggle automatic audio playback
- **Show Transcription**: Show speech-to-text results while speaking
- **Speech Language**: Choose English variant (US, UK, Australian)

## Architecture

```
localchat/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── conversation.py    # Conversation logic & LM Studio integration
│   ├── tts_service.py     # Supertonic TTS wrapper
│   └── main.py            # FastAPI application
├── frontend/
│   └── index.html         # Main HTML page
├── static/
│   ├── css/
│   │   └── styles.css     # Application styles
│   └── js/
│       └── app.js         # Frontend JavaScript
├── assets/                # TTS models (downloaded separately)
│   ├── onnx/              # ONNX model files
│   └── voice_styles/      # Voice style files (F1, F2, M1, M2)
├── .env.example           # Environment template
├── pyproject.toml         # Python dependencies
├── run.py                 # Server runner
└── README.md
```

## API Endpoints

### REST API

- `GET /` - Serve the main application
- `GET /health` - Health check
- `POST /api/speak` - Generate speech from text
- `POST /api/conversation` - Send message and get response with audio
- `POST /api/conversation/new` - Create new conversation
- `DELETE /api/conversation/{id}` - Delete conversation

### WebSocket

- `WS /ws/{client_id}` - Real-time conversation

Message types:
- `message`: Send user text
- `clear`: Clear conversation history
- `ping`: Keep-alive

Response types:
- `ack`: Message received
- `response`: AI text response
- `audio`: Audio data (base64 WAV)
- `cleared`: Conversation cleared
- `pong`: Keep-alive response

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234` | LM Studio server URL |
| `LM_STUDIO_MODEL` | `qwen3-4b-thinking-2507` | Model name in LM Studio |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## Thinking Models

If you're using a "thinking" model (like Qwen thinking variants), the app automatically strips `<think>...</think>` tags from responses, showing only the final answer.

## Troubleshooting

### No audio playing
- Make sure the `assets/` folder contains the ONNX models
- Check browser console for errors

### Microphone not working
- Use `http://localhost:8000` instead of `http://127.0.0.1:8000`
- Grant microphone permissions when prompted
- Try a different browser (Safari/Firefox are less restrictive)

### LM Studio connection failed
- Verify LM Studio server is running
- Check the URL in config matches your LM Studio settings
- Make sure a model is loaded in LM Studio

## Acknowledgments

- [Supertone](https://github.com/supertone-inc/supertonic) for the TTS model
- [LM Studio](https://lmstudio.ai/) for local LLM inference
- Web Speech API for browser-based speech recognition

## License

MIT
