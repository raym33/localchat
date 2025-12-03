"""FastAPI backend for Dia2 English Conversation app."""

import asyncio
import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import config
from .conversation import get_conversation_manager
from .tts_service import get_tts_service

# Try to import pronunciation service (optional dependency)
try:
    from .pronunciation_service import get_pronunciation_service, PRONUNCIATION_AVAILABLE
except ImportError:
    PRONUNCIATION_AVAILABLE = False

    def get_pronunciation_service():
        return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting English Conversation Server...")

    # Initialize TTS service
    tts_service = get_tts_service()
    init_task = asyncio.create_task(tts_service.initialize())

    # Initialize pronunciation service if available
    if PRONUNCIATION_AVAILABLE:
        pronunciation_service = get_pronunciation_service()
        asyncio.create_task(pronunciation_service.initialize())

    yield

    # Cleanup
    logger.info("Shutting down server...")


app = FastAPI(
    title="Dia2 English Conversation",
    description="Real-time English conversation practice with AI using Dia2 TTS",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = PROJECT_ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Pydantic models
class TextInput(BaseModel):
    """Text input for speech generation."""

    text: str
    conversation_id: Optional[str] = None


class ConversationMessage(BaseModel):
    """A message in the conversation."""

    text: str
    conversation_id: str


class ConversationResponse(BaseModel):
    """Response from conversation endpoint."""

    response_text: str
    audio_base64: Optional[str] = None
    conversation_id: str


# REST API endpoints
@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_path = PROJECT_ROOT / "frontend" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Dia2 English Conversation</h1><p>Frontend not found.</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "dia2-english-conversation"}


@app.post("/api/speak", response_model=dict)
async def generate_speech(input_data: TextInput):
    """Generate speech from text.

    Args:
        input_data: Text to convert to speech

    Returns:
        Audio data as base64 encoded WAV
    """
    tts_service = get_tts_service()

    audio_bytes = await tts_service.generate_speech(input_data.text)

    if audio_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to generate speech")

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio": audio_base64,
        "format": "wav",
        "text": input_data.text,
    }


@app.post("/api/conversation", response_model=ConversationResponse)
async def conversation(message: ConversationMessage):
    """Handle a conversation turn.

    Args:
        message: The user's message

    Returns:
        AI response with audio
    """
    conversation_manager = get_conversation_manager()
    tts_service = get_tts_service()

    # Generate AI response
    response_text = await conversation_manager.generate_response(
        message.conversation_id, message.text
    )

    # Generate speech for the response
    audio_bytes = await tts_service.generate_speech(response_text)

    audio_base64 = None
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return ConversationResponse(
        response_text=response_text,
        audio_base64=audio_base64,
        conversation_id=message.conversation_id,
    )


@app.post("/api/conversation/new")
async def new_conversation():
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation_manager = get_conversation_manager()
    conversation_manager.create_conversation(conversation_id)

    return {"conversation_id": conversation_id}


@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    conversation_manager = get_conversation_manager()
    conversation_manager.delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


# WebSocket for real-time conversation
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected")

    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)


connection_manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time conversation."""
    await connection_manager.connect(websocket, client_id)

    conversation_manager = get_conversation_manager()
    tts_service = get_tts_service()

    # Create a conversation for this client
    conversation_manager.create_conversation(client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message_type = data.get("type", "message")

            if message_type == "message":
                user_text = data.get("text", "")

                if not user_text.strip():
                    continue

                # Send acknowledgment
                await connection_manager.send_json(
                    client_id,
                    {"type": "ack", "status": "processing", "user_text": user_text},
                )

                # Generate AI response
                response_text = await conversation_manager.generate_response(
                    client_id, user_text
                )

                # Send text response immediately
                await connection_manager.send_json(
                    client_id,
                    {
                        "type": "response",
                        "text": response_text,
                        "status": "generating_audio",
                    },
                )

                # Generate speech
                audio_bytes = await tts_service.generate_speech(response_text)

                if audio_bytes:
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    await connection_manager.send_json(
                        client_id,
                        {
                            "type": "audio",
                            "audio": audio_base64,
                            "format": "wav",
                        },
                    )

            elif message_type == "clear":
                conversation_manager.clear_conversation(client_id)
                await connection_manager.send_json(
                    client_id, {"type": "cleared", "status": "ok"}
                )

            elif message_type == "ping":
                await connection_manager.send_json(
                    client_id, {"type": "pong"}
                )

            elif message_type == "pronunciation":
                # Handle pronunciation analysis request
                if not PRONUNCIATION_AVAILABLE:
                    await connection_manager.send_json(
                        client_id,
                        {
                            "type": "pronunciation_feedback",
                            "score": 0,
                            "summary": "Pronunciation feedback is not available. Install with: uv sync --extra pronunciation",
                            "errors": [],
                        },
                    )
                    continue

                audio_base64 = data.get("audio", "")
                expected_text = data.get("expected_text", "")

                if not audio_base64 or not expected_text:
                    continue

                pronunciation_service = get_pronunciation_service()
                feedback = await pronunciation_service.analyze_pronunciation(
                    audio_base64, expected_text
                )

                if feedback:
                    await connection_manager.send_json(
                        client_id,
                        {
                            "type": "pronunciation_feedback",
                            "score": feedback.overall_score,
                            "summary": feedback.summary,
                            "transcribed": feedback.transcribed_text,
                            "expected": feedback.original_text,
                            "errors": [
                                {
                                    "word": e.word,
                                    "expected": e.expected_phoneme,
                                    "detected": e.detected_phoneme,
                                    "tip": e.tip,
                                }
                                for e in feedback.errors
                            ],
                        },
                    )
                else:
                    await connection_manager.send_json(
                        client_id,
                        {
                            "type": "pronunciation_feedback",
                            "score": 0,
                            "summary": "Could not analyze pronunciation. Please try again.",
                            "errors": [],
                        },
                    )

    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        conversation_manager.delete_conversation(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
