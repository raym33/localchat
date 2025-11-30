"""Dia2 TTS Service for generating conversational audio."""

import asyncio
import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check if we can import Dia2 (requires the dia2 package to be installed)
DIA2_AVAILABLE = False
try:
    # This will be available when dia2 is installed
    from dia2 import Dia2, GenerationConfig, SamplingConfig

    DIA2_AVAILABLE = True
except ImportError:
    logger.warning("Dia2 not installed. Using mock TTS service.")
    Dia2 = None
    GenerationConfig = None
    SamplingConfig = None


class Dia2Service:
    """Service for generating speech using Dia2 model."""

    def __init__(
        self,
        model_name: str = "nari-labs/Dia2-2B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cfg_scale: float = 3.0,
        temperature: float = 0.8,
        top_k: int = 50,
        use_cuda_graph: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.top_k = top_k
        self.use_cuda_graph = use_cuda_graph
        self.model: Optional[Dia2] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the Dia2 model."""
        if self._initialized:
            return True

        if not DIA2_AVAILABLE:
            logger.warning("Dia2 not available. Running in demo mode.")
            self._initialized = True
            return True

        try:
            logger.info(f"Loading Dia2 model: {self.model_name}")
            # Run in executor to not block the event loop
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: Dia2.from_repo(
                    self.model_name, device=self.device, dtype=self.dtype
                ),
            )
            self._initialized = True
            logger.info("Dia2 model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load Dia2 model: {e}")
            return False

    def _create_config(self) -> "GenerationConfig":
        """Create generation config."""
        if not DIA2_AVAILABLE:
            return None
        return GenerationConfig(
            cfg_scale=self.cfg_scale,
            audio=SamplingConfig(temperature=self.temperature, top_k=self.top_k),
            use_cuda_graph=self.use_cuda_graph,
        )

    async def generate_speech(
        self, text: str, speaker: str = "S1"
    ) -> Optional[bytes]:
        """Generate speech audio from text.

        Args:
            text: The text to convert to speech
            speaker: Speaker tag (S1 or S2)

        Returns:
            WAV audio bytes or None if generation fails
        """
        if not self._initialized:
            await self.initialize()

        # Format text with speaker tag
        formatted_text = f"[{speaker}] {text}"

        if not DIA2_AVAILABLE or self.model is None:
            # Return demo audio (silence with a beep pattern)
            return self._generate_demo_audio(text)

        try:
            loop = asyncio.get_event_loop()

            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name

            config = self._create_config()

            # Run generation in executor
            result = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    formatted_text,
                    config=config,
                    output_wav=output_path,
                    verbose=False,
                ),
            )

            # Read the generated audio
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up temp file
            Path(output_path).unlink(missing_ok=True)

            return audio_bytes

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return self._generate_demo_audio(text)

    def _generate_demo_audio(self, text: str) -> bytes:
        """Generate demo audio when Dia2 is not available."""
        import struct
        import wave

        # Parameters
        sample_rate = 24000
        duration = min(len(text) * 0.05, 3.0)  # Approximate duration based on text length
        num_samples = int(sample_rate * duration)

        # Generate a simple tone sequence
        samples = []
        words = text.split()
        samples_per_word = num_samples // max(len(words), 1)

        for i, word in enumerate(words):
            # Different frequency for each word
            freq = 200 + (hash(word) % 300)
            for j in range(samples_per_word):
                t = j / sample_rate
                # Envelope
                env = min(j / 500, 1.0) * min((samples_per_word - j) / 500, 1.0)
                sample = int(16000 * env * np.sin(2 * np.pi * freq * t))
                samples.append(sample)

        # Pad to full duration if needed
        while len(samples) < num_samples:
            samples.append(0)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        return buffer.getvalue()

    async def generate_conversation_response(
        self, user_text: str, ai_response: str
    ) -> Optional[bytes]:
        """Generate audio for a conversation exchange.

        Args:
            user_text: What the user said (for context, not spoken)
            ai_response: The AI's response to speak

        Returns:
            WAV audio bytes
        """
        # For now, just generate the AI response
        # In a full implementation, we could condition on user audio
        return await self.generate_speech(ai_response, speaker="S1")


# Global service instance
_dia2_service: Optional[Dia2Service] = None


def get_dia2_service() -> Dia2Service:
    """Get or create the Dia2 service instance."""
    global _dia2_service
    if _dia2_service is None:
        from .config import config

        _dia2_service = Dia2Service(
            model_name=config.dia2_model,
            device=config.dia2_device,
            dtype=config.dia2_dtype,
            cfg_scale=config.cfg_scale,
            temperature=config.temperature,
        )
    return _dia2_service
