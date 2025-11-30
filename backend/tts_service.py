"""Supertonic TTS Service for generating conversational audio."""

import asyncio
import io
import json
import logging
import os
import re
import wave
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from unicodedata import normalize

import numpy as np

logger = logging.getLogger(__name__)

# Check if ONNX Runtime is available
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not installed. TTS will not work.")
    ort = None


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """Convert lengths to binary mask."""
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """Split text into chunks by paragraphs and sentences."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        sentences = re.split(pattern, paragraph)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


class UnicodeProcessor:
    """Process text to unicode indices for TTS."""

    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str) -> str:
        text = normalize("NFKD", text)

        # Remove emojis
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        # Replace various dashes and symbols
        replacements = {
            "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
            """: '"', """: '"', "'": "'", "'": "'", "´": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ", "→": " ", "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Remove combining diacritics
        text = re.sub(
            r"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]",
            "", text,
        )

        # Remove special symbols
        text = re.sub(r"[♥☆♡©\\]", "", text)

        # Replace known expressions
        expr_replacements = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
        for k, v in expr_replacements.items():
            text = text.replace(k, v)

        # Fix spacing around punctuation
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)

        # Remove duplicate quotes
        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Add period if needed
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        return length_to_mask(text_ids_lengths)

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(char) for char in text], dtype=np.uint16)

    def __call__(self, text_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            # indexer is a list where index = unicode value, value = token id
            ids = []
            for val in unicode_vals:
                if val < len(self.indexer):
                    ids.append(self.indexer[val] if self.indexer[val] >= 0 else 0)
                else:
                    ids.append(0)
            text_ids[i, : len(unicode_vals)] = np.array(ids, dtype=np.int64)
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask


class Style:
    """Voice style container."""

    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx


class SupertonicTTS:
    """Supertonic Text-to-Speech engine using ONNX."""

    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: "ort.InferenceSession",
        text_enc_ort: "ort.InferenceSession",
        vector_est_ort: "ort.InferenceSession",
        vocoder_ort: "ort.InferenceSession",
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(wav_lengths, self.base_chunk_size, self.chunk_compress_factor)
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self, text_list: list[str], style: Style, total_step: int, speed: float = 1.05
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list)

        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed

        text_emb_onnx, *_ = self.text_enc_ort.run(
            None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
        )

        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)

        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )

        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx

    def synthesize(
        self,
        text: str,
        style: Style,
        total_step: int = 5,
        speed: float = 1.05,
        silence_duration: float = 0.3,
    ) -> tuple[np.ndarray, float]:
        """Synthesize speech from text."""
        text_list = chunk_text(text)
        wav_cat = None
        dur_cat = 0.0

        for chunk in text_list:
            wav, dur_onnx = self._infer([chunk], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = float(dur_onnx[0])
            else:
                silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat += float(dur_onnx[0]) + silence_duration

        return wav_cat[0], dur_cat


def load_voice_style(voice_style_path: str) -> Style:
    """Load a voice style from JSON file."""
    with open(voice_style_path, "r") as f:
        voice_style = json.load(f)

    ttl_dims = voice_style["style_ttl"]["dims"]
    dp_dims = voice_style["style_dp"]["dims"]

    ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
    ttl_style = ttl_data.reshape(1, ttl_dims[1], ttl_dims[2])

    dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
    dp_style = dp_data.reshape(1, dp_dims[1], dp_dims[2])

    return Style(ttl_style, dp_style)


class TTSService:
    """TTS Service using Supertonic ONNX models."""

    def __init__(
        self,
        assets_dir: str = "assets",
        voice_style: str = "F1",  # F1, F2, M1, M2
        total_steps: int = 5,
        speed: float = 1.05,
    ):
        self.assets_dir = Path(assets_dir)
        self.onnx_dir = self.assets_dir / "onnx"
        self.voice_styles_dir = self.assets_dir / "voice_styles"
        self.voice_style_name = voice_style
        self.total_steps = total_steps
        self.speed = speed

        self.tts: Optional[SupertonicTTS] = None
        self.style: Optional[Style] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the TTS engine."""
        if self._initialized:
            return True

        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False

        if not self.onnx_dir.exists():
            logger.error(f"ONNX models not found at {self.onnx_dir}")
            return False

        try:
            logger.info("Loading Supertonic TTS models...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_models)
            self._initialized = True
            logger.info("Supertonic TTS loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load TTS models: {e}")
            return False

    def _load_models(self):
        """Load ONNX models (runs in executor)."""
        opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]

        # Load config
        cfg_path = self.onnx_dir / "tts.json"
        with open(cfg_path, "r") as f:
            cfgs = json.load(f)

        # Load ONNX models
        dp_ort = ort.InferenceSession(
            str(self.onnx_dir / "duration_predictor.onnx"), sess_options=opts, providers=providers
        )
        text_enc_ort = ort.InferenceSession(
            str(self.onnx_dir / "text_encoder.onnx"), sess_options=opts, providers=providers
        )
        vector_est_ort = ort.InferenceSession(
            str(self.onnx_dir / "vector_estimator.onnx"), sess_options=opts, providers=providers
        )
        vocoder_ort = ort.InferenceSession(
            str(self.onnx_dir / "vocoder.onnx"), sess_options=opts, providers=providers
        )

        # Load text processor
        unicode_indexer_path = self.onnx_dir / "unicode_indexer.json"
        text_processor = UnicodeProcessor(str(unicode_indexer_path))

        self.tts = SupertonicTTS(
            cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
        )

        # Load voice style
        voice_style_path = self.voice_styles_dir / f"{self.voice_style_name}.json"
        self.style = load_voice_style(str(voice_style_path))

    def set_voice(self, voice_name: str):
        """Change the voice style (F1, F2, M1, M2)."""
        voice_style_path = self.voice_styles_dir / f"{voice_name}.json"
        if voice_style_path.exists():
            self.style = load_voice_style(str(voice_style_path))
            self.voice_style_name = voice_name
            logger.info(f"Voice changed to {voice_name}")
        else:
            logger.warning(f"Voice style {voice_name} not found")

    async def generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech audio from text.

        Args:
            text: The text to convert to speech

        Returns:
            WAV audio bytes or None if generation fails
        """
        if not self._initialized:
            success = await self.initialize()
            if not success:
                return None

        if not text.strip():
            return None

        try:
            loop = asyncio.get_event_loop()

            # Run synthesis in executor
            wav, duration = await loop.run_in_executor(
                None,
                lambda: self.tts.synthesize(
                    text, self.style, self.total_steps, self.speed
                )
            )

            # Trim to actual duration
            sample_count = int(self.tts.sample_rate * duration)
            wav_trimmed = wav[:sample_count]

            # Convert to WAV bytes
            return self._numpy_to_wav(wav_trimmed, self.tts.sample_rate)

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _numpy_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        # Normalize and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()


# Global service instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create the TTS service instance."""
    global _tts_service
    if _tts_service is None:
        # Get assets path relative to this file
        backend_dir = Path(__file__).parent
        project_dir = backend_dir.parent
        assets_dir = project_dir / "assets"

        _tts_service = TTSService(
            assets_dir=str(assets_dir),
            voice_style="F1",  # Default female voice
            total_steps=5,
            speed=1.05,
        )
    return _tts_service
