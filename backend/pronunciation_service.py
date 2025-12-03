"""Pronunciation feedback service using phoneme comparison.

This module provides pronunciation analysis by comparing user speech phonemes
with reference pronunciations from the CMU Pronouncing Dictionary.

Requires optional dependencies:
    uv sync --extra pronunciation
"""

import asyncio
import base64
import io
import logging
import re
import tempfile
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional pronunciation dependencies
try:
    import whisper
    import cmudict
    from g2p_en import G2p

    PRONUNCIATION_AVAILABLE = True
except ImportError:
    PRONUNCIATION_AVAILABLE = False
    logger.warning(
        "Pronunciation dependencies not installed. "
        "Run 'uv sync --extra pronunciation' to enable pronunciation feedback."
    )


# Common pronunciation errors for Spanish speakers learning English
SPANISH_SPEAKER_TIPS = {
    # Vowels
    ("IH", "IY"): "The 'i' in this word is short, like 'bit', not long like 'beat'.",
    ("IY", "IH"): "The 'ee' sound should be longer here, like 'beat' not 'bit'.",
    ("AE", "EH"): "Open your mouth wider for the 'a' sound, like 'cat' not 'bet'.",
    ("AH", "AA"): "This 'u' is a schwa sound, very short and relaxed.",
    ("UH", "UW"): "This 'oo' is short, like 'book' not 'boot'.",
    # Consonants
    ("V", "B"): "Use your top teeth on your bottom lip for 'v', don't use both lips like 'b'.",
    ("B", "V"): "Use both lips for 'b', not teeth on lip like 'v'.",
    ("Z", "S"): "Vibrate your vocal cords for 'z', it's voiced unlike 's'.",
    ("SH", "CH"): "For 'sh', air flows continuously. 'Ch' has a stop before the air.",
    ("TH", "D"): "Put your tongue between your teeth for 'th', not behind them like 'd'.",
    ("TH", "T"): "Put your tongue between your teeth for 'th', not behind them like 't'.",
    ("TH", "S"): "Put your tongue between your teeth for 'th', not behind them like 's'.",
    ("DH", "D"): "Voiced 'th' (as in 'the'): tongue between teeth, vibrate vocal cords.",
    ("JH", "Y"): "The 'j' sound has more friction, like 'judge' not 'yes'.",
    ("R", "RR"): "English 'r' doesn't roll. Curl your tongue back, don't tap it.",
    ("NG", "N"): "For 'ng', the sound comes from the back of your throat, not the front.",
    ("H", ""): "Don't forget the 'h' sound at the start. Push air from your throat.",
    ("W", "GW"): "For 'w', round your lips without the 'g' sound before it.",
}

# IPA to ARPABET mapping for display
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ",
    "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
    "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}


@dataclass
class PhonemeError:
    """Represents a single phoneme pronunciation error."""

    word: str
    position: int  # Word index in sentence
    expected_phoneme: str
    detected_phoneme: str
    tip: str
    severity: float  # 0.0-1.0, higher is worse


@dataclass
class PronunciationFeedback:
    """Complete pronunciation feedback for a phrase."""

    original_text: str
    transcribed_text: str
    overall_score: float  # 0.0-1.0, 1.0 is perfect
    errors: list[PhonemeError]
    summary: str


class PronunciationService:
    """Service for analyzing pronunciation and providing feedback."""

    def __init__(self, whisper_model: str = "base"):
        """Initialize the pronunciation service.

        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.cmu_dict = None
        self.g2p = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Load models asynchronously."""
        if not PRONUNCIATION_AVAILABLE:
            logger.error("Pronunciation dependencies not installed")
            return False

        if self._initialized:
            return True

        try:
            logger.info(f"Loading Whisper model '{self.whisper_model_name}'...")
            loop = asyncio.get_event_loop()

            # Load models in thread pool
            self.whisper_model = await loop.run_in_executor(
                None, whisper.load_model, self.whisper_model_name
            )
            self.cmu_dict = await loop.run_in_executor(None, cmudict.dict)
            self.g2p = await loop.run_in_executor(None, G2p)

            self._initialized = True
            logger.info("Pronunciation service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pronunciation service: {e}")
            return False

    def _get_reference_phonemes(self, word: str) -> list[str]:
        """Get reference phonemes for a word.

        Uses CMU dictionary first, falls back to G2P for unknown words.
        """
        word_lower = word.lower().strip(".,!?;:'\"")

        if not word_lower:
            return []

        # Try CMU dictionary first
        if self.cmu_dict and word_lower in self.cmu_dict:
            # Return first pronunciation, strip stress markers
            phonemes = self.cmu_dict[word_lower][0]
            return [re.sub(r"\d", "", p) for p in phonemes]

        # Fallback to G2P for unknown words
        if self.g2p:
            phonemes = self.g2p(word_lower)
            # G2P returns a mix of phonemes and punctuation
            return [p for p in phonemes if p.isalpha()]

        return []

    def _compare_phonemes(
        self, expected: list[str], detected: list[str]
    ) -> tuple[float, list[tuple[str, str]]]:
        """Compare expected and detected phonemes.

        Returns:
            Tuple of (similarity score, list of (expected, detected) mismatches)
        """
        if not expected or not detected:
            return 0.0, []

        # Use sequence matcher for alignment
        matcher = SequenceMatcher(None, expected, detected)
        similarity = matcher.ratio()

        mismatches = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    if i < len(expected) and j < len(detected):
                        mismatches.append((expected[i], detected[j]))
            elif tag == "delete":
                for i in range(i1, i2):
                    mismatches.append((expected[i], ""))
            elif tag == "insert":
                for j in range(j1, j2):
                    mismatches.append(("", detected[j]))

        return similarity, mismatches

    def _get_tip_for_error(self, expected: str, detected: str) -> str:
        """Get a learning tip for a phoneme error."""
        # Check our database of common errors
        key = (expected, detected)
        if key in SPANISH_SPEAKER_TIPS:
            return SPANISH_SPEAKER_TIPS[key]

        # Generic tips based on phoneme type
        if expected in ("TH", "DH"):
            return "For 'th' sounds, place your tongue between your teeth."
        if expected in ("V", "W"):
            return "For 'v', use teeth on lip. For 'w', round your lips."
        if expected.endswith("R") or expected == "R":
            return "English 'r' doesn't roll. Curl your tongue back slightly."

        # IPA representation for display
        exp_ipa = ARPABET_TO_IPA.get(expected, expected)
        det_ipa = ARPABET_TO_IPA.get(detected, detected) if detected else "(missing)"

        return f"Expected sound /{exp_ipa}/, but detected /{det_ipa}/."

    async def analyze_pronunciation(
        self, audio_base64: str, expected_text: str
    ) -> Optional[PronunciationFeedback]:
        """Analyze user pronunciation against expected text.

        Args:
            audio_base64: Base64 encoded WAV audio from user
            expected_text: The text the user was supposed to say

        Returns:
            PronunciationFeedback with detailed analysis, or None on error
        """
        if not self._initialized:
            logger.error("Pronunciation service not initialized")
            return None

        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_base64)

            # Save to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                # Transcribe with Whisper
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.whisper_model.transcribe(
                        temp_path,
                        language="en",
                        word_timestamps=True,
                    ),
                )
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

            transcribed_text = result["text"].strip()

            # Compare words
            expected_words = expected_text.lower().split()
            transcribed_words = transcribed_text.lower().split()

            errors = []
            word_scores = []

            # Align words using sequence matcher
            matcher = SequenceMatcher(None, expected_words, transcribed_words)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    # Words match, still check phoneme-level
                    for i in range(i1, i2):
                        word_scores.append(1.0)
                elif tag == "replace":
                    # Different words - check phoneme similarity
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        exp_word = expected_words[i] if i < len(expected_words) else ""
                        det_word = transcribed_words[j] if j < len(transcribed_words) else ""

                        exp_phonemes = self._get_reference_phonemes(exp_word)
                        det_phonemes = self._get_reference_phonemes(det_word)

                        score, mismatches = self._compare_phonemes(exp_phonemes, det_phonemes)
                        word_scores.append(score)

                        # Report significant errors
                        for exp_p, det_p in mismatches[:3]:  # Limit to 3 per word
                            errors.append(
                                PhonemeError(
                                    word=exp_word,
                                    position=i,
                                    expected_phoneme=exp_p,
                                    detected_phoneme=det_p,
                                    tip=self._get_tip_for_error(exp_p, det_p),
                                    severity=1.0 - score,
                                )
                            )
                elif tag == "delete":
                    # Word missing
                    for i in range(i1, i2):
                        word_scores.append(0.0)
                        errors.append(
                            PhonemeError(
                                word=expected_words[i],
                                position=i,
                                expected_phoneme="(word)",
                                detected_phoneme="(missing)",
                                tip=f"The word '{expected_words[i]}' was not detected. "
                                "Try pronouncing it more clearly.",
                                severity=1.0,
                            )
                        )
                elif tag == "insert":
                    # Extra word spoken (not counted as error, just ignored)
                    pass

            # Calculate overall score
            overall_score = sum(word_scores) / len(word_scores) if word_scores else 0.0

            # Generate summary
            if overall_score >= 0.9:
                summary = "Excellent pronunciation! Keep up the great work."
            elif overall_score >= 0.7:
                summary = "Good pronunciation with a few areas to improve."
            elif overall_score >= 0.5:
                summary = "Decent attempt. Focus on the highlighted errors."
            else:
                summary = "Let's work on this together. Practice the sounds below."

            # Sort errors by severity
            errors.sort(key=lambda e: e.severity, reverse=True)

            return PronunciationFeedback(
                original_text=expected_text,
                transcribed_text=transcribed_text,
                overall_score=overall_score,
                errors=errors[:5],  # Limit to top 5 errors
                summary=summary,
            )

        except Exception as e:
            logger.error(f"Pronunciation analysis failed: {e}")
            return None

    def format_feedback_message(self, feedback: PronunciationFeedback) -> str:
        """Format feedback as a readable message for the user."""
        lines = [
            f"**Pronunciation Score: {feedback.overall_score * 100:.0f}%**",
            "",
            feedback.summary,
            "",
        ]

        if feedback.transcribed_text.lower() != feedback.original_text.lower():
            lines.extend(
                [
                    f"You said: *\"{feedback.transcribed_text}\"*",
                    f"Expected: *\"{feedback.original_text}\"*",
                    "",
                ]
            )

        if feedback.errors:
            lines.append("**Tips for improvement:**")
            for i, error in enumerate(feedback.errors, 1):
                lines.append(f"{i}. **{error.word}**: {error.tip}")

        return "\n".join(lines)


# Singleton
_pronunciation_service: Optional[PronunciationService] = None


def get_pronunciation_service() -> PronunciationService:
    """Get or create the pronunciation service singleton."""
    global _pronunciation_service
    if _pronunciation_service is None:
        _pronunciation_service = PronunciationService(whisper_model="base")
    return _pronunciation_service
