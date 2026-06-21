"""
Whisper transcription + suspicious-keyword detector.

Model size is controlled by WHISPER_SIZE env var (default: medium).
'medium' is strongly recommended for Indian-accented English — it has
far better coverage of non-native phonemes than 'small' or 'base'.

Set USE_WHISPER=0 to skip transcription entirely.
"""

import os
import re
import numpy as np
import torch

USE_WHISPER = os.getenv("USE_WHISPER", "1") == "1"

# Seeds Whisper's decoder with Indian English context so it maps
# Indian phonemes (retroflex consonants, vowel shifts) correctly.
# Keep it short — Whisper uses this as prior text, not a system prompt.
_INITIAL_PROMPT = (
    "This is a phone call in Indian English. "
    "The speaker may discuss banking, OTP, account transfers, or personal details."
)

# Phrases fraudsters typically use on social-engineering calls.
SUSPICIOUS_PATTERNS = [
    r"transfer .* (urgent|now|immediately)",
    r"wire .* (funds|money)",
    r"reset .* (password|pin)",
    r"otp",
    r"one[- ]time password",
    r"verification code",
    r"don'?t tell (anyone|the bank)",
    r"emergency",
    r"locked out",
]

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None and USE_WHISPER:
        import whisper
        size = os.getenv("WHISPER_SIZE", "medium")
        print(f"[whisper] loading model: {size}")
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def transcribe(audio: np.ndarray, sr: int = 16_000) -> str:
    """audio: float32 mono at 16 kHz. Returns plain-text transcript."""
    if not USE_WHISPER or len(audio) < sr * 0.5:
        return ""

    model = _load_whisper()
    if model is None:
        return ""

    result = model.transcribe(
        audio,
        language="en",
        initial_prompt=_INITIAL_PROMPT,
        fp16=torch.cuda.is_available(),
        condition_on_previous_text=False,  # avoid hallucination loops on short clips
    )
    return result.get("text", "").strip()


def keyword_flags(transcript: str) -> list[str]:
    """Returns the list of suspicious patterns that matched."""
    if not transcript:
        return []
    hits = []
    lower = transcript.lower()
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, lower):
            hits.append(pat)
    return hits
