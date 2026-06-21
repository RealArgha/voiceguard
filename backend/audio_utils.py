"""
Audio format helpers for Twilio Media Streams.

Twilio sends audio as base64-encoded 8 kHz μ-law mono, in ~20 ms chunks.
Our CNN expects 16 kHz float32 mono. This module bridges the two.

Why we avoid stdlib `audioop`: removed in Python 3.13. We roll our own
μ-law decoder (it's a 256-entry lookup) so the prototype runs on every
modern Python without an extra package.
"""

import base64
import numpy as np
import librosa

# Pre-computed μ-law decode table: byte 0-255 → float in [-1, 1]
# Reference: ITU-T G.711 standard.
_MULAW_TABLE = np.zeros(256, dtype=np.float32)
for i in range(256):
    u = ~i & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    _MULAW_TABLE[i] = (-sample if sign else sample) / 32768.0


def mulaw_to_float(mulaw_bytes: bytes) -> np.ndarray:
    """Decode raw μ-law bytes to float32 PCM in [-1, 1] at 8 kHz."""
    return _MULAW_TABLE[np.frombuffer(mulaw_bytes, dtype=np.uint8)]


def decode_twilio_media(payload_b64: str) -> np.ndarray:
    """One Twilio Media Streams 'media' payload → 8 kHz float32 mono."""
    return mulaw_to_float(base64.b64decode(payload_b64))


def upsample_8k_to_16k(audio_8k: np.ndarray) -> np.ndarray:
    """Resample 8 kHz → 16 kHz so the CNN can consume it."""
    return librosa.resample(audio_8k, orig_sr=8000, target_sr=16000)
