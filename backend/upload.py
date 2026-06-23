"""
File-upload scoring endpoint for the demo.

POST /api/score   multipart file → JSON with CNN prob, risk score, band
GET  /upload      serves the drag-and-drop UI page

This bypasses the browser-mic path entirely and lets you score any
audio file (wav / mp3 / m4a / flac / ogg) through the trained CNN.
Use it on stage for the demo — judges can see real LOW vs HIGH bands
without fighting laptop-mic domain gap.
"""

import os
import shutil
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from .spectrogram import audio_to_logmel, SAMPLE_RATE
from .fusion      import fuse
from .explainer   import explain
from .model       import predict_synthetic_prob


router = APIRouter()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ALLOWED_EXT  = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
MAX_BYTES    = 25 * 1024 * 1024            # 25 MB cap


@router.get("/upload")
def upload_page():
    return FileResponse(FRONTEND_DIR / "upload.html")


@router.post("/api/score")
async def score_file(file: UploadFile = File(...)):
    """Run an uploaded audio file through the full risk pipeline."""

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. "
                                 f"Try one of {sorted(ALLOWED_EXT)}.")

    # Stream to a temp file so librosa can decode any container format.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        size = 0
        while chunk := await file.read(1 << 16):
            size += len(chunk)
            if size > MAX_BYTES:
                tmp.close(); os.unlink(tmp.name)
                raise HTTPException(413, "File too large (>25 MB).")
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    if len(audio) < SAMPLE_RATE * 0.5:
        raise HTTPException(400, "Audio too short (<0.5s).")

    duration_sec = len(audio) / SAMPLE_RATE

    # Score middle 3 seconds (skip silence padding common at start/end).
    target = 3 * SAMPLE_RATE
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    elif len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))

    # Lazy-import MODEL to avoid circular reference at module load.
    from .main import MODEL, DEVICE

    mel = audio_to_logmel(audio).to(DEVICE)
    cnn_prob = predict_synthetic_prob(MODEL, mel)

    risk = fuse(cnn_prob, metadata={})
    note = explain(risk)

    return {
        "filename":   file.filename,
        "duration_s": round(duration_sec, 2),
        "cnn_prob":   risk.cnn_prob,
        "score":      risk.score,
        "band":       risk.band,
        "action":     risk.action,
        "explanation": note,
    }
