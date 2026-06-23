"""
VoiceGuard — FastAPI backend.

Endpoints
---------
GET  /                            → browser demo frontend
WS   /ws/stream                   → live audio in, risk events out
GET  /api/sessions                → list recent sessions (last 50)
GET  /api/sessions/{id}           → single session + event count
GET  /api/sessions/{id}/events    → all events for a session

Audio protocol on /ws/stream
----------------------------
Client sends BINARY frames containing raw float32 PCM, mono, 16 kHz.
Each frame ~ 1 second of audio (16,000 samples × 4 bytes = 64 KB).

Server keeps a rolling 3-second buffer, runs CNN + Whisper every time
a new frame arrives, and emits a JSON event of the form:

    {
      "score": 78.2,
      "band":  "HIGH",
      "action":"BLOCK",
      "cnn_prob": 0.91,
      "keywords": ["otp", "transfer .* urgent"],
      "transcript": "...",
      "explanation": "...",
      "session_id": "uuid"
    }
"""

import asyncio
import json
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

load_dotenv()

from .model        import load_model, predict_synthetic_prob
from .spectrogram  import audio_to_logmel, SAMPLE_RATE
from .transcribe   import transcribe, keyword_flags
from .fusion       import fuse
from .explainer    import explain
from .database     import (
    Event, Session,
    db_available, get_db, init_db, close_db, new_session_id, utcnow,
)
from .redis_client import (
    close_redis, get_cached_events, init_redis,
    publish_event, redis_available,
)


# ── startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()
    yield
    await close_db()
    await close_redis()


# ── app setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="VoiceGuard", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _resolve_weights() -> str:
    if os.getenv("CNN_WEIGHTS"):
        return os.getenv("CNN_WEIGHTS")
    for candidate in [
        "weights/voiceguard_cnn_finetuned.pt",
        "weights/voiceguard_cnn_2021.pt",
        "weights/voiceguard_cnn_aug.pt",
        "weights/voiceguard_cnn.pt",
    ]:
        if Path(candidate).is_file():
            return candidate
    return "weights/voiceguard_cnn.pt"


WEIGHTS_PATH   = _resolve_weights()
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL          = load_model(WEIGHTS_PATH, device=DEVICE)
WINDOW_SECONDS = 3


# ── frontend routes ───────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/agent")
def agent():
    return FileResponse(FRONTEND_DIR / "agent.html")


# ── Twilio + upload routers ───────────────────────────────────────────────────

from .twilio_stream import router as twilio_router   # noqa: E402
from .upload        import router as upload_router   # noqa: E402

app.include_router(twilio_router)
app.include_router(upload_router)


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    async with get_db() as db:
        if db is None:
            return JSONResponse({"error": "database unavailable"}, status_code=503)

        # sessions + event counts in one query
        rows = await db.execute(
            select(
                Session.id,
                Session.started_at,
                Session.ended_at,
                func.count(Event.id).label("event_count"),
            )
            .outerjoin(Event, Event.session_id == Session.id)
            .group_by(Session.id)
            .order_by(Session.started_at.desc())
            .limit(50)
        )
        results = rows.all()

    return [
        {
            "id":          r.id,
            "started_at":  r.started_at.isoformat() if r.started_at else None,
            "ended_at":    r.ended_at.isoformat()   if r.ended_at   else None,
            "event_count": r.event_count,
        }
        for r in results
    ]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    async with get_db() as db:
        if db is None:
            return JSONResponse({"error": "database unavailable"}, status_code=503)

        row = await db.get(Session, session_id)
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)

        count = await db.scalar(
            select(func.count(Event.id)).where(Event.session_id == session_id)
        )

    return {
        "id":          row.id,
        "started_at":  row.started_at.isoformat() if row.started_at else None,
        "ended_at":    row.ended_at.isoformat()   if row.ended_at   else None,
        "meta":        row.meta,
        "event_count": count,
    }


@app.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: str):
    # Try Redis cache first (fast path)
    cached = await get_cached_events(session_id)
    if cached:
        return cached

    async with get_db() as db:
        if db is None:
            return JSONResponse({"error": "database unavailable"}, status_code=503)

        rows = await db.execute(
            select(Event)
            .where(Event.session_id == session_id)
            .order_by(Event.created_at.asc())
        )
        events = rows.scalars().all()

    return [
        {
            "id":         e.id,
            "created_at": e.created_at.isoformat(),
            "score":      e.score,
            "band":       e.band,
            "action":     e.action,
            "cnn_prob":   e.cnn_prob,
            "transcript": e.transcript,
            "keywords":   e.keywords,
        }
        for e in events
    ]


# ── WebSocket stream ──────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    session_id = new_session_id()
    buf:         deque[float] = deque(maxlen=WINDOW_SECONDS * SAMPLE_RATE)
    cnn_history: deque[float] = deque(maxlen=3)
    metadata:    dict         = {}

    # Create session row in DB
    async with get_db() as db:
        if db is not None:
            db.add(Session(
                id=session_id,
                started_at=utcnow(),
                meta=metadata,
            ))

    try:
        while True:
            msg = await ws.receive()

            # ── text frame: control messages ──────────────────────────────
            if "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "init":
                    metadata = payload.get("metadata", {})
                continue

            # ── binary frame: audio ───────────────────────────────────────
            if "bytes" not in msg or msg["bytes"] is None:
                continue

            chunk = np.frombuffer(msg["bytes"], dtype=np.float32)
            buf.extend(chunk.tolist())

            if len(buf) < SAMPLE_RATE:
                continue

            audio_window = np.asarray(buf, dtype=np.float32)
            cnn_prob_raw, transcript = await _analyze(audio_window)

            cnn_history.append(cnn_prob_raw)
            cnn_prob = float(np.mean(cnn_history))

            hits = keyword_flags(transcript)
            risk = fuse(cnn_prob, hits, metadata)
            note = explain(risk)

            event_payload = {
                "score":       risk.score,
                "band":        risk.band,
                "action":      risk.action,
                "cnn_prob":    risk.cnn_prob,
                "keywords":    risk.keyword_hits,
                "explanation": note,
                "session_id":  session_id,
            }

            await ws.send_json(event_payload)

            # Persist to DB + publish to Redis (fire-and-forget style)
            asyncio.create_task(_persist_event(session_id, risk))
            await publish_event(session_id, event_payload)

    except WebSocketDisconnect:
        pass
    finally:
        await _close_session(session_id)


# ── helpers ───────────────────────────────────────────────────────────────────

async def _analyze(audio: np.ndarray) -> tuple[float, str]:
    loop     = asyncio.get_running_loop()
    cnn_task = loop.run_in_executor(None, _run_cnn,   audio)
    stt_task = loop.run_in_executor(None, transcribe, audio)
    return await asyncio.gather(cnn_task, stt_task)


def _run_cnn(audio: np.ndarray) -> float:
    mel = audio_to_logmel(audio).to(DEVICE)
    return predict_synthetic_prob(MODEL, mel)


async def _persist_event(session_id: str, risk) -> None:
    async with get_db() as db:
        if db is None:
            return
        db.add(Event(
            session_id = session_id,
            created_at = utcnow(),
            score      = risk.score,
            band       = risk.band,
            action     = risk.action,
            cnn_prob   = risk.cnn_prob,
            keywords   = risk.keyword_hits or None,
        ))


async def _close_session(session_id: str) -> None:
    async with get_db() as db:
        if db is None:
            return
        row = await db.get(Session, session_id)
        if row:
            row.ended_at = utcnow()
