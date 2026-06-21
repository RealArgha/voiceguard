"""
Twilio integration for VoiceGuard.

Provides
--------
POST /twilio/voice          → TwiML response that starts a Media Stream
WS   /twilio/stream         → Twilio connects here and pushes call audio
WS   /ws/dashboard          → Any client (agent web, customer mobile)
                              subscribes here for live risk events

Flow
----
1. Caller dials your Twilio number.
2. Twilio hits POST /twilio/voice → we return TwiML that:
     - greets the caller
     - opens a <Stream> to ws://<host>/twilio/stream
3. Twilio opens that WebSocket and pushes ~20 ms μ-law chunks.
4. We decode + upsample + buffer + run CNN/Whisper/fuse (same pipeline
   as the browser-mic path) and broadcast every risk event to all
   /ws/dashboard subscribers.
"""

import asyncio
import json
import os
from collections import defaultdict, deque

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from .audio_utils  import decode_twilio_media, upsample_8k_to_16k
from .spectrogram  import audio_to_logmel, SAMPLE_RATE
from .transcribe   import transcribe, keyword_flags
from .fusion       import fuse
from .explainer    import explain
from .model        import predict_synthetic_prob


router = APIRouter()

# Server-relative WebSocket URL that Twilio dials in to.
# We need a public hostname for Twilio to reach us — see TWILIO_SETUP.md.
PUBLIC_HOST = os.getenv("PUBLIC_HOST", "your-ngrok-domain.ngrok-free.app")

WINDOW_SECONDS = 3


# ---------- Dashboard pub/sub ---------------------------------------------
# Every connected dashboard client (agent web, customer mobile) is held in
# this set. When a call produces a risk event we push to all of them.
_dashboard_clients: set[WebSocket] = set()


async def _broadcast(event: dict):
    dead = []
    for ws in _dashboard_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _dashboard_clients.discard(ws)


@router.websocket("/ws/dashboard")
async def dashboard(ws: WebSocket):
    """Agent dashboards + customer mobile apps subscribe here."""
    await ws.accept()
    _dashboard_clients.add(ws)
    try:
        while True:
            # Dashboard is read-only; we only consume to detect disconnect.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _dashboard_clients.discard(ws)


# ---------- TwiML endpoint -------------------------------------------------
@router.post("/twilio/voice")
async def voice_webhook(request: Request):
    """
    Twilio fetches this when a call hits your number.
    We return TwiML that:
      - <Say> a brief greeting (so the caller hears something)
      - <Connect><Stream> opens a bidirectional WS to /twilio/stream
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    from_num = form.get("From", "unknown")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Welcome. Your call is being analyzed for security.</Say>
  <Connect>
    <Stream url="wss://{PUBLIC_HOST}/twilio/stream">
      <Parameter name="call_sid" value="{call_sid}"/>
      <Parameter name="from"     value="{from_num}"/>
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


# ---------- Twilio Media Stream handler -----------------------------------
@router.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    """Receives the live audio Twilio forks from the phone call."""
    await ws.accept()

    # 8 kHz buffer (Twilio's native rate). We resample on each window.
    buf: deque[float] = deque(maxlen=WINDOW_SECONDS * 8000)
    call_meta = {"call_sid": None, "from": None}
    samples_since_last_score = 0

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                params = msg["start"].get("customParameters", {})
                call_meta["call_sid"] = params.get("call_sid")
                call_meta["from"]     = params.get("from")
                print(f"[twilio] call started: {call_meta}")
                continue

            if event == "stop":
                print(f"[twilio] call ended: {call_meta['call_sid']}")
                break

            if event != "media":
                continue

            # Decode 20 ms chunk → 8 kHz float32
            chunk_8k = decode_twilio_media(msg["media"]["payload"])
            buf.extend(chunk_8k.tolist())
            samples_since_last_score += len(chunk_8k)

            # Score roughly once per second (8000 samples at 8 kHz).
            if len(buf) < 8000 or samples_since_last_score < 8000:
                continue
            samples_since_last_score = 0

            # Snapshot, upsample, run pipeline.
            audio_8k  = np.asarray(buf, dtype=np.float32)
            audio_16k = upsample_8k_to_16k(audio_8k)

            cnn_prob, transcript = await _analyze(audio_16k)
            hits  = keyword_flags(transcript)

            # Real-world metadata signals you'd extract from call records.
            metadata = {
                "voip_origin":    call_meta["from"].startswith("+1500")
                                  if call_meta.get("from") else False,
                "unknown_number": False,   # plug your CRM lookup here
            }

            risk = fuse(cnn_prob, hits, metadata)
            note = explain(risk, transcript)

            await _broadcast({
                "call_sid":    call_meta["call_sid"],
                "from":        call_meta["from"],
                "score":       risk.score,
                "band":        risk.band,
                "action":      risk.action,
                "cnn_prob":    risk.cnn_prob,
                "keywords":    risk.keyword_hits,
                "transcript":  transcript,
                "explanation": note,
            })

    except WebSocketDisconnect:
        pass


# ---------- Helpers (shared with main.py path) ----------------------------
# Lazy import to avoid circular reference at module load.
async def _analyze(audio: np.ndarray) -> tuple[float, str]:
    from .main import MODEL, DEVICE        # reuse the loaded model
    loop = asyncio.get_running_loop()

    def _run_cnn():
        mel = audio_to_logmel(audio).to(DEVICE)
        return predict_synthetic_prob(MODEL, mel)

    cnn_task = loop.run_in_executor(None, _run_cnn)
    stt_task = loop.run_in_executor(None, transcribe, audio)
    return await asyncio.gather(cnn_task, stt_task)
