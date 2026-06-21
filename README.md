# VoiceGuard

Real-time AI-cloned-voice detection for bank calls. Flags within 10 s.

```
                          ┌─────────────────────────┐
   ┌──────────────┐       │   FastAPI backend        │
   │ Twilio       │──WS──▶│   ┌─────────┐            │
   │ phone call   │       │   │ μ-law   │            │
   └──────────────┘       │   │ decode  │──┐         │
                          │   └─────────┘  │ buffer  │
                          │                ▼ 3s      │
                          │   ┌─────────────────┐    │
                          │   │ CNN + Whisper   │    │
                          │   │ (parallel)      │    │
                          │   └─────────────────┘    │
                          │                ▼         │
                          │   ┌─────────────────┐    │
                          │   │ Fusion + Claude │    │
                          │   └─────────────────┘    │
                          │                ▼         │
                          │   /ws/dashboard ─────────┼──▶ Agent web (path A)
                          └────────────────┬─────────┘
                                           └────────────▶ Mobile app (path B)
```

## Two deployment paths, one backend

| Path | Audio source | Risk consumer |
|---|---|---|
| **A — Bank-side** | Twilio Media Streams from phone call | Agent dashboard (`/agent`) |
| **B — Customer** | Same Twilio call | Mobile app (subscribes to same broadcast) |

Both share the same CNN, fusion logic, and risk feed. Only the front
end differs.

## Project layout

```
voiceguard/
├── backend/
│   ├── main.py             FastAPI app + dev browser-mic WS
│   ├── twilio_stream.py    Twilio Media Streams handler + /ws/dashboard
│   ├── audio_utils.py      μ-law decode, 8k→16k resample
│   ├── model.py            PyTorch CNN
│   ├── spectrogram.py      librosa log-mel
│   ├── transcribe.py       Whisper + keyword detector
│   ├── fusion.py           Risk fusion + LOW/MED/HIGH bands
│   ├── explainer.py        Claude API (template fallback)
│   └── train.py            ASVspoof 2019 trainer
├── frontend/
│   ├── index.html          Dev tool — browser mic streaming
│   └── agent.html          Path A — agent console (production-style)
├── mobile/                 Path B — Expo / React Native app
│   ├── App.tsx
│   ├── components/
│   │   ├── CallScreen.tsx
│   │   └── RiskGauge.tsx
│   ├── lib/voiceguardApi.ts
│   └── README.md
├── docs/
│   └── TWILIO_SETUP.md     Step-by-step Twilio + ngrok wiring
├── requirements.txt
└── README.md
```

## Quick start (no Twilio yet — browser-mic dev mode)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export USE_WHISPER=0      # skip 500 MB model download for now
uvicorn backend.main:app --reload --port 8000
```

- <http://localhost:8000>        → dev page (use laptop mic)
- <http://localhost:8000/agent>  → agent console (waits for calls)

## Full demo (Path A — phone call into Twilio)

1. Follow **`docs/TWILIO_SETUP.md`** to:
   - buy a Twilio number
   - set up ngrok
   - point Twilio's voice webhook at `https://<ngrok>/twilio/voice`
   - set `export PUBLIC_HOST=<your-ngrok-domain>`
2. `uvicorn backend.main:app --port 8000`
3. `ngrok http 8000` in a second terminal
4. Open `http://localhost:8000/agent` on your laptop
5. Dial your Twilio number from your phone → speak → gauge moves

## Full demo (Path B — customer mobile app)

```bash
cd mobile
npm install
npx expo start
```
Scan the QR code with Expo Go on your phone. See `mobile/README.md`.

## Where to edit what

| Want to change… | File |
|---|---|
| Risk thresholds (LOW/MED/HIGH) | `backend/fusion.py` bottom |
| Fusion weights | `backend/fusion.py` top constants |
| Suspicious keywords | `backend/transcribe.py` `SUSPICIOUS_PATTERNS` |
| CNN architecture | `backend/model.py` |
| Spectrogram params | `backend/spectrogram.py` constants |
| Whisper model size | `WHISPER_SIZE` env var |
| Claude prompt | `backend/explainer.py` `explain()` |
| Agent UI / colors | `frontend/agent.html` |
| Mobile UI | `mobile/components/CallScreen.tsx` |
| Twilio greeting message | `backend/twilio_stream.py` `<Say>` block |

## Train the CNN

The model ships UNTRAINED — scores are random until trained:

```bash
export ASVSPOOF_ROOT=/path/to/ASVspoof2019/LA
python -m backend.train
# saves weights/voiceguard_cnn.pt → picked up on next server start
```

Even untrained, the **keyword + metadata** signals alone produce
sensible MEDIUM/HIGH alerts on fraudulent phrasings, so the end-to-end
loop demos before training is done.
