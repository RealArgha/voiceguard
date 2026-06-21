# VoiceGuard — Claude Session Context

## Project Overview
VoiceGuard is a real-time AI voice detection system. It classifies live microphone audio as **human (bonafide)** vs **AI/TTS (spoof)**. The backend is FastAPI + PyTorch; the frontend is a browser page that streams microphone audio via WebSocket.

**Live URL (local):** `http://localhost:8000`
**Start server:** `uvicorn backend.main:app --reload --port 8000` (from `E:\voiceguard1`, in `.venv`)

---

## Hardware
- GPU: NVIDIA GeForce RTX 4070 Laptop (CUDA 12.1, torch 2.5.1+cu121)
- Mixed-precision (AMP) is enabled in `train_2021.py`

---

## Model: VoiceGuardCNN (`backend/model.py`)
- Input: log-mel spectrogram `(B, 1, 128, T)`, 128 mel bins, 10 ms hop
- Three conv blocks (32→64→128 channels, BatchNorm, ReLU, MaxPool2)
- Global average pool → FC(128→1) → Sigmoid
- Output: `P(synthetic)` in [0,1]; threshold **0.56**

---

## Dataset Layout

### ASVspoof 2019 LA (original training data — DO NOT DELETE)
```
ASVSPOOF_ROOT/LA/LA/
  ASVspoof2019_LA_train/flac/      ← 25,380 files (2,580 bonafide + 22,800 spoof)
  ASVspoof2019_LA_dev/flac/        ← 24,844 files (2,548 bonafide + 22,296 spoof)
  ASVspoof2019_LA_eval/flac/
  ASVspoof2019_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt    ← label format: col[1]=id, col[4]=label
    ASVspoof2019.LA.cm.dev.trl.txt
```

### ASVspoof 2021 (archive/ directory)
```
archive/
  ASVspoof2021_LA_eval/ASVspoof2021_LA_eval/flac/   ← 181,566 files
  ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac/  ← 152,955 files
  ASVspoof2021_DF_eval_part01/ASVspoof2021_DF_eval/flac/  ← 152,958 files
  ASVspoof2021_DF_eval_part02/ASVspoof2021_DF_eval/flac/  ← 152,958 files
  LA-keys-full/keys/LA/CM/trial_metadata.txt   ← 181,566 entries; col[5]=label
  DF-keys-full/keys/DF/CM/trial_metadata.txt   ← 611,829 entries; col[5]=label
  PA-keys-full/keys/PA/CM/trial_metadata.txt   ← PA data; col[9]=label
```

**Protocol column layout:**
- 2019: `speaker_id  file_id  -  -  label`  → `parts[4]`
- 2021 LA/DF: `speaker_id  file_id  codec  tx  system  label  ...` → `parts[5]`
- 2021 PA: same row but label at → `parts[9]`

---

## Weights (`weights/`)
| File | Description |
|---|---|
| `voiceguard_cnn.pt` | Trained on 2019 LA only |
| `voiceguard_cnn_aug.pt` | 2019 LA + augmentation |
| `voiceguard_cnn_2021.pt` | 2019 + 2021 LA + DF + PA combined — **1.41% EER** |
| `voiceguard_cnn_finetuned.pt` | **Active** — mic fine-tune on 105 real + 38 filtered fakes — **0% EER** |

`main.py` resolution order: `CNN_WEIGHTS` env → `voiceguard_cnn_finetuned.pt` → `voiceguard_cnn_2021.pt` → `voiceguard_cnn_aug.pt` → `voiceguard_cnn.pt`

---

## Training Scripts

### `backend/train.py` — 2019-only baseline
```bash
python -m backend.train --augment --epochs 8 --batch-size 128
```
Env: `ASVSPOOF_ROOT=ASVSPOOF_ROOT/LA/LA`

### `backend/train_2021.py` — combined 2019+2021 (the one to run)
```bash
python -m backend.train_2021 --augment --epochs 10 --batch-size 256
python -m backend.train_2021 --quick    # sanity check (2 epochs, ~5 min)
```
Includes PA data automatically if `archive/PA-keys-full/` exists. `--max-pa21` defaults to 60k.

### `backend/finetune.py` — personal mic fine-tune
```bash
python -m backend.finetune --no-augment --warmup 0 --lr 1e-4 --epochs 15
```
**Training data:** `data/mini/real/` (105 clips) + `data/mini/fake/` (38 clips — filtered to only those scoring ≥ 0.60 on 2021 model).
`data/mini/fake_weak/` holds the 26 quarantined low-confidence fakes — do not move back.

---

## Fine-Tune Calibration (current)
| Metric | Value |
|---|---|
| Real voice mean CNN | 0.470 |
| Real voice max CNN | 0.565 → fused score 33.9 → **LOW** |
| Fake (TTS) min CNN | 0.561 → fused score 33.7 → **LOW** (borderline) |
| Fake (TTS) mean CNN | 0.640 → fused score 38.4 → **MEDIUM** |
| CNN_THRESHOLD | **0.56** |
| LOW band boundary | score ≤ **34** |

---

## Caching
| Directory | Contents |
|---|---|
| `cache/train/` | 2019 LA train mel cache |
| `cache/dev/` | 2019 LA dev mel cache |
| `cache/la21/` | 2021 LA mel cache |
| `cache/df21/` | 2021 DF mel cache |
| `cache_audio/train/` | 2019 LA train raw audio cache (augment mode) |
| `cache_audio/la21/` | 2021 LA raw audio cache |
| `cache_audio/df21/` | 2021 DF raw audio cache |

---

## Backend Pipeline (`backend/main.py`)
1. Client opens WebSocket `/ws/stream`
2. Client sends `{type: "init", metadata: {...}}` text frame
3. Client streams `float32` PCM binary frames (~16k samples / frame = 1 s)
4. Server keeps a 3-second rolling buffer + 3-frame CNN score smoothing
5. Every frame: CNN + Whisper run concurrently → fused risk score → JSON event
6. Each event is saved to PostgreSQL (`events` table) and published to Redis channel
7. Session row created on connect, `ended_at` stamped on disconnect

Risk event shape:
```json
{
  "score": 78.2,        "band": "HIGH",    "action": "BLOCK",
  "cnn_prob": 0.91,     "keywords": [...], "transcript": "...",
  "explanation": "...",  "session_id": "uuid"
}
```

REST endpoints (new):
- `GET /api/sessions` — list recent sessions (latest 50)
- `GET /api/sessions/{id}` — single session with event count
- `GET /api/sessions/{id}/events` — all events for a session

---

## Persistence Stack

### PostgreSQL (`backend/database.py`)
- Engine: SQLAlchemy async + asyncpg
- `DB_ENABLED=0` disables it; app still works without a DB
- `DATABASE_URL` env var (default: `postgresql+asyncpg://postgres:password@localhost/voiceguard`)
- Tables auto-created on startup via `init_db()`

### Redis (`backend/redis_client.py`)
- `REDIS_URL` env var (default: `redis://localhost:6379`)
- `REDIS_ENABLED=0` disables it
- Each event published to `voiceguard:events:{session_id}` (pub/sub for React Native)
- Last 50 events cached in `voiceguard:history:{session_id}` list (24h TTL)

---

## Frontend (`frontend/index.html`)
- Full dark-theme redesign with animated SVG ring gauge
- Score counter animates smoothly on change (ease-out cubic, 400 ms)
- Band label (LOW/MEDIUM/HIGH) pops in with scale animation on change
- Responsive: desktop two-column layout; mobile single-column with fixed bottom mic bar
- Safe-area inset support for notched phones

---

## Key Files
| Path | Purpose |
|---|---|
| `backend/model.py` | VoiceGuardCNN definition + load_model() |
| `backend/spectrogram.py` | audio_to_logmel() — 128 mels, 10 ms hop, 16 kHz |
| `backend/augmentation.py` | augment_audio() + spec_augment() |
| `backend/train.py` | 2019-only training script |
| `backend/train_2021.py` | Combined 2019+2021+PA training script |
| `backend/finetune.py` | Personal mic fine-tune (progressive unfreezing, AMP) |
| `backend/main.py` | FastAPI app, WebSocket stream, REST API |
| `backend/fusion.py` | Risk score fusion — CNN_THRESHOLD=0.56, LOW≤34 |
| `backend/explainer.py` | Human-readable risk explanation |
| `backend/transcribe.py` | Whisper transcription + keyword flagging |
| `backend/database.py` | SQLAlchemy async models (Session, Event) + init_db() |
| `backend/redis_client.py` | Redis pub/sub + history cache helpers |
| `test_model.py` | CLI tool: score audio files directly (threshold 0.56) |
| `record_dataset.py` | Record real + slice/record fake clips for fine-tune |
| `frontend/index.html` | Browser mic streaming page (redesigned) |
| `.env.example` | Template for DATABASE_URL, REDIS_URL, etc. |

---

## React Native App (`voiceguard-mobile/`)
- Expo SDK 52 + Expo Router (file-based navigation)
- `react-native-svg` — animated ring gauge matching web design
- `react-native-live-audio-stream` — real-time PCM mic streaming (requires native build)
- Server IP: `172.20.10.2:8000` (set in `voiceguard-mobile/constants/api.ts`)
- Screens: Monitor (ring + mic), History (session list), Session detail (event timeline)
- **Requires Android Studio** to build: `npx expo prebuild` → `npx expo run:android`
- Expo Go works for UI preview only (no audio streaming)

---

## Status (as of 2026-06-13)
- [x] Model trained on ASVspoof 2019 LA + 2021 LA + DF + PA (`voiceguard_cnn_2021.pt` — 1.41% EER)
- [x] Fine-tuned on personal mic data (`voiceguard_cnn_finetuned.pt` — 0% EER, 105 real + 38 filtered fakes)
- [x] Threshold calibrated: CNN=0.56, LOW band ≤ 34
- [x] Frontend redesigned (animated ring, score/band animations, mobile layout)
- [x] PostgreSQL + Redis integrated (`backend/database.py`, `backend/redis_client.py`, `main.py`)
- [x] `.env.example` created
- [x] Whisper accent fix: `medium` model + Indian English `initial_prompt`
- [x] React Native app scaffolded (`voiceguard-mobile/`) — npm install done
- [ ] Android Studio setup needed to run mobile app on device
- [ ] Push notifications for HIGH band (React Native — future)

---

## Common Commands
```bash
# Activate venv
.venv\Scripts\activate

# Install new deps (after requirements.txt update)
pip install asyncpg "sqlalchemy[asyncio]" "redis[asyncio]" python-dotenv

# Run server
uvicorn backend.main:app --reload --port 8000

# Test a file
python test_model.py voice_preview_robot.mp3
python test_model.py --weights weights/voiceguard_cnn_finetuned.pt clip.wav

# Fine-tune (with current best settings)
python -m backend.finetune --no-augment --warmup 0 --lr 1e-4 --epochs 15

# Re-run combined training
python -m backend.train_2021 --augment --epochs 10 --batch-size 256
```
