# VoiceGuard mobile (Path B)

Customer-facing companion app. Shows a live risk gauge while the
customer is on a call with the bank.

## Architecture

```
                  ┌─────────────────────┐
   Phone dialer ──┤  Twilio number      │
                  │  ↓ Media Streams    │
                  │  Backend (FastAPI)  │──┐
                  │  ↓ broadcast        │  │ same WS feed
   Mobile app  ◀──┤  /ws/dashboard     ─┘  │
                  └─────────────────────┘  │
   Agent web   ◀──────────────────────────-┘
```

The mobile app does **not** route audio itself. It opens the native
dialer (`tel:`) to call your Twilio number, and subscribes via WebSocket
to the same risk feed the agent dashboard uses.

This is deliberate — it ships in a day, works in Expo Go, and the pitch
("trusted third-party gauge") is actually stronger than a creepy
hijacking-the-dialer pitch.

## Run it

```bash
cd mobile
npm install
npx expo start
```

Scan the QR code with **Expo Go** on your phone (App Store / Play Store).

## Configure before first run

Edit `components/CallScreen.tsx`:

```ts
const BANK_PHONE_NUMBER = '+15551234567';                  // your Twilio #
const BACKEND_WS_URL    = 'ws://10.0.0.10:8000/ws/dashboard'; // your laptop LAN IP
```

- `BANK_PHONE_NUMBER` — the Twilio number you bought (see `../docs/TWILIO_SETUP.md`).
- `BACKEND_WS_URL` — your laptop's LAN IP (NOT localhost — phone can't reach that). Find it with `ipconfig getifaddr en0` (macOS) or `ip addr show` (Linux). Phone and laptop must be on the same wifi.

## Upgrade path (post-hackathon)

Replace the `Linking.openURL('tel:...')` call with the Twilio Voice
React Native SDK so the call happens inside the app — full in-app VoIP,
no dialer handoff. Requires bare workflow (eject from Expo Go):

```bash
npx expo prebuild
npm install @twilio/voice-react-native-sdk
```

Backend pipeline stays identical.
