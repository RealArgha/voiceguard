# Twilio + ngrok setup

You need a publicly reachable URL for Twilio to send webhooks and audio
to. We use ngrok to tunnel your laptop to the public internet.

## 1. Twilio account

1. Sign up at <https://www.twilio.com/try-twilio> — free trial gives ~$15 credit.
2. From the console: **Phone Numbers → Buy a number** → pick one with
   *Voice* capability. ~$1/month. The trial credit covers it.
3. Copy the **phone number** — that's what users will dial.

## 2. ngrok (free tier is fine)

```bash
# install
brew install ngrok                # macOS
# or download from https://ngrok.com/download

# authenticate (one-time, free signup)
ngrok config add-authtoken <YOUR_TOKEN>
```

## 3. Expose your local backend

In one terminal:
```bash
uvicorn backend.main:app --port 8000
```

In another terminal:
```bash
ngrok http 8000
```

You'll see something like:
```
Forwarding   https://a1b2c3d4.ngrok-free.app -> http://localhost:8000
```

Copy that hostname (without `https://`).

## 4. Tell the backend its public hostname

Set `PUBLIC_HOST` so the TwiML response points Twilio at the right WS URL:

```bash
export PUBLIC_HOST=a1b2c3d4.ngrok-free.app
# Restart uvicorn after setting this
```

## 5. Wire Twilio → your backend

In the Twilio console:

1. **Phone Numbers → Active Numbers → click your number**
2. Scroll to **Voice Configuration**
3. Under **A call comes in**, set:
   - Webhook: `https://a1b2c3d4.ngrok-free.app/twilio/voice`
   - HTTP: `POST`
4. Save.

## 6. Test it

Dial your Twilio number from any phone.

- Caller hears: *"Welcome. Your call is being analyzed for security."*
- Open <http://localhost:8000/agent> — risk gauge updates as you speak.
- Try saying: *"Hi I need to transfer funds urgently, what's my OTP?"*
  → watch the band climb to MEDIUM/HIGH from keyword hits alone (CNN
  still untrained at this stage).

## 7. Common gotchas

| Problem | Fix |
|---|---|
| `wss://your-ngrok-domain.ngrok-free.app/...` in Twilio logs | You forgot to `export PUBLIC_HOST=...`. Restart uvicorn after setting it. |
| Twilio shows 11200 "HTTP retrieval failure" | Your ngrok URL changed. Free tier rotates on restart. Update the Twilio webhook URL. |
| No audio events in backend logs | Twilio's Media Streams uses **wss** (secure WS). ngrok handles this automatically; make sure the TwiML uses `wss://` not `ws://`. |
| Calls cost money | Trial credit covers ~100 minutes. Set a billing alert in Twilio console. |

## 8. Production notes (post-hackathon)

- Replace ngrok with a real domain + TLS cert (Let's Encrypt).
- Validate Twilio signatures on `/twilio/voice` (`X-Twilio-Signature`
  header) so randos can't fake webhooks.
- Use a Twilio TaskRouter to route flagged calls to a fraud specialist.
- Persist evidence reports (audio clip + spectrogram + transcript + risk
  trace) to S3 or equivalent for regulator audit trails.
