import { useEffect, useRef, useState, useCallback } from 'react';
import { Platform, PermissionsAndroid } from 'react-native';
import Constants from 'expo-constants';
import LiveAudioStream from 'react-native-live-audio-stream';
import { ENDPOINTS } from '../constants/api';
import { Band } from '../constants/theme';

export interface RiskEvent {
  score:       number;
  band:        Band;
  action:      string;
  cnn_prob:    number;
  keywords:    string[];
  transcript:  string;
  explanation: string;
  session_id:  string;
}

type Status = 'idle' | 'requesting' | 'recording' | 'error' | 'stopped' | 'no_mic';

const SAMPLE_RATE = 16000;
const SAMPLES_1S  = SAMPLE_RATE;

// Emulators have no mic — react-native-live-audio-stream hard-crashes (SIGABRT)
// on AudioRecord.read() when there is no audio hardware.
const IS_REAL_DEVICE = Constants.isDevice;

export function useVoiceStream() {
  const ws          = useRef<WebSocket | null>(null);
  const accumulator = useRef<number[]>([]);

  const [status,    setStatus]    = useState<Status>(IS_REAL_DEVICE ? 'idle' : 'no_mic');
  const [event,     setEvent]     = useState<RiskEvent | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const stop = useCallback(() => {
    if (IS_REAL_DEVICE) {
      try { LiveAudioStream.stop(); } catch {}
    }
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    accumulator.current = [];
    setStatus(IS_REAL_DEVICE ? 'stopped' : 'no_mic');
  }, []);

  const start = useCallback(async () => {
    if (!IS_REAL_DEVICE) {
      setStatus('no_mic');
      return;
    }

    setStatus('requesting');

    // ── permissions ──────────────────────────────────────────────
    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title:   'Microphone permission',
          message: 'VoiceGuard needs your mic to detect AI voices.',
          buttonPositive: 'Allow',
        },
      );
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
        setStatus('error');
        return;
      }
    }

    // ── WebSocket ─────────────────────────────────────────────────
    const socket = new WebSocket(ENDPOINTS.stream);
    ws.current   = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({ type: 'init', metadata: { unknown_number: false } }));
      setStatus('recording');
    };

    socket.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as RiskEvent;
        setEvent(data);
        if (data.session_id) setSessionId(data.session_id);
      } catch {}
    };

    socket.onerror = () => setStatus('error');
    socket.onclose = () => setStatus((prev) => prev === 'recording' ? 'stopped' : prev);

    // ── audio stream ──────────────────────────────────────────────
    LiveAudioStream.init({
      sampleRate:    SAMPLE_RATE,
      channels:      1,
      bitsPerSample: 16,
      audioSource:   6,
      bufferSize:    4096,
    });

    LiveAudioStream.on('data', (b64: string) => {
      if (socket.readyState !== WebSocket.OPEN) return;

      // base64 → Int16 → Float32
      const binary = atob(b64);
      const bytes  = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const int16  = new Int16Array(bytes.buffer);
      for (let i = 0; i < int16.length; i++) {
        accumulator.current.push(int16[i] / 32768.0);
      }

      // flush whole 1-second chunks
      while (accumulator.current.length >= SAMPLES_1S) {
        const chunk = accumulator.current.splice(0, SAMPLES_1S);
        const f32   = new Float32Array(chunk);
        socket.send(f32.buffer);
      }
    });

    LiveAudioStream.start();
  }, []);

  useEffect(() => () => { stop(); }, []);

  return { status, event, sessionId, start, stop };
}
