/**
 * useRiskFeed — subscribes to the backend's /ws/dashboard broadcast.
 * Returns the most recent risk event, or null until one arrives.
 *
 * The hook auto-reconnects with simple exponential backoff so the gauge
 * keeps working through wifi blips on stage demos.
 */

import { useEffect, useRef, useState } from 'react';

export type RiskEvent = {
  call_sid:    string | null;
  from:        string | null;
  score:       number;
  band:        'LOW' | 'MEDIUM' | 'HIGH';
  action:      'PASS' | 'OTP_CHALLENGE' | 'BLOCK';
  cnn_prob:    number;
  keywords:    string[];
  transcript:  string;
  explanation: string;
};

export function useRiskFeed(url: string): RiskEvent | null {
  const [event, setEvent] = useState<RiskEvent | null>(null);
  const reconnectIn = useRef(1000);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let killed = false;

    const connect = () => {
      if (killed) return;
      ws = new WebSocket(url);

      ws.onopen    = () => { reconnectIn.current = 1000; };
      ws.onmessage = (e) => {
        try { setEvent(JSON.parse(e.data) as RiskEvent); }
        catch { /* ignore malformed frames */ }
      };
      ws.onclose   = () => {
        if (killed) return;
        setTimeout(connect, reconnectIn.current);
        reconnectIn.current = Math.min(reconnectIn.current * 2, 15000);
      };
      ws.onerror   = () => ws?.close();
    };

    connect();
    return () => { killed = true; ws?.close(); };
  }, [url]);

  return event;
}
