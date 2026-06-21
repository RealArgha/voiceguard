import { useState, useCallback } from 'react';
import { ENDPOINTS } from '../constants/api';
import { SessionSummary } from '../components/SessionCard';

export interface SessionEvent {
  id:         number;
  created_at: string;
  score:      number;
  band:       string;
  action:     string;
  cnn_prob:   number;
  transcript: string | null;
  keywords:   string[] | null;
}

export function useSessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res  = await globalThis.fetch(ENDPOINTS.sessions);
      const data = await res.json();
      setSessions(data);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load sessions');
    } finally {
      setLoading(false);
    }
  }, []);

  return { sessions, loading, error, fetch };
}

export function useSessionEvents(sessionId: string) {
  const [events,  setEvents]  = useState<SessionEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res  = await globalThis.fetch(ENDPOINTS.events(sessionId));
      const data = await res.json();
      setEvents(data);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load events');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  return { events, loading, error, fetch };
}
