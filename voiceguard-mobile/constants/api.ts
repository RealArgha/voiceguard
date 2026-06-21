// Change SERVER_IP to your machine's local IP when running on a physical device.
export const SERVER_IP   = '172.20.10.2';
export const SERVER_PORT = 8000;

export const HTTP_BASE = `http://${SERVER_IP}:${SERVER_PORT}`;
export const WS_BASE   = `ws://${SERVER_IP}:${SERVER_PORT}`;

export const ENDPOINTS = {
  stream:   `${WS_BASE}/ws/stream`,
  sessions: `${HTTP_BASE}/api/sessions`,
  session:  (id: string) => `${HTTP_BASE}/api/sessions/${id}`,
  events:   (id: string) => `${HTTP_BASE}/api/sessions/${id}/events`,
};
