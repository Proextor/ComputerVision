export interface DisplayConfig {
  httpUrl: string; wsUrl: string; staleMs: number; mirrorX: boolean; demo: boolean;
  cameraDemo: boolean; ingestUrl: string; captureWidth: number; captureHeight: number;
}

export function deriveWebSocketUrl(httpUrl: string, explicit?: string): string {
  const url = new URL(explicit || httpUrl);
  if (url.username || url.password || url.hash || url.search) throw new Error('Service URLs must not contain credentials, a query or a fragment.');
  if (explicit) {
    if (!['ws:', 'wss:'].includes(url.protocol)) throw new Error('The WebSocket URL must use ws or wss.');
  } else {
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('The HTTP URL must use http or https.');
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = `${url.pathname.replace(/\/$/, '')}/api/v1/scene/stream`;
  }
  return url.toString();
}

export function readConfig(env: Record<string, string | undefined>, search: string): DisplayConfig {
  const httpUrl = env.VITE_VISION_HTTP_URL || (typeof window !== 'undefined' && window.location.port !== '3000' && window.location.origin !== 'null' ? window.location.origin : 'http://127.0.0.1:8000');
  deriveWebSocketUrl(httpUrl); // Validate the HTTP base even with an explicit socket URL.
  const seconds = Number(env.VITE_STALE_SECONDS || 3);
  if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('VITE_STALE_SECONDS must be positive.');
  const captureWidth = Number(env.VITE_CAPTURE_WIDTH || 640), captureHeight = Number(env.VITE_CAPTURE_HEIGHT || 360);
  if (![captureWidth, captureHeight].every(n => Number.isInteger(n) && n >= 64 && n <= 1280) || captureWidth * captureHeight > 921600) throw new Error('Capture dimensions exceed frame limits.');
  const demo = new URLSearchParams(search).get('demo') === '1' || env.VITE_DEMO === 'true';
  return { httpUrl: httpUrl.replace(/\/$/, ''), wsUrl: deriveWebSocketUrl(httpUrl, env.VITE_VISION_WS_URL),
    ingestUrl: deriveWebSocketUrl(httpUrl).replace('/scene/stream', '/camera/ingest'), captureWidth, captureHeight,
    cameraDemo: demo && new URLSearchParams(search).get('camera') === '1',
    staleMs: seconds * 1000, mirrorX: env.VITE_MIRROR_X === 'true',
    demo };
}
