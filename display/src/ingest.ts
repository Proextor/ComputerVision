export type IngestStatus = 'disconnected' | 'connecting' | 'authenticating' | 'sending' | 'error';
export interface IngestState { status: IngestStatus; sessionId: string | null; sent: number; dropped: number }
export const CAPTURE_FPS = 10;
export const MAX_BUFFERED = 256 * 1024;
export function captureDimensions(width: number, height: number, maxWidth = 640, maxHeight = 360) {
  const scale = Math.min(1, maxWidth / width, maxHeight / height);
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}
export function encodeVideo(video: HTMLVideoElement, canvas: HTMLCanvasElement, maxWidth: number, maxHeight: number): Promise<Blob | null> {
  if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return Promise.resolve(null);
  const size = captureDimensions(video.videoWidth, video.videoHeight, maxWidth, maxHeight);
  canvas.width = size.width; canvas.height = size.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return Promise.resolve(null);
  // CSS mirrors only the preview. These source pixels remain unmirrored and uncropped.
  ctx.drawImage(video, 0, 0, size.width, size.height);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .78));
}

type Socket = Pick<WebSocket, 'readyState' | 'bufferedAmount' | 'send' | 'close' | 'onopen' | 'onmessage' | 'onerror' | 'onclose'>;
export class FrameSender {
  private socket: Socket | null = null;
  private timer?: ReturnType<typeof setInterval>;
  private authTimer?: ReturnType<typeof setTimeout>;
  private busy = false;
  private generation = 0;
  private nextAt = 0;
  private maxBytes = 512 * 1024;
  private state: IngestState = { status: 'disconnected', sessionId: null, sent: 0, dropped: 0 };
  private listeners = new Set<(state: IngestState) => void>();
  constructor(private encode: () => Promise<Blob | null>, private socketFactory: (url: string) => Socket = url => new WebSocket(url), private now = () => performance.now()) {}
  getState() { return this.state; }
  subscribe(fn: (state: IngestState) => void) { this.listeners.add(fn); fn(this.state); return () => { this.listeners.delete(fn); }; }
  private update(value: Partial<IngestState>) { this.state = { ...this.state, ...value }; for (const fn of this.listeners) fn(this.state); }
  connect(url: string, token: string): void {
    this.disconnect(); const generation = this.generation;
    if (token.length < 16 || token.length > 256) { this.update({ status: 'error' }); return; }
    this.update({ status: 'connecting', sent: 0, dropped: 0 });
    let socket: Socket;
    try { socket = this.socketFactory(url); } catch { this.update({ status: 'error' }); return; }
    this.socket = socket;
    socket.onopen = () => { if (this.socket === socket) { socket.send(JSON.stringify({ type: 'auth', token })); token = ''; this.update({ status: 'authenticating' }); } };
    socket.onmessage = event => {
      if (this.socket !== socket || this.state.status !== 'authenticating') return;
      try {
        const value = JSON.parse(String(event.data));
        if (value.type !== 'ready' || typeof value.session_id !== 'string' || value.session_id.length > 128 ||
            !Number.isFinite(value.max_frame_bytes) || value.max_frame_bytes <= 0 || value.max_frame_bytes > 512 * 1024 ||
            !Number.isFinite(value.max_fps) || value.max_fps < 1) throw new Error('invalid readiness');
        clearTimeout(this.authTimer); this.maxBytes = value.max_frame_bytes;
        this.update({ status: 'sending', sessionId: value.session_id }); this.nextAt = 0;
        this.timer = setInterval(() => { void this.tick(generation); }, 1000 / Math.min(CAPTURE_FPS, value.max_fps));
      } catch { this.disconnect(); this.update({ status: 'error' }); }
    };
    const failed = () => { if (this.socket === socket) { this.disconnect(); this.update({ status: 'error' }); } };
    socket.onerror = failed; socket.onclose = failed;
    this.authTimer = setTimeout(failed, 7000);
  }
  async tick(generation = this.generation): Promise<void> {
    const socket = this.socket;
    if (!socket || socket.readyState !== 1 || this.state.status !== 'sending' || this.now() + 2 < this.nextAt) return;
    this.nextAt = this.now() + 1000 / CAPTURE_FPS;
    if (this.busy || socket.bufferedAmount > MAX_BUFFERED || document.hidden) { this.update({ dropped: this.state.dropped + 1 }); return; }
    this.busy = true;
    try {
      const blob = await this.encode();
      if (generation !== this.generation || socket !== this.socket || socket.readyState !== 1) return;
      if (!blob || blob.size > this.maxBytes || socket.bufferedAmount > MAX_BUFFERED) { this.update({ dropped: this.state.dropped + 1 }); return; }
      socket.send(blob); this.update({ sent: this.state.sent + 1 });
    } catch { if (generation === this.generation) { this.disconnect(); this.update({ status: 'error' }); } }
    finally { if (generation === this.generation) this.busy = false; }
  }
  disconnect(): void {
    this.generation++; this.busy = false; clearInterval(this.timer); clearTimeout(this.authTimer);
    if (this.socket) { this.socket.onopen = this.socket.onmessage = this.socket.onerror = this.socket.onclose = null; this.socket.close(); this.socket = null; }
    this.update({ status: 'disconnected', sessionId: null });
  }
}
