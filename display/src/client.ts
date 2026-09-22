import { isRecord, parseMessage, type SnapshotEnvelope } from './contracts';
import type { DisplayConfig } from './config';
import { initialTransport, type DataSource, type Listener, type TransportState } from './transport';

export type SocketLike = Pick<WebSocket, 'close' | 'onopen' | 'onclose' | 'onerror' | 'onmessage' | 'readyState'>;
interface Dependencies {
  now: () => number; wallNow: () => number; random: () => number;
  socketFactory: (url: string) => SocketLike;
  healthFetch: (url: string, signal: AbortSignal) => Promise<unknown>;
}
export const reconnectDelay = (attempt: number, random: number): number =>
  Math.min(10_000, 500 * 2 ** Math.min(attempt, 6) * (0.8 + Math.max(0, Math.min(1, random)) * 0.4));

export class VisionClient implements DataSource {
  private readonly deps: Dependencies;
  private state = initialTransport();
  private listeners = new Set<Listener>();
  private socket: SocketLike | null = null;
  private running = false;
  private generation = 0;
  private attempt = 0;
  private reconnectTimer?: ReturnType<typeof setTimeout>;
  private connectTimeout?: ReturnType<typeof setTimeout>;
  private tickTimer?: ReturnType<typeof setInterval>;
  private healthAbort: AbortController | null = null;
  private healthTimeout?: ReturnType<typeof setTimeout>;
  private lastHealth = -Infinity;
  private lastMessage = 0;
  private ageAnchor = 0;
  private ageAt = 0;
  private hasAge = false;

  constructor(private config: DisplayConfig, deps: Partial<Dependencies> = {}) {
    this.deps = {
      now: () => performance.now(), wallNow: () => Date.now(), random: Math.random,
      socketFactory: url => new WebSocket(url),
      healthFetch: async (url, signal) => {
        const response = await fetch(url, { signal });
        if (!response.ok) throw new Error('Health request failed');
        const value: unknown = await response.json(); return value;
      }, ...deps,
    };
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener); listener(this.getState());
    return () => { this.listeners.delete(listener); };
  }
  getState(): TransportState {
    const now = this.deps.now();
    const snapshotAgeSeconds = this.hasAge ? Math.max(0, this.ageAnchor + now - this.ageAt) / 1000 : null;
    return { ...this.state, snapshotAgeSeconds,
      localStale: this.state.connection === 'connected' && (
        now - this.lastMessage > this.config.staleMs ||
        (snapshotAgeSeconds !== null && snapshotAgeSeconds * 1000 > this.config.staleMs)) };
  }
  private emit(delivery?: { envelope: SnapshotEnvelope; replay: boolean }): void {
    const state = this.getState();
    for (const listener of this.listeners) listener(state, delivery);
  }
  start(): void {
    if (this.running) return;
    this.running = true; this.generation++;
    this.state = { ...this.state, connection: 'connecting' };
    this.emit();
    // Deferring creation lets Strict Mode's trial effect clean up before opening a socket.
    this.reconnectTimer = setTimeout(() => this.connect(), 0);
    this.tickTimer = setInterval(() => {
      if (this.state.connection === 'connected' && this.deps.now() - this.lastMessage > Math.max(8000, this.config.staleMs * 3)) this.disconnected();
      if (this.state.connection !== 'connected') this.checkHealth();
      this.emit();
    }, 250);
  }
  stop(): void {
    this.running = false; this.generation++;
    clearTimeout(this.reconnectTimer); clearTimeout(this.connectTimeout); clearTimeout(this.healthTimeout); clearInterval(this.tickTimer);
    this.healthAbort?.abort(); this.healthAbort = null;
    this.releaseSocket();
    this.state = { ...this.state, connection: 'offline' };
  }
  private releaseSocket(): void {
    if (!this.socket) return;
    this.socket.onopen = this.socket.onmessage = this.socket.onclose = this.socket.onerror = null;
    this.socket.close(); this.socket = null;
  }
  private connect(): void {
    if (!this.running || this.socket) return;
    const generation = this.generation;
    let socket: SocketLike;
    try { socket = this.deps.socketFactory(this.config.wsUrl); }
    catch { this.disconnected(); return; }
    this.socket = socket;
    let firstSnapshot = true;
    let lastSequence = 0;
    const current = () => this.running && this.generation === generation && this.socket === socket;
    socket.onopen = () => {
      if (!current()) return;
      clearTimeout(this.connectTimeout);
      clearTimeout(this.healthTimeout);
      this.healthAbort?.abort(); this.healthAbort = null;
      this.lastMessage = this.deps.now();
      this.state = { ...this.state, connection: 'connected', serviceStatus: 'starting' };
      this.emit();
    };
    socket.onmessage = event => {
      if (!current()) return;
      const envelope = parseMessage(event.data);
      if (!envelope) return;
      const now = this.deps.now(); this.lastMessage = now;
      if (envelope.type === 'heartbeat') {
        if (envelope.publication_sequence < lastSequence) return;
        this.state = { ...this.state, heartbeat: envelope, serviceStatus: envelope.service_status,
          publicationSequence: Math.max(lastSequence, envelope.publication_sequence) };
        if (envelope.last_snapshot_age_seconds !== null) {
          this.ageAnchor = envelope.last_snapshot_age_seconds * 1000; this.ageAt = now; this.hasAge = true;
        }
        this.emit(); return;
      }
      if (!firstSnapshot && envelope.publication_sequence <= lastSequence) return;
      lastSequence = envelope.publication_sequence;
      const replay = firstSnapshot; firstSnapshot = false; this.attempt = 0;
      this.ageAnchor = replay ? Math.max(0, this.deps.wallNow() - Date.parse(envelope.published_at)) : 0;
      this.ageAt = now; this.hasAge = true;
      const status = envelope.snapshot.camera_status;
      this.state = { ...this.state, latest: envelope.snapshot, heartbeat: null,
        publicationSequence: envelope.publication_sequence,
        serviceStatus: status === 'ok' ? 'ready' : status === 'eof' ? 'eof' : 'unavailable' };
      this.emit({ envelope, replay });
    };
    socket.onerror = () => { if (current()) this.disconnected(); };
    socket.onclose = () => { if (current()) this.disconnected(); };
    this.connectTimeout = setTimeout(() => { if (current()) this.disconnected(); }, 8000);
  }
  private disconnected(): void {
    if (!this.running) return;
    clearTimeout(this.connectTimeout); this.releaseSocket();
    this.state = { ...this.state, connection: 'reconnecting' };
    this.emit(); this.checkHealth();
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.connect(), reconnectDelay(this.attempt++, this.deps.random()));
  }
  private checkHealth(): void {
    if (!this.running || this.healthAbort || this.deps.now() - this.lastHealth < 5000) return;
    this.lastHealth = this.deps.now();
    const abort = new AbortController(); this.healthAbort = abort;
    const generation = this.generation;
    const timeout = setTimeout(() => abort.abort(), 2500); this.healthTimeout = timeout;
    void this.deps.healthFetch(`${this.config.httpUrl}/health`, abort.signal).then(value => {
      if (this.running && generation === this.generation && this.state.connection !== 'connected' &&
          isRecord(value) && typeof value.service_status === 'string') {
        this.state = { ...this.state, serviceStatus: value.service_status }; this.emit();
      }
    }).catch(() => { /* Socket retries own recovery; health is diagnostic only. */ }).finally(() => {
      clearTimeout(timeout); if (this.healthAbort === abort) this.healthAbort = null;
    });
  }
}
