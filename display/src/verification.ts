/** Sanitized, one-to-one result only. No biometric payload or scene track association. */
export type VerificationStatus = 'idle' | 'pending' | 'verified' | 'rejected' | 'cancelled' | 'expired' | 'error';
export interface VerificationEvent {
  event_id: string; verification_session_id: string; status: VerificationStatus;
  timestamp: string; expires_at?: string | null;
}
export type ProviderState = 'unavailable' | 'demo' | 'ready' | 'stale' | 'error';
export interface VerificationSource {
  readonly state: ProviderState;
  subscribe(listener: (event: unknown) => void): () => void;
  start(): void; stop(): void;
  /** Must be invoked from an explicit user action. */
  begin(): void;
}
const statuses = new Set<VerificationStatus>(['idle', 'pending', 'verified', 'rejected', 'cancelled', 'expired', 'error']);
const iso = (value: unknown): value is string => typeof value === 'string' &&
  /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)$/.test(value) && Number.isFinite(Date.parse(value));
export function parseVerificationEvent(input: unknown, now = Date.now()): VerificationEvent | null {
  if (typeof input !== 'object' || input === null || Array.isArray(input)) return null;
  const v = input as Record<string, unknown>;
  if (typeof v.event_id !== 'string' || !v.event_id.trim() || v.event_id.length > 128 ||
      typeof v.verification_session_id !== 'string' || !v.verification_session_id.trim() || v.verification_session_id.length > 128 ||
      !statuses.has(v.status as VerificationStatus) || !iso(v.timestamp) ||
      !(v.expires_at === undefined || v.expires_at === null || iso(v.expires_at))) return null;
  const timestamp = Date.parse(v.timestamp);
  if (timestamp > now + 5000 || now - timestamp > 30_000 ||
      (v.expires_at && (Date.parse(v.expires_at) <= now || Date.parse(v.expires_at) < timestamp))) return null;
  return { event_id: v.event_id, verification_session_id: v.verification_session_id,
    status: v.status as VerificationStatus, timestamp: v.timestamp,
    expires_at: v.expires_at as string | null | undefined };
}

/** Explicit demo only: never constructed for a live service. */
export class DemoVerificationSource implements VerificationSource {
  readonly state = 'demo';
  private listeners = new Set<(event: unknown) => void>();
  private counter = 0;
  private session: string | null = null;
  private timer?: ReturnType<typeof setTimeout>;
  subscribe(listener: (event: unknown) => void): () => void {
    this.listeners.add(listener); return () => { this.listeners.delete(listener); };
  }
  start(): void { /* User action starts a verification session. */ }
  stop(): void { clearTimeout(this.timer); this.session = null; }
  begin(): void { this.run('verified'); }
  beginPending(): void {
    clearTimeout(this.timer);
    const session = `demo-verification-${++this.counter}`;
    this.session = session; this.emit(session, 'pending');
  }
  finish(result: 'verified' | 'rejected'): void {
    clearTimeout(this.timer);
    if (this.session) this.emit(this.session, result);
  }
  run(result: 'verified' | 'rejected'): void {
    this.beginPending();
    const session = this.session!;
    this.timer = setTimeout(() => {
      if (this.session === session) this.emit(session, result);
    }, 1800);
  }
  private emit(session: string, status: VerificationStatus): void {
    const now = Date.now();
    const event: VerificationEvent = { event_id: `${session}-${status}`, verification_session_id: session,
      status, timestamp: new Date(now).toISOString(), expires_at: new Date(now + 10_000).toISOString() };
    for (const listener of this.listeners) listener(event);
  }
}
