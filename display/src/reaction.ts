import type { Scene, SceneEvent } from './contracts';
import { initialTransport, type TransportState } from './transport';
import { parseVerificationEvent, type ProviderState, type VerificationStatus } from './verification';

export type Baseline = 'offline' | 'stale' | 'camera_unavailable' | 'eof' | 'dormant' | 'waking' | 'present' | 'approaching' | 'active' | 'leaving' | 'verification_pending' | 'verified' | 'verification_retry';
export type Reaction = 'wake' | 'acknowledge' | 'celebrate' | 'wave_back' | 'focus_changed' | 'leaving' | 'verification_pending' | 'verified' | 'verification_retry';
export const REACTIONS: Record<Reaction, { priority: number; duration: number; cooldown: number; caption: string }> = {
  focus_changed: { priority: 1, duration: 1200, cooldown: 2000, caption: 'Right here.' },
  wake: { priority: 2, duration: 1600, cooldown: 4000, caption: 'Hello' },
  acknowledge: { priority: 3, duration: 2000, cooldown: 3000, caption: 'Hello to you, too.' },
  wave_back: { priority: 4, duration: 2800, cooldown: 3500, caption: 'A little wave back.' },
  celebrate: { priority: 5, duration: 1400, cooldown: 4500, caption: 'A shared moment.' },
  leaving: { priority: 2, duration: 1700, cooldown: 0, caption: 'See you soon.' },
  verification_retry: { priority: 6, duration: 2200, cooldown: 0, caption: 'Couldn’t verify — try again' },
  verification_pending: { priority: 7, duration: 30_000, cooldown: 0, caption: 'Verifying…' },
  verified: { priority: 8, duration: 1800, cooldown: 0, caption: 'Verified' },
};
const BASELINES: Record<Baseline, { hue: number; intensity: number; scale: number; caption: string; label: string }> = {
  dormant: { hue: 190, intensity: 0.38, scale: 0.9, caption: 'Quietly here.', label: 'ROOM AT REST' },
  waking: { hue: 168, intensity: 0.72, scale: 1, caption: 'Hello, there.', label: 'WELCOME' },
  leaving: { hue: 190, intensity: 0.38, scale: 0.9, caption: 'See you soon.', label: 'GOODBYE' },
  present: { hue: 168, intensity: 0.72, scale: 1, caption: 'Hello, there.', label: 'PRESENCE DETECTED' },
  approaching: { hue: 38, intensity: 0.85, scale: 1.08, caption: 'A little closer.', label: 'APPROACH DETECTED' },
  active: { hue: 262, intensity: 0.95, scale: 1.04, caption: 'In motion.', label: 'MOVEMENT IN THE ROOM' },
  offline: { hue: 210, intensity: 0.19, scale: 0.86, caption: 'Waiting to connect.', label: 'SERVICE OFFLINE · RECONNECTING' },
  stale: { hue: 215, intensity: 0.24, scale: 0.88, caption: 'Taking a quiet pause.', label: 'LIVE DATA DELAYED' },
  camera_unavailable: { hue: 220, intensity: 0.22, scale: 0.88, caption: 'Waiting for the room.', label: 'CAMERA OR SERVICE UNAVAILABLE' },
  eof: { hue: 220, intensity: 0.29, scale: 0.9, caption: 'Until next time.', label: 'VIDEO COMPLETE' },
  verification_pending: { hue: 190, intensity: 0.7, scale: 1, caption: 'Verifying…', label: 'VERIFICATION IN PROGRESS' },
  verified: { hue: 42, intensity: 0.9, scale: 1.05, caption: 'Verified', label: 'VERIFIED' },
  verification_retry: { hue: 205, intensity: 0.6, scale: 1, caption: 'Couldn’t verify — try again', label: 'TRY AGAIN WHEN READY' },
};
export const isFailure = (baseline: Baseline): boolean => ['offline', 'stale', 'camera_unavailable', 'eof'].includes(baseline);
export interface VisualState {
  baseline: Baseline; transient: Reaction | null; progress: number;
  hue: number; intensity: number; scale: number; focusX: number; activity: number;
  caption: string; label: string; lastEvent: string | null;
  verificationStatus: VerificationStatus | null; verificationSessionId: string | null;
}
export function baselineFor(state: TransportState): Baseline {
  if (!['connected', 'demo'].includes(state.connection)) return 'offline';
  const scene = state.latest;
  if (state.serviceStatus === 'error' || state.serviceStatus === 'unavailable' || state.serviceStatus === 'stopping') return 'camera_unavailable';
  if (state.serviceStatus === 'eof' || scene?.camera_status === 'eof') return 'eof';
  if (scene && scene.camera_status !== 'ok') return 'camera_unavailable';
  if (state.localStale || state.serviceStatus === 'stale' || !scene || state.serviceStatus !== 'ready') return 'stale';
  const room = scene.room_state;
  if (room.occupancy.state === 'empty') return 'dormant';
  if (['moderate', 'high'].includes(room.activity.aggregate_activity_level)) return 'active';
  if (room.interaction.state === 'approaching') return 'approaching';
  if (scene.visible_count > 0 || (room.occupancy.stabilized_count ?? 0) > 0) return 'present';
  return 'stale';
}

export class ReactionController {
  private transport = initialTransport();
  private session: string | null = null;
  private seen = new Set<number>();
  private cooldowns = new Map<Reaction, number>();
  private current: { type: Reaction; started: number } | null = null;
  private lastEvent: string | null = null;
  private focusX = 0;
  private targetX = 0;
  private focusLostAt: number | null = null;
  private verificationProvider: ProviderState = 'unavailable';
  private verificationStatus: VerificationStatus | null = null;
  private verificationSessionId: string | null = null;
  private verificationSeen = new Set<string>();
  private verificationStart = 0;
  private verificationArmedUntil = -Infinity;

  constructor(private mirrorX = false) {}
  get deduplicationSize(): number { return this.seen.size; }
  get verificationCacheSize(): number { return this.verificationSeen.size; }
  get providerState(): ProviderState { return this.verificationProvider; }
  armVerification(now: number): void {
    if (this.verificationProvider === 'ready' || this.verificationProvider === 'demo') this.verificationArmedUntil = now + 5000;
  }
  setVerificationProvider(state: ProviderState): void {
    this.verificationProvider = state;
    if (state !== 'ready' && state !== 'demo') {
      this.verificationStatus = null; this.verificationSessionId = null;
      this.verificationArmedUntil = -Infinity;
      if (this.current && ['verified', 'verification_pending', 'verification_retry'].includes(this.current.type)) this.current = null;
    }
  }
  ingestVerification(input: unknown, now: number, wallNow = Date.now()): boolean {
    if (!['ready', 'demo'].includes(this.verificationProvider) || isFailure(baselineFor(this.transport))) return false;
    const event = parseVerificationEvent(input, wallNow);
    if (!event) return false;
    const key = `${event.verification_session_id}\u0000${event.event_id}`;
    if (this.verificationSeen.has(key)) return false;
    if (event.status === 'pending') {
      if (now > this.verificationArmedUntil) return false;
      this.verificationArmedUntil = -Infinity;
      if (this.verificationSessionId && this.verificationSessionId !== event.verification_session_id &&
          this.current?.type === 'verification_pending') this.current = null;
      this.verificationSessionId = event.verification_session_id;
      this.verificationStart = now;
      this.verificationStatus = 'pending';
      this.current = { type: 'verification_pending', started: now };
    } else {
      // A terminal event requires a user-initiated pending session observed in this page lifetime.
      if (this.verificationSessionId !== event.verification_session_id || this.verificationStatus !== 'pending' ||
          now - this.verificationStart > 30_000) return false;
      this.verificationStatus = event.status;
      this.current = event.status === 'verified' ? { type: 'verified', started: now } :
        ['rejected', 'error'].includes(event.status) ? { type: 'verification_retry', started: now } : null;
    }
    this.verificationSeen.add(key);
    if (this.verificationSeen.size > 256) this.verificationSeen.delete(this.verificationSeen.values().next().value!);
    return true;
  }

  updateTransport(state: TransportState): void {
    this.transport = state;
    if (isFailure(baselineFor(state))) { this.current = null; this.verificationStatus = null; this.verificationSessionId = null; this.verificationArmedUntil = -Infinity; }
    if (this.verificationProvider === 'ready' && state.connection !== 'connected') this.setVerificationProvider('stale');
  }
  ingest(scene: Scene, now: number, replay = false): void {
    if (scene.session_id !== this.session) {
      this.session = scene.session_id; this.seen.clear(); this.cooldowns.clear();
      this.current = null; this.lastEvent = null; this.focusLostAt = null; this.targetX = 0;
      this.verificationStatus = null; this.verificationSessionId = null; this.verificationArmedUntil = -Infinity;
    }
    this.transport = { ...this.transport, latest: scene };
    this.expire(now);
    const candidates: { type: Reaction; event: SceneEvent }[] = [];
    for (const event of scene.events) {
      if (event.session_id !== this.session || this.seen.has(event.event_id)) continue;
      this.seen.add(event.event_id);
      if (this.seen.size > 512) this.seen.delete(this.seen.values().next().value!);
      if (replay || Date.parse(scene.timestamp_utc) - Date.parse(event.timestamp_utc) > 10_000) continue;
      const type = this.mapEvent(event, scene);
      if (type) candidates.push({ type, event });
    }
    if (isFailure(baselineFor(this.transport))) { this.current = null; return; }
    candidates.sort((a, b) => REACTIONS[b.type].priority - REACTIONS[a.type].priority || a.event.event_id - b.event.event_id);
    for (const { type, event } of candidates) {
      const rule = REACTIONS[type];
      if (now - (this.cooldowns.get(type) ?? -Infinity) < rule.cooldown) continue;
      if (this.current && REACTIONS[this.current.type].priority >= rule.priority) continue;
      this.current = { type, started: now }; this.cooldowns.set(type, now);
      this.lastEvent = `${event.type} · #${event.event_id}`; break;
    }
  }
  private mapEvent(event: SceneEvent, scene: Scene): Reaction | null {
    switch (event.type) {
      case 'room_became_occupied': return 'wake';
      case 'room_became_empty': return 'leaving';
      case 'hand_raise_started': return 'acknowledge';
      case 'both_hands_raised_started': return 'celebrate';
      case 'wave_detected': return 'wave_back';
      case 'focus_candidate_changed': {
        const focus = scene.room_state.focus_candidate;
        return focus?.is_visible && focus.score > 0 && focus.evidence.length > 0 ? 'focus_changed' : null;
      }
      default: return null;
    }
  }
  private expire(now: number): void {
    if (this.current && now - this.current.started >= REACTIONS[this.current.type].duration) {
      if (this.current.type === 'verification_pending') { this.verificationStatus = 'expired'; this.verificationSessionId = null; }
      this.current = null;
    }
  }
  step(now: number, dtSeconds: number): VisualState {
    this.expire(now);
    const baseline = baselineFor(this.transport);
    if (isFailure(baseline)) { this.current = null; this.targetX = 0; }
    else {
      const scene = this.transport.latest;
      const focus = scene?.room_state.focus_candidate;
      if (focus && !focus.is_visible) {
        this.focusLostAt ??= now;
        if (now - this.focusLostAt > 800) this.targetX = 0;
      } else {
        this.focusLostAt = null;
        const person = scene?.people.find(p => p.track_id === focus?.track_id);
        const x = person?.anchor_point?.x ?? scene?.room_state.spatial.centroid?.x ?? 0.5;
        this.targetX = Math.max(-0.18, Math.min(0.18, ((this.mirrorX ? 1 - x : x) - 0.5) * 0.36));
      }
    }
    const dt = Number.isFinite(dtSeconds) ? Math.max(0, dtSeconds) : 0;
    this.focusX += (this.targetX - this.focusX) * (1 - Math.exp(-dt / 0.65));
    const active = this.current;
    const effective = active?.type === 'verified' || active?.type === 'verification_pending' || active?.type === 'verification_retry' || active?.type === 'leaving' ? active.type :
      active?.type === 'wake' ? 'waking' : baseline;
    const settings = BASELINES[effective];
    const reaction = this.current;
    return { baseline: isFailure(baseline) ? baseline : effective, transient: reaction?.type ?? null,
      progress: reaction ? Math.max(0, Math.min(1, (now - reaction.started) / REACTIONS[reaction.type].duration)) : 0,
      ...settings, focusX: this.focusX,
      activity: isFailure(baseline) ? 0 : this.transport.latest?.room_state.activity.aggregate_activity_score ?? 0,
      caption: reaction ? REACTIONS[reaction.type].caption : settings.caption,
      label: reaction ? reaction.type.replaceAll('_', ' ').toUpperCase() : settings.label,
      lastEvent: this.lastEvent, verificationStatus: this.verificationStatus,
      verificationSessionId: this.verificationSessionId };
  }
}
