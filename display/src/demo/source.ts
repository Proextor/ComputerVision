import fixtures from './fixtures.json';
import { parseScene, type Scene } from '../contracts';
import { initialTransport, type DataSource, type Listener, type TransportState } from '../transport';

const sequence = [
  [0, 'Quiet room'], [1, 'Someone arrives'], [1, 'Eyes wake'], [2, 'A little closer'],
  [2, 'Verification starts'], [2, 'Verification succeeds'], [2, 'Thumbs-up'],
  [3, 'One hand raised'], [4, 'A wave'], [5, 'Both hands raised'],
  [6, 'Room in motion'], [7, 'A second person'], [8, 'Focus moves'],
  [9, 'Everyone leaves'], [9, 'Goodbye and sleep'], [10, 'Source unavailable'],
  [13, 'New session'], [13, 'Verification retry'],
] as const;
export const DEMO_STEPS = sequence.map(([fixtureIndex, label]) => {
  const f = fixtures[fixtureIndex];
  const snapshot = parseScene(f.snapshot);
  if (!snapshot) throw new Error(`Invalid demo fixture: ${f.label}`);
  return { ...f, label, snapshot };
});
export class DemoSource implements DataSource {
  private listeners = new Set<Listener>();
  private timer?: ReturnType<typeof setInterval>;
  private state: TransportState = initialTransport();
  private index = 0;
  private playing = true;
  private elapsed = 0;
  private previousTime = 0;
  private sequence = 0;
  private eventId = 0;
  private cycle = 0;
  private scene: Scene | null = null;
  private stepListeners = new Set<(index: number) => void>();
  constructor(private now = () => performance.now()) {}
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener); listener(this.state);
    return () => { this.listeners.delete(listener); };
  }
  getState(): TransportState { return this.state; }
  subscribeStep(listener: (index: number) => void): () => void {
    this.stepListeners.add(listener); return () => { this.stepListeners.delete(listener); };
  }
  getControls() { return { index: this.index, playing: this.playing, label: DEMO_STEPS[this.index].label, total: DEMO_STEPS.length }; }
  stepLabel(index: number): string { return DEMO_STEPS[index]?.label ?? ''; }
  start(): void {
    if (this.timer !== undefined) return;
    this.previousTime = this.now(); this.publish(true);
    this.timer = setInterval(() => {
      const now = this.now();
      if (this.playing) this.elapsed += Math.min(1000, now - this.previousTime);
      this.previousTime = now;
      if (this.elapsed >= 6000) this.next(); else this.publish(false);
    }, 200);
  }
  stop(): void { clearInterval(this.timer); this.timer = undefined; }
  toggle(): void { this.playing = !this.playing; this.previousTime = this.now(); this.publish(false); }
  next(): void { if (this.index === DEMO_STEPS.length - 1) this.cycle++; this.select((this.index + 1) % DEMO_STEPS.length); }
  previous(): void { this.select((this.index + DEMO_STEPS.length - 1) % DEMO_STEPS.length); }
  restart(): void { this.cycle++; this.playing = true; this.select(0); }
  selectStep(index: number): void {
    if (Number.isSafeInteger(index) && index >= 0 && index < DEMO_STEPS.length) this.select(index);
  }
  private select(index: number): void { this.index = index; this.elapsed = 0; this.publish(true); }
  private publish(transition: boolean): void {
    const step = DEMO_STEPS[this.index];
    const session = `${step.snapshot.session_id}-${this.cycle}`;
    if (transition || !this.scene) this.scene = { ...step.snapshot, session_id: session,
      events: step.snapshot.events.map(e => ({ ...e, session_id: session, event_id: ++this.eventId })) };
    else this.scene = { ...this.scene, events: [] };
    const stale = step.service_status === 'stale';
    this.state = { connection: 'demo', serviceStatus: step.service_status, latest: this.scene,
      publicationSequence: ++this.sequence, localStale: stale, snapshotAgeSeconds: stale ? 6 : 0,
      heartbeat: { type: 'heartbeat', publication_sequence: this.sequence,
        service_status: step.service_status, last_snapshot_age_seconds: stale ? 6 : 0 } };
    const envelope = { type: 'snapshot' as const, publication_sequence: this.sequence,
      published_at: new Date().toISOString(), snapshot: this.scene };
    for (const listener of this.listeners) listener(this.state, { envelope, replay: false });
    if (transition) for (const listener of this.stepListeners) listener(this.index);
  }
}
