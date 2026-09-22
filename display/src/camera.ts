export type CameraStatus = 'idle' | 'requesting' | 'active' | 'denied' | 'missing' | 'busy' | 'interrupted' | 'unsupported' | 'error';
export interface CameraState { status: CameraStatus; facing: 'user' | 'environment'; stream: MediaStream | null }

/** Local video only. Nothing here records, encodes or uploads frames. */
export class CameraController {
  private state: CameraState = { status: 'idle', facing: 'user', stream: null };
  private listeners = new Set<(state: CameraState) => void>();
  private generation = 0;
  constructor(private media = navigator.mediaDevices) {}
  getState(): CameraState { return this.state; }
  subscribe(fn: (state: CameraState) => void) { this.listeners.add(fn); fn(this.state); return () => { this.listeners.delete(fn); }; }
  private update(value: Partial<CameraState>) { this.state = { ...this.state, ...value }; for (const fn of this.listeners) fn(this.state); }
  private release() {
    for (const track of this.state.stream?.getTracks() ?? []) { track.onended = null; track.onmute = null; track.onunmute = null; track.stop(); }
    this.state = { ...this.state, stream: null };
  }
  async start(facing: CameraState['facing'] = 'user'): Promise<void> {
    const generation = ++this.generation;
    this.release(); this.update({ status: 'requesting', facing });
    if (!this.media?.getUserMedia) { this.update({ status: 'unsupported' }); return; }
    try {
      const stream = await this.media.getUserMedia({ audio: false, video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } } });
      if (generation !== this.generation) { stream.getTracks().forEach(track => track.stop()); return; }
      const track = stream.getVideoTracks()[0];
      if (!track) { stream.getTracks().forEach(t => t.stop()); this.update({ status: 'missing' }); return; }
      const actual = track.getSettings().facingMode;
      this.update({ stream, status: 'active', facing: actual === 'environment' || actual === 'user' ? actual : facing });
      track.onended = () => { if (generation === this.generation) { this.release(); this.update({ status: 'interrupted' }); } };
      track.onmute = () => { if (generation === this.generation) this.update({ status: 'interrupted' }); };
      track.onunmute = () => { if (generation === this.generation) this.update({ status: 'active' }); };
    } catch (error) {
      if (generation !== this.generation) return;
      const name = error instanceof DOMException || error instanceof Error ? error.name : '';
      this.update({ status: name === 'NotAllowedError' || name === 'SecurityError' ? 'denied' :
        name === 'NotFoundError' || name === 'OverconstrainedError' ? 'missing' :
          name === 'NotReadableError' || name === 'AbortError' ? 'busy' : 'error' });
    }
  }
  async switchCamera(): Promise<void> { await this.start(this.state.facing === 'user' ? 'environment' : 'user'); }
  stop(): void { ++this.generation; this.release(); this.update({ status: 'idle' }); }
}
