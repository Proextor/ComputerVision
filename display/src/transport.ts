import type { HeartbeatEnvelope, Scene, SnapshotEnvelope } from './contracts';

export type Connection = 'connecting' | 'connected' | 'reconnecting' | 'offline' | 'demo';
export interface TransportState {
  connection: Connection; serviceStatus: string; latest: Scene | null;
  heartbeat: HeartbeatEnvelope | null; publicationSequence: number;
  snapshotAgeSeconds: number | null; localStale: boolean;
}
export type Listener = (state: TransportState, delivery?: { envelope: SnapshotEnvelope; replay: boolean }) => void;
export interface DataSource {
  start(): void; stop(): void; subscribe(listener: Listener): () => void;
  getState(): TransportState;
}
export const initialTransport = (): TransportState => ({ connection: 'connecting', serviceStatus: 'starting',
  latest: null, heartbeat: null, publicationSequence: 0, snapshotAgeSeconds: null, localStale: false });
