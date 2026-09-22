import fixtures from '../demo/fixtures.json';
import { parseScene, type Scene, type SnapshotEnvelope } from '../contracts';
import { initialTransport, type TransportState } from '../transport';

export const scene = (index = 1): Scene => structuredClone(parseScene(fixtures[index].snapshot)!);
export const envelope = (snapshot = scene(), sequence = 1): SnapshotEnvelope => ({
  type: 'snapshot', publication_sequence: sequence, published_at: new Date().toISOString(), snapshot,
});
export const state = (snapshot = scene()): TransportState => ({ ...initialTransport(),
  connection: 'connected', serviceStatus: 'ready', latest: snapshot, snapshotAgeSeconds: 0 });
