import { beforeEach, describe, expect, it, vi } from 'vitest';
import { reconnectDelay, VisionClient, type SocketLike } from '../client';
import { readConfig } from '../config';
import { envelope, scene } from './helpers';

export class FakeSocket implements SocketLike {
  readyState = 0;
  onopen: WebSocket['onopen'] = null;
  onclose: WebSocket['onclose'] = null;
  onerror: WebSocket['onerror'] = null;
  onmessage: WebSocket['onmessage'] = null;
  close = vi.fn(() => { this.readyState = 3; });
  open() { this.readyState = 1; this.onopen?.call(this as unknown as WebSocket, new Event('open')); }
  message(value: unknown) { this.onmessage?.call(this as unknown as WebSocket, new MessageEvent('message', { data: JSON.stringify(value) })); }
  disconnect() { this.onclose?.call(this as unknown as WebSocket, new CloseEvent('close')); }
}
describe('VisionClient', () => {
  beforeEach(() => vi.useFakeTimers());
  function setup() {
    const sockets: FakeSocket[] = [];
    const health = vi.fn(async () => ({ service_status: 'ready' }));
    const client = new VisionClient(readConfig({}, ''), { now: Date.now, wallNow: Date.now, random: () => 0.5,
      socketFactory: () => { const socket = new FakeSocket(); sockets.push(socket); return socket; }, healthFetch: health });
    const listener = vi.fn(); client.subscribe(listener); client.start(); vi.advanceTimersByTime(0);
    return { client, sockets, health, listener };
  }
  it('connects, parses data, keeps one latest scene and ignores malformed/out-of-order data', () => {
    const { client, sockets, listener, health } = setup(); sockets[0].open();
    sockets[0].message(envelope(scene(1), 2));
    expect(listener.mock.lastCall?.[1].replay).toBe(true);
    sockets[0].message(envelope(scene(2), 3));
    expect(listener.mock.lastCall?.[1].replay).toBe(false);
    sockets[0].message(envelope(scene(1), 2)); sockets[0].message({ nonsense: true });
    expect(client.getState().publicationSequence).toBe(3);
    expect(client.getState().latest?.frame_id).toBe(2);
    vi.advanceTimersByTime(1000); expect(health).not.toHaveBeenCalled(); client.stop();
  });
  it('defers creation so Strict Mode setup-cleanup-setup leaves one socket', () => {
    const factory = vi.fn(() => new FakeSocket());
    const client = new VisionClient(readConfig({}, ''), { socketFactory: factory });
    client.start(); client.stop(); client.start(); client.start(); vi.advanceTimersByTime(0);
    expect(factory).toHaveBeenCalledTimes(1); client.stop();
    expect(factory.mock.results[0].value.close).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
  it('reconnects after bounded backoff and treats initial latest as replay', () => {
    const { client, sockets, listener } = setup(); sockets[0].open(); sockets[0].message(envelope()); sockets[0].disconnect();
    vi.advanceTimersByTime(499); expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1); expect(sockets).toHaveLength(2);
    sockets[1].open(); sockets[1].message(envelope(scene(2), 1));
    expect(listener.mock.lastCall?.[1].replay).toBe(true);
    expect(client.getState().latest?.room_state.interaction.state).toBe('approaching'); client.stop();
    for (let i = 0; i < 100; i++) expect(reconnectDelay(i, 1)).toBeLessThanOrEqual(10000);
    expect(reconnectDelay(1, 0)).not.toBe(reconnectDelay(1, 1));
  });
  it('stops reconnects and ignores callbacks after disposal', () => {
    const { client, sockets } = setup(); sockets[0].disconnect(); client.stop();
    vi.advanceTimersByTime(60000); expect(sockets).toHaveLength(1);
    expect(client.getState().connection).toBe('offline');
    expect(sockets[0].onmessage).toBeNull();
  });
  it('detects local staleness and recovers a silent half-open connection', () => {
    const { client, sockets } = setup(); sockets[0].open(); sockets[0].message(envelope());
    vi.advanceTimersByTime(3250); expect(client.getState().localStale).toBe(true);
    vi.advanceTimersByTime(6000); expect(client.getState().connection).toBe('reconnecting'); client.stop();
  });
  it('heartbeats cannot make an old scene appear fresh', () => {
    const { client, sockets } = setup(); sockets[0].open(); sockets[0].message(envelope());
    sockets[0].message({ type: 'heartbeat', publication_sequence: 1, service_status: 'stale', last_snapshot_age_seconds: 5 });
    expect(client.getState()).toMatchObject({ localStale: true, serviceStatus: 'stale', snapshotAgeSeconds: 5 }); client.stop();
  });
  it('counts the age of a cached latest snapshot on reconnect', () => {
    const { client, sockets } = setup(); sockets[0].open();
    sockets[0].message({ ...envelope(), published_at: new Date(Date.now() - 6000).toISOString() });
    expect(client.getState().localStale).toBe(true); client.stop();
  });
});
