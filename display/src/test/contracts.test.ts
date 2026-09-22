import { describe, expect, it } from 'vitest';
import { parseEnvelope, parseMessage, parseScene } from '../contracts';
import { deriveWebSocketUrl, readConfig } from '../config';
import { envelope, scene } from './helpers';

describe('M5 boundary', () => {
  it('parses snapshots and ignores additive fields', () => {
    const input = { ...envelope(), extra: true, snapshot: { ...scene(), future: { unused: true } } };
    expect(parseMessage(JSON.stringify(input))).toEqual({ ...input, extra: undefined, snapshot: scene() });
    expect(parseScene({ ...scene(), camera_status: 'future' })?.camera_status).toBe('future');
  });
  it('parses heartbeat with explicit unavailable age', () => {
    const heartbeat = { type: 'heartbeat', publication_sequence: 0, service_status: 'starting', last_snapshot_age_seconds: null };
    expect(parseEnvelope(heartbeat)).toEqual(heartbeat);
  });
  it.each([null, [], {}, '{bad', '{"type":"snapshot"}', 32])('rejects malformed input %s', value => {
    expect(parseMessage(value)).toBeNull();
  });
  it('rejects missing fields, invalid coordinates, mismatched count, NaN and wrong schema', () => {
    expect(parseScene({ ...scene(), room_state: undefined })).toBeNull();
    expect(parseScene({ ...scene(), schema_version: '5.0' })).toBeNull();
    expect(parseScene({ ...scene(), visible_count: 999 })).toBeNull();
    const input = scene(); input.people[0].anchor_point = { x: NaN, y: 0.5 };
    expect(parseScene(input)).toBeNull();
    input.people[0].anchor_point = { x: 2, y: 0.5 }; expect(parseScene(input)).toBeNull();
  });
  it('accepts nullable person anchors and IDs', () => {
    const input = scene(); input.people[0] = { track_id: null, anchor_point: null };
    expect(parseScene(input)).not.toBeNull();
  });
});
describe('configuration', () => {
  it('derives ws/wss, preserving base paths and ports', () => {
    expect(deriveWebSocketUrl('http://127.0.0.1:8000')).toBe('ws://127.0.0.1:8000/api/v1/scene/stream');
    expect(deriveWebSocketUrl('https://example.com/base/')).toBe('wss://example.com/base/api/v1/scene/stream');
    expect(deriveWebSocketUrl('http://localhost', 'wss://example.com/live')).toBe('wss://example.com/live');
  });
  it('rejects unsafe URL schemes and credentials', () => {
    expect(() => deriveWebSocketUrl('file:///a')).toThrow();
    expect(() => deriveWebSocketUrl('https://user:password@example.com')).toThrow();
    expect(() => deriveWebSocketUrl('http://localhost?key=x')).toThrow();
    expect(() => readConfig({ VITE_STALE_SECONDS: 'NaN' }, '')).toThrow();
  });
  it('selects explicit demo and mirror options', () => {
    expect(readConfig({ VITE_MIRROR_X: 'true' }, '?demo=1')).toMatchObject({ demo: true, mirrorX: true });
    expect(readConfig({}, '')).toMatchObject({ demo: false, staleMs: 3000 });
  });
});
