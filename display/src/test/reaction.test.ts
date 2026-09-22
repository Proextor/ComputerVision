import { describe, expect, it } from 'vitest';
import { baselineFor, ReactionController } from '../reaction';
import type { Scene } from '../contracts';
import { scene, state } from './helpers';

function controller(input = scene(), mirror = false, replay = false) {
  const c = new ReactionController(mirror); c.updateTransport(state(input)); c.ingest(input, 0, replay); return c;
}
function event(input: Scene, type: string, id: number) {
  return { event_id: id, session_id: input.session_id, timestamp_utc: input.timestamp_utc, type, track_id: 1, metadata: {} };
}
describe('baselines and reactions', () => {
  it.each([[0, 'dormant'], [1, 'waking'], [2, 'approaching'], [6, 'active'], [10, 'camera_unavailable'], [12, 'eof']] as const)('maps fixture %i to %s', (index, expected) => {
    expect(controller(scene(index)).step(0, 0).baseline).toBe(expected);
  });
  it('distinguishes no connection, delayed data and unknown values', () => {
    expect(baselineFor({ ...state(), connection: 'offline' })).toBe('offline');
    expect(baselineFor({ ...state(), localStale: true })).toBe('stale');
    expect(baselineFor({ ...state(), serviceStatus: 'future' })).toBe('stale');
    expect(baselineFor(state({ ...scene(), camera_status: 'future' }))).toBe('camera_unavailable');
  });
  it.each([[1, 'wake'], [3, 'acknowledge'], [4, 'wave_back'], [5, 'celebrate'], [8, 'focus_changed']] as const)('maps event %i to %s', (index, reaction) => {
    expect(controller(scene(index)).step(0, 0).transient).toBe(reaction);
  });
  it('deduplicates events and bounds cache to 512 entries', () => {
    const input = scene(3), c = controller(input);
    c.ingest(input, 5000); expect(c.step(5000, 0).transient).toBeNull();
    input.events = Array.from({ length: 600 }, (_, i) => event(input, 'future', i + 100));
    c.ingest(input, 6000); expect(c.deduplicationSize).toBe(512);
  });
  it('clears cache and cooldown when the session changes', () => {
    const input = scene(3), c = controller(input);
    input.session_id = 'new'; input.events[0].session_id = 'new';
    c.ingest(input, 100); expect(c.step(100, 0).transient).toBe('acknowledge');
    expect(c.deduplicationSize).toBe(1);
  });
  it('chooses deterministic priority and cannot be interrupted by lower events', () => {
    const input = scene(); input.events = ['hand_raise_started', 'wave_detected', 'both_hands_raised_started'].map((type, i) => event(input, type, i + 1));
    const c = controller(input); expect(c.step(0, 0).transient).toBe('celebrate');
    input.events = [event(input, 'wave_detected', 5)]; c.ingest(input, 100);
    expect(c.step(100, 0).transient).toBe('celebrate');
  });
  it('higher priority interrupts and cooldown prevents immediate repeat', () => {
    const input = scene(3), c = controller(input);
    input.events = [event(input, 'both_hands_raised_started', 20)]; c.ingest(input, 100);
    expect(c.step(100, 0).transient).toBe('celebrate');
    input.events = [event(input, 'both_hands_raised_started', 21)]; c.ingest(input, 4000);
    expect(c.step(4000, 0).transient).toBeNull();
    input.events = [event(input, 'both_hands_raised_started', 22)]; c.ingest(input, 5000);
    expect(c.step(5000, 0).transient).toBe('celebrate');
  });
  it('expires into the current baseline, not the baseline at reaction start', () => {
    const c = controller(scene(5)); c.updateTransport(state(scene(2)));
    expect(c.step(3500, 0)).toMatchObject({ baseline: 'approaching', transient: null });
  });
  it('failure overrides reactions; reconnect restores scene without replay', () => {
    const input = scene(5), c = controller(input);
    c.updateTransport({ ...state(input), serviceStatus: 'error' });
    expect(c.step(100, 0)).toMatchObject({ baseline: 'camera_unavailable', transient: null });
    c.updateTransport(state(input)); c.ingest(input, 1000, true);
    expect(c.step(1000, 0)).toMatchObject({ baseline: 'present', transient: null });
  });
  it('suppresses stale event timestamps and focus removal events', () => {
    const input = scene(3); input.events[0].timestamp_utc = '2020-01-01T00:00:00Z';
    expect(controller(input).step(0, 0).transient).toBeNull();
    const focus = scene(8); focus.room_state.focus_candidate = null;
    expect(controller(focus).step(0, 0).transient).toBeNull();
  });
});
describe('focus', () => {
  it('finds current candidate anchor and eases without an instant jump', () => {
    const input = scene(8), c = controller(input);
    expect(c.step(0, 0).focusX).toBe(0);
    const first = c.step(100, 0.1).focusX;
    expect(first).toBeGreaterThan(0); expect(first).toBeLessThan(0.09);
    expect(c.step(10000, 10).focusX).toBeCloseTo(0.09, 4);
  });
  it('uses centroid for a missing focus person and mirrors coordinates', () => {
    const input = scene(); input.room_state.focus_candidate!.track_id = 99;
    input.room_state.spatial.centroid = { x: 0.9, y: 0.5 };
    expect(controller(input).step(10000, 10).focusX).toBeCloseTo(0.144, 4);
    expect(controller(input, true).step(10000, 10).focusX).toBeCloseTo(-0.144, 4);
  });
  it('holds an invisible retained candidate briefly, then returns to centre', () => {
    const input = scene(8), c = controller(input); const x = c.step(10000, 10).focusX;
    input.room_state.focus_candidate!.is_visible = false; c.updateTransport(state(input));
    expect(c.step(10100, 0.1).focusX).toBeCloseTo(x, 4);
    expect(c.step(11000, 1).focusX).toBeLessThan(x / 2);
  });
  it('is time based and handles missing all positions', () => {
    const a = controller(scene(8)), b = controller(scene(8));
    const x = a.step(1000, 1).focusX;
    for (let i = 0; i < 60; i++) b.step(i * 1000 / 60, 1 / 60);
    expect(b.step(1000, 0).focusX).toBeCloseTo(x, 8);
    const input = scene(); input.people[0].anchor_point = null; input.room_state.spatial.centroid = null;
    expect(controller(input).step(1000, 1).focusX).toBe(0);
  });
});
