import { expect, it, vi } from 'vitest';
import { DEMO_STEPS, DemoSource } from '../demo/source';
import { parseScene } from '../contracts';
import { createSource } from '../source';
import { readConfig } from '../config';
import { motionParameters, PARTICLE_LIMIT, pixelRatio } from '../renderer';
import { ReactionController } from '../reaction';
import { scene, state } from './helpers';

it('uses valid schema-compatible fixtures covering all major events and failures', () => {
  expect(DEMO_STEPS).toHaveLength(18);
  for (const step of DEMO_STEPS) expect(parseScene(step.snapshot)).not.toBeNull();
  expect(new Set(DEMO_STEPS.map(s => s.snapshot.session_id)).size).toBe(2);
});
it('demo never opens a WebSocket; play/pause, step, restart and disposal work', () => {
  vi.useFakeTimers(); const socket = vi.spyOn(globalThis, 'WebSocket');
  const { source, demo } = createSource(readConfig({}, '?demo=1'));
  source.start(); vi.advanceTimersByTime(6000); expect(demo?.getControls().index).toBe(1);
  demo?.toggle(); vi.advanceTimersByTime(10000); expect(demo?.getControls().index).toBe(1);
  demo?.next(); expect(demo?.getControls().index).toBe(2);
  demo?.previous(); expect(demo?.getControls().index).toBe(1);
  demo?.restart(); expect(demo?.getControls()).toMatchObject({ index: 0, playing: true });
  source.stop(); expect(vi.getTimerCount()).toBe(0); expect(socket).not.toHaveBeenCalled();
});
it('demo events occur on transitions only and session changes on recovery', () => {
  vi.useFakeTimers(); const demo = new DemoSource(); demo.start(); demo.next();
  expect(demo.getState().latest?.events).toHaveLength(1);
  vi.advanceTimersByTime(200); expect(demo.getState().latest?.events).toHaveLength(0);
  const oldSession = demo.getState().latest?.session_id;
  for (let i = 0; i < 15; i++) demo.next();
  expect(demo.getState().latest?.session_id).not.toBe(oldSession); demo.stop();
});
it('reduced motion removes displacement, wave, expansion and bursts, with bounded rendering resources', () => {
  const c = new ReactionController(); c.updateTransport(state(scene(5))); c.ingest(scene(5), 0);
  const visual = c.step(700, 1);
  expect(motionParameters(visual, true, 1)).toEqual({ offset: 0, wave: 0, burst: 0, scale: 1 });
  expect(motionParameters(visual, false, 1).burst).toBeGreaterThan(0);
  expect(PARTICLE_LIMIT).toBeLessThanOrEqual(128); expect(pixelRatio(4)).toBe(2);
});

