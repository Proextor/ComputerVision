import { describe, expect, it, vi } from 'vitest';
import { ReactionController } from '../reaction';
import { characterBounds, expression, motionParameters, pixelRatio, PARTICLE_LIMIT } from '../renderer';
import { DemoVerificationSource, parseVerificationEvent } from '../verification';
import { createSource } from '../source';
import { readConfig } from '../config';
import { scene, state } from './helpers';

const wall = Date.parse('2026-01-01T12:00:00Z');
const event = (status: string, session = 'consented-1', id = status) => ({
  event_id: id, verification_session_id: session, status,
  timestamp: new Date(wall).toISOString(), expires_at: new Date(wall + 10000).toISOString(),
});
function ready() {
  const c = new ReactionController(); c.updateTransport(state(scene(2)));
  c.ingest(scene(2), 0);
  c.setVerificationProvider('demo'); c.armVerification(0); return c;
}

describe('sanitized verification boundary', () => {
  it('accepts only a fresh, exact result and discards unexpected payload', () => {
    expect(parseVerificationEvent({ ...event('verified'), face_embedding: [1, 2] }, wall))
      .toEqual(event('verified'));
    expect(parseVerificationEvent(event('VERIFIED'), wall)).toBeNull();
    expect(parseVerificationEvent({ ...event('verified'), timestamp: 'yesterday' }, wall)).toBeNull();
    expect(parseVerificationEvent({ ...event('verified'), expires_at: new Date(wall - 1).toISOString() }, wall)).toBeNull();
    expect(parseVerificationEvent(event('verified'), wall + 31000)).toBeNull();
  });
  it('requires a pending session; one valid success plays once and then settles', () => {
    const c = ready();
    expect(c.ingestVerification(event('verified'), 0, wall)).toBe(false);
    expect(c.ingestVerification(event('pending'), 0, wall)).toBe(true);
    expect(c.step(0, 0).transient).toBe('verification_pending');
    expect(c.ingestVerification(event('verified'), 100, wall)).toBe(true);
    expect(c.step(100, 0)).toMatchObject({ baseline: 'verified', transient: 'verified', caption: 'Verified' });
    expect(c.ingestVerification(event('verified'), 200, wall)).toBe(false);
    expect(c.step(2000, .1)).toMatchObject({ baseline: 'approaching', transient: null });
    c.armVerification(2100);
    expect(c.ingestVerification(event('pending', 'consented-2'), 2100, wall)).toBe(true);
    expect(c.ingestVerification(event('verified', 'consented-2'), 2200, wall)).toBe(true);
    expect(c.step(2200, 0).transient).toBe('verified');
  });
  it('pending, rejection, cancellation and malformed results never succeed', () => {
    const c = ready(); c.ingestVerification(event('pending'), 0, wall);
    expect(c.step(0, 0).transient).toBe('verification_pending');
    expect(c.ingestVerification(event('rejected'), 100, wall)).toBe(true);
    expect(c.step(100, 0)).toMatchObject({ baseline: 'verification_retry', transient: 'verification_retry' });
    expect(c.ingestVerification(event('verified', 'consented-1', 'late'), 150, wall)).toBe(false);
    expect(c.ingestVerification({ ...event('verified'), status: 1 }, 200, wall)).toBe(false);
    c.armVerification(300); c.ingestVerification(event('pending', 'consented-2'), 300, wall);
    c.ingestVerification(event('cancelled', 'consented-2'), 400, wall);
    expect(c.step(400, 0).transient).toBeNull();
  });
  it('requires a user arm and ignores replayed or expired success on refresh', () => {
    const c = new ReactionController(); c.updateTransport(state(scene(2))); c.ingest(scene(2), 0);
    c.setVerificationProvider('demo');
    expect(c.ingestVerification(event('pending'), 0, wall)).toBe(false);
    c.armVerification(0);
    expect(c.ingestVerification(event('pending'), 0, wall)).toBe(true);
    expect(c.ingestVerification(event('verified'), 100, wall + 10001)).toBe(false);
    const freshPage = new ReactionController(); freshPage.updateTransport(state(scene(2)));
    freshPage.ingest(scene(2), 0); freshPage.setVerificationProvider('demo');
    expect(freshPage.ingestVerification(event('verified'), 100, wall)).toBe(false);
    expect(freshPage.step(100, 0).transient).toBeNull();
  });
  it('verified outranks gestures; service and provider failures clear success', () => {
    const c = ready(); c.ingestVerification(event('pending'), 0, wall);
    c.ingestVerification(event('verified'), 100, wall);
    c.ingest(scene(5), 200);
    expect(c.step(200, 0).transient).toBe('verified');
    c.updateTransport({ ...state(scene(2)), localStale: true });
    expect(c.step(300, 0)).toMatchObject({ baseline: 'stale', transient: null });
    c.updateTransport(state(scene(2))); c.setVerificationProvider('error');
    expect(c.ingestVerification(event('verified', 'consented-3'), 400, wall)).toBe(false);
  });
  it('bounds event cache, keeps vision tracks separate and clamps suspended frames', () => {
    const c = ready();
    for (let i = 0; i < 300; i++) { c.armVerification(i); c.ingestVerification(event('pending', `s-${i}`, `e-${i}`), i, wall); }
    expect(c.verificationCacheSize).toBe(256);
    expect(Number.isFinite(c.step(1000, .1).focusX)).toBe(true);
    expect(c.step(1000, 0).verificationSessionId).toBe('s-299');
  });
});

describe('demo and layout', () => {
  it('demo success and failure are explicit and no production socket is opened', () => {
    vi.useFakeTimers(); const socket = vi.spyOn(globalThis, 'WebSocket');
    const { source, demo } = createSource(readConfig({}, '?demo=1'));
    const verifier = new DemoVerificationSource(), statuses: string[] = [];
    verifier.subscribe(value => { const parsed = parseVerificationEvent(value); if (parsed) statuses.push(parsed.status); });
    source.start(); demo?.selectStep(4); verifier.beginPending(); demo?.selectStep(5); verifier.finish('verified');
    demo?.selectStep(17); verifier.run('rejected'); vi.advanceTimersByTime(1800);
    expect(statuses).toEqual(['pending', 'verified', 'pending', 'rejected']);
    expect(socket).not.toHaveBeenCalled(); verifier.stop(); source.stop();
  });
  it('sizes from the smaller dimension, caps DPR and preserves reduced-motion thumb', () => {
    expect(characterBounds(320, 568).radius).toBeLessThan(80);
    expect(characterBounds(844, 390).radius).toBeLessThan(100);
    expect(pixelRatio(4)).toBe(2); expect(PARTICLE_LIMIT).toBeLessThanOrEqual(48);
    const c = ready(); c.ingestVerification(event('pending'), 0, wall); c.ingestVerification(event('verified'), 10, wall);
    const visual = c.step(100, 0);
    expect(motionParameters(visual, true, 0).scale).toBe(1);
    expect(motionParameters(visual, false, 0).scale).not.toBe(1);
    expect(expression(visual, true, 0)).toMatchObject({ thumb: true, label: 'Verified', wave: 0, scale: 1 });
    expect(expression({ ...visual, transient: 'verification_retry' }, false, 0)).toMatchObject({ thumb: false, label: 'Try again' });
  });
});
