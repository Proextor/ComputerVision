import { describe, expect, it } from 'vitest';
import esp32 from '../data/esp32-oled.json';
import { parseGuide } from '../task/guideValidation';
import { createTaskSession, reduceTaskSession, type TaskAction, type TaskActionInput, type TaskSession } from '../task/taskSession';
import type { EvidenceRecord } from '../task/evidence';

const parsed = parseGuide(esp32); if (!parsed.ok) throw new Error('fixture guide invalid');
const guide = parsed.value;
const stamp = '2026-10-15T10:00:00.000Z';
let number = 0;
const action = (value: TaskActionInput): TaskAction => ({ ...value, actionId: `a-${++number}`, timestamp: stamp } as TaskAction);
const apply = (session: TaskSession, value: TaskActionInput) => {
  const result = reduceTaskSession(session, guide, action(value)); expect(result.ok).toBe(true); return result.ok ? result.state : session;
};
const started = () => { number = 0; return apply(createTaskSession(guide, 'session-1', stamp), { type: 'start_session' }); };
const evidence = (patch: Partial<EvidenceRecord> = {}): EvidenceRecord => ({
  id: 'e-1', sessionId: 'session-1', guideId: guide.id, guideVersion: guide.version, stepId: 'prepare',
  source: 'functional_test', outcome: 'supported', timestamp: stamp, expiresAt: '2026-10-15T10:10:00.000Z', ...patch,
});
function trustedGuide() {
  return { ...guide, steps: guide.steps.map(step => step.id === 'prepare' ? { ...step, verification: { type: 'trusted_evidence' as const, acceptedSources: ['functional_test' as const] } } : step) };
}
const reduceTrusted = (session: TaskSession, value: TaskActionInput) => reduceTaskSession(session, trustedGuide(), action(value));

describe('task session reducer', () => {
  it('creates a deterministic initial session with prerequisites blocked', () => {
    const session = createTaskSession(guide, 'session-1', stamp);
    expect(session.status).toBe('not_started'); expect(session.stepStates.prepare).toBe('ready'); expect(session.stepStates.ground).toBe('blocked');
  });
  it('blocks unavailable starts, starts available work, and releases prerequisites after confirmation', () => {
    let session = started();
    const blocked = reduceTaskSession(session, guide, action({ type: 'start_step', stepId: 'ground' }));
    expect(blocked).toMatchObject({ ok: false, reason: 'step prerequisites are not complete' });
    session = apply(session, { type: 'start_step', stepId: 'prepare' }); expect(session.stepStates.prepare).toBe('in_progress');
    session = apply(session, { type: 'user_confirm_step', stepId: 'prepare' });
    expect(session.stepStates.prepare).toBe('user_confirmed'); expect(session.stepStates.ground).toBe('ready');
  });
  it('supports awaiting evidence, uncertainty and user confirmation without verification', () => {
    let session = started(); session = apply(session, { type: 'start_step', stepId: 'prepare' });
    session = apply(session, { type: 'request_evidence', stepId: 'prepare' }); expect(session.stepStates.prepare).toBe('awaiting_evidence');
    session = apply(session, { type: 'mark_uncertain', stepId: 'prepare', reason: 'label obscured' }); expect(session.stepStates.prepare).toBe('uncertain');
    session = apply(session, { type: 'start_step', stepId: 'prepare' }); session = apply(session, { type: 'user_confirm_step', stepId: 'prepare' });
    expect(session.stepStates.prepare).toBe('user_confirmed'); expect(session.stepStates.prepare).not.toBe('verified');
  });
  it('maps supporting user evidence to user_confirmed, never verified', () => {
    let session = started(); session = apply(session, { type: 'start_step', stepId: 'prepare' });
    const result = reduceTaskSession(session, guide, action({ type: 'submit_evidence', evidence: evidence({ source: 'user' }) }));
    expect(result.ok).toBe(true); if (result.ok) expect(result.state.stepStates.prepare).toBe('user_confirmed');
  });
  it('accepts only fresh, matching trusted evidence for verified state', () => {
    let session = createTaskSession(trustedGuide(), 'session-1', stamp); number = 0;
    let result = reduceTrusted(session, { type: 'start_session' }); expect(result.ok).toBe(true); if (result.ok) session = result.state;
    result = reduceTrusted(session, { type: 'start_step', stepId: 'prepare' }); if (result.ok) session = result.state;
    result = reduceTrusted(session, { type: 'request_evidence', stepId: 'prepare' }); if (result.ok) session = result.state;
    result = reduceTrusted(session, { type: 'submit_evidence', evidence: evidence() });
    expect(result.ok).toBe(true); if (result.ok) expect(result.state.stepStates.prepare).toBe('verified');
    const user = reduceTrusted(session, { type: 'submit_evidence', evidence: evidence({ source: 'user' }) });
    expect(user.ok).toBe(true); if (user.ok) expect(user.state.stepStates.prepare).toBe('user_confirmed');
  });
  it('rejects stale and cross-session/version evidence safely', () => {
    let session = createTaskSession(trustedGuide(), 'session-1', stamp); number = 0;
    for (const value of [{ type: 'start_session' }, { type: 'start_step', stepId: 'prepare' }, { type: 'request_evidence', stepId: 'prepare' }] as TaskActionInput[]) { const r = reduceTrusted(session, value); if (r.ok) session = r.state; }
    const stale = reduceTrusted(session, { type: 'submit_evidence', evidence: evidence({ expiresAt: stamp }) }); expect(stale.ok).toBe(true); if (stale.ok) expect(stale.state.stepStates.prepare).toBe('uncertain');
    const cross = reduceTrusted(session, { type: 'submit_evidence', evidence: evidence({ sessionId: 'other', id: 'cross' }) }); expect(cross).toMatchObject({ ok: false, reason: 'evidence does not belong to this session and guide version' });
    const version = reduceTrusted(session, { type: 'submit_evidence', evidence: evidence({ guideVersion: '2.0', id: 'version' }) }); expect(version).toMatchObject({ ok: false, reason: 'evidence does not belong to this session and guide version' });
  });
  it('invalidates verified evidence and preserves navigation state', () => {
    let session = createTaskSession(trustedGuide(), 'session-1', stamp); number = 0;
    for (const value of [{ type: 'start_session' }, { type: 'start_step', stepId: 'prepare' }, { type: 'request_evidence', stepId: 'prepare' }, { type: 'submit_evidence', evidence: evidence() }] as TaskActionInput[]) { const result = reduceTrusted(session, value); if (result.ok) session = result.state; }
    const before = session.stepStates; const nav = reduceTrusted(session, { type: 'navigate_to_step', stepId: 'ground' }); expect(nav.ok).toBe(true); if (nav.ok) { expect(nav.state.stepStates).toEqual(before); session = nav.state; }
    const invalidated = reduceTrusted(session, { type: 'invalidate_evidence', evidenceId: 'e-1', reason: 'expired' });
    expect(invalidated.ok).toBe(true); if (invalidated.ok) expect(invalidated.state.stepStates.prepare).toBe('needs_recheck');
  });
  it('is immutable, ordered and replayable with supplied identities and timestamps', () => {
    const initial = createTaskSession(guide, 'session-1', stamp), original = JSON.stringify(initial);
    const actions: TaskAction[] = [
      { type: 'start_session', actionId: '1', timestamp: stamp }, { type: 'start_step', stepId: 'prepare', actionId: '2', timestamp: stamp },
      { type: 'user_confirm_step', stepId: 'prepare', actionId: '3', timestamp: stamp }, { type: 'restart_session', sessionId: 'session-2', actionId: '4', timestamp: stamp },
    ];
    const replay = () => actions.reduce((state, next) => { const result = reduceTaskSession(state, guide, next); if (!result.ok) throw new Error(result.reason); return result.state; }, initial);
    const one = replay(), two = replay(); expect(JSON.stringify(initial)).toBe(original); expect(one).toEqual(two);
    expect(one.actionLog.map(entry => entry.actionId)).toEqual(['4']); expect(one.sessionId).toBe('session-2');
  });
});
