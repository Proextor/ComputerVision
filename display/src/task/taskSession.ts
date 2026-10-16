import { evidenceIsFresh, validateEvidence, type EvidenceRecord } from './evidence';
import type { GuideDefinition, GuideStep } from './guideTypes';

export type TaskStepState = 'ready' | 'in_progress' | 'awaiting_evidence' | 'user_confirmed' | 'verified' | 'uncertain' | 'needs_recheck' | 'blocked';
export type SessionStatus = 'not_started' | 'active' | 'review_required' | 'completed' | 'blocked';
export interface TaskActionLog { actionId: string; timestamp: string; type: TaskAction['type']; stepId?: string; evidenceId?: string; reason?: string }
export interface TaskSession {
  sessionId: string; guideId: string; guideVersion: string; createdAt: string; updatedAt: string;
  activeStepId: string | null; stepStates: Record<string, TaskStepState>;
  actionLog: TaskActionLog[]; evidenceRecords: EvidenceRecord[]; status: SessionStatus;
}
interface ActionBase { actionId: string; timestamp: string }
export type TaskAction =
  | (ActionBase & { type: 'start_session' })
  | (ActionBase & { type: 'start_step'; stepId: string })
  | (ActionBase & { type: 'request_evidence'; stepId: string })
  | (ActionBase & { type: 'user_confirm_step'; stepId: string })
  | (ActionBase & { type: 'mark_uncertain'; stepId: string; reason?: string })
  | (ActionBase & { type: 'submit_evidence'; evidence: unknown })
  | (ActionBase & { type: 'invalidate_evidence'; evidenceId: string; reason?: string })
  | (ActionBase & { type: 'navigate_to_step'; stepId: string })
  | (ActionBase & { type: 'reset_step'; stepId: string })
  | (ActionBase & { type: 'restart_session'; sessionId: string });
type WithoutActionMeta<T> = T extends unknown ? Omit<T, 'actionId' | 'timestamp'> : never;
export type TaskActionInput = WithoutActionMeta<TaskAction>;
export type TransitionResult = { ok: true; state: TaskSession } | { ok: false; state: TaskSession; reason: string };

const complete = (state: TaskStepState) => state === 'user_confirmed' || state === 'verified';
const stepById = (guide: GuideDefinition, id: string): GuideStep | undefined => guide.steps.find(step => step.id === id);
const available = (states: Record<string, TaskStepState>, step: GuideStep) => step.prerequisites.every(id => complete(states[id]));
const actionLog = (action: TaskAction): TaskActionLog => ({ actionId: action.actionId, timestamp: action.timestamp, type: action.type,
  ...('stepId' in action ? { stepId: action.stepId } : {}),
  ...('evidenceId' in action ? { evidenceId: action.evidenceId } : action.type === 'submit_evidence' && typeof action.evidence === 'object' && action.evidence !== null && 'id' in action.evidence && typeof action.evidence.id === 'string' ? { evidenceId: action.evidence.id } : {}),
  ...('reason' in action && action.reason ? { reason: action.reason } : {}) });

function normalizedStates(guide: GuideDefinition, states: Record<string, TaskStepState>): Record<string, TaskStepState> {
  const next = { ...states };
  // Repeat so guide declaration order cannot affect prerequisite release.
  let changed = true;
  while (changed) {
    changed = false;
    for (const step of guide.steps) {
      if (next[step.id] === 'blocked' && available(next, step)) { next[step.id] = 'ready'; changed = true; }
      if (!available(next, step) && ['ready', 'in_progress', 'awaiting_evidence'].includes(next[step.id])) { next[step.id] = 'blocked'; changed = true; }
    }
  }
  return next;
}
function statusFor(guide: GuideDefinition, states: Record<string, TaskStepState>, started: boolean): SessionStatus {
  if (!started) return 'not_started';
  if (guide.steps.every(step => complete(states[step.id]))) return 'completed';
  if (guide.steps.some(step => states[step.id] === 'needs_recheck' || states[step.id] === 'uncertain')) return 'review_required';
  // A blocked step whose prerequisites are already complete was blocked by contradictory evidence.
  if (guide.steps.some(step => states[step.id] === 'blocked' && available(states, step))) return 'blocked';
  return 'active';
}
function append(session: TaskSession, guide: GuideDefinition, action: TaskAction, patch: Partial<TaskSession> = {}): TaskSession {
  const states = normalizedStates(guide, patch.stepStates ?? session.stepStates);
  const started = action.type !== 'start_session' ? session.status !== 'not_started' : true;
  return { ...session, ...patch, stepStates: states, updatedAt: action.timestamp,
    actionLog: [...session.actionLog, actionLog(action)], status: statusFor(guide, states, started) };
}
function reject(state: TaskSession, reason: string): TransitionResult { return { ok: false, state, reason }; }

/** Creates a deterministic, inactive session. IDs and timestamps are always caller supplied. */
export function createTaskSession(guide: GuideDefinition, sessionId: string, timestamp: string): TaskSession {
  const stepStates: Record<string, TaskStepState> = {};
  for (const step of guide.steps) stepStates[step.id] = step.prerequisites.length ? 'blocked' : 'ready';
  return { sessionId, guideId: guide.id, guideVersion: guide.version, createdAt: timestamp, updatedAt: timestamp,
    activeStepId: null, stepStates, actionLog: [], evidenceRecords: [], status: 'not_started' };
}

/** Pure reducer. Contradicting accepted evidence needs recheck when recovery instructions exist, otherwise it blocks. */
export function reduceTaskSession(session: TaskSession, guide: GuideDefinition, action: TaskAction): TransitionResult {
  const target = (id: string) => stepById(guide, id);
  const requireKnown = (id: string) => target(id) ? null : `unknown step ${id}`;
  if (session.guideId !== guide.id || session.guideVersion !== guide.version) return reject(session, 'session guide does not match reducer guide');
  if (action.type === 'restart_session') {
    const restarted = createTaskSession(guide, action.sessionId, action.timestamp);
    return { ok: true, state: { ...restarted, actionLog: [actionLog(action)] } };
  }
  if (action.type === 'start_session') {
    if (session.status !== 'not_started') return reject(session, 'session has already started');
    return { ok: true, state: append(session, guide, action) };
  }
  if (session.status === 'not_started') return reject(session, 'start_session is required first');
  if (action.type === 'navigate_to_step') {
    const error = requireKnown(action.stepId); if (error) return reject(session, error);
    return { ok: true, state: append(session, guide, action, { activeStepId: action.stepId }) };
  }
  if (action.type === 'start_step') {
    const step = target(action.stepId); if (!step) return reject(session, `unknown step ${action.stepId}`);
    if (!available(session.stepStates, step) || session.stepStates[step.id] === 'blocked') return reject(session, 'step prerequisites are not complete');
    if (!['ready', 'uncertain', 'needs_recheck'].includes(session.stepStates[step.id])) return reject(session, `cannot start a ${session.stepStates[step.id]} step`);
    return { ok: true, state: append(session, guide, action, { activeStepId: step.id, stepStates: { ...session.stepStates, [step.id]: 'in_progress' } }) };
  }
  if (action.type === 'request_evidence') {
    const step = target(action.stepId); if (!step) return reject(session, `unknown step ${action.stepId}`);
    if (session.stepStates[step.id] !== 'in_progress') return reject(session, 'evidence can only be requested from an in_progress step');
    return { ok: true, state: append(session, guide, action, { stepStates: { ...session.stepStates, [step.id]: 'awaiting_evidence' } }) };
  }
  if (action.type === 'user_confirm_step') {
    const step = target(action.stepId); if (!step) return reject(session, `unknown step ${action.stepId}`);
    if (!['in_progress', 'awaiting_evidence'].includes(session.stepStates[step.id])) return reject(session, 'user confirmation requires an active step');
    if (step.verification.type === 'informational_only') return reject(session, 'informational steps cannot be confirmed as completed');
    return { ok: true, state: append(session, guide, action, { stepStates: { ...session.stepStates, [step.id]: 'user_confirmed' } }) };
  }
  if (action.type === 'mark_uncertain') {
    const error = requireKnown(action.stepId); if (error) return reject(session, error);
    return { ok: true, state: append(session, guide, action, { stepStates: { ...session.stepStates, [action.stepId]: 'uncertain' } }) };
  }
  if (action.type === 'reset_step') {
    const step = target(action.stepId); if (!step) return reject(session, `unknown step ${action.stepId}`);
    const next = { ...session.stepStates, [step.id]: available(session.stepStates, step) ? 'ready' as const : 'blocked' as const };
    return { ok: true, state: append(session, guide, action, { activeStepId: session.activeStepId === step.id ? null : session.activeStepId, stepStates: next }) };
  }
  if (action.type === 'invalidate_evidence') {
    const evidence = session.evidenceRecords.find(record => record.id === action.evidenceId); if (!evidence) return reject(session, `unknown evidence ${action.evidenceId}`);
    const evidenceRecords = session.evidenceRecords.map(record => record.id === evidence.id ? { ...record, outcome: 'stale' as const, ...(action.reason ? { reason: action.reason } : {}) } : record);
    const state = session.stepStates[evidence.stepId] === 'verified' ? 'needs_recheck' as const : session.stepStates[evidence.stepId];
    return { ok: true, state: append(session, guide, action, { evidenceRecords, stepStates: { ...session.stepStates, [evidence.stepId]: state } }) };
  }
  if (action.type === 'submit_evidence') {
    const parsed = validateEvidence(action.evidence); if (!parsed.ok) return reject(session, `invalid evidence: ${parsed.issues.map(item => item.path).join(', ')}`);
    const evidence = parsed.value, step = target(evidence.stepId);
    if (!step) return reject(session, `unknown step ${evidence.stepId}`);
    if (session.evidenceRecords.some(record => record.id === evidence.id)) return reject(session, `duplicate evidence ${evidence.id}`);
    if (evidence.sessionId !== session.sessionId || evidence.guideId !== guide.id || evidence.guideVersion !== guide.version) return reject(session, 'evidence does not belong to this session and guide version');
    if (!['in_progress', 'awaiting_evidence'].includes(session.stepStates[step.id])) return reject(session, 'evidence requires an active step');
    const evidenceRecords = [...session.evidenceRecords, evidence];
    let next: TaskStepState;
    if (!evidenceIsFresh(evidence, action.timestamp) || evidence.outcome === 'stale' || evidence.outcome === 'insufficient') next = 'uncertain';
    else if (evidence.outcome === 'contradicted') next = step.failureRecovery ? 'needs_recheck' : 'blocked';
    else if (evidence.source === 'user') next = 'user_confirmed';
    else if (step.verification.type === 'trusted_evidence' && step.verification.acceptedSources.includes(evidence.source)) next = 'verified';
    else return reject(session, 'evidence source is not trusted for this step');
    return { ok: true, state: append(session, guide, action, { evidenceRecords, stepStates: { ...session.stepStates, [step.id]: next } }) };
  }
  return reject(session, 'unsupported action');
}
