import type { ValidationIssue, ValidationResult } from './guideTypes';

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type EvidenceSource = 'user' | 'vision' | 'functional_test' | 'system';
export type EvidenceOutcome = 'supported' | 'contradicted' | 'insufficient' | 'stale';
export interface EvidenceRecord {
  id: string; sessionId: string; guideId: string; guideVersion: string; stepId: string;
  source: EvidenceSource; outcome: EvidenceOutcome; timestamp: string;
  confidence?: number; reason?: string; expiresAt?: string; metadata?: Record<string, JsonValue>;
}
const sources = new Set<EvidenceSource>(['user', 'vision', 'functional_test', 'system']);
const outcomes = new Set<EvidenceOutcome>(['supported', 'contradicted', 'insufficient', 'stale']);
const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
const time = (value: unknown): value is string => text(value) && Number.isFinite(Date.parse(value));
function json(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(json);
  return typeof value === 'object' && value !== null && Object.getPrototypeOf(value) === Object.prototype && Object.values(value).every(json);
}
export function validateEvidence(value: unknown): ValidationResult<EvidenceRecord> {
  const issues: ValidationIssue[] = [];
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return { ok: false, issues: [{ path: '$', message: 'evidence must be an object' }] };
  const record = value as Record<string, unknown>;
  for (const key of ['id', 'sessionId', 'guideId', 'guideVersion', 'stepId', 'timestamp'] as const) if (!text(record[key])) issues.push({ path: `$.${key}`, message: 'must be a non-empty string' });
  if (!time(record.timestamp)) issues.push({ path: '$.timestamp', message: 'must be a valid timestamp' });
  if (!sources.has(record.source as EvidenceSource)) issues.push({ path: '$.source', message: 'unsupported evidence source' });
  if (!outcomes.has(record.outcome as EvidenceOutcome)) issues.push({ path: '$.outcome', message: 'unsupported evidence outcome' });
  if (record.confidence !== undefined && (!(typeof record.confidence === 'number') || !Number.isFinite(record.confidence) || record.confidence < 0 || record.confidence > 1)) issues.push({ path: '$.confidence', message: 'must be a finite value between 0 and 1' });
  if (record.reason !== undefined && !text(record.reason)) issues.push({ path: '$.reason', message: 'must be a non-empty string' });
  if (record.expiresAt !== undefined && (!time(record.expiresAt) || Date.parse(record.expiresAt) < Date.parse(record.timestamp as string))) issues.push({ path: '$.expiresAt', message: 'must be a timestamp at or after timestamp' });
  if (record.metadata !== undefined && (!json(record.metadata) || Array.isArray(record.metadata))) issues.push({ path: '$.metadata', message: 'must contain JSON-compatible object values' });
  if (issues.length) return { ok: false, issues };
  return { ok: true, value: {
    id: record.id as string, sessionId: record.sessionId as string, guideId: record.guideId as string,
    guideVersion: record.guideVersion as string, stepId: record.stepId as string,
    source: record.source as EvidenceSource, outcome: record.outcome as EvidenceOutcome, timestamp: record.timestamp as string,
    ...(record.confidence === undefined ? {} : { confidence: record.confidence as number }),
    ...(record.reason === undefined ? {} : { reason: record.reason as string }),
    ...(record.expiresAt === undefined ? {} : { expiresAt: record.expiresAt as string }),
    ...(record.metadata === undefined ? {} : { metadata: record.metadata as Record<string, JsonValue> }),
  } };
}
export const evidenceIsFresh = (evidence: EvidenceRecord, at: string): boolean =>
  evidence.outcome !== 'stale' && (!evidence.expiresAt || Date.parse(evidence.expiresAt) > Date.parse(at));
