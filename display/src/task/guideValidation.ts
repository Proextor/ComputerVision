import { GUIDE_VERSION, type GuideDefinition, type GuideStep, type OverlayPrimitive, type ValidationIssue, type ValidationResult, type VerificationMethod } from './guideTypes';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value) && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);
const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const unit = (value: unknown): value is number => finite(value) && value >= 0 && value <= 1;
const issue = (issues: ValidationIssue[], path: string, message: string) => issues.push({ path, message });
const stringArray = (value: unknown, path: string, issues: ValidationIssue[]): string[] => {
  if (!Array.isArray(value) || !value.every(text)) { issue(issues, path, 'must be an array of non-empty strings'); return []; }
  return value.slice();
};

function overlay(value: unknown, path: string, issues: ValidationIssue[]): OverlayPrimitive | null {
  if (!isRecord(value) || !text(value.type)) { issue(issues, path, 'must be an overlay object with a type'); return null; }
  const label = value.label === undefined ? undefined : text(value.label) ? value.label : (issue(issues, `${path}.label`, 'must be a non-empty string'), undefined);
  if (value.type === 'point') return text(value.anchorId) ? { type: 'point', anchorId: value.anchorId, ...(label ? { label } : {}) } : (issue(issues, `${path}.anchorId`, 'must reference an anchor'), null);
  if (value.type === 'label') return text(value.anchorId) && text(value.text) ? { type: 'label', anchorId: value.anchorId, text: value.text } : (issue(issues, path, 'label overlays require anchorId and text'), null);
  if (value.type === 'path' || value.type === 'arrow') {
    if (!text(value.fromAnchorId) || !text(value.toAnchorId)) { issue(issues, path, `${value.type} overlays require fromAnchorId and toAnchorId`); return null; }
    return { type: value.type, fromAnchorId: value.fromAnchorId, toAnchorId: value.toAnchorId, ...(label ? { label } : {}) };
  }
  if (value.type === 'region') {
    const anchorIds = stringArray(value.anchorIds, `${path}.anchorIds`, issues);
    if (anchorIds.length < 2) issue(issues, `${path}.anchorIds`, 'must contain at least two anchors');
    return { type: 'region', anchorIds, ...(label ? { label } : {}) };
  }
  issue(issues, `${path}.type`, `unsupported overlay type ${String(value.type)}`); return null;
}

function verification(value: unknown, path: string, issues: ValidationIssue[]): VerificationMethod | null {
  if (!isRecord(value) || !text(value.type)) { issue(issues, path, 'must be a verification object with a type'); return null; }
  if (value.type === 'user_confirmation' || value.type === 'informational_only') return { type: value.type };
  if (value.type === 'trusted_evidence') {
    const acceptedSources = stringArray(value.acceptedSources, `${path}.acceptedSources`, issues);
    const allowed = acceptedSources.filter((source): source is 'vision' | 'functional_test' | 'system' => ['vision', 'functional_test', 'system'].includes(source));
    if (!acceptedSources.length || allowed.length !== acceptedSources.length || new Set(allowed).size !== allowed.length) issue(issues, `${path}.acceptedSources`, 'must contain unique trusted sources: vision, functional_test, or system');
    return { type: 'trusted_evidence', acceptedSources: allowed };
  }
  issue(issues, `${path}.type`, `unsupported verification type ${String(value.type)}`); return null;
}

function hasCycle(steps: GuideStep[]): string | null {
  const byId = new Map(steps.map(step => [step.id, step]));
  const visiting = new Set<string>(), done = new Set<string>();
  const visit = (id: string): string | null => {
    if (visiting.has(id)) return id;
    if (done.has(id)) return null;
    visiting.add(id);
    for (const prerequisite of byId.get(id)?.prerequisites ?? []) { const found = visit(prerequisite); if (found) return found; }
    visiting.delete(id); done.add(id); return null;
  };
  for (const step of steps) { const found = visit(step.id); if (found) return found; }
  return null;
}

/** Parses untrusted JSON into a fully cross-referenced guide without throwing for invalid input. */
export function parseGuide(value: unknown): ValidationResult<GuideDefinition> {
  const issues: ValidationIssue[] = [];
  if (!isRecord(value)) return { ok: false, issues: [{ path: '$', message: 'guide must be a plain object' }] };
  const rootText = (key: 'id' | 'title' | 'description' | 'safetyNotice') => text(value[key]) ? value[key] : (issue(issues, `$.${key}`, 'must be a non-empty string'), '');
  const id = rootText('id'), title = rootText('title'), description = rootText('description'), safetyNotice = rootText('safetyNotice');
  if (value.version !== GUIDE_VERSION) issue(issues, '$.version', `unsupported guide version; expected ${GUIDE_VERSION}`);
  if (!finite(value.estimatedMinutes) || value.estimatedMinutes <= 0) issue(issues, '$.estimatedMinutes', 'must be a positive finite number');
  const components = Array.isArray(value.components) ? value.components.map((component, index) => {
    const path = `$.components[${index}]`;
    if (!isRecord(component) || !text(component.id) || !text(component.name) || !text(component.type) || !text(component.description) || !(component.notes === undefined || text(component.notes))) { issue(issues, path, 'must contain id, name, type, description and optional non-empty notes'); return null; }
    return { id: component.id, name: component.name, type: component.type, description: component.description, ...(component.notes ? { notes: component.notes } : {}) };
  }).filter((component): component is NonNullable<typeof component> => component !== null) : (issue(issues, '$.components', 'must be an array'), []);
  if (!components.length) issue(issues, '$.components', 'must not be empty');
  const componentIds = new Set<string>(); for (const component of components) { if (componentIds.has(component.id)) issue(issues, '$.components', `duplicate component id ${component.id}`); componentIds.add(component.id); }
  let workspace: GuideDefinition['workspace'] | null = null;
  if (!isRecord(value.workspace) || value.workspace.type !== 'planar_four_corner' || value.workspace.coordinateSystem !== 'normalized' || !text(value.workspace.calibrationInstructions) || !text(value.workspace.movementWarning) || !(value.workspace.aspectRatio === undefined || (finite(value.workspace.aspectRatio) && value.workspace.aspectRatio > 0))) issue(issues, '$.workspace', 'must be a planar_four_corner normalized workspace with instructions and optional positive aspectRatio');
  else workspace = { type: 'planar_four_corner', coordinateSystem: 'normalized', calibrationInstructions: value.workspace.calibrationInstructions, movementWarning: value.workspace.movementWarning, ...(value.workspace.aspectRatio === undefined ? {} : { aspectRatio: value.workspace.aspectRatio }) };
  const anchors = Array.isArray(value.anchors) ? value.anchors.map((anchor, index) => {
    const path = `$.anchors[${index}]`;
    if (!isRecord(anchor) || !text(anchor.id) || !text(anchor.componentId) || !text(anchor.label) || !unit(anchor.x) || !unit(anchor.y)) { issue(issues, path, 'must contain id, componentId, label and finite normalized x/y coordinates'); return null; }
    if (!componentIds.has(anchor.componentId)) issue(issues, `${path}.componentId`, `unknown component ${anchor.componentId}`);
    return { id: anchor.id, componentId: anchor.componentId, label: anchor.label, x: anchor.x, y: anchor.y };
  }).filter((anchor): anchor is NonNullable<typeof anchor> => anchor !== null) : (issue(issues, '$.anchors', 'must be an array'), []);
  const anchorIds = new Set<string>(); for (const anchor of anchors) { if (anchorIds.has(anchor.id)) issue(issues, '$.anchors', `duplicate anchor id ${anchor.id}`); anchorIds.add(anchor.id); }
  const steps = Array.isArray(value.steps) ? value.steps.map((raw, index) => {
    const path = `$.steps[${index}]`;
    if (!isRecord(raw)) { issue(issues, path, 'must be a step object'); return null; }
    const order = raw.order;
    if (!text(raw.id) || !Number.isSafeInteger(order) || (order as number) < 1 || !text(raw.title) || !text(raw.instruction) || !text(raw.expectedState) || !text(raw.failureRecovery) || !(raw.warning === undefined || text(raw.warning))) { issue(issues, path, 'must contain valid id, positive order, text fields and optional warning'); return null; }
    const prerequisites = stringArray(raw.prerequisites, `${path}.prerequisites`, issues);
    const overlays = Array.isArray(raw.overlays) ? raw.overlays.map((item, overlayIndex) => overlay(item, `${path}.overlays[${overlayIndex}]`, issues)).filter((item): item is OverlayPrimitive => item !== null) : (issue(issues, `${path}.overlays`, 'must be an array'), []);
    const method = verification(raw.verification, `${path}.verification`, issues);
    return method ? { id: raw.id, order: order as number, title: raw.title, instruction: raw.instruction, prerequisites, ...(raw.warning ? { warning: raw.warning } : {}), overlays, expectedState: raw.expectedState, verification: method, failureRecovery: raw.failureRecovery } : null;
  }).filter((step): step is GuideStep => step !== null) : (issue(issues, '$.steps', 'must be an array'), []);
  if (!steps.length) issue(issues, '$.steps', 'must not be empty');
  const stepIds = new Set<string>(), orders = new Set<number>();
  for (const step of steps) { if (stepIds.has(step.id)) issue(issues, '$.steps', `duplicate step id ${step.id}`); stepIds.add(step.id); if (orders.has(step.order)) issue(issues, '$.steps', `duplicate step order ${step.order}`); orders.add(step.order); }
  for (const step of steps) {
    for (const prerequisite of step.prerequisites) { if (!stepIds.has(prerequisite)) issue(issues, `$.steps[${step.id}].prerequisites`, `unknown step ${prerequisite}`); if (prerequisite === step.id) issue(issues, `$.steps[${step.id}].prerequisites`, 'a step cannot depend on itself'); }
    for (const item of step.overlays) { const references = item.type === 'path' || item.type === 'arrow' ? [item.fromAnchorId, item.toAnchorId] : item.type === 'region' ? item.anchorIds : [item.anchorId]; for (const anchorId of references) if (!anchorIds.has(anchorId)) issue(issues, `$.steps[${step.id}].overlays`, `unknown anchor ${anchorId}`); }
  }
  if (!issues.length && hasCycle(steps)) issue(issues, '$.steps', 'cyclic prerequisites are not allowed');
  if (issues.length || !workspace) return { ok: false, issues };
  return { ok: true, value: { id, version: GUIDE_VERSION, title, description, estimatedMinutes: value.estimatedMinutes as number, safetyNotice, components, workspace, anchors, steps } };
}
