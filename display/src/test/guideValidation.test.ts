import { describe, expect, it } from 'vitest';
import esp32 from '../data/esp32-oled.json';
import { parseGuide } from '../task/guideValidation';

const clone = (): Record<string, unknown> => JSON.parse(JSON.stringify(esp32)) as Record<string, unknown>;
const invalid = (value: unknown, fragment: string) => {
  const result = parseGuide(value); expect(result.ok).toBe(false);
  if (!result.ok) expect(result.issues.some(issue => `${issue.path} ${issue.message}`.includes(fragment))).toBe(true);
};

describe('guide validation', () => {
  it('parses the fixed ESP32 guide with normalized, cross-referenced anchors', () => {
    const result = parseGuide(esp32);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.steps).toHaveLength(6);
      expect(result.value.anchors).toHaveLength(8);
      expect(result.value.steps.every(step => step.verification.type === 'user_confirmation')).toBe(true);
    }
  });
  it('returns structured errors for malformed root values and unsupported versions', () => {
    invalid(null, 'plain object'); const guide = clone(); guide.version = '9.0'; invalid(guide, 'unsupported guide version');
    const missing = clone(); delete missing.title; invalid(missing, '$.title');
    const empty = clone(); empty.steps = []; invalid(empty, 'must not be empty');
  });
  it('rejects duplicate ids/orders and invalid coordinates', () => {
    const component = clone(); (component.components as unknown[]).push((component.components as unknown[])[0]); invalid(component, 'duplicate component id');
    const anchor = clone(); (anchor.anchors as Array<Record<string, unknown>>)[0].x = Number.NaN; invalid(anchor, 'finite normalized');
    const duplicateAnchor = clone(); (duplicateAnchor.anchors as Array<Record<string, unknown>>)[1].id = 'esp32-gnd'; invalid(duplicateAnchor, 'duplicate anchor id');
    const steps = clone(); (steps.steps as Array<Record<string, unknown>>)[1].id = 'prepare'; (steps.steps as Array<Record<string, unknown>>)[2].order = 1; invalid(steps, 'duplicate step');
  });
  it('rejects unknown anchor/step links, self links and prerequisite cycles', () => {
    const anchor = clone(); (((anchor.steps as Array<Record<string, unknown>>)[1].overlays as Array<Record<string, unknown>>)[0]).toAnchorId = 'missing'; invalid(anchor, 'unknown anchor');
    const dependency = clone(); (dependency.steps as Array<Record<string, unknown>>)[1].prerequisites = ['missing']; invalid(dependency, 'unknown step');
    const self = clone(); (self.steps as Array<Record<string, unknown>>)[0].prerequisites = ['prepare']; invalid(self, 'cannot depend on itself');
    const cycle = clone(); (cycle.steps as Array<Record<string, unknown>>)[0].prerequisites = ['review']; invalid(cycle, 'cyclic prerequisites');
  });
  it('rejects unsupported overlay/verification forms and unsafe values', () => {
    const overlay = clone(); (((overlay.steps as Array<Record<string, unknown>>)[1].overlays as Array<Record<string, unknown>>)[0]).type = 'pixel'; invalid(overlay, 'unsupported overlay type');
    const verification = clone(); (verification.steps as Array<Record<string, unknown>>)[0].verification = { type: 'camera_connected' }; invalid(verification, 'unsupported verification type');
    const malformed = clone(); malformed.workspace = { type: 'planar_four_corner', coordinateSystem: 'pixels' }; invalid(malformed, 'workspace');
  });
});
