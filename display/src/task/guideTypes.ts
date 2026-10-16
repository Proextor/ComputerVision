export const GUIDE_VERSION = '1.0' as const;
export type GuideVersion = typeof GUIDE_VERSION;

export interface GuideComponent {
  id: string; name: string; type: string; description: string; notes?: string;
}
export interface PlanarWorkspace {
  type: 'planar_four_corner'; coordinateSystem: 'normalized'; aspectRatio?: number;
  calibrationInstructions: string; movementWarning: string;
}
export interface GuideAnchor { id: string; componentId: string; label: string; x: number; y: number }

export type OverlayPrimitive =
  | { type: 'point'; anchorId: string; label?: string }
  | { type: 'label'; anchorId: string; text: string }
  | { type: 'path'; fromAnchorId: string; toAnchorId: string; label?: string }
  | { type: 'arrow'; fromAnchorId: string; toAnchorId: string; label?: string }
  | { type: 'region'; anchorIds: string[]; label?: string };

export type VerificationMethod =
  | { type: 'user_confirmation' }
  | { type: 'trusted_evidence'; acceptedSources: Array<'vision' | 'functional_test' | 'system'> }
  | { type: 'informational_only' };

export interface GuideStep {
  id: string; order: number; title: string; instruction: string; prerequisites: string[];
  warning?: string; overlays: OverlayPrimitive[]; expectedState: string;
  verification: VerificationMethod; failureRecovery: string;
}
export interface GuideDefinition {
  id: string; version: GuideVersion; title: string; description: string; estimatedMinutes: number;
  safetyNotice: string; components: GuideComponent[]; workspace: PlanarWorkspace;
  anchors: GuideAnchor[]; steps: GuideStep[];
}

export interface ValidationIssue { path: string; message: string }
export type ValidationResult<T> = { ok: true; value: T } | { ok: false; issues: ValidationIssue[] };
