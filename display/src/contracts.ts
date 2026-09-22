/** Only the schema 4.0 fields consumed by this display. Unknown fields are discarded. */
export interface Point { x: number; y: number }
export interface Box { x1: number; y1: number; x2: number; y2: number }
export interface Person { track_id: number | null; anchor_point: Point | null; bbox?: Box | null }
export interface Focus { track_id: number; score: number; evidence: string[]; is_visible: boolean }
export interface SceneEvent {
  event_id: number; session_id: string; timestamp_utc: string; type: string;
  track_id: number | null; metadata: Record<string, string | number | boolean | null>;
}
export interface Scene {
  schema_version: '4.0'; session_id: string; frame_id: number; timestamp_utc: string;
  camera_status: string; visible_count: number; people: Person[]; events: SceneEvent[];
  frame_width?: number | null; frame_height?: number | null;
  room_state: {
    occupancy: { state: string; visible_count: number; stabilized_count: number | null };
    spatial: { centroid: Point | null };
    activity: { aggregate_activity_score: number | null; aggregate_activity_level: string };
    interaction: { state: string }; focus_candidate: Focus | null;
  };
}
export interface SnapshotEnvelope {
  type: 'snapshot'; publication_sequence: number; published_at: string; snapshot: Scene;
}
export interface HeartbeatEnvelope {
  type: 'heartbeat'; publication_sequence: number; service_status: string;
  last_snapshot_age_seconds: number | null;
}
export type Envelope = SnapshotEnvelope | HeartbeatEnvelope;

export const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === 'string' && v.length > 0;
const count = (v: unknown): v is number => Number.isSafeInteger(v) && typeof v === 'number' && v >= 0;
const unit = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
const date = (v: unknown): v is string => text(v) && /(?:Z|[+-]\d\d:\d\d)$/.test(v) && Number.isFinite(Date.parse(v));
const age = (v: unknown): v is number | null => v === null || (typeof v === 'number' && Number.isFinite(v) && v >= 0);
function point(v: unknown): Point | null | undefined {
  if (v === null) return null;
  return isRecord(v) && unit(v.x) && unit(v.y) ? { x: v.x, y: v.y } : undefined;
}

export function parseScene(v: unknown): Scene | null {
  if (!isRecord(v) || v.schema_version !== '4.0' || !text(v.session_id) || !count(v.frame_id) ||
      !date(v.timestamp_utc) || !text(v.camera_status) || !count(v.visible_count) ||
      !Array.isArray(v.people) || !Array.isArray(v.events) || !isRecord(v.room_state)) return null;
  const people: Person[] = [];
  for (const person of v.people) {
    if (!isRecord(person) || !(person.track_id === null || count(person.track_id))) return null;
    const anchor = point(person.anchor_point);
    if (anchor === undefined) return null;
    let bbox: Box | null | undefined;
    if (person.bbox === null) bbox = null;
    else if (person.bbox !== undefined) {
      const b = person.bbox;
      if (!isRecord(b) || !unit(b.x1) || !unit(b.x2) || !unit(b.y1) || !unit(b.y2) || b.x1 > b.x2 || b.y1 > b.y2) return null;
      bbox = { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 };
    }
    people.push({ track_id: person.track_id, anchor_point: anchor, ...(bbox !== undefined ? { bbox } : {}) });
  }
  if (people.length !== v.visible_count) return null;
  const room = v.room_state;
  const { occupancy, spatial, activity, interaction } = room;
  if (!isRecord(occupancy) || !text(occupancy.state) || !count(occupancy.visible_count) ||
      occupancy.visible_count !== v.visible_count || !(occupancy.stabilized_count === null || count(occupancy.stabilized_count)) ||
      !isRecord(spatial) || !isRecord(activity) || !isRecord(interaction) || !text(interaction.state) ||
      !(activity.aggregate_activity_score === null || unit(activity.aggregate_activity_score)) ||
      !text(activity.aggregate_activity_level)) return null;
  const centroid = point(spatial.centroid);
  if (centroid === undefined) return null;
  let focus: Focus | null = null;
  if (room.focus_candidate !== null) {
    const f = room.focus_candidate;
    if (!isRecord(f) || !count(f.track_id) || !unit(f.score) || typeof f.is_visible !== 'boolean' ||
        !Array.isArray(f.evidence) || !f.evidence.every(text)) return null;
    focus = { track_id: f.track_id, score: f.score, is_visible: f.is_visible, evidence: f.evidence };
  }
  const events: SceneEvent[] = [];
  for (const event of v.events) {
    if (!isRecord(event) || !count(event.event_id) || event.event_id < 1 || !text(event.type) ||
        event.session_id !== v.session_id || !date(event.timestamp_utc) ||
        !(event.track_id === null || count(event.track_id)) || !isRecord(event.metadata)) return null;
    const metadata: SceneEvent['metadata'] = {};
    for (const [key, value] of Object.entries(event.metadata)) {
      if (value === null || typeof value === 'string' || typeof value === 'boolean' ||
          (typeof value === 'number' && Number.isFinite(value))) metadata[key] = value;
    }
    events.push({ event_id: event.event_id, type: event.type, session_id: v.session_id,
      timestamp_utc: event.timestamp_utc, track_id: event.track_id, metadata });
  }
  if ([v.frame_width, v.frame_height].some(n => n !== undefined && n !== null && (!count(n) || n === 0))) return null;
  return { schema_version: '4.0', session_id: v.session_id, frame_id: v.frame_id,
    ...(v.frame_width !== undefined ? { frame_width: v.frame_width as number | null } : {}),
    ...(v.frame_height !== undefined ? { frame_height: v.frame_height as number | null } : {}),
    timestamp_utc: v.timestamp_utc, camera_status: v.camera_status, visible_count: v.visible_count, people, events,
    room_state: {
      occupancy: { state: occupancy.state, visible_count: occupancy.visible_count, stabilized_count: occupancy.stabilized_count },
      spatial: { centroid }, activity: { aggregate_activity_score: activity.aggregate_activity_score,
        aggregate_activity_level: activity.aggregate_activity_level },
      interaction: { state: interaction.state }, focus_candidate: focus,
    } };
}

export function parseEnvelope(value: unknown): Envelope | null {
  if (!isRecord(value) || !count(value.publication_sequence)) return null;
  if (value.type === 'heartbeat' && text(value.service_status) && age(value.last_snapshot_age_seconds)) {
    return { type: 'heartbeat', publication_sequence: value.publication_sequence,
      service_status: value.service_status, last_snapshot_age_seconds: value.last_snapshot_age_seconds };
  }
  if (value.type === 'snapshot' && value.publication_sequence > 0 && date(value.published_at)) {
    const snapshot = parseScene(value.snapshot);
    if (snapshot) return { type: 'snapshot', publication_sequence: value.publication_sequence,
      published_at: value.published_at, snapshot };
  }
  return null;
}

export function parseMessage(data: unknown): Envelope | null {
  if (typeof data !== 'string') return null;
  try { const value: unknown = JSON.parse(data); return parseEnvelope(value); }
  catch { return null; }
}
