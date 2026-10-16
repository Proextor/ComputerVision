/** Geometry for a video using object-fit: cover inside an overlay viewport. */
export interface Size { width: number; height: number }
export interface Point { x: number; y: number }
export interface NormalizedPoint { x: number; y: number }
export interface CoverGeometry { source: Size; viewport: Size; rendered: { x: number; y: number; width: number; height: number; scale: number } }
export function coverGeometry(source: Size, viewport: Size): CoverGeometry | null {
  if (source.width <= 0 || source.height <= 0 || viewport.width <= 0 || viewport.height <= 0) return null;
  const scale = Math.max(viewport.width / source.width, viewport.height / source.height);
  const width = source.width * scale, height = source.height * scale;
  return { source, viewport, rendered: { x: (viewport.width - width) / 2, y: (viewport.height - height) / 2, width, height, scale } };
}
export function sourceToNormalized(point: Point, source: Size): NormalizedPoint | null { return source.width > 0 && source.height > 0 ? { x: point.x / source.width, y: point.y / source.height } : null; }
export function normalizedToSource(point: NormalizedPoint, source: Size): Point { return { x: point.x * source.width, y: point.y * source.height }; }
/** Overlay coordinates share the viewport origin, including any object-fit crop. */
export function sourceToOverlay(point: Point, geometry: CoverGeometry): Point { return { x: geometry.rendered.x + point.x * geometry.rendered.scale, y: geometry.rendered.y + point.y * geometry.rendered.scale }; }
export function normalizedToOverlay(point: NormalizedPoint, geometry: CoverGeometry): Point { return sourceToOverlay(normalizedToSource(point, geometry.source), geometry); }
