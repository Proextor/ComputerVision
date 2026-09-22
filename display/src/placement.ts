import type { Box } from './contracts';
export interface Point2 { x: number; y: number }
export interface Viewport { width: number; height: number; top: number; right: number; bottom: number; left: number }
export function coverPoint(point: Point2, sourceWidth: number, sourceHeight: number, width: number, height: number, mirrored: boolean): Point2 {
  const scale = Math.max(width / sourceWidth, height / sourceHeight);
  const offsetX = (width - sourceWidth * scale) / 2, offsetY = (height - sourceHeight * scale) / 2;
  const x = point.x * sourceWidth * scale + offsetX;
  return { x: mirrored ? width - x : x, y: point.y * sourceHeight * scale + offsetY };
}
export function overlayTarget(box: Box | null, sourceWidth: number, sourceHeight: number, view: Viewport, mirrored: boolean, unitWidth: number): Point2 {
  const fallback = { x: view.width / 2, y: Math.max(view.top + 70, view.height * .25) };
  let target = fallback;
  if (box && sourceWidth > 0 && sourceHeight > 0) {
    const top = coverPoint({ x: (box.x1 + box.x2) / 2, y: box.y1 }, sourceWidth, sourceHeight, view.width, view.height, mirrored);
    const headVisible = top.x >= 0 && top.x <= view.width && top.y >= 0 && top.y < view.height;
    if (headVisible) {
      // Keep the unit above the head. When there is no room above, place beside the box.
      const above = top.y - 60;
      if (above >= view.top + 50) target = { x: top.x, y: above };
      else {
        const towardLeft = top.x > view.width / 2;
        const side = towardLeft !== mirrored ? box.x1 : box.x2;
        const edge = coverPoint({ x: side, y: box.y1 }, sourceWidth, sourceHeight, view.width, view.height, mirrored);
        const sign = edge.x > top.x ? 1 : -1;
        target = { x: edge.x + sign * (unitWidth / 2 + 16), y: Math.max(view.top + 58, top.y) };
      }
    }
  }
  const half = unitWidth / 2 + 12;
  return { x: Math.max(view.left + half, Math.min(view.width - view.right - half, target.x)),
    y: Math.max(view.top + 56, Math.min(view.height - view.bottom - 72, target.y)) };
}
export class AnchorSmoother {
  private current: Point2 | null = null;
  reset() { this.current = null; }
  step(target: Point2, seconds: number): Point2 {
    if (!this.current) this.current = { ...target };
    const dt = Math.min(.1, Math.max(0, Number.isFinite(seconds) ? seconds : 0));
    const dx = target.x - this.current.x, dy = target.y - this.current.y;
    const distance = Math.hypot(dx, dy);
    const blend = Math.min(1 - Math.exp(-dt / .25), distance ? 260 * dt / distance : 1);
    this.current.x += dx * blend; this.current.y += dy * blend;
    return this.current;
  }
}
