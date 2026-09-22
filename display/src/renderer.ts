import type { Scene } from './contracts';
import { AnchorSmoother, overlayTarget, type Viewport } from './placement';
import { isFailure, type ReactionController, type VisualState } from './reaction';

export const PARTICLE_LIMIT = 8;
export const pixelRatio = (ratio: number): number => Math.max(1, Math.min(2, ratio || 1));
export function characterBounds(width: number, height: number) { return { x: width / 2, y: height * .25, radius: Math.min(width, height) * .15 }; }
export function motionParameters(state: VisualState, reduced: boolean, time: number) {
  const pulse = Math.sin(Math.PI * state.progress);
  return { offset: reduced ? 0 : state.focusX,
    wave: !reduced && state.transient === 'wave_back' ? Math.sin(state.progress * Math.PI * 5) * pulse * .07 : 0,
    burst: !reduced && (state.transient === 'celebrate' || state.transient === 'verified') ? pulse : 0,
    scale: reduced ? 1 : 1 + Math.sin(time * .65) * .006 + (state.transient === 'verified' ? pulse * .025 : 0) };
}
export interface OverlayInput { scene: Scene | null; reliable: boolean; mirrored: boolean }
export function expression(state: VisualState, reduced: boolean, time: number) {
  const verified = state.transient === 'verified';
  const pulse = Math.sin(state.progress * Math.PI);
  const blink = !reduced && Math.sin(time * 1.4) > .998 ? .08 : 1;
  const aperture = verified && !reduced && state.progress < .18 ? Math.max(.08, 1 - Math.sin(state.progress / .18 * Math.PI)) :
    state.transient === 'verification_pending' ? .66 : state.transient === 'verification_retry' ? .55 : blink * (state.baseline === 'approaching' ? 1.07 : 1);
  return { aperture, thumb: verified, hand: state.transient === 'wave_back' || state.transient === 'acknowledge',
    wave: reduced ? 0 : state.transient === 'wave_back' ? Math.sin(state.progress * Math.PI * 5) * 17 * pulse : 0,
    ring: verified ? reduced ? .25 : (1 - state.progress) * .65 : 0,
    scale: reduced ? 1 : state.transient === 'wake' ? .9 + .1 * Math.min(1, state.progress * 5) : 1,
    label: verified ? 'Verified' : state.transient === 'verification_pending' ? 'Verifying' : state.transient === 'verification_retry' ? 'Try again' : state.transient === 'wake' ? 'Hello' : '',
  };
}

export class CharacterAnimator {
  private frame = 0;
  private previous = 0;
  private time = 0;
  private fps = 0;
  private disposed = false;
  private smoother = new AnchorSmoother();
  private view: Viewport = { width: 320, height: 568, top: 64, right: 12, bottom: 70, left: 12 };
  private unitWidth = 180;
  private nodes: Record<string, SVGElement | HTMLElement> = {};
  private observer: ResizeObserver;
  constructor(private root: HTMLElement, private controller: ReactionController, private input: () => OverlayInput, private reduced: () => boolean) {
    for (const name of ['eyes','ring','left-eye','right-eye','left-pupil','right-pupil','thumb','wave','trace','scan','scan-dot','burst','pill']) {
      this.nodes[name] = root.querySelector(`[data-${name}]`)!;
    }
    this.nodes.leftAperture = root.querySelector('[data-aperture="left"]')!;
    this.nodes.rightAperture = root.querySelector('[data-aperture="right"]')!;
    this.observer = new ResizeObserver(this.resize); this.observer.observe(root.parentElement!);
    window.addEventListener('resize', this.resize); window.addEventListener('orientationchange', this.resize);
    document.addEventListener('visibilitychange', this.visibility);
    this.resize(); this.visibility();
  }
  getFPS() { return document.hidden ? 0 : this.fps; }
  private resize = () => {
    const rect = this.root.parentElement!.getBoundingClientRect();
    const style = getComputedStyle(this.root.parentElement!);
    this.view = { width: rect.width || innerWidth, height: rect.height || innerHeight,
      top: 60 + (parseFloat(style.paddingTop) || 0), right: 10 + (parseFloat(style.paddingRight) || 0),
      bottom: 64 + (parseFloat(style.paddingBottom) || 0), left: 10 + (parseFloat(style.paddingLeft) || 0) };
    this.unitWidth = Math.min(210, Math.max(150, Math.min(this.view.width, this.view.height) * .46));
    this.root.style.width = `${this.unitWidth}px`; this.smoother.reset();
  };
  private visibility = () => { cancelAnimationFrame(this.frame); this.previous = 0; if (!document.hidden && !this.disposed) this.frame = requestAnimationFrame(this.draw); };
  private draw = (now: number) => {
    if (this.disposed || document.hidden) return;
    const dt = this.previous ? Math.min(.1, Math.max(0, (now - this.previous) / 1000)) : 0; this.previous = now;
    if (dt > 0) this.fps += (1 / dt - this.fps) * (1 - Math.exp(-dt));
    this.time += dt;
    const state = this.controller.step(now, dt), input = this.input(), reduced = this.reduced();
    const scene = input.scene;
    const focus = input.reliable && !isFailure(state.baseline) && scene?.room_state.focus_candidate?.is_visible ? scene.room_state.focus_candidate : null;
    const box = focus ? scene?.people.find(p => p.track_id === focus.track_id)?.bbox ?? null : null;
    const target = overlayTarget(box, scene?.frame_width ?? 0, scene?.frame_height ?? 0, this.view, input.mirrored, this.unitWidth);
    const pos = this.smoother.step(target, dt);
    this.root.style.transform = `translate3d(${pos.x - this.unitWidth / 2}px,${pos.y - this.unitWidth * .28}px,0)`;
    const p = expression(state, reduced, this.time);
    const dormant = state.baseline === 'dormant';
    const leaving = state.transient === 'leaving';
    this.root.style.opacity = dormant ? '.15' : isFailure(state.baseline) ? '.32' : '1';
    this.nodes.eyes.style.opacity = dormant ? '0' : leaving ? String(1 - state.progress) : '1';
    this.nodes.eyes.setAttribute('transform', `translate(${110 * (1 - p.scale)} ${68 * (1 - p.scale)}) scale(${p.scale})`);
    for (const key of ['leftAperture', 'rightAperture']) { this.nodes[key].setAttribute('height', String(49 * p.aperture)); this.nodes[key].setAttribute('y', String(67.5 - 24.5 * p.aperture)); }
    const pupilOffset = reduced ? 0 : state.transient === 'verification_pending' ? 0 : state.focusX * 25 * (input.mirrored ? -1 : 1);
    this.nodes['left-pupil'].setAttribute('transform', `translate(${pupilOffset} 0)`);
    this.nodes['right-pupil'].setAttribute('transform', `translate(${pupilOffset} 0)`);
    for (const key of ['left-pupil', 'right-pupil']) {
      this.nodes[key].style.filter = p.thumb ? 'brightness(1.3)' : state.baseline === 'approaching' ? 'sepia(.18) brightness(1.08)' : '';
    }
    this.nodes['left-eye'].setAttribute('transform', state.transient === 'verification_retry' ? 'rotate(-5 73 68)' : '');
    this.nodes['right-eye'].setAttribute('transform', state.transient === 'verification_retry' ? 'rotate(-5 147 68)' : '');
    this.nodes.thumb.style.opacity = p.thumb ? '1' : '0';
    const enter = reduced ? 0 : p.thumb ? Math.max(0, 1 - state.progress * 5) * 18 : 0;
    this.nodes.thumb.setAttribute('transform', `translate(${enter} ${enter * .4})`);
    this.nodes.wave.style.opacity = p.hand ? '1' : '0';
    this.nodes.wave.setAttribute('transform', `rotate(${p.wave} 192 126)`);
    this.nodes.ring.style.opacity = String(p.ring);
    const ringScale = reduced ? 1 : 1 + state.progress * .23;
    this.nodes.ring.setAttribute('transform', `translate(${111 * (1 - ringScale)} ${68 * (1 - ringScale)}) scale(${ringScale})`);
    this.nodes.trace.style.opacity = dormant || leaving || state.transient === 'wake' ? '1' : '0';
    this.nodes.trace.style.strokeDasharray = leaving ? `${1 - state.progress} 1` : state.transient === 'wake' ? `${Math.min(1, state.progress * 5)} 1` : '1 0';
    this.nodes.scan.style.opacity = state.transient === 'verification_pending' ? '1' : '0';
    this.nodes['scan-dot'].setAttribute('cx', String(reduced ? 111 : 40 + (Math.sin(this.time * 2) + 1) * 70));
    this.nodes.burst.style.opacity = state.transient === 'celebrate' ? String(reduced ? .5 : Math.sin(state.progress * Math.PI)) : '0';
    this.nodes.pill.textContent = p.label; this.nodes.pill.style.opacity = p.label ? '1' : '0';
    this.frame = requestAnimationFrame(this.draw);
  };
  dispose() { this.disposed = true; cancelAnimationFrame(this.frame); this.observer.disconnect(); window.removeEventListener('resize', this.resize); window.removeEventListener('orientationchange', this.resize); document.removeEventListener('visibilitychange', this.visibility); }
}
