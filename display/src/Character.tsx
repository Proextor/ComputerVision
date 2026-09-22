import { useEffect, useRef } from 'react';
import type { ReactionController } from './reaction';
import { CharacterAnimator, type OverlayInput } from './renderer';

/** Original local vector optics and reaction artwork. No imagery enters this component. */
export function Character({ controller, input, reduced, animatorRef }: {
  controller: ReactionController; input: () => OverlayInput; reduced: () => boolean;
  animatorRef: { current: CharacterAnimator | null };
}) {
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!root.current) return;
    const animator = new CharacterAnimator(root.current, controller, input, reduced);
    animatorRef.current = animator;
    return () => { animator.dispose(); animatorRef.current = null; };
  }, [controller, input, reduced, animatorRef]);
  return <div ref={root} className="reaction-unit" aria-hidden="true">
    <svg viewBox="0 0 240 160" fill="none" className="robot-optics">
      <defs>
        <linearGradient id="rim" x1="70" y1="23" x2="145" y2="113" gradientUnits="userSpaceOnUse"><stop stopColor="#c7d5df"/><stop offset=".25" stopColor="#637986"/><stop offset=".6" stopColor="#1d2b35"/><stop offset="1" stopColor="#536774"/></linearGradient>
        <linearGradient id="glass" x1="92" y1="30" x2="102" y2="109" gradientUnits="userSpaceOnUse"><stop stopColor="#203943"/><stop offset=".4" stopColor="#0c1b24"/><stop offset="1" stopColor="#030b12"/></linearGradient>
        <linearGradient id="optic" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#f7ffff"/><stop offset=".38" stopColor="#9ef5ff"/><stop offset="1" stopColor="#39b8d0"/></linearGradient>
        <linearGradient id="handMetal" x1="179" y1="67" x2="224" y2="129" gradientUnits="userSpaceOnUse"><stop stopColor="#f3e8c9"/><stop offset=".3" stopColor="#d4cba9"/><stop offset=".65" stopColor="#a69871"/><stop offset="1" stopColor="#665b44"/></linearGradient>
        <linearGradient id="handEdge" x1="180" y1="64" x2="228" y2="135" gradientUnits="userSpaceOnUse"><stop stopColor="#fff7dc"/><stop offset="1" stopColor="#6d664f"/></linearGradient>
        <filter id="unitShadow" x="-30%" y="-40%" width="170%" height="190%"><feDropShadow dx="0" dy="5" stdDeviation="5" floodColor="#000" floodOpacity=".38"/></filter>
        <filter id="opticGlow" x="-90%" y="-90%" width="280%" height="280%"><feGaussianBlur stdDeviation="2.7"/></filter>
        <clipPath id="leftAperture"><rect data-aperture="left" x="42" y="43" width="62" height="49" rx="18"/></clipPath>
        <clipPath id="rightAperture"><rect data-aperture="right" x="116" y="43" width="62" height="49" rx="18"/></clipPath>
      </defs>
      <g data-ring><circle cx="111" cy="68" r="59" stroke="#d6e8c5" strokeWidth=".8"/><circle cx="111" cy="68" r="63" stroke="#a7e7ed" strokeWidth=".4" strokeDasharray="2 18"/></g>
      <g data-eyes filter="url(#unitShadow)">
        <g data-left-eye>
          <rect x="36" y="33" width="74" height="69" rx="26" fill="url(#rim)"/>
          <rect x="37.4" y="34.4" width="71.2" height="66.2" rx="25" fill="url(#glass)"/>
          <path d="M48 42Q66 31 93 41" stroke="#d5efff" strokeOpacity=".24" strokeWidth="1.3" strokeLinecap="round"/>
          <g clipPath="url(#leftAperture)"><rect x="42" y="43" width="62" height="49" fill="#05131c"/>
            <g data-left-pupil><rect x="58" y="49" width="31" height="38" rx="13" fill="#7ae9ff" opacity=".65" filter="url(#opticGlow)"/>
              <rect x="62" y="51" width="23" height="34" rx="10" fill="url(#optic)"/><path d="M66 57v12" stroke="#fff" strokeOpacity=".7" strokeWidth="2" strokeLinecap="round"/></g>
            <path d="M43 49L98 81" stroke="#efffff" strokeOpacity=".07" strokeWidth="6"/>
          </g>
          <path d="M54 95h11" stroke="#8ac4cc" strokeWidth="1" strokeOpacity=".4"/>
        </g>
        <g data-right-eye>
          <rect x="110" y="33" width="74" height="69" rx="26" fill="url(#rim)"/>
          <rect x="111.4" y="34.4" width="71.2" height="66.2" rx="25" fill="url(#glass)"/>
          <path d="M122 42Q140 31 167 41" stroke="#d5efff" strokeOpacity=".24" strokeWidth="1.3" strokeLinecap="round"/>
          <g clipPath="url(#rightAperture)"><rect x="116" y="43" width="62" height="49" fill="#05131c"/>
            <g data-right-pupil><rect x="132" y="49" width="31" height="38" rx="13" fill="#7ae9ff" opacity=".65" filter="url(#opticGlow)"/>
              <rect x="136" y="51" width="23" height="34" rx="10" fill="url(#optic)"/><path d="M140 57v12" stroke="#fff" strokeOpacity=".7" strokeWidth="2" strokeLinecap="round"/></g>
            <path d="M117 49L172 81" stroke="#efffff" strokeOpacity=".07" strokeWidth="6"/>
          </g>
          <path d="M156 95h11" stroke="#8ac4cc" strokeWidth="1" strokeOpacity=".4"/>
        </g>
      </g>
      <path data-trace d="M49 109H171" stroke="#b8faff" strokeWidth="1.3" strokeLinecap="round" pathLength="1"/>
      <g data-scan><path d="M39 106H181" stroke="#9ce9f4" strokeOpacity=".18"/><circle cx="40" cy="106" r="2" fill="#c3faff" data-scan-dot/></g>
      <g data-thumb filter="url(#unitShadow)">
        <path d="M177 106l10-11 6-21c1-4 4-6 7-4 7 5 5 15 3 22h14c8 0 12 5 10 12l-5 20c-1 5-5 8-10 8h-21l-14-8z" fill="url(#handMetal)" stroke="url(#handEdge)" strokeWidth="1.5" strokeLinejoin="round"/>
        <path d="M190 99l8-23M206 98h13" stroke="#fff7e1" strokeOpacity=".64" strokeWidth="2" strokeLinecap="round"/>
        <path d="M205 108h17m-18 8h16m-17 8h14" stroke="#665f4d" strokeOpacity=".46" strokeWidth="1.2" strokeLinecap="round"/>
        <rect x="171" y="103" width="12" height="28" rx="4" fill="#263a46" stroke="#9aafad" strokeWidth="1"/>
        <path d="M175 108v17" stroke="#aff5ed" strokeWidth="1.5" strokeLinecap="round"/>
      </g>
      <g data-wave filter="url(#unitShadow)">
        <path d="M185 124l-9-20c-2-6 3-8 6-4l5 8-2-29c0-5 6-6 7-1l3 19V70c0-5 6-5 7 0l1 26 2-20c1-5 7-4 7 1l-1 22 3-13c1-5 7-4 6 1l-3 25c-1 10-6 17-15 18-7 0-13-1-17-6z" fill="url(#handMetal)" stroke="url(#handEdge)" strokeWidth="1.3"/>
        <path d="M197 113q4-9 12-8" stroke="#fff0ca" strokeOpacity=".6" strokeWidth="1.5"/>
        <path d="M220 59q10 7 10 19M178 65q-6 7-6 15" stroke="#c5f7ff" strokeOpacity=".45" strokeWidth="1.5" strokeLinecap="round"/>
      </g>
      <g data-burst stroke="#b5f0ef" strokeWidth="1.4" strokeLinecap="round"><path d="M24 41l-7-7m7 36H12m15 27l-8 8M192 41l7-7m-7 36h12m-15 27l8 8"/></g>
    </svg>
    <span className="reaction-pill" data-pill />
  </div>;
}
