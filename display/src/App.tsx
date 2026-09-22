import { useCallback, useEffect, useRef, useState } from 'react';
import type { DisplayConfig } from './config';
import { createSource } from './source';
import { ReactionController } from './reaction';
import { Character } from './Character';
import type { CharacterAnimator, OverlayInput } from './renderer';
import { useReducedMotion } from './motion';
import { DemoVerificationSource } from './verification';
import { CameraController } from './camera';
import { FrameSender, encodeVideo } from './ingest';
import { initialTransport } from './transport';

function Icon({ name }: { name: 'switch' | 'info' | 'camera' | 'close' | 'play' }) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    {name === 'switch' ? <><path d="M20 10a8 8 0 0 0-14-5L3 8m0-5v5h5M4 14a8 8 0 0 0 14 5l3-3m0 5v-5h-5"/></> :
      name === 'camera' ? <><rect x="3" y="6" width="18" height="14" rx="3"/><path d="M8 6l1-3h6l1 3"/><circle cx="12" cy="13" r="4"/></> :
        name === 'close' ? <path d="m6 6 12 12M18 6 6 18"/> : name === 'play' ? <path d="m9 5 10 7-10 7z"/> :
          <><circle cx="12" cy="12" r="8"/><path d="M12 11v6m0-10v1"/></>}
  </svg>;
}
const cameraMessages = {
  idle: 'Your view. A little more alive.', requesting: 'Waiting for camera permission…', active: '',
  denied: 'Camera permission was denied. Allow it in your browser settings to try again.',
  missing: 'No camera was found on this device.', busy: 'The camera is busy. Close other camera apps and try again.',
  interrupted: 'The camera stream was interrupted.', unsupported: 'Camera access needs localhost or a trusted HTTPS connection.',
  error: 'The camera could not start. Try again.',
};

export function App({ config }: { config: DisplayConfig }) {
  const [{ source, demo }] = useState(() => createSource(config));
  const [controller] = useState(() => new ReactionController(config.mirrorX));
  const [verification] = useState(() => demo ? new DemoVerificationSource() : null);
  const [camera] = useState(() => new CameraController());
  const video = useRef<HTMLVideoElement>(null);
  const [sender] = useState(() => {
    const canvas = document.createElement('canvas');
    return new FrameSender(() => video.current ? encodeVideo(video.current, canvas, config.captureWidth, config.captureHeight) : Promise.resolve(null));
  });
  const [cameraState, setCameraState] = useState(camera.getState());
  const [ingest, setIngest] = useState(sender.getState());
  const [transport, setTransport] = useState(source.getState());
  const [visual, setVisual] = useState(controller.step(performance.now(), 0));
  const [sheet, setSheet] = useState<'diagnostics' | 'connect' | 'demo' | null>(null);
  const [token, setToken] = useState('');
  const [fps, setFps] = useState(0);
  const [notice, setNotice] = useState('');
  const animator = useRef<CharacterAnimator | null>(null);
  const motion = useReducedMotion();
  const renderInput = useRef<OverlayInput>({ scene: null, reliable: false, mirrored: false });
  const reducedRef = useRef(motion.reduced);
  useEffect(() => {
    reducedRef.current = motion.reduced;
    renderInput.current = { scene: transport.latest,
      reliable: !transport.localStale && (Boolean(demo) || (ingest.status === 'sending' && ingest.sessionId === transport.latest?.session_id)),
      mirrored: cameraState.status === 'active' ? cameraState.facing === 'user' : config.mirrorX };
  }, [motion.reduced, transport, cameraState, ingest, demo, config.mirrorX]);
  const input = useCallback(() => renderInput.current, []);
  const reduced = useCallback(() => reducedRef.current, []);

  useEffect(() => {
    controller.setVerificationProvider(verification ? 'demo' : 'unavailable');
    const unsubscribeVerification = verification?.subscribe(event => controller.ingestVerification(event, performance.now()));
    verification?.start();
    const unsubscribeStep = demo?.subscribeStep(index => {
      if (index === 4 || index === 5) { controller.armVerification(performance.now()); verification?.beginPending(); }
      if (index === 6) { controller.armVerification(performance.now()); verification?.beginPending(); verification?.finish('verified'); }
      if (index === 17) { controller.armVerification(performance.now()); verification?.run('rejected'); }
    });
    const unsubscribe = source.subscribe((state, delivery) => {
      if (!demo && state.latest && state.latest.session_id !== sender.getState().sessionId) return;
      controller.updateTransport(state);
      if (delivery) controller.ingest(delivery.envelope.snapshot, performance.now(), delivery.replay);
      setTransport(state);
    });
    const unsubscribeSender = sender.subscribe(state => {
      setIngest(state);
      if (state.status === 'sending') source.start();
      else if (!demo) { source.stop(); const empty = initialTransport(); controller.updateTransport(empty); setTransport(empty); }
    });
    const unsubscribeCamera = camera.subscribe(state => {
      setCameraState(state);
      if (state.status !== 'active' && !demo) sender.disconnect();
    });
    if (demo) source.start();
    const timer = setInterval(() => { setVisual(controller.step(performance.now(), 0)); setFps(animator.current?.getFPS() ?? 0); }, 250);
    return () => {
      clearInterval(timer); unsubscribe(); unsubscribeStep?.(); unsubscribeVerification?.(); unsubscribeSender(); unsubscribeCamera();
      verification?.stop(); source.stop(); sender.disconnect(); camera.stop();
    };
  }, [camera, controller, demo, sender, source, verification]);
  useEffect(() => {
    if (!video.current) return;
    video.current.srcObject = cameraState.stream;
    if (cameraState.stream) void video.current.play().catch(() => setNotice('Tap the camera preview to resume playback.'));
  }, [cameraState.stream]);
  const closeSheet = () => { setSheet(null); setToken(''); };
  async function fullscreen() {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { setNotice('Fullscreen is unavailable. The camera still works in this browser view.'); }
  }
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.repeat || event.ctrlKey || event.altKey || event.metaKey || (event.target instanceof HTMLElement && ['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))) return;
      if (event.key.toLowerCase() === 'd') setSheet(value => value === 'diagnostics' ? null : 'diagnostics');
      else if (event.key.toLowerCase() === 'm') motion.toggle();
      else if (event.key.toLowerCase() === 'f') void fullscreen();
      else if (event.key === 'Escape') { setSheet(null); setToken(''); }
      else return;
      event.preventDefault();
    };
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key);
  }, [motion]);
  const scene = transport.latest, room = scene?.room_state, controls = demo?.getControls();
  const cameraEnabled = !demo || config.cameraDemo;
  const active = cameraState.status === 'active';
  const status = demo ? 'DEMO' : ingest.status === 'sending' ? transport.localStale ? 'Vision delayed' : 'Vision connected' : active ? 'Local preview' : 'Camera off';
  const facts = [
    ['Mode', demo ? config.cameraDemo ? 'Camera overlay demo · no upload' : 'Visual demo' : 'Live browser camera'],
    ['Camera', cameraState.status], ['Facing', cameraState.facing], ['Upload', ingest.status], ['Frames sent / dropped', `${ingest.sent} / ${ingest.dropped}`],
    ['Vision connection', transport.connection], ['Service', transport.serviceStatus], ['Scene session', scene?.session_id], ['Publication', transport.publicationSequence],
    ['Snapshot age', transport.snapshotAgeSeconds === null ? null : `${transport.snapshotAgeSeconds.toFixed(1)} s`],
    ['Visible count', scene?.visible_count], ['Occupancy', room?.occupancy.state], ['Interaction', room?.interaction.state], ['Activity', room?.activity.aggregate_activity_level],
    ['Focus', room?.focus_candidate ? `temporary #${room.focus_candidate.track_id}` : 'none'],
    ['Verification provider', controller.providerState], ['Verification session', visual.verificationSessionId], ['Last verification', visual.verificationStatus],
    ['Baseline', visual.baseline], ['Reaction', visual.transient], ['Animation FPS', fps.toFixed(0)], ['Viewport', `${innerWidth} × ${innerHeight}`], ['Reduced motion', motion.reduced ? 'on' : 'off'],
  ];
  return <main className={`camera-app ${active ? 'has-camera' : 'sample-view'}`} data-reduced-motion={motion.reduced}>
    <div className="sample-background" aria-hidden="true"><div className="sample-window"/><div className="sample-bench"/></div>
    <video ref={video} className={`camera-video ${cameraState.facing === 'user' ? 'mirrored' : ''}`} playsInline muted autoPlay aria-hidden={!active} aria-label="Local camera preview"
      onClick={() => { void video.current?.play().then(() => setNotice('')).catch(() => {}); }} />
    <div className="camera-grade" aria-hidden="true"/>
    <Character controller={controller} input={input} reduced={reduced} animatorRef={animator}/>
    <header className="camera-header">
      <div className="wordmark"><span className="brand-glyph" aria-hidden="true"><i/><i/></span><span>AURA</span></div>
      <div className="header-actions"><span className={`status-chip ${demo ? 'demo-chip' : ''}`}>{status}</span>
        <button className="icon-button" aria-label="Switch camera" title="Switch front/rear camera" disabled={!active} onClick={() => { sender.disconnect(); void camera.switchCamera(); }}><Icon name="switch"/></button>
        <button className="icon-button" aria-label="Diagnostics" aria-expanded={sheet === 'diagnostics'} onClick={() => setSheet('diagnostics')}><Icon name="info"/></button>
      </div>
    </header>
    {cameraEnabled && !active && <section className="camera-prompt">
      <span className="prompt-icon"><Icon name="camera"/></span>
      <h1>{cameraState.status === 'idle' ? 'Bring your space to life.' : 'Let’s get your camera ready.'}</h1>
      <p>{cameraMessages[cameraState.status]}</p>
      <button className="primary-button" disabled={cameraState.status === 'requesting'} onClick={() => { setNotice(''); void camera.start(cameraState.facing); }}>Start camera</button>
      <small>{demo ? 'Camera stays on this device. Demo reactions only.' : 'Preview locally first. Connect when you’re ready.'}</small>
    </section>}
    {cameraState.stream && <span className="camera-indicator"><i/> {active ? 'CAMERA ACTIVE' : 'CAMERA INTERRUPTED'}</span>}
    <div className="bottom-actions">
      {demo ? <button className="glass-button" aria-label="Open demo controls" onClick={() => setSheet('demo')}><Icon name="play"/> Review sequence <span>{String((controls?.index ?? 0) + 1).padStart(2,'0')} / 18</span></button> :
        active && <button className="glass-button" onClick={() => setSheet('connect')}>{ingest.status === 'sending' ? 'Connection settings' : 'Connect vision'} <span>↗</span></button>}
    </div>
    {notice && <button className="notice" onClick={() => setNotice('')}>{notice}</button>}
    <span className="sr-only" role="status" aria-live="polite">{visual.transient ? visual.transient === 'verified' ? 'Verified' : visual.transient === 'verification_retry' ? 'Try again' : visual.caption : status}</span>
    {sheet && <div className="sheet-backdrop" onClick={closeSheet}><aside className="bottom-sheet" aria-label={sheet === 'diagnostics' ? 'Live diagnostics' : sheet === 'connect' ? 'Vision connection' : 'Demo controls'} onClick={e => e.stopPropagation()}>
      <div className="sheet-handle"/><div className="sheet-heading"><h2>{sheet === 'diagnostics' ? 'Behind the lens' : sheet === 'connect' ? 'Connect your vision service' : 'Explore the reactions'}</h2><button className="icon-button" aria-label="Close sheet" onClick={closeSheet}><Icon name="close"/></button></div>
      {sheet === 'connect' && <>
        <p>Connect to send a small live camera stream to your configured vision service. Nothing is recorded.</p>
        <p className="endpoint">{config.httpUrl}</p>
        {ingest.status === 'sending' ? <button className="primary-button" onClick={() => sender.disconnect()}>Disconnect vision</button> :
          <form onSubmit={e => { e.preventDefault(); if (!demo && active) sender.connect(config.ingestUrl, token); setToken(''); }}>
            <label>Ingest token<input type="password" autoComplete="off" value={token} maxLength={256} minLength={16} required onChange={e => setToken(e.target.value)}/></label>
            <button className="primary-button" type="submit" disabled={!active || ingest.status === 'connecting' || ingest.status === 'authenticating'}>Connect and send frames</button>
          </form>}
        {ingest.status === 'error' && <p role="alert">Couldn’t connect. Check the token, service address and whether another camera is connected.</p>}
        <small>The token stays in memory and is cleared when submitted or this sheet closes.</small>
      </>}
      {sheet === 'demo' && demo && controls && <>
        <p className="demo-scenario"><span>DEMO</span> {controls.label}</p>
        <div className="playback"><button aria-label="Previous demo step" onClick={() => demo.previous()}>Previous</button><button aria-label={controls.playing ? 'Pause demo' : 'Play demo'} onClick={() => demo.toggle()}>{controls.playing ? 'Pause' : 'Play'}</button><button aria-label="Next demo step" onClick={() => demo.next()}>Next</button><button aria-label="Restart demo" onClick={() => demo.restart()}>Restart</button></div>
        <label>Scenario<select aria-label="Select demo scenario" value={controls.index} onChange={e => demo.selectStep(Number(e.target.value))}>{Array.from({length:controls.total},(_, i) => <option key={i} value={i}>{i+1}. {demo.stepLabel(i)}</option>)}</select></label>
        <button className="primary-button" aria-label="Begin demo verification" onClick={() => { controller.armVerification(performance.now()); verification?.begin(); closeSheet(); }}>Preview verification</button>
        <small>Simulated results. No identity verification or frame upload.</small>
      </>}
      {sheet === 'diagnostics' && <><div className="diagnostic-controls"><button aria-label="Reduced motion" aria-pressed={motion.reduced} onClick={motion.toggle}>Reduced motion: {motion.reduced ? 'on' : 'off'}</button><button onClick={() => void fullscreen()}>Fullscreen</button>{cameraEnabled && <button onClick={() => camera.stop()}>Stop camera</button>}</div><dl>{facts.map(([name,value]) => <div key={name}><dt>{name}</dt><dd>{value ?? 'unavailable'}</dd></div>)}</dl></>}
    </aside></div>}
  </main>;
}
