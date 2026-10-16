import { useCallback, useEffect, useRef, useState } from 'react';
import { CameraController, type CameraState } from '../../camera';

export type WorkspaceCameraStatus = CameraState['status'] | 'insecure' | 'playback-error';
export interface UseCameraOptions { controller?: CameraController; enumerateDevices?: () => Promise<MediaDeviceInfo[]>; secureContext?: boolean }
/** Owns a local-only MediaStream. It does not encode, send, or upload frames. */
export function useCamera(options: UseCameraOptions = {}) {
  const controllerRef = useRef<CameraController | null>(null);
  if (!controllerRef.current) controllerRef.current = options.controller ?? new CameraController();
  const controller = controllerRef.current;
  const [camera, setCamera] = useState<CameraState>(() => controller.getState());
  const [canSwitch, setCanSwitch] = useState(false), [switchUnsupported, setSwitchUnsupported] = useState(false);
  const [workspaceStatus, setWorkspaceStatus] = useState<WorkspaceCameraStatus>(() => controller.getState().status);
  const [calibrationRevision, setCalibrationRevision] = useState(0);
  const enumerateDevicesRef = useRef(options.enumerateDevices ?? (() => navigator.mediaDevices?.enumerateDevices?.() ?? Promise.resolve([])));
  const enumerateDevices = enumerateDevicesRef.current;
  const secure = options.secureContext ?? (typeof window === 'undefined' || window.isSecureContext !== false);
  const refreshDevices = useCallback(async () => { try { setCanSwitch((await enumerateDevices()).filter(device => device.kind === 'videoinput').length > 1); } catch { setCanSwitch(false); } }, [enumerateDevices]);
  useEffect(() => { const unsubscribe = controller.subscribe(next => { setCamera(next); setWorkspaceStatus(next.status); }); void refreshDevices(); return () => { unsubscribe(); controller.stop(); }; }, [controller, refreshDevices]);
  const start = useCallback(async (facing: CameraState['facing'] = 'environment') => { setSwitchUnsupported(false); if (!secure) { setWorkspaceStatus('insecure'); return; } await controller.start(facing); await refreshDevices(); }, [controller, refreshDevices, secure]);
  const switchCamera = useCallback(async () => { if (!canSwitch) { setSwitchUnsupported(true); return false; } setCalibrationRevision(value => value + 1); setSwitchUnsupported(false); await controller.switchCamera(); await refreshDevices(); return true; }, [canSwitch, controller, refreshDevices]);
  const stop = useCallback(() => { setSwitchUnsupported(false); controller.stop(); }, [controller]);
  const reportPlaybackError = useCallback(() => setWorkspaceStatus('playback-error'), []);
  return { ...camera, workspaceStatus, canSwitch, switchUnsupported, calibrationRevision, start, switchCamera, stop, reportPlaybackError };
}
