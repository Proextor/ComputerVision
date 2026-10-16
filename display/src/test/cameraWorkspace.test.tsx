import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CameraController } from '../camera';
import { CameraSurface } from '../features/camera/CameraSurface';
import { coverGeometry, normalizedToOverlay, sourceToNormalized } from '../features/camera/cameraState';

type FakeTrack = { stop: ReturnType<typeof vi.fn>; onended: (() => void) | null; onmute: (() => void) | null; onunmute: (() => void) | null; getSettings: () => MediaTrackSettings };
function fakeStream(facing: 'user' | 'environment' = 'environment') {
  const track: FakeTrack = { stop: vi.fn(), onended: null, onmute: null, onunmute: null, getSettings: () => ({ facingMode: facing }) };
  return { track, value: { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream };
}
function controller(getUserMedia: ReturnType<typeof vi.fn>) { return new CameraController({ getUserMedia } as unknown as MediaDevices); }
const twoCameras = async () => [{ kind: 'videoinput' }, { kind: 'videoinput' }] as MediaDeviceInfo[];

beforeEach(() => { vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined); });
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe('LUMO local camera workspace', () => {
  it('requests a rear camera only after Start Camera and renders a local preview', async () => {
    const stream = fakeStream(), getUserMedia = vi.fn().mockResolvedValue(stream.value);
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    const socket = vi.fn(); vi.stubGlobal('WebSocket', socket);
    render(<CameraSurface guideTitle="ESP32 + OLED Assembly" onLeave={vi.fn()} controller={controller(getUserMedia)} enumerateDevices={twoCameras}/>);
    expect(getUserMedia).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' }));
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledOnce());
    expect(getUserMedia).toHaveBeenCalledWith(expect.objectContaining({ video: expect.objectContaining({ facingMode: { ideal: 'environment' } }) }));
    await waitFor(() => expect(screen.getByText('Local preview')).toBeTruthy());
    expect(play).toHaveBeenCalled();
    expect(socket).not.toHaveBeenCalled();
    expect(screen.getByText(/does not upload camera frames/i)).toBeTruthy();
  });

  it('handles permission denial, missing media APIs, and insecure contexts without frame upload', async () => {
    const denied = controller(vi.fn().mockRejectedValue(new DOMException('no', 'NotAllowedError')));
    const { unmount } = render(<CameraSurface guideTitle="Guide" onLeave={vi.fn()} controller={denied}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' }));
    await waitFor(() => expect(screen.getAllByText(/permission was denied/i).length).toBeGreaterThan(0)); unmount();
    const missing = new CameraController({} as MediaDevices);
    render(<CameraSurface guideTitle="Guide" onLeave={vi.fn()} controller={missing}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' }));
    await waitFor(() => expect(screen.getAllByText(/does not provide camera access/i).length).toBeGreaterThan(0));
  });

  it('reports an insecure context before requesting media', async () => {
    const getUserMedia = vi.fn();
    render(<CameraSurface guideTitle="Guide" onLeave={vi.fn()} controller={controller(getUserMedia)} secureContext={false}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' }));
    await waitFor(() => expect(screen.getAllByText(/secure HTTPS connection/i).length).toBeGreaterThan(0));
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it('reports a rejected local preview play request', async () => {
    const stream = fakeStream(), getUserMedia = vi.fn().mockResolvedValue(stream.value);
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockRejectedValue(new DOMException('blocked', 'NotAllowedError'));
    render(<CameraSurface guideTitle="Guide" onLeave={vi.fn()} controller={controller(getUserMedia)}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' }));
    await waitFor(() => expect(screen.getAllByText(/could not play the preview/i).length).toBeGreaterThan(0));
  });

  it('switches cameras, invalidates calibration, and cleans up tracks on leave', async () => {
    const rear = fakeStream('environment'), front = fakeStream('user'), getUserMedia = vi.fn().mockResolvedValueOnce(rear.value).mockResolvedValueOnce(front.value);
    const invalidated = vi.fn();
    const view = render(<CameraSurface guideTitle="Guide" onLeave={vi.fn()} controller={controller(getUserMedia)} enumerateDevices={twoCameras} onCalibrationInvalidated={invalidated}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' }));
    await screen.findByText('Local preview');
    fireEvent.click(screen.getByRole('button', { name: 'Switch camera' }));
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(2));
    expect(rear.track.stop).toHaveBeenCalledOnce();
    expect(invalidated).toHaveBeenLastCalledWith(1);
    view.unmount(); expect(front.track.stop).toHaveBeenCalledOnce();
  });

  it('handles an interrupted track and safely supports repeated start and stop', async () => {
    const first = fakeStream(), second = fakeStream(), getUserMedia = vi.fn().mockResolvedValueOnce(first.value).mockResolvedValueOnce(second.value);
    render(<CameraSurface guideTitle="Guide" onLeave={vi.fn()} controller={controller(getUserMedia)} enumerateDevices={twoCameras}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Start Camera' })); await screen.findByText('Local preview');
    first.track.onended?.(); await waitFor(() => expect(screen.getAllByText(/stream was interrupted/i).length).toBeGreaterThan(0));
    fireEvent.click(screen.getByRole('button', { name: 'Try camera again' })); await screen.findByText('Local preview');
    fireEvent.click(screen.getByRole('button', { name: 'Stop camera' }));
    await waitFor(() => expect(screen.getAllByText(/stays off until/i).length).toBeGreaterThan(0));
    expect(first.track.stop).toHaveBeenCalledOnce(); expect(second.track.stop).toHaveBeenCalledOnce();
  });
});

describe('camera geometry contract', () => {
  it('maps source, normalized, and object-fit cover overlay coordinates', () => {
    const geometry = coverGeometry({ width: 640, height: 360 }, { width: 360, height: 800 });
    expect(geometry?.rendered).toMatchObject({ x: -531.1111111111111, y: 0, height: 800 });
    expect(sourceToNormalized({ x: 320, y: 180 }, { width: 640, height: 360 })).toEqual({ x: .5, y: .5 });
    expect(normalizedToOverlay({ x: .5, y: .5 }, geometry!)).toEqual({ x: 180, y: 400 });
  });
});
