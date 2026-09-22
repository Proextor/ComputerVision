import { StrictMode } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { App } from '../App';
import { readConfig } from '../config';

it('requires a camera action, hides diagnostics and supports keyboard controls', () => {
  vi.useFakeTimers();
  const { container, unmount } = render(<StrictMode><App config={readConfig({}, '')} /></StrictMode>);
  expect(screen.getByRole('button',{name:'Start camera'})).toBeTruthy();
  expect(screen.queryByRole('complementary')).toBeNull();
  fireEvent.keyDown(window, { key: 'd' }); expect(screen.getByLabelText('Live diagnostics')).toBeTruthy();
  fireEvent.keyDown(window, { key: 'Escape' }); expect(screen.queryByRole('complementary')).toBeNull();
  fireEvent.keyDown(window, { key: 'm' }); expect(container.querySelector('main')?.dataset.reducedMotion).toBe('true');
  unmount(); expect(vi.getTimerCount()).toBe(0);
});
it('honours reduced motion, keeps demo controls collapsed and needs no backend', () => {
  vi.useFakeTimers(); const socket=vi.spyOn(globalThis,'WebSocket');
  vi.mocked(window.matchMedia).mockReturnValueOnce({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() } as unknown as MediaQueryList)
    .mockReturnValueOnce({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() } as unknown as MediaQueryList);
  render(<App config={readConfig({}, '?demo=1')} />);
  expect(screen.queryByLabelText('Demo controls')).toBeNull();
  fireEvent.click(screen.getByLabelText('Diagnostics'));
  expect(screen.getByLabelText('Reduced motion').getAttribute('aria-pressed')).toBe('true');
  act(() => vi.advanceTimersByTime(250)); expect(socket).not.toHaveBeenCalled();
});
it('camera overlay demo requests only local video and never opens an ingest socket', async () => {
  vi.useFakeTimers(); const socket=vi.spyOn(globalThis,'WebSocket');
  const tracks = [{stop:vi.fn(),getSettings:()=>({facingMode:'user'}),onended:null,onmute:null,onunmute:null}];
  const getUserMedia = vi.fn().mockResolvedValue({getTracks:()=>tracks,getVideoTracks:()=>tracks});
  Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia}});
  vi.spyOn(HTMLMediaElement.prototype,'play').mockResolvedValue(undefined);
  const {container,unmount}=render(<App config={readConfig({},'?demo=1&camera=1')}/>);
  expect(getUserMedia).not.toHaveBeenCalled();
  await act(async()=>fireEvent.click(screen.getByRole('button',{name:'Start camera'})));
  expect(container.querySelector('video')?.classList.contains('mirrored')).toBe(true);
  expect(screen.getByText('CAMERA ACTIVE')).toBeTruthy();
  act(()=>vi.advanceTimersByTime(1000)); expect(socket).not.toHaveBeenCalled();
  unmount(); expect(tracks[0].stop).toHaveBeenCalled();
});
