import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { App } from '../app/App';
import { navigationFromSearch, urlForExperience } from '../app/navigation';
import { readConfig } from '../config';

afterEach(() => window.history.replaceState({}, '', '/'));

describe('LUMO application shell', () => {
  it('uses the guide library by default and opens a camera-off guide workspace', () => {
    render(<App config={readConfig({}, '')}/>);
    expect(screen.getByRole('heading', { name: /spatial guidance/i })).toBeTruthy();
    expect(screen.getByText('ESP32 + OLED tabletop assembly')).toBeTruthy();
    expect(screen.getByText('User-confirmed verification')).toBeTruthy();
    expect(screen.getByText(/without uploading camera frames/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Start camera' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Start Guide' }));
    expect(window.location.search).toContain('experience=guide');
    expect(screen.getByRole('button', { name: 'Start Camera' })).toBeTruthy();
  });
  it('opens Reactive Vision through an explicit URL state and supports browser Back', () => {
    render(<App config={readConfig({}, '')}/>);
    fireEvent.click(screen.getByRole('button', { name: 'Open Reactive Vision' }));
    expect(window.location.search).toContain('experience=reactive');
    expect(screen.getByRole('button', { name: 'Start camera' })).toBeTruthy();
    act(() => { window.history.replaceState({}, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')); });
    expect(screen.getByRole('heading', { name: /spatial guidance/i })).toBeTruthy();
  });
  it('keeps the existing demo query modes reactive', () => {
    expect(navigationFromSearch('?demo=1')).toEqual({ experience: 'reactive' });
    expect(navigationFromSearch('?demo=1&camera=1')).toEqual({ experience: 'reactive' });
    expect(navigationFromSearch('?experience=reactive')).toEqual({ experience: 'reactive' });
    expect(navigationFromSearch('?experience=guide')).toEqual({ experience: 'workspace' });
    expect(navigationFromSearch('')).toEqual({ experience: 'library' });
    expect(urlForExperience('reactive', '?demo=1&camera=1')).toContain('demo=1');
    expect(urlForExperience('reactive', '?demo=1&camera=1')).toContain('experience=reactive');
  });
  it('honours demo configuration even when the browser URL has no query parameters', () => {
    render(<App config={readConfig({}, '?demo=1')}/>);
    expect(screen.getAllByText('DEMO').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: 'Open demo controls' })).toBeTruthy();
  });
});
