import { useCallback, useEffect, useState } from 'react';
import type { DisplayConfig } from '../config';

export type Experience = 'library' | 'workspace' | 'reactive';
export interface NavigationState { experience: Experience }

/** Demo URLs are legacy Reactive Vision URLs and deliberately take precedence. */
export function navigationFromSearch(search: string, demo = false): NavigationState {
  const params = new URLSearchParams(search);
  if (demo || params.get('demo') === '1' || params.get('experience') === 'reactive') return { experience: 'reactive' };
  return { experience: params.get('experience') === 'guide' ? 'workspace' : 'library' };
}

export function urlForExperience(experience: Experience, search: string): string {
  const params = new URLSearchParams(search);
  if (experience === 'reactive') params.set('experience', 'reactive');
  else if (experience === 'workspace') params.set('experience', 'guide');
  else params.delete('experience');
  const query = params.toString();
  return `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
}

export function useNavigation(config: DisplayConfig) {
  const [state, setState] = useState(() => navigationFromSearch(window.location.search, config.demo));
  useEffect(() => {
    const sync = () => setState(navigationFromSearch(window.location.search, config.demo));
    window.addEventListener('popstate', sync); return () => window.removeEventListener('popstate', sync);
  }, [config.demo]);
  const navigate = useCallback((experience: Experience) => {
    const url = urlForExperience(experience, window.location.search);
    window.history.pushState({ experience }, '', url);
    setState(navigationFromSearch(window.location.search, config.demo));
  }, [config.demo]);
  return { ...state, navigate };
}
