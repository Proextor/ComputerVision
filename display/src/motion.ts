import { useEffect, useState } from 'react';

export function useReducedMotion() {
  const [system, setSystem] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [override, setOverride] = useState<boolean | null>(null);
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => setSystem(query.matches);
    query.addEventListener('change', change); change();
    return () => query.removeEventListener('change', change);
  }, []);
  const reduced = override ?? system;
  return { reduced, toggle: () => setOverride(!reduced) };
}
