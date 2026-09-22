import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

Object.defineProperty(window, 'matchMedia', { writable: true, value: vi.fn(() => ({
  matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn(),
})) });
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
HTMLCanvasElement.prototype.getContext = vi.fn(() => null);
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); });
