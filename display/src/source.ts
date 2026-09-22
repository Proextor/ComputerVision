import type { DisplayConfig } from './config';
import { VisionClient } from './client';
import { DemoSource } from './demo/source';
import type { DataSource } from './transport';

export function createSource(config: DisplayConfig): { source: DataSource; demo: DemoSource | null } {
  if (config.demo) { const demo = new DemoSource(); return { source: demo, demo }; }
  return { source: new VisionClient(config), demo: null };
}
