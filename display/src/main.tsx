import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { readConfig } from './config';
import './styles.css';

const root = createRoot(document.getElementById('root')!);
try {
  const config = readConfig(import.meta.env, window.location.search);
  root.render(<StrictMode><App config={config} /></StrictMode>);
} catch (error) {
  root.render(<main className="configuration-error"><h1>Display configuration needs attention.</h1>
    <p>{error instanceof Error ? error.message : 'Invalid environment settings.'}</p><p>Check display/.env.local and restart Vite.</p></main>);
}
