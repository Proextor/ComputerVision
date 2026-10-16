import type { GuideDefinition } from '../../task/guideTypes';
import { GuideCard } from './GuideCard';

export function GuideLibrary({ guide, onOpenReactive, onStartGuide }: { guide: GuideDefinition; onOpenReactive: () => void; onStartGuide: () => void }) {
  return <main className="lumo-library">
    <header className="lumo-library__header">
      <div className="lumo-brand" aria-label="LUMO"><span aria-hidden="true"/><strong>LUMO</strong></div>
      <button className="lumo-button lumo-button--quiet" onClick={onOpenReactive}>Open Reactive Vision</button>
    </header>
    <section className="lumo-library__intro" aria-labelledby="lumo-title">
      <p className="lumo-kicker">GUIDE LIBRARY</p>
      <h1 id="lumo-title">Spatial Guidance <span>&amp; Intelligent Assistance</span></h1>
      <p>Follow clear, tabletop-ready procedures at your own pace. Each step keeps its evidence boundary visible.</p>
    </section>
    <section className="lumo-library__featured" aria-label="Featured guide">
      <GuideCard guide={guide} selected={false} onStart={onStartGuide}/>
    </section>
    <aside className="lumo-library__privacy">
      <span aria-hidden="true">◌</span><p><strong>Local by default.</strong> The basic guide can run locally without uploading camera frames.</p>
    </aside>
  </main>;
}
