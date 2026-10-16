import type { GuideDefinition } from '../../task/guideTypes';

export function GuideCard({ guide, selected, onStart }: { guide: GuideDefinition; selected: boolean; onStart: () => void }) {
  const components = guide.components.map(component => component.name).join(' · ');
  return <article className="guide-card" aria-label={guide.title} data-selected={selected}>
    <div className="guide-card__topline"><span className="guide-card__eyebrow">Featured guide</span><span className="guide-card__version">v{guide.version}</span></div>
    <h2>{guide.title}</h2>
    <p>{guide.description}</p>
    <dl className="guide-card__facts">
      <div><dt>Components</dt><dd>{components}</dd></div>
      <div><dt>Estimated duration</dt><dd>{guide.estimatedMinutes} minutes</dd></div>
      <div><dt>Verification</dt><dd>User-confirmed verification</dd></div>
    </dl>
    <button className="lumo-button lumo-button--primary" onClick={onStart}>{selected ? 'Guide selected' : 'Start Guide'}</button>
  </article>;
}
