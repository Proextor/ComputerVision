import esp32Guide from '../data/esp32-oled.json';
import { App as ReactiveVisionApp } from '../App';
import type { DisplayConfig } from '../config';
import { GuideLibrary } from '../features/library/GuideLibrary';
import { CameraSurface } from '../features/camera/CameraSurface';
import { parseGuide } from '../task/guideValidation';
import { useNavigation } from './navigation';

const guideResult = parseGuide(esp32Guide);

/** The LUMO shell leaves the existing camera application mounted only in explicit Reactive Vision modes. */
export function App({ config }: { config: DisplayConfig }) {
  const { experience, navigate } = useNavigation(config);
  if (experience === 'reactive') return <ReactiveVisionApp config={config}/>;
  if (!guideResult.ok) return <main className="lumo-shell"><p role="alert">The local guide library could not be loaded.</p></main>;
  if (experience === 'workspace') return <div className="lumo-shell"><CameraSurface guideTitle={guideResult.value.title} onLeave={() => navigate('library')}/></div>;
  return <div className="lumo-shell"><GuideLibrary guide={guideResult.value} onStartGuide={() => navigate('workspace')} onOpenReactive={() => navigate('reactive')}/></div>;
}
