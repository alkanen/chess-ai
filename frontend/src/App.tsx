import { useEffect, useState } from 'react';
import { CompareView } from './CompareView';
import { GameView } from './GameView';
import { ReplayView } from './ReplayView';
import { RunsView } from './RunsView';
import { RunView } from './RunView';
import './App.css';

/** The pages there are, in the order they are offered. */
const VIEWS = [
  { name: 'game', label: 'Game', hash: '' },
  { name: 'replay', label: 'Replay', hash: '#replay' },
  { name: 'runs', label: 'Runs', hash: '#runs' },
] as const;

type View = (typeof VIEWS)[number]['name'];

/** Where the address says the viewer is: a view, and for the runs view perhaps one run. */
interface Place {
  view: View;
  /** The run on show, under the runs view; null for the list of them. */
  run: string | null;
  /** The runs compared side by side, under the runs view; null when none are. */
  compare: string[] | null;
}

const RUN_PAGE = /^#runs\/(.+)$/;
const COMPARE_PAGE = /^#compare\/(.+)$/;

/** The place the address names, so that a reload comes back to the one you were on. */
function placeInAddress(): Place {
  const { hash } = window.location;
  const run = RUN_PAGE.exec(hash);
  if (run !== null) {
    return { view: 'runs', run: decodeURIComponent(run[1]), compare: null };
  }
  const compare = COMPARE_PAGE.exec(hash);
  if (compare !== null) {
    // A run's name has no commas in it, so they are what separates one from the next.
    const names = compare[1].split(',').filter((name) => name !== '').map(decodeURIComponent);
    return { view: 'runs', run: null, compare: names };
  }
  const view = VIEWS.find((candidate) => candidate.hash !== '' && candidate.hash === hash);
  return { view: view?.name ?? 'game', run: null, compare: null };
}

/** Which place is on show, and how to go to another view. */
function usePlace(): [Place, (view: View) => void] {
  const [place, setPlace] = useState<Place>(placeInAddress);

  useEffect(() => {
    // The browser's own Back and Forward move between the views, and a link to a run is
    // followed the same way, so follow the address.
    const follow = () => setPlace(placeInAddress());
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);

  return [
    place,
    (next: View) => {
      setPlace({ view: next, run: null, compare: null });
      window.location.hash = VIEWS.find((view) => view.name === next)!.hash;
    },
  ];
}

export function App() {
  const [place, show] = usePlace();
  const { view } = place;

  return (
    <main className="app">
      <header className="app-header">
        <h1>chess-ai</h1>
        <nav aria-label="Views">
          {VIEWS.map(({ name, label }) => (
            <button
              key={name}
              type="button"
              aria-current={view === name ? 'page' : undefined}
              onClick={() => show(name)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      {view === 'game' && <GameView />}
      {view === 'replay' && <ReplayView />}
      {view === 'runs' && place.compare !== null && <CompareView names={place.compare} />}
      {view === 'runs' && place.compare === null && place.run === null && <RunsView />}
      {view === 'runs' && place.run !== null && <RunView key={place.run} name={place.run} />}
    </main>
  );
}
