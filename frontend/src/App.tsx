import { useEffect, useState } from 'react';
import { CompareView } from './CompareView';
import {
  datasetGameInHash,
  datasetInHash,
  type DatasetGameRef,
  type DatasetPage,
} from './datasetPlaces';
import { DatasetsView } from './DatasetsView';
import { DatasetView } from './DatasetView';
import { GameLobby } from './GameLobby';
import { GameView } from './GameView';
import { ReplayView } from './ReplayView';
import { RunsView } from './RunsView';
import { RunView } from './RunView';
import { sampleGameInHash, type SampleGameRef } from './sampleGames';
import './App.css';

/** The pages there are, in the order they are offered. */
const VIEWS = [
  { name: 'game', label: 'Game', hash: '' },
  { name: 'replay', label: 'Replay', hash: '#replay' },
  { name: 'runs', label: 'Runs', hash: '#runs' },
  { name: 'datasets', label: 'Datasets', hash: '#datasets' },
] as const;

type View = (typeof VIEWS)[number]['name'];

/**
 * Where the address says the viewer is: a view, and for the game view perhaps one game, for
 * the runs view perhaps one run, and for the datasets view perhaps one dataset.
 */
interface Place {
  view: View;
  /** The link of the game on show, under the game view; null for starting one. */
  link: string | null;
  /** The run on show, under the runs view; null for the list of them. */
  run: string | null;
  /** The runs compared side by side, under the runs view; null when none are. */
  compare: string[] | null;
  /** The dataset on show, and which page of its games, under the datasets view. */
  dataset?: DatasetPage | null;
  /** The dataset game on show, under the replay view; null for a file or a saved game. */
  datasetGame?: DatasetGameRef | null;
  /** The run's sample game on show, under the replay view; null for a file or a saved game. */
  sampleGame?: SampleGameRef | null;
}

const GAME_PAGE = /^#(?:game|watch)\/(.+)$/;
const RUN_PAGE = /^#runs\/(.+)$/;
const COMPARE_PAGE = /^#compare\/(.+)$/;

/** The place the address names, so that a reload comes back to the one you were on. */
function placeInAddress(): Place {
  const { hash } = window.location;
  const game = GAME_PAGE.exec(hash);
  if (game !== null) {
    // Whether it is a play or a watch link is the server's to say; the address only makes
    // the one it is plain to whoever it is sent to.
    return {
      view: 'game',
      link: decodeURIComponent(game[1]),
      run: null,
      compare: null,
    };
  }
  const dataset = datasetInHash(hash);
  if (dataset !== null) {
    return { view: 'datasets', link: null, run: null, compare: null, dataset };
  }
  const datasetGame = datasetGameInHash(hash);
  if (datasetGame !== null) {
    return { view: 'replay', link: null, run: null, compare: null, datasetGame };
  }
  const sampleGame = sampleGameInHash(hash);
  if (sampleGame !== null) {
    return { view: 'replay', link: null, run: null, compare: null, sampleGame };
  }
  const run = RUN_PAGE.exec(hash);
  if (run !== null) {
    return {
      view: 'runs',
      link: null,
      run: decodeURIComponent(run[1]),
      compare: null,
    };
  }
  const compare = COMPARE_PAGE.exec(hash);
  if (compare !== null) {
    // A run's name has no commas in it, so they are what separates one from the next.
    const names = compare[1]
      .split(',')
      .filter((name) => name !== '')
      .map(decodeURIComponent);
    return { view: 'runs', link: null, run: null, compare: names };
  }
  const view = VIEWS.find((candidate) => candidate.hash !== '' && candidate.hash === hash);
  return { view: view?.name ?? 'game', link: null, run: null, compare: null };
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
      setPlace({ view: next, link: null, run: null, compare: null });
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
      {view === 'game' && place.link === null && <GameLobby />}
      {view === 'game' && place.link !== null && <GameView key={place.link} link={place.link} />}
      {view === 'replay' && (
        <ReplayView
          datasetGame={place.datasetGame ?? null}
          sampleGame={place.sampleGame ?? null}
        />
      )}
      {view === 'runs' && place.compare !== null && <CompareView names={place.compare} />}
      {view === 'runs' && place.compare === null && place.run === null && <RunsView />}
      {view === 'runs' && place.run !== null && <RunView key={place.run} name={place.run} />}
      {view === 'datasets' && place.dataset == null && <DatasetsView />}
      {view === 'datasets' && place.dataset != null && (
        <DatasetView key={place.dataset.name} place={place.dataset} />
      )}
    </main>
  );
}
