import { useEffect, useState } from 'react';
import { GameView } from './GameView';
import { ReplayView } from './ReplayView';
import './App.css';

/** The pages there are, in the order they are offered. */
const VIEWS = [
  { name: 'game', label: 'Game' },
  { name: 'replay', label: 'Replay' },
] as const;

type View = (typeof VIEWS)[number]['name'];

/** The view the address names, so that a reload comes back to the one you were on. */
function viewInAddress(): View {
  return window.location.hash === '#replay' ? 'replay' : 'game';
}

/** Which view is on show, and how to go to another one. */
function useView(): [View, (view: View) => void] {
  const [view, setView] = useState<View>(viewInAddress);

  useEffect(() => {
    // The browser's own Back and Forward move between the views, so follow the address.
    const follow = () => setView(viewInAddress());
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);

  return [
    view,
    (next: View) => {
      setView(next);
      window.location.hash = next === 'replay' ? '#replay' : '';
    },
  ];
}

export function App() {
  const [view, show] = useView();

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
      {view === 'game' ? <GameView /> : <ReplayView />}
    </main>
  );
}
