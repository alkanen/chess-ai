import { useEffect, useState } from 'react';
import { fetchStartPosition, type PositionSnapshot } from './api';
import { Board } from './board/Board';
import { TurnIndicator } from './TurnIndicator';
import './App.css';

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; snapshot: PositionSnapshot }
  | { status: 'failed'; message: string };

export function App() {
  const [state, setState] = useState<LoadState>({ status: 'loading' });

  useEffect(() => {
    const controller = new AbortController();
    fetchStartPosition(controller.signal).then(
      (snapshot) => setState({ status: 'ready', snapshot }),
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setState({ status: 'failed', message: String(error) });
        }
      },
    );
    return () => controller.abort();
  }, []);

  return (
    <main className="app">
      <h1>chess-ai</h1>
      {state.status === 'loading' && <p>Loading…</p>}
      {state.status === 'failed' && <p role="alert">Could not load the position: {state.message}</p>}
      {state.status === 'ready' && (
        <div className="game">
          <TurnIndicator turn={state.snapshot.turn} />
          <div className="board-frame">
            <Board snapshot={state.snapshot} />
          </div>
        </div>
      )}
    </main>
  );
}
