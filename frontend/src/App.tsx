import { Board } from './board/Board';
import { GameStatus } from './GameStatus';
import { NewGameForm } from './NewGameForm';
import { useGameChannel } from './useGameChannel';
import './App.css';

export function App() {
  const { view, connected } = useGameChannel();

  return (
    <main className="app">
      <h1>chess-ai</h1>
      {view === null ? (
        <p>Connecting to the server…</p>
      ) : (
        <>
          <GameStatus position={view.position} inGame={view.game !== null} />
          <div className="game-view">
            <div className="board-frame">
              <Board snapshot={view.position} />
            </div>
            <aside className="side-panel">
              {!connected && <p role="alert">Lost the connection to the server. Reconnecting…</p>}
              {view.game !== null && (
                <dl className="players">
                  <dt>White</dt>
                  <dd>{view.game.white}</dd>
                  <dt>Black</dt>
                  <dd>{view.game.black}</dd>
                </dl>
              )}
              <NewGameForm />
            </aside>
          </div>
        </>
      )}
    </main>
  );
}
