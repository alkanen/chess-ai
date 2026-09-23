import type { GameState, PositionSnapshot } from './api';
import { Board } from './board/Board';
import { GameStatus } from './GameStatus';
import { NewGameForm } from './NewGameForm';
import { useGameChannel } from './useGameChannel';
import './App.css';

/** Whether the side to move is played from this browser, so its moves can be made here. */
function yourTurn(game: GameState | null, position: PositionSnapshot): boolean {
  if (game === null || position.game_over !== null) {
    return false;
  }
  return (position.turn === 'white' ? game.white : game.black).accepts_moves;
}

export function App() {
  const { view, connected, error, movePending, submitMove } = useGameChannel();

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
              <Board
                snapshot={view.position}
                interactive={connected && !movePending && yourTurn(view.game, view.position)}
                onMove={submitMove}
              />
            </div>
            <aside className="side-panel">
              {!connected && <p role="alert">Lost the connection to the server. Reconnecting…</p>}
              {connected && error !== null && <p role="alert">{error}</p>}
              {view.game !== null && (
                <dl className="players">
                  <dt>White</dt>
                  <dd>{view.game.white.name}</dd>
                  <dt>Black</dt>
                  <dd>{view.game.black.name}</dd>
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
