import { useState } from 'react';
import type { GameState, PlayerInfo, PositionSnapshot } from './api';
import { Board } from './board/Board';
import type { Orientation } from './board/geometry';
import { GameControls } from './GameControls';
import { GameStatus } from './GameStatus';
import { MoveList } from './MoveList';
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

/**
 * The side of the board a viewer belongs on: the one they play, if the game gives them
 * exactly one. Watching, or playing both sides, leaves them behind White as usual.
 */
function playersSide(game: GameState | null): Orientation {
  if (game !== null && game.black.accepts_moves && !game.white.accepts_moves) {
    return 'black';
  }
  return 'white';
}

/**
 * Which side is at the bottom of the board, and the flip that turns it round.
 *
 * The board faces the side you are given to play, and stays wherever you last put it
 * until a game hands you the other colour.
 */
function useOrientation(game: GameState | null): [Orientation, () => void] {
  const facing = playersSide(game);
  const [orientation, setOrientation] = useState<Orientation>(facing);
  const [shown, setShown] = useState<Orientation>(facing);
  if (shown !== facing) {
    setShown(facing);
    setOrientation(facing);
  }
  const flip = () => setOrientation((side) => (side === 'white' ? 'black' : 'white'));
  return [orientation, flip];
}

/**
 * How a checkpoint was asked to play, for a side a checkpoint is playing.
 *
 * The name says which run and which step, and this says the rest: two sides of a game can be
 * the same checkpoint at two ratings, and nothing else on the page would tell them apart.
 */
function describeModel(player: PlayerInfo): string | null {
  const model = player.model;
  if (model === null) {
    return null;
  }
  const rating = model.rating !== null ? `plays like ${model.rating}` : 'no rating';
  const choosing =
    model.strategy === 'sample' ? `samples at ${model.temperature}` : 'plays its best move';
  return `${rating}, ${choosing}`;
}

/** One side of the game: who is playing it, and how they were asked to. */
function Player({ player }: { player: PlayerInfo }) {
  const model = describeModel(player);
  return (
    <dd>
      {player.name}
      {model !== null && <span className="model-note">{model}</span>}
    </dd>
  );
}

/** The game the server is playing: the board it is played on, and what is asked of it. */
export function GameView() {
  const { view, connected, error, movePending, submitMove, resign, abort, takeBack } =
    useGameChannel();
  const [orientation, flip] = useOrientation(view?.game ?? null);

  // A game that has ended is still there to look at, but there is nothing left to end.
  const unfinished = view?.position.game_over === null ? view.game : null;

  if (view === null) {
    return <p>Connecting to the server…</p>;
  }
  return (
    <>
      <GameStatus position={view.position} inGame={view.game !== null} />
      <div className="game-view">
        <div className="board-frame">
          <Board
            snapshot={view.position}
            orientation={orientation}
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
              <Player player={view.game.white} />
              <dt>Black</dt>
              <Player player={view.game.black} />
            </dl>
          )}
          <GameControls
            orientation={orientation}
            onFlip={flip}
            pgnGame={view.game?.id ?? null}
            game={unfinished}
            disabled={!connected}
            onResign={resign}
            onAbort={abort}
            onTakeBack={takeBack}
          />
          {view.game !== null && (
            <MoveList moves={view.game.moves} startFen={view.game.start_fen} />
          )}
          <NewGameForm />
        </aside>
      </div>
    </>
  );
}
