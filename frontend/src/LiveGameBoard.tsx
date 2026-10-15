import { useId, useState } from 'react';
import type { Color, LiveGame } from './api';
import { Board } from './board/Board';
import { Player } from './GameView';
import { GameStatus } from './GameStatus';
import { MoveList } from './MoveList';
import { candidateArrows, shownThoughts, whiteOdds } from './thoughts';
import { EvalBar, ThoughtsPanel, useShowThoughts } from './ThoughtsPanel';
// The board and the panel beside it are laid out as the game view's are.
import './App.css';
import './LiveGameBoard.css';

function opposite(side: Color): Color {
  return side === 'white' ? 'black' : 'white';
}

interface LiveGameBoardProps {
  live: LiveGame;
  /** What the game is, among the games its checkpoint plays, as its heading says. */
  caption: string;
}

/**
 * A game the evaluator is playing, move by move as it plays it, with what the model thought of
 * its last move when the viewer asks for that: a sample game, or a game of the ladder.
 */
export function LiveGameBoard({ live, caption }: LiveGameBoardProps) {
  const id = useId();
  const { game } = live;
  // From the checkpoint's side, which against itself is both: White, as usual.
  const modelSide: Color = game.white.model !== null ? 'white' : 'black';
  // Kept as a flip rather than a side, so that it holds from one game to the next whichever
  // side the checkpoint plays in it.
  const [flipped, setFlipped] = useState(false);
  const orientation = flipped ? opposite(modelSide) : modelSide;
  const [showThoughts, setShowThoughts] = useShowThoughts();
  const shown = showThoughts ? shownThoughts(game) : null;
  const board = (
    <div className="board-frame">
      <Board
        snapshot={game.position}
        orientation={orientation}
        arrows={shown === null ? [] : candidateArrows(shown)}
      />
    </div>
  );
  return (
    <section className="live-game" aria-labelledby={`${id}-heading`}>
      <h4 id={`${id}-heading`}>
        <span className="live-badge">Live</span> {caption}
      </h4>
      <GameStatus position={game.position} inGame />
      <div className="game-view">
        {showThoughts ? (
          <div className="board-row">
            <EvalBar
              odds={
                shown?.thoughts.wdl != null ? whiteOdds(shown.thoughts.wdl, shown.side) : null
              }
              orientation={orientation}
            />
            {board}
          </div>
        ) : (
          board
        )}
        <aside className="side-panel">
          <dl className="players">
            <dt>White</dt>
            <Player player={game.white} />
            <dt>Black</dt>
            <Player player={game.black} />
          </dl>
          <p className="flip">
            <button type="button" onClick={() => setFlipped((was) => !was)}>
              {orientation === 'white' ? 'Flip to Black' : 'Flip to White'}
            </button>
          </p>
          <ThoughtsPanel
            game={game}
            shown={shown}
            show={showThoughts}
            onShow={setShowThoughts}
            keepRoom
          />
          <MoveList moves={game.moves} startFen={game.start_fen} />
        </aside>
      </div>
    </section>
  );
}
