import type { Color, GameState } from './api';
import type { Orientation } from './board/geometry';
import './GameControls.css';

/** The sides a viewer may resign: the ones whose moves this browser plays. */
function resignable(game: GameState | null): Color[] {
  if (game === null) {
    return [];
  }
  const sides: Color[] = [];
  if (game.white.accepts_moves) sides.push('white');
  if (game.black.accepts_moves) sides.push('black');
  return sides;
}

interface GameControlsProps {
  /** Which side is at the bottom of the board. */
  orientation: Orientation;
  onFlip: () => void;
  /** The game that can still be ended, or null when there is none to act on. */
  game: GameState | null;
  /** Whether the server is out of reach, so nothing can be asked of the game. */
  disabled: boolean;
  onResign: (color: Color) => void;
  onAbort: () => void;
  onTakeBack: () => void;
}

/**
 * Turning the board round, which is this browser's business alone, and what changes the
 * game itself, which is everyone's: a takeback, a resignation and an abort all reach
 * every viewer.
 */
export function GameControls({
  orientation,
  onFlip,
  game,
  disabled,
  onResign,
  onAbort,
  onTakeBack,
}: GameControlsProps) {
  const sides = resignable(game);
  return (
    <div className="game-controls">
      <button type="button" onClick={onFlip}>
        {orientation === 'white' ? 'Flip to Black' : 'Flip to White'}
      </button>
      {/* Only a game somebody is playing has a move to give back to them. */}
      {sides.length > 0 && (
        <button
          type="button"
          disabled={disabled || game === null || game.moves.length === 0}
          onClick={onTakeBack}
        >
          Take back
        </button>
      )}
      {sides.map((color) => (
        <button key={color} type="button" disabled={disabled} onClick={() => onResign(color)}>
          {/* One side to resign needs no saying which; two do. */}
          {sides.length === 1 ? 'Resign' : `${color === 'white' ? 'White' : 'Black'} resigns`}
        </button>
      ))}
      {game !== null && (
        <button type="button" disabled={disabled} onClick={onAbort}>
          Abort
        </button>
      )}
    </div>
  );
}
