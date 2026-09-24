import type { Orientation } from './board/geometry';
import './ReplayControls.css';

/** Where each button goes, given where the game is now and how long it is. */
const STEPS = [
  { label: 'Start', symbol: '⏮', to: () => 0 },
  { label: 'Back', symbol: '◀', to: (at: number) => at - 1 },
  { label: 'Forward', symbol: '▶', to: (at: number) => at + 1 },
  { label: 'End', symbol: '⏭', to: (_at: number, plies: number) => plies },
] as const;

interface ReplayControlsProps {
  /** How many moves have been played in the position on the board. */
  ply: number;
  /** How many moves the game has, which is as far as it goes. */
  plies: number;
  /** Shows the position after ``ply`` moves. */
  onGo: (ply: number) => void;
  /** Which side is at the bottom of the board. */
  orientation: Orientation;
  onFlip: () => void;
}

/** Stepping through the game, and turning the board round to follow it from either side. */
export function ReplayControls({ ply, plies, onGo, orientation, onFlip }: ReplayControlsProps) {
  return (
    <div className="replay-controls">
      <div className="steps">
        {STEPS.map(({ label, symbol, to }) => {
          // A step that would leave the game stops at its end, and there it is spent.
          const target = Math.min(Math.max(to(ply, plies), 0), plies);
          return (
            <button
              key={label}
              type="button"
              aria-label={label}
              disabled={target === ply}
              onClick={() => onGo(target)}
            >
              <span aria-hidden="true">{symbol}</span>
            </button>
          );
        })}
      </div>
      {/* Not announced as it changes: the turn indicator above the board already is,
          and two things talking over each other is worse than one. */}
      <p className="ply">
        Move {ply} of {plies} <span className="hint">(arrow keys)</span>
      </p>
      <button type="button" onClick={onFlip}>
        {orientation === 'white' ? 'Flip to Black' : 'Flip to White'}
      </button>
    </div>
  );
}
