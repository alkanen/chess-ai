import { useEffect, useRef } from 'react';
import './MoveList.css';

/** A move as the list shows it: its notation, and how far into the game it is. */
interface Played {
  san: string;
  /** How many moves have been played once this one has, which is what it jumps to. */
  ply: number;
}

/** The moves of one full move: White's, and Black's, either of which may be missing. */
interface MovePair {
  number: number;
  white: Played | null;
  black: Played | null;
}

/** Where the numbering starts, read off the position the game began in. */
function firstMove(startFen: string): { number: number; black: boolean } {
  const [, turn, , , , fullmove] = startFen.split(' ');
  const number = Number(fullmove);
  return { number: Number.isFinite(number) && number > 0 ? number : 1, black: turn === 'b' };
}

/**
 * The moves under the numbers they were played at. A game from a custom position starts
 * wherever that position stood, and can start with a move of Black's.
 */
function pairs(moves: readonly { san: string }[], startFen: string): MovePair[] {
  const first = firstMove(startFen);
  const paired: MovePair[] = [];
  moves.forEach(({ san }, index) => {
    const played = { san, ply: index + 1 };
    // Counting from White's move of the first full move, even when it was never played.
    const ply = index + (first.black ? 1 : 0);
    const number = first.number + Math.floor(ply / 2);
    if (ply % 2 === 0) {
      paired.push({ number, white: played, black: null });
    } else {
      const pair = paired.at(-1);
      if (pair?.number === number) {
        pair.black = played;
      } else {
        paired.push({ number, white: null, black: played });
      }
    }
  });
  return paired;
}

interface MoveListProps {
  /** The moves in the order they were played; the list shows their notation alone. */
  moves: readonly { san: string }[];
  /** The position the game began in, which says how the moves are numbered. */
  startFen: string;
  /**
   * How many moves are on the board, for a game being stepped through: the move that
   * many in is the one the board is showing, and 0 is the position it started from.
   */
  current?: number;
  /** Jumps to the position a move leads to. Without it the moves are not for clicking. */
  onSelect?: (ply: number) => void;
}

/** The game so far in standard algebraic notation, a numbered move to a line. */
export function MoveList({ moves, startFen, current, onSelect }: MoveListProps) {
  const scroller = useRef<HTMLDivElement>(null);

  // A long game outgrows the panel: keep the move being looked at in sight. Watched on
  // how many moves there are rather than on the array holding them, because a game that
  // arrives again unchanged — as one does whenever a dropped connection is caught up —
  // is not a reason to drag the reader away from the move they had scrolled to.
  useEffect(() => {
    const list = scroller.current;
    if (list === null) {
      return;
    }
    const shown = list.querySelector('[aria-current]');
    if (shown instanceof HTMLElement) {
      // Measured against the scroller itself rather than by offsetTop, which is counted
      // from whichever ancestor happens to be positioned, and here is none of them.
      const move = shown.getBoundingClientRect();
      const box = list.getBoundingClientRect();
      list.scrollTop += move.top - box.top - (list.clientHeight - move.height) / 2;
    } else if (current === undefined) {
      // A game being played: the end of the list is the move everyone is waiting on.
      list.scrollTop = list.scrollHeight;
    } else {
      // A game being stepped through, standing where it began: no move is on the board,
      // and the end of the game is the one place the reader is certainly not looking.
      list.scrollTop = 0;
    }
  }, [moves.length, current]);

  const lines = pairs(moves, startFen);
  return (
    <section className="move-list" aria-labelledby="move-list-heading">
      <h2 id="move-list-heading">Moves</h2>
      {lines.length === 0 ? (
        <p className="note">No moves yet.</p>
      ) : (
        <div className="moves-scroller" ref={scroller}>
          {/* The numbers are the list markers, so the list starts where the game did. */}
          <ol className="moves" start={lines[0].number}>
            {lines.map(({ number, white, black }) => (
              <li key={number}>
                {/* A game that opens with Black leaves White's place on the line empty. */}
                <Move played={white} current={current} onSelect={onSelect} />
                {black !== null && <Move played={black} current={current} onSelect={onSelect} />}
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

/** One move: something to click when the game is being stepped through, or nothing. */
function Move({
  played,
  current,
  onSelect,
}: {
  played: Played | null;
  current: number | undefined;
  onSelect: ((ply: number) => void) | undefined;
}) {
  if (played === null) {
    return <span className="san">…</span>;
  }
  if (onSelect === undefined) {
    return <span className="san">{played.san}</span>;
  }
  return (
    <button
      type="button"
      className="san"
      // "step" is what this is: one position of a game being stepped through.
      aria-current={played.ply === current ? 'step' : undefined}
      onClick={() => onSelect(played.ply)}
    >
      {played.san}
    </button>
  );
}
