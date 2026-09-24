import { useEffect, useRef } from 'react';
import type { MoveRecord } from './api';
import './MoveList.css';

/** The moves of one full move: White's, and Black's, either of which may be missing. */
interface MovePair {
  number: number;
  white: MoveRecord | null;
  black: MoveRecord | null;
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
function pairs(moves: MoveRecord[], startFen: string): MovePair[] {
  const first = firstMove(startFen);
  const paired: MovePair[] = [];
  moves.forEach((move, index) => {
    // Counting from White's move of the first full move, even when it was never played.
    const ply = index + (first.black ? 1 : 0);
    const number = first.number + Math.floor(ply / 2);
    if (ply % 2 === 0) {
      paired.push({ number, white: move, black: null });
    } else {
      const pair = paired.at(-1);
      if (pair?.number === number) {
        pair.black = move;
      } else {
        paired.push({ number, white: null, black: move });
      }
    }
  });
  return paired;
}

interface MoveListProps {
  moves: MoveRecord[];
  /** The position the game began in, which says how the moves are numbered. */
  startFen: string;
}

/** The game so far in standard algebraic notation, a numbered move to a line. */
export function MoveList({ moves, startFen }: MoveListProps) {
  const scroller = useRef<HTMLDivElement>(null);

  // A long game outgrows the panel, and it is the end of it people are watching.
  useEffect(() => {
    const list = scroller.current;
    if (list !== null) {
      list.scrollTop = list.scrollHeight;
    }
  }, [moves.length]);

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
                <span className="san">{white?.san ?? '…'}</span>
                {black !== null && <span className="san">{black.san}</span>}
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
