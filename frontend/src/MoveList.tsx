import { useEffect, useRef } from 'react';
import type { MoveRecord } from './api';
import './MoveList.css';

/** The moves of one full move: White's, and Black's reply if it has been played. */
interface MovePair {
  number: number;
  white: MoveRecord;
  black: MoveRecord | null;
}

function pairs(moves: MoveRecord[]): MovePair[] {
  const paired: MovePair[] = [];
  for (let ply = 0; ply < moves.length; ply += 2) {
    paired.push({ number: ply / 2 + 1, white: moves[ply], black: moves[ply + 1] ?? null });
  }
  return paired;
}

interface MoveListProps {
  moves: MoveRecord[];
}

/** The game so far in standard algebraic notation, a numbered move to a line. */
export function MoveList({ moves }: MoveListProps) {
  const scroller = useRef<HTMLDivElement>(null);

  // A long game outgrows the panel, and it is the end of it people are watching.
  useEffect(() => {
    const list = scroller.current;
    if (list !== null) {
      list.scrollTop = list.scrollHeight;
    }
  }, [moves.length]);

  return (
    <section className="move-list" aria-labelledby="move-list-heading">
      <h2 id="move-list-heading">Moves</h2>
      {moves.length === 0 ? (
        <p className="note">No moves yet.</p>
      ) : (
        <div className="moves-scroller" ref={scroller}>
          <ol className="moves">
            {pairs(moves).map(({ number, white, black }) => (
              <li key={number}>
                <span className="san">{white.san}</span>
                {black !== null && <span className="san">{black.san}</span>}
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
