import type { PositionSnapshot } from '../api';
import { pieceImage } from './pieces';
import './Board.css';

const FILES = 'abcdefgh';
/** Side of one square in SVG user units. */
const SQUARE = 100;

interface SquarePosition {
  x: number;
  y: number;
}

/** Top-left corner of a square, with White at the bottom. */
function squarePosition(square: string): SquarePosition {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]) - 1;
  return { x: file * SQUARE, y: (7 - rank) * SQUARE };
}

const SQUARES = Array.from({ length: 64 }, (_, index) => {
  const file = index % 8;
  const rank = Math.floor(index / 8);
  return { name: `${FILES[file]}${rank + 1}`, file, rank, light: (file + rank) % 2 === 1 };
});

interface BoardProps {
  snapshot: PositionSnapshot;
}

/** The chessboard as an SVG that scales to the width of its container. */
export function Board({ snapshot }: BoardProps) {
  return (
    <svg className="board" viewBox={`0 0 ${8 * SQUARE} ${8 * SQUARE}`} aria-label="Chessboard">
      {SQUARES.map(({ name, file, rank, light }) => {
        const { x, y } = squarePosition(name);
        const shade = light ? 'light' : 'dark';
        return (
          <g key={name}>
            <rect className={`square ${shade}`} x={x} y={y} width={SQUARE} height={SQUARE} />
            {file === 0 && (
              <text className={`coordinate on-${shade}`} x={x + 4} y={y + 4} dominantBaseline="hanging">
                {rank + 1}
              </text>
            )}
            {rank === 0 && (
              <text
                className={`coordinate on-${shade}`}
                x={x + SQUARE - 4}
                y={y + SQUARE - 4}
                textAnchor="end"
              >
                {FILES[file]}
              </text>
            )}
          </g>
        );
      })}
      {Object.entries(snapshot.pieces).map(([square, piece]) => {
        const { x, y } = squarePosition(square);
        return (
          <image
            key={square}
            href={pieceImage(piece)}
            x={x}
            y={y}
            width={SQUARE}
            height={SQUARE}
            role="img"
            aria-label={`${piece.color} ${piece.type} on ${square}`}
          />
        );
      })}
    </svg>
  );
}
