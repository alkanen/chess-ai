import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import type { LegalMove, Piece, PositionSnapshot } from '../api';
import { pieceImage } from './pieces';
import './Board.css';

const FILES = 'abcdefgh';
/** Side of one square in SVG user units. */
const SQUARE = 100;
const BOARD = 8 * SQUARE;

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

/** How a legal destination is drawn, so that the special moves stand out. */
export type DestinationKind = 'quiet' | 'capture' | 'castling' | 'en-passant';

const DESTINATION_RADIUS: Record<DestinationKind, number> = {
  quiet: 15,
  capture: 42,
  castling: 27,
  'en-passant': 42,
};

function destinationKind(move: LegalMove): DestinationKind {
  if (move.en_passant) return 'en-passant';
  if (move.capture) return 'capture';
  if (move.castling) return 'castling';
  return 'quiet';
}

/** One move per destination square, since the four promotion choices share a square. */
function perDestination(moves: LegalMove[]): LegalMove[] {
  const first = new Map<string, LegalMove>();
  for (const move of moves) {
    if (!first.has(move.to_square)) {
      first.set(move.to_square, move);
    }
  }
  return [...first.values()];
}

/** A piece being dragged, and where it is being held, in SVG user units. */
interface Drag {
  from: string;
  x: number;
  y: number;
}

/** Where a mouse event is on the board, or null when the board has no size to measure. */
function boardPoint(
  svg: SVGSVGElement | null,
  event: { clientX: number; clientY: number },
): SquarePosition | null {
  const box = svg?.getBoundingClientRect();
  if (box === undefined || box.width === 0 || box.height === 0) {
    return null;
  }
  return {
    x: ((event.clientX - box.left) / box.width) * BOARD,
    y: ((event.clientY - box.top) / box.height) * BOARD,
  };
}

function squareCentre(square: string): SquarePosition {
  const { x, y } = squarePosition(square);
  return { x: x + SQUARE / 2, y: y + SQUARE / 2 };
}

interface BoardProps {
  snapshot: PositionSnapshot;
  /** Whether the side to move is played from this browser, so its moves can be made. */
  interactive?: boolean;
  /** Submits a move in UCI. The server decides whether it is played. */
  onMove?: (uci: string) => void;
}

/**
 * The chessboard as an SVG that scales to the width of its container.
 *
 * On an interactive turn, hovering a piece shows the legal destinations the snapshot
 * came with, and a move is made by clicking origin and destination or by dragging.
 */
export function Board({ snapshot, interactive = false, onMove }: BoardProps) {
  const svg = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);

  // A new position, or the turn passing to someone else, calls off any move being made.
  useEffect(() => {
    setSelected(null);
    setDrag(null);
  }, [snapshot, interactive]);

  // Letting go anywhere outside the board drops the piece back where it came from.
  useEffect(() => {
    const drop = () => setDrag(null);
    window.addEventListener('mouseup', drop);
    return () => window.removeEventListener('mouseup', drop);
  }, []);

  const legalMoves = interactive ? snapshot.legal_moves : {};
  const origin = drag?.from ?? selected ?? hovered;
  const destinations = perDestination(origin === null ? [] : (legalMoves[origin] ?? []));

  /** The move between two squares, auto-queening until the promotion picker lands. */
  function moveBetween(from: string, to: string): LegalMove | undefined {
    const candidates = (legalMoves[from] ?? []).filter((move) => move.to_square === to);
    return candidates.find((move) => move.promotion === 'queen') ?? candidates[0];
  }

  function submit(move: LegalMove) {
    setSelected(null);
    setDrag(null);
    onMove?.(move.uci);
  }

  function press(square: string, event: ReactMouseEvent) {
    if (!interactive) {
      return;
    }
    if (event.button !== 0) {
      // The other buttons belong to the browser, and later to annotation arrows. By
      // convention they call off the move being made rather than making one.
      setSelected(null);
      setDrag(null);
      return;
    }
    const move = selected === null ? undefined : moveBetween(selected, square);
    if (move !== undefined) {
      submit(move);
      return;
    }
    if (legalMoves[square] === undefined) {
      setSelected(null);
      return;
    }
    setSelected(square);
    setDrag({ from: square, ...(boardPoint(svg.current, event) ?? squareCentre(square)) });
  }

  function release(square: string) {
    if (drag === null) {
      return;
    }
    const { from } = drag;
    setDrag(null);
    if (square === from) {
      return; // A click rather than a drag: the destination comes with the next click.
    }
    const move = moveBetween(from, square);
    if (move === undefined) {
      setSelected(null);
    } else {
      submit(move);
    }
  }

  function dragTo(event: ReactMouseEvent) {
    const point = drag === null ? null : boardPoint(svg.current, event);
    if (drag !== null && point !== null) {
      setDrag({ ...drag, ...point });
    }
  }

  function drawPiece(square: string, piece: Piece) {
    const dragged = drag?.from === square;
    const { x, y } = dragged
      ? { x: drag.x - SQUARE / 2, y: drag.y - SQUARE / 2 }
      : squarePosition(square);
    return (
      <image
        key={square}
        className={dragged ? 'piece dragged' : 'piece'}
        href={pieceImage(piece)}
        x={x}
        y={y}
        width={SQUARE}
        height={SQUARE}
        role="img"
        aria-label={`${piece.color} ${piece.type} on ${square}`}
      />
    );
  }

  const { last_move: lastMove } = snapshot;
  const lastMoveSquares = lastMove ? [lastMove.from_square, lastMove.to_square] : [];
  const pieces = Object.entries(snapshot.pieces);
  return (
    <svg
      ref={svg}
      className="board"
      viewBox={`0 0 ${BOARD} ${BOARD}`}
      aria-label="Chessboard"
      onMouseLeave={() => setHovered(null)}
      onMouseMove={dragTo}
    >
      {SQUARES.map(({ name, light }) => {
        const { x, y } = squarePosition(name);
        const shade = light ? 'light' : 'dark';
        return (
          <rect
            key={name}
            className={`square ${shade}`}
            x={x}
            y={y}
            width={SQUARE}
            height={SQUARE}
          />
        );
      })}
      {lastMoveSquares.map((square) => {
        const { x, y } = squarePosition(square);
        return (
          <rect
            key={`last-move-${square}`}
            className="last-move"
            data-square={square}
            x={x}
            y={y}
            width={SQUARE}
            height={SQUARE}
          />
        );
      })}
      {origin !== null && destinations.length > 0 && (
        <rect
          className="origin"
          data-square={origin}
          {...squarePosition(origin)}
          width={SQUARE}
          height={SQUARE}
        />
      )}
      {SQUARES.filter(({ file, rank }) => file === 0 || rank === 0).map(
        ({ name, file, rank, light }) => {
          const { x, y } = squarePosition(name);
          const shade = light ? 'light' : 'dark';
          return (
            <g key={`coordinates-${name}`}>
              {file === 0 && (
                <text
                  className={`coordinate on-${shade}`}
                  x={x + 4}
                  y={y + 4}
                  dominantBaseline="hanging"
                >
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
        },
      )}
      {pieces
        .filter(([square]) => drag?.from !== square)
        .map(([square, piece]) => drawPiece(square, piece))}
      {/* Above the pieces, so that a capture is drawn around the piece it takes. */}
      {destinations.map((move) => {
        const kind = destinationKind(move);
        const { x, y } = squareCentre(move.to_square);
        return (
          <circle
            key={`destination-${move.to_square}`}
            className={`destination ${kind}`}
            data-square={move.to_square}
            data-kind={kind}
            cx={x}
            cy={y}
            r={DESTINATION_RADIUS[kind]}
          />
        );
      })}
      {drag !== null &&
        snapshot.pieces[drag.from] !== undefined &&
        drawPiece(drag.from, snapshot.pieces[drag.from])}
      {SQUARES.map(({ name }) => {
        const { x, y } = squarePosition(name);
        const movable = legalMoves[name] !== undefined;
        return (
          <rect
            key={`interaction-${name}`}
            className={movable ? 'interaction movable' : 'interaction'}
            data-square={name}
            x={x}
            y={y}
            width={SQUARE}
            height={SQUARE}
            onMouseEnter={() => setHovered(name)}
            onMouseDown={(event) => press(name, event)}
            onMouseUp={() => release(name)}
          />
        );
      })}
    </svg>
  );
}
