import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import type { LegalMove, Piece, PositionSnapshot } from '../api';
import {
  BOARD,
  FILES,
  SQUARE,
  SQUARES,
  boardPoint,
  coordinatesOn,
  squareCentre,
  squarePosition,
  type Orientation,
} from './geometry';
import { pieceImage } from './pieces';
import { PromotionPicker } from './PromotionPicker';
import './Board.css';

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

interface BoardProps {
  snapshot: PositionSnapshot;
  /** Which side is at the bottom of the board. */
  orientation?: Orientation;
  /** Whether the side to move is played from this browser, so its moves can be made. */
  interactive?: boolean;
  /** Submits a move in UCI. The server decides whether it is played. */
  onMove?: (uci: string) => void;
}

/**
 * The chessboard as an SVG that scales to the width of its container.
 *
 * On an interactive turn, hovering a piece shows the legal destinations the snapshot
 * came with, and a move is made by clicking origin and destination or by dragging. A
 * move onto the last rank waits for the promotion picker before it is submitted.
 */
export function Board({
  snapshot,
  orientation = 'white',
  interactive = false,
  onMove,
}: BoardProps) {
  const svg = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  /** The promotions on offer while the picker is open: one move per piece. */
  const [promoting, setPromoting] = useState<LegalMove[] | null>(null);

  // A new position, or the turn passing to someone else, calls off any move being made.
  useEffect(() => {
    setSelected(null);
    setDrag(null);
    setPromoting(null);
  }, [snapshot, interactive]);

  // Letting go anywhere outside the board drops the piece back where it came from.
  useEffect(() => {
    const drop = () => setDrag(null);
    window.addEventListener('mouseup', drop);
    return () => window.removeEventListener('mouseup', drop);
  }, []);

  // Escape calls off a promotion, as a click outside the picker does.
  useEffect(() => {
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setPromoting(null);
      }
    };
    window.addEventListener('keydown', dismiss);
    return () => window.removeEventListener('keydown', dismiss);
  }, []);

  const legalMoves = interactive ? snapshot.legal_moves : {};
  // The picker covers the board, so nothing underneath is highlighted while it is open.
  const origin = promoting !== null ? null : (drag?.from ?? selected ?? hovered);
  const destinations = perDestination(origin === null ? [] : (legalMoves[origin] ?? []));

  /**
   * Make the move between two squares, if there is one: submit it, or ask which piece
   * the pawn becomes. Returns whether the two squares are a move at all.
   */
  function makeMove(from: string, to: string): boolean {
    const candidates = (legalMoves[from] ?? []).filter((move) => move.to_square === to);
    if (candidates.length === 0) {
      return false;
    }
    setSelected(null);
    setDrag(null);
    // The promotions of one pawn onto one square differ only in the piece it becomes.
    if (candidates[0].promotion !== null) {
      setPromoting(candidates);
    } else {
      onMove?.(candidates[0].uci);
    }
    return true;
  }

  function promote(chosen: LegalMove) {
    setPromoting(null);
    onMove?.(chosen.uci);
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
    if (selected !== null && makeMove(selected, square)) {
      return;
    }
    if (legalMoves[square] === undefined) {
      setSelected(null);
      return;
    }
    setSelected(square);
    setDrag({
      from: square,
      ...(boardPoint(svg.current, event) ?? squareCentre(square, orientation)),
    });
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
    if (!makeMove(from, square)) {
      setSelected(null);
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
      : squarePosition(square, orientation);
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

  const { last_move: lastMove, check_square: checkSquare } = snapshot;
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
      <defs>
        {/* A glow around the king, fading out before it reaches the neighbouring squares. */}
        <radialGradient id="check-glow">
          <stop offset="0%" stopColor="#ff3b30" stopOpacity="0.95" />
          <stop offset="40%" stopColor="#e02419" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#9e0000" stopOpacity="0" />
        </radialGradient>
      </defs>
      {SQUARES.map(({ name, light }) => {
        const { x, y } = squarePosition(name, orientation);
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
        const { x, y } = squarePosition(square, orientation);
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
      {checkSquare !== null && (
        <rect
          className="check"
          data-square={checkSquare}
          {...squarePosition(checkSquare, orientation)}
          width={SQUARE}
          height={SQUARE}
        />
      )}
      {origin !== null && destinations.length > 0 && (
        <rect
          className="origin"
          data-square={origin}
          {...squarePosition(origin, orientation)}
          width={SQUARE}
          height={SQUARE}
        />
      )}
      {SQUARES.map(({ name, file, rank, light }) => {
        const { rankNumber, fileLetter } = coordinatesOn({ file, rank }, orientation);
        if (!rankNumber && !fileLetter) {
          return null;
        }
        const { x, y } = squarePosition(name, orientation);
        const shade = light ? 'light' : 'dark';
        return (
          <g key={`coordinates-${name}`}>
            {rankNumber && (
              <text
                className={`coordinate on-${shade}`}
                x={x + 4}
                y={y + 4}
                dominantBaseline="hanging"
              >
                {rank + 1}
              </text>
            )}
            {fileLetter && (
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
      {pieces
        .filter(([square]) => drag?.from !== square)
        .map(([square, piece]) => drawPiece(square, piece))}
      {/* Above the pieces, so that a capture is drawn around the piece it takes. */}
      {destinations.map((move) => {
        const kind = destinationKind(move);
        const { x, y } = squareCentre(move.to_square, orientation);
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
        const { x, y } = squarePosition(name, orientation);
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
      {/* Above the hit targets, so the board takes nothing while the picker is open. */}
      {promoting !== null && (
        <PromotionPicker
          moves={promoting}
          color={snapshot.turn}
          orientation={orientation}
          onChoose={promote}
          onCancel={() => setPromoting(null)}
        />
      )}
    </svg>
  );
}
