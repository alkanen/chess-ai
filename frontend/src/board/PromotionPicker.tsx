import { useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import type { Color, LegalMove, PieceType } from '../api';
import { BOARD, SQUARE, squarePosition, type Orientation } from './geometry';
import { pieceImage } from './pieces';
import './PromotionPicker.css';

/** The pieces on offer, in the order they are shown: the likeliest choice first. */
const CHOICES: PieceType[] = ['queen', 'rook', 'bishop', 'knight'];

interface PromotionPickerProps {
  /** The promotion moves of one pawn onto one square, one per piece it may become. */
  moves: LegalMove[];
  /** The colour of the promoting pawn, so the choices are drawn in its pieces. */
  color: Color;
  /** Which side is at the bottom of the board, so the stack runs the right way. */
  orientation?: Orientation;
  onChoose: (move: LegalMove) => void;
  /** Called when the picker is dismissed, leaving the pawn where it was. */
  onCancel: () => void;
}

/**
 * The pieces a promoting pawn may become, stacked on the promotion square's file and
 * running into the board, over a backdrop that calls the move off when it is clicked.
 */
export function PromotionPicker({
  moves,
  color,
  orientation = 'white',
  onChoose,
  onCancel,
}: PromotionPickerProps) {
  const first = useRef<SVGGElement>(null);
  // The picker has the board to itself, so it takes the keyboard too: it opens on the
  // queen, Enter or Space takes the choice the keyboard is on, and Escape (handled by
  // the board, which owns the picker) calls the promotion off.
  useEffect(() => first.current?.focus(), []);

  const byPiece = new Map(moves.map((move) => [move.promotion, move]));
  const choices = CHOICES.flatMap((piece) => {
    const move = byPiece.get(piece);
    return move === undefined ? [] : [{ piece, move }];
  });
  const { x, y } = squarePosition(moves[0].to_square, orientation);
  // The stack hangs down from the top of the board and up from the bottom, whichever
  // way round it is, so it always fits.
  const step = y === 0 ? SQUARE : -SQUARE;

  return (
    <g className="promotion-picker" role="group" aria-label="Promotion">
      <rect
        className="promotion-backdrop"
        x={0}
        y={0}
        width={BOARD}
        height={BOARD}
        onMouseDown={onCancel}
      />
      {choices.map(({ piece, move }, index) => {
        const top = y + index * step;
        const take = (event: ReactKeyboardEvent) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault(); // Space would scroll the page.
            onChoose(move);
          }
        };
        return (
          <g
            key={piece}
            ref={index === 0 ? first : undefined}
            className="promotion-choice"
            data-piece={piece}
            role="button"
            tabIndex={0}
            aria-label={`Promote to ${piece}`}
            // The board treats the other buttons as calling the move off, not making
            // one, and a choice is no different: a right-drag is a coming annotation
            // arrow, not a promotion.
            onMouseDown={(event) => (event.button === 0 ? onChoose(move) : onCancel())}
            onKeyDown={take}
          >
            <rect className="choice-square" x={x} y={top} width={SQUARE} height={SQUARE} />
            <image
              href={pieceImage({ color, type: piece })}
              x={x}
              y={top}
              width={SQUARE}
              height={SQUARE}
            />
          </g>
        );
      })}
    </g>
  );
}
