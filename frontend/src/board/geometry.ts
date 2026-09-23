/** Where things sit on the board, in the SVG user units the board is drawn in. */

export const FILES = 'abcdefgh';
/** Side of one square in SVG user units. */
export const SQUARE = 100;
export const BOARD = 8 * SQUARE;

export interface SquarePosition {
  x: number;
  y: number;
}

export const SQUARES = Array.from({ length: 64 }, (_, index) => {
  const file = index % 8;
  const rank = Math.floor(index / 8);
  return { name: `${FILES[file]}${rank + 1}`, file, rank, light: (file + rank) % 2 === 1 };
});

/** Top-left corner of a square, with White at the bottom. */
export function squarePosition(square: string): SquarePosition {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]) - 1;
  return { x: file * SQUARE, y: (7 - rank) * SQUARE };
}

export function squareCentre(square: string): SquarePosition {
  const { x, y } = squarePosition(square);
  return { x: x + SQUARE / 2, y: y + SQUARE / 2 };
}

/** Where a mouse event is on the board, or null when the board has no size to measure. */
export function boardPoint(
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
