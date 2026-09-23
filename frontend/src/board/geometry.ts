/** Where things sit on the board, in the SVG user units the board is drawn in. */

import type { Color } from '../api';

export const FILES = 'abcdefgh';
/** Side of one square in SVG user units. */
export const SQUARE = 100;
export const BOARD = 8 * SQUARE;

/**
 * Which side of the board is at the bottom of the screen. Everything else about a
 * position is the same whichever way round it is drawn, so this is the only place the
 * two orientations differ.
 */
export type Orientation = Color;

export interface SquarePosition {
  x: number;
  y: number;
}

export const SQUARES = Array.from({ length: 64 }, (_, index) => {
  const file = index % 8;
  const rank = Math.floor(index / 8);
  return { name: `${FILES[file]}${rank + 1}`, file, rank, light: (file + rank) % 2 === 1 };
});

/** Top-left corner of a square, with the given side at the bottom. */
export function squarePosition(square: string, orientation: Orientation = 'white'): SquarePosition {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]) - 1;
  return orientation === 'white'
    ? { x: file * SQUARE, y: (7 - rank) * SQUARE }
    : { x: (7 - file) * SQUARE, y: rank * SQUARE };
}

export function squareCentre(square: string, orientation: Orientation = 'white'): SquarePosition {
  const { x, y } = squarePosition(square, orientation);
  return { x: x + SQUARE / 2, y: y + SQUARE / 2 };
}

/**
 * Which coordinates a square carries: the rank number down the leftmost column and the
 * file letter along the bottom row, whichever squares those are once the board has been
 * turned round.
 */
export function coordinatesOn(
  { file, rank }: { file: number; rank: number },
  orientation: Orientation = 'white',
): { rankNumber: boolean; fileLetter: boolean } {
  return orientation === 'white'
    ? { rankNumber: file === 0, fileLetter: rank === 0 }
    : { rankNumber: file === 7, fileLetter: rank === 7 };
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
