/** The squares the board highlights as the last move, in alphabetical order. */
export function lastMoveSquares(container: HTMLElement): string[] {
  return Array.from(
    container.querySelectorAll('.last-move'),
    (rect) => rect.getAttribute('data-square') ?? '',
  ).sort();
}

/**
 * The legal destinations the board is showing, as "square kind" ("d6 en-passant"), in
 * alphabetical order.
 */
export function destinations(container: HTMLElement): string[] {
  return Array.from(
    container.querySelectorAll('.destination'),
    (marker) => `${marker.getAttribute('data-square')} ${marker.getAttribute('data-kind')}`,
  ).sort();
}

/** The square whose legal moves the board is showing, if any. */
export function originSquare(container: HTMLElement): string | null {
  return container.querySelector('.origin')?.getAttribute('data-square') ?? null;
}

/** The hit target of a square, which takes the hovering and the clicks. */
export function square(container: HTMLElement, name: string): Element {
  const target = container.querySelector(`.interaction[data-square="${name}"]`);
  if (target === null) {
    throw new Error(`the board has no square ${name}`);
  }
  return target;
}
