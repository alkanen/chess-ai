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

/** The square of the king the board is highlighting as in check, if any. */
export function checkSquare(container: HTMLElement): string | null {
  return container.querySelector('.check')?.getAttribute('data-square') ?? null;
}

/** Where a square's hit target sits on the board, as "x,y" in SVG user units. */
export function squareAt(container: HTMLElement, name: string): string {
  const target = square(container, name);
  return `${target.getAttribute('x')},${target.getAttribute('y')}`;
}

/** The square whose legal moves the board is showing, if any. */
export function originSquare(container: HTMLElement): string | null {
  return container.querySelector('.origin')?.getAttribute('data-square') ?? null;
}

/** The pieces the promotion picker is offering, in the order it shows them. */
export function promotionChoices(container: HTMLElement): string[] {
  return Array.from(
    container.querySelectorAll('.promotion-choice'),
    (choice) => choice.getAttribute('data-piece') ?? '',
  );
}

/** One piece on offer in the promotion picker. */
export function promotionChoice(container: HTMLElement, piece: string): Element {
  const choice = container.querySelector(`.promotion-choice[data-piece="${piece}"]`);
  if (choice === null) {
    throw new Error(`the promotion picker is not offering a ${piece}`);
  }
  return choice;
}

/** Everything around the promotion picker, clicking which calls the move off. */
export function promotionBackdrop(container: HTMLElement): Element {
  const backdrop = container.querySelector('.promotion-backdrop');
  if (backdrop === null) {
    throw new Error('the promotion picker is not open');
  }
  return backdrop;
}

/** The hit target of a square, which takes the hovering and the clicks. */
export function square(container: HTMLElement, name: string): Element {
  const target = container.querySelector(`.interaction[data-square="${name}"]`);
  if (target === null) {
    throw new Error(`the board has no square ${name}`);
  }
  return target;
}
