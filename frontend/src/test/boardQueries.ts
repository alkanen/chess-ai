/** The squares the board highlights as the last move, in alphabetical order. */
export function lastMoveSquares(container: HTMLElement): string[] {
  return Array.from(
    container.querySelectorAll('.last-move'),
    (rect) => rect.getAttribute('data-square') ?? '',
  ).sort();
}
