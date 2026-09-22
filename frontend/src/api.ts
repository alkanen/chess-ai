export type Color = 'white' | 'black';
export type PieceType = 'pawn' | 'knight' | 'bishop' | 'rook' | 'queen' | 'king';

export interface Piece {
  color: Color;
  type: PieceType;
}

/** A position as the server describes it; mirrors chess_ai.position_view.PositionSnapshot. */
export interface PositionSnapshot {
  fen: string;
  /** The side to move. */
  turn: Color;
  /** Occupied squares, keyed by square name ("e4"). */
  pieces: Record<string, Piece>;
}

/**
 * Resolves an API path against the page's base URL. The server sets the base to its
 * path prefix, so the frontend never needs to know the prefix itself.
 */
export function apiUrl(path: string): string {
  return new URL(`api/${path}`, document.baseURI).href;
}

export async function fetchStartPosition(signal?: AbortSignal): Promise<PositionSnapshot> {
  const response = await fetch(apiUrl('start-position'), { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as PositionSnapshot;
}
