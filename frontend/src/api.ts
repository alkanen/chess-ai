export type Color = 'white' | 'black';
export type PieceType = 'pawn' | 'knight' | 'bishop' | 'rook' | 'queen' | 'king';
export type Result = '1-0' | '0-1' | '1/2-1/2';
export type GameOverReason =
  | 'checkmate'
  | 'stalemate'
  | 'insufficient_material'
  | 'threefold_repetition'
  | 'fifty_move_rule';

export interface Piece {
  color: Color;
  type: PieceType;
}

export interface LastMove {
  from_square: string;
  /** Where the moving piece landed; for castling, the king's destination ("g1"). */
  to_square: string;
}

export interface GameOver {
  result: Result;
  reason: GameOverReason;
}

/** A position as the server describes it; mirrors chess_ai.position_view.PositionSnapshot. */
export interface PositionSnapshot {
  fen: string;
  /** The side to move. */
  turn: Color;
  /** Occupied squares, keyed by square name ("e4"). */
  pieces: Record<string, Piece>;
  /** The move that led to this position, if there is one. */
  last_move: LastMove | null;
  /** Set when the rules end the game in this position. */
  game_over: GameOver | null;
}

export interface CandidateMove {
  uci: string;
  probability: number;
}

/** Probabilities from the point of view of the player who is moving. */
export interface WinDrawLoss {
  win: number;
  draw: number;
  loss: number;
}

/** What a player was considering when it chose its move. */
export interface Thoughts {
  candidates: CandidateMove[];
  wdl: WinDrawLoss | null;
}

export interface MoveRecord {
  uci: string;
  san: string;
  thoughts: Thoughts | null;
}

/** Mirrors chess_ai.game_session.GameState. */
export interface GameState {
  /** The name of the player with the white pieces. */
  white: string;
  /** The name of the player with the black pieces. */
  black: string;
  /** Every move played so far. */
  moves: MoveRecord[];
  position: PositionSnapshot;
}

/** What the game channel sends: the full state first, then every move. */
export type GameEvent =
  | { type: 'no_game'; position: PositionSnapshot }
  | { type: 'state'; game: GameState }
  | { type: 'move'; ply: number; move: MoveRecord; position: PositionSnapshot };

export type PlayerKind = 'random';

export interface NewGameRequest {
  white: PlayerKind;
  black: PlayerKind;
  /** The least number of seconds between two moves. */
  move_delay: number;
}

/**
 * Resolves an API path against the page's base URL. The server sets the base to its
 * path prefix, so the frontend never needs to know the prefix itself.
 */
export function apiUrl(path: string): string {
  return new URL(`api/${path}`, document.baseURI).href;
}

/** The WebSocket URL of the game channel, under the path prefix like every API URL. */
export function gameChannelUrl(): string {
  const url = new URL(apiUrl('game/ws'));
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
}

/** Starts a new game, replacing the current one for every viewer. */
export async function startGame(request: NewGameRequest): Promise<GameState> {
  const response = await fetch(apiUrl('game'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as GameState;
}
