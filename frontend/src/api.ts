export type Color = 'white' | 'black';
export type PieceType = 'pawn' | 'knight' | 'bishop' | 'rook' | 'queen' | 'king';
/** "*" is a game that reached no result, which only an abort leaves behind. */
export type Result = '1-0' | '0-1' | '1/2-1/2' | '*';
export type GameOverReason =
  | 'checkmate'
  | 'stalemate'
  | 'insufficient_material'
  | 'threefold_repetition'
  | 'fifty_move_rule'
  | 'resignation'
  | 'abort';

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

/** One legal move, with what the board needs to draw it without knowing the rules. */
export interface LegalMove {
  /** The move in UCI, including the promotion piece ("e7e8q"). */
  uci: string;
  /** Where the moving piece lands; for castling, the king's destination ("g1"). */
  to_square: string;
  capture: boolean;
  castling: boolean;
  en_passant: boolean;
  /** What a promoting pawn becomes; one legal move per choice, so four per destination. */
  promotion: PieceType | null;
  /** Whether the move gives check. */
  check: boolean;
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
  /**
   * The square of the king in check ("e1"), which is the side to move's, or null. This
   * is the position's check status too: a king is in check exactly when it is set.
   */
  check_square: string | null;
  /** Every legal move for the side to move, keyed by its origin square ("e2"). */
  legal_moves: Record<string, LegalMove[]>;
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

export interface PlayerInfo {
  /** Shown to viewers, such as "Random mover". */
  name: string;
  /** Whether this side's moves are submitted by a viewer rather than played by itself. */
  accepts_moves: boolean;
}

/** Mirrors chess_ai.game_session.GameState. */
export interface GameState {
  /** Tells this game apart from the one that replaces it. */
  id: string;
  /** The player with the white pieces. */
  white: PlayerInfo;
  /** The player with the black pieces. */
  black: PlayerInfo;
  /** Every move played so far. */
  moves: MoveRecord[];
  position: PositionSnapshot;
}

/**
 * What the game channel sends: the full state first, then every move. An error answers
 * something this viewer sent, and reaches nobody else.
 */
export type GameEvent =
  | { type: 'no_game'; position: PositionSnapshot }
  | { type: 'state'; game: GameState }
  | { type: 'move'; ply: number; move: MoveRecord; position: PositionSnapshot }
  | { type: 'game_over'; position: PositionSnapshot }
  | { type: 'error'; message: string };

/**
 * What a viewer sends: a move for a side they play, or an end to the game. Each one
 * names the game it is meant for, since a new game can replace it before it arrives.
 */
export type ViewerMessage = { game: string } & (
  | { type: 'move'; uci: string }
  | { type: 'resign'; color: Color }
  | { type: 'abort' }
);

export type PlayerKind = 'human' | 'random';

/** What each player kind is called in the new-game form. */
export const PLAYER_NAMES: Record<PlayerKind, string> = {
  human: 'Human',
  random: 'Random mover',
};

export interface NewGameRequest {
  white: PlayerKind;
  black: PlayerKind;
  /**
   * The least number of seconds before a move a player works out for itself, so that a
   * game between players that move instantly can be followed. A move a human submits is
   * played as soon as it arrives.
   */
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
