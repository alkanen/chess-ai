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
  /** The position the game began in, which says how its moves are numbered. */
  start_fen: string;
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
  | { type: 'takeback'; ply: number; position: PositionSnapshot }
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
  | { type: 'takeback' }
);

/** One move of a game being replayed, and the position it leads to. */
export interface ReplayMove {
  san: string;
  uci: string;
  /** The position the move leads to, which carries the move as its last move. */
  position: PositionSnapshot;
}

/** One game's headers: enough to pick it out of a file, and not a move of it. */
export interface GameSummary {
  /** Which game of the file this is, counting from zero; how it is asked for. */
  index: number;
  event: string;
  site: string;
  /** As PGN writes it ("2000.11.04"), unknown parts and all ("2000.??.??"). */
  date: string;
  round: string;
  white: string;
  black: string;
  result: Result;
  /** How the game ended, where the file says; games saved here carry it. */
  termination: string | null;
  /** How many moves of the main line there are to step through. */
  plies: number;
}

/** One game of a file, with the position before every move and after it. */
export interface ReplayGame extends GameSummary {
  /** The position the game began in, which says how its moves are numbered. */
  start_fen: string;
  start_position: PositionSnapshot;
  moves: ReplayMove[];
}

/** What a PGN file holds: every game's headers, and one game to step through. */
export interface ReplayFile {
  games: GameSummary[];
  selected: ReplayGame;
}

/** A game in the server's games directory, as the list of them describes it. */
export interface SavedGame {
  /** The file's name, which is what opens it. */
  name: string;
  event: string;
  date: string;
  white: string;
  black: string;
  result: Result;
}

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
  /**
   * The position to start from, which the side it gives the move opens from. Left out,
   * or null, the game starts where games start.
   */
  fen?: string | null;
}

/**
 * Resolves an API path against the page's base URL. The server sets the base to its
 * path prefix, so the frontend never needs to know the prefix itself.
 */
export function apiUrl(path: string): string {
  return new URL(`api/${path}`, document.baseURI).href;
}

/**
 * Where the game a viewer is looking at is downloaded as PGN, in progress or finished.
 *
 * The game is named, as in everything else a viewer sends, because a new game can replace
 * it in the moment before the click: the server then refuses the download rather than
 * handing over a game the viewer has never seen. The server names the file itself.
 */
export function pgnUrl(game: string): string {
  const url = new URL(apiUrl('game/pgn'));
  url.searchParams.set('game', game);
  return url.href;
}

/** What a PGN file is, to a browser that is being handed one or sent one. */
export const PGN_MEDIA_TYPE = 'application/x-chess-pgn';

/** A file the server has handed over, under the name the server gave it. */
export interface PgnFile {
  name: string;
  text: string;
}

const PGN_FILENAME = /filename="([^"]*)"/;

/**
 * The game's PGN, fetched rather than followed as a link.
 *
 * A link has nowhere to put a refusal, and this one can be refused: a game replaced in
 * the moment before the click is turned down rather than swapped for its replacement.
 * Left to the browser, that answer is either a download that fails out of sight or a
 * file full of the error, so it is read here and the viewer is told.
 */
export async function fetchPgn(game: string): Promise<PgnFile> {
  const response = await fetch(pgnUrl(game));
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  const named = PGN_FILENAME.exec(response.headers.get('Content-Disposition') ?? '');
  return { name: named?.[1] || 'game.pgn', text: await response.text() };
}

/** The WebSocket URL of the game channel, under the path prefix like every API URL. */
export function gameChannelUrl(): string {
  const url = new URL(apiUrl('game/ws'));
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
}

/**
 * Why the server would not do what was asked, in its own words where it gave any. It
 * explains a FEN it would not start from, which is the one refusal a person can act on.
 */
async function refusal(response: Response): Promise<string> {
  try {
    const { detail } = (await response.json()) as { detail?: unknown };
    if (typeof detail === 'string' && detail !== '') {
      return detail;
    }
  } catch {
    // No JSON body, or nothing useful in it; the status is all there is to go on.
  }
  return `${response.status} ${response.statusText}`;
}

/** Every game the server has saved, the most recently played first. */
export async function fetchSavedGames(): Promise<SavedGame[]> {
  const response = await fetch(apiUrl('replay/saved'));
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as SavedGame[];
}

/** Opens a saved game, by the `name` the list of saved games gives it. */
export async function openSavedGame(name: string, game = 0): Promise<ReplayFile> {
  const url = new URL(apiUrl(`replay/saved/${encodeURIComponent(name)}`));
  url.searchParams.set('game', String(game));
  return await replayed(url);
}

/**
 * Reads a PGN file, and replays one game of it.
 *
 * The server keeps nothing, so looking at a second game in the same file sends it up
 * again. Files here are the size of a text file, and the answer is the larger of the two.
 */
export async function openPgn(pgn: string, game = 0): Promise<ReplayFile> {
  const url = new URL(apiUrl('replay/pgn'));
  url.searchParams.set('game', String(game));
  return await replayed(url, {
    method: 'POST',
    headers: { 'Content-Type': PGN_MEDIA_TYPE },
    body: pgn,
  });
}

/** A file read back as the positions its games pass through, or why it could not be. */
async function replayed(url: URL, init?: RequestInit): Promise<ReplayFile> {
  const response = await fetch(url.href, init);
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as ReplayFile;
}

/** Starts a new game, replacing the current one for every viewer. */
export async function startGame(request: NewGameRequest): Promise<GameState> {
  const response = await fetch(apiUrl('game'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as GameState;
}
