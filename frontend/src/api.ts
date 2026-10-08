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
  | 'abort'
  /** A player could not go on, such as an engine that died: no result. */
  | 'error';

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
  /** The move as written in the position it was considered in; absent from older thoughts. */
  san?: string | null;
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

/**
 * What the player on move has chosen from, while its move is held back by the move delay: the
 * same thoughts its move carries once it is played. Mirrors chess_ai.game_session.Considering.
 */
export interface Considering {
  /** How many moves had been played: it is about the position after that many. */
  ply: number;
  /** The side whose player is considering. */
  side: Color;
  thoughts: Thoughts;
}

/** How a model turns its distribution over moves into the one move it plays. */
export type SelectionStrategy = 'argmax' | 'sample';

/** Which checkpoint a model player is: mirrors chess_ai.players.ModelDescription. */
export interface ModelDescription {
  /** The training run the weights came from. */
  run: string;
  /** The step the checkpoint was saved at, which is what names it in the run. */
  checkpoint: number;
  /** The rating it was asked to play like, or null for a position that claims none. */
  rating: number | null;
  strategy: SelectionStrategy;
  /** How flat the distribution is sampled from; null when it is not sampled at all. */
  temperature: number | null;
}

/** How strong a Stockfish side plays: mirrors chess_ai.players.StockfishDescription. */
export interface StockfishDescription {
  /** The strength it plays at, within the range Stockfish supports. */
  elo: number;
  /** The strength that was asked for, which is `elo` unless it was out of range. */
  requested_elo: number;
  min_elo: number;
  max_elo: number;
  /** Seconds it thinks about each move. */
  move_time: number;
}

export interface PlayerInfo {
  /** Shown to viewers, such as "Random mover". */
  name: string;
  /** Whether this side's moves are submitted by a viewer rather than played by itself. */
  accepts_moves: boolean;
  /** Which checkpoint is playing this side, for a side a checkpoint is playing. */
  model: ModelDescription | null;
  /** How strong Stockfish plays this side, for a side Stockfish is playing. */
  stockfish: StockfishDescription | null;
}

/** What one of two people at the board has asked the other, awaiting an answer. */
export interface PendingRequest {
  /** Which request this is, which an answer quotes. */
  id: number;
  kind: 'takeback' | 'abort';
  /** The side that asked. */
  by: Color;
}

/** The game waits for a player to take over a side whose player cannot go on. */
export interface Paused {
  /** The side to move, whose player is gone. */
  side: Color;
  /** Why it cannot play, such as a checkpoint deleted since the game began with it. */
  reason: string;
}

/** One side's player was replaced in the middle of the game, and by which. */
export interface Replacement {
  /** How many moves had been played when the new player took over. */
  ply: number;
  side: Color;
  /** The player that could not go on. */
  old: PlayerInfo;
  /** The player that took its place. */
  new: PlayerInfo;
}

/** Mirrors chess_ai.game_session.GameState. */
export interface GameState {
  /** Tells this game apart from every other; it names the PGN file, and reaches nothing. */
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
  /** What one side has asked the other and is waiting to hear about, if anything. */
  request: PendingRequest | null;
  /** Set while the game waits for another player to take over a side that cannot go on. */
  paused: Paused | null;
  /** Every player replaced so far, in the order it happened. */
  replacements: Replacement[];
  /** What the player on move is considering while its move is held back, if it says. */
  considering: Considering | null;
}

/**
 * What a link may do in its game: play one side, abort a game nobody plays by hand
 * ("control"), or only watch.
 */
export type Access = Color | 'control' | 'watch';

/** The links to one game, as the person who started it is given them. */
export interface GameLinks {
  /** Follows the game and can do nothing to it. */
  watch: string;
  /** Plays White, in a game a person plays White in. */
  white: string | null;
  /** Plays Black, in a game a person plays Black in. */
  black: string | null;
  /** Aborts a game nobody plays by hand. */
  control: string | null;
}

/** A game as the link it was reached through sees it: mirrors chess_ai.web.app.SeatView. */
export interface SeatView {
  type: 'state';
  game: GameState;
  access: Access;
  /** The game's watch link, to pass on to anyone who wants to follow it. */
  watch: string;
  /** When the game last changed, as an ISO timestamp. */
  updated: string;
}

/** A game that has just been started, and every link to it. */
export interface NewGame {
  game: GameState;
  links: GameLinks;
}

/**
 * What the game channel sends: the full state first, as the link sees it, then everything
 * that happens. A move, a takeback or the end of the game also ends any request that was
 * waiting for an answer, and whatever the player on move was considering. An error answers
 * something this viewer sent, and reaches nobody else.
 */
export type GameEvent =
  | SeatView
  | { type: 'move'; ply: number; move: MoveRecord; position: PositionSnapshot }
  | { type: 'takeback'; ply: number; position: PositionSnapshot }
  | { type: 'request'; request: PendingRequest | null }
  | { type: 'game_over'; position: PositionSnapshot }
  | { type: 'paused'; paused: Paused }
  | { type: 'replaced'; replacement: Replacement }
  | { type: 'considering'; considering: Considering }
  | { type: 'error'; message: string };

/**
 * What a viewer sends, on behalf of the side their link plays: the link says which game and
 * which side, so nothing here does.
 */
export type ViewerMessage =
  | { type: 'move'; uci: string }
  | { type: 'resign' }
  | { type: 'abort' }
  | { type: 'takeback' }
  | { type: 'answer'; request: number; accept: boolean };

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

/** One PGN file a dataset was built from: mirrors chess_ai.dataset.manifest.SourceInfo. */
export interface DatasetSource {
  path: string;
  bytes: number;
  games_read: number;
  games_kept: number;
  /** What stopped the file being read whole, if anything did. */
  error: string | null;
  /** Whether what failed was the file rather than the PGN in it. */
  went_away: boolean;
  /** The SHA-256 of the file's bytes; absent before datasets had versions. */
  sha256?: string | null;
  /** The Lichess month the file is a dump of, as YYYY-MM, when its name says so. */
  month?: string | null;
}

/** What a dataset is made of, counted as it was built: mirrors manifest.Statistics. */
export interface DatasetStatistics {
  /** Games by result, as PGN writes it. */
  results: Record<string, number>;
  /** Games by time-control class, in the order of the classes: "bullet" first. */
  time_controls: Record<string, number>;
  rating_sources: Record<string, number>;
  /** Players by rating, keyed by the low end of a 100-wide bucket, in rating order. */
  ratings: Record<string, number>;
  /** Players whose rating the file did not give. */
  ratings_unknown: number;
}

/** Mirrors chess_ai.dataset.manifest.Filters; a filter that is not set is left out. */
export interface DatasetFilters {
  min_rating?: number;
  max_rating?: number;
  unknown_rating_passes?: boolean;
  time_controls?: string[];
  exclude_terminations?: string[];
  from?: string;
  until?: string;
  min_clock?: number;
  sample?: number;
  max_games?: number;
}

/** How much of a dataset one split holds: mirrors chess_ai.dataset.manifest.SplitCounts. */
export interface DatasetSplitCounts {
  games: number;
  positions: number;
  /** How many positions are trained on; absent or null when every one of them is. */
  targets?: number | null;
}

/**
 * One build or append, and what it added to the end of the dataset: mirrors
 * chess_ai.dataset.manifest.Version. Its counts are its own share, not the dataset's total.
 */
export interface DatasetVersion {
  version: number;
  /** When it was added, as an ISO timestamp. */
  created: string;
  /** The cap on the dataset's total games it was added under. */
  max_games: number | null;
  sources: DatasetSource[];
  splits: Record<string, DatasetSplitCounts>;
  reached_max_games: boolean;
}

/** A dataset at its latest version, described: mirrors chess_ai.dataset.manifest.Manifest. */
export interface DatasetManifest {
  format_version: number;
  name: string;
  /** When the build finished, as an ISO timestamp. */
  created: string;
  move_vocabulary_size: number;
  validation_fraction: number;
  rating_source: string;
  sources: DatasetSource[];
  /** Which games the build let through; empty when every game was kept. */
  filters: DatasetFilters;
  splits: Record<string, DatasetSplitCounts>;
  /** Games left out, counted by why. */
  skipped: Record<string, number>;
  /** Games the filters left out, counted by which filter. */
  filtered: Record<string, number>;
  /** Positions stored but not trained on, counted by which filter left them out. */
  not_targets: Record<string, number>;
  /** Whether the latest version stopped at its maximum number of games. */
  reached_max_games: boolean;
  statistics: DatasetStatistics;
  /** Every version, oldest first; the totals above are the latest's. */
  versions: DatasetVersion[];
}

/** One dataset, with its manifest or why that cannot be read. */
export interface DatasetSummary {
  name: string;
  manifest: DatasetManifest | null;
  error: string | null;
}

/** One game of a dataset, as a page of its games lists it. */
export interface DatasetGame {
  /** Which game of its split this is, counting from zero; how it is opened. */
  index: number;
  plies: number;
  /** Null where the file gave no rating. */
  white_rating: number | null;
  black_rating: number | null;
  result: Result;
  /** As PGN writes it ("2024.01.05"), with "??" for what the file did not say. */
  date: string;
  time_control: string;
  rating_source: string;
  /** The PGN file the game came from. */
  source: string | null;
  custom_start: boolean;
}

/** One page of a split's games. */
export interface DatasetGames {
  dataset: string;
  split: string;
  /** How many games the split holds. */
  total: number;
  offset: number;
  games: DatasetGame[];
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

export type PlayerKind = 'human' | 'random' | 'model' | 'stockfish';

/** What each player kind is called in the new-game form. */
export const PLAYER_NAMES: Record<PlayerKind, string> = {
  human: 'Human',
  random: 'Random mover',
  model: 'Model',
  stockfish: 'Stockfish',
};

/** Which checkpoint of a run to play: its newest, its best, or the one from a step. */
export type CheckpointChoice = 'latest' | 'best' | number;

/** A checkpoint from a training run, playing by its policy. */
export interface ModelPlayerSpec {
  kind: 'model';
  run: string;
  checkpoint: CheckpointChoice;
  /** The rating to play like, or null to claim no rating at all. */
  rating: number | null;
  strategy: SelectionStrategy;
  /** How flat the distribution is sampled from. Not used when it plays its best move. */
  temperature: number;
  /** Fixes the sampling, so that the same game can be played twice. */
  seed?: number | null;
}

/** Stockfish, held to a strength by its calibrated limit. */
export interface StockfishPlayerSpec {
  kind: 'stockfish';
  /** The Elo to play at; one Stockfish does not support is played at the nearest it does. */
  elo: number;
  /** Seconds Stockfish thinks about each move. */
  move_time: number;
}

/**
 * What one side of a new game is to be played by. A tagged union, because the kinds do
 * not take the same settings; mirrors the server's own.
 */
export type PlayerSpec =
  | { kind: 'human' }
  | { kind: 'random' }
  | ModelPlayerSpec
  | StockfishPlayerSpec;

export interface NewGameRequest {
  white: PlayerSpec;
  black: PlayerSpec;
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

/** Where a run has got to, as its heartbeat says. */
export type RunStatus = 'running' | 'finished' | 'stopped' | 'crashed';

/**
 * One line of a run's metrics log. Every line has a step and says which split its numbers
 * come from; the rest depends on the split ("loss", "learning_rate" for training lines,
 * "top1", "illegal_top_move_rate" for validation ones). A loss that diverged is null.
 */
export interface MetricsRecord {
  step: number;
  split?: string;
  [metric: string]: number | string | null | undefined;
}

/** What people have said about a run: mirrors chess_ai.training.run_store.RunNotes. */
export interface RunNotes {
  /** What to show the run as instead of its name; null to show the name. */
  title: string | null;
  /** Single words, with no whitespace or commas in them. */
  tags: string[];
  notes: string;
}

/** One training run, as the run list and the new-game form show it. */
export interface RunSummary {
  name: string;
  /** What people call the run, to show instead of its name. */
  title?: string | null;
  tags?: string[];
  architecture: string | null;
  /** The name of the dataset the run trains on. */
  dataset?: string | null;
  /** Which version of the dataset it trains on. */
  dataset_version?: number | null;
  /** When the run started, as an ISO timestamp. */
  created: string | null;
  status: RunStatus | null;
  /** Whether the run says it is running and its heartbeat has stopped being updated. */
  stale?: boolean;
  /** When the last heartbeat was written, as an ISO timestamp. */
  updated?: string | null;
  /** How far the run has got, as its last heartbeat said. */
  step: number | null;
  /** How far it is going, which with `step` says how far through it is. */
  steps: number | null;
  epoch?: number | null;
  positions_per_second?: number | null;
  eta_seconds?: number | null;
  /** Why the run crashed, or anything else its last heartbeat had to say. */
  message?: string | null;
  /** How many checkpoints there are to choose between. */
  checkpoints: number;
  latest_train?: MetricsRecord | null;
  latest_validation?: MetricsRecord | null;
}

/** What the hardware said about itself in a heartbeat; any of it may be missing. */
export interface GpuStats {
  name: string | null;
  utilization_percent: number | null;
  memory_used_bytes: number | null;
  memory_total_bytes: number | null;
  process_memory_bytes: number | null;
  temperature_celsius: number | null;
}

/** A run's heartbeat: mirrors chess_ai.training.run_store.Heartbeat. */
export interface Heartbeat {
  status: RunStatus;
  pid: number;
  started: string;
  updated: string;
  step: number;
  steps: number;
  epoch: number;
  positions_per_second: number | null;
  eta_seconds: number | null;
  gpu: GpuStats | null;
  message: string | null;
}

/** What does not change once a run has started: the parts of run.json the dashboard shows. */
export interface RunInfo {
  name: string;
  created: string;
  seed: number;
  code_version: string;
  device: string;
  dataset: {
    name: string;
    /** The version of the dataset the run trains on; absent in a run.json from before. */
    version?: number;
    positions: number;
    train_positions: number;
    /** How many train positions are trained on, when the dataset filtered positions. */
    train_targets?: number | null;
  };
  model: { architecture: string; options: Record<string, unknown>; parameter_count: number };
  steps: number;
  batch_size: number;
  /** The run's config with every default filled in; only the parts the dashboard reads. */
  config?: { optimizer?: { gradient_clip?: number } };
}

/**
 * What the run channel sends: what the run is and where it has got to, first and whenever
 * that changes, and the lines of its metrics log. A metrics event with `reset` set replaces
 * everything before it rather than adding to it.
 */
export type RunEvent =
  | {
      type: 'run';
      name: string;
      info: RunInfo | null;
      heartbeat: Heartbeat | null;
      stale: boolean;
      /** Null while the run's notes cannot be read. */
      notes: RunNotes | null;
    }
  | { type: 'metrics'; reset: boolean; records: MetricsRecord[] }
  | { type: 'error'; message: string };

/** One checkpoint of a run, as the form lists it. */
export interface CheckpointSummary {
  step: number;
  created: string;
  /** The validation metrics measured at this step; a diverged step may have nulls. */
  metrics: Record<string, number | null>;
  /** Whether this is the run's best checkpoint by its own metric. */
  best: boolean;
  /** Whether this is the newest checkpoint of the run. */
  latest: boolean;
}

/** A run's checkpoints, newest first, which is the order they are offered in. */
export interface RunCheckpoints {
  run: string;
  checkpoints: CheckpointSummary[];
}

/**
 * Resolves an API path against the page's base URL. The server sets the base to its
 * path prefix, so the frontend never needs to know the prefix itself.
 */
export function apiUrl(path: string): string {
  return new URL(`api/${path}`, document.baseURI).href;
}

/** The API path of the game a link reaches, with `rest` after it. */
function gamePath(link: string, rest = ''): string {
  return `games/${encodeURIComponent(link)}${rest}`;
}

/**
 * Where a game is downloaded as PGN, in progress or finished, through any of its links.
 * The server names the file itself.
 */
export function pgnUrl(link: string): string {
  return apiUrl(gamePath(link, '/pgn'));
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
 * A link has nowhere to put a refusal, and this one can be refused: a game aborted in the
 * meantime is gone. Left to the browser, that answer is either a download that fails out
 * of sight or a file full of the error, so it is read here and the viewer is told.
 */
export async function fetchPgn(link: string): Promise<PgnFile> {
  const response = await fetch(pgnUrl(link));
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  const named = PGN_FILENAME.exec(response.headers.get('Content-Disposition') ?? '');
  return { name: named?.[1] || 'game.pgn', text: await response.text() };
}

/** The WebSocket URL of the game a link reaches, under the path prefix like every API URL. */
export function gameChannelUrl(link: string): string {
  const url = new URL(apiUrl(gamePath(link, '/ws')));
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
    // What the server says of a request it could not validate: one entry per problem.
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((problem: { msg?: unknown }) =>
          String(problem.msg ?? '').replace(/^Value error, /, ''),
        )
        .join('; ');
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

/** Which Stockfish games are played against here: mirrors chess_ai.stockfish.StockfishInfo. */
export interface StockfishInfo {
  /** What the engine calls itself, such as "Stockfish 17". */
  name: string;
  /** The weakest level it plays, which a lower Elo is raised to. */
  min_elo: number;
  /** The strongest level it plays, which a higher Elo is lowered to. */
  max_elo: number;
}

/** The Stockfish this server plays, or why there is none to play. */
export async function fetchStockfish(): Promise<StockfishInfo> {
  const response = await fetch(apiUrl('stockfish'));
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as StockfishInfo;
}

/**
 * The tag that sets a run aside: left out of lists unless asked for, never offered to play.
 * Mirrors chess_ai.training.run_store.ARCHIVED_TAG.
 */
export const ARCHIVED_TAG = 'archived';

/** Whether a run has been set aside with the archived tag. */
export function isArchived(run: RunSummary): boolean {
  return (run.tags ?? []).includes(ARCHIVED_TAG);
}

/**
 * Every training run on this server, newest first, with where each has got to. Archived runs
 * are left out unless `archived` asks for them.
 */
export async function fetchRuns({ archived = false } = {}): Promise<RunSummary[]> {
  const response = await fetch(apiUrl(archived ? 'runs?archived=true' : 'runs'));
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as RunSummary[];
}

/**
 * Replaces a run's title, tags and notes, all three at once, and gives back what the server
 * kept: it trims a title, and keeps each tag once.
 */
export async function saveRunNotes(run: string, notes: RunNotes): Promise<RunNotes> {
  const response = await fetch(apiUrl(`runs/${encodeURIComponent(run)}/notes`), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(notes),
  });
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as RunNotes;
}

/** The checkpoints of one run, newest first. */
export async function fetchCheckpoints(run: string): Promise<RunCheckpoints> {
  const response = await fetch(apiUrl(`runs/${encodeURIComponent(run)}/checkpoints`));
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as RunCheckpoints;
}

/** The WebSocket URL of one run's live stream, under the path prefix like every API URL. */
export function runChannelUrl(run: string): string {
  const url = new URL(apiUrl(`runs/${encodeURIComponent(run)}/ws`));
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
}

/**
 * A game as a link sees it, or null when the link reaches no game: it never did, or the game
 * was aborted.
 */
export async function fetchGame(link: string): Promise<SeatView | null> {
  const response = await fetch(apiUrl(gamePath(link)));
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as SeatView;
}

/**
 * Starts a new game with the settings the game `link` reaches was started with, and returns
 * the links that reach the new one. A watch link cannot.
 */
export async function rematch(link: string): Promise<NewGame> {
  const response = await fetch(apiUrl(gamePath(link, '/rematch')), { method: 'POST' });
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as NewGame;
}

/** Starts a new game, alongside any others, and returns the links that reach it. */
export async function startGame(request: NewGameRequest): Promise<NewGame> {
  const response = await fetch(apiUrl('games'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as NewGame;
}

/**
 * Hands the side a paused game is waiting on to another checkpoint, which plays on from where
 * the game is. Any link but a watch link may. The game's channel says so to everyone following.
 */
export async function replacePlayer(link: string, player: ModelPlayerSpec): Promise<SeatView> {
  const response = await fetch(apiUrl(gamePath(link, '/replace')), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(player),
  });
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as SeatView;
}

/** Every dataset on this server, by name. */
export async function fetchDatasets(): Promise<DatasetSummary[]> {
  return await fetched<DatasetSummary[]>(apiUrl('datasets'));
}

/** One dataset, by name. */
export async function fetchDataset(name: string): Promise<DatasetSummary> {
  return await fetched<DatasetSummary>(apiUrl(`datasets/${encodeURIComponent(name)}`));
}

/** A path under one split of one dataset. */
function splitPath(dataset: string, split: string, rest: string): string {
  return `datasets/${encodeURIComponent(dataset)}/${encodeURIComponent(split)}/${rest}`;
}

/** Up to `limit` games of one split of a dataset, from game `offset` on. */
export async function fetchDatasetGames(
  dataset: string,
  split: string,
  offset: number,
  limit: number,
): Promise<DatasetGames> {
  const url = new URL(apiUrl(splitPath(dataset, split, 'games')));
  url.searchParams.set('offset', String(offset));
  url.searchParams.set('limit', String(limit));
  return await fetched<DatasetGames>(url.href);
}

/** One game of a dataset, with the position before every move and after it. */
export async function openDatasetGame(
  dataset: string,
  split: string,
  index: number,
): Promise<ReplayGame> {
  return await fetched<ReplayGame>(apiUrl(splitPath(dataset, split, `games/${index}`)));
}

/** What a GET answers with, or why it was refused. */
async function fetched<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await refusal(response));
  }
  return (await response.json()) as T;
}
