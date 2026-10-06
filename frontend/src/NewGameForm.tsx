import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  fetchRuns,
  fetchStockfish,
  isArchived,
  startGame,
  type NewGame,
  type RunSummary,
  type StockfishInfo,
} from './api';
import {
  isPlayable,
  NO_MODEL,
  playerSpec,
  PlayerPicker,
  resolved,
  type PlayerChoice,
} from './PlayerPicker';
import './NewGameForm.css';

/** Choices for the delay between moves, in seconds. */
const MOVE_DELAYS = [0, 0.1, 0.25, 0.5, 1, 2, 5];
const DEFAULT_MOVE_DELAY = 0.5;

/**
 * What `ask` answers, or null until it does, and why it could not; asked the first time
 * `wanted` is true, and only then.
 *
 * The answer is kept however long it takes to arrive, even if whatever wanted it has been
 * changed back meanwhile: it is no less true for that, and the next time it is wanted it is
 * already here. Dropping it instead would leave the question asked and unanswered for good,
 * since a question is only asked once.
 */
function useAskedOnce<T>(wanted: boolean, ask: () => Promise<T>): [T | null, string | null] {
  const [answer, setAnswer] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const asked = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!wanted || asked.current) {
      return;
    }
    asked.current = true;
    ask().then(
      (found) => mounted.current && setAnswer(found),
      (e: unknown) => mounted.current && setError(e instanceof Error ? e.message : String(e)),
    );
  }, [wanted, ask]);

  return [answer, error];
}

/**
 * The runs there are to play against, or null until they arrive, and why there are none.
 *
 * Asked for the first time a side is given to a model, and not before: most games started
 * here have no model in them, and a server with no runs on it should answer no questions
 * about runs. Asked once, because a run that saves its first checkpoint meanwhile is one
 * reload away and nothing here is worth polling the server for.
 */
function useRuns(wanted: boolean): [RunSummary[] | null, string | null] {
  const [runs, error] = useAskedOnce(wanted, fetchRuns);
  // Archived runs are left out here as well as by the server, so that a server still running
  // code from before there were archived runs does not offer them either.
  const playable = useMemo(() => runs?.filter((run) => !isArchived(run)) ?? null, [runs]);
  // An empty list rather than no list when they could not be had: the form has an answer to
  // give, which is that there is nothing here to play against.
  return [error !== null ? [] : playable, error];
}

/**
 * The Stockfish there is to play, or null until the server says, and why there is none.
 *
 * Asked the first time a side is given to Stockfish, and once, as the runs are: the server
 * starts the engine to answer, and the answer is the range of strengths to offer, which
 * differs between versions of Stockfish.
 */
function useStockfish(wanted: boolean): [StockfishInfo | null, string | null] {
  return useAskedOnce(wanted, fetchStockfish);
}

interface NewGameFormProps {
  /** Told of the game once the server has started it, with every link to it. */
  onStarted?: (started: NewGame) => void;
}

/** Starts a new game on the server, alongside any others. */
export function NewGameForm({ onStarted }: NewGameFormProps) {
  const [white, setWhite] = useState<PlayerChoice>({ ...NO_MODEL, kind: 'human' });
  const [black, setBlack] = useState<PlayerChoice>({ ...NO_MODEL, kind: 'random' });
  const [moveDelay, setMoveDelay] = useState(DEFAULT_MOVE_DELAY);
  const [fen, setFen] = useState('');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runs, runsError] = useRuns(white.kind === 'model' || black.kind === 'model');
  const wantsStockfish = white.kind === 'stockfish' || black.kind === 'stockfish';
  const [stockfish, stockfishError] = useStockfish(wantsStockfish);

  // What is actually being asked for: a model chosen before the runs arrived means the
  // run it is being shown, which is the first of them.
  const sides = { white: resolved(white, runs), black: resolved(black, runs) };
  // The delay holds back a move a player works out for itself, so two people at the
  // board have nothing to hold back. The setting is kept, just switched off.
  const pacesNothing = sides.white.kind === 'human' && sides.black.kind === 'human';
  // A model with no run to play is not a player, and neither is a Stockfish the server has
  // said it does not have: the server would only refuse either.
  const ready =
    isPlayable(sides.white) &&
    isPlayable(sides.black) &&
    !(wantsStockfish && stockfishError !== null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStarting(true);
    setError(null);
    try {
      const started = await startGame({
        white: playerSpec(sides.white),
        black: playerSpec(sides.black),
        move_delay: moveDelay,
        // An empty box is not a position: that game starts where games start.
        fen: fen.trim() || null,
      });
      onStarted?.(started);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <form className="new-game" aria-labelledby="new-game-heading" onSubmit={submit}>
      <h2 id="new-game-heading">New game</h2>
      <PlayerPicker
        label="White"
        value={white}
        onChange={setWhite}
        runs={runs}
        stockfish={stockfish}
      />
      <PlayerPicker
        label="Black"
        value={black}
        onChange={setBlack}
        runs={runs}
        stockfish={stockfish}
      />
      {runsError !== null && (
        <p className="note">Could not list the training runs: {runsError}</p>
      )}
      {wantsStockfish && stockfishError !== null && (
        <p className="note">There is no Stockfish to play: {stockfishError}</p>
      )}
      <label>
        Delay between moves{' '}
        <select
          value={moveDelay}
          disabled={pacesNothing}
          aria-describedby={pacesNothing ? 'move-delay-note' : undefined}
          onChange={(e) => setMoveDelay(Number(e.target.value))}
        >
          {MOVE_DELAYS.map((delay) => (
            <option key={delay} value={delay}>
              {delay === 0 ? 'none' : `${delay} s`}
            </option>
          ))}
        </select>
      </label>
      {pacesNothing && (
        <p className="note" id="move-delay-note">
          Both sides are played by hand, so there is nothing to pace.
        </p>
      )}
      <label>
        Start from FEN{' '}
        <input
          type="text"
          value={fen}
          placeholder="the usual starting position"
          spellCheck={false}
          onChange={(e) => setFen(e.target.value)}
        />
      </label>
      <button type="submit" disabled={starting || !ready}>
        Start
      </button>
      {error !== null && <p role="alert">Could not start the game: {error}</p>}
    </form>
  );
}
