import { useEffect, useRef, useState, type FormEvent } from 'react';
import { fetchRuns, startGame, type RunSummary } from './api';
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
 * The runs there are to play against, or null until they arrive, and why there are none.
 *
 * Asked for the first time a side is given to a model, and not before: most games started
 * here have no model in them, and a server with no runs on it should answer no questions
 * about runs. Asked once, because a run that saves its first checkpoint meanwhile is one
 * reload away and nothing here is worth polling the server for.
 */
function useRuns(wanted: boolean): [RunSummary[] | null, string | null] {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const asked = useRef(false);

  useEffect(() => {
    if (!wanted || asked.current) {
      return;
    }
    asked.current = true;
    let dropped = false;
    fetchRuns().then(
      (found) => !dropped && setRuns(found),
      (e: unknown) => {
        if (!dropped) {
          // An empty list rather than no list: the form has an answer to give, which is
          // that there is nothing here to play against.
          setRuns([]);
          setError(e instanceof Error ? e.message : String(e));
        }
      },
    );
    return () => {
      dropped = true;
    };
  }, [wanted]);

  return [runs, error];
}

/** Starts a new game on the server, replacing the current one for every viewer. */
export function NewGameForm() {
  const [white, setWhite] = useState<PlayerChoice>({ ...NO_MODEL, kind: 'human' });
  const [black, setBlack] = useState<PlayerChoice>({ ...NO_MODEL, kind: 'random' });
  const [moveDelay, setMoveDelay] = useState(DEFAULT_MOVE_DELAY);
  const [fen, setFen] = useState('');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runs, runsError] = useRuns(white.kind === 'model' || black.kind === 'model');

  // What is actually being asked for: a model chosen before the runs arrived means the
  // run it is being shown, which is the first of them.
  const sides = { white: resolved(white, runs), black: resolved(black, runs) };
  // The delay holds back a move a player works out for itself, so two people at the
  // board have nothing to hold back. The setting is kept, just switched off.
  const pacesNothing = sides.white.kind === 'human' && sides.black.kind === 'human';
  // A model with no run to play is not a player, and the server would only refuse it.
  const ready = isPlayable(sides.white) && isPlayable(sides.black);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStarting(true);
    setError(null);
    try {
      await startGame({
        white: playerSpec(sides.white),
        black: playerSpec(sides.black),
        move_delay: moveDelay,
        // An empty box is not a position: that game starts where games start.
        fen: fen.trim() || null,
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <form className="new-game" aria-labelledby="new-game-heading" onSubmit={submit}>
      <h2 id="new-game-heading">New game</h2>
      <PlayerPicker label="White" value={white} onChange={setWhite} runs={runs} />
      <PlayerPicker label="Black" value={black} onChange={setBlack} runs={runs} />
      {runsError !== null && (
        <p className="note">Could not list the training runs: {runsError}</p>
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
