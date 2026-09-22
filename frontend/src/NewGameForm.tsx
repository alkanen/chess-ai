import { useState, type FormEvent } from 'react';
import { startGame } from './api';
import './NewGameForm.css';

/** Choices for the delay between moves, in seconds. */
const MOVE_DELAYS = [0, 0.1, 0.25, 0.5, 1, 2, 5];
const DEFAULT_MOVE_DELAY = 0.5;

/** Starts a new game on the server, replacing the current one for every viewer. */
export function NewGameForm() {
  const [moveDelay, setMoveDelay] = useState(DEFAULT_MOVE_DELAY);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStarting(true);
    setError(null);
    try {
      await startGame({ white: 'random', black: 'random', move_delay: moveDelay });
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <form className="new-game" aria-labelledby="new-game-heading" onSubmit={submit}>
      <h2 id="new-game-heading">New game</h2>
      <p>Random mover against random mover</p>
      <label>
        Delay between moves{' '}
        <select value={moveDelay} onChange={(e) => setMoveDelay(Number(e.target.value))}>
          {MOVE_DELAYS.map((delay) => (
            <option key={delay} value={delay}>
              {delay === 0 ? 'none' : `${delay} s`}
            </option>
          ))}
        </select>
      </label>
      <button type="submit" disabled={starting}>
        Start
      </button>
      {error !== null && <p role="alert">Could not start the game: {error}</p>}
    </form>
  );
}
