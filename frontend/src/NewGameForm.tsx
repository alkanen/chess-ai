import { useState, type FormEvent } from 'react';
import { PLAYER_NAMES, startGame, type PlayerKind } from './api';
import './NewGameForm.css';

const PLAYER_KINDS = Object.keys(PLAYER_NAMES) as PlayerKind[];

/** Choices for the delay between moves, in seconds. */
const MOVE_DELAYS = [0, 0.1, 0.25, 0.5, 1, 2, 5];
const DEFAULT_MOVE_DELAY = 0.5;

interface PlayerChoiceProps {
  label: string;
  value: PlayerKind;
  onChange: (kind: PlayerKind) => void;
}

function PlayerChoice({ label, value, onChange }: PlayerChoiceProps) {
  return (
    <label>
      {label}{' '}
      <select value={value} onChange={(e) => onChange(e.target.value as PlayerKind)}>
        {PLAYER_KINDS.map((kind) => (
          <option key={kind} value={kind}>
            {PLAYER_NAMES[kind]}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Starts a new game on the server, replacing the current one for every viewer. */
export function NewGameForm() {
  const [white, setWhite] = useState<PlayerKind>('human');
  const [black, setBlack] = useState<PlayerKind>('random');
  const [moveDelay, setMoveDelay] = useState(DEFAULT_MOVE_DELAY);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The delay holds back a move a player works out for itself, so two people at the
  // board have nothing to hold back. The setting is kept, just switched off.
  const pacesNothing = white === 'human' && black === 'human';

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStarting(true);
    setError(null);
    try {
      await startGame({ white, black, move_delay: moveDelay });
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <form className="new-game" aria-labelledby="new-game-heading" onSubmit={submit}>
      <h2 id="new-game-heading">New game</h2>
      <PlayerChoice label="White" value={white} onChange={setWhite} />
      <PlayerChoice label="Black" value={black} onChange={setBlack} />
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
      <button type="submit" disabled={starting}>
        Start
      </button>
      {error !== null && <p role="alert">Could not start the game: {error}</p>}
    </form>
  );
}
