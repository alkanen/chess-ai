import { useEffect, useState, type FormEvent } from 'react';
import {
  fetchRuns,
  isArchived,
  replacePlayer,
  type Access,
  type Color,
  type ModelPlayerSpec,
  type Paused,
  type PlayerInfo,
  type RunSummary,
} from './api';
import { ModelPicker, NO_MODEL, playerSpec, resolved, type PlayerChoice } from './PlayerPicker';

const SIDES: Record<Color, string> = { white: 'White', black: 'Black' };

/** A checkpoint like the one that is gone: the same run and rating, chosen the same way. */
function likeTheOld(player: PlayerInfo): PlayerChoice {
  const model = player.model;
  return {
    ...NO_MODEL,
    kind: 'model',
    run: model?.run ?? '',
    rating: model?.rating != null ? String(model.rating) : '',
    strategy: model?.strategy ?? 'argmax',
    temperature: model?.temperature ?? NO_MODEL.temperature,
  };
}

/**
 * The run the choice names, or the first there is if that run is not among them: a run that
 * has been deleted is the likeliest reason its checkpoint is gone.
 */
function playable(choice: PlayerChoice, runs: RunSummary[] | null): PlayerChoice {
  if (runs !== null && !runs.some((run) => run.name === choice.run)) {
    return resolved({ ...choice, run: '' }, runs);
  }
  return resolved(choice, runs);
}

interface PausedPanelProps {
  /** The link the game is reached through, which is what asks for another player. */
  link: string;
  paused: Paused;
  /** The player that cannot go on. */
  player: PlayerInfo;
  access: Access;
  /** Whether the server is out of reach, so nothing can be asked of the game. */
  disabled: boolean;
}

/**
 * Why the game is waiting, and, for anyone but a watcher, another checkpoint to play on with.
 * Aborting is the other way out, which the game's own controls offer.
 */
export function PausedPanel({ link, paused, player, access, disabled }: PausedPanelProps) {
  const said = `${SIDES[paused.side]}'s player, ${player.name}, cannot play on: ${paused.reason}`;
  if (access === 'watch') {
    return (
      <p className="request" role="status">
        {said}. The game waits for another to be chosen.
      </p>
    );
  }
  return (
    <ReplaceForm link={link} said={said} side={paused.side} player={player} disabled={disabled} />
  );
}

interface ReplaceFormProps {
  link: string;
  said: string;
  side: Color;
  player: PlayerInfo;
  disabled: boolean;
}

function ReplaceForm({ link, said, side, player, disabled }: ReplaceFormProps) {
  const [choice, setChoice] = useState<PlayerChoice>(() => likeTheOld(player));
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);

  useEffect(() => {
    let dropped = false;
    fetchRuns().then(
      (found) => !dropped && setRuns(found.filter((run) => !isArchived(run))),
      (e: unknown) => {
        if (!dropped) {
          setRuns([]);
          setError(e instanceof Error ? e.message : String(e));
        }
      },
    );
    return () => {
      dropped = true;
    };
  }, []);

  const chosen = playable(choice, runs);

  function submit(event: FormEvent) {
    event.preventDefault();
    setAsking(true);
    setError(null);
    // Left waiting on success: the game's channel says when the new player has taken over,
    // which closes this panel, and asking a second time meanwhile would only be refused.
    replacePlayer(link, playerSpec(chosen) as ModelPlayerSpec).then(
      () => undefined,
      (e: unknown) => {
        setAsking(false);
        setError(e instanceof Error ? e.message : String(e));
      },
    );
  }

  return (
    <form className="request paused" aria-label="Choose another model" onSubmit={submit}>
      <p role="status">{said}.</p>
      <p>Choose another checkpoint to play on with from here, or abort the game.</p>
      <ModelPicker label={SIDES[side]} value={chosen} onChange={setChoice} runs={runs} />
      {error !== null && <p role="alert">{error}</p>}
      <button type="submit" disabled={disabled || asking || chosen.run === ''}>
        Play on with this checkpoint
      </button>
    </form>
  );
}
