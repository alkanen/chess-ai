import { useEffect, useState } from 'react';
import {
  fetchCheckpoints,
  PLAYER_NAMES,
  type CheckpointChoice,
  type CheckpointSummary,
  type PlayerKind,
  type PlayerSpec,
  type RunSummary,
  type SelectionStrategy,
} from './api';

const PLAYER_KINDS = Object.keys(PLAYER_NAMES) as PlayerKind[];

/** What each way of choosing a move is called in the form. */
const STRATEGY_NAMES: Record<SelectionStrategy, string> = {
  argmax: 'its best move',
  sample: 'a sample',
};

/** Temperatures worth offering: under 1 sharpens towards the best move, over 1 flattens. */
const TEMPERATURES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3];
const DEFAULT_TEMPERATURE = 1;

/**
 * Everything one side of the form holds, whichever kind of player is chosen.
 *
 * The model settings are kept while another kind is chosen, as the move delay is: switching
 * to a person and back should not lose the run and rating that were picked.
 */
export interface PlayerChoice {
  kind: PlayerKind;
  /** The run to play, or "" when none has been chosen yet. */
  run: string;
  checkpoint: CheckpointChoice;
  /** The rating to play like, as typed; empty asks the model to claim no rating. */
  rating: string;
  strategy: SelectionStrategy;
  temperature: number;
}

export const NO_MODEL: PlayerChoice = {
  kind: 'human',
  run: '',
  checkpoint: 'best',
  rating: '',
  strategy: 'argmax',
  temperature: DEFAULT_TEMPERATURE,
};

/** A side of the form as the server wants it. */
export function playerSpec(choice: PlayerChoice): PlayerSpec {
  if (choice.kind !== 'model') {
    return { kind: choice.kind };
  }
  return {
    kind: 'model',
    run: choice.run,
    checkpoint: choice.checkpoint,
    // An empty box is not a rating: that model plays a position that claims none.
    rating: choice.rating.trim() === '' ? null : Number(choice.rating),
    strategy: choice.strategy,
    temperature: choice.temperature,
  };
}

/** Whether this side can be played at all, which a model with no run to play cannot. */
export function isPlayable(choice: PlayerChoice): boolean {
  return choice.kind !== 'model' || choice.run !== '';
}

/**
 * The choice with the run it means filled in.
 *
 * A model whose run is "" is one whose runs had not arrived when it was chosen, and what it
 * means is the run it is being shown: the first of them, which is the newest. Resolved here,
 * where the runs are known, rather than written back into the choice when they arrive — a
 * choice that edits itself races with whatever the person is typing at that moment.
 */
export function resolved(choice: PlayerChoice, runs: RunSummary[] | null): PlayerChoice {
  if (choice.kind !== 'model' || choice.run !== '') {
    return choice;
  }
  return { ...choice, run: runs?.[0]?.name ?? '' };
}

/** The checkpoints of `run`, or null until they arrive; [] for a run without any. */
function useCheckpoints(run: string): CheckpointSummary[] | null {
  const [checkpoints, setCheckpoints] = useState<CheckpointSummary[] | null>(null);

  useEffect(() => {
    if (run === '') {
      setCheckpoints(null);
      return;
    }
    // A run being trained saves checkpoints as it goes, so this is asked for whenever the
    // run changes rather than kept: reopening the list is one change of the run away.
    let dropped = false;
    setCheckpoints(null);
    fetchCheckpoints(run).then(
      (found) => !dropped && setCheckpoints(found.checkpoints),
      // A run that has gone, or a server that will not answer, leaves the two choices that
      // need no list: the best checkpoint and the latest one.
      () => !dropped && setCheckpoints([]),
    );
    return () => {
      dropped = true;
    };
  }, [run]);

  return checkpoints;
}

interface PlayerPickerProps {
  /** "White" or "Black": the colour this side plays, and what its fields are labelled by. */
  label: string;
  value: PlayerChoice;
  onChange: (choice: PlayerChoice) => void;
  /** The runs there are to play against, or null until they arrive. */
  runs: RunSummary[] | null;
}

/** One side of the new-game form: what plays it, and the settings that kind takes. */
export function PlayerPicker({ label, value, onChange, runs }: PlayerPickerProps) {
  // What is on show, and what a change to any other field is made against: the run the
  // person can see chosen, which for a model chosen before the runs arrived is the first.
  const choice = resolved(value, runs);
  const checkpoints = useCheckpoints(choice.kind === 'model' ? choice.run : '');

  return (
    <div className="player">
      <label>
        {label}{' '}
        <select
          value={choice.kind}
          onChange={(e) => onChange({ ...choice, kind: e.target.value as PlayerKind })}
        >
          {PLAYER_KINDS.map((kind) => (
            <option key={kind} value={kind}>
              {PLAYER_NAMES[kind]}
            </option>
          ))}
        </select>
      </label>
      {choice.kind === 'model' && (
        <ModelFields
          label={label}
          value={choice}
          onChange={onChange}
          runs={runs}
          checkpoints={checkpoints}
        />
      )}
    </div>
  );
}

interface ModelFieldsProps extends PlayerPickerProps {
  checkpoints: CheckpointSummary[] | null;
}

/** Which checkpoint plays this side, how well it should play, and how it picks its move. */
function ModelFields({ label, value, onChange, runs, checkpoints }: ModelFieldsProps) {
  if (runs !== null && runs.length === 0) {
    return <p className="note">No training runs have been kept here yet.</p>;
  }
  return (
    <div className="model-settings">
      <label>
        {label} run{' '}
        <select
          value={value.run}
          onChange={(e) =>
            // Another run's steps are not this one's, so the checkpoint goes back to
            // the one choice every run has an answer for.
            onChange({ ...value, run: e.target.value, checkpoint: 'best' })
          }
        >
          {runs === null && <option value="">Looking…</option>}
          {runs?.map((run) => (
            <option key={run.name} value={run.name}>
              {describeRun(run)}
            </option>
          ))}
        </select>
      </label>
      <label>
        {label} checkpoint{' '}
        <select
          value={String(value.checkpoint)}
          onChange={(e) => onChange({ ...value, checkpoint: asCheckpoint(e.target.value) })}
        >
          <option value="best">Best</option>
          <option value="latest">Latest</option>
          {checkpoints?.map((checkpoint) => (
            <option key={checkpoint.step} value={String(checkpoint.step)}>
              {describeCheckpoint(checkpoint)}
            </option>
          ))}
        </select>
      </label>
      <label>
        {label} rating{' '}
        <input
          type="number"
          min={0}
          max={4000}
          step={50}
          value={value.rating}
          placeholder="any"
          onChange={(e) => onChange({ ...value, rating: e.target.value })}
        />
      </label>
      <label>
        {label} plays{' '}
        <select
          value={value.strategy}
          onChange={(e) =>
            onChange({ ...value, strategy: e.target.value as SelectionStrategy })
          }
        >
          {(Object.keys(STRATEGY_NAMES) as SelectionStrategy[]).map((strategy) => (
            <option key={strategy} value={strategy}>
              {STRATEGY_NAMES[strategy]}
            </option>
          ))}
        </select>
      </label>
      {value.strategy === 'sample' && (
        <label>
          {label} temperature{' '}
          <select
            value={value.temperature}
            onChange={(e) => onChange({ ...value, temperature: Number(e.target.value) })}
          >
            {TEMPERATURES.map((temperature) => (
              <option key={temperature} value={temperature}>
                {temperature}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}

/** A run in one line: its name, and how far it has got. */
function describeRun(run: RunSummary): string {
  const steps = run.steps !== null ? `, ${run.step ?? 0}/${run.steps} steps` : '';
  return `${run.name} (${run.checkpoints} checkpoints${steps})`;
}

/** A checkpoint in one line: which step it is, and whether it is the best or the newest. */
function describeCheckpoint(checkpoint: CheckpointSummary): string {
  const marks = [checkpoint.best && 'best', checkpoint.latest && 'latest'].filter(Boolean);
  return marks.length > 0 ? `Step ${checkpoint.step} (${marks.join(', ')})` : `Step ${checkpoint.step}`;
}

/** What a checkpoint option's value means: a name, or the number of a step. */
function asCheckpoint(value: string): CheckpointChoice {
  return value === 'best' || value === 'latest' ? value : Number(value);
}
