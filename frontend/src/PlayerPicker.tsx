import { useEffect, useMemo, useState } from 'react';
import {
  fetchCheckpoints,
  fetchLadderPoints,
  PLAYER_NAMES,
  type CheckpointChoice,
  type CheckpointSummary,
  type LadderPoint,
  type PlayerKind,
  type PlayerSpec,
  type RunSummary,
  type SelectionStrategy,
  type StockfishInfo,
} from './api';
import { describeStrength, nearestLadder } from './ladder';
import { describeRun, runLabels } from './runLabels';

const PLAYER_KINDS = Object.keys(PLAYER_NAMES) as PlayerKind[];

/** What each way of choosing a move is called in the form. */
const STRATEGY_NAMES: Record<SelectionStrategy, string> = {
  argmax: 'its best move',
  sample: 'a sample',
};

/** Seconds a move worth offering Stockfish; its levels assume a few seconds a move. */
const MOVE_TIMES = [0.1, 0.25, 0.5, 1, 2, 5];
const DEFAULT_MOVE_TIME = 1;
const DEFAULT_ELO = '1500';

/**
 * The rating a model plays like when none is typed and it plays nobody who gave one: the
 * default rating of games played here. Mirrors chess_ai.web.app.DEFAULT_RATING.
 */
export const DEFAULT_RATING = 2000;

const RATING_KEY = 'chess-ai.my-rating';

/** The rating this browser's person last gave, or '' if none. */
export function rememberedRating(): string {
  try {
    // Checked on the way out as on the way in: an earlier build kept whatever was typed.
    const kept = window.localStorage.getItem(RATING_KEY)?.trim() ?? '';
    return isRating(kept) ? kept : '';
  } catch {
    // Storage this browser will not let the page use: typed again each game.
    return '';
  }
}

/**
 * Keep `rating` as this browser's person's, for the next game. Only a rating the server takes
 * is kept, so a game is never started from a bad one; an empty box keeps what was kept before,
 * since one game played unrated says nothing about the person's rating.
 */
export function rememberRating(rating: string): void {
  if (rating.trim() === '' || !isRating(rating)) {
    return;
  }
  try {
    window.localStorage.setItem(RATING_KEY, rating.trim());
  } catch {
    // Not remembered, which costs typing it again next time and nothing else.
  }
}

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
  /**
   * The rating a model is to play like, as typed. Empty plays like the person on the other
   * side, if they gave a rating, and otherwise like DEFAULT_RATING.
   */
  rating: string;
  /** Whether the model is to claim no rating at all, for a model trained without them. */
  unrated: boolean;
  /** How strong the person playing this side is, as typed; empty claims no rating. */
  humanRating: string;
  strategy: SelectionStrategy;
  temperature: number;
  /** The Elo Stockfish is to play at, as typed. */
  elo: string;
  /** Seconds Stockfish thinks about each move. */
  moveTime: number;
}

export const NO_MODEL: PlayerChoice = {
  kind: 'human',
  run: '',
  checkpoint: 'best',
  rating: '',
  unrated: false,
  humanRating: '',
  strategy: 'argmax',
  temperature: DEFAULT_TEMPERATURE,
  elo: DEFAULT_ELO,
  moveTime: DEFAULT_MOVE_TIME,
};

/** A side of the form as the server wants it. */
export function playerSpec(choice: PlayerChoice): PlayerSpec {
  if (choice.kind === 'stockfish') {
    return { kind: 'stockfish', elo: Number(choice.elo), move_time: choice.moveTime };
  }
  if (choice.kind === 'human') {
    return { kind: 'human', rating: typedRating(choice.humanRating) };
  }
  if (choice.kind !== 'model') {
    return { kind: choice.kind };
  }
  return {
    kind: 'model',
    run: choice.run,
    checkpoint: choice.checkpoint,
    rating: askedRating(choice),
    strategy: choice.strategy,
    temperature: choice.temperature,
  };
}

/** A rating as typed, or null for an empty box. */
function typedRating(typed: string): number | null {
  return typed.trim() === '' ? null : Number(typed);
}

/**
 * The rating a model is to play like, as the server wants it: an empty box is left to the
 * server, which knows who the model plays.
 */
function askedRating(choice: PlayerChoice): number | 'unknown' | null {
  return choice.unrated ? 'unknown' : typedRating(choice.rating);
}

/**
 * The rating a person playing `side` gave, or null if a person does not play it or gave none:
 * what a model on the other side plays like when no rating is typed for it.
 */
export function personsRating(side: PlayerChoice): number | null {
  return side.kind === 'human' ? typedRating(side.humanRating) : null;
}

/**
 * The rating a model will play like, as the server works it out; null when it claims none.
 * `person` is the rating of the person it plays, if one plays it and gave one.
 */
export function playsLike(choice: PlayerChoice, person: number | null): number | null {
  if (choice.unrated) {
    return null;
  }
  return typedRating(choice.rating) ?? person ?? DEFAULT_RATING;
}

/**
 * Whether `typed` is empty or a rating the server takes: a whole number from 0 to 4000, in
 * digits. The boxes are text boxes, so that this sees what was typed: a number box reports
 * "" for what it cannot read, such as "15OO" or a lone "-", which would pass as no rating.
 */
export function isRating(typed: string): boolean {
  if (typed.trim() === '') {
    return true;
  }
  return /^\s*\d+\s*$/.test(typed) && Number(typed) <= 4000;
}

/** Whether a model's rating is one the server takes: claiming none, or a rating that is. */
export function isModelRating(choice: PlayerChoice): boolean {
  return choice.unrated || isRating(choice.rating);
}

/**
 * Whether this side can be played at all, which a model with no run to play cannot, and
 * Stockfish with no Elo to play at cannot either; nor can a side with a rating the server
 * would refuse.
 */
export function isPlayable(choice: PlayerChoice): boolean {
  switch (choice.kind) {
    case 'human':
      return isRating(choice.humanRating);
    case 'model':
      return choice.run !== '' && isModelRating(choice);
    case 'stockfish':
      return choice.elo.trim() !== '' && isRating(choice.elo);
    default:
      return true;
  }
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

/**
 * What `ask` answers about `run`, or null until it does (or while no run is chosen), and
 * `failed` if it will not.
 *
 * A run being trained saves checkpoints and ladders them as it goes, so this is asked again
 * whenever the run changes rather than kept: a fresh answer is one change of the run away.
 */
function useRunAnswer<T>(
  run: string,
  ask: (run: string) => Promise<T>,
  failed: T | null,
): T | null {
  const [answer, setAnswer] = useState<T | null>(null);

  useEffect(() => {
    if (run === '') {
      setAnswer(null);
      return;
    }
    let dropped = false;
    setAnswer(null);
    ask(run).then(
      (found) => !dropped && setAnswer(found),
      () => !dropped && setAnswer(failed),
    );
    return () => {
      dropped = true;
    };
  }, [run, ask, failed]);

  return answer;
}

const askCheckpoints = async (run: string) => (await fetchCheckpoints(run)).checkpoints;
const NO_CHECKPOINTS: CheckpointSummary[] = [];

/** The checkpoints of `run`, or null until they arrive; [] for a run without any. */
function useCheckpoints(run: string): CheckpointSummary[] | null {
  // A run that has gone, or a server that will not answer, leaves the two choices that need
  // no list: the best checkpoint and the latest one.
  return useRunAnswer(run, askCheckpoints, NO_CHECKPOINTS);
}

/**
 * The ladders of `run`'s checkpoints, pruned ones among them; [] for a run without any, and
 * null until they arrive or when they cannot be listed, which is not the same as there being
 * none: the picker says nothing of strength then rather than that no checkpoint has any.
 */
function useLadders(run: string): LadderPoint[] | null {
  return useRunAnswer(run, fetchLadderPoints, null);
}

interface PlayerPickerProps {
  /** "White" or "Black": the colour this side plays, and what its fields are labelled by. */
  label: string;
  value: PlayerChoice;
  onChange: (choice: PlayerChoice) => void;
  /** The runs there are to play against, or null until they arrive. */
  runs: RunSummary[] | null;
  /** The Stockfish there is to play, or null until the server says (or if it has none). */
  stockfish: StockfishInfo | null;
  /** The rating of the person playing the other side, if one does and gave one. */
  person: number | null;
}

/** One side of the new-game form: what plays it, and the settings that kind takes. */
export function PlayerPicker({
  label,
  value,
  onChange,
  runs,
  stockfish,
  person,
}: PlayerPickerProps) {
  // What is on show, and what a change to any other field is made against: the run the
  // person can see chosen, which for a model chosen before the runs arrived is the first.
  const choice = resolved(value, runs);
  const checkpoints = useCheckpoints(choice.kind === 'model' ? choice.run : '');
  const ladders = useLadders(choice.kind === 'model' ? choice.run : '');

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
      {choice.kind === 'human' && <HumanFields label={label} value={choice} onChange={onChange} />}
      {choice.kind === 'model' && (
        <ModelFields
          label={label}
          value={choice}
          onChange={onChange}
          runs={runs}
          person={person}
          checkpoints={checkpoints}
          ladders={ladders}
        />
      )}
      {choice.kind === 'stockfish' && (
        <StockfishFields label={label} value={choice} onChange={onChange} info={stockfish} />
      )}
    </div>
  );
}

interface ModelPickerProps {
  /** "White" or "Black": the colour the model plays, and what its fields are labelled by. */
  label: string;
  value: PlayerChoice;
  onChange: (choice: PlayerChoice) => void;
  /** The runs there are to play, or null until they arrive. */
  runs: RunSummary[] | null;
  /** The rating of the person playing the other side, if one does and gave one. */
  person: number | null;
}

/**
 * A checkpoint to play a side, chosen on its own: for where no other kind of player will do,
 * such as taking over from a checkpoint that is gone.
 */
export function ModelPicker({ label, value, onChange, runs, person }: ModelPickerProps) {
  const choice = resolved({ ...value, kind: 'model' }, runs);
  const checkpoints = useCheckpoints(choice.run);
  const ladders = useLadders(choice.run);
  return (
    <ModelFields
      label={label}
      value={choice}
      onChange={onChange}
      runs={runs}
      person={person}
      checkpoints={checkpoints}
      ladders={ladders}
    />
  );
}

interface HumanFieldsProps {
  label: string;
  value: PlayerChoice;
  onChange: (choice: PlayerChoice) => void;
}

/** How strong the person playing this side is, which a model playing them is told. */
function HumanFields({ label, value, onChange }: HumanFieldsProps) {
  const hint = `${label.toLowerCase()}-rating-hint`;
  return (
    <div className="model-settings">
      <label>
        {label} rating{' '}
        <input
          type="text"
          inputMode="numeric"
          autoComplete="off"
          value={value.humanRating}
          aria-invalid={!isPlayable(value)}
          placeholder="unrated"
          aria-describedby={hint}
          onChange={(e) => onChange({ ...value, humanRating: e.target.value })}
        />
      </label>
      <p className="note" id={hint}>
        A model playing this side is told it, and plays like it unless given a rating of its
        own; left empty, it is told nothing of you. Remembered in this browser if you typed it
        while you were the only person in the game.
      </p>
    </div>
  );
}

interface StockfishFieldsProps {
  label: string;
  value: PlayerChoice;
  onChange: (choice: PlayerChoice) => void;
  info: StockfishInfo | null;
}

/** How strong Stockfish plays this side, and how long it thinks about each move. */
function StockfishFields({ label, value, onChange, info }: StockfishFieldsProps) {
  const hint = `${label.toLowerCase()}-elo-hint`;
  return (
    <div className="model-settings">
      <label>
        {label} Elo{' '}
        <input
          // Checked by isRating, as a rating is, against the range the server takes rather
          // than the one Stockfish plays: asking for 800 is played at the floor, and said to
          // be, rather than refused by the form.
          type="text"
          inputMode="numeric"
          autoComplete="off"
          value={value.elo}
          aria-invalid={!isPlayable(value)}
          aria-describedby={hint}
          onChange={(e) => onChange({ ...value, elo: e.target.value })}
        />
      </label>
      {info !== null && (
        <p className="note" id={hint}>
          {info.name} plays from {info.min_elo} to {info.max_elo}; an Elo outside that is
          played at the nearest end, and the game says so.
        </p>
      )}
      <label>
        {label} time a move{' '}
        <select
          value={value.moveTime}
          onChange={(e) => onChange({ ...value, moveTime: Number(e.target.value) })}
        >
          {MOVE_TIMES.map((seconds) => (
            <option key={seconds} value={seconds}>
              {seconds} s
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

interface ModelFieldsProps extends Omit<PlayerPickerProps, 'stockfish'> {
  checkpoints: CheckpointSummary[] | null;
  ladders: LadderPoint[] | null;
}

/** Which checkpoint plays this side, how well it should play, and how it picks its move. */
function ModelFields({
  label,
  value,
  onChange,
  runs,
  person,
  checkpoints,
  ladders,
}: ModelFieldsProps) {
  const labels = useMemo(() => runLabels(runs ?? []), [runs]);
  const hint = `${label.toLowerCase()}-plays-like-hint`;
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
            <option key={run.name} value={run.name} title={run.name}>
              {describeRun(run, labels.get(run.name) ?? run.name)}
            </option>
          ))}
        </select>
      </label>
      {/* In full, since the select may well be too narrow to show it, and the label it is
          shown by there is only what the run is called, not what it is. */}
      {value.run !== '' && <p className="note run-name">{value.run}</p>}
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
      <CheckpointStrength
        step={chosenStep(value.checkpoint, checkpoints)}
        ladders={ladders}
        asked={playsLike(value, person)}
      />
      <label>
        {label} plays like{' '}
        <input
          type="text"
          inputMode="numeric"
          autoComplete="off"
          value={value.unrated ? '' : value.rating}
          aria-invalid={!isModelRating(value)}
          // What an empty box plays like, which the note under it says why.
          placeholder={value.unrated ? 'none' : String(person ?? DEFAULT_RATING)}
          disabled={value.unrated}
          aria-describedby={hint}
          onChange={(e) => onChange({ ...value, rating: e.target.value })}
        />
      </label>
      <label>
        <input
          type="checkbox"
          checked={value.unrated}
          onChange={(e) => onChange({ ...value, unrated: e.target.checked })}
        />{' '}
        {label} claims no rating
      </label>
      <p className="note" id={hint}>
        {value.unrated
          ? 'Only a model trained on games without ratings plays well like this: one trained ' +
            'on Lichess games, which all have them, plays close to at random.'
          : person !== null
            ? `Left empty, it plays like its opponent’s ${person}.`
            : `Left empty, it plays like ${DEFAULT_RATING}.`}
      </p>
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

/**
 * The step a checkpoint choice means, or null while that is not known: the best and the
 * latest are found in the list of checkpoints, which may not have arrived, or be empty.
 */
function chosenStep(
  checkpoint: CheckpointChoice,
  checkpoints: CheckpointSummary[] | null,
): number | null {
  if (typeof checkpoint === 'number') {
    return checkpoint;
  }
  const found = checkpoints?.find((each) => (checkpoint === 'best' ? each.best : each.latest));
  return found?.step ?? null;
}

interface CheckpointStrengthProps {
  /** The step of the checkpoint chosen, or null while that is not known. */
  step: number | null;
  /** The run's ladders, or null until they arrive. */
  ladders: LadderPoint[] | null;
  /** The rating the checkpoint is to play like, or null for none. */
  asked: number | null;
}

/**
 * How strong the chosen checkpoint played on the Stockfish ladder, so the player knows roughly
 * what they are taking on; for one with no ladder, that it has none, and the nearest that has.
 */
function CheckpointStrength({ step, ladders, asked }: CheckpointStrengthProps) {
  if (step === null || ladders === null) {
    return null;
  }
  const nearest = nearestLadder(step, ladders);
  let text: string;
  if (nearest === null) {
    text = `Step ${step} has not been on the Stockfish ladder, nor has any other checkpoint of this run.`;
  } else if (nearest.step === step) {
    text = `Step ${step} ${describeStrength(nearest, asked)}`;
  } else {
    text =
      `Step ${step} has not been on the Stockfish ladder. ` +
      `The nearest that has, step ${nearest.step}, ${describeStrength(nearest, asked)}`;
  }
  return <p className="note strength">{text}</p>;
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
