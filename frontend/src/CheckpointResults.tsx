/**
 * What the run page's evaluation sections have in common: one suite's results about a run's
 * checkpoints, one checkpoint at a time, chosen with a scrubber that follows the newest until
 * another is chosen, and each result fetched when it is chosen and again whenever it is written
 * again.
 */

import { useEffect, useRef, useState } from 'react';
import type { EvaluationEntry } from './api';
import { formatCount } from './runFormat';
import './CheckpointResults.css';

/**
 * How long the scrubber has to rest on a checkpoint before its result is asked for, so that
 * dragging it across a long run does not ask for every checkpoint on the way.
 */
const SETTLE_MS = 120;

/** How many results are kept in the page, so that going back to one shows it at once. */
const KEPT_RESULTS = 32;

/**
 * The checkpoints that have a result of `suite`, by step, earliest first. Pruned checkpoints are
 * among them: their results are kept after the checkpoints themselves are gone.
 */
export function suiteSteps(evaluations: EvaluationEntry[], suite: string): EvaluationEntry[] {
  return evaluations
    .filter((entry) => entry.suite === suite)
    .sort((one, other) => one.step - other.step);
}

/** A result as one version of it was written, which a newer one replaces. */
export function resultKey(entry: EvaluationEntry): string {
  return `${entry.step}@${entry.updated}`;
}

/** How long to wait before asking again for a result that could not be fetched. */
function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** (failures - 1), 10_000);
}

export interface CheckpointResult<T> {
  /** The last result fetched, and which version of which result it is, or null before any. */
  shown: { key: string; result: T } | null;
  /** Why the result wanted now could not be fetched, while it is being tried again. */
  failed: { key: string; message: string } | null;
}

/**
 * The result `entry` stands for, fetched with `fetchResult`, asked for again whenever it is
 * written again, and the last one there was while it is on its way. `fetchResult` is to be the
 * same function on every render, such as one from the API module.
 *
 * Only the latest one asked for is ever shown: the answer to an earlier one, arriving late, is
 * about a checkpoint the viewer has left or a version since replaced. One that cannot be
 * fetched, such as while the server restarts, is asked for again until it comes, since nothing
 * else would ask: the run channel sends the same results again when it reconnects.
 */
export function useCheckpointResult<T>(
  run: string,
  entry: EvaluationEntry | undefined,
  fetchResult: (run: string, step: number) => Promise<T>,
): CheckpointResult<T> {
  const [state, setState] = useState<CheckpointResult<T>>({ shown: null, failed: null });
  const kept = useRef(new Map<string, T>());
  const key = entry === undefined ? null : resultKey(entry);
  const step = entry?.step;

  useEffect(() => {
    if (key === null || step === undefined) {
      return;
    }
    const known = kept.current.get(key);
    if (known !== undefined) {
      setState({ shown: { key, result: known }, failed: null });
      return;
    }
    let wanted = true;
    let failures = 0;
    const ask = () => {
      fetchResult(run, step).then(
        (result) => {
          kept.current.set(key, result);
          if (kept.current.size > KEPT_RESULTS) {
            // A Map keeps the order things were put in, so the first is the oldest.
            kept.current.delete(kept.current.keys().next().value!);
          }
          if (wanted) {
            setState({ shown: { key, result }, failed: null });
          }
        },
        (error: unknown) => {
          if (!wanted) {
            return;
          }
          const message = error instanceof Error ? error.message : String(error);
          setState((before) => ({ ...before, failed: { key, message } }));
          failures += 1;
          timer = setTimeout(ask, retryDelayMs(failures));
        },
      );
    };
    let timer = setTimeout(ask, SETTLE_MS);
    return () => {
      wanted = false;
      clearTimeout(timer);
    };
  }, [run, key, step, fetchResult]);

  return state;
}

export interface ChosenCheckpoint {
  /** Where the checkpoint shown is among the steps, or -1 when there are none. */
  index: number;
  /** The checkpoint shown, or undefined when there are none. */
  entry: EvaluationEntry | undefined;
  /** Whether the newest is shown because it is the newest, and a newer one will replace it. */
  following: boolean;
  /** Show the checkpoint at `index` among the steps; the last one follows the newest. */
  choose: (index: number) => void;
}

/**
 * Which of `steps` is shown: the newest, followed as newer ones arrive, until another is chosen.
 */
export function useChosenCheckpoint(steps: EvaluationEntry[]): ChosenCheckpoint {
  /** The step chosen, or null to follow the newest. */
  const [chosen, setChosen] = useState<number | null>(null);
  const chosenIndex = chosen === null ? -1 : steps.findIndex((entry) => entry.step === chosen);
  if (chosen !== null && chosenIndex === -1) {
    // A chosen step that has no result any more, such as after the run was started again under
    // the same name, gives way to the newest for good: a step of the same number turning up
    // later is another checkpoint, which nobody chose.
    setChosen(null);
  }
  const index = chosenIndex === -1 ? steps.length - 1 : chosenIndex;
  return {
    index,
    entry: steps.at(index),
    following: chosenIndex === -1,
    // The last one is the newest, now and as newer ones arrive.
    choose: (at: number) => setChosen(at >= steps.length - 1 ? null : steps[at].step),
  };
}

interface CheckpointScrubberProps {
  steps: EvaluationEntry[];
  chosen: ChosenCheckpoint;
}

/** A slider over the checkpoints with a result, with a step either way and which one it is on. */
export function CheckpointScrubber({ steps, chosen }: CheckpointScrubberProps) {
  const { index, entry, following, choose } = chosen;
  return (
    <div className="checkpoint-scrubber">
      <button
        type="button"
        onClick={() => choose(index - 1)}
        disabled={index === 0}
        aria-label="Earlier checkpoint"
      >
        ‹
      </button>
      <input
        type="range"
        min={0}
        max={steps.length - 1}
        step={1}
        value={index}
        onChange={(event) => choose(Number(event.target.value))}
        aria-label="Checkpoint"
        aria-valuetext={`Step ${formatCount(entry?.step)}`}
      />
      <button
        type="button"
        onClick={() => choose(index + 1)}
        disabled={index === steps.length - 1}
        aria-label="Later checkpoint"
      >
        ›
      </button>
      <span className="checkpoint-step">
        Step {formatCount(entry?.step)}{' '}
        <span className="note">
          ({index + 1} of {steps.length}
          {following ? ', newest' : ''})
        </span>
      </span>
    </div>
  );
}
