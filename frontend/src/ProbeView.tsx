import { memo, useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  fetchProbes,
  type EvaluationEntry,
  type ProbeOutcome,
  type ProbeResult,
  type ProbeSetVersion,
} from './api';
import { Board } from './board/Board';
import { CATEGORIES, probedSteps, setMismatch, solved, solvedCount } from './probes';
import { formatCount } from './runFormat';
import { moveArrows, percent, whiteOdds } from './thoughts';
import { CandidateList, described } from './ThoughtsPanel';
import './ThoughtsPanel.css';
import './ProbeView.css';

/**
 * How long the scrubber has to rest on a checkpoint before its result is asked for, so that
 * dragging it across a long run does not ask for every checkpoint on the way.
 */
const SETTLE_MS = 120;

/** How many results are kept in the page, so that going back to one shows it at once. */
const KEPT_RESULTS = 32;

/** A result as one version of it was written, which a newer one replaces. */
function resultKey(entry: EvaluationEntry): string {
  return `${entry.step}@${entry.updated}`;
}

/** How long to wait before asking again for a result that could not be fetched. */
function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** (failures - 1), 10_000);
}

interface ProbeState {
  /** The last result fetched, and which version of which result it is, or null before any. */
  shown: { key: string; result: ProbeResult } | null;
  /** Why the result wanted now could not be fetched, while it is being tried again. */
  failed: { key: string; message: string } | null;
}

/**
 * The result `entry` stands for, asked for again whenever it is written again, and the last
 * one there was while it is on its way.
 *
 * Only the latest one asked for is ever shown: the answer to an earlier one, arriving late, is
 * about a checkpoint the viewer has left or a version since replaced. One that cannot be
 * fetched, such as while the server restarts, is asked for again until it comes, since nothing
 * else would ask: the run channel sends the same results again when it reconnects.
 */
function useProbeResult(run: string, entry: EvaluationEntry | undefined): ProbeState {
  const [state, setState] = useState<ProbeState>({ shown: null, failed: null });
  const kept = useRef(new Map<string, ProbeResult>());
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
      fetchProbes(run, step).then(
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
  }, [run, key, step]);

  return state;
}

interface ProbeCardProps {
  outcome: ProbeOutcome;
}

/** How the position was reached: the moves, and whether they start from a set-up position. */
function lineOf(outcome: ProbeOutcome): string {
  if (outcome.start === null) {
    return outcome.line === '' ? 'Before the first move' : outcome.line;
  }
  return outcome.line === '' ? 'A set-up position' : `From a set-up position: ${outcome.line}`;
}

/** One probe position: the board with the checkpoint's likeliest moves on it, and in words. */
function ProbeCard({ outcome }: ProbeCardProps) {
  const id = useId();
  const marked = (uci: string) => outcome.best.includes(uci);
  const arrows = useMemo(
    () => moveArrows(outcome.top, (uci) => outcome.best.includes(uci)),
    [outcome],
  );
  const verdict = solved(outcome);
  const turn = outcome.snapshot.turn;
  return (
    <article className="probe" aria-labelledby={`${id}-name`} data-probe={outcome.id}>
      <header>
        <h4 id={`${id}-name`}>{outcome.name}</h4>
        {verdict !== null && (
          <span className={verdict ? 'verdict solved' : 'verdict unsolved'}>
            {verdict ? 'Solved' : 'Not solved'}
          </span>
        )}
      </header>
      <p className="probe-line">{lineOf(outcome)}</p>
      {/* From the side to move, which is the side whose moves are shown. */}
      <Board snapshot={outcome.snapshot} orientation={turn} arrows={arrows} />
      <p className="probe-turn">{turn === 'white' ? 'White' : 'Black'} to move</p>
      <CandidateList
        label={`Most likely moves in ${outcome.name}`}
        candidates={outcome.top}
        marked={marked}
        markedAs="solves it"
      />
      {outcome.best_probability !== null && (
        <p className="probe-solution">
          {outcome.best.length === 1 ? 'The solution' : 'The solutions'}:{' '}
          {percent(outcome.best_probability)}
        </p>
      )}
      <p className="probe-odds">{described(whiteOdds(outcome.wdl, turn))}</p>
      {outcome.comment !== null && <p className="note">{outcome.comment}</p>}
    </article>
  );
}

/** Every probe of a result, by category. */
function ProbeGroups({ result }: { result: ProbeResult }) {
  return CATEGORIES.map(({ category, heading }) => {
    const outcomes = result.positions.filter((outcome) => outcome.category === category);
    if (outcomes.length === 0) {
      return null;
    }
    const { solved: count, outOf } = solvedCount(outcomes);
    return (
      <section key={category} className="probe-group" aria-label={heading}>
        <h4 className="probe-group-heading">
          {heading}
          {outOf > 0 && (
            <span className="note">
              {' '}
              solved {count} of {outOf}
            </span>
          )}
        </h4>
        <div className="probe-cards">
          {outcomes.map((outcome) => (
            <ProbeCard key={outcome.id} outcome={outcome} />
          ))}
        </div>
      </section>
    );
  });
}

interface ProbeViewProps {
  run: string;
  /** Which suites have a result about which checkpoints, as the run channel has it. */
  evaluations: EvaluationEntry[];
  /** Whether the run's config has its checkpoints probed. */
  probed: boolean;
  /**
   * The set the evaluator probes with, or null when none has said. From the run channel rather
   * than from each result, so that results kept in the page are compared with the set now.
   */
  currentSet: ProbeSetVersion | null;
}

/**
 * What the run's checkpoints make of the probe positions, one checkpoint at a time, chosen
 * with a scrubber. It shows the newest result until another is chosen, and follows the newest
 * as results arrive; a result written again replaces the one shown.
 *
 * Kept apart from the rest of the run page and only drawn again when the results change, since
 * the page around it changes every second and there are dozens of boards in it.
 */
export const ProbeView = memo(function ProbeView({
  run,
  evaluations,
  probed,
  currentSet,
}: ProbeViewProps) {
  const id = useId();
  const steps = useMemo(() => probedSteps(evaluations), [evaluations]);
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
  const entry = steps.at(index);
  const { shown, failed } = useProbeResult(run, entry);

  if (steps.length === 0) {
    if (!probed) {
      return null;
    }
    return (
      <section className="probe-view" aria-labelledby={`${id}-heading`}>
        <h3 id={`${id}-heading`}>Probe positions</h3>
        <p className="note">
          No checkpoint has been probed yet. <code>chess-ai evaluator</code> probes each one as it
          is saved.
        </p>
      </section>
    );
  }

  function choose(at: number) {
    // The last one is the newest, now and as newer ones arrive.
    setChosen(at >= steps.length - 1 ? null : steps[at].step);
  }

  const result = shown?.result;
  const current = entry !== undefined && shown?.key === resultKey(entry);
  const failure = entry !== undefined && failed?.key === resultKey(entry) ? failed : null;
  const solvedAll = result === undefined ? null : solvedCount(result.positions);
  const mismatch = result === undefined ? null : setMismatch(result, currentSet);
  return (
    <section className="probe-view" aria-labelledby={`${id}-heading`}>
      <h3 id={`${id}-heading`}>Probe positions</h3>
      <div className="probe-scrubber">
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
        <span className="probe-step">
          Step {formatCount(entry?.step)}{' '}
          <span className="note">
            ({index + 1} of {steps.length}
            {chosen === null || chosenIndex === -1 ? ', newest' : ''})
          </span>
        </span>
      </div>
      {failure !== null && (
        <p role="alert">
          Cannot load step {formatCount(entry?.step)} ({failure.message}); trying again.
        </p>
      )}
      {!current && result === undefined && failure === null && <p className="note">Loading…</p>}
      {result !== undefined && (
        <div className={current ? 'probe-result' : 'probe-result loading'} aria-busy={!current}>
          <p className="probe-summary">
            {!current && (
              <span className="note">
                Showing step {formatCount(result.model.checkpoint)} while step{' '}
                {formatCount(entry?.step)} loads.{' '}
              </span>
            )}
            {solvedAll !== null && solvedAll.outOf > 0 && (
              <>
                Solved {solvedAll.solved} of the {solvedAll.outOf} probes that have a solution,{' '}
              </>
            )}
            on {result.probe_set} v{result.probe_set_version}
            {result.model.rating !== null && <>, both sides rated {result.model.rating}</>}.
          </p>
          {mismatch === 'older' && currentSet !== null && (
            <p className="older-set" role="note">
              Measured on {result.probe_set} v{result.probe_set_version}, an older version than the{' '}
              {currentSet.name} v{currentSet.version} the evaluator probes with: this checkpoint has
              been pruned since, or has not been probed again yet.
            </p>
          )}
          {mismatch === 'other' && currentSet !== null && (
            <p className="older-set" role="note">
              Measured on {result.probe_set} v{result.probe_set_version}, not on {currentSet.name} v
              {currentSet.version}, which the evaluator probes with.
            </p>
          )}
          <ProbeGroups result={result} />
        </div>
      )}
    </section>
  );
});
