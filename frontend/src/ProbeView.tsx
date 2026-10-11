import { memo, useId, useMemo } from 'react';
import {
  fetchProbes,
  type EvaluationEntry,
  type ProbeOutcome,
  type ProbeResult,
  type ProbeSetVersion,
} from './api';
import { Board } from './board/Board';
import {
  CheckpointScrubber,
  resultKey,
  suiteSteps,
  useCheckpointResult,
  useChosenCheckpoint,
} from './CheckpointResults';
import { CATEGORIES, PROBE_SUITE, setMismatch, solved, solvedCount } from './probes';
import { formatCount } from './runFormat';
import { moveArrows, percent, whiteOdds } from './thoughts';
import { CandidateList, described } from './ThoughtsPanel';
import './ThoughtsPanel.css';
import './ProbeView.css';

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
  const steps = useMemo(() => suiteSteps(evaluations, PROBE_SUITE), [evaluations]);
  const chosen = useChosenCheckpoint(steps);
  const { entry } = chosen;
  const { shown, failed } = useCheckpointResult(run, entry, fetchProbes);

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

  const result = shown?.result;
  const current = entry !== undefined && shown?.key === resultKey(entry);
  const failure = entry !== undefined && failed?.key === resultKey(entry) ? failed : null;
  const solvedAll = result === undefined ? null : solvedCount(result.positions);
  const mismatch = result === undefined ? null : setMismatch(result, currentSet);
  return (
    <section className="probe-view" aria-labelledby={`${id}-heading`}>
      <h3 id={`${id}-heading`}>Probe positions</h3>
      <CheckpointScrubber steps={steps} chosen={chosen} />
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
