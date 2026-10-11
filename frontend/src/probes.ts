/**
 * What the run page makes of a run's probe results: how a result reads.
 *
 * A probe is a fixed position every checkpoint of a run is shown, to see what it makes of it:
 * its most likely moves, how likely, and who it thinks is winning. Some have a solution, such as
 * a mate in one, and a checkpoint solves one when its most likely move is a solution.
 */

import type {
  ProbeCategory,
  ProbeOutcome,
  ProbeResult,
  ProbeSetVersion,
} from './api';

/** What probing is called among a run's evaluation suites. */
export const PROBE_SUITE = 'probe-positions';

/** The categories in the order the page shows them, with what each is headed. */
export const CATEGORIES: { category: ProbeCategory; heading: string }[] = [
  { category: 'opening', heading: 'Openings' },
  { category: 'middlegame', heading: 'Middlegames' },
  { category: 'tactic', heading: 'Tactics' },
  { category: 'endgame', heading: 'Endgames' },
];

/** Whether the checkpoint's most likely move solves the probe, or null for one without a solution. */
export function solved(outcome: ProbeOutcome): boolean | null {
  if (outcome.best.length === 0) {
    return null;
  }
  return outcome.top.length > 0 && outcome.best.includes(outcome.top[0].uci);
}

/** How many of `outcomes` that have a solution were solved, and how many have one. */
export function solvedCount(outcomes: ProbeOutcome[]): { solved: number; outOf: number } {
  const marked = outcomes.map(solved).filter((each) => each !== null);
  return { solved: marked.filter(Boolean).length, outOf: marked.length };
}

/**
 * How `result` stands to the set the evaluator probes with, `current`: measured on an older
 * version of it, which no result of the current one compares with; on another set altogether;
 * or neither, which includes a newer version, measured on demand before the evaluator was started
 * again with it, and a `current` nobody has said.
 */
export function setMismatch(
  result: ProbeResult,
  current: ProbeSetVersion | null,
): 'older' | 'other' | null {
  if (current === null) {
    return null;
  }
  if (current.name !== result.probe_set) {
    return 'other';
  }
  return result.probe_set_version < current.version ? 'older' : null;
}
