import type { RunSummary } from './api';
import { formatCount, formatRate } from './runFormat';

/**
 * Names up to this long, less the timestamp they start with, are shown whole. Run names are made
 * of every setting that differs from the experiment's defaults and reach well over a hundred
 * characters, which no select can show.
 */
const SHORT_NAME = 24;

/** A run name's leading timestamp, as runs are named here: "2026-10-02T19h17m_". */
const STARTED = /^(\d{4}-\d\d-\d\dT\d\dh\d\dm)_/;

/**
 * What to call each run where there is little room, such as in a select: its title if it has
 * one, else its name without the timestamp it starts with, cut down to the experiment when it
 * is long. What comes first is what tells runs apart, since a narrow select shows only that.
 *
 * Every run gets a label of its own. Two runs with the same label, such as two runs of one sweep
 * given one title, are told apart by when each started; two that started in the same minute are
 * shown by their whole names, since what tells them apart is the part that was cut.
 */
export function runLabels(runs: RunSummary[]): Map<string, string> {
  const labels = new Map(runs.map((run) => [run.name, firstLabel(run)]));
  for (const run of shared(runs, labels)) {
    const started = STARTED.exec(run.name)?.[1];
    const label = labels.get(run.name) ?? run.name;
    labels.set(run.name, started !== undefined ? `${label} (${started})` : run.name);
  }
  // Two runs with one label that started in the same minute are left with their names.
  for (const run of shared(runs, labels)) {
    labels.set(run.name, run.name);
  }
  return labels;
}

/** A run in one line: its label, how far it has got, and how many checkpoints it has. */
export function describeRun(run: RunSummary, label: string): string {
  const checkpoints = `${run.checkpoints} ${run.checkpoints === 1 ? 'checkpoint' : 'checkpoints'}`;
  if (run.steps === null) {
    return `${label} · ${checkpoints}`;
  }
  return `${label} · ${progress(run.step ?? 0, run.steps)} steps, ${checkpoints}`;
}

/**
 * How far a run has got, briefly, without ever making one that is still training look done:
 * the picker is where a checkpoint is chosen, and a run still training has a "best" that may
 * yet move. The step is rounded down for that, and when the two would still read alike they
 * are given in full.
 */
function progress(step: number, steps: number): string {
  if (step >= steps) {
    return formatRate(steps);
  }
  const [done, total] = [roundedDown(step), formatRate(steps)];
  return done === total ? `${formatCount(step)}/${formatCount(steps)}` : `${done}/${total}`;
}

/** A count as `formatRate` gives it, but never more than it is. */
function roundedDown(value: number): string {
  if (value >= 1e6) {
    return `${(Math.floor(value / 1e4) / 100).toFixed(2)}M`;
  }
  if (value >= 1e3) {
    return `${Math.floor(value / 1e3)}k`;
  }
  return Math.floor(value).toString();
}

function firstLabel(run: RunSummary): string {
  const title = run.title?.trim();
  if (title) {
    return title;
  }
  const name = run.name.replace(STARTED, '');
  if (name === '' || name.length <= SHORT_NAME) {
    return name || run.name;
  }
  const [experiment] = name.split('_');
  return experiment === name ? name : `${experiment}…`;
}

/** The runs whose label some other run has too. */
function shared(runs: RunSummary[], labels: Map<string, string>): RunSummary[] {
  const counts = new Map<string, number>();
  for (const label of labels.values()) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return runs.filter((run) => (counts.get(labels.get(run.name) ?? '') ?? 0) > 1);
}
