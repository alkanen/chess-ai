import type { MetricsRecord, RunInfo } from './api';
import { decimalsOf, formatCount, formatDuration, formatNumber, formatPercent } from './runFormat';

/**
 * The most lines a chart can tell apart. There are eight categorical colours and never more:
 * a ninth line would need one that cannot be told from one of these.
 */
export const MAX_SERIES = 8;

/** A metric from the lines of one split of the log. */
export interface Metric {
  split: 'train' | 'validation';
  metric: string;
}

/** One line on a run's chart. */
export interface SeriesSpec extends Metric {
  label: string;
}

/** What a chart is and how it shows its values, whatever its lines are. */
export interface ChartLook {
  id: string;
  title: string;
  format: (value: number) => string;
  /**
   * A value as a y-axis tick, with ticks `step` apart: as `format` puts it, or with more
   * digits where a chart zoomed in has ticks closer together than `format` tells apart.
   */
  formatTick: (value: number, step: number) => string;
  /** Whether the y-axis is logarithmic, which keeps a loss's early drop from flattening the rest. */
  logarithmic?: boolean;
  /**
   * Whether the chart is left out for a log that has none of its metric, rather than shown
   * empty: a metric runs did not always log, which an older run will never have.
   */
  optional?: boolean;
}

/** A level drawn across a run's chart, from the run's own settings rather than its log. */
export interface Reference {
  label: string;
  /** The level for this run, or null for a run that has none. */
  value: (info: RunInfo) => number | null;
}

export interface ChartSpec extends ChartLook {
  series: SeriesSpec[];
  reference?: Reference;
}

/** How a chart of plain numbers shows them, to `digits` significant digits or as ticks need. */
function numbers(digits = 4): Pick<ChartLook, 'format' | 'formatTick'> {
  return {
    format: (value) => formatNumber(value, digits),
    formatTick: (value, step) => {
      // The digits from the value's first down to the step's last, and one to spare, since
      // formatNumber writes a very small or large value in exponent form to one digit fewer.
      const needed = value === 0 ? 0 : Math.floor(Math.log10(Math.abs(value))) + 2 + decimalsOf(step);
      return formatNumber(value, Math.max(digits, needed));
    },
  };
}

/** How a chart of fractions shows them, as percentages to a decimal or as many as ticks need. */
const PERCENTAGES: Pick<ChartLook, 'format' | 'formatTick'> = {
  format: (value) => formatPercent(value),
  formatTick: (value, step) => formatPercent(value, Math.max(1, decimalsOf(step * 100))),
};

/** The gradient norm a run clips at, or null for a run that does not clip. */
export function gradientClip(info: RunInfo): number | null {
  const clip = info.config?.optimizer?.gradient_clip;
  return typeof clip === 'number' && clip > 0 ? clip : null;
}

/** The charts of a run, in the order they are shown. */
export const CHARTS: ChartSpec[] = [
  {
    id: 'loss',
    title: 'Loss',
    series: [
      { label: 'train', split: 'train', metric: 'loss' },
      { label: 'validation', split: 'validation', metric: 'loss' },
    ],
    ...numbers(),
    logarithmic: true,
  },
  {
    id: 'accuracy',
    title: 'Accuracy (validation)',
    series: [
      { label: 'top-1', split: 'validation', metric: 'top1' },
      { label: 'top-5', split: 'validation', metric: 'top5' },
    ],
    ...PERCENTAGES,
  },
  {
    id: 'learning-rate',
    title: 'Learning rate',
    series: [{ label: 'learning rate', split: 'train', metric: 'learning_rate' }],
    ...numbers(3),
  },
  {
    id: 'illegal',
    title: 'Illegal top-move rate (validation)',
    series: [{ label: 'illegal top move', split: 'validation', metric: 'illegal_top_move_rate' }],
    ...PERCENTAGES,
  },
  {
    id: 'gradient-norm',
    title: 'Gradient norm, before clipping',
    series: [{ label: 'gradient norm', split: 'train', metric: 'gradient_norm' }],
    reference: { label: 'clipped above', value: gradientClip },
    ...numbers(3),
    logarithmic: true,
    optional: true,
  },
  {
    id: 'clipped',
    title: 'Steps clipped',
    series: [{ label: 'clipped', split: 'train', metric: 'clipped_fraction' }],
    ...PERCENTAGES,
    optional: true,
  },
];

/**
 * A chart's lines as uPlot takes them: the x values, and each series' value at every one of
 * them. Validation is measured far less often than training is logged, and runs compared on
 * one chart are logged at x values of their own, so a series has a null wherever it has
 * nothing at that x, as it does where a value diverged.
 */
export type ChartData = [number[], ...(number | null)[][]];

/** What a chart's x-axis can measure a run's progress in. */
export type XAxisId = 'step' | 'positions' | 'time';

export interface XAxis {
  id: XAxisId;
  label: string;
  /** A value on the axis, for its ticks. */
  format: (value: number) => string;
  /** Where along the run a value was logged, for a caption: "step 1,500", "1.2M positions". */
  describe: (value: number) => string;
  /** The distances between ticks the axis may use, where uPlot's own would read badly. */
  increments?: number[];
}

/** A count of positions in three figures and a suffix, such as 12_288_000 as "12.3M". */
export function formatPositions(value: number): string {
  for (const [size, suffix] of [
    [1e9, 'G'],
    [1e6, 'M'],
    [1e3, 'k'],
  ] as const) {
    if (value >= size) {
      return `${Number((value / size).toPrecision(3))}${suffix}`;
    }
  }
  return formatCount(value);
}

const MINUTE = 60;
const HOUR = 60 * MINUTE;

/**
 * The x-axes a chart can be drawn against, the first of them the default. Steps compare
 * runs with the same batch size; positions seen compare runs with different ones; wall time
 * compares how much of the machine each took to get there.
 */
export const X_AXES: XAxis[] = [
  {
    id: 'step',
    label: 'Step',
    format: formatCount,
    describe: (value) => `step ${formatCount(value)}`,
  },
  {
    id: 'positions',
    label: 'Positions seen',
    format: formatPositions,
    describe: (value) => `${formatPositions(value)} positions`,
  },
  {
    id: 'time',
    label: 'Wall time',
    format: formatDuration,
    describe: (value) => `${formatDuration(value)} in`,
    increments: [1, 5, 10, 30, MINUTE, 5 * MINUTE, 10 * MINUTE, 30 * MINUTE, HOUR, 2 * HOUR, 6 * HOUR, 12 * HOUR, 24 * HOUR, 48 * HOUR, 168 * HOUR],
  },
];

export function xAxis(id: XAxisId): XAxis {
  return X_AXES.find((axis) => axis.id === id)!;
}

/**
 * Where each line of a run's log goes along `axis`, or null for one the log does not say
 * enough to place.
 *
 * Every line of a log written now says how many positions the run had trained on and how
 * many seconds it had been going. An older log says so less: its training lines have the
 * seconds and not the positions, which the run's batch size makes up for, and its validation
 * lines have neither, so they are placed at the seconds of the line logged before them.
 */
export function placeRecords(
  records: MetricsRecord[],
  axis: XAxisId,
  batchSize: number | null = null,
): (number | null)[] {
  let elapsed: number | null = null;
  return records.map((record) => {
    switch (axis) {
      case 'step':
        return record.step;
      case 'positions': {
        const seen = record.positions_seen;
        if (typeof seen === 'number') {
          return seen;
        }
        return batchSize == null ? null : record.step * batchSize;
      }
      case 'time':
        if (typeof record.elapsed === 'number') {
          elapsed = record.elapsed;
          return elapsed;
        }
        return elapsed;
    }
  });
}

/** One line of a chart: its value at each x it has one at. */
type Line = [x: number, y: number | null][];

/** Lines that were logged at x values of their own, lined up on every x any of them has. */
export function alignLines(lines: Line[]): ChartData {
  const byX = new Map<number, (number | null)[]>();
  lines.forEach((line, index) => {
    for (const [x, y] of line) {
      let values = byX.get(x);
      if (values === undefined) {
        values = lines.map(() => null);
        byX.set(x, values);
      }
      values[index] = y;
    }
  });
  const xs = [...byX.keys()].sort((a, b) => a - b);
  return [xs, ...lines.map((_, index) => xs.map((x) => byX.get(x)![index]))];
}

/** One metric of one split of a run's log, at the x values `xs` place each line at. */
function line(records: MetricsRecord[], xs: (number | null)[], series: Metric): Line {
  const found: Line = [];
  records.forEach((record, index) => {
    const x = xs[index];
    if (x == null || record.split !== series.split || !(series.metric in record)) {
      return;
    }
    const value = record[series.metric];
    found.push([x, typeof value === 'number' ? value : null]);
  });
  return found;
}

/** Whether any line of `records` has `metric`, which an optional chart is shown for. */
export function logs(records: MetricsRecord[], metric: Metric): boolean {
  return records.some((record) => record.split === metric.split && metric.metric in record);
}

/**
 * The lines of one run's chart, against `axis`, and then a level across it at `reference` if
 * that is not null. The level is drawn wherever the chart's first line has a value, so it
 * spans what the run has logged rather than an axis of its own.
 */
export function chartData(
  records: MetricsRecord[],
  spec: ChartSpec,
  axis: XAxisId = 'step',
  batchSize: number | null = null,
  reference: number | null = null,
): ChartData {
  const xs = placeRecords(records, axis, batchSize);
  const lines = spec.series.map((series) => line(records, xs, series));
  if (reference !== null) {
    lines.push(lines[0].map(([x]): [number, number] => [x, reference]));
  }
  return alignLines(lines);
}

/** A chart of runs side by side: one metric, a line for each run. */
export interface ComparisonChart extends ChartLook {
  metric: Metric;
}

/** The charts of runs compared, in the order they are shown. */
export const COMPARISON_CHARTS: ComparisonChart[] = [
  {
    id: 'train-loss',
    title: 'Training loss',
    metric: { split: 'train', metric: 'loss' },
    ...numbers(),
    logarithmic: true,
  },
  {
    id: 'validation-loss',
    title: 'Validation loss',
    metric: { split: 'validation', metric: 'loss' },
    ...numbers(),
    logarithmic: true,
  },
  {
    id: 'top1',
    title: 'Top-1 accuracy (validation)',
    metric: { split: 'validation', metric: 'top1' },
    ...PERCENTAGES,
  },
  {
    id: 'top5',
    title: 'Top-5 accuracy (validation)',
    metric: { split: 'validation', metric: 'top5' },
    ...PERCENTAGES,
  },
  {
    id: 'illegal',
    title: 'Illegal top-move rate (validation)',
    metric: { split: 'validation', metric: 'illegal_top_move_rate' },
    ...PERCENTAGES,
  },
  {
    id: 'learning-rate',
    title: 'Learning rate',
    metric: { split: 'train', metric: 'learning_rate' },
    ...numbers(3),
  },
  {
    id: 'gradient-norm',
    title: 'Gradient norm, before clipping',
    metric: { split: 'train', metric: 'gradient_norm' },
    ...numbers(3),
    logarithmic: true,
    optional: true,
  },
  {
    id: 'clipped',
    title: 'Steps clipped',
    metric: { split: 'train', metric: 'clipped_fraction' },
    ...PERCENTAGES,
    optional: true,
  },
];

/** One run on a comparison chart. */
export interface ComparedRun {
  records: MetricsRecord[];
  /** What makes up for a log too old to say how many positions it had seen. */
  batchSize: number | null;
}

/** The lines of a comparison chart against `axis`, one for each of `runs` in their order. */
export function comparisonData(
  runs: ComparedRun[],
  chart: ComparisonChart,
  axis: XAxisId,
): ChartData {
  return alignLines(
    runs.map(({ records, batchSize }) =>
      line(records, placeRecords(records, axis, batchSize), chart.metric),
    ),
  );
}

/** The last value each series has, with the x it was logged at, for the caption. */
export function latestValues(data: ChartData): ({ x: number; value: number } | null)[] {
  const [xs, ...series] = data;
  return series.map((values) => {
    for (let i = values.length - 1; i >= 0; i -= 1) {
      const value = values[i];
      if (value != null) {
        return { x: xs[i], value };
      }
    }
    return null;
  });
}
