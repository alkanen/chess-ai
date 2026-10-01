import type { MetricsRecord } from './api';
import { formatNumber, formatPercent } from './runFormat';

/** One line on a chart: a metric from the lines of one split of the log. */
export interface SeriesSpec {
  label: string;
  split: 'train' | 'validation';
  metric: string;
}

export interface ChartSpec {
  id: string;
  title: string;
  series: SeriesSpec[];
  format: (value: number) => string;
  /** Whether the y-axis is logarithmic, which keeps a loss's early drop from flattening the rest. */
  logarithmic?: boolean;
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
    format: (value) => formatNumber(value),
    logarithmic: true,
  },
  {
    id: 'accuracy',
    title: 'Accuracy (validation)',
    series: [
      { label: 'top-1', split: 'validation', metric: 'top1' },
      { label: 'top-5', split: 'validation', metric: 'top5' },
    ],
    format: formatPercent,
  },
  {
    id: 'learning-rate',
    title: 'Learning rate',
    series: [{ label: 'learning rate', split: 'train', metric: 'learning_rate' }],
    format: (value) => formatNumber(value, 3),
  },
  {
    id: 'illegal',
    title: 'Illegal top-move rate (validation)',
    series: [{ label: 'illegal top move', split: 'validation', metric: 'illegal_top_move_rate' }],
    format: formatPercent,
  },
];

/**
 * A chart's lines as uPlot takes them: the steps, and each series' value at every one of
 * them. Validation is measured far less often than training is logged, so a series has a
 * null wherever it has nothing at that step, as it does where a value diverged.
 */
export type ChartData = [number[], ...(number | null)[][]];

export function chartData(records: MetricsRecord[], spec: ChartSpec): ChartData {
  const byStep = new Map<number, (number | null)[]>();
  for (const record of records) {
    spec.series.forEach((series, index) => {
      if (record.split !== series.split || !(series.metric in record)) {
        return;
      }
      const value = record[series.metric];
      let values = byStep.get(record.step);
      if (values === undefined) {
        values = spec.series.map(() => null);
        byStep.set(record.step, values);
      }
      values[index] = typeof value === 'number' ? value : null;
    });
  }
  const steps = [...byStep.keys()].sort((a, b) => a - b);
  return [
    steps,
    ...spec.series.map((_, index) => steps.map((step) => byStep.get(step)![index])),
  ];
}

/** The last value each series has, with the step it was logged at, for the caption. */
export function latestValues(data: ChartData): ({ step: number; value: number } | null)[] {
  const [steps, ...series] = data;
  return series.map((values) => {
    for (let i = values.length - 1; i >= 0; i -= 1) {
      const value = values[i];
      if (value != null) {
        return { step: steps[i], value };
      }
    }
    return null;
  });
}
