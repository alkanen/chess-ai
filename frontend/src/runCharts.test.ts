import { describe, expect, it } from 'vitest';
import type { MetricsRecord } from './api';
import {
  CHARTS,
  COMPARISON_CHARTS,
  chartData,
  comparisonData,
  formatPositions,
  gradientClip,
  latestValues,
  logs,
  placeRecords,
} from './runCharts';
import { formatAgo, formatDuration, formatNumber, formatPercent, runState } from './runFormat';

const LOSS = CHARTS.find((chart) => chart.id === 'loss')!;

describe('chartData', () => {
  it('lines up training and validation on the steps either was logged at', () => {
    const records: MetricsRecord[] = [
      { step: 50, split: 'train', loss: 3.0, learning_rate: 0.001 },
      { step: 100, split: 'train', loss: 2.5 },
      { step: 100, split: 'validation', loss: 2.8, top1: 0.2 },
      { step: 150, split: 'train', loss: 2.2 },
    ];

    expect(chartData(records, LOSS)).toEqual([
      [50, 100, 150],
      [3.0, 2.5, 2.2],
      [null, 2.8, null],
    ]);
  });

  it('keeps a diverged value as a gap rather than dropping the step', () => {
    const records: MetricsRecord[] = [
      { step: 1, split: 'train', loss: 3.0 },
      { step: 2, split: 'train', loss: null },
    ];

    expect(chartData(records, LOSS)).toEqual([
      [1, 2],
      [3.0, null],
      [null, null],
    ]);
  });

  it('has no steps for a log with nothing the chart draws', () => {
    const accuracy = CHARTS.find((chart) => chart.id === 'accuracy')!;

    expect(chartData([{ step: 1, split: 'train', loss: 3.0 }], accuracy)).toEqual([[], [], []]);
  });

  it('draws a level across the chart wherever its first line has a value', () => {
    const gradient = CHARTS.find((chart) => chart.id === 'gradient-norm')!;
    const records: MetricsRecord[] = [
      { step: 50, split: 'train', gradient_norm: 2.5 },
      { step: 100, split: 'validation', loss: 2.0 },
      { step: 100, split: 'train', gradient_norm: 0.8 },
    ];

    expect(chartData(records, gradient, 'step', null, 1.0)).toEqual([
      [50, 100],
      [2.5, 0.8],
      [1.0, 1.0],
    ]);
    expect(chartData(records, gradient)).toEqual([
      [50, 100],
      [2.5, 0.8],
    ]);
  });

  it('says whether a log has a metric at all, which an optional chart is shown for', () => {
    const records: MetricsRecord[] = [{ step: 50, split: 'train', loss: 3.0 }];

    expect(logs(records, { split: 'train', metric: 'loss' })).toBe(true);
    expect(logs(records, { split: 'train', metric: 'gradient_norm' })).toBe(false);
    expect(logs(records, { split: 'validation', metric: 'loss' })).toBe(false);
  });

  it('finds the last value of each series, wherever it was logged', () => {
    expect(
      latestValues([
        [1, 2, 3],
        [5, 4, null],
        [null, null, null],
      ]),
    ).toEqual([{ x: 2, value: 4 }, null]);
  });
});

describe('placeRecords', () => {
  const records: MetricsRecord[] = [
    { step: 10, split: 'train', loss: 3.0, elapsed: 5, positions_seen: 160 },
    { step: 10, split: 'validation', loss: 3.2, elapsed: 5, positions_seen: 160 },
    { step: 20, split: 'train', loss: 2.5, elapsed: 11, positions_seen: 320 },
  ];

  it('places each line by its step, the positions seen, or the seconds it was logged at', () => {
    expect(placeRecords(records, 'step')).toEqual([10, 10, 20]);
    expect(placeRecords(records, 'positions')).toEqual([160, 160, 320]);
    expect(placeRecords(records, 'time')).toEqual([5, 5, 11]);
  });

  it('makes up for a log too old to say how many positions it had seen', () => {
    const old: MetricsRecord[] = [
      { step: 10, split: 'train', loss: 3.0, elapsed: 5 },
      { step: 10, split: 'validation', loss: 3.2 },
      { step: 15, split: 'validation', loss: 3.1 },
      { step: 20, split: 'train', loss: 2.5, elapsed: 11 },
    ];

    expect(placeRecords(old, 'positions', 16)).toEqual([160, 160, 240, 320]);
    // Without a batch size there is nothing to make it up from.
    expect(placeRecords(old, 'positions', null)).toEqual([null, null, null, null]);
    // Validation lines without seconds of their own take those of the line before them.
    expect(placeRecords(old, 'time')).toEqual([5, 5, 5, 11]);
  });

  it('leaves out what the log cannot place', () => {
    const old: MetricsRecord[] = [{ step: 10, split: 'validation', loss: 3.2 }];

    expect(chartData(old, LOSS, 'time')).toEqual([[], [], []]);
  });

  it('charts a run against wall time', () => {
    expect(chartData(records, LOSS, 'time')).toEqual([
      [5, 11],
      [3.0, 2.5],
      [3.2, null],
    ]);
  });
});

describe('comparisonData', () => {
  const TRAIN_LOSS = COMPARISON_CHARTS.find((chart) => chart.id === 'train-loss')!;

  it('gives each run a line, lined up on every x any of them was logged at', () => {
    const small: MetricsRecord[] = [
      { step: 1, split: 'train', loss: 4.0, positions_seen: 100 },
      { step: 2, split: 'train', loss: 3.0, positions_seen: 200 },
    ];
    const big: MetricsRecord[] = [
      { step: 1, split: 'train', loss: 3.5, positions_seen: 200 },
      { step: 1, split: 'validation', loss: 3.9, positions_seen: 200 },
    ];
    const runs = [
      { records: small, batchSize: 100 },
      { records: big, batchSize: 200 },
    ];

    expect(comparisonData(runs, TRAIN_LOSS, 'step')).toEqual([
      [1, 2],
      [4.0, 3.0],
      [3.5, null],
    ]);
    expect(comparisonData(runs, TRAIN_LOSS, 'positions')).toEqual([
      [100, 200],
      [4.0, 3.0],
      [null, 3.5],
    ]);
  });

  it('has a line for a run that has logged nothing yet', () => {
    expect(comparisonData([{ records: [], batchSize: null }], TRAIN_LOSS, 'step')).toEqual([[], []]);
  });
});

describe('runState', () => {
  it('takes a run at its word unless it says it is running and has stopped saying so', () => {
    expect(runState('running', false)).toBe('running');
    expect(runState('running', true)).toBe('stale');
    expect(runState('finished', false)).toBe('finished');
    expect(runState('crashed', undefined)).toBe('crashed');
    expect(runState(null, false)).toBe('starting');
  });
});

describe('formatting', () => {
  it('writes numbers the way a chart reader wants them', () => {
    expect(formatNumber(2.512345)).toBe('2.512');
    expect(formatNumber(0.0002)).toBe('2.00e-4');
    expect(formatNumber(null)).toBe('–');
    expect(formatPercent(0.4594)).toBe('45.9%');
    expect(formatDuration(3900)).toBe('1h 05m');
    expect(formatDuration(75)).toBe('1m 15s');
    expect(formatPositions(10_000_000)).toBe('10M');
    expect(formatPositions(12_288_000)).toBe('12.3M');
    expect(formatPositions(2_500_000_000)).toBe('2.5G');
    expect(formatPositions(800)).toBe('800');
    expect(formatAgo('2026-10-01T12:00:00Z', Date.parse('2026-10-01T12:00:30Z'))).toBe(
      '30s ago',
    );
  });
});

describe('gradientClip', () => {
  const run = {
    name: 'a',
    created: '2026-10-01T10:00:00Z',
    seed: 1,
    code_version: 'test',
    device: 'cpu',
    dataset: { name: 'd', positions: 1, train_positions: 1 },
    model: { architecture: 'mlp', options: {}, parameter_count: 1 },
    steps: 1,
    batch_size: 1,
  };

  it('is the threshold a run clips at, and nothing for a run that does not clip', () => {
    expect(gradientClip({ ...run, config: { optimizer: { gradient_clip: 1.0 } } })).toBe(1.0);
    expect(gradientClip({ ...run, config: { optimizer: { gradient_clip: 0 } } })).toBeNull();
    expect(gradientClip(run)).toBeNull();
  });
});
