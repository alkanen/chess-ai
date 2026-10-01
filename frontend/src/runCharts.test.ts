import { describe, expect, it } from 'vitest';
import type { MetricsRecord } from './api';
import { CHARTS, chartData, latestValues } from './runCharts';
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

  it('finds the last value of each series, wherever it was logged', () => {
    expect(
      latestValues([
        [1, 2, 3],
        [5, 4, null],
        [null, null, null],
      ]),
    ).toEqual([{ step: 2, value: 4 }, null]);
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
    expect(formatAgo('2026-10-01T12:00:00Z', Date.parse('2026-10-01T12:00:30Z'))).toBe(
      '30s ago',
    );
  });
});
