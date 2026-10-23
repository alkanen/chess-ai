import { render } from '@testing-library/react';
import { useLayoutEffect } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { LadderPoint } from './api';
import { EloChart, triangleCorners } from './EloChart';
import { FakeUPlot } from './test/fakeUPlot';
import { ladderPoint } from './test/ladder';

vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

const below = (step: number, high: number) =>
  ladderPoint(step, { rating: null, low: null, high, beyond: 'below_floor' });

/** What the legend says of series `series` at `index`, as uPlot asks it on a hover. */
function legend(chart: FakeUPlot, series: number, index: number): string {
  const value = chart.options.series[series].value as (...args: unknown[]) => string;
  return value(chart, 0, series, index);
}

describe('EloChart', () => {
  beforeEach(() => {
    FakeUPlot.instances = [];
  });

  it('describes a checkpoint past the ladder from the points the chart has, and none past them', () => {
    const { rerender } = render(<EloChart points={[below(1000, 994), below(2000, 1179)]} />);
    const chart = FakeUPlot.withSeries('Elo');
    expect(legend(chart, 4, 1)).toBe('below 1350, at most 1179');

    // Fewer points, as when another run's ladders arrive: a stale index has nothing to describe.
    rerender(<EloChart points={[below(3000, 1300)]} />);

    expect(chart.data[4]).toEqual([1300]);
    expect(legend(chart, 4, 0)).toBe('below 1350, at most 1300');
    expect(legend(chart, 4, 1)).toBe('–');
  });

  it('describes the points the chart is drawing until it is given the new ones', () => {
    const said: string[] = [];
    // Its layout effect runs once the chart has rendered new points and before the chart's
    // effects hand them to uPlot: when a hover would ask the legend about the old data.
    function Hovered({ points }: { points: LadderPoint[] }) {
      useLayoutEffect(() => {
        const [chart] = FakeUPlot.live();
        if (chart !== undefined) {
          said.push(`${chart.data[4].length} drawn: ${legend(chart, 4, 1)}`);
        }
      });
      return <EloChart points={points} />;
    }
    const { rerender } = render(<Hovered points={[below(1000, 994), below(2000, 1179)]} />);

    rerender(<Hovered points={[below(3000, 1300)]} />);

    expect(said).toEqual(['2 drawn: below 1350, at most 1179']);
  });

  it('hands the chart its data once when it is made', () => {
    const setData = vi.spyOn(FakeUPlot.prototype, 'setData');
    render(<EloChart points={[ladderPoint(1000), ladderPoint(2000)]} />);
    const chart = FakeUPlot.withSeries('Elo');

    const given = setData.mock.calls.length + (chart.madeWith?.[0]?.length ? 1 : 0);

    expect(given).toBe(1);
    expect(chart.data[0]).toEqual([1000, 2000]);
    setData.mockRestore();
  });

  it('puts no hover dot over a mark past the ladder, which would hide which way it points', () => {
    render(<EloChart points={[ladderPoint(1000)]} />);
    const chart = FakeUPlot.withSeries('Elo');
    const size = chart.options.cursor?.points?.size as (self: unknown, series: number) => number;
    const labels = chart.options.series.map((series) => series.label);

    expect(size(chart, labels.indexOf('Below the ladder'))).toBe(0);
    expect(size(chart, labels.indexOf('Above the ladder'))).toBe(0);
    expect(size(chart, labels.indexOf('Elo'))).toBe(8);
  });

  it('draws a mark past the ladder from the bound, pointing past it, even on whole pixels', () => {
    expect(triangleCorners(100.4, 50.6, 11, true)).toEqual([
      [94, 51],
      [106, 51],
      [100, 62],
    ]);
    expect(triangleCorners(100.4, 50.6, 11, false)).toEqual([
      [94, 51],
      [106, 51],
      [100, 40],
    ]);
  });
});
