import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { logRange, logTicks } from './chartScale';
import { latestValues, type ChartData, type ChartLook, type XAxis } from './runCharts';
import './MetricChart.css';

const HEIGHT = 220;

/** The categorical slots, in the order lines take them, for a page without the tokens. */
const SERIES_FALLBACKS = [
  '#2a78d6',
  '#eb6834',
  '#1baf7a',
  '#eda100',
  '#e87ba4',
  '#008300',
  '#4a3aa7',
  '#e34948',
];

/** The colours a chart is drawn in, read from the CSS tokens of the scheme in use. */
interface Palette {
  series: string[];
  text: string;
  grid: string;
}

function readPalette(element: HTMLElement): Palette {
  const style = getComputedStyle(element);
  const token = (name: string, fallback: string) =>
    style.getPropertyValue(name).trim() || fallback;
  return {
    series: SERIES_FALLBACKS.map((fallback, index) => token(`--series-${index + 1}`, fallback)),
    text: token('--chart-text', '#52514e'),
    grid: token('--chart-grid', 'rgb(128 128 128 / 20%)'),
  };
}

/** Whether the page is in its dark scheme, followed as the system setting changes. */
function useDarkScheme(): boolean {
  const [dark, setDark] = useState(() => darkQuery()?.matches ?? false);
  useEffect(() => {
    const query = darkQuery();
    if (query === null) {
      return;
    }
    const follow = () => setDark(query.matches);
    query.addEventListener('change', follow);
    return () => query.removeEventListener('change', follow);
  }, []);
  return dark;
}

function darkQuery(): MediaQueryList | null {
  return typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;
}

/** The least distance between ticks next to each other, which their labels must tell apart. */
function smallestGap(ticks: number[]): number {
  let gap = Infinity;
  for (let index = 1; index < ticks.length; index += 1) {
    gap = Math.min(gap, Math.abs(ticks[index] - ticks[index - 1]));
  }
  return gap;
}

/**
 * The least and greatest of the levels shown on `chart`, its last `levels` series, for a
 * y-axis with no data series shown to fit: the viewer has hidden them in the legend.
 */
function shownLevels(chart: uPlot, levels: number): [number | null, number | null] {
  let least: number | null = null;
  let greatest: number | null = null;
  for (let index = chart.series.length - levels; index < chart.series.length; index += 1) {
    if (chart.series[index].show === false) {
      continue;
    }
    for (const value of chart.data[index]) {
      if (value != null) {
        least = least === null ? value : Math.min(least, value);
        greatest = greatest === null ? value : Math.max(greatest, value);
      }
    }
  }
  return [least, greatest];
}

interface MetricChartProps {
  chart: ChartLook;
  /** What each line is called, in the order of the series in `data`. */
  labels: string[];
  x: XAxis;
  data: ChartData;
  /**
   * How many of the last series are levels from the run's settings rather than its log,
   * which the y-axis is not fitted to: a gradient norm far below where it would be clipped
   * would otherwise be squashed at the bottom of the chart.
   */
  levels?: number;
}

/**
 * One chart of metrics against `x`, drawn by uPlot: one run's, or several runs' side by side.
 *
 * The chart is made once and handed new data as the run logs it. Dragging across it zooms
 * in, and a zoomed chart stays where it was put while new lines arrive, until a double
 * click lets it follow the run again.
 */
export function MetricChart({ chart: spec, labels, x, data, levels = 0 }: MetricChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const zoomed = useRef(false);
  const dark = useDarkScheme();
  const latest = useMemo(() => latestValues(data), [data]);
  const empty = data[0].length === 0;
  // The newest data, for a chart made after it arrived.
  const current = useRef(data);
  current.current = data;
  // The labels as a value rather than an array, so that a new array saying the same thing does
  // not make the chart again; a renamed run does.
  const named = labels.join('\n');
  const currentLabels = useRef(labels);
  currentLabels.current = labels;

  useEffect(() => {
    const element = container.current;
    if (element === null || empty) {
      return;
    }
    const palette = readPalette(element);
    const axis = {
      stroke: palette.text,
      grid: { stroke: palette.grid, width: 1 },
      ticks: { stroke: palette.grid, width: 1 },
    };
    const options: uPlot.Options = {
      width: element.clientWidth || 600,
      height: HEIGHT,
      // uPlot fits a linear y-axis to the data in view itself; a logarithmic one it would round
      // out to powers of ten, so it is given a range and ticks that fit the data instead. Levels
      // are left out of the fit, so where only levels are shown it fits them instead.
      scales: {
        x: { time: false },
        y: {
          ...(spec.logarithmic ? { distr: 3 } : {}),
          ...(spec.logarithmic || levels > 0
            ? {
                // uPlot's types say otherwise, but it passes nulls for a y-axis with no data.
                range: (made: uPlot, dataMin: number | null, dataMax: number | null) => {
                  let [min, max] = [dataMin, dataMax];
                  if (min === null && levels > 0) {
                    [min, max] = shownLevels(made, levels);
                  }
                  if (spec.logarithmic) {
                    return logRange(min, max);
                  }
                  // What uPlot gives a linear y-axis of its own.
                  return min === null || max === null
                    ? [null, null]
                    : uPlot.rangeNum(min, max, 0.1, true);
                },
              }
            : {}),
        },
      },
      axes: [
        {
          ...axis,
          values: (_, splits) => splits.map((value) => x.format(value)),
          ...(x.increments === undefined ? {} : { incrs: x.increments }),
        },
        {
          ...axis,
          size: 64,
          values: (_, splits) => {
            const step = smallestGap(splits);
            return splits.map((value) => spec.formatTick(value, step));
          },
          ...(spec.logarithmic
            ? { splits: (_, __, min, max) => logTicks(min, max), filter: (_, splits) => splits }
            : {}),
        },
      ],
      series: [
        { label: x.label, value: (_, at) => (at == null ? '–' : x.format(at)) },
        ...currentLabels.current.map((label, index, all) => ({
          label,
          auto: index < all.length - levels,
          stroke: palette.series[index],
          width: 2,
          // Validation is measured every so many steps, and runs side by side are logged at
          // x values of their own, so each line is joined across the x values it has nothing
          // at. uPlot marks each point only while they are far enough
          // apart to tell apart; a long run's hundreds of them would be a solid band.
          spanGaps: true,
          points: { size: 8 },
          value: (_: uPlot, value: number | null) => (value == null ? '–' : spec.format(value)),
        })),
      ],
      hooks: { setSelect: [() => (zoomed.current = true)] },
    };
    // A new chart fits whatever it is made with, so a zoom on the chart it replaces is gone.
    zoomed.current = false;
    const made = new uPlot(options, current.current as uPlot.AlignedData, element);
    chart.current = made;
    const unzoom = () => (zoomed.current = false);
    made.over?.addEventListener('dblclick', unzoom);

    const resized =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => made.setSize({ width: element.clientWidth, height: HEIGHT }));
    resized?.observe(element);
    return () => {
      resized?.disconnect();
      made.over?.removeEventListener('dblclick', unzoom);
      made.destroy();
      chart.current = null;
    };
  }, [spec, x, named, dark, empty, levels]);

  useEffect(() => {
    chart.current?.setData(data as uPlot.AlignedData, !zoomed.current);
  }, [data]);

  const caption = labels
    .map((label, index) => {
      const value = latest[index];
      return value === null
        ? `${label}: –`
        : `${label} ${spec.format(value.value)} at ${x.describe(value.x)}`;
    })
    .join('; ');

  return (
    <figure className="metric-chart" aria-labelledby={`chart-${spec.id}`}>
      <figcaption id={`chart-${spec.id}`}>
        <span className="title">{spec.title}</span>
        <span className="latest">{empty ? 'Nothing logged yet' : caption}</span>
      </figcaption>
      {!empty && <div className="plot" ref={container} />}
    </figure>
  );
}
