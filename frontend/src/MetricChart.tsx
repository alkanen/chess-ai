import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
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

interface MetricChartProps {
  chart: ChartLook;
  /** What each line is called, in the order of the series in `data`. */
  labels: string[];
  x: XAxis;
  data: ChartData;
}

/**
 * One chart of metrics against `x`, drawn by uPlot: one run's, or several runs' side by side.
 *
 * The chart is made once and handed new data as the run logs it. Dragging across it zooms
 * in, and a zoomed chart stays where it was put while new lines arrive, until a double
 * click lets it follow the run again.
 */
export function MetricChart({ chart: spec, labels, x, data }: MetricChartProps) {
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
      scales: { x: { time: false }, y: spec.logarithmic ? { distr: 3 } : {} },
      axes: [
        {
          ...axis,
          values: (_, splits) => splits.map((value) => x.format(value)),
          ...(x.increments === undefined ? {} : { incrs: x.increments }),
        },
        { ...axis, size: 64, values: (_, splits) => splits.map((value) => spec.format(value)) },
      ],
      series: [
        { label: x.label, value: (_, at) => (at == null ? '–' : x.format(at)) },
        ...currentLabels.current.map((label, index) => ({
          label,
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
  }, [spec, x, named, dark, empty]);

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
