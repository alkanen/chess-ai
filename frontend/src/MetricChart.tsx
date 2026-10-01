import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { latestValues, type ChartData, type ChartSpec } from './runCharts';
import { formatCount } from './runFormat';
import './MetricChart.css';

const HEIGHT = 220;

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
    series: [token('--series-1', '#2a78d6'), token('--series-2', '#eb6834')],
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
  spec: ChartSpec;
  data: ChartData;
}

/**
 * One chart of a run's metrics against the step, drawn by uPlot.
 *
 * The chart is made once and handed new data as the run logs it. Dragging across it zooms
 * in, and a zoomed chart stays where it was put while new lines arrive, until a double
 * click lets it follow the run again.
 */
export function MetricChart({ spec, data }: MetricChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const zoomed = useRef(false);
  const dark = useDarkScheme();
  const latest = useMemo(() => latestValues(data), [data]);
  const empty = data[0].length === 0;
  // The newest data, for a chart made after it arrived.
  const current = useRef(data);
  current.current = data;

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
        { ...axis, values: (_, splits) => splits.map((step) => formatCount(step)) },
        { ...axis, size: 64, values: (_, splits) => splits.map((value) => spec.format(value)) },
      ],
      series: [
        { label: 'step', value: (_, step) => (step == null ? '–' : formatCount(step)) },
        ...spec.series.map((series, index) => ({
          label: series.label,
          stroke: palette.series[index],
          width: 2,
          // Validation is measured every so many steps, so its points are joined across the
          // steps it has nothing at. uPlot marks each point only while they are far enough
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
  }, [spec, dark, empty]);

  useEffect(() => {
    chart.current?.setData(data as uPlot.AlignedData, !zoomed.current);
  }, [data]);

  const caption = spec.series
    .map((series, index) => {
      const value = latest[index];
      return value === null
        ? `${series.label}: –`
        : `${series.label} ${spec.format(value.value)} at step ${formatCount(value.step)}`;
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
