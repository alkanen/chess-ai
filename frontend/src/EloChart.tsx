import { useEffect, useMemo, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { LadderPoint } from './api';
import { describeEstimate, ELO_SERIES, eloChartData, moveTimes } from './ladder';
import { readPalette, useDarkScheme } from './MetricChart';
import { formatCount } from './runFormat';
import './MetricChart.css';

const HEIGHT = 220;

/** A colour of the palette, made see-through for the band its range is drawn as. */
function translucent(color: string, alpha: number): string {
  const hex = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(color);
  if (hex === null) {
    return `color-mix(in srgb, ${color} ${Math.round(alpha * 100)}%, transparent)`;
  }
  const [red, green, blue] = hex.slice(1).map((part) => parseInt(part, 16));
  return `rgb(${red} ${green} ${blue} / ${alpha})`;
}

interface EloChartProps {
  points: LadderPoint[];
}

/**
 * The checkpoints' Elo over training, with the range each is good to as a band, and those
 * beyond the ladder marked at the end of it they are past.
 *
 * Kept apart from the metric charts, which have several x-axes to choose from and log-scaled
 * series: a ladder is about a checkpoint, which is a step.
 */
export function EloChart({ points }: EloChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const dark = useDarkScheme();
  const data = useMemo(() => eloChartData(points), [points]);
  const empty = points.length === 0;
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
    const [elo, beyond] = palette.series;
    const rounded = (_: uPlot, value: number | null) =>
      value == null ? '–' : String(Math.round(value));
    const options: uPlot.Options = {
      width: element.clientWidth || 600,
      height: HEIGHT,
      scales: { x: { time: false } },
      axes: [
        { ...axis, values: (_, splits) => splits.map((value) => formatCount(value)) },
        { ...axis, size: 56 },
      ],
      series: [
        { label: 'Step', value: (_, at) => (at == null ? '–' : formatCount(at)) },
        { label: ELO_SERIES[0], stroke: elo, width: 2, points: { size: 8 }, value: rounded },
        // The range is the band between these two, which say what its ends are on hover.
        { label: ELO_SERIES[1], stroke: translucent(elo, 0.5), width: 1, value: rounded },
        { label: ELO_SERIES[2], stroke: translucent(elo, 0.5), width: 1, value: rounded },
        ...[ELO_SERIES[3], ELO_SERIES[4]].map((label) => ({
          label,
          stroke: beyond,
          // Marks alone: a checkpoint below the ladder is somewhere below it, not on a line.
          width: 0,
          paths: () => null,
          points: { show: true, size: 9, fill: beyond },
          value: rounded,
        })),
      ],
      bands: [{ series: [3, 2], fill: translucent(elo, 0.15) }],
    };
    const made = new uPlot(options, current.current as uPlot.AlignedData, element);
    chart.current = made;
    const resized =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => made.setSize({ width: element.clientWidth, height: HEIGHT }));
    resized?.observe(element);
    return () => {
      resized?.disconnect();
      made.destroy();
      chart.current = null;
    };
  }, [empty, dark]);

  useEffect(() => {
    chart.current?.setData(data as uPlot.AlignedData);
  }, [data]);

  const latest = points.length === 0 ? null : points[points.length - 1];
  const times = moveTimes(points);
  return (
    <figure className="metric-chart elo-chart" aria-labelledby="elo-chart-title">
      <figcaption id="elo-chart-title">
        <span className="title">
          Elo on Stockfish’s scale at {times.map((time) => `${time} s`).join(' and ')} a move
        </span>
        <span className="latest">
          {latest === null
            ? 'No ladder yet'
            : `${describeEstimate(latest.estimate)} at step ${formatCount(latest.step)}, ` +
              `a ${Math.round(latest.estimate.confidence * 100)}% range`}
        </span>
      </figcaption>
      {times.length > 1 && (
        <p className="note">
          These ladders were played at different move times, at which Stockfish plays differently
          strong: compare ratings measured at one move time only.
        </p>
      )}
      {!empty && <div className="plot" ref={container} />}
    </figure>
  );
}
