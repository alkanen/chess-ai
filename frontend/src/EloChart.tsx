import { useEffect, useMemo, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { LadderPoint } from './api';
import { describeEstimate, ELO_SERIES, eloChart, ladderLimits } from './ladder';
import { touchToZoom, useZoomed } from './chartZoom';
import { readPalette, ResetZoom, useDarkScheme } from './MetricChart';
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

/**
 * The corners of a mark for a checkpoint beyond the ladder, in canvas pixels: its base on the
 * bound at (`x`, `y`) and its tip `size` past it, down for one below the ladder and up for one
 * above, since the checkpoint lies past the bound rather than on it. Whole pixels, as uPlot
 * puts its own points, so the mark is as sharp as they are.
 */
export function triangleCorners(
  x: number,
  y: number,
  size: number,
  down: boolean,
): [number, number][] {
  // The centre rounded first and the sides a whole number from it, so that an odd size
  // still gives a mark with its tip over the middle of its base.
  const left = Math.round(x);
  const top = Math.round(y);
  const half = Math.round(size / 2);
  return [
    [left - half, top],
    [left + half, top],
    [left, Math.round(down ? top + size : top - size)],
  ];
}

/** Draws a series' points as the marks of :func:`triangleCorners`. */
function triangles(down: boolean): uPlot.Series.Points.PathBuilder {
  return (chart, seriesIdx, idx0, idx1) => {
    const path = new Path2D();
    const xs = chart.data[0];
    const ys = chart.data[seriesIdx];
    for (let index = idx0; index <= idx1; index++) {
      const y = ys[index];
      if (y == null) {
        continue;
      }
      const [first, ...rest] = triangleCorners(
        chart.valToPos(xs[index], 'x', true),
        chart.valToPos(y, 'y', true),
        MARK_SIZE * devicePixelRatio,
        down,
      );
      path.moveTo(...first);
      rest.forEach((corner) => path.lineTo(...corner));
      path.closePath();
    }
    return { stroke: path, fill: path };
  };
}

const MARK_SIZE = 11;

/** The series drawn as marks beyond the ladder, below it and above it, by label. */
const BEYOND_SERIES = new Set([ELO_SERIES[3], ELO_SERIES[4]]);

type Charted = ReturnType<typeof eloChart>;

/**
 * Hands `made` the data of `charted`, and notes in `given` that it has it: the legend describes
 * the points of what was given, and nothing is handed over twice. The one way data reaches the
 * chart.
 */
function draw(made: uPlot, given: { current: Charted | null }, charted: Charted, rescale: boolean) {
  given.current = charted;
  made.setData(charted.data as uPlot.AlignedData, rescale);
}

interface EloChartProps {
  points: LadderPoint[];
}

/**
 * The checkpoints' Elo over training, with the range each is good to as a band, and those
 * beyond the ladder marked with a triangle pointing past the bound they are drawn at.
 *
 * Kept apart from the metric charts, which have several x-axes to choose from and log-scaled
 * series: a ladder is about a checkpoint, which is a step. It zooms as they do, and a zoomed
 * chart stays where it was put as new checkpoints are laddered.
 */
export function EloChart({ points }: EloChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const dark = useDarkScheme();
  const [zoomed, zoomedNow, setZoomed] = useZoomed();
  const charted = useMemo(() => eloChart(points), [points]);
  const empty = points.length === 0;
  const current = useRef(charted);
  current.current = charted;
  // What uPlot has now, whose points are what an index from it means. Not the latest: the
  // chart is given those only once the render is done, and is hovered with the old ones until
  // then.
  const given = useRef<Charted | null>(null);

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
        ...[ELO_SERIES[3], ELO_SERIES[4]].map((label, which) => ({
          label,
          stroke: beyond,
          // Marks alone: a checkpoint below the ladder is somewhere below it, not on a line.
          width: 0,
          paths: () => null,
          points: { show: true, size: MARK_SIZE, fill: beyond, paths: triangles(which === 0) },
          // The bound alone would read as a rating, which it is not.
          value: (_: uPlot, value: number | null, _series: number, index: number | null) => {
            const point =
              value == null || index == null ? undefined : given.current?.points[index];
            return point === undefined ? '–' : describeEstimate(point.estimate);
          },
        })),
      ],
      bands: [{ series: [3, 2], fill: translucent(elo, 0.15) }],
      cursor: {
        points: {
          // uPlot's hover dot is a disc of the point's size, which would cover a mark's
          // direction just as it is pointed at; the legend says what it is instead.
          size: (made, series) =>
            BEYOND_SERIES.has(made.series[series].label as string)
              ? 0
              : (made.series[series].points?.size ?? 0),
        },
      },
      hooks: { setSelect: [() => setZoomed(true)] },
    };
    setZoomed(false);
    const made = new uPlot(options, undefined, element);
    draw(made, given, current.current, true);
    chart.current = made;
    const unzoom = () => setZoomed(false);
    made.over.addEventListener('dblclick', unzoom);
    const untouch = touchToZoom(made);
    const resized =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => made.setSize({ width: element.clientWidth, height: HEIGHT }));
    resized?.observe(element);
    return () => {
      resized?.disconnect();
      untouch();
      made.over.removeEventListener('dblclick', unzoom);
      made.destroy();
      chart.current = null;
    };
  }, [empty, dark, setZoomed]);

  useEffect(() => {
    // Not when the chart was made with these just now, in the same commit.
    if (chart.current !== null && given.current !== charted) {
      draw(chart.current, given, charted, !zoomedNow.current);
    }
  }, [charted, zoomedNow]);

  const resetZoom = () => {
    if (chart.current !== null) {
      draw(chart.current, given, current.current, true);
    }
    setZoomed(false);
  };

  const latest = points.length === 0 ? null : points[points.length - 1];
  const limits = ladderLimits(points);
  return (
    <figure className="metric-chart elo-chart" aria-labelledby="elo-chart-title">
      <figcaption id="elo-chart-title">
        <span className="title">
          Elo on Stockfish’s scale at {limits.join(' and ')}
        </span>
        <span className="latest">
          {latest === null
            ? 'No ladder yet'
            : `${describeEstimate(latest.estimate)} at step ${formatCount(latest.step)}, ` +
              `a ${Math.round(latest.estimate.confidence * 100)}% range`}
        </span>
      </figcaption>
      {limits.length > 1 && (
        <p className="note">
          These ladders were played with Stockfish held to different limits, at which it plays
          differently strong: compare ratings measured at one limit only.
        </p>
      )}
      {!empty && (
        <div className="plot-area">
          <div className="plot" ref={container} />
          {zoomed && <ResetZoom onClick={resetZoom} />}
        </div>
      )}
    </figure>
  );
}
