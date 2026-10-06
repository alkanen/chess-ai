import { useId } from 'react';
import type { DatasetManifest } from './api';
import { formatCount, formatPercent } from './runFormat';

/** How wide a bar of the rating histogram is; mirrors chess_ai.dataset.manifest.RATING_BUCKET. */
export const RATING_BUCKET = 100;

/** What each result is called, from white's side, in the order they are listed. */
const RESULTS: [string, string][] = [
  ['1-0', 'White won'],
  ['1/2-1/2', 'Draw'],
  ['0-1', 'Black won'],
];

/** One bar of a distribution: what it counts, and how many. */
export type Bar = [label: string, count: number];

/**
 * The rating histogram, one bar per bucket from the lowest rating to the highest, the empty
 * buckets between them included: a gap is part of the shape, and the shape is the point.
 * Players with no rating come last, as a bar of their own.
 */
export function ratingBars(statistics: DatasetManifest['statistics']): Bar[] {
  const buckets = Object.keys(statistics.ratings).map(Number);
  const bars: Bar[] = [];
  if (buckets.length > 0) {
    for (let low = Math.min(...buckets); low <= Math.max(...buckets); low += RATING_BUCKET) {
      bars.push([`${low}–${low + RATING_BUCKET - 1}`, statistics.ratings[String(low)] ?? 0]);
    }
  }
  if (statistics.ratings_unknown > 0) {
    bars.push(['unknown', statistics.ratings_unknown]);
  }
  return bars;
}

/** Results in the order white won, drawn, black won, and anything else after them. */
export function resultBars(results: Record<string, number>): Bar[] {
  const known = RESULTS.filter(([result]) => result in results).map(
    ([result, label]): Bar => [label, results[result]],
  );
  const other = Object.entries(results).filter(([result]) => !RESULTS.some(([r]) => r === result));
  return [...known, ...other];
}

/** Counts as bars in the order the manifest gives them, except that "unknown" comes last. */
export function unknownLast(counts: Record<string, number>): Bar[] {
  const bars = Object.entries(counts);
  const unknown = bars.filter(([name]) => name === 'unknown');
  return [...bars.filter(([name]) => name !== 'unknown'), ...unknown];
}

/** A name the manifest writes with underscores, as words: "unsupported_variant" as such. */
function words(name: string): string {
  return name.replaceAll('_', ' ');
}

interface DistributionProps {
  title: string;
  bars: Bar[];
  /** What is counted, for the share column's heading and the empty case: "games", "players". */
  of: string;
}

/**
 * A distribution as a table whose last column is a bar: the counts and shares can be read off
 * exactly, and the bars say at a glance what the numbers only say on a second look.
 */
export function Distribution({ title, bars, of }: DistributionProps) {
  const total = bars.reduce((sum, [, count]) => sum + count, 0);
  const longest = Math.max(0, ...bars.map(([, count]) => count));
  const caption = useId();
  return (
    <figure className="distribution">
      <figcaption id={caption}>{title}</figcaption>
      {total === 0 ? (
        <p className="note">No {of}.</p>
      ) : (
        <table aria-labelledby={caption}>
          <thead>
            <tr>
              <th scope="col">{title}</th>
              <th scope="col" className="number">
                {of[0].toUpperCase() + of.slice(1)}
              </th>
              <th scope="col" className="number">
                Share
              </th>
              <th scope="col" aria-hidden="true" />
            </tr>
          </thead>
          <tbody>
            {bars.map(([label, count]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td className="number">{formatCount(count)}</td>
                <td className="number">{formatPercent(count / total)}</td>
                <td className="bar" aria-hidden="true">
                  {count > 0 && <span style={{ width: `${(count / longest) * 100}%` }} />}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </figure>
  );
}

/** What a dataset is made of: how strong the players were, how games ended, how fast. */
export function DatasetStatistics({ manifest }: { manifest: DatasetManifest }) {
  const { statistics } = manifest;
  return (
    <section className="dataset-statistics" aria-labelledby="dataset-statistics-heading">
      <h3 id="dataset-statistics-heading">Statistics</h3>
      <div className="distributions">
        <Distribution title="Rating" of="players" bars={ratingBars(statistics)} />
        <div className="stack">
          <Distribution title="Result" of="games" bars={resultBars(statistics.results)} />
          <Distribution
            title="Time control"
            of="games"
            bars={unknownLast(statistics.time_controls)}
          />
          <Distribution
            title="Rating source"
            of="games"
            bars={unknownLast(statistics.rating_sources)}
          />
          <Distribution
            title="Skipped"
            of="games"
            bars={Object.entries(manifest.skipped).map(([reason, count]): Bar => [
              words(reason),
              count,
            ])}
          />
        </div>
      </div>
    </section>
  );
}
