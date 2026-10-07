import { useEffect, useState } from 'react';
import {
  fetchDatasets,
  type DatasetFilters,
  type DatasetManifest,
  type DatasetSummary,
} from './api';
import { datasetHash, DEFAULT_SPLIT } from './datasetPlaces';
import { formatCount } from './runFormat';
import './DatasetsView.css';

/** When a dataset was built, in the viewer's own time. */
export function formatBuilt(created: string): string {
  return new Date(created).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

/** The last part of a path, which is what tells one source file from the next. */
export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

/** How many games a dataset holds, which is its splits added up. */
export function totalOf(manifest: DatasetManifest, what: 'games' | 'positions'): number {
  return Object.values(manifest.splits).reduce((sum, split) => sum + split[what], 0);
}

/**
 * The filters a dataset was built with, a phrase per filter; mirrors describe_filters in
 * chess_ai.dataset.summary, which says the same on the command line.
 */
export function filterPhrases(filters: DatasetFilters): string[] {
  const phrases: string[] = [];
  const { min_rating: min, max_rating: max } = filters;
  if (min !== undefined || max !== undefined) {
    const band =
      max === undefined
        ? `at least ${min}`
        : min === undefined
          ? `at most ${max}`
          : `${min} to ${max}`;
    const unknown = filters.unknown_rating_passes ? 'passes' : 'does not';
    phrases.push(`player to move rated ${band}; an unknown rating ${unknown}`);
  }
  if (filters.min_clock !== undefined) {
    phrases.push(`player to move with at least ${filters.min_clock}s on the clock`);
  }
  if (filters.time_controls !== undefined) {
    phrases.push(`time controls ${filters.time_controls.join(', ')}`);
  }
  if (filters.exclude_terminations !== undefined && filters.exclude_terminations.length > 0) {
    const names = filters.exclude_terminations.map((name) => name.replaceAll('_', ' '));
    phrases.push(`not ended by ${names.join(', ')}`);
  }
  if (filters.from !== undefined || filters.until !== undefined) {
    const from = filters.from !== undefined ? ` from ${filters.from}` : '';
    const until = filters.until !== undefined ? ` until ${filters.until}` : '';
    phrases.push(`played${from}${until}`);
  }
  if (filters.sample !== undefined) {
    phrases.push(`a ${filters.sample} sample, by a hash of each game`);
  }
  if (filters.max_games !== undefined) {
    phrases.push(`at most ${formatCount(filters.max_games)} games`);
  }
  return phrases;
}

/** The filters a dataset was built with, in a few words. */
export function describeFilters(filters: DatasetFilters): string {
  const phrases = filterPhrases(filters);
  return phrases.length === 0 ? 'none' : phrases.join('; ');
}

/** How many of a dataset's positions are trained on, or null when every one of them is. */
export function trainedOn(manifest: DatasetManifest): number | null {
  const splits = Object.values(manifest.splits);
  if (splits.every((split) => split.targets == null)) {
    return null;
  }
  return splits.reduce((sum, split) => sum + (split.targets ?? split.positions), 0);
}

/** Whether a source was not read whole, which leaves the dataset short of its games. */
export function sourceFailed(manifest: DatasetManifest): boolean {
  return manifest.sources.some((source) => source.error !== null);
}

/**
 * Every dataset on the server, with what it was built from and what is in it.
 *
 * Asked for once, when the page is opened: a dataset takes minutes to hours to build, and a
 * finished one is one page change away.
 */
export function DatasetsView() {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let dropped = false;
    fetchDatasets().then(
      (found) => !dropped && setDatasets(found),
      (e: unknown) => !dropped && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      dropped = true;
    };
  }, []);

  return (
    <section className="datasets-view" aria-labelledby="datasets-heading">
      <h2 id="datasets-heading">Datasets</h2>
      {error !== null && <p role="alert">Could not list the datasets: {error}</p>}
      {error === null && datasets === null && <p className="note">Looking…</p>}
      {datasets !== null && datasets.length === 0 && (
        <p className="note">
          No datasets have been built here yet. Build one with <code>chess-ai dataset build</code>.
        </p>
      )}
      {datasets !== null && datasets.length > 0 && (
        <div className="table-frame">
          <table>
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col">Built</th>
                <th scope="col">Sources</th>
                <th scope="col">Filters</th>
                <th scope="col" className="number">
                  Games
                </th>
                <th scope="col" className="number">
                  Positions
                </th>
                <th scope="col" className="number">
                  Skipped
                </th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <DatasetRow key={dataset.name} dataset={dataset} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function DatasetRow({ dataset }: { dataset: DatasetSummary }) {
  const { name, manifest } = dataset;
  const link = (
    <a href={datasetHash({ name, split: DEFAULT_SPLIT, page: 0 })} className="name">
      {name}
    </a>
  );
  if (manifest === null) {
    return (
      <tr>
        <th scope="row">{link}</th>
        <td colSpan={6} className="unreadable">
          Cannot be read: {dataset.error}
        </td>
      </tr>
    );
  }
  const skipped = Object.values(manifest.skipped).reduce((sum, count) => sum + count, 0);
  const latest = manifest.versions.at(-1);
  return (
    <tr>
      <th scope="row">{link}</th>
      <td>
        {formatBuilt(manifest.created)}
        {latest !== undefined && latest.version > 1 && (
          <span className="note version">
            v{latest.version} added {formatBuilt(latest.created)}
          </span>
        )}
      </td>
      <td className="sources" title={manifest.sources.map((source) => source.path).join('\n')}>
        {manifest.sources.map((source) => fileName(source.path)).join(', ')}
        {sourceFailed(manifest) && <span className="warning"> (not all read)</span>}
      </td>
      <td>{describeFilters(manifest.filters)}</td>
      <td className="number">{formatCount(totalOf(manifest, 'games'))}</td>
      <td className="number">{formatCount(totalOf(manifest, 'positions'))}</td>
      <td className="number">{formatCount(skipped)}</td>
    </tr>
  );
}
