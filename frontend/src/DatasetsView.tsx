import { useEffect, useState } from 'react';
import { fetchDatasets, type DatasetManifest, type DatasetSummary } from './api';
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

/** The filters a dataset was built with, in a few words. */
export function describeFilters(filters: DatasetManifest['filters']): string {
  const entries = Object.entries(filters);
  if (entries.length === 0) {
    return 'none';
  }
  return entries
    .map(([name, value]) => `${name.replaceAll('_', ' ')} ${JSON.stringify(value)}`)
    .join(', ');
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
  return (
    <tr>
      <th scope="row">{link}</th>
      <td>{formatBuilt(manifest.created)}</td>
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
