import { useEffect, useState } from 'react';
import {
  fetchDataset,
  fetchDatasetGames,
  type DatasetGames,
  type DatasetManifest,
  type DatasetSource,
  type DatasetSummary,
} from './api';
import { DatasetStatistics } from './DatasetStatistics';
import {
  datasetGameHash,
  datasetHash,
  PAGE_SIZE,
  type DatasetPage,
} from './datasetPlaces';
import { describeFilters, fileName, formatBuilt, totalOf, trainedOn } from './DatasetsView';
import { describeResult } from './GameStatus';
import { formatBytes, formatCount, formatPercent } from './runFormat';
import './DatasetsView.css';

/** One dataset: what it was built from, what is in it, and its games a page at a time. */
export function DatasetView({ place }: { place: DatasetPage }) {
  const { name } = place;
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let dropped = false;
    fetchDataset(name).then(
      (found) => !dropped && setDataset(found),
      (e: unknown) => !dropped && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      dropped = true;
    };
  }, [name]);

  const manifest = dataset?.manifest ?? null;
  return (
    <section className="datasets-view dataset-view" aria-labelledby="dataset-heading">
      <a href="#datasets">← All datasets</a>
      <h2 id="dataset-heading">{name}</h2>
      {error !== null && <p role="alert">Could not open that dataset: {error}</p>}
      {error === null && dataset === null && <p className="note">Looking…</p>}
      {dataset !== null && manifest === null && (
        <p role="alert">This dataset cannot be read: {dataset.error}</p>
      )}
      {manifest !== null && (
        <>
          <DatasetFacts manifest={manifest} />
          <DatasetStatistics manifest={manifest} />
          <DatasetGameList manifest={manifest} place={place} />
        </>
      )}
    </section>
  );
}

/** A count of games or positions, and how the two splits share it. */
function splitCounts(manifest: DatasetManifest, what: 'games' | 'positions'): string {
  const splits = Object.entries(manifest.splits)
    .map(([split, counts]) => `${split} ${formatCount(counts[what])}`)
    .join(', ');
  return `${formatCount(totalOf(manifest, what))} (${splits})`;
}

function DatasetFacts({ manifest }: { manifest: DatasetManifest }) {
  const trained = trainedOn(manifest);
  return (
    <>
      <dl className="facts">
        <dt>Built</dt>
        <dd>{formatBuilt(manifest.created)}</dd>
        <dt>Games</dt>
        <dd>{splitCounts(manifest, 'games')}</dd>
        <dt>Positions</dt>
        <dd>{splitCounts(manifest, 'positions')}</dd>
        {trained !== null && (
          <>
            <dt>Trained on</dt>
            <dd>
              {formatCount(trained)} positions; the rest are there for the games around them
            </dd>
          </>
        )}
        <dt>Held back</dt>
        <dd>{formatPercent(manifest.validation_fraction)} of games, for validation</dd>
        <dt>Filters</dt>
        <dd>
          {describeFilters(manifest.filters)}
          {manifest.reached_max_games && '; stopped at the maximum'}
        </dd>
        <dt>Ratings</dt>
        <dd>{manifest.rating_source}</dd>
        <dt>Format</dt>
        <dd>
          version {manifest.format_version}, move vocabulary {manifest.move_vocabulary_size}
        </dd>
      </dl>
      <DatasetSources sources={manifest.sources} />
    </>
  );
}

/** More source files than this are folded away, so that the statistics stay in sight. */
const SOURCES_SHOWN = 5;

function SourceItem({ source }: { source: DatasetSource }) {
  return (
    <li>
      <span className="path" title={source.path}>
        {source.path}
      </span>{' '}
      <span className="note">
        {formatBytes(source.bytes)}, {formatCount(source.games_read)} games read,{' '}
        {formatCount(source.games_kept)} kept
      </span>
      {source.error !== null && (
        <p className="warning">
          {source.went_away
            ? source.games_read === 0
              ? 'Nothing read from it'
              : 'Went away part-way through'
            : 'Not read whole'}
          : {source.error}
        </p>
      )}
    </li>
  );
}

/**
 * The files a dataset was built from. A dataset of a few hundred files lists them folded away
 * behind their totals; a file that was not read whole is listed either way, because it is the
 * first thing to know about a dataset that looks smaller than it should.
 */
function DatasetSources({ sources }: { sources: DatasetSource[] }) {
  const failed = sources.filter((source) => source.error !== null);
  const sum = (count: (source: DatasetSource) => number) =>
    sources.reduce((total, source) => total + count(source), 0);
  return (
    <section aria-labelledby="dataset-sources-heading">
      <h3 id="dataset-sources-heading">Sources</h3>
      {sources.length <= SOURCES_SHOWN ? (
        <ul className="sources">
          {sources.map((source) => (
            <SourceItem key={source.path} source={source} />
          ))}
        </ul>
      ) : (
        <>
          {failed.length > 0 && (
            <ul className="sources">
              {failed.map((source) => (
                <SourceItem key={source.path} source={source} />
              ))}
            </ul>
          )}
          <details>
            <summary>
              {formatCount(sources.length)} files, {formatBytes(sum((s) => s.bytes))},{' '}
              {formatCount(sum((s) => s.games_read))} games read,{' '}
              {formatCount(sum((s) => s.games_kept))} kept
            </summary>
            <ul className="sources">
              {sources.map((source) => (
                <SourceItem key={source.path} source={source} />
              ))}
            </ul>
          </details>
        </>
      )}
    </section>
  );
}

/** What a side of a dataset game is called: its rating, since the dataset keeps no names. */
function rating(value: number | null): string {
  return value === null ? '?' : String(value);
}

/**
 * Which games a page holds, out of how many. A page past the last game, which a rebuild that
 * shrank the dataset or an address typed by hand leads to, says so rather than claiming a game.
 */
function describeRange(first: number, total: number): string {
  if (total === 0) {
    return 'No games';
  }
  if (first >= total) {
    return `Past the last of ${formatCount(total)} games`;
  }
  const last = Math.min(first + PAGE_SIZE, total);
  return `Games ${formatCount(first + 1)}–${formatCount(last)} of ${formatCount(total)}`;
}

function DatasetGameList({ manifest, place }: { manifest: DatasetManifest; place: DatasetPage }) {
  const { name, split, page } = place;
  const [games, setGames] = useState<DatasetGames | null>(null);
  const [error, setError] = useState<string | null>(null);
  const splits = Object.keys(manifest.splits);

  useEffect(() => {
    let dropped = false;
    setError(null);
    fetchDatasetGames(name, split, page * PAGE_SIZE, PAGE_SIZE).then(
      (found) => {
        if (!dropped) {
          setGames(found);
        }
      },
      (e: unknown) => {
        if (!dropped) {
          setGames(null);
          setError(e instanceof Error ? e.message : String(e));
        }
      },
    );
    return () => {
      dropped = true;
    };
  }, [name, split, page]);

  // Only what was asked for last is shown; the previous page stays up until it arrives.
  const shown = games !== null && games.split === split ? games : null;
  const listed = manifest.splits[split]?.games ?? 0;
  // The rows come from whatever is on the disk when each page is asked for, and the manifest
  // from when the dataset was opened. A rebuild in between makes the two disagree, and the
  // pager has to agree with the rows it is paging through.
  const total = shown?.total ?? listed;
  const changed = shown !== null && shown.total !== listed;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const first = page * PAGE_SIZE;
  const go = (to: DatasetPage) => {
    window.location.hash = datasetHash(to);
  };

  return (
    <section className="dataset-games" aria-labelledby="dataset-games-heading">
      <h3 id="dataset-games-heading">Games</h3>
      <div className="pager">
        {splits.length > 1 && (
          <label>
            Split{' '}
            <select value={split} onChange={(e) => go({ name, split: e.target.value, page: 0 })}>
              {splits.map((one) => (
                <option key={one} value={one}>
                  {one}
                </option>
              ))}
            </select>
          </label>
        )}
        <button
          type="button"
          disabled={page === 0}
          onClick={() => go({ name, split, page: page - 1 })}
        >
          Previous
        </button>
        <span className="range" role="status">
          {describeRange(first, total)}
        </span>
        <button
          type="button"
          disabled={page + 1 >= pages}
          onClick={() => go({ name, split, page: page + 1 })}
        >
          Next
        </button>
      </div>
      {error !== null && <p role="alert">Could not list the games: {error}</p>}
      {changed && (
        <p className="warning">
          This dataset has changed since this page was opened: the games listed are the ones
          there now, and the rest of the page describes the dataset as it was. Reload to see it
          as it is.
        </p>
      )}
      {shown !== null && shown.games.length > 0 && (
        <div className="table-frame">
          <table>
            <thead>
              <tr>
                <th scope="col" className="number">
                  #
                </th>
                <th scope="col" className="number">
                  White
                </th>
                <th scope="col" className="number">
                  Black
                </th>
                <th scope="col">Result</th>
                <th scope="col">Date</th>
                <th scope="col">Time control</th>
                <th scope="col" className="number">
                  Moves
                </th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody>
              {shown.games.map((game) => (
                <tr key={game.index}>
                  <th scope="row" className="number">
                    <a
                      href={datasetGameHash({ name, split, index: game.index })}
                      aria-label={`Replay game ${game.index + 1}`}
                    >
                      {formatCount(game.index + 1)}
                    </a>
                  </th>
                  <td className="number">{rating(game.white_rating)}</td>
                  <td className="number">{rating(game.black_rating)}</td>
                  <td>{describeResult(game.result)}</td>
                  <td>{game.date}</td>
                  <td>{game.time_control}</td>
                  <td className="number">{formatCount(Math.ceil(game.plies / 2))}</td>
                  <td title={game.source ?? undefined}>
                    {game.source === null ? '?' : fileName(game.source)}
                    {game.custom_start && ' (set-up position)'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
