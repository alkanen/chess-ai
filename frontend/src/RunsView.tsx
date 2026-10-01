import { useEffect, useState } from 'react';
import { fetchRuns, type RunSummary } from './api';
import { MAX_SERIES } from './runCharts';
import { RunStateBadge } from './RunState';
import { formatAgo, formatCount, formatNumber, formatPercent } from './runFormat';
import { TagList } from './TagList';
import './RunsView.css';

/**
 * How often the list is asked for again. Every run's state, step and latest metrics are in
 * it, and a run going stale is something only time passing shows.
 */
export const RUNS_REFRESH_MS = 5_000;

/** Every run there is, kept up to date while the page is on show. */
function useRunList(): [RunSummary[] | null, string | null] {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let dropped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    function refresh() {
      fetchRuns()
        .then(
          (found) => {
            if (!dropped) {
              setRuns(found);
              setError(null);
            }
          },
          // What was listed before stays up, with a word that it may be out of date.
          (e: unknown) => !dropped && setError(e instanceof Error ? e.message : String(e)),
        )
        .finally(() => {
          if (!dropped) {
            timer = setTimeout(refresh, RUNS_REFRESH_MS);
          }
        });
    }
    refresh();
    return () => {
      dropped = true;
      clearTimeout(timer);
    };
  }, []);

  return [runs, error];
}

/** Where a run has got to, as steps done of steps asked for. */
function progress(run: RunSummary): string {
  if (run.step == null) {
    return '–';
  }
  if (run.steps == null || run.steps === 0) {
    return formatCount(run.step);
  }
  return `${formatCount(run.step)} / ${formatCount(run.steps)} (${Math.floor((run.step / run.steps) * 100)}%)`;
}

function metric(record: RunSummary['latest_train'], name: string): number | null {
  const value = record?.[name];
  return typeof value === 'number' ? value : null;
}

/** Where the runs called `names` are compared, in the address. */
export function compareHash(names: string[]): string {
  return `#compare/${names.map(encodeURIComponent).join(',')}`;
}

/** Every tag any of `runs` has, in alphabetical order. */
function allTags(runs: RunSummary[]): string[] {
  return [...new Set(runs.flatMap((run) => run.tags ?? []))].sort((a, b) => a.localeCompare(b));
}

/**
 * The runs dashboard: every run with its state and its latest numbers, each one to open,
 * filtered by a tag if one is picked, and any of them to be compared side by side.
 */
export function RunsView() {
  const [runs, error] = useRunList();
  const [tag, setTag] = useState<string | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const tags = runs === null ? [] : allTags(runs);
  // A tag no run has any more filters nothing out, rather than everything.
  const filter = tag !== null && tags.includes(tag) ? tag : null;
  const shown = runs?.filter((run) => filter === null || (run.tags ?? []).includes(filter)) ?? [];
  // Only runs that are still there; chosen ones the filter hides stay chosen.
  const compared = chosen.filter((name) => runs?.some((run) => run.name === name));
  const full = compared.length >= MAX_SERIES;

  function choose(name: string, on: boolean) {
    setChosen((before) => (on ? [...before, name] : before.filter((other) => other !== name)));
  }

  return (
    <section className="runs-view" aria-labelledby="runs-heading">
      <h2 id="runs-heading">Training runs</h2>
      {error !== null && <p role="alert">Could not list the training runs: {error}</p>}
      {error === null && runs === null && <p className="note">Looking…</p>}
      {runs !== null && runs.length === 0 && (
        <p className="note">No training runs here yet. Start one with chess-ai train.</p>
      )}
      {runs !== null && runs.length > 0 && (
        <div className="runs-tools">
          {tags.length > 0 && (
            <div className="tag-filter" role="group" aria-label="Filter by tag">
              <span className="label">Tags</span>
              <button type="button" aria-pressed={filter === null} onClick={() => setTag(null)}>
                All
              </button>
              <TagList
                tags={tags}
                picked={filter}
                onPick={(picked) => setTag(picked === filter ? null : picked)}
              />
            </div>
          )}
          <div className="compare-bar">
            {compared.length >= 2 ? (
              <a className="compare" href={compareHash(compared)}>
                Compare {compared.length} runs
              </a>
            ) : (
              <span className="note">Tick two or more runs to compare them on the same charts.</span>
            )}
            {compared.length > 0 && (
              <button type="button" onClick={() => setChosen([])}>
                Clear
              </button>
            )}
          </div>
        </div>
      )}
      {runs !== null && runs.length > 0 && shown.length === 0 && (
        <p className="note">No run is tagged {filter}.</p>
      )}
      {shown.length > 0 && (
        <div className="table-frame">
          <table>
            <thead>
              <tr>
                <th scope="col">
                  <span className="visually-hidden">Compare</span>
                </th>
                <th scope="col">State</th>
                <th scope="col">Run</th>
                <th scope="col">Tags</th>
                <th scope="col">Architecture</th>
                <th scope="col">Dataset</th>
                <th scope="col">Progress</th>
                <th scope="col" className="number">
                  Train loss
                </th>
                <th scope="col" className="number">
                  Val. loss
                </th>
                <th scope="col" className="number">
                  Top-1
                </th>
                <th scope="col" className="number">
                  Illegal
                </th>
                <th scope="col">Heartbeat</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((run) => (
                <tr key={run.name}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`Compare ${run.name}`}
                      checked={compared.includes(run.name)}
                      disabled={full && !compared.includes(run.name)}
                      title={
                        full && !compared.includes(run.name)
                          ? `A chart can tell ${MAX_SERIES} runs apart`
                          : undefined
                      }
                      onChange={(e) => choose(run.name, e.target.checked)}
                    />
                  </td>
                  <td>
                    <RunStateBadge status={run.status} stale={run.stale} />
                  </td>
                  <th scope="row" className="name">
                    <a href={`#runs/${encodeURIComponent(run.name)}`}>{run.title ?? run.name}</a>
                    {run.title != null && <span className="run-name">{run.name}</span>}
                  </th>
                  <td>
                    <TagList tags={run.tags ?? []} picked={filter} onPick={setTag} />
                  </td>
                  <td>{run.architecture ?? '–'}</td>
                  <td>{run.dataset ?? '–'}</td>
                  <td className="progress">{progress(run)}</td>
                  <td className="number">{formatNumber(metric(run.latest_train, 'loss'))}</td>
                  <td className="number">
                    {formatNumber(metric(run.latest_validation, 'loss'))}
                  </td>
                  <td className="number">
                    {formatPercent(metric(run.latest_validation, 'top1'))}
                  </td>
                  <td className="number">
                    {formatPercent(metric(run.latest_validation, 'illegal_top_move_rate'))}
                  </td>
                  <td className="updated">{formatAgo(run.updated)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
