import { useEffect, useState } from 'react';
import { fetchRuns, type RunSummary } from './api';
import { RunStateBadge } from './RunState';
import { formatAgo, formatCount, formatNumber, formatPercent } from './runFormat';
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

/** The runs dashboard: every run with its state and its latest numbers, each one to open. */
export function RunsView() {
  const [runs, error] = useRunList();

  return (
    <section className="runs-view" aria-labelledby="runs-heading">
      <h2 id="runs-heading">Training runs</h2>
      {error !== null && <p role="alert">Could not list the training runs: {error}</p>}
      {error === null && runs === null && <p className="note">Looking…</p>}
      {runs !== null && runs.length === 0 && (
        <p className="note">No training runs here yet. Start one with chess-ai train.</p>
      )}
      {runs !== null && runs.length > 0 && (
        <div className="table-frame">
          <table>
            <thead>
              <tr>
                <th scope="col">State</th>
                <th scope="col">Run</th>
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
              {runs.map((run) => (
                <tr key={run.name}>
                  <td>
                    <RunStateBadge status={run.status} stale={run.stale} />
                  </td>
                  <th scope="row" className="name">
                    <a href={`#runs/${encodeURIComponent(run.name)}`}>{run.name}</a>
                  </th>
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
