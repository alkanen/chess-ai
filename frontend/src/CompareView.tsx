import { useMemo, useRef } from 'react';
import type { MetricsRecord } from './api';
import { MetricChart } from './MetricChart';
import {
  COMPARISON_CHARTS,
  logs,
  comparisonData,
  MAX_SERIES,
  xAxis,
  type ComparedRun,
} from './runCharts';
import { compareHash } from './RunsView';
import { RunStateBadge } from './RunState';
import { useRunChannels } from './useRunChannel';
import { useXAxis, XAxisPicker } from './XAxisPicker';
import './RunsView.css';
import './RunView.css';
import './CompareView.css';

const NO_RECORDS: MetricsRecord[] = [];

/**
 * `runs`, as the same array for as long as every run in it has the same log and batch size.
 *
 * The page is rendered on every message from every run compared, and most of those are a
 * heartbeat that moved: up to eight of them a second, none of which changes a chart. A log is
 * replaced by a new array only when lines are added to it, so comparing the arrays is what
 * tells a message that changes the charts from one that does not, and the charts are worked
 * out again, and redrawn, only for the first kind.
 */
function useComparedRuns(runs: ComparedRun[]): ComparedRun[] {
  const kept = useRef(runs);
  const same =
    kept.current.length === runs.length &&
    kept.current.every(
      (run, index) =>
        // Two empty logs are the same log: a run's first message brings it an empty one of
        // its own, in place of none at all.
        (run.records === runs[index].records ||
          (run.records.length === 0 && runs[index].records.length === 0)) &&
        run.batchSize === runs[index].batchSize,
    );
  if (!same) {
    kept.current = runs;
  }
  return kept.current;
}

interface CompareViewProps {
  names: string[];
}

/**
 * Runs side by side: each chart one metric, with a line for every run, all of it followed
 * live. As many runs as a chart has colours to tell apart; any more are left out, and said
 * to be.
 */
export function CompareView({ names }: CompareViewProps) {
  const compared = names.slice(0, MAX_SERIES);
  const left = names.slice(MAX_SERIES);
  const channels = useRunChannels(compared);
  const [axis, setAxis] = useXAxis();
  const labels = compared.map((name, index) => channels[index].run?.notes?.title ?? name);
  const runs = useComparedRuns(
    channels.map(({ run }) => ({
      records: run?.metrics ?? NO_RECORDS,
      batchSize: run?.info?.batch_size ?? null,
    })),
  );
  const charts = useMemo(
    () =>
      COMPARISON_CHARTS.filter(
        (chart) => !chart.optional || runs.some(({ records }) => logs(records, chart.metric)),
      ).map((chart) => ({ chart, data: comparisonData(runs, chart, axis) })),
    [runs, axis],
  );

  return (
    <section className="run-view compare-view" aria-labelledby="compare-heading">
      <p className="back">
        <a href="#runs">← All runs</a>
      </p>
      <header>
        <h2 id="compare-heading">Comparing {compared.length} runs</h2>
      </header>
      {left.length > 0 && (
        <p role="alert">
          A chart can tell {MAX_SERIES} runs apart, so {left.join(', ')}{' '}
          {left.length === 1 ? 'is' : 'are'} left out.
        </p>
      )}
      <ol className="compared-runs chart-palette" aria-label="Runs compared">
        {compared.map((name, index) => {
          const { run, error } = channels[index];
          const others = compared.filter((other) => other !== name);
          return (
            <li key={name}>
              <span
                className="swatch"
                aria-hidden="true"
                style={{ background: `var(--series-${index + 1})` }}
              />
              <a href={`#runs/${encodeURIComponent(name)}`}>{labels[index]}</a>
              {labels[index] !== name && <span className="run-name">{name}</span>}
              {run !== null && (
                <RunStateBadge status={run.heartbeat?.status} stale={run.stale} />
              )}
              {run?.info != null && (
                <span className="note">
                  {run.info.model.architecture}, batch {run.info.batch_size.toLocaleString('en-US')}
                </span>
              )}
              {error !== null && <span className="error">{error}</span>}
              {others.length > 0 && (
                <a className="remove" href={compareHash(others)} aria-label={`Stop comparing ${name}`}>
                  Remove
                </a>
              )}
            </li>
          );
        })}
      </ol>
      <XAxisPicker value={axis} onChange={setAxis} />
      <div className="charts">
        {charts.map(({ chart, data }) => (
          <MetricChart key={chart.id} chart={chart} labels={labels} x={xAxis(axis)} data={data} />
        ))}
      </div>
    </section>
  );
}
