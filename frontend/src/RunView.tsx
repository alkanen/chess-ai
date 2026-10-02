import { useEffect, useMemo, useState } from 'react';
import type { GpuStats } from './api';
import { MetricChart } from './MetricChart';
import { CHARTS, chartData, logs, xAxis } from './runCharts';
import { RunStateBadge } from './RunState';
import {
  formatAgo,
  formatCount,
  formatDuration,
  formatNumber,
  formatRate,
} from './runFormat';
import { RunNotesPanel, useShownNotes } from './RunNotesPanel';
import { useRunChannel } from './useRunChannel';
import { useXAxis, XAxisPicker } from './XAxisPicker';
import './RunsView.css';
import './RunView.css';

function gibibytes(bytes: number | null): string {
  return bytes == null ? '–' : `${(bytes / 2 ** 30).toFixed(1)} GiB`;
}

function describeGpu(gpu: GpuStats): string {
  const parts = [
    gpu.utilization_percent != null && `${Math.round(gpu.utilization_percent)}% busy`,
    gpu.process_memory_bytes != null && `${gibibytes(gpu.process_memory_bytes)} held by the run`,
    gpu.memory_used_bytes != null &&
      `${gibibytes(gpu.memory_used_bytes)} / ${gibibytes(gpu.memory_total_bytes)} in use`,
    gpu.temperature_celsius != null && `${Math.round(gpu.temperature_celsius)} °C`,
  ].filter(Boolean);
  return [gpu.name, parts.join(', ')].filter(Boolean).join(': ') || '–';
}

/**
 * The time now, moving on every second while `ticking`. How long ago a heartbeat was is the
 * one thing on the page that changes with nothing sent, and a run that has stopped reporting
 * is when it matters most.
 */
function useNow(ticking: boolean): number {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (!ticking) {
      return;
    }
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, [ticking]);
  return now;
}

interface RunViewProps {
  name: string;
}

/** One run: what it is, where it has got to, and its metrics charted live as it trains. */
export function RunView({ name }: RunViewProps) {
  const { run, error, connected } = useRunChannel(name);
  const [axis, setAxis] = useXAxis();
  const metrics = run?.metrics;
  const info = run?.info ?? null;
  const batchSize = info?.batch_size ?? null;
  // As one string, so that the charts are worked out again when a level changes rather than
  // whenever a heartbeat brings the same info in a new object.
  const referenceKey = JSON.stringify(
    CHARTS.map((spec) => (spec.reference && info ? spec.reference.value(info) : null)),
  );
  const charts = useMemo(() => {
    const references = JSON.parse(referenceKey) as (number | null)[];
    const records = metrics ?? [];
    return CHARTS.flatMap((spec, index) => {
      if (spec.optional && !logs(records, spec.series[0])) {
        return [];
      }
      const reference = references[index];
      const labels = spec.series.map((series) => series.label);
      if (reference !== null && spec.reference) {
        labels.push(spec.reference.label);
      }
      return [{ spec, labels, data: chartData(records, spec, axis, batchSize, reference) }];
    });
  }, [metrics, axis, batchSize, referenceKey]);
  const [notes, showSaved] = useShownNotes(run?.notes ?? null);
  const title = notes?.title ?? null;
  const beat = run?.heartbeat ?? null;
  const running = beat?.status === 'running' && !run?.stale;
  const now = useNow(beat !== null);

  return (
    <section className="run-view" aria-labelledby="run-heading">
      <p className="back">
        <a href="#runs">← All runs</a>
      </p>
      <header>
        <h2 id="run-heading">{title ?? name}</h2>
        {title !== null && <span className="run-name">{name}</span>}
        {run !== null && <RunStateBadge status={beat?.status} stale={run.stale} />}
        {run !== null && !connected && error === null && (
          <span className="note">Reconnecting…</span>
        )}
      </header>
      {error !== null && <p role="alert">{error}</p>}
      {error === null && run === null && <p className="note">Connecting to the server…</p>}
      {run !== null && (
        <>
          {beat?.message != null && (
            <p className={beat.status === 'crashed' ? 'message crashed' : 'message'}>
              {beat.message}
            </p>
          )}
          <dl className="run-facts">
            <dt>Architecture</dt>
            <dd>
              {info === null
                ? '–'
                : `${info.model.architecture}, ${formatCount(info.model.parameter_count)} parameters`}
            </dd>
            <dt>Dataset</dt>
            <dd>
              {info === null
                ? '–'
                : `${info.dataset.name}, ${formatCount(info.dataset.train_positions)} training positions`}
            </dd>
            <dt>Step</dt>
            <dd>
              {formatCount(beat?.step)} of {formatCount(beat?.steps ?? info?.steps)}
              {beat !== null && `, epoch ${formatNumber(beat.epoch, 3)}`}
            </dd>
            <dt>Throughput</dt>
            <dd>
              {beat?.positions_per_second == null
                ? '–'
                : `${formatRate(beat.positions_per_second)} positions/s`}
            </dd>
            {running && (
              <>
                <dt>Time left</dt>
                <dd>{formatDuration(beat.eta_seconds)}</dd>
              </>
            )}
            {beat?.gpu != null && (
              <>
                <dt>GPU</dt>
                <dd>{describeGpu(beat.gpu)}</dd>
              </>
            )}
            <dt>Heartbeat</dt>
            <dd>{formatAgo(beat?.updated, now)}</dd>
          </dl>
          <RunNotesPanel run={name} notes={notes} onSaved={showSaved} />
          <XAxisPicker value={axis} onChange={setAxis} />
          <div className="charts">
            {charts.map(({ spec, labels, data }) => (
              <MetricChart
                key={spec.id}
                chart={spec}
                labels={labels}
                x={xAxis(axis)}
                data={data}
              />
            ))}
          </div>
        </>
      )}
    </section>
  );
}
