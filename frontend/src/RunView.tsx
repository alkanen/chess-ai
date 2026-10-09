import { useEffect, useMemo, useState } from 'react';
import type { GpuStats } from './api';
import { MetricChart } from './MetricChart';
import { PROBE_SUITE } from './probes';
import { ProbeView } from './ProbeView';
import { CHARTS, chartData, logs, xAxis } from './runCharts';
import { RunStateBadge } from './RunState';
import {
  datasetLabel,
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

function gibibytes(bytes: number): string {
  return (bytes / 2 ** 30).toFixed(1);
}

/**
 * The GPU's memory as far as the heartbeat knows it. What the run holds says whether a bigger
 * batch would fit, and what the card has in use says whether something else is crowding it,
 * so both are shown; any of the three figures may be missing.
 */
function describeGpuMemory(gpu: GpuStats): string {
  const used = gpu.memory_used_bytes;
  const total = gpu.memory_total_bytes;
  const parts = [
    gpu.process_memory_bytes != null && `${gibibytes(gpu.process_memory_bytes)} GiB held by the run`,
    used != null &&
      (total != null
        ? `${gibibytes(used)} of ${gibibytes(total)} GiB in use on the card`
        : `${gibibytes(used)} GiB in use on the card`),
  ].filter(Boolean);
  return parts.join(', ') || '–';
}

/** A figure NVML may not have given, rounded and with its unit, or a dash. */
function rounded(value: number | null, unit: string): string {
  return value == null ? '–' : `${Math.round(value)}${unit}`;
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
      const level = reference !== null ? spec.reference : undefined;
      if (level) {
        labels.push(level.label);
      }
      const levels = level ? 1 : 0;
      return [{ spec, labels, levels, data: chartData(records, spec, axis, batchSize, reference) }];
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
                : `${datasetLabel(info.dataset.name, info.dataset.version ?? 1)}, ${formatCount(info.dataset.train_targets ?? info.dataset.train_positions)} training positions`}
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
            <dt>Device</dt>
            <dd>{info?.device ?? '–'}</dd>
            {/* Like the time left, a dead run's last figures are not shown as if current. */}
            {running && beat.gpu != null && (
              <>
                <dt>GPU busy</dt>
                <dd>{rounded(beat.gpu.utilization_percent, '%')}</dd>
                <dt>GPU memory</dt>
                <dd>{describeGpuMemory(beat.gpu)}</dd>
                <dt>GPU temperature</dt>
                <dd>{rounded(beat.gpu.temperature_celsius, ' °C')}</dd>
              </>
            )}
            <dt>Heartbeat</dt>
            <dd>{formatAgo(beat?.updated, now)}</dd>
          </dl>
          <RunNotesPanel run={name} notes={notes} onSaved={showSaved} />
          <XAxisPicker value={axis} onChange={setAxis} />
          <div className="charts">
            {charts.map(({ spec, labels, levels, data }) => (
              <MetricChart
                key={spec.id}
                chart={spec}
                labels={labels}
                levels={levels}
                x={xAxis(axis)}
                data={data}
              />
            ))}
          </div>
          {/* Another run's checkpoints are others: nothing chosen or kept of one carries over. */}
          <ProbeView
            key={name}
            run={name}
            evaluations={run.evaluations}
            currentSet={run.probeSet}
            probed={info?.config?.evaluation?.suites?.includes(PROBE_SUITE) ?? true}
          />
        </>
      )}
    </section>
  );
}
