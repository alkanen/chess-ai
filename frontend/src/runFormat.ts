import type { RunStatus } from './api';

/** What the dashboard calls a run's state; "stale" is a running run that stopped reporting. */
export type RunState = RunStatus | 'stale' | 'starting';

/**
 * A run's state as the dashboard shows it. A run that says it is running and has not
 * rewritten its heartbeat for a while has most likely died without saying so, and is shown
 * as stale rather than taken at its word; a run with no heartbeat yet is starting.
 */
export function runState(status: RunStatus | null | undefined, stale: boolean | undefined): RunState {
  if (status == null) {
    return 'starting';
  }
  return status === 'running' && stale ? 'stale' : status;
}

/** A sentence on what each state means, for a tooltip. */
export const STATE_DESCRIPTIONS: Record<RunState, string> = {
  starting: 'Has not written a heartbeat yet.',
  running: 'Training, and reporting as it goes.',
  stale: 'Says it is running, but has stopped reporting: its trainer has probably died.',
  finished: 'Trained every step it was asked to.',
  stopped: 'Asked to stop, and saved a checkpoint on the way out.',
  crashed: 'Ended with an error.',
};

/** A number to a few significant digits, without the noise of a float's full length. */
export function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value == null) {
    return '–';
  }
  if (value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e6)) {
    return value.toExponential(digits - 2);
  }
  return Number(value.toPrecision(digits)).toString();
}

/** A fraction as a percentage, such as 0.4594 as "45.9%". */
export function formatPercent(value: number | null | undefined, decimals = 1): string {
  return value == null ? '–' : `${(value * 100).toFixed(decimals)}%`;
}

/**
 * How many decimals it takes to write `value` without float noise, such as 2 for 0.25 and 6
 * for 2.5e-5: what ticks that far apart need to be told apart.
 */
export function decimalsOf(value: number): number {
  if (!Number.isFinite(value) || value === 0) {
    return 0;
  }
  const [mantissa, exponent = '0'] = Math.abs(value).toPrecision(6).split('e');
  const fraction = mantissa.includes('.') ? mantissa.split('.')[1].replace(/0+$/, '') : '';
  return Math.max(0, fraction.length - Number(exponent));
}

/** A count with its thousands separated, such as 15000 as "15,000". */
export function formatCount(value: number | null | undefined): string {
  return value == null ? '–' : Math.round(value).toLocaleString('en-US');
}

/** A rate with an SI suffix, such as 344538 as "345k". */
export function formatRate(value: number | null | undefined): string {
  if (value == null) {
    return '–';
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(2)}M`;
  }
  if (value >= 1e3) {
    return `${Math.round(value / 1e3)}k`;
  }
  return Math.round(value).toString();
}

/** A length of time in the largest two units that matter, such as "1h 05m". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) {
    return '–';
  }
  const whole = Math.max(0, Math.round(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const rest = whole % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, '0')}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(rest).padStart(2, '0')}s`;
  }
  return `${rest}s`;
}

/** How long ago an ISO timestamp was, as of `now`, such as "3m 12s ago". */
export function formatAgo(timestamp: string | null | undefined, now: number = Date.now()): string {
  if (timestamp == null) {
    return '–';
  }
  return `${formatDuration((now - Date.parse(timestamp)) / 1000)} ago`;
}

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

/** A size in the unit that makes it readable, as the command line writes one: "1.2 MB". */
export function formatBytes(count: number): string {
  let size = count;
  for (const [place, unit] of BYTE_UNITS.entries()) {
    if (size < 1024 || place === BYTE_UNITS.length - 1) {
      return unit === 'B' ? `${size.toFixed(0)} B` : `${size.toFixed(1)} ${unit}`;
    }
    size /= 1024;
  }
  return `${count} B`;
}
