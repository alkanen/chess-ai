import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Heartbeat, RunEvent, RunInfo } from './api';
import { RunView } from './RunView';
import { FakeUPlot } from './test/fakeUPlot';
import { FakeWebSocket } from './test/fakeWebSocket';

vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

const INFO: RunInfo = {
  name: 'mlp-big',
  created: '2026-10-01T10:00:00Z',
  seed: 7,
  code_version: 'git abc1234',
  device: 'cuda',
  dataset: { name: 'lichess-2024', positions: 1_000_000, train_positions: 950_000 },
  model: { architecture: 'mlp', options: { width: 2048 }, parameter_count: 12_345_678 },
  steps: 15_000,
  batch_size: 16_384,
};

function heartbeat(changes: Partial<Heartbeat> = {}): Heartbeat {
  return {
    status: 'running',
    pid: 1,
    started: '2026-10-01T10:00:00Z',
    updated: new Date().toISOString(),
    step: 1_000,
    steps: 15_000,
    epoch: 0.25,
    positions_per_second: 344_538,
    eta_seconds: 3_900,
    gpu: null,
    message: null,
    ...changes,
  };
}

function runEvent(changes: Partial<Extract<RunEvent, { type: 'run' }>> = {}): RunEvent {
  return { type: 'run', name: 'mlp-big', info: INFO, heartbeat: heartbeat(), stale: false, ...changes };
}

describe('RunView', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    FakeWebSocket.instances = [];
    FakeUPlot.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('follows the run channel under the path prefix', () => {
    render(<RunView name="mlp big" />);

    const url = new URL('/chess/api/runs/mlp%20big/ws', window.location.href);
    url.protocol = 'ws:';
    expect(FakeWebSocket.latest.url).toBe(url.href);
    expect(screen.getByText('Connecting to the server…')).toBeInTheDocument();
  });

  it('says what the run is and where it has got to', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent());

    expect(screen.getByRole('heading', { name: 'mlp-big' })).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('mlp, 12,345,678 parameters')).toBeInTheDocument();
    expect(screen.getByText('lichess-2024, 950,000 training positions')).toBeInTheDocument();
    expect(screen.getByText('1,000 of 15,000, epoch 0.25')).toBeInTheDocument();
    expect(screen.getByText('345k positions/s')).toBeInTheDocument();
    expect(screen.getByText('1h 05m')).toBeInTheDocument();
  });

  it('charts the whole log, and then what is appended to it, without reloading', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent());
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 50, split: 'train', loss: 3.0, learning_rate: 0.001 },
        { step: 100, split: 'train', loss: 2.5, learning_rate: 0.002 },
      ],
    });

    const loss = FakeUPlot.withSeries('validation');
    expect(loss.data).toEqual([
      [50, 100],
      [3.0, 2.5],
      [null, null],
    ]);
    expect(FakeUPlot.withSeries('learning rate').data).toEqual([
      [50, 100],
      [0.001, 0.002],
    ]);
    // Nothing validated yet, so there is nothing to chart accuracy with.
    expect(() => FakeUPlot.withSeries('top-1')).toThrow();
    expect(screen.getAllByText('Nothing logged yet')).toHaveLength(2);

    socket.deliver({
      type: 'metrics',
      reset: false,
      records: [
        { step: 100, split: 'validation', loss: 2.7, top1: 0.3, top5: 0.6, illegal_top_move_rate: 0.1 },
      ],
    });

    // The same chart, handed the new data rather than made again.
    expect(FakeUPlot.withSeries('validation')).toBe(loss);
    expect(loss.data).toEqual([
      [50, 100],
      [3.0, 2.5],
      [null, 2.7],
    ]);
    expect(FakeUPlot.withSeries('top-1').data).toEqual([[100], [0.3], [0.6]]);
    expect(screen.getByText('top-1 30.0% at step 100; top-5 60.0% at step 100')).toBeInTheDocument();
  });

  it('follows the run again on a chart made anew after a zoom', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent());
    socket.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 9 }] });
    FakeUPlot.withSeries('validation').select();
    // The run is started again under the same name: the log empties, and the chart with it.
    socket.deliver({ type: 'metrics', reset: true, records: [] });
    socket.deliver({ type: 'metrics', reset: false, records: [{ step: 1, split: 'train', loss: 5 }] });

    socket.deliver({ type: 'metrics', reset: false, records: [{ step: 2, split: 'train', loss: 4 }] });

    const remade = FakeUPlot.withSeries('validation');
    expect(remade.data[0]).toEqual([1, 2]);
    expect(remade.rescaled).toBe(true);
  });

  it('keeps a zoomed chart where it was put while new lines arrive', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent());
    socket.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 9 }] });
    const chart = FakeUPlot.withSeries('validation');
    chart.select();

    socket.deliver({ type: 'metrics', reset: false, records: [{ step: 2, split: 'train', loss: 8 }] });

    expect(chart.rescaled).toBe(false);
  });

  it('starts the charts again when the log is replaced', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent());
    socket.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 9 }] });

    socket.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 5 }] });

    expect(FakeUPlot.withSeries('validation').data[1]).toEqual([5]);
  });

  it('flags a run that has stopped reporting', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent());

    FakeWebSocket.latest.deliver(runEvent({ stale: true }));

    expect(screen.getByText('stale')).toHaveAttribute(
      'title',
      expect.stringContaining('stopped reporting'),
    );
    expect(screen.queryByText('running')).not.toBeInTheDocument();
    // Nothing is left to wait for on a run that has died.
    expect(screen.queryByText('Time left')).not.toBeInTheDocument();
  });

  it('shows why a run crashed', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(
      runEvent({ heartbeat: heartbeat({ status: 'crashed', message: 'RuntimeError: CUDA out of memory' }) }),
    );

    expect(screen.getByText('crashed')).toBeInTheDocument();
    expect(screen.getByText('RuntimeError: CUDA out of memory')).toBeInTheDocument();
  });

  it('keeps counting how long ago a run that has stopped reporting last did', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-01T12:00:00Z'));
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(
      runEvent({ stale: true, heartbeat: heartbeat({ updated: '2026-10-01T11:58:00Z' }) }),
    );
    expect(screen.getByText('2m 00s ago')).toBeInTheDocument();

    // The server says nothing more about a run that has gone quiet.
    act(() => vi.advanceTimersByTime(60_000));

    expect(screen.getByText('3m 00s ago')).toBeInTheDocument();
  });

  it('stops asking for a run the server says is not there', () => {
    vi.useFakeTimers();
    render(<RunView name="missing" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver({ type: 'error', message: "no run called 'missing' is kept here" });

    socket.disconnect();
    act(() => vi.advanceTimersByTime(60_000));

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(screen.getByRole('alert')).toHaveTextContent("no run called 'missing' is kept here");
    expect(screen.queryByText('Reconnecting…')).not.toBeInTheDocument();
  });

  it('says so when there is no such run', () => {
    render(<RunView name="missing" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver({ type: 'error', message: "no run called 'missing' is kept here" });

    expect(screen.getByRole('alert')).toHaveTextContent("no run called 'missing' is kept here");
  });

  it('reconnects when the connection drops, and is sent the whole run again', () => {
    vi.useFakeTimers();
    render(<RunView name="mlp-big" />);
    const first = FakeWebSocket.latest;
    first.open();
    first.deliver(runEvent());
    first.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 9 }] });

    first.disconnect();
    expect(screen.getByText('Reconnecting…')).toBeInTheDocument();
    vi.advanceTimersByTime(1_000);
    const second = FakeWebSocket.latest;
    expect(second).not.toBe(first);
    second.open();
    second.deliver(runEvent());
    second.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 1, split: 'train', loss: 9 },
        { step: 2, split: 'train', loss: 8 },
      ],
    });

    expect(screen.queryByText('Reconnecting…')).not.toBeInTheDocument();
    expect(FakeUPlot.withSeries('validation').data[1]).toEqual([9, 8]);
  });
});
