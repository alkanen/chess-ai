import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunEvent, RunInfo } from './api';
import { CompareView } from './CompareView';
import { MAX_SERIES } from './runCharts';
import { FakeUPlot } from './test/fakeUPlot';
import { FakeWebSocket } from './test/fakeWebSocket';

vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

function info(name: string, batchSize: number): RunInfo {
  return {
    name,
    created: '2026-10-01T10:00:00Z',
    seed: 7,
    code_version: 'git abc1234',
    device: 'cuda',
    dataset: { name: 'lichess-2024', positions: 1_000_000, train_positions: 950_000 },
    model: { architecture: 'mlp', options: {}, parameter_count: 1_000 },
    steps: 100,
    batch_size: batchSize,
  };
}

function runEvent(name: string, batchSize: number, title: string | null = null): RunEvent {
  return {
    type: 'run',
    name,
    info: info(name, batchSize),
    heartbeat: null,
    stale: false,
    notes: { title, tags: [], notes: '' },
  };
}

/** The socket following the run called `name`. */
function socketFor(name: string): FakeWebSocket {
  const socket = FakeWebSocket.instances.find((candidate) =>
    candidate.url.endsWith(`/runs/${name}/ws`),
  );
  if (socket === undefined) {
    throw new Error(`nothing follows ${name}`);
  }
  return socket;
}

describe('CompareView', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    FakeWebSocket.instances = [];
    FakeUPlot.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
    window.localStorage.clear();
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('follows every run compared, and overlays them on the same charts', () => {
    render(<CompareView names={['small', 'big']} />);
    expect(FakeWebSocket.instances).toHaveLength(2);
    const small = socketFor('small');
    const big = socketFor('big');
    small.open();
    small.deliver(runEvent('small', 100, 'Small batch'));
    small.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 1, split: 'train', loss: 4.0, positions_seen: 100 },
        { step: 2, split: 'train', loss: 3.0, positions_seen: 200 },
      ],
    });
    big.open();
    big.deliver(runEvent('big', 200));
    big.deliver({
      type: 'metrics',
      reset: true,
      records: [{ step: 1, split: 'train', loss: 3.5, positions_seen: 200 }],
    });

    const chart = FakeUPlot.withSeries('Small batch');
    expect(chart.options.series.map((series) => series.label)).toEqual([
      'Step',
      'Small batch',
      'big',
    ]);
    expect(chart.data).toEqual([
      [1, 2],
      [4.0, 3.0],
      [3.5, null],
    ]);

    // Runs with different batch sizes line up on the positions they have seen.
    fireEvent.click(screen.getByRole('radio', { name: 'Positions seen' }));

    expect(FakeUPlot.withSeries('Small batch').data).toEqual([
      [100, 200],
      [4.0, 3.0],
      [null, 3.5],
    ]);
  });

  it('fits the y-axis of the loss charts to the losses rather than to powers of ten', () => {
    render(<CompareView names={['a', 'b']} />);
    const a = socketFor('a');
    a.open();
    a.deliver(runEvent('a', 8));
    a.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 1, split: 'train', loss: 1.52, gradient_norm: 0.35 },
        { step: 1, split: 'validation', loss: 1.44, top1: 0.3 },
      ],
    });

    for (const title of ['Training loss', 'Validation loss', 'Gradient norm, before clipping']) {
      const figure = screen.getByText(title).closest('figure')!;
      const chart = FakeUPlot.live().find((candidate) => figure.contains(candidate.target))!;
      const [low, high] = chart.yRange(1.44, 1.52)!;
      expect(low).toBeGreaterThan(1.4);
      expect(high).toBeLessThan(1.6);
    }
    const figure = screen.getByText('Top-1 accuracy (validation)').closest('figure')!;
    const top1 = FakeUPlot.live().find((candidate) => figure.contains(candidate.target))!;
    expect(top1.yRange(0.3, 0.4)).toBeNull();
  });

  it('adds what each run logs as it trains', () => {
    render(<CompareView names={['a', 'b']} />);
    const a = socketFor('a');
    a.open();
    a.deliver(runEvent('a', 8));
    a.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 4 }] });
    const chart = FakeUPlot.withSeries('a');

    a.deliver({ type: 'metrics', reset: false, records: [{ step: 2, split: 'train', loss: 3 }] });

    expect(FakeUPlot.withSeries('a')).toBe(chart);
    expect(chart.data).toEqual([
      [1, 2],
      [4, 3],
      [null, null],
    ]);
  });

  it('redraws nothing when only a heartbeat changes', () => {
    render(<CompareView names={['a', 'b']} />);
    const [a, b] = [socketFor('a'), socketFor('b')];
    a.open();
    a.deliver(runEvent('a', 8));
    a.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 4 }] });
    b.open();
    b.deliver(runEvent('b', 8));
    b.deliver({ type: 'metrics', reset: true, records: [] });
    const charts = FakeUPlot.live();
    const drawn = charts.map((chart) => chart.data);

    a.deliver({ ...(runEvent('a', 8) as Extract<RunEvent, { type: 'run' }>), stale: true });
    b.deliver({ ...(runEvent('b', 8) as Extract<RunEvent, { type: 'run' }>), stale: true });

    expect(FakeUPlot.live()).toEqual(charts);
    // Handed no new data, not even an equal copy of the old: setData was never called.
    FakeUPlot.live().forEach((chart, index) => expect(chart.data).toBe(drawn[index]));
  });

  it('overlays the gradient norm once any run compared has logged one', () => {
    render(<CompareView names={['old', 'new']} />);
    const [old, current] = [socketFor('old'), socketFor('new')];
    old.open();
    old.deliver(runEvent('old', 8));
    old.deliver({ type: 'metrics', reset: true, records: [{ step: 1, split: 'train', loss: 4 }] });

    expect(screen.queryByText('Gradient norm, before clipping')).toBeNull();

    current.open();
    current.deliver(runEvent('new', 8));
    current.deliver({
      type: 'metrics',
      reset: true,
      records: [{ step: 1, split: 'train', loss: 3, gradient_norm: 0.5, clipped_fraction: 0 }],
    });

    const figure = screen.getByText('Gradient norm, before clipping').closest('figure')!;
    const chart = FakeUPlot.live().find((candidate) => figure.contains(candidate.target))!;
    expect(chart.data).toEqual([[1], [null], [0.5]]);
  });

  it('names each run with a link to it, and a way to leave it out', () => {
    render(<CompareView names={['a', 'b', 'c']} />);
    socketFor('a').open();
    socketFor('a').deliver(runEvent('a', 8, 'Run A'));

    const key = within(screen.getByRole('list', { name: 'Runs compared' }));
    expect(key.getByRole('link', { name: 'Run A' })).toHaveAttribute('href', '#runs/a');
    expect(key.getByRole('link', { name: 'Stop comparing b' })).toHaveAttribute(
      'href',
      '#compare/a,c',
    );
  });

  it('says which runs are left out when more are asked for than a chart can tell apart', () => {
    const names = Array.from({ length: MAX_SERIES + 2 }, (_, index) => `run-${index}`);

    render(<CompareView names={names} />);

    expect(FakeWebSocket.instances).toHaveLength(MAX_SERIES);
    expect(screen.getByRole('alert')).toHaveTextContent(
      `run-${MAX_SERIES}, run-${MAX_SERIES + 1} are left out`,
    );
  });

  it('says so of a run that is not there, and goes on comparing the rest', () => {
    render(<CompareView names={['a', 'ghost']} />);
    socketFor('ghost').open();
    socketFor('ghost').deliver({ type: 'error', message: "no run called 'ghost' is kept here" });

    expect(screen.getByText("no run called 'ghost' is kept here")).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Comparing 2 runs' })).toBeInTheDocument();
  });
});
