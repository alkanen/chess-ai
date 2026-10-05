import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GpuStats, Heartbeat, RunEvent, RunInfo } from './api';
import { RunView } from './RunView';
import { FakeUPlot } from './test/fakeUPlot';
import { FakeWebSocket } from './test/fakeWebSocket';

vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

const INFO: RunInfo = {
  name: 'mlp-big',
  created: '2026-10-01T10:00:00Z',
  seed: 7,
  code_version: 'git abc1234',
  device: 'cuda:0 NVIDIA GeForce RTX 4090, 24 GiB',
  dataset: { name: 'lichess-2024', positions: 1_000_000, train_positions: 950_000 },
  model: { architecture: 'mlp', options: { width: 2048 }, parameter_count: 12_345_678 },
  steps: 15_000,
  batch_size: 16_384,
};

const GIB = 2 ** 30;

const GPU: GpuStats = {
  name: 'NVIDIA GeForce RTX 4090',
  utilization_percent: 87,
  memory_used_bytes: 20 * GIB,
  memory_total_bytes: 24 * GIB,
  process_memory_bytes: 6 * GIB,
  temperature_celsius: 64,
};

/** What the run's facts say against `label`. */
function fact(label: string): string | null {
  const term = screen.getByText(label, { selector: 'dt' });
  return term.nextElementSibling?.textContent ?? null;
}

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
  return {
    type: 'run',
    name: 'mlp-big',
    info: INFO,
    heartbeat: heartbeat(),
    stale: false,
    notes: { title: null, tags: [], notes: '' },
    ...changes,
  };
}

describe('RunView', () => {
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

  it('shows what the GPU says about itself, and follows it as the heartbeat changes', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ heartbeat: heartbeat({ gpu: GPU }) }));

    expect(fact('Device')).toBe('cuda:0 NVIDIA GeForce RTX 4090, 24 GiB');
    expect(fact('GPU busy')).toBe('87%');
    expect(fact('GPU memory')).toBe('6.0 GiB held by the run, 20.0 of 24.0 GiB in use on the card');
    expect(fact('GPU temperature')).toBe('64 °C');

    socket.deliver(
      runEvent({
        heartbeat: heartbeat({
          step: 2_000,
          gpu: { ...GPU, utilization_percent: 12.4, temperature_celsius: 71.6 },
        }),
      }),
    );

    expect(fact('Step')).toBe('2,000 of 15,000, epoch 0.25');
    expect(fact('GPU busy')).toBe('12%');
    expect(fact('GPU temperature')).toBe('72 °C');
  });

  it('shows the memory figures of a GPU that NVML could not be asked about', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(
      runEvent({
        heartbeat: heartbeat({
          gpu: {
            ...GPU,
            name: null,
            utilization_percent: null,
            temperature_celsius: null,
          },
        }),
      }),
    );

    expect(fact('GPU busy')).toBe('–');
    expect(fact('GPU memory')).toBe('6.0 GiB held by the run, 20.0 of 24.0 GiB in use on the card');
    expect(fact('GPU temperature')).toBe('–');
  });

  it.each([
    [{ process_memory_bytes: null }, '20.0 of 24.0 GiB in use on the card'],
    [{ memory_total_bytes: null }, '6.0 GiB held by the run, 20.0 GiB in use on the card'],
    [{ memory_used_bytes: null, memory_total_bytes: null }, '6.0 GiB held by the run'],
    [{ memory_used_bytes: null, memory_total_bytes: null, process_memory_bytes: null }, '–'],
  ])('says only what it was told about GPU memory: %o', (missing, shown) => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent({ heartbeat: heartbeat({ gpu: { ...GPU, ...missing } }) }));

    expect(fact('GPU memory')).toBe(shown);
  });

  it('shows no GPU figures for a run on the CPU, but says what it runs on', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(
      runEvent({ info: { ...INFO, device: 'cpu (16 threads)' }, heartbeat: heartbeat({ gpu: null }) }),
    );

    expect(fact('Device')).toBe('cpu (16 threads)');
    expect(screen.queryByText('GPU busy')).not.toBeInTheDocument();
    expect(screen.queryByText('GPU memory')).not.toBeInTheDocument();
  });

  it('does not pass a dead run’s last GPU figures off as current', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent({ stale: true, heartbeat: heartbeat({ gpu: GPU }) }));

    expect(screen.queryByText('GPU busy')).not.toBeInTheDocument();
    expect(screen.queryByText('GPU memory')).not.toBeInTheDocument();
    expect(screen.queryByText('GPU temperature')).not.toBeInTheDocument();
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

  it('charts the gradient norm against the clip threshold, and how often it was clipped', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ info: { ...INFO, config: { optimizer: { gradient_clip: 1.0 } } } }));
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 50, split: 'train', loss: 3.0, gradient_norm: 2.5, clipped_fraction: 1.0 },
        { step: 100, split: 'train', loss: 2.5, gradient_norm: 0.8, clipped_fraction: 0.25 },
      ],
    });

    expect(FakeUPlot.withSeries('gradient norm').data).toEqual([
      [50, 100],
      [2.5, 0.8],
      [1.0, 1.0],
    ]);
    expect(screen.getByText('gradient norm 0.8 at step 100; clipped above 1 at step 100')).toBeInTheDocument();
    expect(FakeUPlot.withSeries('clipped').data).toEqual([
      [50, 100],
      [1.0, 0.25],
    ]);
  });

  it('fits the y-axis of every chart to its data rather than to the clip threshold', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ info: { ...INFO, config: { optimizer: { gradient_clip: 1.0 } } } }));
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 50, split: 'train', loss: 1.52, learning_rate: 0.001, gradient_norm: 0.35, clipped_fraction: 0 },
        { step: 100, split: 'validation', loss: 1.44, top1: 0.3, top5: 0.6, illegal_top_move_rate: 0.1 },
      ],
    });

    // The logarithmic charts fit their data rather than the powers of ten around it.
    for (const label of ['validation', 'gradient norm']) {
      const [low, high] = FakeUPlot.withSeries(label).yRange(1.44, 1.52)!;
      expect(low).toBeGreaterThan(1.4);
      expect(high).toBeLessThan(1.6);
    }
    // uPlot fits the linear ones to the data in view itself.
    for (const label of ['top-1', 'learning rate', 'illegal top move', 'clipped']) {
      expect(FakeUPlot.withSeries(label).yRange(0.3, 0.6)).toBeNull();
    }
    // The clip threshold is drawn, but the axis is fitted to the gradient norm alone.
    const norm = FakeUPlot.withSeries('gradient norm').options.series;
    expect(norm.map((series) => series.auto)).toEqual([undefined, true, false]);
  });

  it('fits the y-axis to the clip threshold when the gradient norm is hidden', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ info: { ...INFO, config: { optimizer: { gradient_clip: 1.0 } } } }));
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [{ step: 50, split: 'train', loss: 1.5, gradient_norm: 0.35 }],
    });
    const chart = FakeUPlot.withSeries('gradient norm');
    // The viewer clicks the gradient norm in the legend, which leaves uPlot nothing to fit.
    chart.options.series[1].show = false;

    const [low, high] = chart.yRange(null, null)!;

    expect(low).toBeLessThan(1);
    expect(low).toBeGreaterThan(0.9);
    expect(high).toBeGreaterThan(1);
    expect(high).toBeLessThan(1.1);
    // With both hidden there is nothing to fit, as uPlot has it.
    chart.options.series[2].show = false;
    expect(chart.yRange(null, null)).toEqual([null, null]);
  });

  it('draws no threshold for a run that does not clip', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ info: { ...INFO, config: { optimizer: { gradient_clip: 0 } } } }));
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [{ step: 50, split: 'train', loss: 3.0, gradient_norm: 2.5, clipped_fraction: 0 }],
    });

    expect(FakeUPlot.withSeries('gradient norm').options.series).toHaveLength(2);
    expect(FakeUPlot.withSeries('gradient norm').data).toEqual([[50], [2.5]]);
  });

  it('leaves out the gradient charts for a run logged before there were any', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ info: { ...INFO, config: { optimizer: { gradient_clip: 1.0 } } } }));
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [{ step: 50, split: 'train', loss: 3.0, learning_rate: 0.001 }],
    });

    expect(() => FakeUPlot.withSeries('gradient norm')).toThrow();
    expect(screen.queryByText('Gradient norm, before clipping')).toBeNull();
    expect(screen.queryByText('Steps clipped')).toBeNull();
  });

  it('works out no chart again when a heartbeat brings the same run in a new object', () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    const configured = { ...INFO, config: { optimizer: { gradient_clip: 1.0 } } };
    socket.open();
    socket.deliver(runEvent({ info: configured }));
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [{ step: 50, split: 'train', loss: 3.0, gradient_norm: 2.5, clipped_fraction: 1 }],
    });
    const drawn = FakeUPlot.live().map((chart) => chart.data);

    socket.deliver(runEvent({ info: structuredClone(configured), heartbeat: heartbeat({ step: 60 }) }));

    FakeUPlot.live().forEach((chart, index) => expect(chart.data).toBe(drawn[index]));
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

  it('shows a run by its title, with its name, tags and notes', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(
      runEvent({ notes: { title: 'Wide MLP', tags: ['mlp', 'wide'], notes: 'Width 2048.' } }),
    );

    expect(screen.getByRole('heading', { name: 'Wide MLP' })).toBeInTheDocument();
    expect(screen.getByText('mlp-big')).toBeInTheDocument();
    expect(within(screen.getByRole('list', { name: 'Tags' })).getByText('wide')).toBeInTheDocument();
    expect(screen.getByText('Width 2048.')).toBeInTheDocument();
  });

  it('follows notes changed elsewhere, such as from the command line', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent());
    expect(screen.getByText('No notes.')).toBeInTheDocument();

    FakeWebSocket.latest.deliver(runEvent({ notes: { title: 'Renamed', tags: [], notes: '' } }));

    expect(screen.getByRole('heading', { name: 'Renamed' })).toBeInTheDocument();
  });

  it('saves a title, tags and notes, and shows what the server kept', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(Response.json({ title: 'Wide MLP', tags: ['mlp', 'wide'], notes: 'Better.' }));
    vi.stubGlobal('fetch', fetch);
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent({ notes: { title: null, tags: ['mlp'], notes: '' } }));

    fireEvent.click(screen.getByRole('button', { name: 'Edit title, tags and notes' }));
    expect(screen.getByLabelText('Tags')).toHaveValue('mlp');
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Wide MLP' } });
    fireEvent.change(screen.getByLabelText('Tags'), { target: { value: 'mlp, wide' } });
    fireEvent.change(screen.getByLabelText('Notes'), { target: { value: 'Better.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(await screen.findByRole('heading', { name: 'Wide MLP' })).toBeInTheDocument();
    expect(screen.queryByRole('form', { name: 'Edit notes' })).not.toBeInTheDocument();
    expect(screen.getByText('Better.')).toBeInTheDocument();
    const [url, request] = fetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://localhost:3000/chess/api/runs/mlp-big/notes');
    expect(request.method).toBe('PUT');
    expect(JSON.parse(request.body as string)).toEqual({
      title: 'Wide MLP',
      tags: ['mlp', 'wide'],
      notes: 'Better.',
    });
  });

  it('does not flash the old notes back when a message read before the save arrives after it', async () => {
    const saved = { title: 'New', tags: [], notes: '' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json(saved)));
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    const old = { title: 'Old', tags: [], notes: '' };
    socket.deliver(runEvent({ notes: old }));
    fireEvent.click(screen.getByRole('button', { name: 'Edit title, tags and notes' }));
    fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'New' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByRole('heading', { name: 'New' })).toBeInTheDocument();

    // The server read the notes just before the save, and sends them because the heartbeat moved.
    socket.deliver(runEvent({ notes: { ...old }, heartbeat: heartbeat({ step: 1_001 }) }));
    expect(screen.getByRole('heading', { name: 'New' })).toBeInTheDocument();

    socket.deliver(runEvent({ notes: { ...saved }, heartbeat: heartbeat({ step: 1_002 }) }));
    expect(screen.getByRole('heading', { name: 'New' })).toBeInTheDocument();

    // And whatever is written after that, from anywhere, is followed again.
    socket.deliver(runEvent({ notes: { title: 'Newer', tags: [], notes: '' } }));
    expect(screen.getByRole('heading', { name: 'Newer' })).toBeInTheDocument();
  });

  it('does not flash the first of two quick saves back while the second is on its way', async () => {
    const first = { title: 'First', tags: [], notes: '' };
    const second = { title: 'Second', tags: [], notes: '' };
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValueOnce(Response.json(first)).mockResolvedValueOnce(Response.json(second)),
    );
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent({ notes: { title: 'Old', tags: [], notes: '' } }));
    for (const title of ['First', 'Second']) {
      fireEvent.click(screen.getByRole('button', { name: 'Edit title, tags and notes' }));
      fireEvent.change(screen.getByLabelText('Title'), { target: { value: title } });
      fireEvent.click(screen.getByRole('button', { name: 'Save' }));
      expect(await screen.findByRole('heading', { name: title })).toBeInTheDocument();
    }

    // Read between the two saves, and sent after both.
    socket.deliver(runEvent({ notes: { ...first }, heartbeat: heartbeat({ step: 1_001 }) }));
    expect(screen.getByRole('heading', { name: 'Second' })).toBeInTheDocument();

    socket.deliver(runEvent({ notes: { ...second }, heartbeat: heartbeat({ step: 1_002 }) }));
    socket.deliver(runEvent({ notes: { title: 'Elsewhere', tags: [], notes: '' } }));
    expect(screen.getByRole('heading', { name: 'Elsewhere' })).toBeInTheDocument();
  });

  it('keeps what was typed and says why when the notes are refused', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        Response.json(
          { detail: [{ msg: "Value error, invalid tag 'a/b c': a tag is one word" }] },
          { status: 422 },
        ),
      ),
    );
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent());
    fireEvent.click(screen.getByRole('button', { name: 'Edit title, tags and notes' }));
    fireEvent.change(screen.getByLabelText('Notes'), { target: { value: 'Keep me.' } });

    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      "Could not save: invalid tag 'a/b c': a tag is one word",
    );
    expect(screen.getByLabelText('Notes')).toHaveValue('Keep me.');
  });

  it('puts the notes back as they were when editing is cancelled', () => {
    render(<RunView name="mlp-big" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(runEvent({ notes: { title: null, tags: [], notes: 'Old.' } }));
    fireEvent.click(screen.getByRole('button', { name: 'Edit title, tags and notes' }));
    fireEvent.change(screen.getByLabelText('Notes'), { target: { value: 'New.' } });

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(screen.getByText('Old.')).toBeInTheDocument();
  });

  it('charts against positions seen or wall time, and remembers which', async () => {
    render(<RunView name="mlp-big" />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(runEvent());
    socket.deliver({
      type: 'metrics',
      reset: true,
      records: [
        { step: 50, split: 'train', loss: 3.0, elapsed: 30, positions_seen: 819_200 },
        { step: 100, split: 'train', loss: 2.5, elapsed: 61, positions_seen: 1_638_400 },
      ],
    });

    fireEvent.click(screen.getByRole('radio', { name: 'Positions seen' }));

    const chart = FakeUPlot.withSeries('validation');
    expect(chart.data[0]).toEqual([819_200, 1_638_400]);
    expect(chart.options.series[0].label).toBe('Positions seen');
    expect(screen.getByText(/train 2\.5 at 1\.64M positions/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('radio', { name: 'Wall time' }));

    expect(FakeUPlot.withSeries('validation').data[0]).toEqual([30, 61]);
    await waitFor(() =>
      expect(window.localStorage.getItem('chess-ai.chart-x-axis')).toBe('time'),
    );
  });
});
