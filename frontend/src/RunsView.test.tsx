import { act, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunSummary } from './api';
import { RUNS_REFRESH_MS, RunsView } from './RunsView';

function summary(changes: Partial<RunSummary> = {}): RunSummary {
  return {
    name: 'mlp-big',
    architecture: 'mlp',
    dataset: 'lichess-2024',
    created: '2026-10-01T10:00:00Z',
    status: 'running',
    stale: false,
    updated: new Date().toISOString(),
    step: 7_500,
    steps: 15_000,
    epoch: 0.2,
    positions_per_second: 340_000,
    eta_seconds: 600,
    message: null,
    checkpoints: 3,
    latest_train: { step: 7_500, split: 'train', loss: 2.51234 },
    latest_validation: {
      step: 7_000,
      split: 'validation',
      loss: 2.6,
      top1: 0.4594,
      illegal_top_move_rate: 0.012,
    },
    ...changes,
  };
}

/** The server lists these runs, once each time it is asked, the last of them from then on. */
function serving(...answers: RunSummary[][]) {
  const fetch = vi.fn();
  for (const answer of answers) {
    fetch.mockResolvedValueOnce(Response.json(answer));
  }
  fetch.mockResolvedValue(Response.json(answers.at(-1)));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

/** The cells of the row for the run called `name`. */
function row(name: string) {
  return within(screen.getByRole('rowheader', { name }).closest('tr')!);
}

describe('RunsView', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('lists every run with its state, what it is, and its latest metrics', async () => {
    const fetch = serving([summary(), summary({ name: 'old', status: 'finished', step: 15_000 })]);

    render(<RunsView />);

    expect(await screen.findByRole('rowheader', { name: 'mlp-big' })).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith('http://localhost:3000/chess/api/runs');
    const running = row('mlp-big');
    expect(running.getByText('running')).toBeInTheDocument();
    expect(running.getByText('mlp')).toBeInTheDocument();
    expect(running.getByText('lichess-2024')).toBeInTheDocument();
    expect(running.getByText('7,500 / 15,000 (50%)')).toBeInTheDocument();
    expect(running.getByText('2.512')).toBeInTheDocument();
    expect(running.getByText('45.9%')).toBeInTheDocument();
    expect(running.getByText('1.2%')).toBeInTheDocument();
    expect(running.getByRole('link', { name: 'mlp-big' })).toHaveAttribute('href', '#runs/mlp-big');
    expect(row('old').getByText('finished')).toBeInTheDocument();
  });

  it('flags a run whose heartbeat has stopped', async () => {
    serving([summary({ stale: true })]);

    render(<RunsView />);

    const badge = await screen.findByText('stale');
    expect(badge).toHaveAttribute('title', expect.stringContaining('stopped reporting'));
  });

  it('shows a run that has not logged anything yet without numbers', async () => {
    serving([summary({ status: null, step: null, latest_train: null, latest_validation: null })]);

    render(<RunsView />);

    expect(await screen.findByText('starting')).toBeInTheDocument();
  });

  it('keeps itself up to date', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    serving([summary()], [summary({ status: 'crashed', message: 'boom' })]);
    render(<RunsView />);
    expect(await screen.findByText('running')).toBeInTheDocument();

    await act(() => vi.advanceTimersByTimeAsync(RUNS_REFRESH_MS));

    expect(await screen.findByText('crashed')).toBeInTheDocument();
  });

  it('says when there are no runs', async () => {
    serving([]);

    render(<RunsView />);

    expect(await screen.findByText(/No training runs here yet/)).toBeInTheDocument();
  });

  it('says why the runs could not be listed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('', { status: 500, statusText: 'Server Error' })),
    );

    render(<RunsView />);

    expect(await screen.findByRole('alert')).toHaveTextContent('500 Server Error');
  });
});
