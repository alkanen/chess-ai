import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunSummary } from './api';
import { MAX_SERIES } from './runCharts';
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

  it('shows a run by its title, with its name and its tags', async () => {
    serving([summary({ title: 'Wide MLP', tags: ['mlp', 'wide'] })]);

    render(<RunsView />);

    const link = await screen.findByRole('link', { name: 'Wide MLP' });
    expect(link).toHaveAttribute('href', '#runs/mlp-big');
    const named = within(link.closest('tr')!);
    expect(named.getByText('mlp-big')).toBeInTheDocument();
    expect(within(named.getByRole('list', { name: 'Tags' })).getAllByRole('listitem')).toHaveLength(2);
  });

  it('filters the list by a tag, and back to every run', async () => {
    serving([
      summary({ name: 'a', tags: ['mlp', 'wide'] }),
      summary({ name: 'b', tags: ['mlp'] }),
      summary({ name: 'c', tags: [] }),
    ]);
    render(<RunsView />);
    await screen.findByRole('rowheader', { name: 'a' });
    const filter = within(screen.getByRole('group', { name: 'Filter by tag' }));

    fireEvent.click(filter.getByRole('button', { name: 'wide' }));

    expect(screen.getByRole('rowheader', { name: 'a' })).toBeInTheDocument();
    expect(screen.queryByRole('rowheader', { name: 'b' })).not.toBeInTheDocument();
    expect(screen.queryByRole('rowheader', { name: 'c' })).not.toBeInTheDocument();
    expect(filter.getByRole('button', { name: 'wide' })).toHaveAttribute('aria-pressed', 'true');

    // A tag in a run's row filters by it too.
    fireEvent.click(filter.getByRole('button', { name: 'All' }));
    fireEvent.click(row('b').getByRole('button', { name: 'mlp' }));

    expect(screen.getAllByRole('rowheader').map((header) => header.textContent)).toEqual(['a', 'b']);

    fireEvent.click(filter.getByRole('button', { name: 'All' }));

    expect(screen.getAllByRole('rowheader')).toHaveLength(3);
  });

  it('offers to compare the runs ticked, once there are two of them', async () => {
    serving([summary({ name: 'a' }), summary({ name: 'b' }), summary({ name: 'c' })]);
    render(<RunsView />);
    await screen.findByRole('rowheader', { name: 'a' });

    fireEvent.click(screen.getByRole('checkbox', { name: 'Compare a' }));
    expect(screen.queryByRole('link', { name: /Compare/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Compare c' }));

    expect(screen.getByRole('link', { name: 'Compare 2 runs' })).toHaveAttribute(
      'href',
      '#compare/a,c',
    );
  });

  it('keeps runs ticked while a filter hides them', async () => {
    serving([summary({ name: 'a', tags: ['x'] }), summary({ name: 'b', tags: ['y'] })]);
    render(<RunsView />);
    await screen.findByRole('rowheader', { name: 'a' });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Compare a' }));

    fireEvent.click(
      within(screen.getByRole('group', { name: 'Filter by tag' })).getByRole('button', { name: 'y' }),
    );
    fireEvent.click(screen.getByRole('checkbox', { name: 'Compare b' }));

    expect(screen.getByRole('link', { name: 'Compare 2 runs' })).toHaveAttribute(
      'href',
      '#compare/a,b',
    );
  });

  it('lets no more runs be ticked than a chart can tell apart', async () => {
    const names = Array.from({ length: MAX_SERIES + 1 }, (_, index) => `run-${index}`);
    serving(names.map((name) => summary({ name })));
    render(<RunsView />);
    await screen.findByRole('rowheader', { name: 'run-0' });

    names.slice(0, MAX_SERIES).forEach((name) =>
      fireEvent.click(screen.getByRole('checkbox', { name: `Compare ${name}` })),
    );

    expect(screen.getByRole('checkbox', { name: `Compare run-${MAX_SERIES}` })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: 'Compare run-0' })).toBeEnabled();
  });
});
