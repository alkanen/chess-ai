import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Paused, PlayerInfo, RunSummary } from './api';
import { PausedPanel } from './PausedPanel';

const RUNS: RunSummary[] = [
  {
    name: 'newer',
    architecture: 'resnet',
    created: '2026-10-05T10:00:00Z',
    status: 'running',
    step: 400,
    steps: 8000,
    checkpoints: 1,
  },
  {
    name: 'tiny',
    architecture: 'mlp',
    created: '2026-09-20T10:00:00Z',
    status: 'finished',
    step: 4,
    steps: 4,
    checkpoints: 2,
  },
];

const GONE: PlayerInfo = {
  name: 'tiny step 2',
  accepts_moves: false,
  model: { run: 'tiny', checkpoint: 2, rating: 1500, strategy: 'argmax', temperature: null },
  stockfish: null,
};

const PAUSED: Paused = { side: 'black', reason: 'run tiny has no checkpoint from step 2' };

/** A server with `runs` on it, which takes any replacement it is sent. */
function serving(runs: RunSummary[] = RUNS, replaced: Response = Response.json({})) {
  return vi.fn((url: string) => {
    if (url.endsWith('/api/runs')) {
      return Promise.resolve(Response.json(runs));
    }
    if (url.endsWith('/replace')) {
      return Promise.resolve(replaced);
    }
    const asked = /\/api\/runs\/([^/]+)\/checkpoints$/.exec(url);
    if (asked !== null) {
      return Promise.resolve(Response.json({ run: decodeURIComponent(asked[1]), checkpoints: [] }));
    }
    return Promise.resolve(Response.json({}));
  });
}

function posted(fetch: ReturnType<typeof vi.fn>): [string, unknown][] {
  return (fetch.mock.calls as [string, RequestInit?][])
    .filter(([, init]) => init?.method === 'POST')
    .map(([url, init]) => [url, JSON.parse(init?.body as string)]);
}

describe('PausedPanel', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('tells a watcher why the game waits, and offers nothing to do about it', () => {
    vi.stubGlobal('fetch', serving());
    render(
      <PausedPanel link="watching" paused={PAUSED} player={GONE} access="watch" disabled={false} />,
    );

    expect(screen.getByRole('status')).toHaveTextContent(
      "Black's player, tiny step 2, cannot play on: run tiny has no checkpoint from step 2",
    );
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('offers the person playing a checkpoint of the same run, asked to play the same way', async () => {
    const fetch = serving();
    vi.stubGlobal('fetch', fetch);
    render(
      <PausedPanel link="mine" paused={PAUSED} player={GONE} access="white" disabled={false} />,
    );

    await vi.waitFor(() => expect(screen.getByLabelText('Black run')).toHaveValue('tiny'));
    expect(screen.getByLabelText('Black rating')).toHaveValue(1500);
    fireEvent.click(screen.getByRole('button', { name: 'Play on with this checkpoint' }));

    await vi.waitFor(() => expect(posted(fetch)).toHaveLength(1));
    // Taken: what remains is for the game to say the new player is there.
    expect(screen.getByRole('button', { name: 'Play on with this checkpoint' })).toBeDisabled();
    const [[url, body]] = posted(fetch);
    expect(url).toMatch(/\/chess\/api\/games\/mine\/replace$/);
    expect(body).toEqual({
      kind: 'model',
      run: 'tiny',
      checkpoint: 'best',
      rating: 1500,
      strategy: 'argmax',
      temperature: 1,
    });
  });

  it('offers another run when the one that played is gone', async () => {
    vi.stubGlobal('fetch', serving([RUNS[0]]));
    render(
      <PausedPanel link="mine" paused={PAUSED} player={GONE} access="white" disabled={false} />,
    );

    await vi.waitFor(() => expect(screen.getByLabelText('Black run')).toHaveValue('newer'));
  });

  it('says why the server would not take the checkpoint chosen', async () => {
    const refused = Response.json(
      { detail: 'run tiny has not saved a checkpoint yet' },
      { status: 400 },
    );
    vi.stubGlobal('fetch', serving(RUNS, refused));
    render(
      <PausedPanel link="mine" paused={PAUSED} player={GONE} access="white" disabled={false} />,
    );
    await vi.waitFor(() => expect(screen.getByLabelText('Black run')).toHaveValue('tiny'));

    fireEvent.click(screen.getByRole('button', { name: 'Play on with this checkpoint' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'run tiny has not saved a checkpoint yet',
    );
    expect(screen.getByRole('button', { name: 'Play on with this checkpoint' })).toBeEnabled();
  });
});
