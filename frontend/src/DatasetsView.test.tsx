import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DatasetGame, DatasetManifest, DatasetSummary } from './api';
import { App } from './App';
import { datasetGameInHash, datasetHash, datasetInHash, PAGE_SIZE } from './datasetPlaces';
import { ratingBars, resultBars, unknownLast } from './DatasetStatistics';
import { filterPhrases } from './DatasetsView';
import { foolsMateFile } from './test/replayFile';

// uPlot draws on a canvas, which jsdom does not have; see RunView.test.tsx for the charts.
vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

const MANIFEST: DatasetManifest = {
  format_version: 1,
  name: 'lichess-2024',
  created: '2026-10-01T12:00:00Z',
  move_vocabulary_size: 1968,
  validation_fraction: 0.02,
  rating_source: 'auto',
  sources: [
    {
      path: '/data/pgn/lichess_2024-01.pgn',
      bytes: 3 * 1024 * 1024,
      games_read: 130,
      games_kept: 120,
      error: null,
      went_away: false,
    },
    {
      path: '/data/pgn/broken.pgn',
      bytes: 2048,
      games_read: 5,
      games_kept: 3,
      error: 'ValueError: not PGN',
      went_away: false,
    },
  ],
  filters: {},
  splits: { train: { games: 120, positions: 9000 }, validation: { games: 3, positions: 210 } },
  skipped: { no_result: 10, unsupported_variant: 2 },
  filtered: {},
  not_targets: {},
  reached_max_games: false,
  statistics: {
    results: { '1-0': 60, '0-1': 50, '1/2-1/2': 13 },
    time_controls: { bullet: 20, blitz: 80, rapid: 23 },
    rating_sources: { lichess: 123 },
    ratings: { '1400': 100, '1600': 140 },
    ratings_unknown: 6,
  },
};

const DATASETS: DatasetSummary[] = [
  { name: 'lichess-2024', manifest: MANIFEST, error: null },
  { name: 'old', manifest: null, error: 'old.json is dataset format version 0; rebuild' },
];

function game(index: number): DatasetGame {
  return {
    index,
    plies: 41,
    white_rating: 1500 + index,
    black_rating: index === 0 ? null : 1600,
    result: '1-0',
    date: '2024.01.05',
    time_control: 'blitz',
    rating_source: 'lichess',
    source: '/data/pgn/lichess_2024-01.pgn',
    custom_start: false,
  };
}

/** The server: the datasets above, and the train split's 120 games a page at a time. */
function server(url: string): Promise<Response> {
  const asked = new URL(url);
  const path = asked.pathname.replace('/chess/api/', '');
  if (path === 'datasets') {
    return Promise.resolve(Response.json(DATASETS));
  }
  if (path === 'datasets/lichess-2024') {
    return Promise.resolve(Response.json(DATASETS[0]));
  }
  const page = /^datasets\/lichess-2024\/(\w+)\/games$/.exec(path);
  if (page !== null) {
    const offset = Number(asked.searchParams.get('offset'));
    const limit = Number(asked.searchParams.get('limit'));
    const total = MANIFEST.splits[page[1]].games;
    const games = [];
    for (let index = offset; index < Math.min(offset + limit, total); index++) {
      games.push(game(index));
    }
    return Promise.resolve(
      Response.json({ dataset: 'lichess-2024', split: page[1], total, offset, games }),
    );
  }
  if (/^datasets\/lichess-2024\/train\/games\/\d+$/.test(path)) {
    return Promise.resolve(
      Response.json({ ...foolsMateFile.selected, white: 'rated 1500', black: 'rating unknown' }),
    );
  }
  if (path === 'replay/saved') {
    return Promise.resolve(Response.json([]));
  }
  return Promise.resolve(Response.json({ detail: `no ${path}` }, { status: 404 }));
}

describe('filterPhrases', () => {
  it('says a limit of 0, which is a filter like any other', () => {
    // A minimum rating of 0 means "rated players only", and 0 is falsy.
    expect(filterPhrases({ min_rating: 0, min_clock: 0 })).toEqual([
      'player to move rated at least 0; an unknown rating does not',
      'player to move with at least 0s on the clock',
    ]);
    expect(filterPhrases({ max_rating: 0 })).toEqual([
      'player to move rated at most 0; an unknown rating does not',
    ]);
  });

  it('says nothing for no filters', () => {
    expect(filterPhrases({})).toEqual([]);
  });
});

describe('the datasets pages', () => {
  let fetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    fetch = vi.fn(server);
    vi.stubGlobal('fetch', fetch);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
    window.location.hash = '';
  });

  function open(hash: string) {
    window.location.hash = hash;
    return render(<App />);
  }

  it('lists every dataset with when it was built, its sources, filters and counts', async () => {
    open('#datasets');

    const row = (await screen.findByRole('link', { name: 'lichess-2024' })).closest('tr')!;
    const cells = within(row).getAllByRole('cell').map((cell) => cell.textContent);
    expect(cells[0]).toMatch(/2026/);
    expect(cells[1]).toBe('lichess_2024-01.pgn, broken.pgn (not all read)');
    expect(cells[2]).toBe('none');
    expect(cells.slice(3)).toEqual(['123', '9,210', '12']);
    expect(screen.getByRole('link', { name: 'lichess-2024' })).toHaveAttribute(
      'href',
      '#datasets/lichess-2024',
    );
  });

  it('lists a dataset that cannot be read, with the reason', async () => {
    open('#datasets');

    expect(
      await screen.findByText(/Cannot be read: old.json is dataset format/),
    ).toBeInTheDocument();
  });

  it('shows the statistics of one dataset', async () => {
    open('#datasets/lichess-2024');

    const ratings = await screen.findByRole('table', { name: 'Rating' });
    const rows = within(ratings)
      .getAllByRole('row')
      .slice(1)
      .map((row) => row.textContent);
    // The empty bucket between the two is part of the shape, and is drawn.
    expect(rows).toEqual([
      '1400–149910040.7%',
      '1500–159900.0%',
      '1600–169914056.9%',
      'unknown62.4%',
    ]);
    const table = (name: string) => within(screen.getByRole('table', { name }));
    expect(table('Result').getByText('White won')).toBeInTheDocument();
    expect(table('Time control').getByText('blitz')).toBeInTheDocument();
    expect(table('Skipped').getByText('unsupported variant')).toBeInTheDocument();
    expect(screen.getByText('Not read whole: ValueError: not PGN')).toBeInTheDocument();
    expect(screen.getByText('123 (train 120, validation 3)')).toBeInTheDocument();
  });

  it('folds many sources away behind their totals, with failed files still in sight', async () => {
    const many = Array.from({ length: 6 }, (_, n) => ({
      ...MANIFEST.sources[0],
      path: `/data/pgn/part-${n}.pgn`,
    }));
    const manifest = { ...MANIFEST, sources: [...many, MANIFEST.sources[1]] };
    fetch.mockImplementation((url: string) =>
      new URL(url).pathname === '/chess/api/datasets/lichess-2024'
        ? Promise.resolve(Response.json({ name: 'lichess-2024', manifest, error: null }))
        : server(url),
    );
    open('#datasets/lichess-2024');

    const summary = await screen.findByText(/^7 files, 18\.0 MB, 785 games read, 723 kept$/);
    expect(summary.closest('details')).not.toHaveAttribute('open');
    const failed = screen.getAllByText('Not read whole: ValueError: not PGN');
    expect(failed.filter((shown) => shown.closest('details') === null)).toHaveLength(1);
  });

  it('shows what a filtered dataset let through and what it trains on', async () => {
    const manifest = {
      ...MANIFEST,
      filters: { min_rating: 2000, time_controls: ['rapid', 'classical'], max_games: 5000 },
      splits: {
        train: { games: 120, positions: 9000, targets: 6000 },
        validation: { games: 3, positions: 210, targets: 140 },
      },
      filtered: { rating: 40, time_control: 7 },
      not_targets: { rating: 3000, clock: 70 },
      reached_max_games: true,
    };
    fetch.mockImplementation((url: string) =>
      new URL(url).pathname === '/chess/api/datasets/lichess-2024'
        ? Promise.resolve(Response.json({ name: 'lichess-2024', manifest, error: null }))
        : server(url),
    );
    open('#datasets/lichess-2024');

    expect(
      await screen.findByText(
        'player to move rated at least 2000; an unknown rating does not; ' +
          'time controls rapid, classical; at most 5,000 games; stopped at the maximum',
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/^6,140 positions; the rest are there/)).toBeInTheDocument();
    const table = (name: string) => within(screen.getByRole('table', { name }));
    expect(table('Filtered out').getByText('time control')).toBeInTheDocument();
    expect(table('Not trained on').getByText("mover's clock")).toBeInTheDocument();
  });

  it('says nothing of training targets for a dataset that trains on every position', async () => {
    open('#datasets/lichess-2024');

    await screen.findByRole('table', { name: 'Rating' });
    expect(screen.queryByText('Trained on')).not.toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Filtered out' })).not.toBeInTheDocument();
  });

  it('pages through a split of the games, keeping the page in the address', async () => {
    open('#datasets/lichess-2024');

    expect(await screen.findByRole('status')).toHaveTextContent('Games 1–50 of 120');
    expect(await screen.findByRole('link', { name: 'Replay game 1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Next' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Next' }));

    await screen.findByRole('link', { name: 'Replay game 101' });
    expect(window.location.hash).toBe('#datasets/lichess-2024/train/3');
    expect(screen.getByRole('status')).toHaveTextContent('Games 101–120 of 120');
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
    expect(screen.queryByRole('link', { name: 'Replay game 1' })).not.toBeInTheDocument();
  });

  it('pages by what the server has now, and says when the dataset changed under it', async () => {
    // The page was opened on a manifest of 120 training games, and the dataset has since been
    // rebuilt with 200: the rows come from the new one, so the pager has to follow them.
    fetch.mockImplementation((url: string) => {
      const asked = new URL(url);
      if (!asked.pathname.endsWith('/train/games')) {
        return server(url);
      }
      const offset = Number(asked.searchParams.get('offset'));
      const games = Array.from({ length: Math.min(PAGE_SIZE, 200 - offset) }, (_, n) =>
        game(offset + n),
      );
      return Promise.resolve(
        Response.json({ dataset: 'lichess-2024', split: 'train', total: 200, offset, games }),
      );
    });
    open('#datasets/lichess-2024/train/3');

    await screen.findByRole('link', { name: 'Replay game 101' });
    expect(screen.getByRole('status')).toHaveTextContent('Games 101–150 of 200');
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();
    expect(screen.getByText(/has changed since this page was opened/)).toBeInTheDocument();
  });

  it('says so when a page is past the last game', async () => {
    open('#datasets/lichess-2024/train/9');

    // Not "Games 120–120 of 120", which would claim a game the list does not show.
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent('Past the last of 120 games'),
    );
    expect(screen.queryByRole('link', { name: /^Replay game/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeEnabled();
  });

  it('lists the games of the other split when it is chosen', async () => {
    open('#datasets/lichess-2024');
    await screen.findByRole('link', { name: 'Replay game 1' });

    fireEvent.change(screen.getByRole('combobox', { name: 'Split' }), {
      target: { value: 'validation' },
    });

    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Games 1–3 of 3'));
    expect(window.location.hash).toBe('#datasets/lichess-2024/validation/1');
  });

  it('says what each game was, with a rating the file did not give as unknown', async () => {
    open('#datasets/lichess-2024');

    const row = (await screen.findByRole('link', { name: 'Replay game 1' })).closest('tr')!;
    expect(within(row).getAllByRole('cell').map((cell) => cell.textContent)).toEqual([
      '1500',
      '?',
      '1–0',
      '2024.01.05',
      'blitz',
      '21',
      'lichess_2024-01.pgn',
    ]);
  });

  it('opens a game in the replay viewer, with a way back to its page', async () => {
    open('#datasets/lichess-2024/train/2');
    const link = await screen.findByRole('link', { name: 'Replay game 60' });
    expect(link).toHaveAttribute('href', '#replay/dataset/lichess-2024/train/60');

    window.location.hash = link.getAttribute('href')!;

    expect(await screen.findByText('rated 1500')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      expect.stringMatching(/\/chess\/api\/datasets\/lichess-2024\/train\/games\/59$/),
    );
    expect(screen.getByRole('button', { name: 'Replay' })).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('link', { name: 'lichess-2024, train game 60' })).toHaveAttribute(
      'href',
      '#datasets/lichess-2024/train/2',
    );
  });

  it('takes the address off a dataset game when another game is opened', async () => {
    fetch.mockImplementation((url: string) =>
      new URL(url).pathname === '/chess/api/replay/saved'
        ? Promise.resolve(
            Response.json([
              {
                name: 'saved.pgn',
                event: 'chess-ai game',
                date: '2026.09.24',
                white: 'Human',
                black: 'Random mover',
                result: '0-1',
              },
            ]),
          )
        : new URL(url).pathname.startsWith('/chess/api/replay/saved/')
          ? Promise.resolve(Response.json(foolsMateFile))
          : server(url),
    );
    open('#replay/dataset/lichess-2024/train/60');
    await screen.findByText('rated 1500');

    fireEvent.click(await screen.findByRole('button', { name: /Human – Random mover/ }));

    expect(await screen.findByText('saved.pgn')).toBeInTheDocument();
    expect(window.location.hash).toBe('#replay');
  });

  it('says so when a dataset is not there', async () => {
    open('#datasets/missing');

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not open that dataset: no datasets/missing',
    );
  });
});

describe('dataset addresses', () => {
  it('round-trip a page, the first page of the train split being the short one', () => {
    expect(datasetHash({ name: 'a.b', split: 'train', page: 0 })).toBe('#datasets/a.b');
    expect(datasetInHash('#datasets/a.b')).toEqual({ name: 'a.b', split: 'train', page: 0 });
    expect(datasetInHash('#datasets/a.b/validation/4')).toEqual({
      name: 'a.b',
      split: 'validation',
      page: 3,
    });
    expect(datasetInHash('#datasets')).toBeNull();
  });

  it('name a game from 1, and refuse a game 0', () => {
    expect(datasetGameInHash('#replay/dataset/d/train/1')).toEqual({
      name: 'd',
      split: 'train',
      index: 0,
    });
    expect(datasetGameInHash('#replay/dataset/d/train/0')).toBeNull();
    expect(PAGE_SIZE).toBe(50);
  });
});

describe('the distributions', () => {
  it('list results as white won, draw, black won', () => {
    expect(resultBars({ '0-1': 1, '1-0': 2, '1/2-1/2': 3 })).toEqual([
      ['White won', 2],
      ['Draw', 3],
      ['Black won', 1],
    ]);
  });

  it('list an unknown time control after the known ones', () => {
    expect(unknownLast({ unknown: 3, bullet: 1, blitz: 2 })).toEqual([
      ['bullet', 1],
      ['blitz', 2],
      ['unknown', 3],
    ]);
  });

  it('have no rating bars for a dataset with no ratings at all', () => {
    expect(ratingBars({ ...MANIFEST.statistics, ratings: {}, ratings_unknown: 0 })).toEqual([]);
  });
});
