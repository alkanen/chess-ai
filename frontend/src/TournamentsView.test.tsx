import { act, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Result, TournamentPage, TournamentSummary } from './api';
import { App } from './App';
import { tournamentGameHash, tournamentGameInHash } from './tournaments';
import { foolsMateFile } from './test/replayFile';

// uPlot draws on a canvas, which jsdom does not have; see RunView.test.tsx for the charts.
vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

const ENTRANTS = ['tiny step 2 (rating 1500)', 'tiny step 4 (rating 1800)', 'Stockfish 1400'];

const SUMMARIES: TournamentSummary[] = [
  {
    name: 'cup',
    created: '2026-10-28T10:00:00Z',
    entrants: ENTRANTS,
    games_played: 4,
    games_total: 6,
    playing: true,
    error: null,
  },
  {
    name: 'broken',
    created: '2026-10-27T10:00:00Z',
    entrants: [],
    games_played: 0,
    games_total: 0,
    playing: false,
    error: 'line 3 of games.jsonl is not a game',
  },
];

const cell = (points: number, games: number) => ({ points, games });

const PAGE: TournamentPage = {
  name: 'cup',
  definition: {
    name: 'cup',
    entrants: [
      {
        kind: 'model',
        run: 'tiny',
        step: 2,
        checkpoint_written: null,
        rating: 1500,
        strategy: 'argmax',
        temperature: 1,
      },
      {
        kind: 'model',
        run: 'tiny',
        step: 4,
        checkpoint_written: null,
        rating: 1800,
        strategy: 'argmax',
        temperature: 1,
      },
      { kind: 'stockfish', elo: 1400, move_time: 0.1 },
    ],
    games_per_pairing: 2,
    openings: 'standard',
    opening_set: 'standard v1',
    seed: 7,
    anchor: null,
    created: '2026-10-28T10:00:00Z',
  },
  entrants: ENTRANTS,
  leaderboard: {
    standings: [
      { entrant: 1, rating: 120.4, low: -60.2, high: 300.9, points: 2.5, games: 3 },
      { entrant: 2, rating: -10, low: -190, high: 170, points: 1, games: 2 },
      { entrant: 0, rating: -110.4, low: -290, high: 69.6, points: 0.5, games: 3 },
    ],
    crosstable: [
      [null, cell(0.5, 2), cell(0, 1)],
      [cell(1.5, 2), null, cell(1, 1)],
      [cell(1, 1), cell(0, 1), null],
    ],
    games_played: 4,
    games_total: 6,
  },
  games: [
    game(0, 1, 0, 1, '1/2-1/2'),
    game(1, 1, 0, 2, '0-1'),
    game(2, 1, 1, 2, '1-0'),
    game(3, 2, 1, 0, '1-0'),
  ],
  playing: true,
};

function game(index: number, round: number, white: number, black: number, result: Result) {
  return {
    index,
    round,
    white,
    black,
    result,
    termination: 'checkmate' as const,
    opening: 'Ruy Lopez',
    plies: 41,
  };
}

/** The server: the tournaments above, a tournament game replayed as the fool's mate. */
function server(url: string): Promise<Response> {
  const asked = new URL(url);
  const path = asked.pathname.replace('/chess/api/', '');
  if (path === 'tournaments') {
    return Promise.resolve(Response.json(SUMMARIES));
  }
  if (path === 'tournaments/cup') {
    return Promise.resolve(Response.json(PAGE));
  }
  if (path === 'tournaments/cup/replay') {
    const index = Number(asked.searchParams.get('game'));
    return Promise.resolve(
      Response.json({ ...foolsMateFile, selected: { ...foolsMateFile.selected, index } }),
    );
  }
  if (path === 'replay/saved') {
    return Promise.resolve(Response.json([]));
  }
  return Promise.resolve(Response.json({ detail: `no ${path}` }, { status: 404 }));
}

describe('tournament game addresses', () => {
  it('number the games from 1 and read back what they wrote', () => {
    const hash = tournamentGameHash({ name: 'cup 2', index: 3 });
    expect(hash).toBe('#replay/tournament/cup%202/4');
    expect(tournamentGameInHash(hash)).toEqual({ name: 'cup 2', index: 3 });
    expect(tournamentGameInHash('#replay/tournament/cup/0')).toBeNull();
    expect(tournamentGameInHash('#replay/ladder/cup/2/1')).toBeNull();
  });
});

describe('the tournaments pages', () => {
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
    vi.useRealTimers();
    window.location.hash = '';
  });

  function open(hash: string) {
    window.location.hash = hash;
    return render(<App />);
  }

  it('lists every tournament with its entrants and how far it has got', async () => {
    open('#tournaments');

    const row = (await screen.findByRole('link', { name: 'cup' })).closest('tr')!;
    const cells = within(row).getAllByRole('cell').map((one) => one.textContent);
    expect(cells).toContain(ENTRANTS.join(', '));
    expect(cells).toContain('4 of 6 (being played)');
    expect(screen.getByRole('link', { name: 'cup' })).toHaveAttribute('href', '#tournaments/cup');
    const broken = screen.getByRole('link', { name: 'broken' }).closest('tr')!;
    expect(broken).toHaveTextContent('Cannot be read: line 3 of games.jsonl is not a game');
  });

  it('shows a tournament best first, with its ranges, crosstable and games', async () => {
    open('#tournaments/cup');

    const leaderboard = await screen.findByRole('table', { name: 'Leaderboard' });
    const rows = within(leaderboard)
      .getAllByRole('row')
      .slice(1)
      .map((row) => within(row).getAllByRole('cell').map((one) => one.textContent));
    expect(rows).toEqual([
      ['1', '2 tiny step 4 (rating 1800)', '+120', '−60 to +301', '2½/3'],
      ['2', '3 Stockfish 1400', '−10', '−190 to +170', '1/2'],
      ['3', '1 tiny step 2 (rating 1500)', '−110', '−290 to +70', '½/3'],
    ]);
    expect(screen.getByText(/4 of 6 games/)).toHaveTextContent(/being played/);
    expect(screen.getByText(/around an average of 0/)).toBeInTheDocument();

    const crosstable = screen.getByRole('table', { name: 'Crosstable' });
    const second = within(crosstable).getAllByRole('row')[2];
    expect(within(second).getAllByRole('cell').map((one) => one.textContent)).toEqual([
      '1½/2',
      '–',
      '1/1',
    ]);

    const games = screen.getByRole('table', { name: 'Games' });
    const fourth = within(games).getAllByRole('row')[4];
    expect(fourth).toHaveTextContent('2 tiny step 4 (rating 1800)');
    expect(fourth).toHaveTextContent('1 tiny step 2 (rating 1500)');
    expect(within(fourth).getByRole('link', { name: '4' })).toHaveAttribute(
      'href',
      '#replay/tournament/cup/4',
    );
    expect(screen.getByRole('link', { name: /download/i }).getAttribute('href')).toMatch(
      /\/chess\/api\/tournaments\/cup\/pgn$/,
    );
  });

  it('says what the ratings are measured from when the tournament is anchored', async () => {
    const anchored = {
      ...PAGE,
      definition: { ...PAGE.definition, anchor: { entrant: 2, rating: 1400 } },
      leaderboard: {
        ...PAGE.leaderboard,
        standings: PAGE.leaderboard.standings.map((one) => ({
          ...one,
          rating: one.rating + 1410,
          low: one.low + 1410,
          high: one.high + 1410,
        })),
      },
    };
    fetch.mockImplementation((url: string) =>
      url.endsWith('tournaments/cup')
        ? Promise.resolve(Response.json(anchored))
        : server(url),
    );
    open('#tournaments/cup');

    const leaderboard = await screen.findByRole('table', { name: 'Leaderboard' });
    expect(within(within(leaderboard).getAllByRole('row')[1]).getAllByRole('cell')[2]).toHaveTextContent(
      /^1530$/,
    );
    expect(screen.getByText(/from 3 Stockfish 1400 at 1400/)).toBeInTheDocument();
  });

  it('asks again while the tournament is being played, and stops once it is not', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let playing = true;
    fetch.mockImplementation((url: string) =>
      url.endsWith('tournaments/cup')
        ? Promise.resolve(Response.json({ ...PAGE, playing }))
        : server(url),
    );
    open('#tournaments/cup');
    await screen.findByRole('table', { name: 'Leaderboard' });
    const asked = () => fetch.mock.calls.filter(([url]) => url.endsWith('tournaments/cup')).length;
    expect(asked()).toBe(1);

    playing = false;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(asked()).toBe(2);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(asked()).toBe(2);
  });

  it('replays a tournament game, with a link back to the tournament', async () => {
    open(tournamentGameHash({ name: 'cup', index: 2 }));

    expect(await screen.findByRole('link', { name: 'cup' })).toHaveAttribute(
      'href',
      '#tournaments/cup',
    );
    const replayed = fetch.mock.calls.map(([url]) => String(url)).find((url) => url.includes('replay?'));
    expect(replayed).toContain('/chess/api/tournaments/cup/replay?game=2');
  });
});
