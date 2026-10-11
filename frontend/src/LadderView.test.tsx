import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { EvaluationEntry, LadderPoint, LadderResult } from './api';
import { LadderView } from './LadderView';
import { FakeUPlot } from './test/fakeUPlot';
import { ladderPoint, ladderResult } from './test/ladder';
import { checkpointPlayer, liveGame, STOCKFISH_PLAYER } from './test/sampleGames';

vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

function entry(step: number, updated = '2026-10-11T12:00:00Z'): EvaluationEntry {
  return { step, suite: 'stockfish-ladder', updated };
}

/**
 * Answers the requests for a run's ladders, to chart, with `points`, and those for a step's
 * ladder with what `answer` gives for that step.
 */
function serve(
  answer: (step: number) => LadderResult = (step) => ladderResult(step),
  points: () => LadderPoint[] = () => [ladderPoint(1000), ladderPoint(2000)],
) {
  const fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (/evaluations\/stockfish-ladder$/.test(url)) {
      return Response.json(points());
    }
    const match = /evaluations\/(\d+)\/stockfish-ladder$/.exec(url);
    if (match === null) {
      throw new Error(`unexpected request ${url}`);
    }
    return Response.json(answer(Number(match[1])));
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

function calls(fetch: ReturnType<typeof serve>, pattern: RegExp): number {
  return fetch.mock.calls.filter(([input]) => pattern.test(String(input))).length;
}

const POINTS = /evaluations\/stockfish-ladder$/;

/** The rows of a table, as their cells read. */
function rows(table: HTMLElement): string[][] {
  return within(table)
    .getAllByRole('row')
    .slice(1)
    .map((row) => within(row).getAllByRole('cell').map((cell) => cell.textContent ?? ''));
}

describe('LadderView', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    window.localStorage.clear();
    FakeUPlot.instances = [];
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('shows the newest checkpoint’s Elo, its score at each level, and its games', async () => {
    serve();
    render(
      <LadderView
        run="tiny run"
        evaluations={[entry(2_000), entry(1_000), { ...entry(3_000), suite: 'sample-games' }]}
        liveGame={null}
        laddered
      />,
    );

    expect(await screen.findByText('1420 (1360 to 1481)')).toBeVisible();
    expect(screen.getByText(/^Step /).textContent).toBe('Step 2,000 (2 of 2, newest)');
    const [levels, games] = screen.getAllByRole('table');
    expect(rows(levels)).toEqual([
      ['Stockfish 1350', '1.5/2 (+1 =1 −0)', '75%'],
      ['Stockfish 1500', '0/2 (+0 =0 −2)', '0%'],
    ]);
    expect(rows(games)[1]).toEqual([
      'Game 2',
      'Stockfish 1350, checkpoint as Black',
      'Italian Game',
      'Draw by threefold repetition (½–½)',
      '30',
    ]);
    expect(screen.getByRole('link', { name: 'Game 4' })).toHaveAttribute(
      'href',
      '#replay/ladder/tiny%20run/2000/4',
    );
    expect(
      screen.getByText(/rated 2000; 2 games a level; Stockfish at 0.1 s a move/),
    ).toBeInTheDocument();
  });

  it('charts the Elo of every checkpoint laddered, with its range as a band', async () => {
    serve(undefined, () => [
      ladderPoint(1_000, { rating: null, low: null, high: 1400, beyond: 'below_floor' }),
      ladderPoint(2_000),
    ]);
    render(
      <LadderView run="tiny" evaluations={[entry(1_000), entry(2_000)]} liveGame={null} laddered />,
    );

    await waitFor(() => expect(FakeUPlot.live()).toHaveLength(1));
    const chart = FakeUPlot.withSeries('Elo');
    expect(chart.data).toEqual([
      [1000, 2000],
      [null, 1420.4],
      [null, 1360.2],
      [null, 1480.6],
      [1350, null],
      [null, null],
    ]);
    expect(chart.options.bands).toEqual([expect.objectContaining({ series: [3, 2] })]);
    expect(screen.getByText(/Elo on Stockfish’s scale at 0.1 s a move/)).toBeInTheDocument();
    expect(screen.getByText('1420 (1360 to 1481) at step 2,000, a 95% range')).toBeInTheDocument();
  });

  it('asks for the chart again when a ladder is added, and not for the same ones again', async () => {
    const fetch = serve();
    const { rerender } = render(
      <LadderView run="tiny" evaluations={[entry(1_000)]} liveGame={null} laddered />,
    );
    await screen.findByText('1420 (1360 to 1481)');
    expect(calls(fetch, POINTS)).toBe(1);

    rerender(<LadderView run="tiny" evaluations={[entry(1_000)]} liveGame={null} laddered />);
    rerender(
      <LadderView run="tiny" evaluations={[entry(1_000), entry(2_000)]} liveGame={null} laddered />,
    );

    await waitFor(() => expect(calls(fetch, POINTS)).toBe(2));
  });

  it('steps back to an earlier checkpoint’s ladder', async () => {
    serve();
    render(
      <LadderView run="tiny" evaluations={[entry(1_000), entry(2_000)]} liveGame={null} laddered />,
    );
    await screen.findByRole('link', { name: 'Game 1' });

    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));

    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'Game 1' })).toHaveAttribute(
        'href',
        '#replay/ladder/tiny/1000/1',
      ),
    );
  });

  it('links to the games of a ladder played before they were listed', async () => {
    serve((step) => ladderResult(step, { games: [] }));
    render(<LadderView run="tiny" evaluations={[entry(1_000)]} liveGame={null} laddered />);

    expect(await screen.findByRole('link', { name: 'Replay the games' })).toHaveAttribute(
      'href',
      '#replay/ladder/tiny/1000/1',
    );
  });

  it('shows the ladder game being played, and leaves a sample game to its own section', () => {
    serve();
    const game = liveGame({ suite: 'stockfish-ladder', index: 12, games: 100, level: 1500 });
    game.game = { ...game.game, white: STOCKFISH_PLAYER, black: checkpointPlayer(500) };
    const { rerender } = render(
      <LadderView run="tiny" evaluations={[]} liveGame={game} laddered />,
    );

    expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent(
      'Step 500 is playing ladder game 13 of 100, against Stockfish 1500',
    );

    rerender(<LadderView run="tiny" evaluations={[]} liveGame={liveGame()} laddered />);

    expect(screen.queryByRole('heading', { level: 4 })).toBeNull();
  });

  it('is left out for a run that is not laddered and has no ladders', () => {
    serve();
    const { container } = render(
      <LadderView run="tiny" evaluations={[]} liveGame={liveGame()} laddered={false} />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
