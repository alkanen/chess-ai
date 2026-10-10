import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { EvaluationEntry, SampleGamesResult } from './api';
import { describeEnding, describeOpponent, sampleGameHash, sampleGameInHash } from './sampleGames';
import { SampleGamesView } from './SampleGamesView';
import { lastMoveSquares } from './test/boardQueries';
import {
  checkpointPlayer,
  liveGame,
  sampleGamesResult,
  STOCKFISH_PLAYER,
} from './test/sampleGames';

function entry(step: number, updated = '2026-10-10T12:00:00Z'): EvaluationEntry {
  return { step, suite: 'sample-games', updated };
}

/** The step a request asked for the sample games of. */
function stepOf(input: RequestInfo | URL): number {
  const match = /evaluations\/(\d+)\/sample-games$/.exec(String(input));
  if (match === null) {
    throw new Error(`unexpected request ${String(input)}`);
  }
  return Number(match[1]);
}

/** Answers every request for a step's games with what `answer` gives for that step. */
function serve(answer: (step: number) => SampleGamesResult | Promise<SampleGamesResult>) {
  const fetch = vi.fn(async (input: RequestInfo | URL) =>
    Response.json(await answer(stepOf(input))),
  );
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

/** The rows of the game list, as their cells read. */
function rows(): string[][] {
  const table = screen.getByRole('table');
  return within(table)
    .getAllByRole('row')
    .slice(1)
    .map((row) => within(row).getAllByRole('cell').map((cell) => cell.textContent ?? ''));
}

describe('SampleGamesView', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    window.localStorage.clear();
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('lists the newest checkpoint’s games, each linked to the replay viewer', async () => {
    const fetch = serve((step) => sampleGamesResult(step));
    render(
      <SampleGamesView
        run="tiny run"
        evaluations={[entry(1_000), entry(500), { ...entry(1_500), suite: 'probe-positions' }]}
        liveGame={null}
        playing
      />,
    );

    expect(await screen.findByRole('table')).toBeVisible();
    expect(fetch).toHaveBeenCalledOnce();
    expect(String(fetch.mock.calls[0][0])).toMatch(
      /\/chess\/api\/runs\/tiny%20run\/evaluations\/1000\/sample-games$/,
    );
    expect(screen.getByText(/^Step /).textContent).toBe('Step 1,000 (2 of 2, newest)');
    expect(rows()).toEqual([
      ['Game 1', 'Itself', 'Italian Game', 'White wins by checkmate (1–0)', '21'],
      ['Game 2', 'Itself', 'Sicilian Defence', 'Draw by threefold repetition (½–½)', '30'],
      ['Game 3', 'Stockfish, checkpoint as Black', 'Italian Game', 'unfinished', '6'],
    ]);
    expect(screen.getByRole('link', { name: 'Game 2' })).toHaveAttribute(
      'href',
      '#replay/run/tiny%20run/1000/2',
    );
    expect(
      screen.getByText(/both sides rated 2000; openings from curated v1; Stockfish at Elo 1350/),
    ).toBeInTheDocument();
  });

  it('steps back to an earlier checkpoint’s games', async () => {
    const fetch = serve((step) => sampleGamesResult(step));
    render(
      <SampleGamesView
        run="tiny"
        evaluations={[entry(500), entry(1_000)]}
        liveGame={null}
        playing
      />,
    );
    await screen.findByRole('table');

    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));

    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'Game 1' })).toHaveAttribute(
        'href',
        '#replay/run/tiny/500/1',
      ),
    );
    expect(screen.getByText(/^Step /).textContent).toBe('Step 500 (1 of 2)');
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('asks again for a checkpoint whose games were played again', async () => {
    let plays = 0;
    const fetch = serve((step) => {
      plays += 1;
      return sampleGamesResult(step, { openings: `curated v${plays}` });
    });
    const { rerender } = render(
      <SampleGamesView run="tiny" evaluations={[entry(500)]} liveGame={null} playing />,
    );
    expect(await screen.findByText(/curated v1/)).toBeInTheDocument();

    rerender(
      <SampleGamesView
        run="tiny"
        evaluations={[entry(500, '2026-10-10T13:00:00Z')]}
        liveGame={null}
        playing
      />,
    );

    expect(await screen.findByText(/curated v2/)).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('keeps asking for games that cannot be fetched, and says so meanwhile', async () => {
    vi.useFakeTimers();
    try {
      let refusals = 1;
      const fetch = vi.fn(async (input: RequestInfo | URL) => {
        if (refusals > 0) {
          refusals -= 1;
          return Response.json({ detail: 'the server is restarting' }, { status: 503 });
        }
        return Response.json(sampleGamesResult(stepOf(input)));
      });
      vi.stubGlobal('fetch', fetch);
      render(<SampleGamesView run="tiny" evaluations={[entry(500)]} liveGame={null} playing />);

      await act(() => vi.advanceTimersByTimeAsync(200));
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Cannot load step 500 (the server is restarting); trying again.',
      );

      await act(() => vi.advanceTimersByTimeAsync(1_000));
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
      expect(screen.getByRole('table')).toBeInTheDocument();
      expect(fetch).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('says how to get games when no checkpoint has played any', () => {
    render(<SampleGamesView run="tiny" evaluations={[]} liveGame={null} playing />);

    expect(screen.getByRole('heading', { name: 'Sample games' })).toBeInTheDocument();
    expect(screen.getByText(/No checkpoint has finished its sample games yet/)).toBeInTheDocument();
  });

  it('leaves itself out for a run that plays none and has none', () => {
    render(<SampleGamesView run="tiny" evaluations={[]} liveGame={null} playing={false} />);

    expect(screen.queryByRole('heading', { name: 'Sample games' })).not.toBeInTheDocument();
  });

  describe('the game being played', () => {
    it('shows the board as it stands, who is playing and the moves so far', () => {
      const { container } = render(
        <SampleGamesView run="tiny" evaluations={[]} liveGame={liveGame()} playing />,
      );

      const live = screen.getByRole('region', { name: /Live/ });
      expect(within(live).getByRole('heading', { level: 4 })).toHaveTextContent(
        'Live Step 500 is playing game 1 of 2, against itself',
      );
      expect(within(live).getByRole('status')).toHaveTextContent('White to move');
      expect(lastMoveSquares(container)).toEqual(['e5', 'e7']);
      expect(within(live).getAllByText('tiny@500')).toHaveLength(2);
      expect(within(live).getByText('e5')).toBeInTheDocument();
      // Against itself, from White's side.
      expect(within(live).getByRole('button', { name: 'Flip to Black' })).toBeInTheDocument();
      // The games played so far are still there below it, once there are any.
      expect(
        screen.getByText(/No checkpoint has finished its sample games yet/),
      ).toBeInTheDocument();
    });

    it('shows what the checkpoint thought of its last move when asked', () => {
      const { container } = render(
        <SampleGamesView run="tiny" evaluations={[]} liveGame={liveGame()} playing />,
      );
      expect(container.querySelectorAll('.thought-arrow')).toHaveLength(0);

      fireEvent.click(screen.getByRole('checkbox', { name: 'Show what the model is thinking' }));

      expect(screen.getByText('Black before playing e5')).toBeInTheDocument();
      expect(container.querySelectorAll('.thought-arrow')).toHaveLength(2);
      expect(screen.getByRole('img', { name: /^Evaluation: White wins 20%/ })).toBeInTheDocument();
    });

    it('faces the checkpoint’s side against Stockfish, and keeps a flip from game to game', () => {
      const black = liveGame({
        opponent: 'stockfish',
        index: 2,
        games: 3,
      });
      black.game = { ...black.game, white: STOCKFISH_PLAYER, black: checkpointPlayer(500) };
      const { rerender } = render(
        <SampleGamesView run="tiny" evaluations={[]} liveGame={black} playing />,
      );
      expect(screen.getByRole('heading', { level: 4 })).toHaveTextContent(
        'game 3 of 3, against Stockfish',
      );
      expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: 'Flip to White' }));
      rerender(<SampleGamesView run="tiny" evaluations={[]} liveGame={liveGame()} playing />);

      // The checkpoint plays White now, and the board stays flipped away from it.
      expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();
    });

    it('goes when the games are over', () => {
      const { rerender } = render(
        <SampleGamesView run="tiny" evaluations={[]} liveGame={liveGame()} playing />,
      );

      rerender(<SampleGamesView run="tiny" evaluations={[]} liveGame={null} playing />);

      expect(screen.queryByRole('region', { name: /Live/ })).not.toBeInTheDocument();
    });
  });
});

describe('sample game addresses', () => {
  it('name a game from 1, round-trip a run name, and refuse a game 0', () => {
    const ref = { run: 'tiny run/2', step: 1_000, index: 0 };
    expect(sampleGameHash(ref)).toBe('#replay/run/tiny%20run%2F2/1000/1');
    expect(sampleGameInHash(sampleGameHash(ref))).toEqual(ref);
    expect(sampleGameInHash('#replay/run/tiny/1000/0')).toBeNull();
    expect(sampleGameInHash('#runs/tiny')).toBeNull();
  });
});

describe('describing a sample game', () => {
  it('says whom the checkpoint played, and with which pieces', () => {
    expect(describeOpponent({ opponent: 'self', model_color: 'both' })).toBe('Itself');
    expect(describeOpponent({ opponent: 'stockfish', model_color: 'white' })).toBe(
      'Stockfish, checkpoint as White',
    );
  });

  it('says how it ended, or that a game stopped by an error has no result', () => {
    expect(describeEnding({ result: '0-1', termination: 'resignation' })).toBe(
      'Black wins by resignation (0–1)',
    );
    expect(describeEnding({ result: '*', termination: 'error' })).toBe(
      'Game stopped: a player could not go on',
    );
  });
});
