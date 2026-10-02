import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CheckpointSummary, RunSummary } from './api';
import { NewGameForm } from './NewGameForm';

const HUMAN = { kind: 'human' };
const RANDOM = { kind: 'random' };

const RUNS: RunSummary[] = [
  {
    name: 'mlp-baseline',
    architecture: 'mlp',
    created: '2026-09-20T10:00:00Z',
    status: 'finished',
    step: 12000,
    steps: 12000,
    checkpoints: 2,
  },
  {
    name: 'mlp-wider',
    architecture: 'mlp',
    created: '2026-09-18T10:00:00Z',
    status: 'running',
    step: 400,
    steps: 8000,
    checkpoints: 1,
  },
];

const CHECKPOINTS: Record<string, CheckpointSummary[]> = {
  'mlp-baseline': [
    { step: 12000, created: '2026-09-20T12:00:00Z', metrics: {}, best: false, latest: true },
    { step: 6000, created: '2026-09-20T11:00:00Z', metrics: {}, best: true, latest: false },
  ],
  'mlp-wider': [
    { step: 400, created: '2026-09-18T11:00:00Z', metrics: {}, best: true, latest: true },
  ],
};

/** A server with `runs` on it, answering each request with a body of its own. */
function serving(runs: RunSummary[] = RUNS) {
  return vi.fn((url: string) => {
    if (url.endsWith('/api/runs')) {
      return Promise.resolve(Response.json(runs));
    }
    if (url.endsWith('/api/stockfish')) {
      return Promise.resolve(
        Response.json({ name: 'Stockfish 14.1', min_elo: 1350, max_elo: 2850 }),
      );
    }
    const asked = /\/api\/runs\/([^/]+)\/checkpoints$/.exec(url);
    if (asked !== null) {
      const run = decodeURIComponent(asked[1]);
      return Promise.resolve(Response.json({ run, checkpoints: CHECKPOINTS[run] ?? [] }));
    }
    return Promise.resolve(Response.json({}));
  });
}

/** Everything the form has posted so far, which is every game it has started. */
function posted(fetch: ReturnType<typeof vi.fn>): unknown[] {
  return (fetch.mock.calls as [string, RequestInit?][])
    .filter(([, init]) => init?.method === 'POST')
    .map(([, init]) => JSON.parse(init?.body as string));
}

/** Starts a game with the form as it stands and returns what was posted. */
async function start(fetch: ReturnType<typeof vi.fn>): Promise<unknown> {
  const before = posted(fetch).length;
  fireEvent.click(screen.getByRole('button', { name: 'Start' }));
  await vi.waitFor(() => expect(posted(fetch).length).toBe(before + 1));
  return posted(fetch)[before];
}

/** Gives `side` to a model and waits for the run it is offered to arrive. */
async function chooseModel(side: 'White' | 'Black'): Promise<void> {
  fireEvent.change(screen.getByLabelText(side), { target: { value: 'model' } });
  await screen.findByLabelText(`${side} run`);
  await vi.waitFor(() =>
    expect(screen.getByLabelText(`${side} run`)).toHaveValue('mlp-baseline'),
  );
}

describe('NewGameForm', () => {
  let fetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    fetch = serving();
    vi.stubGlobal('fetch', fetch);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('starts a game with the chosen delay under the path prefix', async () => {
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Delay between moves'), { target: { value: '2' } });
    const posted = await start(fetch);

    const [url, init] = fetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(new URL('/chess/api/game', window.location.href).href);
    expect(init.method).toBe('POST');
    expect(posted).toEqual({ white: HUMAN, black: RANDOM, move_delay: 2, fen: null });
    expect(await screen.findByRole('button', { name: 'Start' })).toBeEnabled();
  });

  it('offers you the white pieces against a random mover to begin with', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('White')).toHaveDisplayValue('Human');
    expect(screen.getByLabelText('Black')).toHaveDisplayValue('Random mover');
    expect(screen.getByLabelText('Delay between moves')).toHaveDisplayValue('0.5 s');
  });

  it('asks the server nothing about runs until a model is chosen', () => {
    render(<NewGameForm />);

    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    ['human', 'human'],
    ['human', 'random'],
    ['random', 'random'],
  ])('starts %s against %s', async (white, black) => {
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('White'), { target: { value: white } });
    fireEvent.change(screen.getByLabelText('Black'), { target: { value: black } });

    expect(await start(fetch)).toEqual({
      white: { kind: white },
      black: { kind: black },
      move_delay: 0.5,
      fen: null,
    });
  });

  it('paces nothing when both sides are played by hand, and says so', () => {
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'human' } });

    expect(screen.getByLabelText('Delay between moves')).toBeDisabled();
    expect(screen.getByText(/nothing to pace/i)).toBeInTheDocument();
  });

  it('paces the moves of a player that moves for itself', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('Delay between moves')).toBeEnabled();
    expect(screen.queryByText(/nothing to pace/i)).not.toBeInTheDocument();
  });

  it('keeps the chosen delay while both sides are played by hand', async () => {
    render(<NewGameForm />);
    fireEvent.change(screen.getByLabelText('Delay between moves'), { target: { value: '2' } });

    fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'human' } });
    fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'random' } });

    const delay = screen.getByLabelText('Delay between moves');
    expect(delay).toBeEnabled();
    expect(delay).toHaveDisplayValue('2 s');
    expect(await start(fetch)).toEqual({
      white: HUMAN,
      black: RANDOM,
      move_delay: 2,
      fen: null,
    });
  });

  it('reports a server error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));
    render(<NewGameForm />);

    fireEvent.click(screen.getByRole('button', { name: 'Start' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not start the game');
  });

  it('starts from a position typed into the FEN box', async () => {
    const fen = 'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2';
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Start from FEN'), { target: { value: ` ${fen} ` } });

    expect(await start(fetch)).toEqual({
      white: HUMAN,
      black: RANDOM,
      move_delay: 0.5,
      // Typed-in positions come with whatever was pasted around them.
      fen,
    });
  });

  it('says why the server would not start from the FEN', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(Response.json({ detail: 'White has no king' }, { status: 400 })),
    );
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Start from FEN'), { target: { value: '8/8/8' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not start the game: White has no king',
    );
  });

  it('keeps the FEN box empty for a game from the usual starting position', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('Start from FEN')).toHaveValue('');
  });

  describe('playing a checkpoint', () => {
    it('offers the runs there are, newest first, and plays the first of them', async () => {
      render(<NewGameForm />);

      await chooseModel('Black');

      expect(screen.getByLabelText('Black run')).toHaveDisplayValue(
        'mlp-baseline (2 checkpoints, 12000/12000 steps)',
      );
      expect(
        screen.getByRole('option', { name: 'mlp-wider (1 checkpoints, 400/8000 steps)' }),
      ).toBeInTheDocument();
      expect(await start(fetch)).toEqual({
        white: HUMAN,
        black: {
          kind: 'model',
          run: 'mlp-baseline',
          checkpoint: 'best',
          rating: null,
          strategy: 'argmax',
          temperature: 1,
        },
        move_delay: 0.5,
        fen: null,
      });
    });

    it('offers the best and the latest checkpoint as well as each one by step', async () => {
      render(<NewGameForm />);

      await chooseModel('White');

      const checkpoint = await screen.findByLabelText('White checkpoint');
      await vi.waitFor(() =>
        expect(screen.getByRole('option', { name: 'Step 12000 (latest)' })).toBeInTheDocument(),
      );
      expect(screen.getByRole('option', { name: 'Step 6000 (best)' })).toBeInTheDocument();
      expect(checkpoint).toHaveDisplayValue('Best');

      fireEvent.change(checkpoint, { target: { value: '6000' } });

      expect(await start(fetch)).toMatchObject({ white: { checkpoint: 6000 } });
    });

    it('asks for another run’s checkpoints and goes back to its best one', async () => {
      render(<NewGameForm />);
      await chooseModel('White');
      fireEvent.change(await screen.findByLabelText('White checkpoint'), {
        target: { value: '6000' },
      });

      fireEvent.change(screen.getByLabelText('White run'), { target: { value: 'mlp-wider' } });

      await vi.waitFor(() =>
        expect(screen.getByRole('option', { name: 'Step 400 (best, latest)' })).toBeInTheDocument(),
      );
      expect(screen.getByLabelText('White checkpoint')).toHaveDisplayValue('Best');
      expect(await start(fetch)).toMatchObject({
        white: { run: 'mlp-wider', checkpoint: 'best' },
      });
    });

    it('plays like the rating that was typed in', async () => {
      render(<NewGameForm />);
      await chooseModel('White');

      fireEvent.change(screen.getByLabelText('White rating'), { target: { value: '1600' } });

      expect(await start(fetch)).toMatchObject({ white: { rating: 1600 } });
    });

    it('claims no rating at all when the box is left empty', async () => {
      render(<NewGameForm />);
      await chooseModel('White');

      expect(screen.getByLabelText('White rating')).toHaveValue(null);
      expect(await start(fetch)).toMatchObject({ white: { rating: null } });
    });

    it('offers a temperature only when the model samples its move', async () => {
      render(<NewGameForm />);
      await chooseModel('White');

      expect(screen.queryByLabelText('White temperature')).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText('White plays'), { target: { value: 'sample' } });
      fireEvent.change(screen.getByLabelText('White temperature'), { target: { value: '1.5' } });

      expect(await start(fetch)).toMatchObject({
        white: { strategy: 'sample', temperature: 1.5 },
      });
    });

    it('sets the two sides up as different checkpoints of the same run', async () => {
      render(<NewGameForm />);

      await chooseModel('White');
      await chooseModel('Black');
      fireEvent.change(await screen.findByLabelText('Black checkpoint'), {
        target: { value: '6000' },
      });

      expect(await start(fetch)).toMatchObject({
        white: { kind: 'model', checkpoint: 'best' },
        black: { kind: 'model', checkpoint: 6000 },
      });
    });

    it('keeps the run and rating while the side is given back to a person', async () => {
      render(<NewGameForm />);
      await chooseModel('White');
      fireEvent.change(screen.getByLabelText('White rating'), { target: { value: '1200' } });

      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'human' } });
      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'model' } });

      expect(screen.getByLabelText('White rating')).toHaveValue(1200);
      expect(await start(fetch)).toMatchObject({ white: { run: 'mlp-baseline', rating: 1200 } });
    });

    it('lets any whole rating through the browser’s own checks', async () => {
      render(<NewGameForm />);
      await chooseModel('White');

      fireEvent.change(screen.getByLabelText('White rating'), { target: { value: '1337' } });

      expect(screen.getByLabelText('White rating')).toBeValid();
    });

    it('keeps the runs that arrive after the side was given back to a person', async () => {
      let answer: (response: Response) => void = () => {};
      const answering = serving();
      fetch.mockImplementation((url: string) =>
        url.endsWith('/api/runs')
          ? new Promise<Response>((resolve) => (answer = resolve))
          : answering(url),
      );
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'model' } });
      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'human' } });
      await act(async () => answer(Response.json(RUNS)));
      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'model' } });

      await vi.waitFor(() =>
        expect(screen.getByLabelText('White run')).toHaveValue('mlp-baseline'),
      );
    });

    it('says when there is nothing here to play against, and will not start', async () => {
      vi.stubGlobal('fetch', serving([]));
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'model' } });

      expect(await screen.findByText(/No training runs have been kept here yet/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Start' })).toBeDisabled();
    });

    it('says why the runs could not be listed', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'model' } });

      expect(
        await screen.findByText(/Could not list the training runs/),
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Start' })).toBeDisabled();
    });
  });

  describe('playing Stockfish', () => {
    it('plays either side at the Elo typed in, for the time a move chosen', async () => {
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'stockfish' } });
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });
      fireEvent.change(screen.getByLabelText('White Elo'), { target: { value: '1800' } });
      fireEvent.change(screen.getByLabelText('Black time a move'), { target: { value: '0.25' } });

      expect(await start(fetch)).toMatchObject({
        white: { kind: 'stockfish', elo: 1800, move_time: 1 },
        black: { kind: 'stockfish', elo: 1500, move_time: 0.25 },
      });
    });

    it('asks the server which Stockfish it has, once, and nothing about runs', async () => {
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });
      fireEvent.change(screen.getByLabelText('White'), { target: { value: 'stockfish' } });

      await screen.findAllByText(/Stockfish 14.1 plays/);
      const asked = (fetch.mock.calls as [string][]).map(([url]) => new URL(url).pathname);
      expect(asked).toEqual(['/chess/api/stockfish']);
    });

    it('will not start without an Elo to play at', () => {
      render(<NewGameForm />);
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });

      fireEvent.change(screen.getByLabelText('Black Elo'), { target: { value: '' } });

      expect(screen.getByRole('button', { name: 'Start' })).toBeDisabled();
    });

    it('says which strengths the Stockfish here plays, moving others into range', async () => {
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });

      expect(
        await screen.findByText(/Stockfish 14.1 plays from 1350 to 2850/),
      ).toBeInTheDocument();
    });

    it('asks for an Elo out of that range all the same, to be played at the nearest', async () => {
      render(<NewGameForm />);
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });
      await screen.findByText(/Stockfish 14.1 plays/);

      fireEvent.change(screen.getByLabelText('Black Elo'), { target: { value: '800' } });

      expect(await start(fetch)).toMatchObject({ black: { kind: 'stockfish', elo: 800 } });
    });

    it('lets any whole Elo through the browser’s own checks, not only multiples of 50', () => {
      render(<NewGameForm />);
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });

      for (const elo of ['1320', '1337', '3190']) {
        fireEvent.change(screen.getByLabelText('Black Elo'), { target: { value: elo } });
        // A step mismatch here is a form the browser will not submit, with no game started.
        expect(screen.getByLabelText('Black Elo')).toBeValid();
      }
    });

    it('keeps the answer that arrives after the side was given to someone else', async () => {
      let answer: (response: Response) => void = () => {};
      fetch.mockImplementation(
        (url: string) =>
          new Promise<Response>((resolve) => {
            answer = resolve;
            expect(url).toMatch(/\/api\/stockfish$/);
          }),
      );
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });
      // Changed back before the server has started the engine to answer.
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'random' } });
      await act(async () => {
        answer(Response.json({ name: 'Stockfish 14.1', min_elo: 1350, max_elo: 2850 }));
      });
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });

      expect(await screen.findByText(/Stockfish 14.1 plays from 1350/)).toBeInTheDocument();
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('still says there is no Stockfish when the answer came after the side was changed', async () => {
      let answer: (response: Response) => void = () => {};
      fetch.mockImplementation(
        () => new Promise<Response>((resolve) => (answer = resolve)),
      );
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'random' } });
      await act(async () => {
        answer(Response.json({ detail: 'Stockfish was not found' }, { status: 500 }));
      });
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });

      expect(await screen.findByText(/There is no Stockfish to play/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Start' })).toBeDisabled();
    });

    it('says why there is no Stockfish to play, and will not start', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue(
          Response.json({ detail: "Stockfish was not found at 'stockfish'" }, { status: 500 }),
        ),
      );
      render(<NewGameForm />);

      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'stockfish' } });

      expect(
        await screen.findByText(/There is no Stockfish to play: .*not found/),
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Start' })).toBeDisabled();
      // Given to someone else, the side no longer needs it, and the game can start.
      fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'random' } });
      expect(screen.getByRole('button', { name: 'Start' })).toBeEnabled();
    });
  });
});
