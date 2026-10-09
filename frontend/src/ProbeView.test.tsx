import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { EvaluationEntry, PositionSnapshot, ProbeOutcome, ProbeResult } from './api';
import { ProbeView } from './ProbeView';

const POSITION: PositionSnapshot = {
  fen: 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
  turn: 'white',
  pieces: {
    e1: { color: 'white', type: 'king' },
    h5: { color: 'white', type: 'queen' },
    c4: { color: 'white', type: 'bishop' },
    e8: { color: 'black', type: 'king' },
    f7: { color: 'black', type: 'pawn' },
  },
  last_move: { from_square: 'g8', to_square: 'f6' },
  check_square: null,
  legal_moves: {},
  game_over: null,
};

function outcome(changes: Partial<ProbeOutcome> = {}): ProbeOutcome {
  return {
    id: 'scholars-mate',
    name: "Scholar's mate",
    category: 'tactic',
    comment: 'Mate in one.',
    start: null,
    line: '1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6',
    fen: POSITION.fen,
    best: ['h5f7'],
    top: [
      { uci: 'h5f7', san: 'Qxf7#', probability: 0.7 },
      { uci: 'h5e5', san: 'Qxe5+', probability: 0.2 },
    ],
    wdl: { win: 0.8, draw: 0.15, loss: 0.05 },
    illegal_mass: 0.01,
    best_probability: 0.7,
    snapshot: POSITION,
    ...changes,
  };
}

const OPENING = outcome({
  id: 'start',
  name: 'Starting position',
  category: 'opening',
  comment: null,
  line: '',
  best: [],
  top: [
    { uci: 'e2e4', san: 'e4', probability: 0.4 },
    { uci: 'd2d4', san: 'd4', probability: 0.35 },
  ],
  best_probability: null,
  wdl: { win: 0.3, draw: 0.5, loss: 0.2 },
});

function result(step: number, changes: Partial<ProbeResult> = {}): ProbeResult {
  return {
    model: { run: 'tiny', checkpoint: step, rating: 2000 },
    finished: '2026-10-09T12:00:00Z',
    probe_set: 'standard',
    probe_set_version: 1,
    positions: [OPENING, outcome()],
    ...changes,
  };
}

const V1 = { name: 'standard', version: 1 };
const V2 = { name: 'standard', version: 2 };

function entry(step: number, updated = '2026-10-09T12:00:00Z'): EvaluationEntry {
  return { step, suite: 'probe-positions', updated };
}

/** The step a request asked for the probe result of. */
function stepOf(input: RequestInfo | URL): number {
  const match = /evaluations\/(\d+)\/probe-positions$/.exec(String(input));
  if (match === null) {
    throw new Error(`unexpected request ${String(input)}`);
  }
  return Number(match[1]);
}

/** Answers every request for a step's result with what `answer` gives for that step. */
function serve(answer: (step: number) => ProbeResult | Promise<ProbeResult>) {
  const fetch = vi.fn(async (input: RequestInfo | URL) =>
    Response.json(await answer(stepOf(input))),
  );
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

function shownStep(): string | null {
  return screen.getByText(/^Step /).textContent;
}

describe('ProbeView', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('shows the newest checkpoint’s moves on each probe, as arrows and in words', async () => {
    const fetch = serve((step) => result(step));
    render(<ProbeView currentSet={V1} run="tiny run" evaluations={[entry(4), entry(2)]} probed />);

    const mate = await screen.findByRole('article', { name: "Scholar's mate" });
    expect(fetch).toHaveBeenCalledOnce();
    expect(String(fetch.mock.calls[0][0])).toMatch(
      /\/chess\/api\/runs\/tiny%20run\/evaluations\/4\/probe-positions$/,
    );
    expect(shownStep()).toBe('Step 4 (2 of 2, newest)');
    expect(within(mate).getByText('Solved')).toBeInTheDocument();
    expect(within(mate).getByText('1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6')).toBeInTheDocument();
    expect(within(mate).getByText('White to move')).toBeInTheDocument();
    expect(within(mate).getByText('Mate in one.')).toBeInTheDocument();
    expect(within(mate).getByText('The solution: 70%')).toBeInTheDocument();
    expect(within(mate).getByText('White wins 80%, draw 15%, Black wins 5%')).toBeInTheDocument();
    const moves = within(mate).getByRole('list', { name: "Most likely moves in Scholar's mate" });
    expect([...moves.querySelectorAll('li')].map((item) => item.textContent)).toEqual([
      'Qxf7# (solves it)70%',
      'Qxe5+20%',
    ]);
    // Least likely first, so the likeliest is drawn on top; the solution in its own colour.
    const arrows = [...mate.querySelectorAll('.thought-arrow')];
    expect(arrows.map((arrow) => arrow.getAttribute('data-to'))).toEqual(['e5', 'f7']);
    expect(arrows[1]).toHaveClass('played');
    expect(arrows[0]).not.toHaveClass('played');
  });

  it('says how each position was reached', async () => {
    const setUp = { start: '8/8/8/8/8/2k5/8/K1Q5 w - - 0 1' };
    serve((step) =>
      result(step, {
        positions: [
          OPENING,
          outcome({ id: 'a', name: 'Set up', ...setUp, line: '' }),
          outcome({ id: 'b', name: 'Set up, then played', ...setUp, line: '1. Kb1 Kb3' }),
        ],
      }),
    );
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />);

    const line = async (name: string) =>
      (await screen.findByRole('article', { name })).querySelector('.probe-line')?.textContent;
    expect(await line('Starting position')).toBe('Before the first move');
    expect(await line('Set up')).toBe('A set-up position');
    expect(await line('Set up, then played')).toBe('From a set-up position: 1. Kb1 Kb3');
  });

  it('groups the probes by category, and counts what was solved', async () => {
    serve((step) => result(step));
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />);

    const openings = await screen.findByRole('region', { name: 'Openings' });
    const tactics = screen.getByRole('region', { name: 'Tactics' });
    expect(within(openings).getByRole('article', { name: 'Starting position' })).toBeVisible();
    expect(within(tactics).getByRole('article', { name: "Scholar's mate" })).toBeVisible();
    // A position without a solution is neither solved nor failed.
    expect(within(openings).queryByText(/solved/i)).not.toBeInTheDocument();
    expect(within(tactics).getByRole('heading', { name: /^Tactics/ })).toHaveTextContent(
      'Tactics solved 1 of 1',
    );
    expect(screen.queryByRole('region', { name: 'Endgames' })).not.toBeInTheDocument();
    expect(
      screen.getByText(/Solved 1 of the 1 probes that have a solution, on standard v1/),
    ).toHaveTextContent('both sides rated 2000.');
  });

  it('shows the black side’s moves from the black side, and the odds from White’s', async () => {
    const black = outcome({
      snapshot: { ...POSITION, turn: 'black' },
      wdl: { win: 0.6, draw: 0.3, loss: 0.1 },
      top: [{ uci: 'e8e7', san: 'Ke7', probability: 1 }],
    });
    serve((step) => result(step, { positions: [black] }));
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />);

    const card = await screen.findByRole('article', { name: "Scholar's mate" });
    expect(within(card).getByText('Black to move')).toBeInTheDocument();
    expect(within(card).getByText('Not solved')).toBeInTheDocument();
    expect(within(card).getByText('White wins 10%, draw 30%, Black wins 60%')).toBeInTheDocument();
  });

  it('steps back through the checkpoints, and leaves the newest to follow', async () => {
    const fetch = serve((step) => result(step));
    const { rerender } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2), entry(4), entry(6)]} probed />,
    );
    await screen.findByText(/on standard v1/);

    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));
    expect(shownStep()).toBe('Step 4 (2 of 3)');
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(stepOf(fetch.mock.calls[1][0])).toBe(4);

    // A newer result arriving leaves the one chosen where it is.
    rerender(
      <ProbeView
        currentSet={V1}
        run="tiny"
        evaluations={[entry(2), entry(4), entry(6), entry(8)]}
        probed
      />,
    );
    expect(shownStep()).toBe('Step 4 (2 of 4)');

    fireEvent.change(screen.getByRole('slider', { name: 'Checkpoint' }), {
      target: { value: '3' },
    });
    expect(shownStep()).toBe('Step 8 (4 of 4, newest)');
    rerender(
      <ProbeView
        currentSet={V1}
        run="tiny"
        evaluations={[entry(2), entry(4), entry(6), entry(8), entry(10)]}
        probed
      />,
    );
    expect(shownStep()).toBe('Step 10 (5 of 5, newest)');
    await waitFor(() => expect(stepOf(fetch.mock.lastCall![0])).toBe(10));
  });

  it('follows the newest checkpoint as its results arrive', async () => {
    serve((step) => result(step));
    const { rerender } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />,
    );
    await screen.findByText(/on standard v1/);
    expect(screen.queryByRole('slider')).toHaveAttribute('max', '0');

    rerender(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2), entry(4)]} probed />);

    await waitFor(() => expect(screen.queryByText(/while step/)).not.toBeInTheDocument());
    expect(shownStep()).toBe('Step 4 (2 of 2, newest)');
  });

  it('asks for a result again when it is written again', async () => {
    let wdl = { win: 0.8, draw: 0.15, loss: 0.05 };
    const fetch = serve((step) => result(step, { positions: [outcome({ wdl })] }));
    const { rerender } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />,
    );
    await screen.findByText('White wins 80%, draw 15%, Black wins 5%');

    wdl = { win: 0.1, draw: 0.2, loss: 0.7 };
    rerender(
      <ProbeView
        currentSet={V1}
        run="tiny"
        evaluations={[entry(2, '2026-10-09T13:00:00Z')]}
        probed
      />,
    );

    await screen.findByText('White wins 10%, draw 20%, Black wins 70%');
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('does not ask for a result it has again when it goes back to it', async () => {
    const fetch = serve((step) => result(step));
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2), entry(4)]} probed />);
    await screen.findByText(/on standard v1/);
    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText(/while step/)).not.toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: 'Later checkpoint' }));

    expect(screen.queryByText(/while step/)).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('asks only for where the scrubber comes to rest when it is dragged along', async () => {
    const fetch = serve((step) => result(step));
    const steps = [2, 4, 6, 8, 10].map((step) => entry(step));
    render(<ProbeView currentSet={V1} run="tiny" evaluations={steps} probed />);
    await screen.findByText(/on standard v1/);

    const slider = screen.getByRole('slider', { name: 'Checkpoint' });
    for (const value of ['3', '2', '1', '0']) {
      fireEvent.change(slider, { target: { value } });
      // As long as a drag stays on a mark, which is not long.
      await act(() => new Promise((resolve) => setTimeout(resolve, 40)));
    }

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(fetch.mock.calls.map(([input]) => stepOf(input))).toEqual([10, 2]);
  });

  it('never shows a late answer about a checkpoint it has left', async () => {
    let answerFour: ((value: ProbeResult) => void) | null = null;
    serve((step) =>
      step === 4
        ? new Promise<ProbeResult>((resolve) => {
            answerFour = resolve;
          })
        : result(step),
    );
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2), entry(4)]} probed />);
    await waitFor(() => expect(answerFour).not.toBeNull());

    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));
    await screen.findByText(/on standard v1/);
    await act(async () => answerFour!(result(4, { probe_set_version: 9 })));

    expect(shownStep()).toBe('Step 2 (1 of 2)');
    expect(screen.getByText(/on standard v1/)).toBeInTheDocument();
    expect(screen.queryByText(/v9/)).not.toBeInTheDocument();
  });

  it('tries again when a result cannot be fetched, keeping the boards it has meanwhile', async () => {
    let failures = 1;
    const fetch = serve((step) => {
      if (step === 4 && failures > 0) {
        failures -= 1;
        throw new TypeError('Failed to fetch');
      }
      return result(step);
    });
    const { rerender } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />,
    );
    await screen.findByText(/on standard v1/);

    // The server restarted, say, and the run channel sends the same results again.
    rerender(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2), entry(4)]} probed />);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Cannot load step 4 (Failed to fetch); trying again',
    );
    expect(screen.getByRole('article', { name: "Scholar's mate" })).toBeVisible();
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument(), {
      timeout: 3000,
    });
    expect(screen.queryByText(/while step/)).not.toBeInTheDocument();
    expect(fetch.mock.calls.map(([input]) => stepOf(input))).toEqual([2, 4, 4]);
  });

  it('stops trying for a checkpoint it has left', async () => {
    const fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    vi.stubGlobal('fetch', fetch);
    const { unmount } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />,
    );
    await screen.findByRole('alert');

    unmount();
    await new Promise((resolve) => setTimeout(resolve, 1500));

    expect(fetch).toHaveBeenCalledOnce();
  });

  it('follows the newest again once the checkpoint chosen has no result any more', async () => {
    serve((step) => result(step));
    const { rerender } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2), entry(4)]} probed />,
    );
    await screen.findByText(/on standard v1/);
    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));
    expect(shownStep()).toBe('Step 2 (1 of 2)');

    // The run is started again under the same name, and its results start afresh.
    rerender(
      <ProbeView
        currentSet={V1}
        run="tiny"
        evaluations={[entry(1, '2026-10-10T09:00:00Z')]}
        probed
      />,
    );
    expect(shownStep()).toBe('Step 1 (1 of 1, newest)');
    const again = [1, 2, 3].map((step) => entry(step, '2026-10-10T09:00:00Z'));
    rerender(<ProbeView currentSet={V1} run="tiny" evaluations={again} probed />);

    expect(shownStep()).toBe('Step 3 (3 of 3, newest)');
  });

  it('marks a result measured on an older version of the set the evaluator uses', async () => {
    serve((step) => result(step, { probe_set_version: step === 2 ? 1 : 2 }));
    render(<ProbeView currentSet={V2} run="tiny" evaluations={[entry(2), entry(4)]} probed />);
    await screen.findByText(/on standard v2/);
    expect(screen.queryByRole('note')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Earlier checkpoint' }));

    expect(await screen.findByRole('note')).toHaveTextContent(
      'Measured on standard v1, an older version than the standard v2 the evaluator probes ' +
        'with: this checkpoint has been pruned since, or has not been probed again yet.',
    );
  });

  it('never calls a result on a newer version than the evaluator’s older', async () => {
    // Measured on demand with a set the evaluator has not been started again with.
    serve((step) => result(step, { probe_set_version: 2 }));
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />);

    await screen.findByText(/on standard v2/);
    expect(screen.queryByRole('note')).not.toBeInTheDocument();
  });

  it('marks a result measured on another set than the evaluator’s', async () => {
    serve((step) => result(step, { probe_set: 'mine', probe_set_version: 3 }));
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />);

    expect(await screen.findByRole('note')).toHaveTextContent(
      'Measured on mine v3, not on standard v1, which the evaluator probes with.',
    );
  });

  it('marks a result it already has once the evaluator probes with a newer set', async () => {
    const fetch = serve((step) => result(step));
    const { rerender } = render(
      <ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />,
    );
    await screen.findByText(/on standard v1/);
    expect(screen.queryByRole('note')).not.toBeInTheDocument();

    rerender(<ProbeView currentSet={V2} run="tiny" evaluations={[entry(2)]} probed />);

    expect(screen.getByRole('note')).toHaveTextContent('older version than the standard v2');
    expect(fetch).toHaveBeenCalledOnce();
  });

  it('marks nothing when no evaluator has said which set it uses', async () => {
    serve((step) => result(step));
    render(<ProbeView currentSet={null} run="tiny" evaluations={[entry(2)]} probed />);

    await screen.findByText(/on standard v1/);
    expect(screen.queryByRole('note')).not.toBeInTheDocument();
  });

  it('says why a result cannot be shown', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          Response.json({ detail: 'cannot read probe-positions.json' }, { status: 500 }),
        ),
    );
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[entry(2)]} probed />);

    expect(await screen.findByRole('alert')).toHaveTextContent('cannot read probe-positions.json');
  });

  it('leaves out the results of other suites', async () => {
    const fetch = serve((step) => result(step));
    render(
      <ProbeView
        currentSet={V1}
        run="tiny"
        evaluations={[entry(2), { step: 4, suite: 'ladder', updated: '2026-10-09T12:00:00Z' }]}
        probed
      />,
    );

    await screen.findByText(/on standard v1/);
    expect(shownStep()).toBe('Step 2 (1 of 1, newest)');
    expect(stepOf(fetch.mock.calls[0][0])).toBe(2);
  });

  it('says how checkpoints get probed before any has been', () => {
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[]} probed />);

    expect(screen.getByText(/No checkpoint has been probed yet/)).toHaveTextContent(
      'chess-ai evaluator probes each one as it is saved.',
    );
  });

  it('is left out for a run whose checkpoints are not probed', () => {
    render(<ProbeView currentSet={V1} run="tiny" evaluations={[]} probed={false} />);

    expect(screen.queryByRole('heading', { name: 'Probe positions' })).not.toBeInTheDocument();
  });
});
