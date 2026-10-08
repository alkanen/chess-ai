import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ModelDescription, Thoughts } from './api';
import { GameView } from './GameView';
import { FakeWebSocket } from './test/fakeWebSocket';
import { foolsMateMoves, foolsMateStart } from './test/foolsMate';

const MODEL: ModelDescription = {
  run: 'tiny',
  checkpoint: 4,
  rating: 1600,
  strategy: 'argmax',
  temperature: null,
} as unknown as ModelDescription;

/** What Black's model thinks of the position after 1. f3. */
const THOUGHTS: Thoughts = {
  candidates: [
    { uci: 'e7e5', san: 'e5', probability: 0.55 },
    { uci: 'd7d5', san: 'd5', probability: 0.3 },
  ],
  wdl: { win: 0.4, draw: 0.35, loss: 0.25 },
};

const [f3, e5, g4] = foolsMateMoves;

/** Fool's mate between a person playing White and a model playing Black, as White sees it. */
function following() {
  render(<GameView link="a-link" />);
  const socket = FakeWebSocket.latest;
  socket.open();
  socket.deliver({
    ...foolsMateStart,
    access: 'white',
    game: {
      ...foolsMateStart.game,
      white: { name: 'Human', accepts_moves: true, model: null, stockfish: null },
      black: { name: 'tiny step 4', accepts_moves: false, model: MODEL, stockfish: null },
    },
  });
  return socket;
}

function arrows(): string[] {
  return Array.from(document.querySelectorAll('.thought-arrow')).map(
    (arrow) => `${arrow.getAttribute('data-from')}${arrow.getAttribute('data-to')}`,
  );
}

describe('the thoughts overlay', () => {
  beforeEach(() => {
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
    window.localStorage.clear();
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('is the same while the model waits out the delay as once its move is played', () => {
    const socket = following();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Show what the model is thinking' }));
    socket.deliver(f3);

    socket.deliver({
      type: 'considering',
      considering: { ply: 1, side: 'black', thoughts: THOUGHTS },
    });
    const considering = arrows();
    const bar = screen.getByRole('img', { name: /^Evaluation:/ }).getAttribute('aria-label');
    expect(considering).toEqual(['d7d5', 'e7e5']);
    expect(screen.getByText('Black is considering…')).toBeInTheDocument();
    // From White's side, so Black's chance of winning is the model's own.
    expect(bar).toBe('Evaluation: White wins 25%, draw 35%, Black wins 40%');

    socket.deliver({ ...e5, move: { ...e5.move, thoughts: THOUGHTS } });

    expect(arrows()).toEqual(considering);
    expect(screen.getByRole('img', { name: /^Evaluation:/ })).toHaveAttribute('aria-label', bar);
    expect(screen.getByText('Black before playing e5')).toBeInTheDocument();
    expect(document.querySelector('.thought-arrow.played')).toHaveAttribute('data-to', 'e5');

    // About a position two moves back once White has replied, so it is put away.
    socket.deliver(g4);
    expect(arrows()).toEqual([]);
    expect(screen.getByText(/Shown when the model has just moved/)).toBeInTheDocument();
  });

  it('shows nothing until the viewer asks for it', () => {
    const socket = following();
    socket.deliver(f3);
    socket.deliver({ ...e5, move: { ...e5.move, thoughts: THOUGHTS } });

    expect(arrows()).toEqual([]);
    expect(screen.queryByRole('img', { name: /^Evaluation:/ })).not.toBeInTheDocument();
  });

  it('is not offered in a game no model plays', () => {
    render(<GameView link="a-link" />);
    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(foolsMateStart);

    expect(
      screen.queryByRole('checkbox', { name: 'Show what the model is thinking' }),
    ).not.toBeInTheDocument();
  });
});
