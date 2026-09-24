import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { GameState, PlayerKind } from './api';
import { GameControls } from './GameControls';
import startPosition from './test/fixtures/start-position.json';

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true },
  random: { name: 'Random mover', accepts_moves: false },
} satisfies Record<PlayerKind, { name: string; accepts_moves: boolean }>;

const POSITION = startPosition as GameState['position'];

function gameOf(white: PlayerKind, black: PlayerKind, moves = 0): GameState {
  return {
    id: 'a-game',
    white: PLAYERS[white],
    black: PLAYERS[black],
    start_fen: POSITION.fen,
    // Only how many there are decides what can be taken back, not what they were.
    moves: Array.from({ length: moves }, () => ({ uci: 'e2e4', san: 'e4', thoughts: null })),
    position: POSITION,
  };
}

function renderControls(overrides: Partial<Parameters<typeof GameControls>[0]> = {}) {
  const props = {
    orientation: 'white' as const,
    onFlip: vi.fn(),
    game: gameOf('human', 'random', 1),
    disabled: false,
    onResign: vi.fn(),
    onAbort: vi.fn(),
    onTakeBack: vi.fn(),
    ...overrides,
  };
  render(<GameControls {...props} />);
  return props;
}

/** Every control on offer, in the order it is shown. */
function buttons(): string[] {
  return screen.getAllByRole('button').map((button) => button.textContent ?? '');
}

describe('GameControls', () => {
  it('offers to turn the board round to the other side', () => {
    const { onFlip } = renderControls({ orientation: 'white' });

    fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

    expect(onFlip).toHaveBeenCalledOnce();
  });

  it('names the side the board would turn to', () => {
    renderControls({ orientation: 'black' });

    expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();
  });

  it('resigns the one side the viewer plays', () => {
    const { onResign } = renderControls({ game: gameOf('human', 'random') });

    fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('white');
  });

  it('resigns for Black when Black is the side the viewer plays', () => {
    const { onResign } = renderControls({ game: gameOf('random', 'human') });

    fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('black');
  });

  it('asks which side resigns when the viewer plays both', () => {
    const { onResign } = renderControls({ game: gameOf('human', 'human', 1) });

    expect(buttons()).toEqual([
      'Flip to Black',
      'Take back',
      'White resigns',
      'Black resigns',
      'Abort',
    ]);

    fireEvent.click(screen.getByRole('button', { name: 'Black resigns' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('black');
  });

  it('offers no resignation in a game the viewer only watches, but still an abort', () => {
    renderControls({ game: gameOf('random', 'random', 1) });

    // Nobody plays a game of two players that move for themselves, so nobody takes a
    // move back in one either.
    expect(buttons()).toEqual(['Flip to Black', 'Abort']);
  });

  it('takes back the last move of a game the viewer plays', () => {
    const { onTakeBack } = renderControls({ game: gameOf('human', 'random', 1) });

    fireEvent.click(screen.getByRole('button', { name: 'Take back' }));

    expect(onTakeBack).toHaveBeenCalledOnce();
  });

  it('has nothing to take back before the first move', () => {
    renderControls({ game: gameOf('human', 'random', 0) });

    expect(screen.getByRole('button', { name: 'Take back' })).toBeDisabled();
  });

  it('aborts the game', () => {
    const { onAbort } = renderControls();

    fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

    expect(onAbort).toHaveBeenCalledOnce();
  });

  it('offers only the flip when there is no game to end', () => {
    renderControls({ game: null });

    expect(buttons()).toEqual(['Flip to Black']);
  });

  it('keeps the flip working, and ends nothing, while the server is out of reach', () => {
    const { onFlip } = renderControls({ disabled: true });

    fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

    expect(onFlip).toHaveBeenCalledOnce();
    expect(screen.getByRole('button', { name: 'Resign' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Abort' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Take back' })).toBeDisabled();
  });
});
