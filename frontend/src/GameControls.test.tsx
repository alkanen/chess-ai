import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { GameState, PlayerKind } from './api';
import { GameControls } from './GameControls';
import startPosition from './test/fixtures/start-position.json';

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true },
  random: { name: 'Random mover', accepts_moves: false },
} satisfies Record<PlayerKind, { name: string; accepts_moves: boolean }>;

function gameOf(white: PlayerKind, black: PlayerKind): GameState {
  return {
    id: 'a-game',
    white: PLAYERS[white],
    black: PLAYERS[black],
    moves: [],
    position: startPosition as GameState['position'],
  };
}

function renderControls(overrides: Partial<Parameters<typeof GameControls>[0]> = {}) {
  const props = {
    orientation: 'white' as const,
    onFlip: vi.fn(),
    game: gameOf('human', 'random'),
    disabled: false,
    onResign: vi.fn(),
    onAbort: vi.fn(),
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
    const { onResign } = renderControls({ game: gameOf('human', 'human') });

    expect(buttons()).toEqual(['Flip to Black', 'White resigns', 'Black resigns', 'Abort']);

    fireEvent.click(screen.getByRole('button', { name: 'Black resigns' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('black');
  });

  it('offers no resignation in a game the viewer only watches, but still an abort', () => {
    renderControls({ game: gameOf('random', 'random') });

    expect(buttons()).toEqual(['Flip to Black', 'Abort']);
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
  });
});
