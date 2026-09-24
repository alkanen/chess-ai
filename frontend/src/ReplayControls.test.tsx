import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ReplayControls } from './ReplayControls';

/** The controls a third of the way through a four-move game, unless told otherwise. */
function show(changes: { ply?: number; plies?: number; onGo?: () => void } = {}) {
  const onGo = changes.onGo ?? vi.fn();
  render(
    <ReplayControls
      ply={changes.ply ?? 1}
      plies={changes.plies ?? 4}
      onGo={onGo}
      orientation="white"
      onFlip={vi.fn()}
    />,
  );
  return onGo;
}

/** The step buttons that can still be pressed, in the order they are offered. */
function offered(): string[] {
  return screen
    .getAllByRole('button')
    .filter((button) => !(button as HTMLButtonElement).disabled)
    .map((button) => button.getAttribute('aria-label') ?? button.textContent ?? '');
}

describe('ReplayControls', () => {
  it('says how far into the game the board is', () => {
    show({ ply: 1, plies: 4 });

    expect(screen.getByText(/Move 1 of 4/)).toBeInTheDocument();
  });

  it.each([
    ['Start', 0],
    ['Back', 0],
    ['Forward', 2],
    ['End', 4],
  ])('steps to the right move when %s is pressed', (label, expected) => {
    const onGo = show({ ply: 1, plies: 4 });

    fireEvent.click(screen.getByRole('button', { name: label }));

    expect(onGo).toHaveBeenCalledExactlyOnceWith(expected);
  });

  it('offers nothing but going forward at the start of the game', () => {
    show({ ply: 0, plies: 4 });

    expect(offered()).toEqual(['Forward', 'End', 'Flip to Black']);
  });

  it('offers nothing but going back at the end of the game', () => {
    show({ ply: 4, plies: 4 });

    expect(offered()).toEqual(['Start', 'Back', 'Flip to Black']);
  });

  it('turns the board round', () => {
    const onFlip = vi.fn();
    render(
      <ReplayControls ply={1} plies={4} onGo={vi.fn()} orientation="black" onFlip={onFlip} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Flip to White' }));

    expect(onFlip).toHaveBeenCalledOnce();
  });
});
