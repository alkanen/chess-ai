import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { MoveRecord } from './api';
import { foolsMateMoves } from './test/foolsMate';
import { MoveList } from './MoveList';

function movesUpTo(count: number): MoveRecord[] {
  return foolsMateMoves.slice(0, count).map((event) => event.move);
}

/** The move list as it reads, one numbered move to a line. */
function lines(): string[] {
  return screen
    .getAllByRole('listitem')
    .map((item) => Array.from(item.querySelectorAll('.san'), (san) => san.textContent).join(' '));
}

describe('MoveList', () => {
  it('pairs the moves under their move numbers', () => {
    render(<MoveList moves={movesUpTo(4)} />);

    expect(lines()).toEqual(['f3 e5', 'g4 Qh4#']);
  });

  it('shows a move on its own until it has been answered', () => {
    render(<MoveList moves={movesUpTo(3)} />);

    expect(lines()).toEqual(['f3 e5', 'g4']);
  });

  it('says so before the first move', () => {
    render(<MoveList moves={[]} />);

    expect(screen.queryAllByRole('listitem')).toEqual([]);
    expect(screen.getByText('No moves yet.')).toBeInTheDocument();
  });

  it('grows as the game is played', () => {
    const { rerender } = render(<MoveList moves={movesUpTo(1)} />);
    expect(lines()).toEqual(['f3']);

    rerender(<MoveList moves={movesUpTo(4)} />);

    expect(lines()).toEqual(['f3 e5', 'g4 Qh4#']);
  });
});
