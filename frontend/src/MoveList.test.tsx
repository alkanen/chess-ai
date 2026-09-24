import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { MoveRecord } from './api';
import { foolsMateMoves } from './test/foolsMate';
import { MoveList } from './MoveList';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function movesUpTo(count: number): MoveRecord[] {
  return foolsMateMoves.slice(0, count).map((event) => event.move);
}

/** Moves that are only there to be counted; what they are does not matter here. */
function named(...sans: string[]): MoveRecord[] {
  return sans.map((san) => ({ uci: 'a1a1', san, thoughts: null }));
}

/** The move list as it reads, one numbered move to a line. */
function lines(): string[] {
  return screen
    .getAllByRole('listitem')
    .map((item) => Array.from(item.querySelectorAll('.san'), (san) => san.textContent).join(' '));
}

/** The number each line is marked with, which the list itself counts out. */
function firstNumber(): number {
  return Number(screen.getByRole('list').getAttribute('start') ?? 1);
}

describe('MoveList', () => {
  it('pairs the moves under their move numbers', () => {
    render(<MoveList moves={movesUpTo(4)} startFen={START} />);

    expect(lines()).toEqual(['f3 e5', 'g4 Qh4#']);
    expect(firstNumber()).toBe(1);
  });

  it('shows a move on its own until it has been answered', () => {
    render(<MoveList moves={movesUpTo(3)} startFen={START} />);

    expect(lines()).toEqual(['f3 e5', 'g4']);
  });

  it('says so before the first move', () => {
    render(<MoveList moves={[]} startFen={START} />);

    expect(screen.queryAllByRole('listitem')).toEqual([]);
    expect(screen.getByText('No moves yet.')).toBeInTheDocument();
  });

  it('grows as the game is played', () => {
    const { rerender } = render(<MoveList moves={movesUpTo(1)} startFen={START} />);
    expect(lines()).toEqual(['f3']);

    rerender(<MoveList moves={movesUpTo(4)} startFen={START} />);

    expect(lines()).toEqual(['f3 e5', 'g4 Qh4#']);
  });

  it('shrinks again when moves are taken back', () => {
    const { rerender } = render(<MoveList moves={movesUpTo(4)} startFen={START} />);

    rerender(<MoveList moves={movesUpTo(2)} startFen={START} />);

    expect(lines()).toEqual(['f3 e5']);
  });

  it('counts from the move number the game started at', () => {
    const fen = 'r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 8 7';

    render(<MoveList moves={named('d3', 'd6', 'Bg5')} startFen={fen} />);

    expect(lines()).toEqual(['d3 d6', 'Bg5']);
    expect(firstNumber()).toBe(7);
  });

  it('opens with a half-line when the game started with Black to move', () => {
    const fen = 'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2';

    render(<MoveList moves={named('Qxd5', 'Nc3', 'Qa5')} startFen={fen} />);

    // Black's move stands alone under its number, as "2… Qxd5" does in a book.
    expect(lines()).toEqual(['… Qxd5', 'Nc3 Qa5']);
    expect(firstNumber()).toBe(2);
  });
});
