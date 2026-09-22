import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { PositionSnapshot } from '../api';
import { lastMoveSquares } from '../test/boardQueries';
import startPosition from '../test/fixtures/start-position.json';
import { foolsMateMoves } from '../test/foolsMate';
import { Board } from './Board';

function pieceLabels(): string[] {
  return screen.getAllByRole('img').map((piece) => piece.getAttribute('aria-label') ?? '');
}

describe('Board', () => {
  it('draws every piece of the starting position on its square', () => {
    render(<Board snapshot={startPosition as PositionSnapshot} />);

    expect(pieceLabels()).toHaveLength(32);
    expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'white queen on d1' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'black queen on d8' })).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'black knight on g8' })).toBeInTheDocument();
    for (const file of 'abcdefgh') {
      expect(screen.getByRole('img', { name: `white pawn on ${file}2` })).toBeInTheDocument();
      expect(screen.getByRole('img', { name: `black pawn on ${file}7` })).toBeInTheDocument();
    }
  });

  it('draws only the pieces in the snapshot', () => {
    const snapshot: PositionSnapshot = {
      fen: '4k3/8/8/8/4P3/8/8/4K3 b - - 0 1',
      turn: 'black',
      pieces: {
        e1: { color: 'white', type: 'king' },
        e4: { color: 'white', type: 'pawn' },
        e8: { color: 'black', type: 'king' },
      },
      last_move: null,
      game_over: null,
    };

    render(<Board snapshot={snapshot} />);

    expect(pieceLabels().sort()).toEqual([
      'black king on e8',
      'white king on e1',
      'white pawn on e4',
    ]);
  });

  it('places White at the bottom', () => {
    const snapshot: PositionSnapshot = {
      fen: '7k/8/8/8/8/8/8/K7 w - - 0 1',
      turn: 'white',
      pieces: {
        a1: { color: 'white', type: 'king' },
        h8: { color: 'black', type: 'king' },
      },
      last_move: null,
      game_over: null,
    };

    render(<Board snapshot={snapshot} />);

    const a1 = screen.getByRole('img', { name: 'white king on a1' });
    const h8 = screen.getByRole('img', { name: 'black king on h8' });
    expect([a1.getAttribute('x'), a1.getAttribute('y')]).toEqual(['0', '700']);
    expect([h8.getAttribute('x'), h8.getAttribute('y')]).toEqual(['700', '0']);
  });

  it('highlights the squares of the last move', () => {
    const mate = foolsMateMoves[3];

    const { container } = render(<Board snapshot={mate.position} />);

    expect(lastMoveSquares(container)).toEqual(['d8', 'h4']);
    expect(screen.getByRole('img', { name: 'black queen on h4' })).toBeInTheDocument();
  });

  it('highlights nothing before the first move', () => {
    const { container } = render(<Board snapshot={startPosition as PositionSnapshot} />);

    expect(lastMoveSquares(container)).toEqual([]);
  });
});
