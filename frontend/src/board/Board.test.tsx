import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { PositionSnapshot } from '../api';
import { destinations, lastMoveSquares, originSquare, square } from '../test/boardQueries';
import startPosition from '../test/fixtures/start-position.json';
import { foolsMateMoves } from '../test/foolsMate';
import { castling, enPassant, pin, promotion } from '../test/positions';
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
      legal_moves: {},
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
      legal_moves: {},
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

  describe('legal-move hover', () => {
    it('shows the hovered piece its legal destinations, castling included', () => {
      const { container } = render(<Board snapshot={castling} interactive />);

      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toEqual([
        'c1 castling',
        'd1 quiet',
        'd2 quiet',
        'e2 quiet',
        'f1 quiet',
        'f2 quiet',
        'g1 castling',
      ]);
      expect(originSquare(container)).toBe('e1');
    });

    it('draws an en passant capture apart from the quiet push', () => {
      const { container } = render(<Board snapshot={enPassant} interactive />);

      fireEvent.mouseEnter(square(container, 'e5'));

      expect(destinations(container)).toEqual(['d6 en-passant', 'e6 quiet']);
    });

    it('draws a capture apart from a quiet move, and promotions only once each', () => {
      const { container } = render(<Board snapshot={promotion} interactive />);

      fireEvent.mouseEnter(square(container, 'b7'));

      expect(destinations(container)).toEqual(['a8 capture', 'b8 quiet']);
    });

    it('shows nothing for a pinned piece that cannot move', () => {
      const { container } = render(<Board snapshot={pin} interactive />);

      fireEvent.mouseEnter(square(container, 'e4'));

      expect(destinations(container)).toEqual([]);
      expect(originSquare(container)).toBeNull();
    });

    it('shows nothing for the opponent\'s pieces', () => {
      const { container } = render(<Board snapshot={castling} interactive />);

      fireEvent.mouseEnter(square(container, 'e8'));

      expect(destinations(container)).toEqual([]);
    });

    it('shows nothing while it is not your turn', () => {
      const { container } = render(<Board snapshot={castling} />);

      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toEqual([]);
      expect(originSquare(container)).toBeNull();
    });

    it('stops showing destinations once the pointer leaves the board', () => {
      const { container } = render(<Board snapshot={castling} interactive />);
      fireEvent.mouseEnter(square(container, 'e1'));

      fireEvent.mouseLeave(container.querySelector('.board') as Element);

      expect(destinations(container)).toEqual([]);
    });
  });

  describe('making a move', () => {
    it('submits the move of a click on the piece and a click on the destination', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} interactive onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));

      expect(originSquare(container)).toBe('e1');
      expect(onMove).not.toHaveBeenCalled();

      fireEvent.mouseDown(square(container, 'g1'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('e1g1');
    });

    it('submits the move of a piece dragged onto a destination', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} interactive onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseEnter(square(container, 'c1'));
      fireEvent.mouseUp(square(container, 'c1'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('e1c1');
    });

    it('keeps showing the destinations of the piece being dragged', () => {
      const { container } = render(<Board snapshot={castling} interactive />);

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseEnter(square(container, 'a1'));

      expect(originSquare(container)).toBe('e1');
      expect(destinations(container)).toContain('g1 castling');
    });

    it('auto-queens a promotion until the promotion picker lands', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'b7'));
      fireEvent.mouseUp(square(container, 'b7'));
      fireEvent.mouseDown(square(container, 'a8'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('b7a8q');
    });

    it('submits nothing for a piece dragged onto a square it may not reach', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} interactive onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e4'));

      expect(onMove).not.toHaveBeenCalled();
      expect(originSquare(container)).toBeNull();
    });

    it('submits nothing when it is not your turn', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(onMove).not.toHaveBeenCalled();
    });

    it('makes no move with a button other than the primary one', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} interactive onMove={onMove} />);
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));

      fireEvent.mouseDown(square(container, 'g1'), { button: 2 });

      expect(onMove).not.toHaveBeenCalled();
    });

    it('lets a non-primary button call off the move being made', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} interactive onMove={onMove} />);
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      expect(originSquare(container)).toBe('e1');

      fireEvent.mouseDown(square(container, 'a1'), { button: 2 });
      fireEvent.mouseUp(square(container, 'a1'), { button: 2 });

      expect(originSquare(container)).toBeNull();
      expect(onMove).not.toHaveBeenCalled();
    });

    it('drops the selection when a click lands on a square with nothing to move', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={castling} interactive onMove={onMove} />);
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));

      fireEvent.mouseDown(square(container, 'e4'));
      fireEvent.mouseUp(square(container, 'e4'));

      expect(onMove).not.toHaveBeenCalled();
      expect(originSquare(container)).toBeNull();
    });
  });
});
