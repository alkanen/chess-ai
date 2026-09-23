import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { PositionSnapshot } from '../api';
import {
  checkSquare,
  destinations,
  lastMoveSquares,
  originSquare,
  promotionBackdrop,
  promotionChoice,
  promotionChoices,
  square,
  squareAt,
} from '../test/boardQueries';
import startPosition from '../test/fixtures/start-position.json';
import { foolsMateMoves } from '../test/foolsMate';
import {
  blackPromotion,
  castling,
  check,
  enPassant,
  pin,
  promotion,
} from '../test/positions';
import { Board } from './Board';
import { pieceImage } from './pieces';

function pieceLabels(): string[] {
  return screen.getAllByRole('img').map((piece) => piece.getAttribute('aria-label') ?? '');
}

/** The rank numbers and file letters around the board, in the order they are drawn. */
function coordinates(container: HTMLElement) {
  const texts = Array.from(container.querySelectorAll('.coordinate'));
  return {
    ranks: texts.filter((text) => /^[1-8]$/.test(text.textContent ?? '')),
    files: texts.filter((text) => /^[a-h]$/.test(text.textContent ?? '')),
  };
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
      check_square: null,
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
      check_square: null,
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

  describe('promotion', () => {
    /** Moves the b7 pawn to b8 by clicking both squares, which promotes it. */
    function clickToLastRank(container: HTMLElement) {
      fireEvent.mouseDown(square(container, 'b7'));
      fireEvent.mouseUp(square(container, 'b7'));
      fireEvent.mouseDown(square(container, 'b8'));
      fireEvent.mouseUp(square(container, 'b8'));
    }

    it('asks which piece the pawn becomes instead of submitting a move', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);

      clickToLastRank(container);

      expect(promotionChoices(container)).toEqual(['queen', 'rook', 'bishop', 'knight']);
      expect(screen.getByRole('button', { name: 'Promote to knight' })).toBeInTheDocument();
      expect(onMove).not.toHaveBeenCalled();
    });

    it('submits the piece that is chosen', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);

      fireEvent.mouseDown(promotionChoice(container, 'knight'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('b7b8n');
      expect(promotionChoices(container)).toEqual([]);
    });

    it('asks on a capture-promotion too', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'b7'));
      fireEvent.mouseUp(square(container, 'a8'));

      expect(promotionChoices(container)).toEqual(['queen', 'rook', 'bishop', 'knight']);

      fireEvent.mouseDown(promotionChoice(container, 'rook'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('b7a8r');
    });

    it('offers Black the same choice, in Black\'s pieces', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={blackPromotion} interactive onMove={onMove} />);

      fireEvent.mouseDown(square(container, 'b2'));
      fireEvent.mouseUp(square(container, 'b1'));

      expect(promotionChoices(container)).toEqual(['queen', 'rook', 'bishop', 'knight']);
      const queen = promotionChoice(container, 'queen').querySelector('image');
      expect(queen?.getAttribute('href')).toBe(pieceImage({ color: 'black', type: 'queen' }));

      fireEvent.mouseDown(promotionChoice(container, 'bishop'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('b2b1b');
    });

    it('stacks the choices from the promotion square towards the mover', () => {
      const { container } = render(<Board snapshot={promotion} interactive />);
      clickToLastRank(container);

      const ys = ['queen', 'rook', 'bishop', 'knight'].map((piece) =>
        promotionChoice(container, piece).querySelector('image')?.getAttribute('y'),
      );

      // b8 is the top rank, so the choices hang down the b-file and stay on the board.
      expect(ys).toEqual(['0', '100', '200', '300']);
    });

    it('submits nothing and returns the pawn when the picker is dismissed', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);

      fireEvent.mouseDown(promotionBackdrop(container));

      expect(onMove).not.toHaveBeenCalled();
      expect(promotionChoices(container)).toEqual([]);
      expect(screen.getByRole('img', { name: 'white pawn on b7' })).toBeInTheDocument();
      expect(originSquare(container)).toBeNull();
    });

    it('makes no promotion with a button other than the primary one', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);

      fireEvent.mouseDown(promotionChoice(container, 'queen'), { button: 2 });

      // The board calls a move off on the other buttons, and so does the picker.
      expect(onMove).not.toHaveBeenCalled();
      expect(promotionChoices(container)).toEqual([]);
    });

    it('submits the choice Enter lands on', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);

      fireEvent.keyDown(promotionChoice(container, 'rook'), { key: 'Enter' });

      expect(onMove).toHaveBeenCalledExactlyOnceWith('b7b8r');
      expect(promotionChoices(container)).toEqual([]);
    });

    it('submits the choice Space lands on', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);

      fireEvent.keyDown(promotionChoice(container, 'knight'), { key: ' ' });

      expect(onMove).toHaveBeenCalledExactlyOnceWith('b7b8n');
    });

    it('puts the choices in the tab order and starts on the queen', () => {
      const { container } = render(<Board snapshot={promotion} interactive />);

      clickToLastRank(container);

      for (const piece of ['queen', 'rook', 'bishop', 'knight']) {
        expect(promotionChoice(container, piece)).toHaveAttribute('tabindex', '0');
      }
      expect(document.activeElement).toBe(promotionChoice(container, 'queen'));
    });

    it('lets Escape call the promotion off', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);

      fireEvent.keyDown(window, { key: 'Escape' });

      expect(promotionChoices(container)).toEqual([]);
      expect(onMove).not.toHaveBeenCalled();
    });

    it('takes the next move once a dismissed promotion is behind it', () => {
      const onMove = vi.fn();
      const { container } = render(<Board snapshot={promotion} interactive onMove={onMove} />);
      clickToLastRank(container);
      fireEvent.mouseDown(promotionBackdrop(container));

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'e2'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('e1e2');
    });

    it('calls the promotion off when the position moves on', () => {
      const onMove = vi.fn();
      const { container, rerender } = render(
        <Board snapshot={promotion} interactive onMove={onMove} />,
      );
      clickToLastRank(container);

      rerender(<Board snapshot={castling} interactive onMove={onMove} />);

      expect(promotionChoices(container)).toEqual([]);
      expect(onMove).not.toHaveBeenCalled();
    });

    it('shows no highlights under the picker', () => {
      const { container } = render(<Board snapshot={promotion} interactive />);
      clickToLastRank(container);

      expect(destinations(container)).toEqual([]);
      expect(originSquare(container)).toBeNull();
    });
  });
  describe('the king in check', () => {
    it('highlights the square of the king that is in check', () => {
      const { container } = render(<Board snapshot={check} />);

      expect(checkSquare(container)).toBe('e8');
    });

    it('highlights the mated king too', () => {
      const { container } = render(<Board snapshot={foolsMateMoves[3].position} />);

      expect(checkSquare(container)).toBe('e1');
    });

    it('highlights nothing when no king is in check', () => {
      const { container } = render(<Board snapshot={startPosition as PositionSnapshot} />);

      expect(checkSquare(container)).toBeNull();
    });

    it('follows the king to the other end of a flipped board', () => {
      const { container } = render(<Board snapshot={check} orientation="black" />);

      expect(checkSquare(container)).toBe('e8');
      // The eighth rank is at the bottom of the screen with Black at the bottom.
      expect(container.querySelector('.check')?.getAttribute('y')).toBe('700');
    });
  });

  describe('flipped, with Black at the bottom', () => {
    it('places Black at the bottom and White at the top', () => {
      const snapshot: PositionSnapshot = {
        fen: '7k/8/8/8/8/8/8/K7 w - - 0 1',
        turn: 'white',
        pieces: {
          a1: { color: 'white', type: 'king' },
          h8: { color: 'black', type: 'king' },
        },
        last_move: null,
        check_square: null,
        legal_moves: {},
        game_over: null,
      };

      render(<Board snapshot={snapshot} orientation="black" />);

      const a1 = screen.getByRole('img', { name: 'white king on a1' });
      const h8 = screen.getByRole('img', { name: 'black king on h8' });
      expect([a1.getAttribute('x'), a1.getAttribute('y')]).toEqual(['700', '0']);
      expect([h8.getAttribute('x'), h8.getAttribute('y')]).toEqual(['0', '700']);
    });

    it('turns the squares that take the clicks round with the board', () => {
      const { container } = render(<Board snapshot={castling} orientation="black" />);

      expect(squareAt(container, 'a1')).toBe('700,0');
      expect(squareAt(container, 'h8')).toBe('0,700');
      expect(squareAt(container, 'e1')).toBe('300,0');
    });

    it('keeps the coordinates down the left edge and along the bottom', () => {
      const { container } = render(<Board snapshot={castling} orientation="black" />);

      const { ranks, files } = coordinates(container);
      expect(ranks.map((text) => text.textContent).sort().join('')).toBe('12345678');
      expect(files.map((text) => text.textContent).sort().join('')).toBe('abcdefgh');
      // The leftmost column is the h-file once flipped, and the bottom row the eighth rank.
      expect(ranks.map((text) => text.getAttribute('x'))).toEqual(Array(8).fill('4'));
      expect(files.map((text) => text.getAttribute('y'))).toEqual(Array(8).fill('796'));
      // Within those, the first rank is at the top of the board and the a-file at its right.
      const rankY = Object.fromEntries(ranks.map((t) => [t.textContent, t.getAttribute('y')]));
      const fileX = Object.fromEntries(files.map((t) => [t.textContent, t.getAttribute('x')]));
      expect([rankY['1'], rankY['8']]).toEqual(['4', '704']);
      expect([fileX['a'], fileX['h']]).toEqual(['796', '96']);
    });

    it('shows the legal destinations of a hovered piece', () => {
      const { container } = render(<Board snapshot={castling} orientation="black" interactive />);

      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toContain('g1 castling');
      expect(originSquare(container)).toBe('e1');
    });

    it('submits the move of two clicks, as it does the right way up', () => {
      const onMove = vi.fn();
      const { container } = render(
        <Board snapshot={castling} orientation="black" interactive onMove={onMove} />,
      );

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'c1'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('e1c1');
    });

    it('submits the move of a dragged piece', () => {
      const onMove = vi.fn();
      const { container } = render(
        <Board snapshot={castling} orientation="black" interactive onMove={onMove} />,
      );

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'g1'));

      expect(onMove).toHaveBeenCalledExactlyOnceWith('e1g1');
    });

    it('stacks the promotion choices from the promotion square towards the mover', () => {
      const { container } = render(<Board snapshot={promotion} orientation="black" interactive />);

      fireEvent.mouseDown(square(container, 'b7'));
      fireEvent.mouseUp(square(container, 'b8'));

      const ys = ['queen', 'rook', 'bishop', 'knight'].map((piece) =>
        promotionChoice(container, piece).querySelector('image')?.getAttribute('y'),
      );
      // Flipped, b8 is at the bottom of the screen, so the choices climb the board.
      expect(ys).toEqual(['700', '600', '500', '400']);
    });
  });
});
