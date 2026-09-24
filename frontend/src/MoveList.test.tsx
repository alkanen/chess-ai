import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
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

  describe('while a game is being stepped through', () => {
    it('jumps to the position a move led to when the move is clicked', () => {
      const onSelect = vi.fn();
      render(<MoveList moves={movesUpTo(4)} startFen={START} current={0} onSelect={onSelect} />);

      fireEvent.click(screen.getByRole('button', { name: 'g4' }));

      // The third move of the game, so three moves have been played once it has.
      expect(onSelect).toHaveBeenCalledExactlyOnceWith(3);
    });

    it('marks the move the board is showing', () => {
      const { rerender } = render(
        <MoveList moves={movesUpTo(4)} startFen={START} current={2} onSelect={vi.fn()} />,
      );
      expect(screen.getByRole('button', { name: 'e5' })).toHaveAttribute('aria-current', 'step');

      rerender(<MoveList moves={movesUpTo(4)} startFen={START} current={4} onSelect={vi.fn()} />);

      expect(screen.getByRole('button', { name: 'e5' })).not.toHaveAttribute('aria-current');
      expect(screen.getByRole('button', { name: 'Qh4#' })).toHaveAttribute('aria-current', 'step');
    });

    it('marks no move at the position the game started from', () => {
      render(<MoveList moves={movesUpTo(4)} startFen={START} current={0} onSelect={vi.fn()} />);

      expect(screen.queryByRole('button', { current: 'step' })).not.toBeInTheDocument();
      expect(lines()).toEqual(['f3 e5', 'g4 Qh4#']);
    });
  });

  describe('keeping the move being looked at in view', () => {
    /** jsdom lays nothing out, so the scroller is given a height to scroll within. */
    const SCROLLABLE = 500;
    let height: PropertyDescriptor | undefined;

    beforeEach(() => {
      height = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollHeight');
      Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
        value: SCROLLABLE,
        configurable: true,
      });
    });

    afterEach(() => {
      if (height === undefined) {
        Reflect.deleteProperty(HTMLElement.prototype, 'scrollHeight');
      } else {
        Object.defineProperty(HTMLElement.prototype, 'scrollHeight', height);
      }
    });

    /** The box the list scrolls inside. */
    function scroller(): HTMLElement {
      const box = screen.getByRole('list').parentElement;
      if (box === null) throw new Error('the list is not in a scroller');
      return box;
    }

    it('follows a game being played to its last move', () => {
      render(<MoveList moves={movesUpTo(4)} startFen={START} />);

      expect(scroller().scrollTop).toBe(SCROLLABLE);
    });

    it('stays at the first move at the position a replayed game began in', () => {
      // No move is on the board at the start, so there is none to scroll to — and the
      // end of the game, which is where a game being played is watched, is the one
      // place the reader is certainly not looking.
      render(<MoveList moves={movesUpTo(4)} startFen={START} current={0} onSelect={vi.fn()} />);

      expect(scroller().scrollTop).toBe(0);
    });

    it('leaves a game being played where the reader left it when it arrives again', () => {
      // A dropped connection is caught up with a fresh state event, whose moves are
      // parsed anew: the same game, in a new array. Someone who had scrolled up to
      // look at an earlier move is not to be dragged back to the end by that.
      const played = named('e4', 'e5', 'Nf3', 'Nc6');
      const { rerender } = render(<MoveList moves={played} startFen={START} />);
      scroller().scrollTop = 120;

      rerender(<MoveList moves={named('e4', 'e5', 'Nf3', 'Nc6')} startFen={START} />);

      expect(scroller().scrollTop).toBe(120);
    });

    it('follows a game being played to each new move as it arrives', () => {
      const { rerender } = render(<MoveList moves={movesUpTo(2)} startFen={START} />);
      scroller().scrollTop = 0;

      rerender(<MoveList moves={movesUpTo(3)} startFen={START} />);

      expect(scroller().scrollTop).toBe(SCROLLABLE);
    });
  });

  it('leaves the moves unclickable while a game is being played', () => {
    render(<MoveList moves={movesUpTo(4)} startFen={START} />);

    expect(screen.queryAllByRole('button')).toEqual([]);
  });

  it('opens with a half-line when the game started with Black to move', () => {
    const fen = 'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2';

    render(<MoveList moves={named('Qxd5', 'Nc3', 'Qa5')} startFen={fen} />);

    // Black's move stands alone under its number, as "2… Qxd5" does in a book.
    expect(lines()).toEqual(['… Qxd5', 'Nc3 Qa5']);
    expect(firstNumber()).toBe(2);
  });
});
