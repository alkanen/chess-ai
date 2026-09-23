import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PlayerKind, PositionSnapshot } from './api';
import { App } from './App';
import { destinations, lastMoveSquares, square } from './test/boardQueries';
import { FakeWebSocket } from './test/fakeWebSocket';
import startPosition from './test/fixtures/start-position.json';
import { foolsMateMoves, foolsMateStart, type StateEvent } from './test/foolsMate';
import { castling, drawnByFiftyMoves } from './test/positions';

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true },
  random: { name: 'Random mover', accepts_moves: false },
} satisfies Record<PlayerKind, { name: string; accepts_moves: boolean }>;

/** A game at the given position, between the given kinds of player. */
function gameOf(white: PlayerKind, black: PlayerKind, position: PositionSnapshot): StateEvent {
  return {
    type: 'state',
    game: { white: PLAYERS[white], black: PLAYERS[black], moves: [], position },
  };
}

/** The state a viewer who connects after `count` moves receives. */
function stateAfter(count: number): StateEvent {
  const moves = foolsMateMoves.slice(0, count);
  return {
    type: 'state',
    game: {
      ...foolsMateStart.game,
      moves: moves.map((event) => event.move),
      position: moves.at(-1)?.position ?? foolsMateStart.game.position,
    },
  };
}

describe('App', () => {
  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('follows the game channel under the path prefix', () => {
    render(<App />);

    const url = new URL('/chess/api/game/ws', window.location.href);
    url.protocol = 'ws:';
    expect(FakeWebSocket.latest.url).toBe(url.href);
    expect(screen.getByText('Connecting to the server…')).toBeInTheDocument();
  });

  it('shows the starting position before any game', () => {
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({ type: 'no_game', position: startPosition as PositionSnapshot });

    expect(screen.getByRole('status')).toHaveTextContent('No game in progress');
    expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start' })).toBeInTheDocument();
  });

  it('plays the moves onto the board as they arrive, then shows the result', () => {
    const { container } = render(<App />);
    const socket = FakeWebSocket.latest;
    socket.open();

    socket.deliver(foolsMateStart);

    expect(screen.getByRole('status')).toHaveTextContent('White to move');
    expect(lastMoveSquares(container)).toEqual([]);

    socket.deliver(foolsMateMoves[0]);

    expect(screen.getByRole('status')).toHaveTextContent('Black to move');
    expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();
    expect(lastMoveSquares(container)).toEqual(['f2', 'f3']);

    for (const move of foolsMateMoves.slice(1)) {
      socket.deliver(move);
    }

    expect(screen.getByRole('status')).toHaveTextContent('Black wins by checkmate (0–1)');
    expect(screen.getByRole('img', { name: 'black queen on h4' })).toBeInTheDocument();
    expect(lastMoveSquares(container)).toEqual(['d8', 'h4']);
  });

  it('names the players', () => {
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({
      type: 'state',
      game: {
        ...foolsMateStart.game,
        white: { name: 'Random mover', accepts_moves: false },
        black: { name: 'Someone else', accepts_moves: false },
      },
    });

    const players = screen.getAllByRole('definition').map((element) => element.textContent);
    expect(players).toEqual(['Random mover', 'Someone else']);
  });

  it('reconnects after losing the connection and catches up with the game', () => {
    vi.useFakeTimers();
    render(<App />);
    const first = FakeWebSocket.latest;
    first.open();
    first.deliver(foolsMateStart);
    first.deliver(foolsMateMoves[0]);

    first.disconnect();

    expect(screen.getByRole('alert')).toHaveTextContent('Reconnecting');
    expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(1000));
    const second = FakeWebSocket.latest;
    expect(second).not.toBe(first);
    second.open();
    expect(screen.getByRole('alert')).toHaveTextContent('Reconnecting');
    second.deliver(stateAfter(3));

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'white pawn on g4' })).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('Black to move');

    second.deliver(foolsMateMoves[3]);

    expect(screen.getByRole('status')).toHaveTextContent('Black wins by checkmate');
  });

  it('waits longer between attempts while the server stays unreachable', () => {
    vi.useFakeTimers();
    render(<App />);

    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it('keeps backing off while connections open but close before sending anything', () => {
    vi.useFakeTimers();
    render(<App />);

    FakeWebSocket.latest.open();
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.latest.open();
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it('starts backing off afresh once a connection delivers the game', () => {
    vi.useFakeTimers();
    render(<App />);
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(2000));
    expect(FakeWebSocket.instances).toHaveLength(3);

    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(foolsMateStart);
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));

    expect(FakeWebSocket.instances).toHaveLength(4);
  });

  it('closes the connection for good when it goes away', () => {
    vi.useFakeTimers();
    const { unmount } = render(<App />);
    const socket = FakeWebSocket.latest;

    unmount();

    expect(socket.closed).toBe(true);
    act(() => vi.advanceTimersByTime(60_000));
    expect(FakeWebSocket.instances).toEqual([socket]);
  });

  describe('making a move', () => {
    it('sends the move a player makes on the board', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(socket.sent).toEqual([JSON.stringify({ type: 'move', uci: 'e1g1' })]);
    });

    it('takes no second move while the first is still on its way to the server', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      fireEvent.mouseEnter(square(container, 'a1'));
      fireEvent.mouseDown(square(container, 'a1'));
      fireEvent.mouseUp(square(container, 'a1'));
      fireEvent.mouseDown(square(container, 'b1'));

      expect(destinations(container)).toEqual([]);
      expect(socket.sent).toEqual([JSON.stringify({ type: 'move', uci: 'e1g1' })]);
    });

    it('takes moves again once the server has answered', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      socket.deliver({ type: 'error', message: 'e1g1 is not a legal move here' });
      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toContain('g1 castling');
    });

    it('shows the rejection of a move and leaves the board as it was', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      socket.deliver({ type: 'error', message: 'e1g1 is not a legal move in ...' });

      expect(screen.getByRole('alert')).toHaveTextContent('e1g1 is not a legal move');
      expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
      expect(screen.getByRole('status')).toHaveTextContent('White to move');
    });

    it('drops the rejection once the game moves on', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      socket.deliver({ type: 'error', message: 'it is not your turn' });
      expect(screen.getByRole('alert')).toBeInTheDocument();

      socket.deliver(gameOf('human', 'random', castling));

      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('takes moves again when the server goes quiet without closing the connection', () => {
      vi.useFakeTimers();
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));
      expect(destinations(container)).toEqual([]);

      // The socket never closes, so nothing else tells the board the move is lost.
      act(() => vi.advanceTimersByTime(30_000));

      expect(screen.getByRole('alert')).toHaveTextContent('has not answered');
      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');
    });

    it('takes moves again after a dropped connection has been remade', () => {
      vi.useFakeTimers();
      const { container } = render(<App />);
      const first = FakeWebSocket.latest;
      first.open();
      first.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      first.disconnect();
      act(() => vi.advanceTimersByTime(1000));
      const second = FakeWebSocket.latest;
      second.open();
      second.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toContain('g1 castling');
    });

    it('takes no move while a player that moves for itself is to move', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'human', castling));

      fireEvent.mouseEnter(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(destinations(container)).toEqual([]);
      expect(socket.sent).toEqual([]);
    });

    it('takes no move once the game is over, moves or no moves', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));

      fireEvent.mouseEnter(square(container, 'e2'));
      fireEvent.mouseDown(square(container, 'e2'));
      fireEvent.mouseUp(square(container, 'e3'));

      expect(screen.getByRole('status')).toHaveTextContent('Draw by the fifty-move rule');
      expect(destinations(container)).toEqual([]);
      expect(socket.sent).toEqual([]);
    });

    it('takes no move while the connection is down', () => {
      vi.useFakeTimers();
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      socket.disconnect();
      fireEvent.mouseEnter(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(destinations(container)).toEqual([]);
      expect(socket.sent).toEqual([]);
    });
  });
});
