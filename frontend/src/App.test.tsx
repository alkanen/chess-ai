import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PositionSnapshot } from './api';
import { App } from './App';
import { lastMoveSquares } from './test/boardQueries';
import { FakeWebSocket } from './test/fakeWebSocket';
import startPosition from './test/fixtures/start-position.json';
import { foolsMateMoves, foolsMateStart, type StateEvent } from './test/foolsMate';

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

    FakeWebSocket.latest.send({ type: 'no_game', position: startPosition as PositionSnapshot });

    expect(screen.getByRole('status')).toHaveTextContent('No game in progress');
    expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start' })).toBeInTheDocument();
  });

  it('plays the moves onto the board as they arrive, then shows the result', () => {
    const { container } = render(<App />);
    const socket = FakeWebSocket.latest;
    socket.open();

    socket.send(foolsMateStart);

    expect(screen.getByRole('status')).toHaveTextContent('White to move');
    expect(lastMoveSquares(container)).toEqual([]);

    socket.send(foolsMateMoves[0]);

    expect(screen.getByRole('status')).toHaveTextContent('Black to move');
    expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();
    expect(lastMoveSquares(container)).toEqual(['f2', 'f3']);

    for (const move of foolsMateMoves.slice(1)) {
      socket.send(move);
    }

    expect(screen.getByRole('status')).toHaveTextContent('Black wins by checkmate (0–1)');
    expect(screen.getByRole('img', { name: 'black queen on h4' })).toBeInTheDocument();
    expect(lastMoveSquares(container)).toEqual(['d8', 'h4']);
  });

  it('names the players', () => {
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.send({
      type: 'state',
      game: { ...foolsMateStart.game, white: 'Random mover', black: 'Someone else' },
    });

    const players = screen.getAllByRole('definition').map((element) => element.textContent);
    expect(players).toEqual(['Random mover', 'Someone else']);
  });

  it('reconnects after losing the connection and catches up with the game', () => {
    vi.useFakeTimers();
    render(<App />);
    const first = FakeWebSocket.latest;
    first.open();
    first.send(foolsMateStart);
    first.send(foolsMateMoves[0]);

    first.disconnect();

    expect(screen.getByRole('alert')).toHaveTextContent('Reconnecting');
    expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(1000));
    const second = FakeWebSocket.latest;
    expect(second).not.toBe(first);
    second.open();
    expect(screen.getByRole('alert')).toHaveTextContent('Reconnecting');
    second.send(stateAfter(3));

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'white pawn on g4' })).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('Black to move');

    second.send(foolsMateMoves[3]);

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
    FakeWebSocket.latest.send(foolsMateStart);
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
});
