import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { GameState } from './api';
import { FakeWebSocket } from './test/fakeWebSocket';
import startPosition from './test/fixtures/start-position.json';
import { useGameChannel } from './useGameChannel';

const GAME = {
  id: 'a-game',
  white: { name: 'Human', accepts_moves: true },
  black: { name: 'Random mover', accepts_moves: false },
  moves: [],
  position: startPosition,
} as unknown as GameState;

/** A channel following a game, on an open connection. */
function following() {
  const channel = renderHook(() => useGameChannel());
  const socket = FakeWebSocket.latest;
  socket.open();
  socket.deliver({ type: 'state', game: GAME });
  return { channel, socket };
}

describe('useGameChannel', () => {
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

  it('waits for the server to answer a submitted move', () => {
    const { channel, socket } = following();

    act(() => channel.result.current.submitMove('e2e4'));

    expect(socket.sent.map((message) => JSON.parse(message))).toEqual([
      { game: 'a-game', type: 'move', uci: 'e2e4' },
    ]);
    expect(channel.result.current.movePending).toBe(true);
  });

  it('stops waiting for an answer when the connection drops', () => {
    vi.useFakeTimers();
    const { channel, socket } = following();
    act(() => channel.result.current.submitMove('e2e4'));

    socket.disconnect();

    expect(channel.result.current.movePending).toBe(false);
  });

  it('sends nothing through a socket that has closed, and does not wait for it', () => {
    vi.useFakeTimers();
    const { channel, socket } = following();
    socket.disconnect();

    act(() => channel.result.current.submitMove('e2e4'));

    expect(socket.sent).toEqual([]);
    expect(channel.result.current.movePending).toBe(false);
  });

  it('keeps waiting while an answer is merely slow', () => {
    vi.useFakeTimers();
    const { channel } = following();
    act(() => channel.result.current.submitMove('e2e4'));

    act(() => vi.advanceTimersByTime(1000));

    expect(channel.result.current.movePending).toBe(true);
    expect(channel.result.current.error).toBeNull();
  });

  it('gives up on a connection that stays open but stops answering', () => {
    vi.useFakeTimers();
    const { channel } = following();
    act(() => channel.result.current.submitMove('e2e4'));

    act(() => vi.advanceTimersByTime(30_000));

    expect(channel.result.current.movePending).toBe(false);
    expect(channel.result.current.connected).toBe(true);
    expect(channel.result.current.error).not.toBeNull();
  });

  it('stops waiting as soon as the server answers', () => {
    vi.useFakeTimers();
    const { channel, socket } = following();
    act(() => channel.result.current.submitMove('e2e4'));

    socket.deliver({ type: 'error', message: 'e2e4 is not a legal move here' });

    expect(channel.result.current.movePending).toBe(false);
    expect(channel.result.current.error).toBe('e2e4 is not a legal move here');
  });
});
