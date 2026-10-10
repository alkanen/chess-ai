import { renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LiveGame } from './api';
import { FakeWebSocket } from './test/fakeWebSocket';
import { useRunChannel } from './useRunChannel';

const RESULT = { step: 2000, suite: 'probe-positions', updated: '2026-10-09T12:00:00Z' };
const V1 = { name: 'standard', version: 1 };

describe('useRunChannel', () => {
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
  });

  it('keeps which checkpoints have evaluation results, replaced whole by each event', () => {
    const channel = renderHook(() => useRunChannel('live'));
    const socket = FakeWebSocket.latest;
    socket.open();

    socket.deliver({ type: 'evaluations', results: [RESULT], current_set: V1 });
    socket.deliver({
      type: 'run',
      name: 'live',
      info: null,
      heartbeat: null,
      stale: false,
      notes: null,
    });
    expect(channel.result.current.run?.evaluations).toEqual([RESULT]);
    expect(channel.result.current.run?.probeSet).toEqual(V1);

    const later = { ...RESULT, step: 4000 };
    socket.deliver({ type: 'evaluations', results: [RESULT, later], current_set: null });
    expect(channel.result.current.run?.evaluations).toEqual([RESULT, later]);
    expect(channel.result.current.run?.probeSet).toBeNull();
    expect(channel.result.current.run?.metrics).toEqual([]);
  });

  it('keeps the sample game being played until the server says it is over', () => {
    const channel = renderHook(() => useRunChannel('live'));
    const socket = FakeWebSocket.latest;
    socket.open();
    const live: LiveGame = {
      run: 'live',
      step: 2000,
      index: 0,
      games: 2,
      opponent: 'self',
      updated: '2026-10-10T12:00:00Z',
      // Only passed along here: what is in it is the run page's to draw.
      game: { id: 'g', moves: [] } as unknown as LiveGame['game'],
    };

    socket.deliver({ type: 'live_game', live });
    socket.deliver({
      type: 'run',
      name: 'live',
      info: null,
      heartbeat: null,
      stale: false,
      notes: null,
    });
    expect(channel.result.current.run?.liveGame).toEqual(live);

    socket.deliver({ type: 'live_game', live: null });
    expect(channel.result.current.run?.liveGame).toBeNull();
  });
});
