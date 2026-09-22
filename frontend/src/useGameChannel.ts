import { useEffect, useReducer, useState } from 'react';
import { gameChannelUrl, type GameEvent, type GameState, type PositionSnapshot } from './api';

/** What the game view shows. */
export interface GameView {
  /** The current game, or null before any game has been started. */
  game: GameState | null;
  /** The position on the board: the game's, or the starting position without a game. */
  position: PositionSnapshot;
}

export interface GameChannel {
  /** Null until the server has sent the first state. */
  view: GameView | null;
  /** Whether the current connection has delivered the game, so the view is live. */
  connected: boolean;
}

const MAX_RETRY_DELAY_MS = 10_000;

/** How long to wait before reconnecting after the given number of failed attempts. */
function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** failures, MAX_RETRY_DELAY_MS);
}

function applyEvent(view: GameView | null, event: GameEvent): GameView | null {
  switch (event.type) {
    case 'no_game':
      return { game: null, position: event.position };
    case 'state':
      return { game: event.game, position: event.game.position };
    case 'move': {
      if (view?.game == null) {
        return view;
      }
      const game = {
        ...view.game,
        moves: [...view.game.moves, event.move],
        position: event.position,
      };
      return { game, position: game.position };
    }
  }
}

/**
 * Follows the server's current game over a WebSocket, reconnecting when the connection
 * drops. The server sends the full state on every connection, so nothing is lost.
 */
export function useGameChannel(): GameChannel {
  const [view, dispatch] = useReducer(applyEvent, null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let socket: WebSocket;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    let stopped = false;

    function connect() {
      socket = new WebSocket(gameChannelUrl());
      socket.onmessage = (message: MessageEvent<string>) => {
        // The server starts every connection with the game's state, so a message (rather
        // than the socket opening) shows that a connection works.
        failures = 0;
        setConnected(true);
        dispatch(JSON.parse(message.data) as GameEvent);
      };
      socket.onclose = () => {
        setConnected(false);
        if (!stopped) {
          retryTimer = setTimeout(connect, retryDelayMs(failures));
          failures += 1;
        }
      };
    }

    connect();
    return () => {
      stopped = true;
      clearTimeout(retryTimer);
      socket.close();
    };
  }, []);

  return { view, connected };
}
