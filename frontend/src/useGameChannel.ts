import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import {
  gameChannelUrl,
  type GameEvent,
  type GameState,
  type PositionSnapshot,
  type SubmitMove,
} from './api';

/** What the game view shows. */
export interface GameView {
  /** The current game, or null before any game has been started. */
  game: GameState | null;
  /** The position on the board: the game's, or the starting position without a game. */
  position: PositionSnapshot;
}

interface ChannelState {
  /** Null until the server has sent the first state. */
  view: GameView | null;
  /** Why the server would not act on what we last sent, until the game moves on. */
  error: string | null;
}

export interface GameChannel extends ChannelState {
  /** Whether the current connection has delivered the game, so the view is live. */
  connected: boolean;
  /**
   * Whether a submitted move is still waiting for the server's answer. The wait is
   * bounded, so the board never stays shut for longer than {@link ANSWER_TIMEOUT_MS}.
   */
  movePending: boolean;
  /** Plays a move, in UCI, for the side to move. The server has the final say. */
  submitMove: (uci: string) => void;
}

const MAX_RETRY_DELAY_MS = 10_000;

/**
 * How long the board waits for the server's answer to a submitted move before taking
 * input again. A connection can stay open long after it has stopped carrying anything,
 * and there is no heartbeat to notice, so the wait has to end by itself.
 */
const ANSWER_TIMEOUT_MS = 5_000;

const NO_ANSWER = 'The server has not answered. Your move may not have arrived.';

const DISCONNECTED: ChannelState = { view: null, error: null };

/** How long to wait before reconnecting after the given number of failed attempts. */
function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** failures, MAX_RETRY_DELAY_MS);
}

function applyEvent(state: ChannelState, event: GameEvent): ChannelState {
  switch (event.type) {
    case 'no_game':
      return { view: { game: null, position: event.position }, error: null };
    case 'state':
      return { view: { game: event.game, position: event.game.position }, error: null };
    case 'move': {
      if (state.view?.game == null) {
        return { ...state, error: null };
      }
      const game = {
        ...state.view.game,
        moves: [...state.view.game.moves, event.move],
        position: event.position,
      };
      return { view: { game, position: game.position }, error: null };
    }
    case 'error':
      return { ...state, error: event.message };
  }
}

/**
 * Follows the server's current game over a WebSocket, reconnecting when the connection
 * drops. The server sends the full state on every connection, so nothing is lost, and
 * it is the only judge of the moves submitted through it.
 */
export function useGameChannel(): GameChannel {
  const [{ view, error }, dispatch] = useReducer(applyEvent, DISCONNECTED);
  const [connected, setConnected] = useState(false);
  const [movePending, setMovePending] = useState(false);
  const socket = useRef<WebSocket | null>(null);
  const answer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  /** Stop waiting for an answer to a submitted move, so the board takes input again. */
  const stopWaiting = useCallback(() => {
    clearTimeout(answer.current);
    setMovePending(false);
  }, []);

  useEffect(() => {
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    let stopped = false;

    function connect() {
      const opened = new WebSocket(gameChannelUrl());
      socket.current = opened;
      opened.onmessage = (message: MessageEvent<string>) => {
        // The server starts every connection with the game's state, so a message (rather
        // than the socket opening) shows that a connection works.
        failures = 0;
        setConnected(true);
        // Whatever the server has to say, it has answered any move we submitted.
        stopWaiting();
        dispatch(JSON.parse(message.data) as GameEvent);
      };
      opened.onclose = () => {
        setConnected(false);
        // Nothing can arrive on this socket now, least of all an answer.
        stopWaiting();
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
      clearTimeout(answer.current);
      socket.current?.close();
      socket.current = null;
    };
  }, [stopWaiting]);

  const submitMove = useCallback(
    (uci: string) => {
      const open = socket.current;
      // A socket that is closing or closed throws nothing and reports nothing: it
      // discards what it is given. Waiting for an answer to that would never end.
      if (open === null || open.readyState !== WebSocket.OPEN) {
        return;
      }
      const message: SubmitMove = { type: 'move', uci };
      open.send(JSON.stringify(message));
      setMovePending(true);
      clearTimeout(answer.current);
      answer.current = setTimeout(() => {
        stopWaiting();
        dispatch({ type: 'error', message: NO_ANSWER });
      }, ANSWER_TIMEOUT_MS);
    },
    [stopWaiting],
  );

  return { view, error, connected, movePending, submitMove };
}
