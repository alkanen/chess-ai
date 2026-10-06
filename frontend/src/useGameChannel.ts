import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import {
  gameChannelUrl,
  type Access,
  type GameEvent,
  type GameState,
  type PositionSnapshot,
  type ViewerMessage,
} from './api';

/** What the game view shows. */
export interface GameView {
  game: GameState;
  /** The position on the board, which is the game's. */
  position: PositionSnapshot;
  /** What the link this game was reached through may do in it. */
  access: Access;
  /** The game's watch link. */
  watch: string;
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
   * Why the link reaches no game, once the server has said it does not: it never did, or
   * the game was aborted. Nothing is followed after that.
   */
  missing: string | null;
  /**
   * Whether a submitted move is still waiting for the server's answer. The wait is
   * bounded, so the board never stays shut for longer than {@link ANSWER_TIMEOUT_MS}.
   */
  movePending: boolean;
  /** Plays a move, in UCI, for the link's side. The server has the final say. */
  submitMove: (uci: string) => void;
  /** Resigns the game for the link's side, ending it for every viewer. */
  resign: () => void;
  /** Aborts the game, or asks the other person to agree to it. */
  abort: () => void;
  /** Takes back the side's last move, or asks the other person to agree to it. */
  takeBack: () => void;
  /**
   * Agrees to, or declines, what the other person asked, naming the request: an answer that
   * arrives late must not agree to something asked after it.
   */
  answer: (request: number, accept: boolean) => void;
}

const MAX_RETRY_DELAY_MS = 10_000;

/**
 * How long the board waits for the server's answer to a submitted move before taking
 * input again. A connection can stay open long after it has stopped carrying anything,
 * and there is no heartbeat to notice, so the wait has to end by itself.
 */
const ANSWER_TIMEOUT_MS = 5_000;

const NO_ANSWER = 'The server has not answered. Your move may not have arrived.';

/**
 * How the server closes a connection it has nothing more to send on: the game has ended, or
 * (with an error before it) the link reaches no game. Either is final, and reconnecting would
 * only be told the same again. Any other close, such as the server restarting, is reconnected.
 */
const DONE = 1000;
const NO_SUCH_GAME = 1008;

const DISCONNECTED: ChannelState = { view: null, error: null };

/** How long to wait before reconnecting after the given number of failed attempts. */
function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** failures, MAX_RETRY_DELAY_MS);
}

function applyEvent(state: ChannelState, event: GameEvent): ChannelState {
  switch (event.type) {
    case 'state':
      return {
        view: {
          game: event.game,
          position: event.game.position,
          access: event.access,
          watch: event.watch,
        },
        error: null,
      };
    case 'move': {
      if (state.view === null) {
        return { ...state, error: null };
      }
      const game = {
        ...state.view.game,
        moves: [...state.view.game.moves, event.move],
        position: event.position,
        // A request was about the position before the move.
        request: null,
      };
      return {
        view: { ...state.view, game, position: game.position },
        error: null,
      };
    }
    case 'takeback': {
      // The moves after the takeback never happened, and the position is the one the
      // game stood in before them.
      if (state.view === null) {
        return { ...state, error: null };
      }
      const game = {
        ...state.view.game,
        moves: state.view.game.moves.slice(0, event.ply),
        position: event.position,
        request: null,
      };
      return {
        view: { ...state.view, game, position: event.position },
        error: null,
      };
    }
    case 'request': {
      if (state.view === null) {
        return state;
      }
      const game = { ...state.view.game, request: event.request };
      return { view: { ...state.view, game }, error: null };
    }
    case 'game_over': {
      if (state.view === null) {
        return { ...state, error: null };
      }
      const game = {
        ...state.view.game,
        position: event.position,
        request: null,
      };
      return {
        view: { ...state.view, game, position: event.position },
        error: null,
      };
    }
    case 'error':
      return { ...state, error: event.message };
  }
}

/**
 * Follows the game a link reaches over a WebSocket, reconnecting when the connection drops.
 * The server sends the full state on every connection, so nothing is lost, and it is the
 * only judge of what is asked through it.
 */
export function useGameChannel(link: string): GameChannel {
  const [{ view, error }, dispatch] = useReducer(applyEvent, DISCONNECTED);
  const [connected, setConnected] = useState(false);
  const [missing, setMissing] = useState<string | null>(null);
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
    // What the server said last, which is why it is closing if it is.
    let lastError: string | null = null;

    function connect() {
      const opened = new WebSocket(gameChannelUrl(link));
      socket.current = opened;
      opened.onmessage = (message: MessageEvent<string>) => {
        const event = JSON.parse(message.data) as GameEvent;
        lastError = event.type === 'error' ? event.message : null;
        // The server starts every connection with the game's state, so a state (rather
        // than the socket opening, or an error) shows that a connection works.
        if (event.type === 'state') {
          failures = 0;
          setConnected(true);
        }
        // Whatever the server has to say, it has answered any move we submitted.
        stopWaiting();
        dispatch(event);
      };
      opened.onclose = (closing?: CloseEvent) => {
        setConnected(false);
        // Nothing can arrive on this socket now, least of all an answer.
        stopWaiting();
        if (closing?.code === NO_SUCH_GAME) {
          setMissing(lastError ?? 'There is no such game.');
          return;
        }
        if (closing?.code === DONE || stopped) {
          return;
        }
        retryTimer = setTimeout(connect, retryDelayMs(failures));
        failures += 1;
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
  }, [link, stopWaiting]);

  /** Sends a message, and says whether the connection was there to take it. */
  const send = useCallback((message: ViewerMessage): boolean => {
    const open = socket.current;
    // A socket that is closing or closed throws nothing and reports nothing: it
    // discards what it is given. Waiting for an answer to that would never end.
    if (open === null || open.readyState !== WebSocket.OPEN) {
      return false;
    }
    open.send(JSON.stringify(message));
    return true;
  }, []);

  const submitMove = useCallback(
    (uci: string) => {
      if (!send({ type: 'move', uci })) {
        return;
      }
      setMovePending(true);
      clearTimeout(answer.current);
      answer.current = setTimeout(() => {
        stopWaiting();
        dispatch({ type: 'error', message: NO_ANSWER });
      }, ANSWER_TIMEOUT_MS);
    },
    [send, stopWaiting],
  );

  // None of these hold anything up on the board: each answers itself with what it changed,
  // or with an error, and a move made meanwhile is the server's to sort out.
  const resign = useCallback(() => void send({ type: 'resign' }), [send]);
  const abort = useCallback(() => void send({ type: 'abort' }), [send]);
  const takeBack = useCallback(() => void send({ type: 'takeback' }), [send]);
  const answerRequest = useCallback(
    (request: number, accept: boolean) => void send({ type: 'answer', request, accept }),
    [send],
  );

  return {
    view,
    error,
    connected,
    missing,
    movePending,
    submitMove,
    resign,
    abort,
    takeBack,
    answer: answerRequest,
  };
}
