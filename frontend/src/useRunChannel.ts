import { useEffect, useReducer, useState } from 'react';
import { runChannelUrl, type Heartbeat, type MetricsRecord, type RunEvent, type RunInfo } from './api';

/** One run, as far as the run channel has told it. */
export interface RunView {
  /** What the run is, or null while its run.json cannot be read. */
  info: RunInfo | null;
  /** Where it has got to, or null before it has written a heartbeat. */
  heartbeat: Heartbeat | null;
  /** Whether it says it is running and has stopped saying so. */
  stale: boolean;
  /** Every line of its metrics log so far, oldest first. */
  metrics: MetricsRecord[];
}

export interface RunChannel {
  /** Null until the server has said what the run is. */
  run: RunView | null;
  /** Why the server will not stream this run, such as there being no run of that name. */
  error: string | null;
  /** Whether the current connection has delivered the run, so what is shown is live. */
  connected: boolean;
}

interface ChannelState {
  run: RunView | null;
  error: string | null;
}

const MAX_RETRY_DELAY_MS = 10_000;

const NOTHING: ChannelState = { run: null, error: null };

function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** failures, MAX_RETRY_DELAY_MS);
}

function applyEvent(state: ChannelState, event: RunEvent | 'reset'): ChannelState {
  if (event === 'reset') {
    return NOTHING;
  }
  switch (event.type) {
    case 'run': {
      const metrics = state.run?.metrics ?? [];
      const { info, heartbeat, stale } = event;
      return { run: { info, heartbeat, stale, metrics }, error: null };
    }
    case 'metrics': {
      const run = state.run ?? { info: null, heartbeat: null, stale: false, metrics: [] };
      const metrics = event.reset ? event.records : [...run.metrics, ...event.records];
      return { run: { ...run, metrics }, error: null };
    }
    case 'error':
      return { ...state, error: event.message };
  }
}

/**
 * Follows one training run over a WebSocket, reconnecting when the connection drops. The
 * server starts every connection with the whole run, so a reconnect loses nothing. A run the
 * server refuses to stream is not asked for again.
 */
export function useRunChannel(name: string): RunChannel {
  const [{ run, error }, dispatch] = useReducer(applyEvent, NOTHING);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    let stopped = false;
    let refused = false;
    let socket: WebSocket | null = null;
    // Another run is another history; nothing of the one before it carries over.
    dispatch('reset');

    function connect() {
      const opened = new WebSocket(runChannelUrl(name));
      socket = opened;
      opened.onmessage = (message: MessageEvent<string>) => {
        const event = JSON.parse(message.data) as RunEvent;
        if (event.type === 'error') {
          // The server will not stream this run, such as one that is not there, and closes
          // the connection after saying so. Asking again would only be told the same thing.
          refused = true;
        } else {
          // Only the run itself shows that the connection works and the backoff can restart.
          failures = 0;
          setConnected(true);
        }
        dispatch(event);
      };
      opened.onclose = () => {
        setConnected(false);
        if (!stopped && !refused) {
          retryTimer = setTimeout(connect, retryDelayMs(failures));
          failures += 1;
        }
      };
    }

    connect();
    return () => {
      stopped = true;
      clearTimeout(retryTimer);
      socket?.close();
    };
  }, [name]);

  return { run, error, connected };
}
