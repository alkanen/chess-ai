import { useEffect, useReducer, useState } from 'react';
import {
  runChannelUrl,
  type EvaluationEntry,
  type LiveGame,
  type Heartbeat,
  type MetricsRecord,
  type ProbeSetVersion,
  type RunEvent,
  type RunInfo,
  type RunNotes,
} from './api';

/** One run, as far as the run channel has told it. */
export interface RunView {
  /** What the run is, or null while its run.json cannot be read. */
  info: RunInfo | null;
  /** Where it has got to, or null before it has written a heartbeat. */
  heartbeat: Heartbeat | null;
  /** Whether it says it is running and has stopped saying so. */
  stale: boolean;
  /** Its title, tags and notes, or null while they cannot be read. */
  notes: RunNotes | null;
  /** Every line of its metrics log so far, oldest first. */
  metrics: MetricsRecord[];
  /** Which suites have a result about which checkpoints, by step and suite. */
  evaluations: EvaluationEntry[];
  /** The probe set the evaluator probes with, or null when none has said. */
  probeSet: ProbeSetVersion | null;
  /** The sample game being played with one of its checkpoints, or null for none. */
  liveGame: LiveGame | null;
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

/** A run the channel has sent something about before saying what it is. */
const UNKNOWN_RUN: RunView = {
  info: null,
  heartbeat: null,
  stale: false,
  notes: null,
  metrics: [],
  evaluations: [],
  probeSet: null,
  liveGame: null,
};

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
      const evaluations = state.run?.evaluations ?? [];
      const probeSet = state.run?.probeSet ?? null;
      const liveGame = state.run?.liveGame ?? null;
      const { info, heartbeat, stale, notes } = event;
      return {
        run: { info, heartbeat, stale, notes, metrics, evaluations, probeSet, liveGame },
        error: null,
      };
    }
    case 'metrics': {
      const run = state.run ?? UNKNOWN_RUN;
      const metrics = event.reset ? event.records : [...run.metrics, ...event.records];
      return { run: { ...run, metrics }, error: null };
    }
    case 'evaluations': {
      const run = state.run ?? UNKNOWN_RUN;
      return {
        run: { ...run, evaluations: event.results, probeSet: event.current_set },
        error: null,
      };
    }
    case 'live_game': {
      const run = state.run ?? UNKNOWN_RUN;
      return { run: { ...run, liveGame: event.live }, error: null };
    }
    case 'error':
      return { ...state, error: event.message };
  }
}

/**
 * Follows the run called `name` over a WebSocket until the returned function is called,
 * reconnecting when the connection drops. The server starts every connection with the whole
 * run, so a reconnect loses nothing. A run the server refuses to stream is not asked for again.
 */
function followRun(
  name: string,
  onEvent: (event: RunEvent) => void,
  onConnected: (connected: boolean) => void,
): () => void {
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let failures = 0;
  let stopped = false;
  let refused = false;
  let socket: WebSocket | null = null;

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
        onConnected(true);
      }
      onEvent(event);
    };
    opened.onclose = () => {
      onConnected(false);
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
}

/** Follows one training run; see followRun. */
export function useRunChannel(name: string): RunChannel {
  const [{ run, error }, dispatch] = useReducer(applyEvent, NOTHING);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // Another run is another history; nothing of the one before it carries over.
    dispatch('reset');
    return followRun(name, dispatch, setConnected);
  }, [name]);

  return { run, error, connected };
}

type ChannelsAction = { type: 'reset' } | { type: 'event'; name: string; event: RunEvent };

function applyChannelsAction(
  states: Record<string, ChannelState>,
  action: ChannelsAction,
): Record<string, ChannelState> {
  if (action.type === 'reset') {
    return {};
  }
  return { ...states, [action.name]: applyEvent(states[action.name] ?? NOTHING, action.event) };
}

/** Follows several training runs at once, one connection each; in the order of `names`. */
export function useRunChannels(names: string[]): RunChannel[] {
  const [states, dispatch] = useReducer(applyChannelsAction, {});
  const [connected, setConnected] = useState<Record<string, boolean>>({});
  // As a value, so that a new array naming the same runs does not reconnect to them all.
  const key = names.join('\n');

  useEffect(() => {
    dispatch({ type: 'reset' });
    setConnected({});
    const stops = key
      .split('\n')
      .filter((name) => name !== '')
      .map((name) =>
        followRun(
          name,
          (event) => dispatch({ type: 'event', name, event }),
          (now) => setConnected((all) => ({ ...all, [name]: now })),
        ),
      );
    return () => stops.forEach((stop) => stop());
  }, [key]);

  return names.map((name) => ({
    run: states[name]?.run ?? null,
    error: states[name]?.error ?? null,
    connected: connected[name] ?? false,
  }));
}
