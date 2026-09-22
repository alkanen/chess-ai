import type { GameEvent } from '../api';
import events from './fixtures/fools-mate-events.json';

export type StateEvent = Extract<GameEvent, { type: 'state' }>;
export type MoveEvent = Extract<GameEvent, { type: 'move' }>;

/**
 * A scripted game of fool's mate, as the server sends it: the full state, then four
 * moves. A pytest test keeps the fixture in step with the server.
 */
export const [foolsMateStart, ...foolsMateMoves] = events as unknown as [
  StateEvent,
  ...MoveEvent[],
];
