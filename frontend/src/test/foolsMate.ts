import type { GameEvent } from '../api';
import events from './fixtures/fools-mate-events.json';

export type StateEvent = Extract<GameEvent, { type: 'state' }>;
export type MoveEvent = Extract<GameEvent, { type: 'move' }>;

const [start, ...moves] = events as unknown as [
  Omit<StateEvent, 'access' | 'watch' | 'updated'>,
  ...MoveEvent[],
];

/**
 * A scripted game of fool's mate, as the server sends it: the full state, then four
 * moves. A pytest test keeps the fixture in step with the server.
 *
 * The state is the game's own, which the server sends as the link that is following it sees
 * it: this one is a watch link's.
 */
export const foolsMateStart: StateEvent = {
  ...start,
  access: 'watch',
  watch: 'the-watch-link',
  updated: '2026-10-05T12:00:00Z',
};
export const foolsMateMoves: MoveEvent[] = moves;
