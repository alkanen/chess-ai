/** How Stockfish is held to a strength of play beyond its Elo: the time or the nodes a move. */

import { formatCount } from './runFormat';

/** What a Stockfish side, or a ladder's level, was held to for each move. */
export interface StockfishLimit {
  /** Seconds a move, or null when it was held to `nodes` instead. */
  move_time: number | null;
  /** Positions searched a move, or null (or missing, from before it could be) when it was held
   * to `move_time` instead. */
  nodes?: number | null;
}

/** What Stockfish was held to, such as "0.1 s a move" or "100,000 nodes a move". */
export function describeLimit(limit: StockfishLimit): string {
  if (limit.nodes != null) {
    return `${formatCount(limit.nodes)} nodes a move`;
  }
  return `${limit.move_time} s a move`;
}

/**
 * The limits a set of ladders or levels were played at, each once, time limits first and the
 * shortest first, then node limits, the fewest first: ratings measured at one are not
 * comparable with those measured at another.
 */
export function describeLimits(limits: StockfishLimit[]): string[] {
  const sorted = [...limits].sort(
    (one, other) =>
      Number(one.nodes != null) - Number(other.nodes != null) ||
      (one.nodes ?? one.move_time ?? 0) - (other.nodes ?? other.move_time ?? 0),
  );
  return [...new Set(sorted.map(describeLimit))];
}
