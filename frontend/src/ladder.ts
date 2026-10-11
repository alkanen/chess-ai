/**
 * What the run page makes of a run's Stockfish ladders: each checkpoint's Elo with its range,
 * how it did at each level, and where in the replay viewer one of its games is.
 *
 * Games are numbered from 1 in an address, as they are on the page, and from 0 everywhere else.
 */

import type { LadderEstimate, LadderGame, LadderLevel, LadderPoint } from './api';
import type { ChartData } from './runCharts';
import type { SampleGameRef } from './sampleGames';

/** What the ladder is called among a run's evaluation suites. */
export const LADDER_SUITE = 'stockfish-ladder';

/** One ladder game of one checkpoint of a run: a run, a step and a game, as a sample game is. */
export type LadderGameRef = SampleGameRef;

const LADDER_GAME = /^#replay\/ladder\/([^/]+)\/(\d+)\/(\d+)$/;

/** The address of one ladder game in the replay viewer. */
export function ladderGameHash({ run, step, index }: LadderGameRef): string {
  return `#replay/ladder/${encodeURIComponent(run)}/${step}/${index + 1}`;
}

/** The ladder game an address opens in the replay viewer, or null for any other address. */
export function ladderGameInHash(hash: string): LadderGameRef | null {
  const found = LADDER_GAME.exec(hash);
  if (found === null || Number(found[3]) < 1) {
    return null;
  }
  return {
    run: decodeURIComponent(found[1]),
    step: Number(found[2]),
    index: Number(found[3]) - 1,
  };
}

/** The estimate in a few words, such as "1523 (1460 to 1590)" or "below 1350, at most 1420". */
export function describeEstimate(estimate: LadderEstimate): string {
  const round = (value: number | null) => (value === null ? '?' : String(Math.round(value)));
  if (estimate.beyond === 'below_floor') {
    const top = estimate.high === null ? '' : `, at most ${round(estimate.high)}`;
    return `below ${round(estimate.floor)}${top}`;
  }
  if (estimate.beyond === 'above_ceiling') {
    const bottom = estimate.low === null ? '' : `, at least ${round(estimate.low)}`;
    return `above ${round(estimate.ceiling)}${bottom}`;
  }
  return `${round(estimate.rating)} (${round(estimate.low)} to ${round(estimate.high)})`;
}

/** The checkpoint's score against a level, such as "12.5/20 (+10 =5 −5)". */
export function describeScore(level: LadderLevel): string {
  const games = level.wins + level.draws + level.losses;
  const points = level.wins + level.draws / 2;
  return `${points}/${games} (+${level.wins} =${level.draws} −${level.losses})`;
}

/** Which side the checkpoint played, against which level. */
export function describeLadderOpponent(game: Pick<LadderGame, 'elo' | 'model_color'>): string {
  return `Stockfish ${game.elo}, checkpoint as ${game.model_color === 'white' ? 'White' : 'Black'}`;
}

/**
 * The move times the ladders were played at, fastest first: ratings measured at one are not
 * comparable with those measured at another.
 */
export function moveTimes(points: LadderPoint[]): number[] {
  return [...new Set(points.map((point) => point.move_time))].sort((one, other) => one - other);
}

/** The series of the Elo chart, in the order `eloChartData` gives them. */
export const ELO_SERIES = ['Elo', 'Range low', 'Range high', 'Below the ladder', 'Above the ladder'];

/**
 * The Elo chart's data: by step, the most likely rating and its range where there is one, and
 * a checkpoint beyond the ladder drawn at the end of it it is past, rather than at a number the
 * results do not give.
 */
export function eloChartData(points: LadderPoint[]): ChartData {
  const sorted = [...points].sort((one, other) => one.step - other.step);
  const finite = (point: LadderPoint) => point.estimate.beyond === null;
  return [
    sorted.map((point) => point.step),
    sorted.map((point) => (finite(point) ? point.estimate.rating : null)),
    sorted.map((point) => (finite(point) ? point.estimate.low : null)),
    sorted.map((point) => (finite(point) ? point.estimate.high : null)),
    sorted.map((point) => (point.estimate.beyond === 'below_floor' ? point.estimate.floor : null)),
    sorted.map((point) =>
      point.estimate.beyond === 'above_ceiling' ? point.estimate.ceiling : null,
    ),
  ];
}
