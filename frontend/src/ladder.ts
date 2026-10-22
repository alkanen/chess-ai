/**
 * What the run page makes of a run's Stockfish ladders: each checkpoint's Elo with its range,
 * how it did at each level, and where in the replay viewer one of its games is.
 *
 * Games are numbered from 1 in an address, as they are on the page, and from 0 everywhere else.
 */

import type { LadderEstimate, LadderGame, LadderLevel, LadderPoint } from './api';
import type { ChartData } from './runCharts';
import type { SampleGameRef } from './sampleGames';
import { describeLimit, describeLimits } from './stockfishLimit';

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
 * The limits the ladders were played at, such as "0.1 s a move" or "100,000 nodes a move", each
 * once: ratings measured at one are not comparable with those measured at another.
 */
export function ladderLimits(points: LadderPoint[]): string[] {
  return describeLimits(points);
}

/** The series of the Elo chart, in the order `eloChart` gives them. */
export const ELO_SERIES = ['Elo', 'Range low', 'Range high', 'Below the ladder', 'Above the ladder'];

/**
 * The Elo chart's data: by step, the most likely rating and its range where there is one, and
 * a checkpoint beyond the ladder drawn where it is likely to be at the nearest, rather than at
 * a number the results do not give: at the end of the ladder it is past, or at the far end of
 * its range where that lies beyond it. One that lost almost every game to the lowest level is
 * likely far below that level, and drawn there; one that only just fell short of it, at it.
 *
 * With the points in the order of the columns, which is by step, so that an index into the
 * data names the same checkpoint in both.
 */
export function eloChart(points: LadderPoint[]): { points: LadderPoint[]; data: ChartData } {
  const sorted = [...points].sort((one, other) => one.step - other.step);
  const finite = (point: LadderPoint) => point.estimate.beyond === null;
  const below = ({ estimate }: LadderPoint) =>
    estimate.beyond !== 'below_floor'
      ? null
      : Math.min(estimate.floor, estimate.high ?? estimate.floor);
  const above = ({ estimate }: LadderPoint) =>
    estimate.beyond !== 'above_ceiling'
      ? null
      : Math.max(estimate.ceiling, estimate.low ?? estimate.ceiling);
  return {
    points: sorted,
    data: [
      sorted.map((point) => point.step),
      sorted.map((point) => (finite(point) ? point.estimate.rating : null)),
      sorted.map((point) => (finite(point) ? point.estimate.low : null)),
      sorted.map((point) => (finite(point) ? point.estimate.high : null)),
      sorted.map(below),
      sorted.map(above),
    ],
  };
}

/**
 * The ladder that speaks for the checkpoint from `step`: its own, or failing that the one
 * nearest it by step, the earlier of two as near; null when no checkpoint has one.
 */
export function nearestLadder(step: number, points: LadderPoint[]): LadderPoint | null {
  let nearest: LadderPoint | null = null;
  for (const point of points) {
    const distance = Math.abs(point.step - step);
    const best = nearest === null ? Infinity : Math.abs(nearest.step - step);
    if (distance < best || (distance === best && nearest !== null && point.step < nearest.step)) {
      nearest = point;
    }
  }
  return nearest;
}

/**
 * How strong a checkpoint played on the ladder, in words for a player choosing it, after a
 * subject such as "Step 4000": which Stockfish setting it scored about as well as, with its
 * range, and which rating it was asked to play like there. `asked` is the rating the player
 * wants it to play like, which it may well play differently at.
 *
 * Stockfish's Elo setting at a fraction of a second a move is a scale of its own, so the
 * words never call the number a rating.
 */
export function describeStrength(point: LadderPoint, asked: number | null): string {
  const round = (value: number | null) => (value === null ? '?' : String(Math.round(value)));
  const { estimate } = point;
  let strength: string;
  if (estimate.beyond === 'below_floor') {
    const top = estimate.high === null ? '' : ` (likely at most ${round(estimate.high)})`;
    strength = `below Stockfish ${round(estimate.floor)}, the ladder’s lowest level${top}`;
  } else if (estimate.beyond === 'above_ceiling') {
    const bottom = estimate.low === null ? '' : ` (likely at least ${round(estimate.low)})`;
    strength = `above Stockfish ${round(estimate.ceiling)}, the ladder’s highest level${bottom}`;
  } else {
    strength =
      `about as well as Stockfish set to ${round(estimate.rating)} ` +
      `(likely ${round(estimate.low)} to ${round(estimate.high)})`;
  }
  const rating =
    point.rating === null
      ? 'asked to claim no rating'
      : `asked to play like a player rated ${point.rating}`;
  const sentences = [
    `played ${strength} at ${describeLimit(point)} on the Stockfish ladder, ${rating}.`,
    'That is Stockfish’s own scale, not a Lichess rating.',
  ];
  if (asked !== point.rating) {
    sentences.push(
      `You have asked for ${asked === null ? 'no rating' : asked}, so it may play differently.`,
    );
  }
  return sentences.join(' ');
}
