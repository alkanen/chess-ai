import type { LadderEstimate, LadderPoint, LadderResult } from '../api';

/** An estimate between the levels of a 1350–1500 ladder, unless `changes` say otherwise. */
export function estimate(changes: Partial<LadderEstimate> = {}): LadderEstimate {
  return {
    rating: 1420.4,
    low: 1360.2,
    high: 1480.6,
    beyond: null,
    floor: 1350,
    ceiling: 1500,
    confidence: 0.95,
    games: 4,
    points: 2,
    ...changes,
  };
}

/** A checkpoint's ladder of two levels, two games each. */
export function ladderResult(step: number, changes: Partial<LadderResult> = {}): LadderResult {
  return {
    model: { run: 'tiny', checkpoint: step, rating: 2000, strategy: 'argmax', temperature: null },
    started: '2026-10-11T12:00:00Z',
    finished: '2026-10-11T12:10:00Z',
    openings: 'standard v1',
    games_per_level: 2,
    levels: [
      { elo: 1350, requested_elo: 1350, move_time: 0.1, wins: 1, draws: 1, losses: 0 },
      { elo: 1500, requested_elo: 1500, move_time: 0.1, wins: 0, draws: 0, losses: 2 },
    ],
    estimate: estimate(),
    games: [
      {
        index: 0,
        elo: 1350,
        model_color: 'white',
        opening: 'Italian Game',
        result: '1-0',
        termination: 'checkmate',
        plies: 41,
      },
      {
        index: 1,
        elo: 1350,
        model_color: 'black',
        opening: 'Italian Game',
        result: '1/2-1/2',
        termination: 'threefold_repetition',
        plies: 60,
      },
      {
        index: 2,
        elo: 1500,
        model_color: 'white',
        opening: 'Sicilian Defence',
        result: '0-1',
        termination: 'checkmate',
        plies: 30,
      },
      {
        index: 3,
        elo: 1500,
        model_color: 'black',
        opening: 'Sicilian Defence',
        result: '1-0',
        termination: 'checkmate',
        plies: 33,
      },
    ],
    ...changes,
  };
}

export function ladderPoint(step: number, changes: Partial<LadderEstimate> = {}): LadderPoint {
  return {
    step,
    estimate: estimate(changes),
    levels: [1350, 1500],
    games_per_level: 2,
    move_time: 0.1,
    rating: 2000,
  };
}
