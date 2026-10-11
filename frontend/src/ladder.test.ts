import { describe, expect, it } from 'vitest';
import {
  describeEstimate,
  describeScore,
  eloChartData,
  ladderGameHash,
  ladderGameInHash,
} from './ladder';
import { sampleGameInHash } from './sampleGames';
import { estimate, ladderPoint } from './test/ladder';

describe('ladder', () => {
  it('names a ladder game in an address of its own, counting games from 1', () => {
    const hash = ladderGameHash({ run: 'tiny run', step: 4000, index: 2 });

    expect(hash).toBe('#replay/ladder/tiny%20run/4000/3');
    expect(ladderGameInHash(hash)).toEqual({ run: 'tiny run', step: 4000, index: 2 });
    expect(sampleGameInHash(hash)).toBeNull();
    expect(ladderGameInHash('#replay/run/tiny/4000/3')).toBeNull();
    expect(ladderGameInHash('#replay/ladder/tiny/4000/0')).toBeNull();
  });

  it('describes an estimate, or the end of the ladder it is past', () => {
    expect(describeEstimate(estimate())).toBe('1420 (1360 to 1481)');
    expect(describeEstimate(estimate({ rating: 1420, low: null }))).toBe('1420 (? to 1481)');
    expect(
      describeEstimate(estimate({ rating: null, low: null, high: 1402.2, beyond: 'below_floor' })),
    ).toBe('below 1350, at most 1402');
    expect(
      describeEstimate(estimate({ rating: null, low: null, high: null, beyond: 'below_floor' })),
    ).toBe('below 1350');
    expect(
      describeEstimate(estimate({ rating: null, low: 1600, high: null, beyond: 'above_ceiling' })),
    ).toBe('above 1500, at least 1600');
  });

  it('describes a score against a level', () => {
    expect(
      describeScore({
        elo: 1350,
        requested_elo: 1350,
        move_time: 0.1,
        wins: 10,
        draws: 5,
        losses: 5,
      }),
    ).toBe('12.5/20 (+10 =5 −5)');
  });

  it('charts a rating with its range, and one past the ladder at its end', () => {
    const data = eloChartData([
      ladderPoint(3000, { rating: 1600, low: 1550, high: null, beyond: 'above_ceiling' }),
      ladderPoint(1000, { rating: null, low: null, high: 1400, beyond: 'below_floor' }),
      ladderPoint(2000),
    ]);

    expect(data).toEqual([
      [1000, 2000, 3000],
      [null, 1420.4, null],
      [null, 1360.2, null],
      [null, 1480.6, null],
      [1350, null, null],
      [null, null, 1500],
    ]);
  });
});
