import { describe, expect, it } from 'vitest';
import {
  describeEstimate,
  describeScore,
  describeStrength,
  eloChartData,
  ladderGameHash,
  ladderGameInHash,
  nearestLadder,
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

  it('takes a checkpoint’s own ladder, or else the nearest, the earlier of two as near', () => {
    const points = [ladderPoint(2000), ladderPoint(4000), ladderPoint(8000)];

    expect(nearestLadder(4000, points)?.step).toBe(4000);
    expect(nearestLadder(5000, points)?.step).toBe(4000);
    expect(nearestLadder(6000, points)?.step).toBe(4000);
    expect(nearestLadder(7000, points)?.step).toBe(8000);
    expect(nearestLadder(9000, points)?.step).toBe(8000);
    expect(nearestLadder(4000, [])).toBeNull();
  });

  it('describes a strength on Stockfish’s scale, at the rating the ladder asked for', () => {
    expect(describeStrength(ladderPoint(4000), 2000)).toBe(
      'played about as well as Stockfish set to 1420 (likely 1360 to 1481) at 0.1 s a move ' +
        'on the Stockfish ladder, asked to play like a player rated 2000. ' +
        'That is Stockfish’s own scale, not a Lichess rating.',
    );
  });

  it('says a strength may not hold at a rating other than the ladder’s', () => {
    expect(describeStrength(ladderPoint(4000), 1500)).toMatch(
      /rated 2000\. .* You have asked for 1500, so it may play differently\.$/,
    );
    expect(describeStrength(ladderPoint(4000), null)).toMatch(
      /You have asked for no rating, so it may play differently\.$/,
    );
    const unrated = { ...ladderPoint(4000), rating: null };
    expect(describeStrength(unrated, null)).toMatch(/asked to claim no rating\. That is [^.]*\.$/);
    expect(describeStrength(unrated, 2000)).toMatch(/You have asked for 2000/);
  });

  it('describes a strength past either end of the ladder', () => {
    expect(
      describeStrength(
        ladderPoint(4000, { rating: null, low: null, high: 1402.2, beyond: 'below_floor' }),
        2000,
      ),
    ).toMatch(/^played below Stockfish 1350, the ladder’s lowest level \(likely at most 1402\) at/);
    expect(
      describeStrength(
        ladderPoint(4000, { rating: null, low: null, high: null, beyond: 'below_floor' }),
        2000,
      ),
    ).toMatch(/^played below Stockfish 1350, the ladder’s lowest level at/);
    expect(
      describeStrength(
        ladderPoint(4000, { rating: null, low: 1600, high: null, beyond: 'above_ceiling' }),
        2000,
      ),
    ).toMatch(/^played above Stockfish 1500, the ladder’s highest level \(likely at least 1600\) at/);
  });
});
