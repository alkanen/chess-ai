import { describe, expect, it } from 'vitest';
import { describeLimit, describeLimits } from './stockfishLimit';

describe('describeLimit', () => {
  it('says the time a move, or the nodes a move when Stockfish was held to those', () => {
    expect(describeLimit({ move_time: 0.1 })).toBe('0.1 s a move');
    expect(describeLimit({ move_time: 0.1, nodes: null })).toBe('0.1 s a move');
    expect(describeLimit({ move_time: null, nodes: 100_000 })).toBe('100,000 nodes a move');
  });

  it('lists each limit once, times before nodes and the least of each first', () => {
    expect(
      describeLimits([
        { move_time: null, nodes: 200_000 },
        { move_time: 0.2 },
        { move_time: null, nodes: 50_000 },
        { move_time: 0.1, nodes: null },
        { move_time: 0.2 },
      ]),
    ).toEqual(['0.1 s a move', '0.2 s a move', '50,000 nodes a move', '200,000 nodes a move']);
  });
});
