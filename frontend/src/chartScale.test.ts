import { describe, expect, it } from 'vitest';
import { logRange, logTicks } from './chartScale';

describe('logRange', () => {
  it('fits losses within one power of ten closely rather than out to 1 and 10', () => {
    const [low, high] = logRange(1.44, 1.52);
    expect(low).toBeGreaterThan(1.42);
    expect(low).toBeLessThan(1.44);
    expect(high).toBeGreaterThan(1.52);
    expect(high).toBeLessThan(1.54);
  });

  it('fits a gradient norm of 0.2 to 0.35 rather than widening it to 1', () => {
    const [low, high] = logRange(0.2, 0.35);
    expect(low).toBeGreaterThan(0.18);
    expect(high).toBeLessThan(0.4);
  });

  it('pads evenly in log space, so a range over decades keeps a margin at both ends', () => {
    const [low, high] = logRange(0.01, 10);
    expect(Math.log10(0.01) - Math.log10(low!)).toBeCloseTo(0.3);
    expect(Math.log10(high!) - Math.log10(10)).toBeCloseTo(0.3);
  });

  it('gives flat data some height around its value', () => {
    const [low, high] = logRange(2, 2);
    expect(low).toBeLessThan(2);
    expect(high).toBeGreaterThan(2);
    expect(high! / low!).toBeLessThan(1.1);
  });

  it('has no range for a chart with nothing in view', () => {
    expect(logRange(null, null)).toEqual([null, null]);
  });
});

describe('logTicks', () => {
  it('ticks a range within one power of ten at round values', () => {
    expect(logTicks(1.43, 1.53)).toEqual([1.44, 1.46, 1.48, 1.5, 1.52]);
    // At the ends of the range too, where float arithmetic could put them just outside it.
    expect(logTicks(1.44, 1.52)).toEqual([1.44, 1.46, 1.48, 1.5, 1.52]);
  });

  it('ticks a range over several powers of ten at the powers', () => {
    expect(logTicks(0.005, 20)).toEqual([0.01, 0.1, 1, 10]);
  });

  it('adds 2 and 5 times the powers to a range of about one power of ten', () => {
    expect(logTicks(0.15, 3)).toEqual([0.2, 0.5, 1, 2]);
  });

  it('ticks a zoomed range at three to six values', () => {
    expect(logTicks(1.81, 1.925)).toEqual([1.82, 1.84, 1.86, 1.88, 1.9, 1.92]);
    for (const [min, max] of [
      [1.9, 5.5],
      [0.18, 0.4],
      [1.4401, 1.4409],
      [1.0, 1.1],
      [1.0, 1.24],
      [3, 30],
    ]) {
      const ticks = logTicks(min, max);
      expect(ticks.length).toBeGreaterThanOrEqual(3);
      expect(ticks.length).toBeLessThanOrEqual(6);
    }
  });
});
