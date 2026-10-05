/**
 * The y-range and ticks of a chart with a logarithmic y-axis.
 *
 * uPlot's own log range rounds out to the powers of ten around the data, so a loss between
 * 1.44 and 1.52 is drawn from 1 to 10 however far the chart is zoomed in. These fit the data
 * instead, and tick a range within one power of ten at round values that can still be read.
 */

/** How much is added above and below the data, as a fraction of its span in log space. */
const PADDING = 0.1;

/** The least span in powers of ten a range is given, for data that is flat or nearly so. */
const LEAST_SPAN = 0.02;

/** How many ticks an axis should have at the least and at the most. */
const FEWEST_TICKS = 3;
const MOST_TICKS = 6;

/**
 * The y-range for data from `min` to `max` on a logarithmic axis: the data with a margin
 * either side, padded in log space. Null for a chart with nothing in view, as uPlot has it.
 */
export function logRange(
  min: number | null,
  max: number | null,
): [number | null, number | null] {
  if (min == null || max == null || !(min > 0) || !(max > 0)) {
    return [null, null];
  }
  const low = Math.log10(min);
  const high = Math.log10(max);
  const span = Math.max(high - low, LEAST_SPAN);
  const middle = (low + high) / 2;
  const pad = span * PADDING;
  return [10 ** (Math.min(low, middle - span / 2) - pad), 10 ** (Math.max(high, middle + span / 2) + pad)];
}

/**
 * Where to put ticks between `min` and `max` on a logarithmic axis. A range of several powers
 * of ten is ticked at the powers themselves, a narrower one at 1, 2 and 5 times them too, and
 * one too narrow for even those at evenly spaced round values, such as 1.44, 1.46, ….
 */
export function logTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > min)) {
    return [];
  }
  for (const mantissas of [[1], [1, 2, 5]]) {
    const ticks: number[] = [];
    for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power += 1) {
      for (const mantissa of mantissas) {
        const tick = round(mantissa * 10 ** power);
        if (tick >= min && tick <= max) {
          ticks.push(tick);
        }
      }
    }
    if (ticks.length >= FEWEST_TICKS) {
      return ticks;
    }
  }
  const step = roundStep((max - min) / MOST_TICKS);
  const ticks: number[] = [];
  for (let tick = round(Math.ceil(round(min / step)) * step); tick <= max; tick = round(tick + step)) {
    ticks.push(tick);
  }
  return ticks;
}

/**
 * The smallest of 1, 2, 2.5 or 5 times a power of ten that is at least `raw`. None is more
 * than twice the one before, so a range given at most six ticks gets three at the least.
 */
function roundStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find((mantissa) => mantissa * power >= raw * (1 - 1e-9))!;
  return step * power;
}

/** `value` without the noise of float arithmetic, such as 1.4600000000000002 as 1.46. */
function round(value: number): number {
  return Number(value.toPrecision(12));
}
