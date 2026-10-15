import type uPlot from 'uplot';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HOLD_MS, SLOP_PX, touchToZoom } from './chartZoom';
import { FakeUPlot } from './test/fakeUPlot';

/** The plot sits 100 px in and 50 px down the page, 400 px wide and 200 px tall. */
const BOUNDS = { left: 100, top: 50, width: 400, height: 200 };

/**
 * A finger event on the chart, with the fingers down after it at `points` (client x, y).
 * Returns the event, to see whether the chart kept it from the browser.
 */
function touch(chart: FakeUPlot, type: string, ...points: [number, number][]): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, 'touches', {
    value: points.map(([clientX, clientY]) => ({ clientX, clientY })),
  });
  chart.over.dispatchEvent(event);
  return event;
}

describe('touchToZoom', () => {
  let chart: FakeUPlot;
  let selected: number;
  let detach: () => void;

  beforeEach(() => {
    vi.useFakeTimers();
    selected = 0;
    chart = new FakeUPlot(
      { width: 400, height: 200, series: [], hooks: { setSelect: [() => (selected += 1)] } },
      [[]],
      document.createElement('div'),
    );
    chart.over.getBoundingClientRect = () => ({ ...BOUNDS }) as DOMRect;
    detach = touchToZoom(chart as unknown as uPlot);
  });

  afterEach(() => {
    detach();
    vi.useRealTimers();
  });

  /** A finger goes down at (`x`, 100) and rests there until the chart takes it as held. */
  function hold(x: number) {
    touch(chart, 'touchstart', [x, 100]);
    vi.advanceTimersByTime(HOLD_MS);
  }

  it('zooms into the range a held finger drags across', () => {
    hold(200);
    touch(chart, 'touchmove', [260, 120]);
    touch(chart, 'touchmove', [350, 140]);
    const lift = touch(chart, 'touchend');

    expect(selected).toBe(1);
    expect(chart.xScale).toEqual({ min: 1000, max: 2500 });
    expect(chart.select.width).toBe(0);
    // The browser sends no mouse events after it, which would move the readout.
    expect(lift.defaultPrevented).toBe(true);
  });

  it('draws the selection while dragging, without zooming yet', () => {
    hold(300);
    const move = touch(chart, 'touchmove', [150, 120]);

    expect(chart.select).toEqual({ left: 50, width: 150, top: 0, height: 200 });
    expect(chart.cursor).toEqual({ left: 50, top: 70 });
    expect(selected).toBe(0);
    expect(chart.xScale).toBeNull();
    // The page does not scroll under the selection.
    expect(move.defaultPrevented).toBe(true);
  });

  it('keeps the selection inside the plot when the finger goes past its edge', () => {
    hold(400);
    touch(chart, 'touchmove', [900, 100]);
    touch(chart, 'touchend');

    expect(chart.xScale).toEqual({ min: 3000, max: 4000 });
  });

  it('shows a held finger as a line under it', () => {
    hold(200);

    expect(chart.over).toHaveClass('touch-selecting');
    expect(chart.cursor).toEqual({ left: 100, top: 50 });

    touch(chart, 'touchend');
    expect(chart.over).not.toHaveClass('touch-selecting');
  });

  it('lets a finger that moves before it is held scroll the page', () => {
    touch(chart, 'touchstart', [200, 100]);
    vi.advanceTimersByTime(HOLD_MS / 2);
    const swipe = touch(chart, 'touchmove', [200, 100 + SLOP_PX + 1]);
    vi.advanceTimersByTime(HOLD_MS);
    const more = touch(chart, 'touchmove', [200, 200]);
    touch(chart, 'touchend');

    expect(swipe.defaultPrevented).toBe(false);
    expect(more.defaultPrevented).toBe(false);
    expect(chart.over).not.toHaveClass('touch-selecting');
    expect(selected).toBe(0);
    expect(chart.xScale).toBeNull();
  });

  it('takes a finger that trembles a little while held as resting', () => {
    touch(chart, 'touchstart', [200, 100]);
    touch(chart, 'touchmove', [203, 104]);
    vi.advanceTimersByTime(HOLD_MS);
    touch(chart, 'touchmove', [300, 100]);
    touch(chart, 'touchend');

    expect(chart.xScale).toEqual({ min: 1000, max: 2000 });
  });

  it('leaves a tap to the browser, which shows the values under it', () => {
    touch(chart, 'touchstart', [200, 100]);
    vi.advanceTimersByTime(HOLD_MS / 2);
    const lift = touch(chart, 'touchend');
    vi.advanceTimersByTime(HOLD_MS);

    expect(lift.defaultPrevented).toBe(false);
    expect(chart.over).not.toHaveClass('touch-selecting');
  });

  it('shows the values where a tap was after the browser says the mouse has left', () => {
    touch(chart, 'touchstart', [200, 100]);
    touch(chart, 'touchend');
    // uPlot takes the values away on this; the chart is in its place by then.
    chart.cursor = { left: -10, top: -10 };
    chart.over.dispatchEvent(new Event('mouseleave'));

    expect(chart.cursor).toEqual({ left: 100, top: 50 });
  });

  it('lets the mouse leave as usual when no tap was just made', () => {
    touch(chart, 'touchstart', [200, 100]);
    touch(chart, 'touchend');
    chart.over.dispatchEvent(new Event('mouseleave'));
    chart.cursor = { left: -10, top: -10 };
    chart.over.dispatchEvent(new Event('mouseleave'));
    // A swipe is not a tap.
    touch(chart, 'touchstart', [200, 100]);
    touch(chart, 'touchmove', [200, 200]);
    touch(chart, 'touchend');
    chart.over.dispatchEvent(new Event('mouseleave'));

    expect(chart.cursor).toEqual({ left: -10, top: -10 });
  });

  it('leaves a pinch to the browser', () => {
    touch(chart, 'touchstart', [200, 100]);
    touch(chart, 'touchstart', [200, 100], [300, 100]);
    vi.advanceTimersByTime(HOLD_MS);
    const pinch = touch(chart, 'touchmove', [180, 100], [320, 100]);

    expect(pinch.defaultPrevented).toBe(false);
    expect(chart.over).not.toHaveClass('touch-selecting');
  });

  it('stops selecting when a second finger comes down', () => {
    hold(200);
    touch(chart, 'touchmove', [300, 100]);
    const pinch = touch(chart, 'touchmove', [300, 100], [350, 100]);
    touch(chart, 'touchend');

    expect(pinch.defaultPrevented).toBe(false);
    expect(chart.select.width).toBe(0);
    expect(chart.xScale).toBeNull();
  });

  it('does not zoom when a held finger is lifted where it went down', () => {
    hold(200);
    touch(chart, 'touchend');

    expect(selected).toBe(0);
    expect(chart.xScale).toBeNull();
  });

  it('drops a selection the browser cancels', () => {
    hold(200);
    touch(chart, 'touchmove', [300, 100]);
    touch(chart, 'touchcancel');
    touch(chart, 'touchend');

    expect(chart.select.width).toBe(0);
    expect(chart.xScale).toBeNull();
  });

  it('keeps the browser’s long-press menu off the chart while a finger is on it', () => {
    touch(chart, 'touchstart', [200, 100]);
    vi.advanceTimersByTime(HOLD_MS * 2);
    const held = new Event('contextmenu', { cancelable: true });
    chart.over.dispatchEvent(held);
    touch(chart, 'touchend');
    // A right click with a mouse is left alone.
    const clicked = new Event('contextmenu', { cancelable: true });
    chart.over.dispatchEvent(clicked);

    expect(held.defaultPrevented).toBe(true);
    expect(clicked.defaultPrevented).toBe(false);
  });

  it('stops listening once taken off', () => {
    touch(chart, 'touchstart', [200, 100]);
    detach();
    vi.advanceTimersByTime(HOLD_MS);
    touch(chart, 'touchmove', [300, 100]);
    touch(chart, 'touchend');

    expect(chart.over).not.toHaveClass('touch-selecting');
    expect(chart.xScale).toBeNull();
  });
});
