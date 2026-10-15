import { useCallback, useRef, useState, type RefObject } from 'react';
import type uPlot from 'uplot';

/**
 * How long a finger rests on a chart, in milliseconds, before dragging it selects a range to
 * zoom into rather than scrolling the page.
 */
export const HOLD_MS = 350;

/** How far a finger may wander, in CSS pixels, while it is held and still count as resting. */
export const SLOP_PX = 10;

/** A selection narrower than this, in CSS pixels, is a finger lifted where it was held. */
const LEAST_SELECTION_PX = 2;

/**
 * Whether a chart is zoomed, as a value to draw with and as a ref the chart's callbacks read
 * when new data arrives, and a setter for both.
 */
export function useZoomed(): [boolean, RefObject<boolean>, (zoomed: boolean) => void] {
  const [zoomed, setState] = useState(false);
  const current = useRef(false);
  const setZoomed = useCallback((value: boolean) => {
    current.current = value;
    setState(value);
  }, []);
  return [zoomed, current, setZoomed];
}

/**
 * Lets a finger zoom `chart` into a range, as dragging with a mouse does, which uPlot only
 * does for a mouse: a touch drag sends no mouse events.
 *
 * A finger held still on the chart for `HOLD_MS` and then dragged selects the range it
 * crosses, drawn as uPlot draws a mouse's, and lifting it zooms the x-axis to it through the
 * same `setSelect` hook. Everything else is left to the browser: a finger that moves before
 * then scrolls the page, a tap shows the values under it (the browser sends it on as mouse
 * events), and a second finger is a pinch, which zooms the page.
 *
 * Chrome follows a tap's mouse events with a `mouseleave`, on which uPlot takes the values it
 * has just shown away again, so after a tap they are put back where it was.
 *
 * Touch events rather than pointer events, because only a touch event can stop the page
 * scrolling once the hold has started; a pointer event cannot be cancelled. Returns a
 * function that takes the listeners off again.
 */
export function touchToZoom(chart: uPlot): () => void {
  const over = chart.over;
  let timer: ReturnType<typeof setTimeout> | null = null;
  // While a finger is down: where it went down, and whether it has been held long enough.
  let start: { x: number; y: number } | null = null;
  let holding = false;
  let bounds: DOMRect | null = null;
  let from = 0;
  let selection = { left: 0, width: 0 };
  // Where a tap just was, to show the values at once the browser says the mouse has left.
  let tapped: { left: number; top: number } | null = null;

  const offset = (clientX: number) =>
    Math.min(Math.max(clientX - bounds!.left, 0), bounds!.width);

  const reset = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (holding) {
      chart.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
      over.classList.remove('touch-selecting');
    }
    start = null;
    holding = false;
  };

  const select = (clientX: number, clientY: number) => {
    const to = offset(clientX);
    selection = { left: Math.min(from, to), width: Math.abs(to - from) };
    chart.setSelect({ ...selection, top: 0, height: bounds!.height }, false);
    chart.setCursor({ left: to, top: clientY - bounds!.top });
  };

  const hold = () => {
    timer = null;
    if (start === null) {
      return;
    }
    holding = true;
    bounds = over.getBoundingClientRect();
    from = offset(start.x);
    // The class widens the empty selection to a line under the finger, to say it is held.
    over.classList.add('touch-selecting');
    select(start.x, start.y);
  };

  const touchStart = (event: TouchEvent) => {
    reset();
    tapped = null;
    if (event.touches.length === 1) {
      const touch = event.touches[0];
      start = { x: touch.clientX, y: touch.clientY };
      timer = setTimeout(hold, HOLD_MS);
    }
  };

  const touchMove = (event: TouchEvent) => {
    if (start === null) {
      return;
    }
    if (event.touches.length !== 1) {
      reset();
      return;
    }
    const touch = event.touches[0];
    if (holding) {
      event.preventDefault();
      select(touch.clientX, touch.clientY);
    } else if (Math.hypot(touch.clientX - start.x, touch.clientY - start.y) > SLOP_PX) {
      // A swipe: the page scrolls.
      reset();
    }
  };

  const touchEnd = (event: TouchEvent) => {
    if (holding) {
      // No mouse events after it, which would show the values where the finger was lifted.
      event.preventDefault();
      if (selection.width >= LEAST_SELECTION_PX) {
        const { left, width } = selection;
        chart.setSelect({ left, width, top: 0, height: bounds!.height });
        chart.setScale('x', {
          min: chart.posToVal(left, 'x'),
          max: chart.posToVal(left + width, 'x'),
        });
      }
    } else if (start !== null && event.touches.length === 0) {
      const { left, top } = over.getBoundingClientRect();
      tapped = { left: start.x - left, top: start.y - top };
    }
    reset();
  };

  // Added after uPlot's own listener, so it runs after uPlot has taken the values away.
  const mouseLeave = () => {
    if (tapped !== null) {
      chart.setCursor(tapped);
      tapped = null;
    }
  };

  // A long press is the browser's too: it would open a menu over the chart.
  const contextMenu = (event: Event) => {
    if (start !== null) {
      event.preventDefault();
    }
  };

  over.addEventListener('touchstart', touchStart, { passive: true });
  over.addEventListener('touchmove', touchMove, { passive: false });
  over.addEventListener('touchend', touchEnd);
  over.addEventListener('touchcancel', reset);
  over.addEventListener('contextmenu', contextMenu);
  over.addEventListener('mouseleave', mouseLeave);
  return () => {
    reset();
    over.removeEventListener('touchstart', touchStart);
    over.removeEventListener('touchmove', touchMove);
    over.removeEventListener('touchend', touchEnd);
    over.removeEventListener('touchcancel', reset);
    over.removeEventListener('contextmenu', contextMenu);
    over.removeEventListener('mouseleave', mouseLeave);
  };
}
