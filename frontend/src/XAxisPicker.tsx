import { useId, useState } from 'react';
import { X_AXES, type XAxisId } from './runCharts';
import './XAxisPicker.css';

const STORAGE_KEY = 'chess-ai.chart-x-axis';

function storedAxis(): XAxisId {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return X_AXES.find((axis) => axis.id === stored)?.id ?? X_AXES[0].id;
  } catch {
    // Storage this browser will not let the page use; the default does.
    return X_AXES[0].id;
  }
}

/**
 * Which x-axis the charts are drawn against, remembered in this browser: whoever compares
 * runs by positions seen wants the next run they open charted the same way.
 */
export function useXAxis(): [XAxisId, (axis: XAxisId) => void] {
  const [axis, setAxis] = useState<XAxisId>(storedAxis);
  return [
    axis,
    (next: XAxisId) => {
      setAxis(next);
      try {
        window.localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // Not remembered, which costs a click next time and nothing else.
      }
    },
  ];
}

interface XAxisPickerProps {
  value: XAxisId;
  onChange: (axis: XAxisId) => void;
}

/** A choice of what the charts' x-axis measures: steps, positions seen or wall time. */
export function XAxisPicker({ value, onChange }: XAxisPickerProps) {
  const name = useId();
  return (
    <fieldset className="x-axis-picker">
      <legend>Chart against</legend>
      {X_AXES.map((axis) => (
        <label key={axis.id}>
          <input
            type="radio"
            name={name}
            value={axis.id}
            checked={value === axis.id}
            onChange={() => onChange(axis.id)}
          />
          {axis.label}
        </label>
      ))}
    </fieldset>
  );
}
