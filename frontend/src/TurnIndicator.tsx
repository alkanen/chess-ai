import type { Color } from './api';
import './TurnIndicator.css';

interface TurnIndicatorProps {
  turn: Color;
}

export function TurnIndicator({ turn }: TurnIndicatorProps) {
  return (
    <p className="turn-indicator" role="status">
      <span className={`turn-swatch ${turn}`} aria-hidden="true" />
      {turn === 'white' ? 'White' : 'Black'} to move
    </p>
  );
}
