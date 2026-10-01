import type { RunStatus } from './api';
import { runState, STATE_DESCRIPTIONS } from './runFormat';
import './RunState.css';

const ICONS: Record<string, string> = { stale: '⚠', crashed: '✕', running: '●', finished: '✓' };

interface RunStateBadgeProps {
  status: RunStatus | null | undefined;
  stale: boolean | undefined;
}

/** A run's state as a labelled badge, never as a colour alone. */
export function RunStateBadge({ status, stale }: RunStateBadgeProps) {
  const state = runState(status, stale);
  return (
    <span className={`run-state run-state-${state}`} title={STATE_DESCRIPTIONS[state]}>
      {ICONS[state] !== undefined && (
        <span className="icon" aria-hidden="true">
          {ICONS[state]}
        </span>
      )}
      {state}
    </span>
  );
}
