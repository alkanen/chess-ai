import type { Access, Color, PendingRequest } from './api';

const SIDES: Record<Color, string> = { white: 'White', black: 'Black' };

const ASKED: Record<PendingRequest['kind'], string> = {
  takeback: 'to take back their last move',
  abort: 'to abort the game, which deletes it',
};

interface RequestPanelProps {
  request: PendingRequest;
  access: Access;
  disabled: boolean;
  /** Agrees to, or declines, the request named by its id. */
  onAnswer: (request: number, accept: boolean) => void;
}

/**
 * What one person has asked the other: for the one asked, with the answers to give; for the
 * one who asked, that the answer is awaited; for anyone watching, what is going on.
 */
export function RequestPanel({ request, access, disabled, onAnswer }: RequestPanelProps) {
  const by = SIDES[request.by];
  if (access === request.by) {
    const what = request.kind === 'takeback' ? 'take back your last move' : 'abort the game';
    return (
      <p className="request" role="status">
        You asked to {what}. Waiting for your opponent to answer.
      </p>
    );
  }
  if (access === 'white' || access === 'black') {
    return (
      <div className="request" role="group" aria-label="Request">
        <p>
          {by} asks {ASKED[request.kind]}.
        </p>
        <button type="button" disabled={disabled} onClick={() => onAnswer(request.id, true)}>
          Agree
        </button>
        <button type="button" disabled={disabled} onClick={() => onAnswer(request.id, false)}>
          Decline
        </button>
      </div>
    );
  }
  return (
    <p className="request" role="status">
      {by} has asked {ASKED[request.kind]}.
    </p>
  );
}
