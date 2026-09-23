import type { GameOver, GameOverReason, PositionSnapshot, Result } from './api';
import { TurnIndicator } from './TurnIndicator';
import './GameStatus.css';

/** An abort is described on its own, so it is the one reason with nothing to say here. */
const REASONS: Record<Exclude<GameOverReason, 'abort'>, string> = {
  checkmate: 'checkmate',
  stalemate: 'stalemate',
  insufficient_material: 'insufficient material',
  threefold_repetition: 'threefold repetition',
  fifty_move_rule: 'the fifty-move rule',
  resignation: 'resignation',
};

const SCORES: Record<Exclude<Result, '*'>, string> = {
  '1-0': '1–0',
  '0-1': '0–1',
  '1/2-1/2': '½–½',
};

export function describeGameOver({ result, reason }: GameOver): string {
  // An abort is the one ending that leaves no result, so there is nothing to score.
  if (result === '*' || reason === 'abort') {
    return 'Game aborted';
  }
  const outcome =
    result === '1-0' ? 'White wins by' : result === '0-1' ? 'Black wins by' : 'Draw by';
  return `${outcome} ${REASONS[reason]} (${SCORES[result]})`;
}

interface GameStatusProps {
  position: PositionSnapshot;
  /** Whether a game is being shown, rather than the board before any game. */
  inGame: boolean;
}

/** Whose turn it is, or how the game ended. */
export function GameStatus({ position, inGame }: GameStatusProps) {
  if (position.game_over !== null) {
    return (
      <p className="game-status game-over" role="status">
        {describeGameOver(position.game_over)}
      </p>
    );
  }
  if (!inGame) {
    return (
      <p className="game-status" role="status">
        No game in progress
      </p>
    );
  }
  return <TurnIndicator turn={position.turn} />;
}
