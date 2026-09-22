import type { GameOver, GameOverReason, PositionSnapshot } from './api';
import { TurnIndicator } from './TurnIndicator';
import './GameStatus.css';

const REASONS: Record<GameOverReason, string> = {
  checkmate: 'checkmate',
  stalemate: 'stalemate',
  insufficient_material: 'insufficient material',
  threefold_repetition: 'threefold repetition',
  fifty_move_rule: 'the fifty-move rule',
};

const SCORES = { '1-0': '1–0', '0-1': '0–1', '1/2-1/2': '½–½' } as const;

export function describeGameOver({ result, reason }: GameOver): string {
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
