import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { GameOver, PositionSnapshot } from './api';
import { GameStatus } from './GameStatus';
import startPosition from './test/fixtures/start-position.json';

function position(changes: Partial<PositionSnapshot>): PositionSnapshot {
  return { ...(startPosition as PositionSnapshot), ...changes };
}

describe('GameStatus', () => {
  it('shows whose turn it is during a game', () => {
    render(<GameStatus position={position({ turn: 'black' })} inGame />);

    expect(screen.getByRole('status')).toHaveTextContent('Black to move');
  });

  it('says so when there is no game', () => {
    render(<GameStatus position={position({})} inGame={false} />);

    expect(screen.getByRole('status')).toHaveTextContent('No game in progress');
  });

  it.each<[GameOver, string]>([
    [{ result: '1-0', reason: 'checkmate' }, 'White wins by checkmate (1–0)'],
    [{ result: '0-1', reason: 'checkmate' }, 'Black wins by checkmate (0–1)'],
    [{ result: '1/2-1/2', reason: 'stalemate' }, 'Draw by stalemate (½–½)'],
    [
      { result: '1/2-1/2', reason: 'insufficient_material' },
      'Draw by insufficient material (½–½)',
    ],
    [{ result: '1/2-1/2', reason: 'threefold_repetition' }, 'Draw by threefold repetition (½–½)'],
    [{ result: '1/2-1/2', reason: 'fifty_move_rule' }, 'Draw by the fifty-move rule (½–½)'],
    [{ result: '0-1', reason: 'resignation' }, 'Black wins by resignation (0–1)'],
    [{ result: '1-0', reason: 'resignation' }, 'White wins by resignation (1–0)'],
  ])('shows the result and reason of a finished game: %o', (gameOver, text) => {
    render(<GameStatus position={position({ game_over: gameOver })} inGame />);

    expect(screen.getByRole('status')).toHaveTextContent(text);
    expect(screen.getByRole('status')).not.toHaveTextContent('to move');
  });

  it('shows an aborted game as having no result at all', () => {
    const aborted = position({ game_over: { result: '*', reason: 'abort' } });

    render(<GameStatus position={aborted} inGame />);

    expect(screen.getByRole('status')).toHaveTextContent('Game aborted');
    expect(screen.getByRole('status')).not.toHaveTextContent('Draw');
  });
});
