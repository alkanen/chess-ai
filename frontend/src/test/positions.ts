import type { PositionSnapshot } from '../api';
import positions from './fixtures/positions.json';

/**
 * Hand-picked positions, exactly as the server describes them. A pytest test keeps the
 * fixture in step with the snapshot of each position's FEN.
 */
export const { castling, drawnByFiftyMoves, enPassant, pin, promotion } =
  positions as unknown as Record<
    'castling' | 'drawnByFiftyMoves' | 'enPassant' | 'pin' | 'promotion',
    PositionSnapshot
  >;
