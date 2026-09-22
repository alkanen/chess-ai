// The cburnett piece set; see pieces/cburnett/LICENSE.md.
import bB from './pieces/cburnett/bB.svg';
import bK from './pieces/cburnett/bK.svg';
import bN from './pieces/cburnett/bN.svg';
import bP from './pieces/cburnett/bP.svg';
import bQ from './pieces/cburnett/bQ.svg';
import bR from './pieces/cburnett/bR.svg';
import wB from './pieces/cburnett/wB.svg';
import wK from './pieces/cburnett/wK.svg';
import wN from './pieces/cburnett/wN.svg';
import wP from './pieces/cburnett/wP.svg';
import wQ from './pieces/cburnett/wQ.svg';
import wR from './pieces/cburnett/wR.svg';
import type { Color, Piece, PieceType } from '../api';

const IMAGES: Record<Color, Record<PieceType, string>> = {
  white: { king: wK, queen: wQ, rook: wR, bishop: wB, knight: wN, pawn: wP },
  black: { king: bK, queen: bQ, rook: bR, bishop: bB, knight: bN, pawn: bP },
};

export function pieceImage(piece: Piece): string {
  return IMAGES[piece.color][piece.type];
}
