"""Position view: turns a position into the snapshot the browser renders.

The snapshot is the only representation of a position the browser receives; the
browser has no rules engine of its own.
"""

from typing import Literal

import chess
from pydantic import BaseModel

Color = Literal["white", "black"]
PieceType = Literal["pawn", "knight", "bishop", "rook", "queen", "king"]


class Piece(BaseModel):
    color: Color
    type: PieceType


class PositionSnapshot(BaseModel):
    fen: str
    turn: Color
    """The side to move."""
    pieces: dict[str, Piece]
    """Occupied squares, keyed by square name ("e4")."""


def _color(color: chess.Color) -> Color:
    return "white" if color == chess.WHITE else "black"


def snapshot(board: chess.Board) -> PositionSnapshot:
    """Describe ``board`` for display."""
    pieces = {
        chess.square_name(square): Piece(
            color=_color(piece.color),
            type=chess.piece_name(piece.piece_type),  # type: ignore[arg-type]
        )
        for square, piece in board.piece_map().items()
    }
    return PositionSnapshot(fen=board.fen(), turn=_color(board.turn), pieces=pieces)
