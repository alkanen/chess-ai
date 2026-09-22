"""Position view: turns a position into the snapshot the browser renders.

The snapshot is the only representation of a position the browser receives; the
browser has no rules engine of its own.
"""

from typing import Literal

import chess
from pydantic import BaseModel

Color = Literal["white", "black"]
PieceType = Literal["pawn", "knight", "bishop", "rook", "queen", "king"]
Result = Literal["1-0", "0-1", "1/2-1/2"]
GameOverReason = Literal[
    "checkmate",
    "stalemate",
    "insufficient_material",
    "threefold_repetition",
    "fifty_move_rule",
]


class Piece(BaseModel):
    color: Color
    type: PieceType


class LastMove(BaseModel):
    from_square: str
    to_square: str
    """Where the moving piece landed; for castling, the king's destination ("g1")."""


class GameOver(BaseModel):
    result: Result
    reason: GameOverReason


class PositionSnapshot(BaseModel):
    fen: str
    turn: Color
    """The side to move."""
    pieces: dict[str, Piece]
    """Occupied squares, keyed by square name ("e4")."""
    last_move: LastMove | None
    """The move that led to this position, if the board's history has one."""
    game_over: GameOver | None
    """Set when the rules end the game in this position."""


def _color(color: chess.Color) -> Color:
    return "white" if color == chess.WHITE else "black"


def snapshot(board: chess.Board) -> PositionSnapshot:
    """Describe ``board``, using its move history for the last move and repetitions."""
    pieces = {
        chess.square_name(square): Piece(
            color=_color(piece.color),
            type=chess.piece_name(piece.piece_type),  # type: ignore[arg-type]
        )
        for square, piece in board.piece_map().items()
    }
    last_move = None
    if board.move_stack:
        move = board.peek()
        last_move = LastMove(
            from_square=chess.square_name(move.from_square),
            to_square=chess.square_name(move.to_square),
        )
    return PositionSnapshot(
        fen=board.fen(),
        turn=_color(board.turn),
        pieces=pieces,
        last_move=last_move,
        game_over=_game_over(board),
    )


def _game_over(board: chess.Board) -> GameOver | None:
    """The game's end in this position, claiming repetition and fifty-move draws at once.

    The draws are claimed when they are on the board: the position has occurred for the
    third time, or 50 moves have passed without a capture or pawn move. python-chess's
    ``claim_draw`` also ends the game one move early when a draw *could* be reached with
    the next move, which would leave viewers looking for a repetition that isn't there.
    """
    if board.is_checkmate():
        winner = not board.turn
        return GameOver(result="1-0" if winner == chess.WHITE else "0-1", reason="checkmate")
    if board.is_stalemate():
        return GameOver(result="1/2-1/2", reason="stalemate")
    if board.is_insufficient_material():
        return GameOver(result="1/2-1/2", reason="insufficient_material")
    if board.is_fifty_moves():
        return GameOver(result="1/2-1/2", reason="fifty_move_rule")
    if board.is_repetition(3):
        return GameOver(result="1/2-1/2", reason="threefold_repetition")
    return None
