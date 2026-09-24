"""Reading PGN back with python-chess, which is how tests check what was written out."""

import io

import chess
import chess.pgn


def read_back(pgn: str) -> chess.pgn.Game:
    """The game python-chess reads in ``pgn``, which has to be a game it can read."""
    record = chess.pgn.read_game(io.StringIO(pgn))
    assert record is not None, "python-chess found no game in the PGN"
    # Anything python-chess could not make sense of, such as a move that is not legal in
    # the position before it, is collected here rather than raised.
    assert record.errors == []
    return record


def replayed(pgn: str) -> str:
    """The position ``pgn``'s moves lead to, replayed from the position it starts in."""
    record = read_back(pgn)
    board = record.board()
    for move in record.mainline_moves():
        board.push(move)
    return board.fen()
