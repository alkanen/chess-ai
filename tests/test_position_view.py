import chess

from chess_ai.position_view import snapshot


def test_starting_position():
    view = snapshot(chess.Board())

    assert view.fen == chess.STARTING_FEN
    assert view.turn == "white"
    assert len(view.pieces) == 32
    assert view.pieces["e1"].model_dump() == {"color": "white", "type": "king"}
    assert view.pieces["d8"].model_dump() == {"color": "black", "type": "queen"}
    assert view.pieces["g1"].model_dump() == {"color": "white", "type": "knight"}
    assert view.pieces["c7"].model_dump() == {"color": "black", "type": "pawn"}
    assert "e4" not in view.pieces


def test_after_a_move_black_is_to_move():
    board = chess.Board()
    board.push_san("e4")

    view = snapshot(board)

    assert view.turn == "black"
    assert view.pieces["e4"].model_dump() == {"color": "white", "type": "pawn"}
    assert "e2" not in view.pieces
