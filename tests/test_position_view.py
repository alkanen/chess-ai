import chess
import pytest

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


def play(board: chess.Board, moves: str) -> chess.Board:
    for san in moves.split():
        board.push_san(san)
    return board


def test_last_move():
    assert snapshot(chess.Board()).last_move is None

    view = snapshot(play(chess.Board(), "e4"))

    assert view.last_move is not None
    assert (view.last_move.from_square, view.last_move.to_square) == ("e2", "e4")


def test_castling_last_move_is_the_kings_two_square_move():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_san("O-O")

    last_move = snapshot(board).last_move

    assert last_move is not None
    assert (last_move.from_square, last_move.to_square) == ("e1", "g1")


def test_a_game_in_progress_is_not_over():
    assert snapshot(chess.Board()).game_over is None
    assert snapshot(play(chess.Board(), "e4 e5 Qh5 Nc6 Bc4 Nf6")).game_over is None


@pytest.mark.parametrize(
    ("fen", "result", "reason"),
    [
        # Fool's mate.
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "0-1", "checkmate"),
        ("R5k1/5ppp/8/8/8/8/8/6K1 b - - 1 1", "1-0", "checkmate"),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "1/2-1/2", "stalemate"),
        ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", "1/2-1/2", "insufficient_material"),
        ("8/8/4k3/8/8/4K3/8/4B3 b - - 0 1", "1/2-1/2", "insufficient_material"),
        ("8/8/4k3/8/8/4K3/4R3/8 b - - 100 80", "1/2-1/2", "fifty_move_rule"),
    ],
)
def test_game_over(fen, result, reason):
    game_over = snapshot(chess.Board(fen)).game_over

    assert game_over is not None
    assert (game_over.result, game_over.reason) == (result, reason)


def test_fifty_move_rule_needs_the_full_hundred_half_moves():
    board = chess.Board("8/8/4k3/8/8/4K3/4R3/8 w - - 99 80")
    assert snapshot(board).game_over is None

    board.push_san("Ra2")

    game_over = snapshot(board).game_over
    assert game_over is not None
    assert game_over.reason == "fifty_move_rule"


def test_checkmate_on_the_hundredth_half_move_is_checkmate():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80")
    board.push_san("Ra8#")

    game_over = snapshot(board).game_over

    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("1-0", "checkmate")


def test_threefold_repetition_ends_the_game_when_it_is_on_the_board():
    # The starting position occurs for the second time after four half-moves and for
    # the third time after eight.
    board = play(chess.Board(), "Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1")
    assert snapshot(board).game_over is None

    board.push_san("Ng8")

    game_over = snapshot(board).game_over
    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("1/2-1/2", "threefold_repetition")
