import json
from pathlib import Path

import chess
import pytest

from chess_ai.position_view import InvalidFenError, board_from_fen, snapshot

FRONTEND_FIXTURES = Path(__file__).parents[1] / "frontend" / "src" / "test" / "fixtures"


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


@pytest.mark.parametrize(
    ("fen", "square"),
    [
        (chess.STARTING_FEN, None),
        ("4k3/8/8/8/8/8/8/K3R3 b - - 0 1", "e8"),  # Rook check down the e-file.
        ("4k3/8/8/8/8/8/8/K3R3 w - - 0 1", None),  # The same board, but White to move.
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "e1"),  # Fool's mate.
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", None),  # Stalemate is not check.
    ],
)
def test_check_names_the_square_of_the_king_in_check(fen, square):
    assert snapshot(chess.Board(fen)).check_square == square


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


def test_starting_position_offers_the_twenty_opening_moves():
    view = snapshot(chess.Board())

    assert sum(len(moves) for moves in view.legal_moves.values()) == 20
    assert [move.uci for move in view.legal_moves["e2"]] == ["e2e3", "e2e4"]
    assert [move.uci for move in view.legal_moves["g1"]] == ["g1f3", "g1h3"]
    assert not any(
        move.capture or move.castling or move.en_passant or move.promotion or move.check
        for moves in view.legal_moves.values()
        for move in moves
    )


def test_legal_moves_are_the_side_to_moves_only():
    view = snapshot(play(chess.Board(), "e4"))

    assert view.turn == "black"
    assert {square[1] for square in view.legal_moves} == {"7", "8"}


def test_captures_are_flagged():
    view = snapshot(play(chess.Board(), "e4 d5"))

    assert {move.to_square: move.capture for move in view.legal_moves["e4"]} == {
        "d5": True,
        "e5": False,
    }


def test_moves_that_give_check_are_flagged():
    view = snapshot(chess.Board("4k3/8/8/8/8/8/8/4KR2 w - - 0 1"))

    checks = [move.uci for moves in view.legal_moves.values() for move in moves if move.check]
    assert checks == ["f1f8"]


def test_castling_is_the_kings_two_square_move():
    moves = snapshot(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")).legal_moves["e1"]

    castling = [move for move in moves if move.castling]
    assert [(move.uci, move.to_square) for move in castling] == [("e1c1", "c1"), ("e1g1", "g1")]
    assert not any(move.capture or move.en_passant or move.promotion for move in castling)


@pytest.mark.parametrize(
    ("fen", "castles"),
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["c1", "g1"]),  # rights intact
        ("r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1", []),  # rights lost
        ("r3k2r/8/8/8/8/8/8/R2QK1NR w KQkq - 0 1", []),  # both paths blocked
        ("r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1", []),  # out of check
        ("r3k2r/8/8/8/5r2/8/8/R3K2R w KQkq - 0 1", ["c1"]),  # kingside through check
        ("r3k2r/8/8/8/6r1/8/8/R3K2R w KQkq - 0 1", ["c1"]),  # kingside into check
        ("r3k2r/8/8/8/3r4/8/8/R3K2R w KQkq - 0 1", ["g1"]),  # queenside through check
        ("r3k2r/8/8/8/2r5/8/8/R3K2R w KQkq - 0 1", ["g1"]),  # queenside into check
        # The rook's square is allowed to be attacked; only the king's path matters.
        ("r3k2r/8/8/8/1r6/8/8/R3K2R w KQkq - 0 1", ["c1", "g1"]),
    ],
)
def test_castling_is_offered_only_when_it_is_legal(fen, castles):
    moves = snapshot(chess.Board(fen)).legal_moves.get("e1", [])

    assert [move.to_square for move in moves if move.castling] == castles


def test_en_passant_is_a_flagged_capture():
    moves = snapshot(chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")).legal_moves["e5"]

    assert [(move.uci, move.capture, move.en_passant) for move in moves] == [
        ("e5d6", True, True),
        ("e5e6", False, False),
    ]


def test_en_passant_is_gone_once_the_chance_has_passed():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    board.push_san("d5")
    assert [move.uci for move in snapshot(board).legal_moves["e5"] if move.en_passant] == ["e5d6"]

    play(board, "Ke2 Ke7")

    assert not any(move.en_passant for move in snapshot(board).legal_moves["e5"])


def test_promotion_offers_every_piece_and_flags_capture_promotions():
    moves = snapshot(chess.Board("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1")).legal_moves["b7"]

    assert [(move.uci, move.to_square, move.promotion, move.capture) for move in moves] == [
        ("b7a8b", "a8", "bishop", True),
        ("b7a8n", "a8", "knight", True),
        ("b7a8q", "a8", "queen", True),
        ("b7a8r", "a8", "rook", True),
        ("b7b8b", "b8", "bishop", False),
        ("b7b8n", "b8", "knight", False),
        ("b7b8q", "b8", "queen", False),
        ("b7b8r", "b8", "rook", False),
    ]


def test_a_pinned_piece_has_no_moves_that_expose_its_king():
    view = snapshot(chess.Board("4r2k/8/8/8/4N3/8/8/4K3 w - - 0 1"))

    assert "e4" not in view.legal_moves


def test_a_pinned_piece_may_still_move_along_the_pin():
    view = snapshot(chess.Board("4r2k/8/8/8/4R3/8/8/4K3 w - - 0 1"))

    assert [move.to_square for move in view.legal_moves["e4"]] == [
        "e2",
        "e3",
        "e5",
        "e6",
        "e7",
        "e8",
    ]


def test_a_king_in_check_is_offered_only_the_moves_that_answer_it():
    view = snapshot(chess.Board("4k3/8/8/8/8/8/4q3/4K2R w K - 0 1"))

    assert {square: [move.uci for move in moves] for square, moves in view.legal_moves.items()} == {
        "e1": ["e1e2"]
    }


@pytest.mark.parametrize(
    "fen",
    [
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # checkmate
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    ],
)
def test_a_finished_game_offers_no_moves(fen):
    assert snapshot(chess.Board(fen)).legal_moves == {}


def test_frontend_position_fixtures_are_what_the_server_sends():
    """The Vitest fixtures must stay the snapshots of the positions they name."""
    fixtures = json.loads((FRONTEND_FIXTURES / "positions.json").read_text(encoding="utf-8"))

    assert set(fixtures) == {
        "blackPromotion",
        "castling",
        "check",
        "drawnByFiftyMoves",
        "enPassant",
        "pin",
        "promotion",
    }
    for name, view in fixtures.items():
        assert view == snapshot(chess.Board(view["fen"])).model_dump(mode="json"), name


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",  # black to move
        "8/8/8/8/8/5k2/6q1/7K w - - 10 60",  # an endgame, with clocks well along
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # en passant
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",  # one castling right left
    ],
)
def test_a_position_a_game_can_start_from_is_read_back_unchanged(fen):
    assert board_from_fen(fen).fen() == fen


def test_a_fen_is_read_with_the_spaces_around_it_trimmed():
    """People paste FENs, and a pasted one brings whatever was around it."""
    assert board_from_fen(f"  {chess.STARTING_FEN}\n").fen() == chess.STARTING_FEN


def test_a_fen_without_its_clocks_is_given_the_ones_a_game_starts_with():
    assert board_from_fen("4k3/8/8/8/8/8/8/4K3 w - -").fen() == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


@pytest.mark.parametrize(
    ("fen", "problem"),
    [
        ("", "not a FEN"),
        ("hello", "not a FEN"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", "not a FEN"),
        ("rnbqkbnr/pppppppp/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "not a FEN"),  # seven ranks
        ("8/8/8/8/8/8/8/8 w - - 0 1", "no pieces on the board"),
        ("4k3/8/8/8/8/8/8/8 w - - 0 1", "White has no king"),
        ("8/8/8/8/8/8/8/4K3 w - - 0 1", "Black has no king"),
        ("4k3/8/8/8/8/8/8/3KK3 w - - 0 1", "more than one king"),
        ("4k3/8/8/8/8/8/8/P3K3 w - - 0 1", "pawn stands on a back rank"),
        ("4k3/8/8/8/8/8/8/4K3 w K - 0 1", "castling rights do not match"),
        ("4k3/8/8/8/8/8/8/4K3 w - e6 0 1", "no pawn has just passed"),
        # Black is in check with White to move, so Black's last move was never legal.
        ("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1", "has just moved is left in check"),
    ],
)
def test_a_fen_that_cannot_be_played_from_says_what_is_wrong_with_it(fen, problem):
    with pytest.raises(InvalidFenError, match=problem):
        board_from_fen(fen)


def test_a_position_that_is_already_over_can_still_be_started_from():
    """A game from a finished position is a short game, not an invalid one."""
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"

    assert snapshot(board_from_fen(mate)).game_over is not None


def test_an_en_passant_square_no_pawn_can_take_on_is_dropped():
    """python-chess keeps only the en passant squares that are worth something.

    A game started from such a FEN reports the tidied one as the position it began in,
    which is the same position by every rule that decides a game.
    """
    # 1. e4 c5: Black's pawn passed c6, but no white pawn is there to take it.
    idle = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"

    assert board_from_fen(idle).fen() == idle.replace(" c6 ", " - ")
