"""What a dataset stores per position: the packing, and getting the position back out of it."""

import random

import chess
import numpy as np
import pytest

from chess_ai.dataset.records import (
    GAME_DTYPE,
    NO_SQUARE,
    POSITION_DTYPE,
    PositionFlags,
    RatingSource,
    Result,
    TimeControl,
    position_record,
    unpack_board,
)


def packed(board: chess.Board, **overrides) -> np.void:
    """``board`` as one position record, which is what the builder writes per ply."""
    fields = dict(
        move=0,
        result=Result.DRAW,
        mover_rating=1500,
        opponent_rating=1500,
        mover_rating_known=True,
        opponent_rating_known=True,
        rating_source=RatingSource.UNKNOWN,
        time_control=TimeControl.BLITZ,
        ply=0,
    )
    return np.array([position_record(board, **{**fields, **overrides})], dtype=POSITION_DTYPE)[0]


def positions_of(seed: int, plies: int = 200) -> list[chess.Board]:
    """Positions from a random game, which between them exercise every kind of state."""
    rng = random.Random(seed)
    board = chess.Board()
    boards = [board.copy()]
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        boards.append(board.copy())
    return boards


AWKWARD = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 12 34",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "8/8/8/8/8/5k2/8/5K2 w - - 99 120",
    "8/P6k/8/8/8/8/p6K/8 w - - 0 1",
]


@pytest.mark.parametrize("fen", AWKWARD)
def test_a_position_survives_being_packed(fen):
    board = chess.Board(fen)

    assert unpack_board(packed(board)).fen() == board.fen()


def test_every_position_of_a_game_survives_being_packed():
    for seed in range(20):
        for board in positions_of(seed):
            back = unpack_board(packed(board))

            assert back.fen() == board.fen()
            assert {move.uci() for move in back.legal_moves} == {
                move.uci() for move in board.legal_moves
            }


def test_a_record_is_a_fixed_size_of_the_promised_order():
    # Memory mapping needs a fixed size, and the PRD asks for 50-100 bytes per position.
    assert 50 <= POSITION_DTYPE.itemsize <= 100
    assert POSITION_DTYPE.itemsize == 52
    assert not POSITION_DTYPE.hasobject
    # Packed, so that the file holds records rather than records and padding.
    assert POSITION_DTYPE.itemsize == sum(
        POSITION_DTYPE.fields[name][0].itemsize for name in POSITION_DTYPE.names
    )
    assert GAME_DTYPE.itemsize == sum(
        GAME_DTYPE.fields[name][0].itemsize for name in GAME_DTYPE.names
    )


def test_a_double_step_no_pawn_can_answer_records_no_en_passant():
    # python-chess keeps the square behind any double step and a FEN prints one, but a
    # square nothing can capture on changes no legal move.
    board = chess.Board()
    board.push_uci("a2a4")

    assert board.ep_square == chess.A3
    assert packed(board)["en_passant"] == NO_SQUARE

    # A double step a pawn beside it can answer is recorded, because that is a legal move.
    capturable = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    capturable.push_uci("d7d5")

    assert capturable.has_legal_en_passant()
    assert packed(capturable)["en_passant"] == chess.D6


def test_castling_rights_and_side_to_move_are_flags():
    white = packed(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"))

    assert white["flags"] & PositionFlags.WHITE_TO_MOVE
    assert white["flags"] & PositionFlags.WHITE_KINGSIDE
    assert white["flags"] & PositionFlags.BLACK_QUEENSIDE
    assert not white["flags"] & PositionFlags.WHITE_QUEENSIDE
    assert not white["flags"] & PositionFlags.BLACK_KINGSIDE

    black = packed(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b - - 0 1"))

    assert not black["flags"] & PositionFlags.WHITE_TO_MOVE


def test_a_clock_past_what_a_record_holds_is_capped():
    # No game of chess gets here; a record that cannot hold it must not wrap around to 0.
    board = chess.Board("8/8/8/8/8/5k2/8/5K2 w - - 0 1")
    board.halfmove_clock = 400

    assert packed(board)["halfmove_clock"] == 255


def test_an_unknown_rating_is_a_flag_on_the_position():
    board = chess.Board()

    known = packed(board)
    mover_unknown = packed(board, mover_rating_known=False, mover_rating=0)

    assert not known["flags"] & PositionFlags.MOVER_RATING_UNKNOWN
    assert not known["flags"] & PositionFlags.OPPONENT_RATING_UNKNOWN
    assert mover_unknown["flags"] & PositionFlags.MOVER_RATING_UNKNOWN
    assert not mover_unknown["flags"] & PositionFlags.OPPONENT_RATING_UNKNOWN


def test_what_a_position_was_told_it_is_what_it_holds():
    record = packed(
        chess.Board(),
        move=42,
        result=Result.WIN,
        mover_rating=2001,
        opponent_rating=1799,
        rating_source=RatingSource.LICHESS,
        time_control=TimeControl.RAPID,
        ply=7,
    )

    assert int(record["move"]) == 42
    assert int(record["result"]) == Result.WIN
    assert int(record["mover_rating"]) == 2001
    assert int(record["opponent_rating"]) == 1799
    assert int(record["rating_source"]) == RatingSource.LICHESS
    assert int(record["time_control"]) == TimeControl.RAPID
    assert int(record["ply"]) == 7
    assert int(record["game"]) == 0, "which game a record belongs to is the writer's to fill in"


def test_a_result_read_from_the_other_side():
    assert Result.WIN.opponent == Result.LOSS
    assert Result.LOSS.opponent == Result.WIN
    assert Result.DRAW.opponent == Result.DRAW
