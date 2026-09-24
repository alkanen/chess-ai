"""The move vocabulary, checked against python-chess over a large sample of positions."""

import random

import chess
import numpy as np
import pytest

from chess_ai.move_codec import (
    MIRRORED_INDEX,
    VOCABULARY,
    VOCABULARY_SIZE,
    legal_mask,
    mirrored_index,
    move_at,
    move_index,
)


def playout(seed: int, plies: int = 120) -> list[chess.Board]:
    """The positions a random game passes through, as a source of varied positions."""
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


@pytest.fixture(scope="module")
def sampled_positions() -> list[chess.Board]:
    """A few thousand positions from random playouts, plus hand-picked awkward ones."""
    boards = [board for seed in range(40) for board in playout(seed)]
    boards += [
        # Every promotion, including capture-promotions, for both colours.
        chess.Board("6k1/4P3/8/8/8/8/8/4K3 w - - 0 1"),
        chess.Board("3rr1k1/4P3/8/8/8/8/8/4K3 w - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/3p4/4K1RR b - - 0 1"),
        # Castling both ways for both colours, and en passant.
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"),
    ]
    return boards


def test_vocabulary_is_the_expected_size():
    # Every (from, to) pair a queen or a knight can make, plus the promotions: the PRD's
    # "on the order of 1,900 entries". The number is part of the format, so it is pinned.
    assert VOCABULARY_SIZE == 1968
    assert len(set(VOCABULARY)) == VOCABULARY_SIZE


def test_every_index_round_trips():
    for index in range(VOCABULARY_SIZE):
        assert move_index(move_at(index)) == index


def test_every_legal_move_round_trips(sampled_positions):
    moves = 0
    for board in sampled_positions:
        for move in board.legal_moves:
            assert move_at(move_index(move)) == move
            moves += 1
    # Guards the sample itself: a fixture that stopped producing positions would leave
    # every assertion above unrun.
    assert moves > 10_000


def test_castling_is_a_two_square_king_move():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    castles = {move.uci() for move in board.legal_moves if board.is_castling(move)}

    assert castles == {"e1g1", "e1c1"}
    for uci in castles:
        assert move_at(move_index(chess.Move.from_uci(uci))).uci() == uci


def test_mask_matches_python_chess_exactly(sampled_positions):
    for board in sampled_positions:
        expected = np.zeros(VOCABULARY_SIZE, dtype=bool)
        for move in board.legal_moves:
            expected[move_index(move)] = True

        mask = legal_mask(board)

        assert mask.dtype == bool
        assert mask.shape == (VOCABULARY_SIZE,)
        np.testing.assert_array_equal(mask, expected)


def test_mask_of_a_finished_game_is_empty():
    checkmated = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):  # Fool's mate.
        checkmated.push_uci(uci)

    assert checkmated.is_checkmate()
    assert not legal_mask(checkmated).any()


def test_mirroring_is_an_involution():
    twice = MIRRORED_INDEX[MIRRORED_INDEX]

    np.testing.assert_array_equal(twice, np.arange(VOCABULARY_SIZE, dtype=np.uint16))


def test_mirroring_agrees_with_the_mirrored_board(sampled_positions):
    for board in sampled_positions:
        mirrored_board = board.mirror()
        expected = {mirrored_index(move_index(move)) for move in board.legal_moves}

        assert {move_index(move) for move in mirrored_board.legal_moves} == expected


def test_mirroring_keeps_the_promotion_piece():
    promotion = chess.Move.from_uci("e7d8r")

    mirrored = move_at(mirrored_index(move_index(promotion)))

    assert mirrored == chess.Move.from_uci("e2d1r")


def test_a_move_no_piece_could_make_is_not_in_the_vocabulary():
    with pytest.raises(KeyError):
        move_index(chess.Move.from_uci("a1b4"))
