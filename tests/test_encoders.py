"""The board-planes encoder, against positions whose planes and features can be read by eye."""

import chess
import numpy as np
import pytest
from training_helpers import record, records

from chess_ai.dataset import RatingSource, Result, TimeControl
from chess_ai.encoders import (
    DEFAULT_RATING_SCALE,
    GLOBAL_FEATURES,
    PIECE_PLANES,
    BoardPlanesEncoder,
    create_encoder,
    encoder_names,
)
from chess_ai.move_codec import VOCABULARY_SIZE
from chess_ai.registry import RegistryError

WHITE_PAWNS = 0
WHITE_KING = 5
BLACK_PAWNS = 6
BLACK_KING = 11
"""Plane numbers, from the piece codes a record packs; see PIECE_PLANES."""


def features(board: chess.Board, **options) -> dict[str, float]:
    """The global features for ``board``, by name, which is how a test can read them."""
    bundle = create_encoder("board-planes").encode(record(board, **options))
    return dict(zip(GLOBAL_FEATURES, bundle.globals[0].tolist(), strict=True))


def test_the_encoder_is_registered_under_its_name():
    assert encoder_names() == ["board-planes"]
    assert isinstance(create_encoder("board-planes"), BoardPlanesEncoder)


def test_an_unknown_encoder_says_what_there_is():
    with pytest.raises(RegistryError, match="no encoder called 'planes'.*board-planes"):
        create_encoder("planes")


def test_an_unknown_option_is_refused_rather_than_ignored():
    with pytest.raises(RegistryError, match="rating_scale"):
        create_encoder("board-planes", rating_sale=1000.0)


def test_the_spec_describes_what_encode_actually_returns():
    encoder = create_encoder("board-planes")
    bundle = encoder.encode(records(chess.Board(), chess.Board()))

    spec = encoder.spec
    assert bundle.spatial.shape == (2, *spec.spatial_shape)
    assert bundle.globals.shape == (2, spec.global_features)
    assert bundle.sequence is None, "this encoder produces no move sequence"
    assert spec.policy_size == VOCABULARY_SIZE
    assert spec.spatial_channels == PIECE_PLANES
    assert (bundle.spatial.dtype, bundle.globals.dtype) == (np.float32, np.float32)


def test_the_spec_records_the_options_it_was_made_with():
    assert create_encoder("board-planes", rating_scale=4000.0).spec.options == {
        "rating_scale": 4000.0
    }


def test_the_opening_position_puts_every_piece_on_its_own_plane():
    planes = create_encoder("board-planes").encode(record(chess.Board())).spatial[0]

    assert planes.sum() == 32, "thirty-two pieces, each on exactly one plane"
    assert planes[WHITE_PAWNS, 1].tolist() == [1] * 8, "white's pawns on rank 2"
    assert planes[BLACK_PAWNS, 6].tolist() == [1] * 8, "black's pawns on rank 7"
    # Rank first then file, so e1 is [0][4] and e8 is [7][4].
    assert planes[WHITE_KING][0][4] == 1 and planes[WHITE_KING].sum() == 1
    assert planes[BLACK_KING][7][4] == 1 and planes[BLACK_KING].sum() == 1


def test_the_planes_are_absolute_so_the_same_pieces_stay_on_the_same_planes():
    """Black to move does not move black's pieces onto white's planes."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")

    planes = create_encoder("board-planes").encode(record(board)).spatial[0]

    assert planes[WHITE_KING][0][4] == 1, "white's king is still on white's plane on e1"
    assert planes[BLACK_KING][7][4] == 1


def test_an_empty_square_is_empty_on_every_plane():
    board = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")

    planes = create_encoder("board-planes").encode(record(board)).spatial[0]

    assert planes.sum() == 2
    assert planes[:, 3, 3].tolist() == [0] * PIECE_PLANES, "d4 holds nothing"


def test_side_to_move_and_castling_rights_are_read_off_the_position():
    white = features(chess.Board())
    assert white["side_to_move"] == 1.0
    assert (white["white_kingside"], white["white_queenside"]) == (1.0, 1.0)
    assert (white["black_kingside"], white["black_queenside"]) == (1.0, 1.0)

    # Black to move, and only black may still castle, on the queen's side only.
    black = features(chess.Board("r3k3/8/8/8/8/8/8/4K3 b q - 0 1"))
    assert black["side_to_move"] == 0.0
    assert (black["white_kingside"], black["white_queenside"]) == (0.0, 0.0)
    assert (black["black_kingside"], black["black_queenside"]) == (0.0, 1.0)


def test_en_passant_is_flagged_only_when_the_capture_can_be_played():
    available = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 3")
    assert features(available)["en_passant"] == 1.0

    # The same en-passant square, but no white pawn that could take on it.
    unavailable = chess.Board("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 3")
    assert features(unavailable)["en_passant"] == 0.0


def test_the_halfmove_clock_is_one_at_the_fifty_move_draw():
    board = chess.Board()
    board.halfmove_clock = 100

    assert features(board)["halfmove_clock"] == 1.0
    board.halfmove_clock = 25
    assert features(board)["halfmove_clock"] == 0.25


def test_ratings_are_divided_by_the_rating_scale():
    board = chess.Board()

    scaled = features(board, mover_rating=2500, opponent_rating=1000)

    assert scaled["mover_rating"] == pytest.approx(2500 / DEFAULT_RATING_SCALE)
    assert scaled["opponent_rating"] == pytest.approx(1000 / DEFAULT_RATING_SCALE)
    assert (scaled["mover_rating_unknown"], scaled["opponent_rating_unknown"]) == (0.0, 0.0)


def test_the_rating_scale_is_configurable():
    bundle = create_encoder("board-planes", rating_scale=2000.0).encode(
        record(chess.Board(), mover_rating=1000, opponent_rating=2000)
    )
    scaled = dict(zip(GLOBAL_FEATURES, bundle.globals[0].tolist(), strict=True))

    assert (scaled["mover_rating"], scaled["opponent_rating"]) == (0.5, 1.0)


def test_a_rating_scale_of_zero_is_refused():
    with pytest.raises(ValueError, match="rating_scale must be positive"):
        create_encoder("board-planes", rating_scale=0.0)


def test_an_unknown_rating_is_zero_with_its_flag_set():
    """Zero and flagged, so that "we were not told" cannot be read as "rated zero"."""
    unknown = features(
        chess.Board(),
        mover_rating=0,
        mover_rating_known=False,
        opponent_rating=1800,
    )

    assert (unknown["mover_rating"], unknown["mover_rating_unknown"]) == (0.0, 1.0)
    assert unknown["opponent_rating"] == pytest.approx(1800 / DEFAULT_RATING_SCALE)
    assert unknown["opponent_rating_unknown"] == 0.0


def test_the_features_ignore_what_is_not_a_feature():
    """The result, the rating pool and the time control are recorded but not shown."""
    board = chess.Board()

    one = features(board, result=Result.WIN, rating_source=RatingSource.FIDE)
    other = features(board, result=Result.LOSS, time_control=TimeControl.CORRESPONDENCE)

    assert one == other


def test_a_batch_encodes_each_position_in_the_order_given():
    boards = [
        chess.Board(),
        chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
    ]
    encoder = create_encoder("board-planes")

    bundle = encoder.encode(records(*boards))

    for index, board in enumerate(boards):
        alone = encoder.encode(record(board))
        assert np.array_equal(bundle.spatial[index], alone.spatial[0])
        assert np.array_equal(bundle.globals[index], alone.globals[0])
