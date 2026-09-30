"""The board-planes encoder, against positions whose planes and features can be read by eye."""

import chess
import numpy as np
import pytest
from training_helpers import playout, record, records

from chess_ai.dataset import RatingSource, Result, TimeControl
from chess_ai.encoders import (
    DEFAULT_RATING_SCALE,
    GLOBAL_FEATURES,
    PIECE_PLANES,
    BoardPlanesEncoder,
    create_encoder,
    encoder_names,
)
from chess_ai.move_codec import VOCABULARY_SIZE, legal_mask, move_index
from chess_ai.registry import RegistryError

WHITE_PAWNS = 0
WHITE_KNIGHTS = 1
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


def test_a_live_position_encodes_as_the_same_position_out_of_a_dataset():
    """The point of encoding a board through a record: play sees what training saw."""
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    encoder = create_encoder("board-planes")

    live = encoder.encode_board(board, mover_rating=1600, opponent_rating=1800)
    stored = encoder.encode(record(board, mover_rating=1600, opponent_rating=1800))

    assert np.array_equal(live.spatial, stored.spatial)
    assert np.array_equal(live.globals, stored.globals)


def test_a_live_position_is_one_example_shaped_as_the_spec_says():
    encoder = create_encoder("board-planes")

    bundle = encoder.encode_board(chess.Board())

    assert bundle.spatial.shape == (1, *encoder.spec.spatial_shape)
    assert bundle.globals.shape == (1, encoder.spec.global_features)
    assert bundle.sequence is None


def test_a_live_position_with_no_rating_says_so_rather_than_guessing():
    """The caller of a game has no ratings to hand, and a guess would be a feature."""
    encoder = create_encoder("board-planes")

    unknown = dict(
        zip(GLOBAL_FEATURES, encoder.encode_board(chess.Board()).globals[0].tolist(), strict=True)
    )

    assert (unknown["mover_rating"], unknown["mover_rating_unknown"]) == (0.0, 1.0)
    assert (unknown["opponent_rating"], unknown["opponent_rating_unknown"]) == (0.0, 1.0)


def test_only_the_rating_that_was_given_is_known():
    encoder = create_encoder("board-planes")

    half = dict(
        zip(
            GLOBAL_FEATURES,
            encoder.encode_board(chess.Board(), mover_rating=2000).globals[0].tolist(),
            strict=True,
        )
    )

    assert half["mover_rating"] == pytest.approx(2000 / DEFAULT_RATING_SCALE)
    assert (half["mover_rating_unknown"], half["opponent_rating_unknown"]) == (0.0, 1.0)


def played(*moves: str) -> chess.Board:
    """The board after ``moves`` from the opening position, with those moves behind it."""
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    return board


def test_history_adds_a_set_of_piece_planes_per_earlier_position():
    encoder = create_encoder("board-planes", history=3)

    assert encoder.spec.spatial_channels == 4 * PIECE_PLANES
    assert encoder.encode_board(chess.Board()).spatial.shape == (1, 4 * PIECE_PLANES, 8, 8)


@pytest.mark.parametrize("history", [-1, 1.5, True, "2"])
def test_a_history_that_is_not_a_number_of_positions_is_refused(history):
    with pytest.raises(ValueError, match="history must be a whole number"):
        create_encoder("board-planes", history=history)


def test_an_orientation_that_does_not_exist_says_which_do():
    with pytest.raises(ValueError, match="absolute, side-to-move.*'mover'"):
        create_encoder("board-planes", orientation="mover")


def test_the_spec_records_the_new_options_only_when_they_are_set():
    """So that a spec written before the options existed still describes the same encoder."""
    assert create_encoder("board-planes").spec.options == {"rating_scale": DEFAULT_RATING_SCALE}
    assert create_encoder("board-planes", history=0, orientation="absolute").spec == (
        create_encoder("board-planes").spec
    )
    assert create_encoder("board-planes", history=2, orientation="side-to-move").spec.options == {
        "rating_scale": DEFAULT_RATING_SCALE,
        "history": 2,
        "orientation": "side-to-move",
    }


def test_the_history_planes_are_the_positions_the_game_came_through_most_recent_first():
    """1. e4 e5 2. Nf3, read by eye: each frame back undoes one more move."""
    board = played("e4", "e5", "Nf3")

    planes = create_encoder("board-planes", history=2).encode_board(board).spatial[0]
    now, before_nf3, before_e5 = planes.reshape(3, PIECE_PLANES, 8, 8)

    # Now: the knight is on f3, and both e-pawns have moved.
    assert now[WHITE_KNIGHTS][2][5] == 1 and now[WHITE_KNIGHTS][0][6] == 0
    assert now[WHITE_PAWNS][3][4] == 1 and now[BLACK_PAWNS][4][4] == 1
    # One ply back: the knight is still on g1, and the pawns are where they are now.
    assert before_nf3[WHITE_KNIGHTS][0][6] == 1 and before_nf3[WHITE_KNIGHTS][2][5] == 0
    assert before_nf3[WHITE_PAWNS][3][4] == 1 and before_nf3[BLACK_PAWNS][4][4] == 1
    # Two plies back: black has not yet answered 1. e4.
    assert before_e5[WHITE_KNIGHTS][0][6] == 1
    assert before_e5[WHITE_PAWNS][3][4] == 1
    assert before_e5[BLACK_PAWNS][6][4] == 1 and before_e5[BLACK_PAWNS][4][4] == 0
    assert [frame.sum() for frame in (now, before_nf3, before_e5)] == [32, 32, 32]


def test_each_history_frame_is_that_position_encoded_on_its_own():
    board = played("d4", "Nf6", "c4", "e6", "Nc3", "Bb4")
    plain = create_encoder("board-planes")

    planes = create_encoder("board-planes", history=4).encode_board(board).spatial[0]

    earlier = board.copy()
    for frame in planes.reshape(5, PIECE_PLANES, 8, 8):
        assert np.array_equal(frame, plain.encode_board(earlier).spatial[0])
        earlier.pop()


def test_history_from_before_the_game_began_is_zero():
    board = played("e4")

    planes = create_encoder("board-planes", history=3).encode_board(board).spatial[0]
    now, opening, *before_the_game = planes.reshape(4, PIECE_PLANES, 8, 8)

    assert now[WHITE_PAWNS][3][4] == 1
    assert np.array_equal(
        opening, create_encoder("board-planes").encode(record(chess.Board())).spatial[0]
    )
    assert all(frame.sum() == 0 for frame in before_the_game)


def test_a_position_set_up_from_a_fen_has_no_history_to_show():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")

    planes = create_encoder("board-planes", history=2).encode_board(board).spatial[0]

    assert planes[:PIECE_PLANES].sum() == 32
    assert planes[PIECE_PLANES:].sum() == 0


def test_records_given_without_their_history_encode_as_positions_with_none():
    encoder = create_encoder("board-planes", history=2)

    bundle = encoder.encode(records(chess.Board(), played("e4")))

    assert bundle.spatial.shape == (2, *encoder.spec.spatial_shape)
    assert bundle.spatial[:, PIECE_PLANES:].sum() == 0
    assert np.array_equal(
        bundle.spatial[:, :PIECE_PLANES],
        create_encoder("board-planes").encode(records(chess.Board(), played("e4"))).spatial,
    )


def test_history_leaves_the_global_features_alone():
    board = played("e4", "e5", "Nf3")

    with_history = create_encoder("board-planes", history=2).encode_board(board, mover_rating=1700)
    without = create_encoder("board-planes").encode_board(board, mover_rating=1700)

    assert np.array_equal(with_history.globals, without.globals)


def oriented(**options):
    return create_encoder("board-planes", orientation="side-to-move", **options)


def sampled() -> list[chess.Board]:
    """Positions from random playouts, with castling rights, en passant and both movers."""
    return [board for seed in range(12) for board in playout(seed, plies=60)]


def test_white_to_move_is_shown_as_it_is():
    board = played("e4", "e5")

    turned, absolute = (
        oriented().encode_board(board),
        create_encoder("board-planes").encode_board(board),
    )

    assert np.array_equal(turned.spatial, absolute.spatial)
    assert np.array_equal(turned.globals, absolute.globals)


def test_black_to_move_is_shown_with_the_mover_at_the_bottom_on_the_movers_planes():
    """After 1. e4, by eye: black's pieces are "white's", and white's e4 pawn is on "e5"."""
    planes = oriented().encode_board(played("e4")).spatial[0]

    assert planes[WHITE_PAWNS, 1].tolist() == [1] * 8, "the mover's pawns, on the second rank"
    assert planes[WHITE_KING][0][4] == 1, "the mover's king on e1"
    assert planes[BLACK_KING][7][4] == 1
    assert planes[BLACK_PAWNS][4][4] == 1, "the opponent's e-pawn, two squares down from rank 7"
    assert planes[BLACK_PAWNS, 6].tolist() == [1, 1, 1, 1, 0, 1, 1, 1]
    assert planes.sum() == 32


def test_the_files_stay_where_they_are_so_kingside_is_still_kingside():
    # Black to move, with a rook on h8 and nothing on the a-file.
    board = chess.Board("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")

    bundle = oriented().encode(record(board))
    named = dict(zip(GLOBAL_FEATURES, bundle.globals[0].tolist(), strict=True))

    assert bundle.spatial[0][3][0][7] == 1, "the mover's rook on h1, not a1"
    assert (named["white_kingside"], named["white_queenside"]) == (1.0, 0.0), "the mover's pair"
    assert (named["black_kingside"], named["black_queenside"]) == (0.0, 0.0)
    assert named["side_to_move"] == 0.0, "which colour is really moving is still said"


def test_the_oriented_encoding_is_the_absolute_encoding_of_the_mirrored_board():
    """Planes and castling features together: python-chess's mirror is the reference."""
    absolute = create_encoder("board-planes")
    side_to_move = GLOBAL_FEATURES.index("side_to_move")
    black_to_move = [
        board for board in sampled() if board.turn == chess.BLACK and any(board.legal_moves)
    ]
    assert any(board.has_castling_rights(chess.BLACK) for board in black_to_move)

    for board in black_to_move:
        turned = oriented().encode(record(board))
        mirrored = absolute.encode(record(board.mirror()))

        assert np.array_equal(turned.spatial, mirrored.spatial), board.fen()
        assert np.array_equal(
            np.delete(turned.globals, side_to_move, axis=1),
            np.delete(mirrored.globals, side_to_move, axis=1),
        ), board.fen()


def test_turning_the_board_twice_gives_back_the_board():
    from chess_ai.encoders.planes import _piece_codes, _turn

    codes = _piece_codes(records(*sampled()[:50])["board"])

    assert not np.array_equal(_turn(codes), codes)
    assert np.array_equal(_turn(_turn(codes)), codes)


def test_mirroring_the_moves_is_an_involution():
    encoder = oriented()
    every_move = np.arange(VOCABULARY_SIZE)

    for white_to_move in (True, False):
        there = encoder.model_moves(every_move, white_to_move)
        assert np.array_equal(encoder.board_moves(there, white_to_move), every_move)
    assert np.array_equal(encoder.model_moves(every_move, True), every_move)
    assert not np.array_equal(encoder.model_moves(every_move, False), every_move)


def test_the_mirrored_moves_are_the_legal_moves_of_the_board_the_model_was_shown():
    """The codec and the planes have to turn together, or the policy points at the wrong board."""
    encoder = oriented()

    for board in sampled():
        shown = board if board.turn == chess.WHITE else board.mirror()
        legal = np.flatnonzero(legal_mask(board))

        in_model_terms = encoder.model_moves(legal, board.turn == chess.WHITE)

        assert set(in_model_terms.tolist()) == set(np.flatnonzero(legal_mask(shown)).tolist())


def test_moves_are_mirrored_per_position_in_a_mixed_batch():
    encoder = oriented()
    e7e5, e2e4 = (move_index(chess.Move.from_uci(uci)) for uci in ("e7e5", "e2e4"))

    mirrored = encoder.model_moves(np.array([e2e4, e7e5]), np.array([True, False]))

    assert mirrored.tolist() == [e2e4, e2e4], "a double push of the e-pawn, whoever plays it"
    assert mirrored.dtype == np.int64


def test_an_absolute_encoder_leaves_the_moves_alone():
    encoder = create_encoder("board-planes")
    every_move = np.arange(VOCABULARY_SIZE)

    assert np.array_equal(encoder.model_moves(every_move, False), every_move)
    assert np.array_equal(encoder.board_moves(every_move, False), every_move)


def test_the_history_turns_with_the_position_it_belongs_to():
    """After 1. e4 black is to move, and the opening position behind it is shown turned too."""
    planes = oriented(history=1).encode_board(played("e4")).spatial[0]
    now, opening = planes.reshape(2, PIECE_PLANES, 8, 8)

    assert now[BLACK_PAWNS][4][4] == 1, "white's pawn on e4, seen from black's side"
    assert opening[WHITE_PAWNS, 1].tolist() == [1] * 8, "black's pawns at the bottom"
    assert opening[BLACK_PAWNS, 6].tolist() == [1] * 8, "and white's, unmoved, at the top"
    assert opening[WHITE_KING][0][4] == 1
