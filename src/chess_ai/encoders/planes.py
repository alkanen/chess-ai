"""The board-planes encoder: a plane per piece type and colour, plus a global feature vector.

Two parts, for the reason the PRD gives: what is laid out on the board goes on the board, and
what is not — whose turn it is, what may still be castled, how good the players are — goes in a
flat vector beside it. Baking the globals into planes would force every architecture to read
them as geometry, and a rating is not a property of a square.

The planes are absolute: white's pieces are white's wherever they are, and rank 0 of the array
is rank 1 of the board. Orienting the board from the side to move is an option of its own, and
it has to mirror the castling features and the move codec along with the planes, which is why it
is a separate slice of work rather than a flag here.

Everything is done over a whole batch of dataset records at once. This runs in data-loader
workers on every position of every epoch, so it is vectorised numpy throughout: no ``chess.Board``
is built, and the packed board in each record is unpacked with shifts and masks.
"""

from typing import Final

import chess
import numpy as np

from chess_ai.dataset import POSITION_DTYPE, PositionFlags, live_position
from chess_ai.dataset.records import NO_SQUARE
from chess_ai.encoders.registry import ENCODERS
from chess_ai.encoders.spec import BOARD_SIZE, EncoderSpec, InputBundle
from chess_ai.move_codec import VOCABULARY_SIZE

NAME: Final = "board-planes"
"""The name this encoder is chosen by in an experiment config."""

PIECE_PLANES: Final = 12
"""One per piece type and colour, indexed by the packed piece code minus one.

The codes are python-chess's piece types, white 1-6 and black 7-12, so plane 0 is white's
pawns and plane 11 is black's king. See :func:`chess_ai.dataset.records.pack_board`.
"""

SQUARES: Final = BOARD_SIZE * BOARD_SIZE

DEFAULT_RATING_SCALE: Final = 5000.0
"""What a rating is divided by, per the PRD: well above today's top ratings, so a rating of
3000 is 0.6 and there is room left for the pools to inflate without the feature leaving
the range the network was trained on."""

HALFMOVE_CLOCK_SCALE: Final = 100.0
"""What the halfmove clock is divided by: 1.0 is the fifty-move draw, which is the only
value on that counter the rules attach a meaning to."""

GLOBAL_FEATURES: Final = (
    "side_to_move",
    "white_kingside",
    "white_queenside",
    "black_kingside",
    "black_queenside",
    "en_passant",
    "halfmove_clock",
    "mover_rating",
    "opponent_rating",
    "mover_rating_unknown",
    "opponent_rating_unknown",
)
"""The global vector's features, in order. Part of the input contract: a checkpoint trained
against this order reads a differently ordered vector as nonsense, so appending to it is a
change of encoder rather than an addition to this one.

``side_to_move`` is 1.0 when white is to move. The castling features are absolute, as the
planes are. ``en_passant`` is 1.0 only when the capture can actually be played, which is what
the dataset records. The ratings are the mover's and their opponent's, each divided by the
rating scale and 0.0 when the game did not say, with the unknown flags saying which is which —
a rating of 0.0 with the flag clear cannot happen, so the network can tell missing from low.
"""

_CASTLING_BITS: Final = (
    PositionFlags.WHITE_KINGSIDE,
    PositionFlags.WHITE_QUEENSIDE,
    PositionFlags.BLACK_KINGSIDE,
    PositionFlags.BLACK_QUEENSIDE,
)


@ENCODERS.register(NAME)
class BoardPlanesEncoder:
    """Dataset position records to spatial planes and global features."""

    def __init__(self, *, rating_scale: float = DEFAULT_RATING_SCALE) -> None:
        if rating_scale <= 0:
            raise ValueError(f"rating_scale must be positive, not {rating_scale}")
        self.rating_scale = float(rating_scale)

    @property
    def spec(self) -> EncoderSpec:
        """The shapes this encoder produces, which is what a model is built from."""
        return EncoderSpec(
            encoder=NAME,
            spatial_channels=PIECE_PLANES,
            board_size=BOARD_SIZE,
            global_features=len(GLOBAL_FEATURES),
            policy_size=VOCABULARY_SIZE,
            options={"rating_scale": self.rating_scale},
        )

    def encode(self, positions: np.ndarray) -> InputBundle:
        """Encode a batch of :data:`~chess_ai.dataset.POSITION_DTYPE` records.

        Takes the records as they come out of a split reader, in the order given, and returns
        arrays whose first axis is the batch.
        """
        positions = np.asarray(positions, dtype=POSITION_DTYPE).reshape(-1)
        return InputBundle(spatial=self._planes(positions), globals=self._globals(positions))

    def encode_board(
        self,
        board: chess.Board,
        *,
        mover_rating: int | None = None,
        opponent_rating: int | None = None,
    ) -> InputBundle:
        """Encode one position being played, as a batch of one.

        Through a record and :meth:`encode`, rather than reading the board directly: the two
        paths then cannot drift apart, and a position played here is encoded as the same
        position out of a dataset would be.
        """
        return self.encode(
            live_position(board, mover_rating=mover_rating, opponent_rating=opponent_rating)
        )

    def _planes(self, positions: np.ndarray) -> np.ndarray:
        """The piece planes, as (batch, :data:`PIECE_PLANES`, 8, 8) ``float32``."""
        codes = _piece_codes(positions["board"])
        planes = np.zeros((len(positions), PIECE_PLANES, SQUARES), dtype=np.float32)
        # Only the occupied squares, scattered in one pass: a board holds at most 32 pieces, so
        # this touches a twentieth of what a plane-by-plane comparison would.
        example, square = np.nonzero(codes)
        planes[example, codes[example, square] - 1, square] = 1.0
        return planes.reshape(len(positions), PIECE_PLANES, BOARD_SIZE, BOARD_SIZE)

    def _globals(self, positions: np.ndarray) -> np.ndarray:
        """The global features, as (batch, ``len(GLOBAL_FEATURES)``) ``float32``."""
        flags = positions["flags"]
        out = np.empty((len(positions), len(GLOBAL_FEATURES)), dtype=np.float32)
        out[:, 0] = (flags & PositionFlags.WHITE_TO_MOVE) != 0
        for offset, bit in enumerate(_CASTLING_BITS, start=1):
            out[:, offset] = (flags & bit) != 0
        out[:, 5] = positions["en_passant"] != NO_SQUARE
        out[:, 6] = positions["halfmove_clock"] / HALFMOVE_CLOCK_SCALE
        for offset, (rating, bit) in enumerate(
            (
                ("mover_rating", PositionFlags.MOVER_RATING_UNKNOWN),
                ("opponent_rating", PositionFlags.OPPONENT_RATING_UNKNOWN),
            ),
            start=7,
        ):
            unknown = (flags & bit) != 0
            out[:, offset] = np.where(unknown, 0.0, positions[rating] / self.rating_scale)
            out[:, offset + 2] = unknown
        return out


def _piece_codes(packed: np.ndarray) -> np.ndarray:
    """The packed boards as (batch, 64) piece codes, 0 where a square is empty.

    A record holds a nibble per square with a1 in the low nibble of the first byte, so the low
    nibbles are the even-numbered squares and the high nibbles the odd-numbered ones. Square
    ``rank * 8 + file`` lands at ``[rank, file]`` once reshaped, so the array's first board axis
    is the rank counting up from white's.
    """
    codes = np.empty((len(packed), SQUARES), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes
