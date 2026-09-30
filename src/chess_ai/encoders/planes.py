"""The board-planes encoder: a plane per piece type and colour, plus a global feature vector.

Two parts, for the reason the PRD gives: what is laid out on the board goes on the board, and
what is not — whose turn it is, what may still be castled, how good the players are — goes in a
flat vector beside it. Baking the globals into planes would force every architecture to read
them as geometry, and a rating is not a property of a square.

By default the planes are absolute: white's pieces are white's wherever they are, and rank 0 of
the array is rank 1 of the board. Two options change what is shown, and both default to off:

- ``history`` adds the positions the game passed through on its way here, as further sets of
  piece planes behind the current one, most recent first. A game that has not yet played that
  many moves has nothing to show, and those planes are all zero.
- ``orientation = "side-to-move"`` turns the board around whenever black is to move, so the
  mover's pieces are always on the first six planes and always play up the array. That is three
  things done together or not at all: the planes are mirrored, the castling features become the
  mover's and the opponent's, and the moves the model predicts are mirrored with them — see
  :meth:`BoardPlanesEncoder.model_moves`. Mirror only some of them and the global features
  quietly disagree with the board they describe.

Everything is done over a whole batch of dataset records at once. This runs in data-loader
workers on every position of every epoch, so it is vectorised numpy throughout: no ``chess.Board``
is built, and the packed board in each record is unpacked with shifts and masks.
"""

from typing import Final

import chess
import numpy as np

from chess_ai.dataset import POSITION_DTYPE, PositionFlags, live_history
from chess_ai.dataset.records import BOARD_BYTES, NO_SQUARE
from chess_ai.encoders.registry import ENCODERS
from chess_ai.encoders.spec import BOARD_SIZE, EncoderSpec, InputBundle
from chess_ai.move_codec import MIRRORED_INDEX, VOCABULARY_SIZE

NAME: Final = "board-planes"
"""The name this encoder is chosen by in an experiment config."""

PIECE_PLANES: Final = 12
"""One per piece type and colour, indexed by the packed piece code minus one.

The codes are python-chess's piece types, white 1-6 and black 7-12, so plane 0 is white's
pawns and plane 11 is black's king. See :func:`chess_ai.dataset.records.pack_board`.
"""

SQUARES: Final = BOARD_SIZE * BOARD_SIZE

ABSOLUTE: Final = "absolute"
"""The orientation that shows every position from white's side of the board."""

SIDE_TO_MOVE: Final = "side-to-move"
"""The orientation that shows every position from the side of whoever is to move."""

ORIENTATIONS: Final = (ABSOLUTE, SIDE_TO_MOVE)

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
planes are — and like the planes they follow the orientation: from the side to move, the two
named for white are the mover's and the two named for black are the opponent's. ``en_passant``
is 1.0 only when the capture can actually be played, which is what the dataset records. The
ratings are the mover's and their opponent's, each divided by the rating scale and 0.0 when the
game did not say, with the unknown flags saying which is which — a rating of 0.0 with the flag
clear cannot happen, so the network can tell missing from low.
"""

_CASTLING_BITS: Final = (
    PositionFlags.WHITE_KINGSIDE,
    PositionFlags.WHITE_QUEENSIDE,
    PositionFlags.BLACK_KINGSIDE,
    PositionFlags.BLACK_QUEENSIDE,
)

_OTHER_COLOUR: Final = np.array(
    [0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 13, 14, 15], dtype=np.uint8
)
"""Each packed piece code as the same piece of the other colour, indexed by the code."""


@ENCODERS.register(NAME)
class BoardPlanesEncoder:
    """Dataset position records to spatial planes and global features."""

    def __init__(
        self,
        *,
        rating_scale: float = DEFAULT_RATING_SCALE,
        history: int = 0,
        orientation: str = ABSOLUTE,
    ) -> None:
        if rating_scale <= 0:
            raise ValueError(f"rating_scale must be positive, not {rating_scale}")
        if isinstance(history, bool) or not isinstance(history, int) or history < 0:
            raise ValueError(f"history must be a whole number of positions, not {history!r}")
        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {', '.join(ORIENTATIONS)}, not {orientation!r}"
            )
        self.rating_scale = float(rating_scale)
        self.history = history
        self.orientation = orientation

    @property
    def spec(self) -> EncoderSpec:
        """The shapes this encoder produces, which is what a model is built from."""
        options: dict[str, float | int | str] = {"rating_scale": self.rating_scale}
        # Recorded only when they are set. A checkpoint is loaded by rebuilding its encoder and
        # comparing specs, so an option written down at its default would make every spec from
        # before the option existed look like a different encoder.
        if self.history:
            options["history"] = self.history
        if self._oriented:
            options["orientation"] = self.orientation
        return EncoderSpec(
            encoder=NAME,
            spatial_channels=PIECE_PLANES * (self.history + 1),
            board_size=BOARD_SIZE,
            global_features=len(GLOBAL_FEATURES),
            policy_size=VOCABULARY_SIZE,
            options=options,
        )

    @property
    def _oriented(self) -> bool:
        return self.orientation == SIDE_TO_MOVE

    def encode(self, positions: np.ndarray) -> InputBundle:
        """Encode a batch of :data:`~chess_ai.dataset.POSITION_DTYPE` records.

        Takes the records as they come out of a split reader, in the order given, and returns
        arrays whose first axis is the batch. A two-dimensional array is each position followed
        by the ones before it in its game, as :meth:`~chess_ai.dataset.SplitReader.position_history`
        returns them; a flat one is positions whose history is not known, which encode as
        positions with none.
        """
        positions = np.asarray(positions, dtype=POSITION_DTYPE)
        frames = positions if positions.ndim == 2 else positions.reshape(-1, 1)
        current = frames[:, 0]
        turned = self._turned(current)
        return InputBundle(
            spatial=self._planes(frames, turned), globals=self._globals(current, turned)
        )

    def encode_board(
        self,
        board: chess.Board,
        *,
        mover_rating: int | None = None,
        opponent_rating: int | None = None,
    ) -> InputBundle:
        """Encode one position being played, as a batch of one.

        Through records and :meth:`encode`, rather than reading the board directly: the two
        paths then cannot drift apart, and a position played here is encoded as the same
        position out of a dataset would be. The history is read off the moves ``board`` has had
        played on it, so a board set up from a FEN has none, exactly as a dataset's game that
        began from one has none.
        """
        return self.encode(
            live_history(
                board, self.history, mover_rating=mover_rating, opponent_rating=opponent_rating
            )
        )

    def model_moves(self, moves: np.ndarray, white_to_move: np.ndarray | bool) -> np.ndarray:
        """Move indices as played on the board, as the indices the model predicts them by.

        The same indices unless the board is oriented from the side to move: then black's moves
        are shown to the model mirrored, as the board was, so that a pawn push is the same move
        whichever colour makes it.
        """
        moves = np.asarray(moves, dtype=np.int64)
        if not self._oriented:
            return moves
        return np.where(white_to_move, moves, MIRRORED_INDEX[moves]).astype(np.int64)

    def board_moves(self, moves: np.ndarray, white_to_move: np.ndarray | bool) -> np.ndarray:
        """The model's move indices as moves on the real board: :meth:`model_moves` undone.

        Mirroring is its own inverse, so this is the same mapping, under the name that says
        which way it is being used.
        """
        return self.model_moves(moves, white_to_move)

    def _turned(self, current: np.ndarray) -> np.ndarray:
        """Which examples are shown turned around: black's moves, when oriented from the mover."""
        if not self._oriented:
            return np.zeros(len(current), dtype=bool)
        return (current["flags"] & PositionFlags.WHITE_TO_MOVE) == 0

    def _planes(self, frames: np.ndarray, turned: np.ndarray) -> np.ndarray:
        """The piece planes, as (batch, (history + 1) * :data:`PIECE_PLANES`, 8, 8) ``uint8``.

        The current position's twelve planes first, then each earlier position's, most recent
        first. A history frame the batch does not hold, or holds as a blank record, is twelve
        planes of zeros: an empty packed board scatters nothing.
        """
        batch = len(frames)
        shown = frames[:, : self.history + 1]
        codes = _piece_codes(shown["board"].reshape(-1, BOARD_BYTES)).reshape(batch, -1, SQUARES)
        if turned.any():
            # Every frame of an example turns with its current position, earlier ones included:
            # the mover's pieces stay on the same planes all the way back through the history.
            codes[turned] = _turn(codes[turned])
        planes = np.zeros((batch, (self.history + 1) * PIECE_PLANES, SQUARES), dtype=np.uint8)
        # Only the occupied squares, scattered in one pass: a board holds at most 32 pieces, so
        # this touches a twentieth of what a plane-by-plane comparison would.
        example, frame, square = np.nonzero(codes)
        plane = frame * PIECE_PLANES + codes[example, frame, square] - 1
        planes[example, plane, square] = 1
        return planes.reshape(batch, -1, BOARD_SIZE, BOARD_SIZE)

    def _globals(self, positions: np.ndarray, turned: np.ndarray) -> np.ndarray:
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
        if turned.any():
            # The board was turned, so the castling rights turn with it: black's pair is now
            # the pair that describes the pieces at the bottom.
            out[turned, 1:5] = out[turned][:, [3, 4, 1, 2]]
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


def _turn(codes: np.ndarray) -> np.ndarray:
    """Piece codes of shape (..., 64) as the same boards seen from the other side.

    The colours swap and the ranks reverse, which is what :meth:`chess.Board.mirror` does: a1
    becomes a8 and files stay where they are, so kingside is still kingside.
    """
    ranks = _OTHER_COLOUR[codes].reshape(*codes.shape[:-1], BOARD_SIZE, BOARD_SIZE)
    return ranks[..., ::-1, :].reshape(codes.shape)
