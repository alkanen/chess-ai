"""What a dataset stores per position and per game, and how it is laid out in bytes.

The records say what happened, not what a network should see: a packed board rather than
planes, a move index rather than a one-hot vector, a rating rather than a normalized
feature. That is what lets one dataset feed every encoder and every architecture, and it
is why the encoders are a separate module — nothing here knows they exist.

Each stream is a plain array of fixed-size records, which is what makes a dataset
memory-mappable: record ``i`` is at offset ``i * itemsize``, so the trainer can take a
random batch out of tens of gigabytes without reading, parsing or holding the rest. The
dtypes below are therefore packed and little-endian, and :data:`FORMAT_VERSION` changes
whenever they do.

Three streams make up a split:

- *positions*, one per ply: the position, and the move played in it
- *games*, one per game: the ratings and the rest of what the whole game says, and where
  the game's plies are
- *moves*, one per ply: just the move index, laid out end to end per game, because a
  sequence model wants a game's moves contiguous and two bytes wide rather than gathered
  out of 52-byte position records
"""

from enum import IntEnum
from typing import Final

import chess
import numpy as np

FORMAT_VERSION: Final = 1
"""The version of the on-disk layout, recorded in the manifest and checked when reading."""


class Result(IntEnum):
    """A game's result from one player's point of view."""

    LOSS = 0
    DRAW = 1
    WIN = 2

    @property
    def opponent(self) -> "Result":
        """The same game from the other player's side."""
        return Result(2 - self.value)


class RatingSource(IntEnum):
    """Which pool a game's ratings come from.

    Ratings from different pools are not comparable — an online blitz rating runs well
    above a FIDE one — so the pool is recorded per game rather than assumed per dataset,
    and an experiment can account for it.
    """

    UNKNOWN = 0
    LICHESS = 1
    CHESSCOM = 2
    FIDE = 3
    OTHER = 4


class TimeControl(IntEnum):
    """How fast the game was played, by the class its time control falls into."""

    UNKNOWN = 0
    BULLET = 1
    BLITZ = 2
    RAPID = 3
    CLASSICAL = 4
    CORRESPONDENCE = 5


class PositionFlags(IntEnum):
    """Bits of a position record's ``flags`` field."""

    WHITE_TO_MOVE = 1 << 0
    WHITE_KINGSIDE = 1 << 1
    WHITE_QUEENSIDE = 1 << 2
    BLACK_KINGSIDE = 1 << 3
    BLACK_QUEENSIDE = 1 << 4
    MOVER_RATING_UNKNOWN = 1 << 5
    OPPONENT_RATING_UNKNOWN = 1 << 6


class GameFlags(IntEnum):
    """Bits of a game record's ``flags`` field."""

    WHITE_RATING_UNKNOWN = 1 << 0
    BLACK_RATING_UNKNOWN = 1 << 1
    CUSTOM_START = 1 << 2
    """The game began somewhere other than where games begin."""


BOARD_BYTES: Final = 32
"""A nibble per square: 0 for empty, else the piece code, a1 in the low nibble first."""

NO_SQUARE: Final = 64
"""What an en-passant field holds when there is no en-passant capture to be made."""

MAX_CLOCK: Final = 255
"""The largest halfmove clock a record can hold; see :func:`position_record`."""

POSITION_DTYPE: Final = np.dtype(
    [
        ("board", np.uint8, (BOARD_BYTES,)),
        ("flags", np.uint8),
        ("en_passant", np.uint8),
        ("halfmove_clock", np.uint8),
        ("result", np.uint8),
        ("rating_source", np.uint8),
        ("time_control", np.uint8),
        ("fullmove_number", "<u2"),
        ("move", "<u2"),
        ("mover_rating", "<u2"),
        ("opponent_rating", "<u2"),
        ("ply", "<u2"),
        ("game", "<u4"),
    ],
    align=False,
)
"""One position and the move played in it, from the mover's point of view.

``result`` is the game's result for the player to move, ``mover_rating`` and
``opponent_rating`` are that player's and their opponent's, and ``move`` is the index in
the move vocabulary of the move actually played. ``game`` is the record's game, as an
index into the same split's game stream. ``ply`` counts from 0 at the game's first
position, so it is also the position's place in its game.
"""

GAME_DTYPE: Final = np.dtype(
    [
        ("ply_offset", "<u8"),
        ("ply_count", "<u2"),
        ("white_rating", "<u2"),
        ("black_rating", "<u2"),
        ("date", "<u4"),
        ("flags", np.uint8),
        ("result", np.uint8),
        ("rating_source", np.uint8),
        ("time_control", np.uint8),
        ("source", "<u4"),
    ],
    align=False,
)
"""One game: who played it how well, how it ended, and which plies are its own.

Every ply of a game becomes exactly one position and one move, so ``ply_offset`` and
``ply_count`` locate the game in both streams at once. ``result`` is from white's point
of view, ``date`` is ``yyyymmdd`` (0 when the file did not say), and ``source`` is the
game's file, as an index into the manifest's sources.
"""

MOVE_DTYPE: Final = np.dtype("<u2")
"""A move index, which is all the move stream holds."""

assert POSITION_DTYPE.itemsize == 52, "the PRD asks for roughly 50-100 bytes per position"


def pack_board(board: chess.Board) -> np.ndarray:
    """``board``'s piece placement as :data:`BOARD_BYTES` bytes, a nibble per square.

    Read off the bitboards rather than out of ``piece_map``, because this runs once per ply
    of every game in a dump of tens of millions: building 32 ``Piece`` objects per position
    to throw them all away again was the slowest thing a build did.
    """
    packed = bytearray(BOARD_BYTES)
    black = board.occupied_co[chess.BLACK]
    # The bitboards in the order of python-chess's piece-type numbers, which are the codes.
    for code, remaining in enumerate(
        (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings),
        start=1,
    ):
        while remaining:
            lowest = remaining & -remaining
            square = lowest.bit_length() - 1
            remaining ^= lowest
            packed[square >> 1] |= (code + 6 if lowest & black else code) << (4 * (square & 1))
    return np.frombuffer(bytes(packed), dtype=np.uint8)


_CASTLING_FLAGS: Final = (
    (PositionFlags.WHITE_KINGSIDE, chess.BB_H1),
    (PositionFlags.WHITE_QUEENSIDE, chess.BB_A1),
    (PositionFlags.BLACK_KINGSIDE, chess.BB_H8),
    (PositionFlags.BLACK_QUEENSIDE, chess.BB_A8),
)


def board_flags(board: chess.Board) -> int:
    """``board``'s side to move and castling rights, as :class:`PositionFlags` bits."""
    flags = PositionFlags.WHITE_TO_MOVE if board.turn == chess.WHITE else 0
    rights = board.clean_castling_rights()
    for bit, square_mask in _CASTLING_FLAGS:
        if rights & square_mask:
            flags |= bit
    return int(flags)


def en_passant_field(board: chess.Board) -> int:
    """The square an en-passant capture can be made on, or :data:`NO_SQUARE`.

    Only a capture that can actually be played is recorded. python-chess keeps the square
    behind any double step, and a FEN prints one, but a square no pawn can capture on
    changes no legal move, and leaving it out makes two positions that play identically
    pack identically.
    """
    return board.ep_square if board.has_legal_en_passant() else NO_SQUARE


def position_record(
    board: chess.Board,
    *,
    move: int,
    result: Result,
    mover_rating: int,
    opponent_rating: int,
    mover_rating_known: bool,
    opponent_rating_known: bool,
    rating_source: RatingSource,
    time_control: TimeControl,
    ply: int,
) -> tuple:
    """One position record's fields, in :data:`POSITION_DTYPE`'s order, ready to be assigned.

    A tuple rather than a filled-in record because assigning one is a single call into numpy
    instead of one per field, and this is called once per ply of every game in a dump. The
    game field is left at 0: only the writer knows which game index a game is getting.

    The halfmove clock is capped at :data:`MAX_CLOCK`, which no game reaches: a position 255
    halfmoves from the last capture or pawn move is 100 past the draw.
    """
    flags = board_flags(board)
    if not mover_rating_known:
        flags |= PositionFlags.MOVER_RATING_UNKNOWN
    if not opponent_rating_known:
        flags |= PositionFlags.OPPONENT_RATING_UNKNOWN
    return (
        pack_board(board),
        flags,
        en_passant_field(board),
        min(board.halfmove_clock, MAX_CLOCK),
        result,
        rating_source,
        time_control,
        min(board.fullmove_number, 0xFFFF),
        move,
        mover_rating,
        opponent_rating,
        ply,
        0,
    )


def unpack_board(position: np.void) -> chess.Board:
    """The position ``position`` was packed from, as a board the rules apply to again.

    The board plays the same moves as the one it was packed from and prints the same FEN.
    """
    board = chess.Board.empty()
    # As bytes rather than a field at a time: indexing a numpy array 64 times to read 32
    # bytes costs more than the copy does.
    packed = position["board"].tobytes()
    for square in chess.SQUARES:
        code = (packed[square >> 1] >> (4 * (square & 1))) & 0xF
        if code:
            color = chess.WHITE if code <= 6 else chess.BLACK
            board.set_piece_at(square, chess.Piece(code - (0 if color else 6), color))
    flags = int(position["flags"])
    board.turn = chess.WHITE if flags & PositionFlags.WHITE_TO_MOVE else chess.BLACK
    rights = chess.BB_EMPTY
    for bit, square_mask in _CASTLING_FLAGS:
        if flags & bit:
            rights |= square_mask
    board.castling_rights = rights
    en_passant = int(position["en_passant"])
    board.ep_square = None if en_passant == NO_SQUARE else en_passant
    board.halfmove_clock = int(position["halfmove_clock"])
    board.fullmove_number = int(position["fullmove_number"])
    return board
