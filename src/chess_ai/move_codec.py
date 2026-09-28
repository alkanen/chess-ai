"""Move codec: the one move vocabulary everything here shares.

A network cannot emit a move, only a number, so every move a game can contain is given
an index in a fixed, flat vocabulary. Policy outputs, dataset records, checkpoints and
the inference engine all speak in these indices, so the vocabulary and its order are
part of the on-disk format: appending to it would change what a trained checkpoint's
outputs mean, and reordering it would silently invalidate every model ever trained.

The vocabulary holds every *geometrically* possible move rather than every legal one,
which is what makes it fixed. A move is a (from-square, to-square, promotion piece)
triple, and the geometry is a queen's and a knight's: no piece moves anywhere else, and
castling is a two-square king move, as in UCI. Whether a move is legal is a property of
a position, not of the vocabulary, and :func:`legal_mask` answers it per position.

Nothing in here needs a promotion-less move from the seventh rank to the eighth, because
a pawn reaching the last rank must promote. Such an index exists all the same, as the
queen move it also is, and simply never comes up as a pawn's.
"""

from typing import Final

import chess
import numpy as np

PROMOTION_PIECES: Final = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
"""What a pawn may become, queen first because that is what it almost always is."""


def _vocabulary() -> tuple[chess.Move, ...]:
    """Every geometrically possible move, in the order that fixes their indices.

    Sorted by from-square, then to-square, then promotion piece, so the order follows
    from the rules rather than from the way this happens to be written: the same
    vocabulary comes back from any implementation that agrees on the sort.
    """
    moves: list[chess.Move] = []
    for from_square in chess.SQUARES:
        for to_square in sorted(_reachable(from_square)):
            moves.append(chess.Move(from_square, to_square))
            if _promotes(from_square, to_square):
                moves.extend(
                    chess.Move(from_square, to_square, promotion=piece)
                    for piece in sorted(PROMOTION_PIECES)
                )
    return tuple(moves)


def _reachable(from_square: chess.Square) -> set[chess.Square]:
    """Where a piece on ``from_square`` could ever land, on an otherwise empty board.

    A queen and a knight between them cover every way any piece moves, pawns included:
    a pawn's push and its diagonal capture are both queen moves, and the king's castling
    is its two squares along the rank.
    """
    board = chess.Board.empty()
    squares: set[chess.Square] = set()
    for piece_type in (chess.QUEEN, chess.KNIGHT):
        board.set_piece_at(from_square, chess.Piece(piece_type, chess.WHITE))
        squares.update(move.to_square for move in board.legal_moves)
    return squares


def _promotes(from_square: chess.Square, to_square: chess.Square) -> bool:
    """Whether a pawn moving between these squares would have to promote.

    That is a move onto the last rank from the one before it, for either colour, and by
    a file at most: the two ranks a pawn crosses to promote are adjacent, so the queen
    moves between them that a pawn could not make are ruled out by the file alone.
    """
    if abs(chess.square_file(from_square) - chess.square_file(to_square)) > 1:
        return False
    ranks = (chess.square_rank(from_square), chess.square_rank(to_square))
    return ranks in {(6, 7), (1, 0)}


VOCABULARY: Final = _vocabulary()
"""Every move, by index. Its order is part of the on-disk format; see the module docstring."""

VOCABULARY_SIZE: Final = len(VOCABULARY)
"""How many moves there are, which is how wide a policy output is."""

_INDICES: Final = {move: index for index, move in enumerate(VOCABULARY)}


def move_index(move: chess.Move) -> int:
    """``move``'s index in the vocabulary.

    Raises :exc:`KeyError` for a move no piece could make, which is a bug in the caller
    rather than an unusual position: every move python-chess generates is in here.
    """
    return _INDICES[move]


def move_at(index: int) -> chess.Move:
    """The move at ``index``, which is :func:`move_index`'s inverse."""
    return VOCABULARY[index]


def legal_mask(board: chess.Board) -> np.ndarray:
    """Which vocabulary entries are legal moves in ``board``, as a boolean array.

    This is what masks a policy output down to moves that can actually be played.
    """
    mask = np.zeros(VOCABULARY_SIZE, dtype=bool)
    mask[[_INDICES[move] for move in board.legal_moves]] = True
    return mask


def _mirrored_index() -> np.ndarray:
    """Each move's index once the board is turned around; see :func:`mirrored_index`."""
    mirrored = np.empty(VOCABULARY_SIZE, dtype=np.uint16)
    for index, move in enumerate(VOCABULARY):
        flipped = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
        mirrored[index] = _INDICES[flipped]
    return mirrored


MIRRORED_INDEX: Final = _mirrored_index()
"""Every move's mirrored index, for mirroring a whole batch of them at once."""
MIRRORED_INDEX.flags.writeable = False


def mirrored_index(index: int) -> int:
    """The same move played on a board turned around, so that both colours play up.

    An encoder that orients the board from the side to move sees black's moves as white's
    and has to say which move that makes it. The move keeps its shape and its promotion
    piece and changes rank: a1 becomes a8. Mirroring twice gives back the move, which is
    what makes this safe to apply on the way in and on the way out.
    """
    return int(MIRRORED_INDEX[index])
