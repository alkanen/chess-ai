"""Turning one PGN game into the records a dataset stores, or deciding to leave it out.

PGN is written by hand as often as by software, and a dump of millions of games contains
every way that can go wrong: games that end mid-move, moves that are not legal, positions
that are not positions, headers left as "?", variants that are not chess. A build that
crashes on any of them is a build that cannot finish, so a game that cannot be turned into
records is skipped, counted under a reason, and forgotten.

What the headers say is read here too, because it is guesswork everywhere else: PGN has one
``TimeControl`` tag covering sudden death, increments, several periods and no clock at all,
and nothing at all that says which rating pool an ``Elo`` came from.
"""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import chess
import chess.pgn
import numpy as np

from chess_ai.dataset.records import (
    GAME_DTYPE,
    MOVE_DTYPE,
    POSITION_DTYPE,
    GameFlags,
    RatingSource,
    Result,
    TimeControl,
    position_record,
)
from chess_ai.move_codec import move_index

MAX_PLIES: Final = 0xFFFF
"""The longest game a record can describe, which no game of chess reaches."""

MAX_RATING: Final = 0xFFFF

STANDARD_VARIANTS: Final = frozenset({"", "standard", "chess", "normal", "from position"})
"""``Variant`` values that still mean ordinary chess; anything else is out of scope.

"From Position" is what Lichess calls a standard game that started somewhere unusual, which
is a legitimate game of chess and kept, with :attr:`~.records.GameFlags.CUSTOM_START` set.
"""

_RESULTS: Final = {"1-0": Result.WIN, "0-1": Result.LOSS, "1/2-1/2": Result.DRAW}
"""The result from white's point of view. "*" is not in here; see :attr:`SkipReason.NO_RESULT`."""

RESULT_NAMES: Final = {result: text for text, result in _RESULTS.items()}
"""How PGN writes each result, for the statistics."""

MOVES_ASSUMED: Final = 40
"""Moves a time control's increment is estimated over, which is how Lichess classifies one."""

_TIME_CONTROL_LIMITS: Final = (
    (180, TimeControl.BULLET),
    (480, TimeControl.BLITZ),
    (1500, TimeControl.RAPID),
    (21600, TimeControl.CLASSICAL),
)
"""Estimated seconds per player, and the class a game up to that estimate belongs to.

The first three bounds are the ones Lichess sorts its own pools by. The last separates a
classical game from a correspondence one: six hours on one player's clock is well past any
game played at a board and well short of a day per move.
"""


class SkipReason(StrEnum):
    """Why a game was left out of a build. Every skipped game is counted under one of these."""

    NO_MOVES = "no_moves"
    """Nothing to learn from: an empty game, or prose the parser read as one."""
    NO_RESULT = "no_result"
    """Result "*": a game still being played, or abandoned before it had one. The value
    target is the result from the mover's side, so a game without one has none."""
    ILLEGAL_MOVE = "illegal_move"
    """A move the parser could not play, which truncates the game where it appears."""
    BAD_POSITION = "bad_position"
    """The ``FEN`` header is not a position, or not a legal one."""
    UNSUPPORTED_VARIANT = "unsupported_variant"
    """Not standard chess; only standard chess is in scope."""
    TOO_LONG = "too_long"
    """More plies than a record can count, which means the file is not a game."""
    UNREADABLE = "unreadable"
    """Anything else the parser or this code could not make sense of."""


class GameSkipped(Exception):
    """One game is being left out. Carries the :class:`SkipReason` the build counts it under."""

    def __init__(self, reason: SkipReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class GameRecords:
    """One game as a dataset stores it: the game, its positions, and the moves played.

    ``identity`` is what the game is, as text, and is all the split hash looks at; see
    :func:`split_of`.
    """

    game: np.ndarray
    positions: np.ndarray
    moves: np.ndarray
    identity: str


def game_records(
    record: chess.pgn.Game, *, source: int, rating_source: RatingSource
) -> GameRecords:
    """``record`` as dataset records, or raise :exc:`GameSkipped` saying why it cannot be.

    ``source`` is the index of the file the game came from. ``rating_source`` is the pool to
    record for it, which the caller has either been told or read off the headers.
    """
    headers = record.headers
    if headers.get("Variant", "").strip().lower() not in STANDARD_VARIANTS:
        raise GameSkipped(SkipReason.UNSUPPORTED_VARIANT)
    result = _RESULTS.get(headers.get("Result", "*"))
    if result is None:
        raise GameSkipped(SkipReason.NO_RESULT)
    try:
        board = record.board()
    except ValueError as e:
        raise GameSkipped(SkipReason.BAD_POSITION) from e
    if not board.is_valid():
        raise GameSkipped(SkipReason.BAD_POSITION)
    if record.errors:
        # A move the parser could not play is left here rather than raised, and the game it
        # was reading stops at that move. Half a game is not the game the file recorded.
        raise GameSkipped(SkipReason.ILLEGAL_MOVE)
    moves = list(record.mainline_moves())
    if not moves:
        raise GameSkipped(SkipReason.NO_MOVES)
    if len(moves) > MAX_PLIES:
        raise GameSkipped(SkipReason.TOO_LONG)

    custom_start = board != chess.Board()
    white_rating, white_known = _rating(headers.get("WhiteElo"))
    black_rating, black_known = _rating(headers.get("BlackElo"))
    time_control = time_control_class(headers.get("TimeControl"))

    # Built as tuples and handed to numpy in one go: a field at a time is a call into numpy
    # per field per ply, and a dump has hundreds of millions of plies.
    fields = []
    indices = []
    for ply, move in enumerate(moves):
        try:
            index = move_index(move)
        except KeyError as e:
            # No move python-chess generates is outside the vocabulary, so this is a move it
            # did not generate: a drop, or a null move written as one.
            raise GameSkipped(SkipReason.ILLEGAL_MOVE) from e
        white_to_move = board.turn == chess.WHITE
        indices.append(index)
        fields.append(
            position_record(
                board,
                move=index,
                result=result if white_to_move else result.opponent,
                mover_rating=white_rating if white_to_move else black_rating,
                opponent_rating=black_rating if white_to_move else white_rating,
                mover_rating_known=white_known if white_to_move else black_known,
                opponent_rating_known=black_known if white_to_move else white_known,
                rating_source=rating_source,
                time_control=time_control,
                ply=ply,
            )
        )
        board.push(move)
    positions = np.array(fields, dtype=POSITION_DTYPE)

    game_flags = GameFlags.CUSTOM_START if custom_start else 0
    if not white_known:
        game_flags |= GameFlags.WHITE_RATING_UNKNOWN
    if not black_known:
        game_flags |= GameFlags.BLACK_RATING_UNKNOWN
    game = np.array(
        [
            (
                0,  # The writer fills in where in the split this game's plies went.
                len(moves),
                white_rating,
                black_rating,
                _date(headers),
                game_flags,
                result,
                rating_source,
                time_control,
                source,
            )
        ],
        dtype=GAME_DTYPE,
    )
    identity = [_identity_header(headers, name) for name in ("Site", "Date", "White", "Black")]
    return GameRecords(
        game=game,
        positions=positions,
        moves=np.array(indices, dtype=MOVE_DTYPE),
        identity="|".join([*identity, *(move.uci() for move in moves)]),
    )


def split_of(identity: str, validation_fraction: float) -> bool:
    """Whether the game called ``identity`` belongs to the validation split.

    The split is decided per game, by a hash of the game itself, for two reasons. Splitting
    by position would put a game's opening in training and its endgame in validation, and a
    model that has seen the first half of a game has an unearned head start on the rest of
    it. And a hash means no state: the same game lands in the same split in every dataset it
    is ever built into, so a fine-tuning set cannot hand the model a game the pretraining
    set validated against.
    """
    digest = hashlib.blake2b(identity.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < validation_fraction


def rating_source_of(record: chess.pgn.Game) -> RatingSource:
    """Which rating pool ``record``'s ratings come from, as far as its headers say.

    Only the sites that say so in a URL can be recognized; a file that does not name where
    it came from gets :attr:`~.records.RatingSource.UNKNOWN`, which a build can override for
    a source it knows the provenance of.
    """
    said = " ".join(record.headers.get(name, "") for name in ("Site", "Link")).lower()
    if "lichess.org" in said:
        return RatingSource.LICHESS
    if "chess.com" in said:
        return RatingSource.CHESSCOM
    return RatingSource.UNKNOWN


def time_control_class(header: str | None) -> TimeControl:
    """The class ``header``'s time control falls into, by how long it gives a player.

    PGN's ``TimeControl`` is a colon-separated list of periods, each "seconds",
    "moves/seconds" or "*seconds" for a sandclock, any of them with a "+increment". "-" is a
    game played without a clock, and "?" is a game that did not say. The estimate is the
    first period's time plus its increment over :data:`MOVES_ASSUMED` moves, which is how
    Lichess sorts the same games into the same pools.
    """
    text = (header or "").strip()
    if not text or text == "?":
        return TimeControl.UNKNOWN
    if text == "-":
        return TimeControl.CORRESPONDENCE
    period = text.split(":")[0]
    base, _, increment = period.partition("+")
    if base.startswith("*"):
        # A sandclock gives a player the whole game, which says nothing about its pace.
        return TimeControl.UNKNOWN
    seconds = base.rpartition("/")[2]
    try:
        estimate = float(seconds) + MOVES_ASSUMED * float(increment or 0)
    except ValueError:
        return TimeControl.UNKNOWN
    for limit, time_control in _TIME_CONTROL_LIMITS:
        if estimate < limit:
            return time_control
    return TimeControl.CORRESPONDENCE


def _rating(header: str | None) -> tuple[int, bool]:
    """A rating and whether the file gave one.

    A missing, unparseable or "?" rating is kept as a game with the rating unknown rather
    than thrown away: an unrated source is still a source of moves, and the flag lets a
    model and an experiment tell the difference.
    """
    try:
        rating = int((header or "").strip())
    except ValueError:
        return 0, False
    if not 0 <= rating <= MAX_RATING:
        return 0, False
    return rating, True


def _date(headers: chess.pgn.Headers) -> int:
    """When the game was played, as ``yyyymmdd``, or 0 if the file did not say.

    PGN writes an unknown part of a date as "?", so a game known only to the month keeps its
    year and month and loses its day.
    """
    for name in ("UTCDate", "Date"):
        parts = headers.get(name, "").strip().split(".")
        if len(parts) != 3:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        month, day = (_number(part) for part in parts[1:])
        if 0 <= year <= 9999:
            return year * 10000 + month * 100 + day
    return 0


def _number(part: str) -> int:
    """One part of a date, with PGN's "??" for an unknown one reading as 0."""
    try:
        return int(part)
    except ValueError:
        return 0


def _identity_header(headers: chess.pgn.Headers, name: str) -> str:
    """One header as the split hash sees it, with PGN's "?" for unknown reading as empty."""
    value = headers.get(name, "").strip()
    return "" if value == "?" else value
