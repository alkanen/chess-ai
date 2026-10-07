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
from datetime import date
from enum import StrEnum
from typing import Final

import chess
import chess.pgn
import numpy as np

from chess_ai.dataset.filters import (
    FilterReason,
    Screen,
    Termination,
    sampled,
    termination_of,
)
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
    RULES_INFRACTION = "rules_infraction"
    """Ended by the site for a breach of its rules, which is mostly cheating: moves an engine
    chose, under a human's rating. Left out of every dataset, whatever its filters say."""
    UNREADABLE = "unreadable"
    """Anything else the parser or this code could not make sense of."""


class GameSkipped(Exception):
    """One game is being left out. Carries the reason the build counts it under.

    A :class:`SkipReason` for a game that is not usable chess, and a
    :class:`~chess_ai.dataset.filters.FilterReason` for one the filters did not let through.
    """

    def __init__(self, reason: SkipReason | FilterReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class GameRecords:
    """One game as a dataset stores it: the game, its positions, and the moves played.

    ``identity`` is what the game is, as text, and is all the split hash looks at; see
    :func:`split_of`. ``targets`` says which positions are trained on, one boolean each, and
    ``not_targets`` how many are not, for the mover's rating and for the clock.
    """

    game: np.ndarray
    positions: np.ndarray
    moves: np.ndarray
    identity: str
    targets: np.ndarray
    not_targets: tuple[int, int] = (0, 0)


NO_SCREEN: Final = Screen()
"""The filters of a build that has none."""


def header_verdict(
    headers: chess.pgn.Headers, screen: Screen = NO_SCREEN
) -> SkipReason | FilterReason | None:
    """Why a game is left out on its headers alone, or ``None`` if they do not say it is.

    Everything here is decided before a single move is parsed, which is what lets the reading
    skip the moves of a game the answer is already known for; see :func:`skips_moves`.
    """
    if headers.get("Variant", "").strip().lower() not in STANDARD_VARIANTS:
        return SkipReason.UNSUPPORTED_VARIANT
    if headers.get("Result", "*") not in _RESULTS:
        return SkipReason.NO_RESULT
    termination = termination_of(headers.get("Termination"))
    if termination is Termination.RULES_INFRACTION:
        return SkipReason.RULES_INFRACTION
    if not screen.per_game:
        # Not worth reading the rest of the headers for, once per game of a dump.
        return None
    return screen.game_reason(
        termination=termination,
        time_control=time_control_class(headers.get("TimeControl")),
        date=_date(headers),
        white=_rating(headers.get("WhiteElo")),
        black=_rating(headers.get("BlackElo")),
    )


def skips_moves(headers: chess.pgn.Headers, screen: Screen) -> bool:
    """Whether the moves of a game with these headers need not be parsed at all.

    Only for what this project added -- the filters, and leaving out games ended for cheating.
    Games skipped for not being chess keep being parsed as they always were, so that a build
    without filters reads exactly what it read before there were any.
    """
    verdict = header_verdict(headers, screen)
    return isinstance(verdict, FilterReason) or verdict is SkipReason.RULES_INFRACTION


def game_records(
    record: chess.pgn.Game,
    *,
    source: int,
    rating_source: RatingSource,
    screen: Screen = NO_SCREEN,
) -> GameRecords:
    """``record`` as dataset records, or raise :exc:`GameSkipped` saying why it cannot be.

    ``source`` is the index of the file the game came from. ``rating_source`` is the pool to
    record for it, which the caller has either been told or read off the headers. ``screen`` is
    the build's filters.

    What the headers say is decided first, so a game is counted under the same reason whether or
    not its moves were parsed; see :func:`skips_moves`.
    """
    headers = record.headers
    verdict = header_verdict(headers, screen)
    if verdict is not None:
        raise GameSkipped(verdict)
    result = _RESULTS[headers["Result"]]
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
    movers = []
    for ply, move in enumerate(moves):
        try:
            index = move_index(move)
        except KeyError as e:
            # No move python-chess generates is outside the vocabulary, so this is a move it
            # did not generate: a drop, or a null move written as one.
            raise GameSkipped(SkipReason.ILLEGAL_MOVE) from e
        white_to_move = board.turn == chess.WHITE
        movers.append(white_to_move)
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
    targets, rating_left_out, clock_left_out = screen.targets(
        movers,
        _clocks(record, len(moves), _starting_time(headers.get("TimeControl")))
        if screen.min_clock is not None
        else None,
        white_passes=screen.rating_passes(white_rating, white_known),
        black_passes=screen.rating_passes(black_rating, black_known),
    )
    if not targets.any():
        # The headers let it through, so one of the players passes the rating. Either that
        # player never moved, or every time they did they were short of time.
        raise GameSkipped(FilterReason.CLOCK if clock_left_out else FilterReason.RATING)
    identity = [_identity_header(headers, name) for name in ("Site", "Date", "White", "Black")]
    identity_text = "|".join([*identity, *(move.uci() for move in moves)])
    if screen.sample is not None and not sampled(identity_text, screen.sample):
        raise GameSkipped(FilterReason.SAMPLE)
    return GameRecords(
        game=game,
        positions=positions,
        moves=np.array(indices, dtype=MOVE_DTYPE),
        identity=identity_text,
        targets=targets,
        not_targets=(rating_left_out, clock_left_out),
    )


def _clocks(record: chess.pgn.Game, plies: int, start: float | None) -> list[float | None]:
    """How many seconds the player to move had left in each position, where the file says.

    A ``[%clk]`` comment is the time left *after* the move it follows, so what the mover had
    when they chose a move is what their previous move left them: two plies back. Before their
    first move each side had ``start``, the time control's starting time. Where neither says --
    a game whose file records no clocks, or does not say its time control -- the position gets
    ``None``: one the file says nothing about is not one known to be rushed.
    """
    after = [node.clock() for node in record.mainline()]
    return [after[ply - 2] if ply >= 2 else start for ply in range(plies)]


def _starting_time(header: str | None) -> float | None:
    """The seconds a ``TimeControl`` header gives each player at the start, if it says.

    Read the way :func:`time_control_class` reads it: the first period's time. A sandclock, a game
    without a clock, or a header that is not one says nothing.
    """
    text = (header or "").strip()
    base = text.split(":")[0].partition("+")[0]
    if base.startswith("*"):
        return None
    try:
        return float(base.rpartition("/")[2])
    except ValueError:
        return None


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
    year and month and loses its day. A part that is not a date at all is treated the same way,
    because a field read as ``yyyymmdd`` has to be one: scraped PGN carries "2024.99.99" and
    "2024.02.30", and a day of 99 stored as a day would quietly poison every date range, month
    bucket and chart that ever reads it. A day without a month it belongs to goes the same way.
    """
    for name in ("UTCDate", "Date"):
        parts = headers.get(name, "").strip().split(".")
        if len(parts) != 3:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if not 1 <= year <= 9999:
            continue
        month = _within(parts[1], 12)
        day = _within(parts[2], 31) if month else 0
        if day and not _is_a_day(year, month, day):
            day = 0
        return year * 10000 + month * 100 + day
    return 0


def _within(part: str, most: int) -> int:
    """One part of a date, or 0 for PGN's "??" and for anything that is not that part."""
    try:
        value = int(part)
    except ValueError:
        return 0
    return value if 1 <= value <= most else 0


def _is_a_day(year: int, month: int, day: int) -> bool:
    """Whether that month of that year has such a day, which February decides for itself."""
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _identity_header(headers: chess.pgn.Headers, name: str) -> str:
    """One header as the split hash sees it, with PGN's "?" for unknown reading as empty."""
    value = headers.get(name, "").strip()
    return "" if value == "?" else value
