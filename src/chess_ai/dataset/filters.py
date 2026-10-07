"""Which games a build lets through, and which of their positions are trained on.

A filter answers one of two questions. Most are about a *game* and are answered from its
headers alone -- how fast it was played, how it ended, when -- so a game that fails one is left
out whole, and its moves are never even parsed: in a build that keeps a few percent of a dump,
not parsing the rest is most of what makes it fast.

The others are about a *position*: the rating of the player to move, and how much time they had
left. A game between a 2300 and an 1800 is half a game worth learning from, but the history an
encoder reads and the move sequence a sequence model reads are both the whole game, so its
positions are all stored and the 1800's are marked as not a training target. A game none of whose
positions is a target is left out.

What this module does not decide is anything about whether a game is *chess*: that is
:mod:`~chess_ai.dataset.games`, and a broken game is skipped there whatever the filters say.
"""

import calendar
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from chess_ai.dataset.records import TimeControl

SAMPLE_PERSON: Final = b"sample"
"""What sets the sampling hash apart from the split's, which hashes the same text.

Without it the two would be the same number: every validation game (hash below 0.02) would be in
every sample (hash below 0.05), and a 5% sample would be 40% validation.
"""


class Termination(StrEnum):
    """How a game ended, as far as its ``Termination`` header says.

    Lichess writes one of a handful of words; chess.com writes a sentence ("X won on time"); this
    project's own games write the reason ("checkmate"). They are sorted into these, by
    :func:`termination_of`.
    """

    NORMAL = "normal"
    """Played to an end on the board, by mate, resignation or a draw."""
    TIME_FORFEIT = "time_forfeit"
    """A flag fell. The moves are real, though the last few are often a scramble."""
    ABANDONED = "abandoned"
    """A player left, which mostly happens a few moves in."""
    RULES_INFRACTION = "rules_infraction"
    """Lichess ended the game for a breach of its rules, mostly cheating: always left out."""
    UNTERMINATED = "unterminated"
    """The game was cut off, usually by the server, without being decided."""
    UNKNOWN = "unknown"
    """No header, or one that says nothing recognisable."""


_TERMINATIONS: Final = {
    "normal": Termination.NORMAL,
    "time forfeit": Termination.TIME_FORFEIT,
    "abandoned": Termination.ABANDONED,
    "rules infraction": Termination.RULES_INFRACTION,
    "unterminated": Termination.UNTERMINATED,
    # This project's own saved games; see chess_ai.pgn.TERMINATIONS.
    "checkmate": Termination.NORMAL,
    "stalemate": Termination.NORMAL,
    "insufficient material": Termination.NORMAL,
    "threefold repetition": Termination.NORMAL,
    "fifty-move rule": Termination.NORMAL,
    "resignation": Termination.NORMAL,
}
"""Whole headers that name a termination, lowercased."""

_TERMINATION_WORDS: Final = (
    ("abandon", Termination.ABANDONED),
    ("on time", Termination.TIME_FORFEIT),
    ("timeout", Termination.TIME_FORFEIT),
    ("time forfeit", Termination.TIME_FORFEIT),
    ("checkmate", Termination.NORMAL),
    ("resignation", Termination.NORMAL),
    ("agreement", Termination.NORMAL),
    ("repetition", Termination.NORMAL),
    ("stalemate", Termination.NORMAL),
    ("insufficient material", Termination.NORMAL),
    ("50-move", Termination.NORMAL),
    ("50 move", Termination.NORMAL),
)
"""Words that place a sentence, in the order they are looked for.

chess.com's "Game drawn by timeout vs insufficient material" is a flag that fell with nothing
left to mate with, so the clock is looked for before the material.
"""


def termination_of(header: str | None) -> Termination:
    """The :class:`Termination` a ``Termination`` header describes."""
    text = (header or "").strip().lower()
    if not text:
        return Termination.UNKNOWN
    exact = _TERMINATIONS.get(text)
    if exact is not None:
        return exact
    # chess.com writes "<winner> won by checkmate", and a name can hold any of the words below
    # ("NoTimeout"), a space, or " won " itself; the reason after the last " won " never does.
    _, won, reason = text.rpartition(" won ")
    if won:
        text = reason
    for word, termination in _TERMINATION_WORDS:
        if word in text:
            return termination
    return Termination.UNKNOWN


class FilterReason(StrEnum):
    """Why a game a build could read was left out, or a stored position is not trained on."""

    TERMINATION = "termination"
    TIME_CONTROL = "time_control"
    DATE = "date"
    RATING = "rating"
    """For a game: neither player's rating passes. For a position: the mover's does not."""
    CLOCK = "clock"
    """For a game: no position of it is a target. For a position: the mover was short of time."""
    SAMPLE = "sample"
    """The sample left it out; see :func:`sampled`."""


_DATE = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")


def date_bounds(text: str) -> tuple[int, int]:
    """The first and last day ``text`` covers, as ``yyyymmdd``.

    ``text`` is "YYYY", "YYYY-MM" or "YYYY-MM-DD"; a year or a month covers all of itself, so
    ``2024-06`` runs from the first of June to the thirtieth. Raises :exc:`ValueError` for
    anything else, including a day the month does not have.
    """
    match = _DATE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"{text!r} is not a date: write YYYY, YYYY-MM or YYYY-MM-DD")
    year = int(match[1])
    if year < 1:
        raise ValueError(f"{text!r} is not a date: there is no year 0")
    if match[2] is None:
        return year * 10000 + 101, year * 10000 + 1231
    month = int(match[2])
    if not 1 <= month <= 12:
        raise ValueError(f"{text!r} is not a date: there is no month {month}")
    last = calendar.monthrange(year, month)[1]
    if match[3] is None:
        return year * 10000 + month * 100 + 1, year * 10000 + month * 100 + last
    day = int(match[3])
    if not 1 <= day <= last:
        raise ValueError(f"{text!r} is not a date: that month has no day {day}")
    day_number = year * 10000 + month * 100 + day
    return day_number, day_number


def game_period(date: int) -> tuple[int, int] | None:
    """The first and last day a stored ``yyyymmdd`` could be, or ``None`` for no date at all.

    PGN writes the part of a date it does not know as "??", which a record stores as 0, so a
    game of "2024.03.??" could have been played on any day of March.
    """
    year, month, day = date // 10000, date // 100 % 100, date % 100
    if not year:
        return None
    if not month:
        return year * 10000 + 101, year * 10000 + 1231
    if not day:
        last = calendar.monthrange(year, month)[1]
        return year * 10000 + month * 100 + 1, year * 10000 + month * 100 + last
    return date, date


def sampled(identity: str, fraction: float) -> bool:
    """Whether the game called ``identity`` is in a sample of ``fraction`` of all games.

    A hash of the game, like the split, so the same game is in or out of every sample of that
    size, and a smaller sample is part of every larger one. Keyed apart from the split's hash; see
    :data:`SAMPLE_PERSON`.
    """
    digest = hashlib.blake2b(
        identity.encode("utf-8", "surrogatepass"), digest_size=8, person=SAMPLE_PERSON
    ).digest()
    return int.from_bytes(digest, "big") / 2**64 < fraction


@dataclass(frozen=True)
class Screen:
    """The filters of one build, in the form the reading checks them in.

    Made from a :class:`~chess_ai.dataset.manifest.Filters` by
    :meth:`~chess_ai.dataset.manifest.Filters.screen`, once per build, and handed to every
    worker: the checks run once per game of a dump, so nothing in them parses a filter again.
    """

    min_rating: int | None = None
    max_rating: int | None = None
    unknown_rating_passes: bool = False
    time_controls: frozenset[TimeControl] | None = None
    excluded_terminations: frozenset[Termination] = frozenset()
    first_day: int | None = None
    last_day: int | None = None
    min_clock: float | None = None
    sample: float | None = None

    @property
    def rates(self) -> bool:
        """Whether a rating limit is set, which is what makes ratings matter at all."""
        return self.min_rating is not None or self.max_rating is not None

    @property
    def per_game(self) -> bool:
        """Whether anything here can leave a game out on its headers."""
        return (
            self.rates
            or self.time_controls is not None
            or bool(self.excluded_terminations)
            or self.first_day is not None
            or self.last_day is not None
        )

    @property
    def per_position(self) -> bool:
        """Whether some positions of a kept game may not be training targets."""
        return self.rates or self.min_clock is not None

    def rating_passes(self, rating: int, known: bool) -> bool:
        """Whether a player of ``rating`` passes the rating limits."""
        if not self.rates:
            return True
        if not known:
            return self.unknown_rating_passes
        if self.min_rating is not None and rating < self.min_rating:
            return False
        return self.max_rating is None or rating <= self.max_rating

    def game_reason(
        self,
        *,
        termination: Termination,
        time_control: TimeControl,
        date: int,
        white: tuple[int, bool],
        black: tuple[int, bool],
    ) -> FilterReason | None:
        """Why a game is left out on what its headers say, or ``None`` if it is not.

        The rating passes when either player's does: the other player's positions are then
        stored and not trained on.
        """
        if termination in self.excluded_terminations:
            return FilterReason.TERMINATION
        if self.time_controls is not None and time_control not in self.time_controls:
            return FilterReason.TIME_CONTROL
        if self.first_day is not None or self.last_day is not None:
            period = game_period(date)
            if period is None:
                return FilterReason.DATE
            if self.first_day is not None and period[0] < self.first_day:
                return FilterReason.DATE
            if self.last_day is not None and period[1] > self.last_day:
                return FilterReason.DATE
        if not (self.rating_passes(*white) or self.rating_passes(*black)):
            return FilterReason.RATING
        return None

    def targets(
        self,
        white_to_move: Sequence[bool],
        clocks: Sequence[float | None] | None,
        *,
        white_passes: bool,
        black_passes: bool,
    ) -> tuple[np.ndarray, int, int]:
        """Which of a game's positions are training targets, and how many are not, by why.

        ``white_to_move`` says who moves in each position, and ``clocks`` how many seconds the
        mover had left when they did, ``None`` where the file did not say. Returns the mask, the
        positions left out for the mover's rating, and those left out for the clock; a position
        that fails both counts as the rating's.
        """
        movers = np.asarray(white_to_move, dtype=bool)
        rated = np.where(movers, white_passes, black_passes)
        timed = np.ones(len(movers), dtype=bool)
        if self.min_clock is not None and clocks is not None:
            timed = np.array(
                [clock is None or clock >= self.min_clock for clock in clocks], dtype=bool
            )
        targets = rated & timed
        return targets, int(np.count_nonzero(~rated)), int(np.count_nonzero(rated & ~timed))
