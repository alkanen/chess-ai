"""Rating math: from results against opponents of known rating, to a rating with a range.

Pure functions, knowing nothing of chess: a result is points scored over games against an
opponent with a rating, and a rating is a number on the logistic Elo curve, on which a player
``d`` points stronger than another is expected to score ``1 / (1 + 10 ** (-d / 400))``.

The estimate is the rating that makes the results most likely, which is the one at which the
points expected against those opponents add up to the points scored. Its range comes from the
likelihood itself rather than from a standard error, so that it is lopsided where the evidence
is: a player who scored 1 out of 20 against the weakest opponent is known to be weaker than it
by more than they are known to be stronger than anything.

Points are counted as if each were a win and each half point half a win, which is how draws are
handled in Elo's own model. That treats a draw as a coin-flip it is not, so a match with many
draws is reported with a slightly wider range than it has earned: on the safe side.

Two cases have no estimate at all. A player who lost every game is less likely the stronger
they are, without end, and one who won every game likewise the weaker: the most likely rating
is infinitely far below the lowest opponent, or above the highest. Such a player is reported as
**below the floor** or **above the ceiling**, the weakest and the strongest opponent faced, with
only the side of the range that the results do pin down. An estimate that is finite but outside
the opponents' ratings is reported the same way, since all it says beyond the floor or ceiling
is how far the curve was extrapolated, which one draw more or less would move by hundreds.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

ELO_SCALE = 400.0
"""How many points of rating difference make a player ten times as likely to win as to lose."""

DEFAULT_CONFIDENCE = 0.95

_LIKELIHOOD_BOUND = 4000.0
"""How far beyond the opponents' ratings the range is looked for before it is taken to have
none on that side: further than any result short of every game can push it."""

_TOLERANCE = 1e-6
"""How close to the true rating the bisections stop, in rating points, and so how far past the
floor or ceiling an estimate may land before it is taken to be past it: an even score against
one opponent is that opponent's rating, not a hair above it."""


@dataclass(frozen=True)
class Result:
    """What a player scored over some games against one opponent of known rating."""

    opponent: float
    points: float
    """Wins plus half the draws."""
    games: int

    def __post_init__(self) -> None:
        if self.games < 0 or not 0 <= self.points <= self.games:
            raise ValueError(f"{self.points} points from {self.games} games is not a result")
        if not math.isfinite(self.opponent):
            raise ValueError(f"an opponent's rating is a number, not {self.opponent}")


class Beyond(StrEnum):
    """Which end of the opponents' ratings a player was found to be past."""

    FLOOR = "below_floor"
    CEILING = "above_ceiling"


@dataclass(frozen=True)
class Estimate:
    """A rating found from results, with the range it is good to, or the end it is past."""

    rating: float | None
    """The most likely rating, or ``None`` when the player is :attr:`beyond` the opponents."""
    low: float | None
    """The bottom of the range, or ``None`` when the results put no bottom to it."""
    high: float | None
    """The top of the range, or ``None`` when the results put no top to it."""
    beyond: Beyond | None
    """Set when the results put the player below every opponent or above them all."""
    floor: float
    """The weakest opponent's rating."""
    ceiling: float
    """The strongest opponent's rating."""
    confidence: float
    """How likely the range is to hold the true rating, such as 0.95."""
    games: int
    points: float

    def describe(self) -> str:
        """The estimate in a few words, such as ``1523 (1460 to 1590)`` or ``below 1350``."""
        if self.beyond is Beyond.FLOOR:
            top = f", at most {self.high:.0f}" if self.high is not None else ""
            return f"below {self.floor:.0f}{top}"
        if self.beyond is Beyond.CEILING:
            bottom = f", at least {self.low:.0f}" if self.low is not None else ""
            return f"above {self.ceiling:.0f}{bottom}"
        assert self.rating is not None
        low = f"{self.low:.0f}" if self.low is not None else "?"
        high = f"{self.high:.0f}" if self.high is not None else "?"
        return f"{self.rating:.0f} ({low} to {high})"


def expected_score(rating: float, opponent: float) -> float:
    """The points a player of ``rating`` is expected to score a game against ``opponent``."""
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / ELO_SCALE))


def estimate_rating(
    results: Iterable[Result], *, confidence: float = DEFAULT_CONFIDENCE
) -> Estimate:
    """The maximum-likelihood rating of a player with ``results``, and a range around it.

    The range is the likelihood-ratio interval: every rating whose results are not much less
    likely than the best one's, with "much" set by ``confidence`` through the chi-squared
    distribution with one degree of freedom.

    Raises:
        ValueError: there are no games among ``results``, or ``confidence`` is not between
            0 and 1.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"a confidence is between 0 and 1, not {confidence}")
    played = [result for result in results if result.games > 0]
    if not played:
        raise ValueError("a rating needs at least one game")
    floor = min(result.opponent for result in played)
    ceiling = max(result.opponent for result in played)
    games = sum(result.games for result in played)
    points = sum(result.points for result in played)

    if points <= 0 or points >= games:
        # No most likely rating, only a limit that every result approaches: certainty of the
        # results that were had, whose logarithm is nothing.
        best, best_likelihood = None, 0.0
    else:
        best = _bisect(
            lambda rating: _expected_points(played, rating) - points,
            floor - _LIKELIHOOD_BOUND,
            ceiling + _LIKELIHOOD_BOUND,
        )
        best_likelihood = _log_likelihood(played, best)

    # Twice the drop in log-likelihood from the best rating is chi-squared with one degree of
    # freedom, whose quantile is the square of the normal one for the two-sided confidence.
    allowed_drop = _normal_quantile(0.5 + confidence / 2) ** 2 / 2

    def excess_drop(rating: float) -> float:
        return best_likelihood - _log_likelihood(played, rating) - allowed_drop

    lowest, highest = floor - _LIKELIHOOD_BOUND, ceiling + _LIKELIHOOD_BOUND
    low = high = None
    if points > 0:
        start = best if best is not None else highest
        if excess_drop(lowest) > 0:
            low = _bisect(lambda rating: -excess_drop(rating), lowest, start)
    if points < games:
        start = best if best is not None else lowest
        if excess_drop(highest) > 0:
            high = _bisect(excess_drop, start, highest)

    beyond = None
    if best is None:
        beyond = Beyond.FLOOR if points <= 0 else Beyond.CEILING
    elif best < floor - _TOLERANCE:
        beyond = Beyond.FLOOR
    elif best > ceiling + _TOLERANCE:
        beyond = Beyond.CEILING
    return Estimate(
        rating=best if beyond is None else None,
        # Past the floor, the bottom of the range is as much an extrapolation as the estimate
        # was, and past the ceiling the top is.
        low=None if beyond is Beyond.FLOOR else low,
        high=None if beyond is Beyond.CEILING else high,
        beyond=beyond,
        floor=floor,
        ceiling=ceiling,
        confidence=confidence,
        games=games,
        points=points,
    )


def _expected_points(results: list[Result], rating: float) -> float:
    return sum(result.games * expected_score(rating, result.opponent) for result in results)


def _log_likelihood(results: list[Result], rating: float) -> float:
    """How likely ``results`` are for a player of ``rating``, as a natural logarithm."""
    total = 0.0
    for result in results:
        # Both terms through log1p of a power, rather than log of the expected score, so that a
        # rating thousands of points away gives a large finite number rather than log(0).
        difference = (result.opponent - rating) / ELO_SCALE
        if result.points:
            total -= result.points * _log1p_pow10(difference)
        if result.games - result.points:
            total -= (result.games - result.points) * _log1p_pow10(-difference)
    return total


def _log1p_pow10(exponent: float) -> float:
    """``ln(1 + 10 ** exponent)``, without overflowing for large exponents."""
    if exponent > 20:
        return exponent * math.log(10) + math.log1p(10.0**-exponent)
    return math.log1p(10.0**exponent)


def _bisect(rising, low: float, high: float) -> float:
    """Where ``rising``, which goes from negative at ``low`` to positive at ``high``, is zero."""
    while high - low > _TOLERANCE:
        middle = (low + high) / 2
        if rising(middle) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _normal_quantile(probability: float) -> float:
    """The standard normal quantile, found by bisecting the error function."""
    return _bisect(lambda z: (1 + math.erf(z / math.sqrt(2))) / 2 - probability, -10.0, 10.0)
