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

A round robin is the other kind of results: players of unknown rating against each other, each
rated against the rest at once by :func:`rate_round_robin`. Nobody's rating is known there, only
the gaps between them, so the ratings come around an average of nothing unless one player is
anchored to a rating known from elsewhere.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

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


VIRTUAL_DRAWS = 1
"""How many drawn games :func:`rate_round_robin` adds to every pairing before it rates anyone:
enough that an entrant who won or lost every game still has a most likely rating, a few hundred
points from the others rather than infinitely far, and few enough to be lost among real games.
Draws, so that they pull no two entrants apart, only a little towards each other."""


@dataclass(frozen=True)
class Pairing:
    """What two entrants of a round robin scored against each other, by their places in it."""

    first: int
    second: int
    points: float
    """What the first scored: wins plus half the draws."""
    games: int

    def __post_init__(self) -> None:
        if self.games < 0 or not 0 <= self.points <= self.games:
            raise ValueError(f"{self.points} points from {self.games} games is not a result")
        if self.first == self.second:
            raise ValueError(f"entrant {self.first} cannot be paired with itself")


@dataclass(frozen=True)
class Standing:
    """An entrant's rating relative to the rest of a round robin, with the range it is good to."""

    rating: float
    low: float
    high: float
    games: int
    points: float


def rate_round_robin(
    entrants: int,
    pairings: Iterable[Pairing],
    *,
    anchor: tuple[int, float] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[Standing]:
    """The rating of each of ``entrants`` from what they scored against each other, in order.

    Every rating at once, as the ones that make all the results together most likely, after
    :data:`VIRTUAL_DRAWS` are added to every pairing. Only the gaps between them are known, so
    they are given around an average of nothing, unless ``anchor`` puts one entrant, by its
    place, at a rating known some other way: then every range is how far from that entrant
    the other could be, and the anchored entrant's own range is its rating alone.

    The range is the normal approximation from how sharply the likelihood falls away from its
    peak: symmetric, unlike :func:`estimate_rating`'s, which the virtual draws make close
    enough even for an entrant who won every game.

    Raises:
        ValueError: ``confidence`` is not between 0 and 1, or ``pairings`` do not connect
            every entrant to every other, through others if need be, which leaves the gap
            between the parts unknown.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"a confidence is between 0 and 1, not {confidence}")
    pairs = list(pairings)
    for pair in pairs:
        if not (0 <= pair.first < entrants and 0 <= pair.second < entrants):
            raise ValueError(f"a pairing of entrants {pair.first} and {pair.second} of {entrants}")
    if anchor is not None and not 0 <= anchor[0] < entrants:
        raise ValueError(f"entrant {anchor[0]} of {entrants} cannot be the anchor")
    _require_connected(entrants, pairs)

    # Rated with one entrant held at nothing, which takes away the one way of changing the
    # ratings that changes no result: moving them all together.
    held = anchor[0] if anchor is not None else 0
    free = [i for i in range(entrants) if i != held]
    scale = math.log(10) / ELO_SCALE
    ratings = np.zeros(entrants)
    for _ in range(_NEWTON_STEPS):
        gradient, information = _slopes(entrants, pairs, ratings, scale)
        step = np.linalg.solve(information[np.ix_(free, free)], gradient[free])
        # Held to a few hundred points a step, so that a first guess far from the answer is
        # walked towards it rather than overshot.
        step = np.clip(step, -_LARGEST_STEP, _LARGEST_STEP)
        ratings[free] += step
        if np.max(np.abs(step), initial=0.0) < _TOLERANCE:
            break
    _, information = _slopes(entrants, pairs, ratings, scale)

    # The spread of each rating about the held one, from the inverse of the information with
    # that one fixed. Without an anchor, about the average instead: centring spreads the held
    # entrant's share back out among them all.
    covariance = np.zeros((entrants, entrants))
    covariance[np.ix_(free, free)] = np.linalg.inv(information[np.ix_(free, free)])
    if anchor is None:
        centring = np.eye(entrants) - 1.0 / entrants
        covariance = centring @ covariance @ centring
        ratings -= ratings.mean()
    else:
        ratings += anchor[1]
    width = _normal_quantile(0.5 + confidence / 2) * np.sqrt(np.clip(np.diag(covariance), 0, None))

    games = [0] * entrants
    points = [0.0] * entrants
    for pair in pairs:
        games[pair.first] += pair.games
        games[pair.second] += pair.games
        points[pair.first] += pair.points
        points[pair.second] += pair.games - pair.points
    return [
        Standing(
            rating=float(ratings[i]),
            low=float(ratings[i] - width[i]),
            high=float(ratings[i] + width[i]),
            games=games[i],
            points=points[i],
        )
        for i in range(entrants)
    ]


_NEWTON_STEPS = 100
_LARGEST_STEP = 400.0


def _slopes(
    entrants: int, pairs: list[Pairing], ratings: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """How the log-likelihood rises with each rating, and the information: minus its curvature."""
    gradient = np.zeros(entrants)
    information = np.zeros((entrants, entrants))
    for pair in pairs:
        games = pair.games + VIRTUAL_DRAWS
        points = pair.points + VIRTUAL_DRAWS / 2
        expected = expected_score(ratings[pair.first], ratings[pair.second])
        slope = scale * (points - games * expected)
        gradient[pair.first] += slope
        gradient[pair.second] -= slope
        curvature = scale**2 * games * expected * (1 - expected)
        information[pair.first, pair.first] += curvature
        information[pair.second, pair.second] += curvature
        information[pair.first, pair.second] -= curvature
        information[pair.second, pair.first] -= curvature
    # In rating points per point of log-likelihood: Newton's step is the solution in ratings.
    return gradient, information


def _require_connected(entrants: int, pairs: list[Pairing]) -> None:
    reached = {0} if entrants else set()
    grew = True
    while grew:
        grew = False
        for pair in pairs:
            if (pair.first in reached) != (pair.second in reached):
                reached |= {pair.first, pair.second}
                grew = True
    if len(reached) < entrants:
        raise ValueError("the pairings leave some entrants with no way to compare them")
