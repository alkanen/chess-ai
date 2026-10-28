import math
import random

import pytest

from chess_ai.rating import (
    Beyond,
    Pairing,
    Result,
    estimate_rating,
    expected_score,
    rate_round_robin,
)

LADDER = [1350, 1500, 1700, 1900, 2100]


def play(rng: random.Random, rating: float, opponent: float, games: int) -> Result:
    """``games`` games drawn from the Elo curve, a fifth of them drawn where they could be."""
    points = 0.0
    for _ in range(games):
        expected = expected_score(rating, opponent)
        # A draw is as likely as a fifth of the less likely decisive result allows, which keeps
        # the expected score of a game what the curve says it is.
        draw = 0.4 * min(expected, 1 - expected)
        roll = rng.random()
        if roll < draw:
            points += 0.5
        elif roll < draw + expected - draw / 2:
            points += 1.0
    return Result(opponent=opponent, points=points, games=games)


def test_expected_score_follows_the_logistic_elo_curve():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert expected_score(1500, 1900) == pytest.approx(1 / 11)


def test_an_even_score_against_one_opponent_is_that_opponents_rating():
    estimate = estimate_rating([Result(opponent=1600, points=10, games=20)])

    assert estimate.rating == pytest.approx(1600, abs=0.01)
    assert estimate.beyond is None
    assert estimate.low < 1600 < estimate.high
    assert estimate.high - 1600 == pytest.approx(1600 - estimate.low, abs=0.01), "symmetric"


def test_the_estimate_is_where_the_expected_points_add_up_to_the_points_scored():
    results = [Result(1350, 15, 20), Result(1700, 8.5, 20), Result(2100, 1, 20)]

    estimate = estimate_rating(results)

    expected = sum(r.games * expected_score(estimate.rating, r.opponent) for r in results)
    assert expected == pytest.approx(24.5, abs=1e-4)
    assert estimate.games == 60
    assert estimate.points == 24.5


@pytest.mark.parametrize("true_rating", [1450, 1650, 1800, 2000])
def test_known_ratings_are_recovered_within_the_range_as_often_as_it_claims(true_rating):
    rng = random.Random(true_rating)
    trials, covered = 400, 0
    for _ in range(trials):
        results = [play(rng, true_rating, level, 20) for level in LADDER]
        estimate = estimate_rating(results)
        low = estimate.low if estimate.low is not None else float("-inf")
        high = estimate.high if estimate.high is not None else float("inf")
        covered += low <= true_rating <= high
    # 95% claimed; the draws make the range wider than it need be, so it holds more often.
    assert covered / trials >= 0.93


def test_more_games_narrow_the_range():
    few = estimate_rating([Result(1500, 6, 10), Result(1700, 4, 10)])
    many = estimate_rating([Result(1500, 60, 100), Result(1700, 40, 100)])

    assert many.high - many.low < (few.high - few.low) / 2


def test_a_higher_confidence_widens_the_range():
    results = [Result(1500, 12, 20), Result(1700, 7, 20)]

    narrow = estimate_rating(results, confidence=0.8)
    wide = estimate_rating(results, confidence=0.99)

    assert wide.low < narrow.low < narrow.rating < narrow.high < wide.high


def test_losing_every_game_is_below_the_floor_with_only_a_top_to_the_range():
    estimate = estimate_rating([Result(opponent, 0, 20) for opponent in LADDER])

    assert estimate.beyond is Beyond.FLOOR
    assert estimate.rating is None
    assert estimate.low is None
    assert estimate.high is not None
    assert estimate.high < 1350, "a hundred lost games say it is weaker than the weakest"
    assert estimate.describe() == f"below 1350, at most {estimate.high:.0f}"


def test_winning_every_game_is_above_the_ceiling_with_only_a_bottom_to_the_range():
    estimate = estimate_rating([Result(opponent, 20, 20) for opponent in LADDER])

    assert estimate.beyond is Beyond.CEILING
    assert estimate.rating is None
    assert estimate.high is None
    assert estimate.low > 2100
    assert estimate.describe() == f"above 2100, at least {estimate.low:.0f}"


def test_two_lost_games_leave_the_range_above_the_floor():
    estimate = estimate_rating([Result(1350, 0, 2)])

    assert estimate.beyond is Beyond.FLOOR
    assert estimate.high > 1350, "two games cannot show it is weaker than the floor"


def test_an_estimate_extrapolated_below_the_floor_is_reported_as_below_it():
    estimate = estimate_rating([Result(1350, 0.5, 20), Result(1500, 0, 20)])

    assert estimate.beyond is Beyond.FLOOR
    assert estimate.rating is None
    assert estimate.low is None
    assert estimate.high is not None


def test_an_estimate_extrapolated_above_the_ceiling_is_reported_as_above_it():
    estimate = estimate_rating([Result(1350, 20, 20), Result(1500, 19.5, 20)])

    assert estimate.beyond is Beyond.CEILING
    assert estimate.rating is None
    assert estimate.high is None
    assert estimate.low is not None


def test_one_drawn_game_gives_its_opponents_rating_with_a_wide_range():
    estimate = estimate_rating([Result(1700, 0.5, 1)])

    assert estimate.rating == pytest.approx(1700, abs=0.01)
    assert estimate.high - estimate.low > 800


def test_one_won_game_is_above_the_ceiling():
    estimate = estimate_rating([Result(1700, 1, 1)])

    assert estimate.beyond is Beyond.CEILING
    assert estimate.low < 1700, "one game cannot show it is stronger than its opponent"


def test_opponents_with_no_games_are_left_out():
    estimate = estimate_rating([Result(1350, 0, 0), Result(1700, 5, 10), Result(2100, 0, 0)])

    assert (estimate.floor, estimate.ceiling) == (1700, 1700)
    assert estimate.rating == pytest.approx(1700, abs=0.01)


def test_the_same_opponent_twice_counts_as_one():
    twice = estimate_rating([Result(1500, 7, 10), Result(1500, 5, 10)])
    once = estimate_rating([Result(1500, 12, 20)])

    assert twice.rating == pytest.approx(once.rating)
    assert twice.low == pytest.approx(once.low)


@pytest.mark.parametrize("results", [[], [Result(1500, 0, 0)]], ids=["no results", "no games"])
def test_a_rating_needs_a_game(results):
    with pytest.raises(ValueError, match="at least one game"):
        estimate_rating(results)


@pytest.mark.parametrize(
    ("points", "games"), [(-1, 10), (11, 10), (0, -1)], ids=["negative", "too many", "no games"]
)
def test_a_result_cannot_score_more_than_its_games(points, games):
    with pytest.raises(ValueError, match="is not a result"):
        Result(1500, points, games)


@pytest.mark.parametrize("confidence", [0, 1, 1.5])
def test_the_confidence_is_a_probability(confidence):
    with pytest.raises(ValueError, match="between 0 and 1"):
        estimate_rating([Result(1500, 5, 10)], confidence=confidence)


def test_a_described_estimate_gives_the_rating_and_its_range():
    estimate = estimate_rating([Result(1600, 10, 20)])

    assert estimate.describe() == f"1600 ({estimate.low:.0f} to {estimate.high:.0f})"


def round_robin(rng: random.Random, ratings: list[float], games: int) -> list[Pairing]:
    """Every pair of ``ratings`` playing ``games`` games drawn from the Elo curve."""
    return [
        Pairing(
            first=i, second=j, points=play(rng, ratings[i], ratings[j], games).points, games=games
        )
        for i in range(len(ratings))
        for j in range(i + 1, len(ratings))
    ]


def test_a_round_robin_recovers_the_entrants_order_and_the_gaps_between_them():
    true = [1200.0, 1450.0, 1500.0, 1700.0, 2000.0]

    standings = rate_round_robin(len(true), round_robin(random.Random(1), true, games=200))

    ratings = [standing.rating for standing in standings]
    assert sorted(range(len(true)), key=ratings.__getitem__) == list(range(len(true)))
    for i in range(1, len(true)):
        assert ratings[i] - ratings[0] == pytest.approx(true[i] - true[0], abs=40)


def test_anchoring_an_entrant_moves_every_rating_by_as_much_and_measures_ranges_from_it():
    pairings = round_robin(random.Random(2), [1300.0, 1500.0, 1800.0], games=40)
    relative = rate_round_robin(3, pairings)

    anchored = rate_round_robin(3, pairings, anchor=(1, 1600.0))

    assert anchored[1].rating == 1600.0
    assert (anchored[1].low, anchored[1].high) == (1600.0, 1600.0), "it is where it was put"
    shift = 1600.0 - relative[1].rating
    for before, after in zip(relative, anchored, strict=True):
        assert after.rating == pytest.approx(before.rating + shift)
    # Measured from one entrant rather than from the average of all, the others' ranges carry
    # that entrant's uncertainty as well as their own.
    for i in (0, 2):
        assert anchored[i].high - anchored[i].low > relative[i].high - relative[i].low
        assert anchored[i].low < anchored[i].rating < anchored[i].high


def test_an_entrant_who_won_every_game_still_has_a_rating_above_the_rest():
    pairings = [Pairing(0, 1, points=10, games=10), Pairing(0, 2, 10, 10), Pairing(1, 2, 5, 10)]

    standings = rate_round_robin(3, pairings)

    assert math.isfinite(standings[0].rating)
    assert standings[0].low > max(standings[1].rating, standings[2].rating)


def test_the_ranges_hold_the_true_ratings_about_as_often_as_they_claim():
    true = [1400.0, 1500.0, 1650.0, 1900.0]
    rng = random.Random(3)
    trials, covered, checked = 200, 0, 0
    for _ in range(trials):
        standings = rate_round_robin(4, round_robin(rng, true, games=20), anchor=(0, true[0]))
        for standing, rating in zip(standings[1:], true[1:], strict=True):
            checked += 1
            covered += standing.low <= rating <= standing.high
    assert covered / checked == pytest.approx(0.95, abs=0.04)


def test_more_games_narrow_the_ranges():
    true = [1400.0, 1500.0, 1650.0]
    few = rate_round_robin(3, round_robin(random.Random(4), true, games=10))
    many = rate_round_robin(3, round_robin(random.Random(4), true, games=200))

    for short, long in zip(few, many, strict=True):
        assert long.high - long.low < (short.high - short.low) / 3


def test_entrants_the_pairings_do_not_connect_cannot_be_rated():
    with pytest.raises(ValueError, match="no way to compare"):
        rate_round_robin(4, [Pairing(0, 1, 5, 10), Pairing(2, 3, 5, 10)])
