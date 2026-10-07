"""Dataset filters: which games a build keeps, and which of their positions are trained on."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from dataset_helpers import TINY_SHARDS, build, fixture, shard_bytes
from pydantic import Field

from chess_ai.dataset import (
    SPLITS,
    TRAIN,
    VALIDATION,
    DatasetError,
    SkipReason,
    builder,
    load_manifest,
    open_dataset,
    summarize,
)
from chess_ai.dataset.filters import (
    Termination,
    date_bounds,
    game_period,
    sampled,
    termination_of,
)
from chess_ai.dataset.games import split_of
from chess_ai.dataset.manifest import Filters
from chess_ai.dataset.records import PositionFlags
from chess_ai.dataset.store import dataset_path

OPENING = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
"""Six plies, three for each side, which every made-up game here plays unless told otherwise."""

_SITES = itertools.count()


def game(
    *,
    white: int | None = 1500,
    black: int | None = 1500,
    time_control: str = "600+0",
    termination: str | None = "Normal",
    date: str | None = "2024.01.05",
    result: str = "1-0",
    moves: list[str] = OPENING,
    clocks: list[int] | None = None,
    bad_move: bool = False,
) -> str:
    """One game as PGN, with whatever headers and clock comments a test needs.

    A rating of ``None`` leaves the header out. ``clocks`` is the seconds left after each move,
    written as Lichess writes them. ``bad_move`` ends the moves with one that cannot be played.
    """
    headers = {
        "Event": "Rated game",
        "Site": f"https://lichess.org/{next(_SITES):08d}",
        "Date": date,
        "White": "white",
        "Black": "black",
        "Result": result,
        "WhiteElo": None if white is None else str(white),
        "BlackElo": None if black is None else str(black),
        "TimeControl": time_control,
        "Termination": termination,
    }
    lines = [f'[{name} "{value}"]' for name, value in headers.items() if value is not None]
    tokens = []
    for ply, move in enumerate(moves):
        if ply % 2 == 0:
            tokens.append(f"{ply // 2 + 1}.")
        tokens.append(move)
        if clocks is not None:
            seconds = clocks[ply]
            tokens.append(
                f"{{ [%clk {seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}] }}"
            )
    if bad_move:
        tokens.append("Ke7e2")
    return "\n".join(lines) + "\n\n" + " ".join([*tokens, result]) + "\n\n"


def pgn(tmp_path: Path, *games: str, name: str = "games.pgn") -> str:
    """``games`` written to one file, as a build takes it."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(games))
    return str(path)


def built(tmp_path: Path, *games: str, **filters):
    """A dataset of ``games`` built with ``filters``, everything in its train split."""
    return build(
        tmp_path,
        pgn(tmp_path, *games),
        validation_fraction=0.0,
        filters=Filters(**filters),
    )


def target_positions(tmp_path: Path, split: str = TRAIN) -> np.ndarray:
    """Every training target of ``split`` of the dataset "test", as position records."""
    with open_dataset("test", data_dir=tmp_path) as dataset:
        reader = dataset[split]
        return reader.positions(reader.target_positions(np.arange(reader.targets)))


def stored_positions(tmp_path: Path, split: str = TRAIN) -> np.ndarray:
    with open_dataset("test", data_dir=tmp_path) as dataset:
        reader = dataset[split]
        return reader.positions(np.arange(len(reader)))


# --- What a game's headers say ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Normal", Termination.NORMAL),
        ("Time forfeit", Termination.TIME_FORFEIT),
        ("Abandoned", Termination.ABANDONED),
        ("Rules infraction", Termination.RULES_INFRACTION),
        ("Unterminated", Termination.UNTERMINATED),
        ("alice won by resignation", Termination.NORMAL),
        ("bob won by checkmate", Termination.NORMAL),
        ("Game drawn by agreement", Termination.NORMAL),
        ("Game drawn by repetition", Termination.NORMAL),
        ("alice won on time", Termination.TIME_FORFEIT),
        ("Game drawn by timeout vs insufficient material", Termination.TIME_FORFEIT),
        ("bob won - game abandoned", Termination.ABANDONED),
        ("checkmate", Termination.NORMAL),
        ("fifty-move rule", Termination.NORMAL),
        ("abandoned", Termination.ABANDONED),
        ("", Termination.UNKNOWN),
        (None, Termination.UNKNOWN),
        ("the moon fell", Termination.UNKNOWN),
    ],
)
def test_a_termination_header_names_how_the_game_ended(header, expected):
    assert termination_of(header) is expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("NoTimeout won by checkmate", Termination.NORMAL),
        ("Abandonia won by resignation", Termination.NORMAL),
        ("Checkmate4U won on time", Termination.TIME_FORFEIT),
        ("Resignation won - game abandoned", Termination.ABANDONED),
        ("Player NoTimeout won by checkmate", Termination.NORMAL),
        ("Player 1 won on time", Termination.TIME_FORFEIT),
        ("Player won Timeout won by checkmate", Termination.NORMAL),
    ],
)
def test_a_winners_name_does_not_say_how_the_game_ended(header, expected):
    # chess.com starts the sentence with the winner's name, and names contain any word -- and
    # spaces, in an anonymised or hand-edited export.
    assert termination_of(header) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024", (20240101, 20241231)),
        ("2024-02", (20240201, 20240229)),
        ("2023-02", (20230201, 20230228)),
        ("2024-06-15", (20240615, 20240615)),
    ],
)
def test_a_date_covers_the_whole_of_what_it_names(text, expected):
    assert date_bounds(text) == expected


@pytest.mark.parametrize("text", ["24", "2024-13", "2024-02-30", "2024/01", "0000", "yesterday"])
def test_a_date_that_is_not_one_is_refused(text):
    with pytest.raises(ValueError, match="is not a date"):
        date_bounds(text)


def test_a_partial_date_could_be_any_day_of_what_it_knows():
    assert game_period(20240300) == (20240301, 20240331)
    assert game_period(20240000) == (20240101, 20241231)
    assert game_period(20240315) == (20240315, 20240315)
    assert game_period(0) is None


# --- The filters, as written down -------------------------------------------------------------


def test_no_filters_are_written_as_none():
    assert Filters().model_dump() == {}
    assert Filters.model_validate_json(Filters().model_dump_json()) == Filters()


def test_the_filters_set_are_written_and_read_back():
    filters = Filters.model_validate(
        {"min_rating": 2000, "from": "2024-01", "time_controls": ["classical", "rapid"]}
    )

    assert filters.model_dump() == {
        "min_rating": 2000,
        "time_controls": ["rapid", "classical"],
        "from": "2024-01",
    }
    assert Filters.model_validate(filters.model_dump()) == filters


def test_a_filter_of_zero_is_still_written_down():
    # 0 == False in Python, and --min-rating 0 means "rated players only".
    filters = Filters(min_rating=0, max_rating=0, min_clock=0)

    assert filters.model_dump() == {"min_rating": 0, "max_rating": 0, "min_clock": 0}
    assert Filters.model_validate(filters.model_dump()) == filters


def test_the_filters_are_written_the_same_by_alias():
    # Which is how the web server sends them.
    filters = Filters.model_validate({"from": "2024-01", "min_rating": 0})

    assert filters.model_dump(by_alias=True) == {"min_rating": 0, "from": "2024-01"}
    assert Filters().model_dump(by_alias=True) == {}


def test_a_field_renamed_only_on_the_way_out_still_dumps():
    # A dump by alias uses the serialization alias, which is not always the alias.
    class Renamed(Filters):
        until: str | None = Field(default=None, serialization_alias="to")

    assert Renamed(until="2024").model_dump(by_alias=True) == {"to": "2024"}
    assert Renamed().model_dump(by_alias=True) == {}


def test_a_minimum_rating_of_zero_shows_in_the_summary(tmp_path):
    manifest = built(tmp_path, game(white=None, black=1500), min_rating=0)

    on_disk = load_manifest(dataset_path(tmp_path, "test"))
    assert on_disk.filters == manifest.filters == Filters(min_rating=0)
    assert "player to move rated at least 0" in summarize(on_disk)


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"min_rating": 2000, "max_rating": 1900}, "above the maximum"),
        ({"time_controls": ["blitz", "hyperbullet"]}, "no such time control"),
        ({"time_controls": []}, "keep no games"),
        ({"exclude_terminations": ["flagged"]}, "no such termination"),
        ({"exclude_terminations": ["rules_infraction"]}, "always left out"),
        ({"from": "2024-06", "until": "2024-01"}, "is empty"),
        ({"until": "2024-02-30"}, "is not a date"),
        ({"sample": 0}, "greater than 0"),
        ({"sample": 1.5}, "less than or equal to 1"),
        ({"max_games": 0}, "greater than 0"),
        # TOML has a literal inf, and JSON would write it as null: a filter lost on the way out.
        ({"min_clock": float("inf")}, "finite number"),
        ({"min_clock": float("nan")}, "finite number"),
    ],
)
def test_filters_that_make_no_sense_are_refused(filters, message):
    with pytest.raises(ValueError, match=message):
        Filters.model_validate(filters)


# --- Games ended for cheating -----------------------------------------------------------------


def test_a_game_ended_for_a_rules_infraction_is_always_left_out(tmp_path):
    manifest = built(tmp_path, game(), game(termination="Rules infraction"))

    assert manifest.games == 1
    assert manifest.skipped == {SkipReason.RULES_INFRACTION.value: 1}
    assert manifest.filters == Filters(), "it is not a filter: there is none to turn off"


# --- The rating of the player to move ---------------------------------------------------------


def test_a_game_where_neither_player_passes_the_rating_is_left_out(tmp_path):
    manifest = built(
        tmp_path, game(white=2300, black=2100), game(white=1800, black=1900), min_rating=2000
    )

    assert manifest.games == 1
    assert manifest.filtered == {"rating": 1}


def test_only_the_positions_of_the_player_who_passes_are_trained_on(tmp_path):
    manifest = built(tmp_path, game(white=2300, black=1800), min_rating=2000)

    stored = stored_positions(tmp_path)
    targets = target_positions(tmp_path)
    assert len(stored) == len(OPENING), "the whole game is stored, for the history around it"
    assert len(targets) == 3
    assert set(targets["mover_rating"].tolist()) == {2300}
    assert (targets["flags"] & PositionFlags.WHITE_TO_MOVE).all()
    assert manifest.splits[TRAIN].targets == 3
    assert manifest.trained_on == 3
    assert manifest.not_targets == {"rating": 3}


@pytest.mark.parametrize(
    ("limits", "kept_ratings"),
    [
        ({"min_rating": 1500, "max_rating": 1999}, {1500, 1999}),
        ({"min_rating": 1500}, {1500, 1999, 2400}),
        ({"max_rating": 1999}, {1200, 1500, 1999}),
        ({}, {1200, 1500, 1999, 2400, None}),
    ],
)
def test_a_rating_band_has_either_end_or_neither(tmp_path, limits, kept_ratings):
    # Each game has the same player on both sides, so a game is kept exactly when its rating is.
    ratings = [1200, 1500, 1999, 2400, None]
    built(tmp_path, *(game(white=rating, black=rating) for rating in ratings), **limits)

    targets = target_positions(tmp_path)
    unknown = (targets["flags"] & PositionFlags.MOVER_RATING_UNKNOWN) != 0
    found = set(targets["mover_rating"][~unknown].tolist())
    if unknown.any():
        found.add(None)
    assert found == kept_ratings


def test_an_unknown_rating_fails_a_limit_unless_it_is_let_through(tmp_path):
    games = [game(white=None, black=None), game(white=2100, black=2100)]

    assert built(tmp_path / "strict", *games, min_rating=2000).games == 1
    assert (
        built(tmp_path / "lenient", *games, min_rating=2000, unknown_rating_passes=True).games == 2
    )


# --- Time controls, terminations and dates ----------------------------------------------------


def test_only_the_time_controls_asked_for_are_kept(tmp_path):
    manifest = built(
        tmp_path,
        game(time_control="60+0"),
        game(time_control="300+0"),
        game(time_control="900+10"),
        game(time_control="5400+30"),
        game(time_control="-"),
        game(time_control="?"),
        time_controls=["rapid", "classical"],
    )

    assert manifest.statistics.time_controls == {"rapid": 1, "classical": 1}
    assert manifest.filtered == {"time_control": 4}


def test_an_unknown_time_control_is_kept_only_when_asked_for(tmp_path):
    manifest = built(
        tmp_path,
        game(time_control="?"),
        game(time_control="60+0"),
        time_controls=["unknown"],
    )

    assert manifest.statistics.time_controls == {"unknown": 1}


def test_the_terminations_excluded_are_left_out(tmp_path):
    manifest = built(
        tmp_path,
        game(termination="Normal"),
        game(termination="Time forfeit"),
        game(termination="carol won on time"),
        game(termination="Abandoned"),
        exclude_terminations=["time_forfeit"],
    )

    assert manifest.games == 2
    assert manifest.filtered == {"termination": 2}


@pytest.mark.parametrize(
    ("date", "kept"),
    [
        ("2024.03.15", True),
        ("2024.03.01", True),
        ("2024.06.30", True),
        ("2024.02.29", False),
        ("2024.07.01", False),
        ("2024.04.??", True),
        ("2024.03.??", True),
        ("2024.??.??", False),
        ("????.??.??", False),
        (None, False),
    ],
)
def test_a_date_range_keeps_the_games_known_to_be_inside_it(tmp_path, date, kept):
    # Beside a game that is always kept, because a build that keeps nothing is refused.
    manifest = built(
        tmp_path,
        game(date="2024.05.01"),
        game(date=date),
        **{"from": "2024-03", "until": "2024-06"},
    )

    assert manifest.games == 1 + kept
    assert manifest.filtered == ({} if kept else {"date": 1})


def test_a_partial_date_needs_all_of_it_inside_the_range(tmp_path):
    manifest = built(
        tmp_path, game(date="2024.05.01"), game(date="2024.03.??"), **{"from": "2024-03-15"}
    )

    assert manifest.filtered == {"date": 1}


# --- The clock --------------------------------------------------------------------------------


def test_a_position_whose_mover_was_short_of_time_is_not_trained_on(tmp_path):
    # The clock after each move. What the mover had when choosing is what their previous move
    # left them: white had 60s at ply 2 and 20s at ply 4, black 50s at ply 3 and 10s at ply 5.
    clocks = [60, 50, 20, 10, 5, 3]
    manifest = built(tmp_path, game(clocks=clocks), min_clock=30)

    targets = target_positions(tmp_path)
    assert targets["ply"].tolist() == [0, 1, 2, 3], "each side started with the game's 600s"
    assert manifest.not_targets == {"clock": 2}


def test_a_game_without_clocks_passes_the_clock(tmp_path):
    manifest = built(tmp_path, game(), game(time_control="?"), min_clock=30)

    assert manifest.trained_on == 2 * len(OPENING)


def test_a_sides_first_move_is_timed_by_the_time_control(tmp_path):
    # A 15-second game: nobody ever had 30 seconds, not even before their first move.
    manifest = built(
        tmp_path, game(), game(time_control="15+0", clocks=[14, 13, 10, 9, 5, 3]), min_clock=30
    )

    assert manifest.games == 1
    assert manifest.filtered == {"clock": 1}


def test_a_game_whose_rated_player_never_moves_is_left_out_for_the_rating(tmp_path):
    manifest = built(
        tmp_path,
        game(white=2100, black=2100),
        game(white=1000, black=2100, moves=["e4"]),
        min_rating=2000,
    )

    assert manifest.filtered == {"rating": 1}


# --- Sampling and the maximum -----------------------------------------------------------------


def test_a_smaller_sample_is_part_of_a_larger_one():
    identities = [f"game {n}" for n in range(5000)]

    small = {identity for identity in identities if sampled(identity, 0.05)}
    large = {identity for identity in identities if sampled(identity, 0.10)}

    assert small < large
    assert 150 < len(small) < 350


def test_a_sample_does_not_follow_the_split():
    # Hashed the same way, a 5% sample would hold every 2% validation game, and be 40% of them.
    identities = [f"game {n}" for n in range(20000)]
    sample = [identity for identity in identities if sampled(identity, 0.05)]

    held_back = sum(split_of(identity, 0.02) for identity in sample) / len(sample)

    assert held_back < 0.05


def test_a_sample_keeps_the_same_games_every_time(tmp_path):
    games = [game() for _ in range(60)]

    first = built(tmp_path / "a", *games, sample=0.5)
    second = built(tmp_path / "b", *games, sample=0.5)

    assert first.games == second.games
    assert 10 < first.games < 50
    assert first.filtered == {"sample": 60 - first.games}
    assert shard_bytes(dataset_path(tmp_path / "a", "test")) == shard_bytes(
        dataset_path(tmp_path / "b", "test")
    )


def test_the_maximum_keeps_the_first_games_and_stops_reading(tmp_path):
    sites = [game(white=1500 + n, black=1500 + n) for n in range(20)]

    manifest = built(tmp_path, *sites, max_games=5)

    assert manifest.games == 5
    assert manifest.reached_max_games
    assert manifest.sources[0].games_read == 5, "nothing past the fifth game was read"
    assert sorted(set(stored_positions(tmp_path)["mover_rating"].tolist())) == [
        1500,
        1501,
        1502,
        1503,
        1504,
    ]


def test_the_maximum_counts_games_after_the_other_filters(tmp_path):
    games = [game(time_control="60+0"), game(), game(time_control="60+0"), game(), game()]

    manifest = built(tmp_path, *games, max_games=2, time_controls=["rapid"])

    assert manifest.games == 2
    assert manifest.filtered == {"time_control": 2}
    assert manifest.sources[0].games_read == 4


def test_a_maximum_the_sources_do_not_reach_is_not_reached(tmp_path):
    manifest = built(tmp_path, game(), game(), max_games=5)

    assert manifest.games == 2
    assert not manifest.reached_max_games


def test_the_sources_after_the_maximum_are_not_read(tmp_path):
    first = pgn(tmp_path, game(), game(), name="a.pgn")
    second = pgn(tmp_path, game(), name="b.pgn")

    manifest = build(tmp_path, first, second, filters=Filters(max_games=2))

    assert [source.games_read for source in manifest.sources] == [2, 0]
    assert all(source.error is None for source in manifest.sources)


# --- Header filters and broken games ----------------------------------------------------------


def test_a_game_the_headers_leave_out_is_not_parsed_and_counted_as_filtered(tmp_path):
    # Its moves cannot be played, which the build would count as an illegal move had it parsed
    # them. It did not: the time control had already decided.
    manifest = built(
        tmp_path, game(), game(time_control="60+0", bad_move=True), time_controls=["rapid"]
    )

    assert manifest.filtered == {"time_control": 1}
    assert manifest.skipped == {}


def test_a_game_that_is_not_chess_is_skipped_as_such_whatever_the_filters(tmp_path):
    manifest = built(tmp_path, game(), game(bad_move=True), min_rating=1000)

    assert manifest.skipped == {SkipReason.ILLEGAL_MOVE.value: 1}


def test_a_build_the_filters_leave_empty_says_so(tmp_path):
    with pytest.raises(DatasetError, match="the filters left out all 2 games"):
        built(tmp_path, game(), game(), min_rating=3000)


# --- Reading in several processes -------------------------------------------------------------


def _many_games() -> list[str]:
    """Games varied enough that every filter has something to do in every piece."""
    games = []
    for n in range(120):
        games.append(
            game(
                white=1400 + 13 * n % 900,
                black=1500 + 7 * n % 900,
                time_control=["60+0", "300+0", "900+10"][n % 3],
                clocks=[600 - 7 * ply - n % 50 for ply in range(len(OPENING))] if n % 2 else None,
                termination=["Normal", "Time forfeit", "Abandoned"][n % 3 - 1],
            )
        )
    return games


@pytest.mark.parametrize(
    "filters",
    [
        {"min_rating": 1800, "min_clock": 560},
        {"max_games": 37, "sample": 0.7, "time_controls": ["blitz", "rapid"]},
        {"max_games": 1, "min_rating": 1900},
    ],
)
def test_reading_in_several_processes_gives_the_same_filtered_dataset(tmp_path, filters):
    source = pgn(tmp_path, *_many_games())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 900)
        parallel = build(
            tmp_path / "parallel", source, filters=Filters(**filters), workers=3, shards=TINY_SHARDS
        )
    serial = build(
        tmp_path / "serial", source, filters=Filters(**filters), workers=1, shards=TINY_SHARDS
    )

    assert shard_bytes(dataset_path(tmp_path / "parallel", "test")) == shard_bytes(
        dataset_path(tmp_path / "serial", "test")
    )
    ignored = {"created", "sources"}
    assert parallel.model_dump(exclude=ignored) == serial.model_dump(exclude=ignored)
    assert [s.games_read for s in parallel.sources] == [s.games_read for s in serial.sources]


# --- What the manifest says, and reading it back ----------------------------------------------


def test_the_filters_are_read_back_from_the_manifest_as_they_were_written(tmp_path):
    filters = {"min_rating": 0, "from": "2024", "until": "2024-12", "min_clock": 0.5}
    manifest = built(tmp_path, game(white=None, black=1500), **filters)

    written = json.loads((dataset_path(tmp_path, "test") / "manifest.json").read_text())
    assert written["filters"] == filters
    assert load_manifest(dataset_path(tmp_path, "test")).filters == manifest.filters


def test_the_manifest_records_the_filters_and_stats_shows_them(tmp_path):
    manifest = built(
        tmp_path,
        game(white=2300, black=1800),
        game(time_control="60+0", white=2300),
        min_rating=2000,
        time_controls=["rapid"],
        exclude_terminations=["abandoned"],
        **{"from": "2024", "until": "2024-06"},
    )

    on_disk = load_manifest(dataset_path(tmp_path, "test"))
    assert on_disk.filters == manifest.filters
    assert on_disk.filters.min_rating == 2000
    summary = summarize(on_disk)
    assert "player to move rated at least 2000; an unknown rating does not" in summary
    assert "time controls rapid" in summary
    assert "not ended by abandoned" in summary
    assert "played from 2024 until 2024-06" in summary
    assert "trained on 3 (train 3, validation 0) positions" in summary
    assert "games the filters left out  1" in summary
    assert "positions stored but not trained on  3" in summary


def test_a_dataset_without_filters_says_so(tmp_path):
    build(tmp_path, fixture("lichess.pgn"))

    summary = summarize(load_manifest(dataset_path(tmp_path, "test")))
    assert "filters    none: every game in every source" in summary
    assert "trained on" not in summary


def test_an_unfiltered_dataset_has_no_target_stream(tmp_path):
    manifest = build(tmp_path, fixture("lichess.pgn"))

    assert all(counts.targets is None for counts in manifest.splits.values())
    assert not list(dataset_path(tmp_path, "test").rglob("targets"))


def test_a_dataset_of_the_first_format_reads_as_unfiltered(tmp_path):
    build(tmp_path, fixture("lichess.pgn"))
    path = dataset_path(tmp_path, "test") / "manifest.json"
    path.write_text(path.read_text().replace('"format_version": 2', '"format_version": 1'))

    with open_dataset("test", data_dir=tmp_path) as dataset:
        assert dataset.manifest.format_version == 1
        assert dataset.manifest.filters == Filters()
        for split in SPLITS:
            reader = dataset[split]
            assert reader.targets == len(reader)
            every = np.arange(len(reader))
            assert reader.target_positions(every).tolist() == every.tolist()


def test_targets_land_in_the_split_their_game_went_to(tmp_path):
    games = [game(white=2300, black=1800) for _ in range(40)]

    manifest = build(
        tmp_path, pgn(tmp_path, *games), validation_fraction=0.3, filters=Filters(min_rating=2000)
    )

    assert manifest.splits[VALIDATION].games > 0
    for split in SPLITS:
        targets = target_positions(tmp_path, split)
        assert len(targets) == manifest.splits[split].targets == 3 * manifest.splits[split].games
        assert set(targets["mover_rating"].tolist()) == {2300}
