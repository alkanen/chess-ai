"""Building datasets out of the fixture PGN files: what is kept, what is not, and the split."""

import ast
import errno
import io
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.pgn
import pytest
from dataset_helpers import (
    FIXTURES,
    GOOD_GAMES,
    CountingPath,
    build,
    fixture,
    flat_pgn,
    move_sequences,
    separated_pgn,
    shard_bytes,
)

from chess_ai.dataset import (
    SPLITS,
    TRAIN,
    VALIDATION,
    DatasetError,
    GameFlags,
    PositionFlags,
    Progress,
    ProgressPrinter,
    RatingSource,
    Result,
    SkipReason,
    TimeControl,
    builder,
    list_datasets,
    load_manifest,
    open_dataset,
    sources,
    store,
    unpack_board,
)
from chess_ai.dataset.builder import DEFAULT_VALIDATION_FRACTION
from chess_ai.dataset.games import split_of, time_control_class
from chess_ai.dataset.progress import format_progress
from chess_ai.dataset.records import POSITION_DTYPE
from chess_ai.dataset.sources import (
    ByteRange,
    Source,
    game_ranges,
    games_in_range,
    resolve_sources,
)
from chess_ai.dataset.store import dataset_path
from chess_ai.move_codec import VOCABULARY_SIZE, move_at


def test_a_build_keeps_the_games_it_can_read(tmp_path):
    manifest = build(tmp_path)

    assert manifest.games == GOOD_GAMES
    assert manifest.positions == sum(counts.positions for counts in manifest.splits.values())
    assert manifest.positions > 100


def test_broken_games_are_skipped_and_counted_by_reason(tmp_path):
    # Beside a file that does give games, because a build that keeps none of them is refused.
    manifest = build(tmp_path, "malformed.pgn", "lichess.pgn")

    assert manifest.sources[0].games_kept == 0, "none of the broken ones"
    assert manifest.skipped == {
        SkipReason.NO_MOVES.value: 1,
        SkipReason.NO_RESULT.value: 2,
        SkipReason.ILLEGAL_MOVE.value: 1,
        SkipReason.BAD_POSITION.value: 1,
        SkipReason.UNSUPPORTED_VARIANT.value: 1,
    }
    assert manifest.games_skipped == 6


def test_a_file_of_nothing_but_broken_games_does_not_end_the_build(tmp_path):
    manifest = build(tmp_path, "malformed.pgn", "lichess.pgn")

    assert manifest.games == 4
    assert [(source.games_read, source.games_kept) for source in manifest.sources] == [
        (6, 0),
        (4, 4),
    ]


def test_a_file_that_is_not_pgn_at_all_is_read_without_crashing(tmp_path):
    junk = tmp_path / "junk.pgn"
    junk.write_bytes(bytes(range(256)) * 50)

    manifest = build(tmp_path / "data", str(junk), "lichess.pgn")

    assert manifest.games == 4, "the other source's games, which the junk did not stop"
    assert manifest.sources[0].games_kept == 0
    assert manifest.games_skipped > 0


def test_games_with_missing_ratings_are_kept_and_flagged(tmp_path):
    build(tmp_path, "unrated.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)

    unrated = [
        split.game(index)
        for index in range(split.games)
        if int(split.game(index)["flags"]) & GameFlags.WHITE_RATING_UNKNOWN
    ]

    # The club-night game gives no ratings at all; the correspondence game gives black's only.
    assert len(unrated) == 2
    assert {(int(game["white_rating"]), int(game["black_rating"])) for game in unrated} == {
        (0, 0),
        (0, 2100),
    }
    both_unknown = [game for game in unrated if int(game["flags"]) & GameFlags.BLACK_RATING_UNKNOWN]
    assert len(both_unknown) == 1


def test_a_position_says_whether_its_own_players_were_rated(tmp_path):
    build(tmp_path, "unrated.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)
    half_rated = _game_with(split, lambda game: int(game["black_rating"]) == 2100)

    positions = split.game_positions(half_rated)

    for position in positions:
        white_to_move = bool(position["flags"] & PositionFlags.WHITE_TO_MOVE)
        mover_unknown = bool(position["flags"] & PositionFlags.MOVER_RATING_UNKNOWN)
        # White's rating is the missing one, so it is the mover's on white's turns.
        assert mover_unknown == white_to_move
        assert bool(position["flags"] & PositionFlags.OPPONENT_RATING_UNKNOWN) != white_to_move


def test_the_rating_pool_is_read_from_each_game(tmp_path):
    manifest = build(tmp_path)

    assert manifest.rating_source == "auto"
    assert manifest.statistics.rating_sources == {
        RatingSource.UNKNOWN.name.lower(): 2,
        RatingSource.LICHESS.name.lower(): 5,
        RatingSource.CHESSCOM.name.lower(): 1,
    }


def test_the_rating_pool_can_be_given_for_every_game(tmp_path):
    manifest = build(tmp_path, rating_source=RatingSource.FIDE)

    assert manifest.rating_source == "fide"
    assert manifest.statistics.rating_sources == {"fide": GOOD_GAMES}


def test_the_split_puts_a_whole_game_on_one_side(tmp_path):
    build(tmp_path, validation_fraction=0.5)
    dataset = open_dataset("test", data_dir=tmp_path)

    train = move_sequences(dataset[TRAIN])
    validation = move_sequences(dataset[VALIDATION])

    assert set(train).isdisjoint(validation)
    assert len(train) + len(validation) == GOOD_GAMES
    assert train and validation, "a half-and-half split of eight games should use both sides"


def test_every_position_belongs_to_a_game_of_its_own_split(tmp_path):
    build(tmp_path, validation_fraction=0.5)
    dataset = open_dataset("test", data_dir=tmp_path)

    for name in SPLITS:
        split = dataset[name]
        seen = 0
        for index in range(split.games):
            game = split.game(index)
            for ply, position in enumerate(split.game_positions(index)):
                assert int(position["game"]) == index
                assert int(position["ply"]) == ply
                assert int(position["result"]) == (
                    int(game["result"])
                    if position["flags"] & PositionFlags.WHITE_TO_MOVE
                    else Result(int(game["result"])).opponent
                )
                seen += 1
        assert seen == len(split), "the split's positions are exactly its games' positions"


def test_the_split_is_the_same_whichever_dataset_a_game_is_built_into(tmp_path):
    # The hash is of the game, not of the build, so a game validated against in one dataset
    # cannot be trained on in another.
    everything = build(tmp_path / "all", validation_fraction=0.5)
    lichess_only = build(tmp_path / "some", "lichess.pgn", validation_fraction=0.5)

    held_back = set(move_sequences(open_dataset("test", data_dir=tmp_path / "all")[VALIDATION]))
    also_held_back = set(
        move_sequences(open_dataset("test", data_dir=tmp_path / "some")[VALIDATION])
    )

    assert everything.games == GOOD_GAMES and lichess_only.games == 4
    assert also_held_back <= held_back


def test_the_validation_fraction_decides_how_much_is_held_back():
    identities = [f"game {number}" for number in range(2000)]

    held_back = sum(split_of(identity, 0.25) for identity in identities)

    assert 0.2 < held_back / len(identities) < 0.3
    assert not any(split_of(identity, 0.0) for identity in identities)
    assert all(split_of(identity, 1.0) for identity in identities)


def test_nothing_is_held_back_when_nothing_should_be(tmp_path):
    manifest = build(tmp_path, validation_fraction=0.0)

    assert manifest.splits[VALIDATION].games == 0
    assert manifest.splits[TRAIN].games == GOOD_GAMES


def test_a_fraction_that_is_not_one_is_refused(tmp_path):
    with pytest.raises(DatasetError, match="between 0 and 1"):
        build(tmp_path, validation_fraction=1.5)


def test_the_moves_recorded_are_the_moves_played(tmp_path):
    build(tmp_path)
    dataset = open_dataset("test", data_dir=tmp_path)

    replayed = 0
    for name in SPLITS:
        split = dataset[name]
        for index in range(split.games):
            positions = split.game_positions(index)
            moves = split.move_sequence(index)
            # Replaying a game's own moves from its first position has to walk through the
            # rest of its positions exactly, which is the whole dataset checked against itself.
            board = unpack_board(positions[0])
            for ply, position in enumerate(positions):
                assert board.fen() == unpack_board(position).fen()
                assert moves[ply] == position["move"]
                move = move_at(int(position["move"]))
                assert board.is_legal(move), f"{move} is not legal in {board.fen()}"
                board.push(move)
                replayed += 1
    assert replayed == len(dataset[TRAIN]) + len(dataset[VALIDATION])


def test_a_game_that_started_somewhere_unusual_keeps_where_it_started(tmp_path):
    build(tmp_path, "custom-start.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)
    game = split.game(0)

    assert int(game["flags"]) & GameFlags.CUSTOM_START
    assert split.board(int(game["ply_offset"])).fen() == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert not int(split.game(0)["flags"]) & GameFlags.WHITE_RATING_UNKNOWN


def test_promotions_are_recorded_as_promotions(tmp_path):
    build(tmp_path, "custom-start.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)

    promotions = [
        move_at(int(position["move"]))
        for position in split.game_positions(0)
        if move_at(int(position["move"])).promotion is not None
    ]

    assert [move.uci() for move in promotions] == ["e7e8q"]


def test_time_controls_are_sorted_into_classes(tmp_path):
    manifest = build(tmp_path)

    assert manifest.statistics.time_controls == {
        TimeControl.UNKNOWN.name.lower(): 1,
        TimeControl.BULLET.name.lower(): 1,
        TimeControl.BLITZ.name.lower(): 2,
        TimeControl.RAPID.name.lower(): 2,
        TimeControl.CLASSICAL.name.lower(): 1,
        TimeControl.CORRESPONDENCE.name.lower(): 1,
    }


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, TimeControl.UNKNOWN),
        ("", TimeControl.UNKNOWN),
        ("?", TimeControl.UNKNOWN),
        ("-", TimeControl.CORRESPONDENCE),
        ("1/86400", TimeControl.CORRESPONDENCE),
        ("15", TimeControl.BULLET),
        ("60+0", TimeControl.BULLET),
        ("120+1", TimeControl.BULLET),
        ("180+0", TimeControl.BLITZ),
        ("300+0", TimeControl.BLITZ),
        ("180+3", TimeControl.BLITZ),
        ("600+0", TimeControl.RAPID),
        ("600+5", TimeControl.RAPID),
        ("1800+0", TimeControl.CLASSICAL),
        ("40/7200+30", TimeControl.CLASSICAL),
        ("40/9000:1800+30", TimeControl.CLASSICAL),
        ("*180", TimeControl.UNKNOWN),
        ("nonsense", TimeControl.UNKNOWN),
    ],
)
def test_a_time_control_header_names_a_class(header, expected):
    assert time_control_class(header) == expected


def test_results_are_counted_from_whites_side(tmp_path):
    manifest = build(tmp_path)

    assert manifest.statistics.results == {"1-0": 4, "0-1": 2, "1/2-1/2": 2}
    assert sum(manifest.statistics.results.values()) == GOOD_GAMES


def test_ratings_are_counted_per_player_in_buckets(tmp_path):
    manifest = build(tmp_path)
    statistics = manifest.statistics

    assert sum(statistics.ratings.values()) + statistics.ratings_unknown == GOOD_GAMES * 2
    assert statistics.ratings_unknown == 3
    assert statistics.ratings["2400"] == 2
    assert [int(bucket) for bucket in statistics.ratings] == sorted(
        int(bucket) for bucket in statistics.ratings
    )


def test_the_manifest_records_what_the_dataset_was_built_from(tmp_path):
    when = datetime(2024, 5, 17, 9, 30, tzinfo=UTC)

    manifest = build(tmp_path, "lichess.pgn", "unrated.pgn", now=when)

    assert manifest.name == "test"
    assert manifest.created == when
    assert manifest.format_version == 1
    assert manifest.move_vocabulary_size == VOCABULARY_SIZE
    assert manifest.validation_fraction == DEFAULT_VALIDATION_FRACTION
    assert manifest.filters.model_dump() == {}
    assert [source.path for source in manifest.sources] == [
        fixture("lichess.pgn"),
        fixture("unrated.pgn"),
    ]
    assert [source.bytes for source in manifest.sources] == [
        (FIXTURES / "lichess.pgn").stat().st_size,
        (FIXTURES / "unrated.pgn").stat().st_size,
    ]
    assert load_manifest(dataset_path(tmp_path, "test")) == manifest


def test_a_directory_of_pgn_files_is_read_in_name_order(tmp_path):
    manifest = build(tmp_path)

    assert [source.path for source in manifest.sources] == [
        fixture(name)
        for name in ("custom-start.pgn", "lichess.pgn", "malformed.pgn", "unrated.pgn")
    ]


def test_a_glob_names_the_files_it_matches(tmp_path):
    manifest = build(tmp_path, str(FIXTURES / "*rated*.pgn"))

    assert [source.path for source in manifest.sources] == [fixture("unrated.pgn")]
    assert manifest.games == 3


def test_a_file_named_twice_is_read_once(tmp_path):
    manifest = build(tmp_path, "lichess.pgn", "lichess.pgn")

    assert len(manifest.sources) == 1
    assert manifest.games == 4


def test_a_source_that_matches_nothing_is_an_error(tmp_path):
    with pytest.raises(DatasetError, match="no PGN files match"):
        build(tmp_path, str(tmp_path / "*.pgn"))
    with pytest.raises(DatasetError, match="no such PGN file"):
        build(tmp_path, str(tmp_path / "nowhere.pgn"))


def test_a_name_that_cannot_be_a_directory_is_an_error(tmp_path):
    from chess_ai.dataset import build_dataset

    for name in ("../escape", "with/slash", ".hidden", ""):
        with pytest.raises(DatasetError, match="invalid dataset name"):
            build_dataset(name, [str(FIXTURES)], data_dir=tmp_path)


def test_building_over_a_dataset_needs_saying_so(tmp_path):
    build(tmp_path, "lichess.pgn")

    with pytest.raises(DatasetError, match="already in"):
        build(tmp_path, "unrated.pgn")

    replaced = build(tmp_path, "unrated.pgn", overwrite=True)

    assert replaced.games == 3
    assert [source.path for source in replaced.sources] == [fixture("unrated.pgn")]


def test_the_build_reports_progress_as_it_goes(tmp_path):
    reports: list[Progress] = []

    manifest = build(tmp_path, progress=reports.append)

    assert reports, "a build should say what it is doing"
    final = reports[-1]
    assert final.done
    assert final.games_kept == manifest.games == GOOD_GAMES
    assert final.positions == manifest.positions
    assert final.games_read == manifest.games + manifest.games_skipped
    assert final.bytes_read == final.bytes_total == sum(s.bytes for s in manifest.sources)
    assert final.fraction == 1.0
    assert final.seconds_remaining is None
    assert final.games_per_second > 0


def test_a_failure_nothing_expected_is_counted_and_said_out_loud(tmp_path, caplog):
    # Counting a game as unreadable is right for a broken game and would hide a broken build,
    # so the first unexpected failure has to be visible rather than only counted.
    real_game_records = builder.game_records
    calls = {"n": 0}

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("something nothing expected")
        return real_game_records(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "game_records", explode)
        manifest = build(tmp_path, "lichess.pgn")

    assert manifest.games == 2, "the games it could still read"
    assert manifest.skipped == {SkipReason.UNREADABLE.value: 2}
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "said once, not once per game"
    assert "something nothing expected" in caplog.text


def test_a_split_of_no_games_is_still_a_split(tmp_path):
    # Holding nothing back leaves the validation split with nothing in it, and a reader has to
    # cope with that as readily as with a split that has something. A dataset of no games at all
    # is refused now, so an empty split is the only empty thing a reader can be handed.
    build(tmp_path / "data", "lichess.pgn", validation_fraction=0.0)

    dataset = open_dataset("test", data_dir=tmp_path / "data")

    assert dataset.manifest.games == 4
    assert len(dataset[VALIDATION]) == 0
    assert dataset[VALIDATION].games == 0


def _all_games(data_dir):
    """The training split of a build that held nothing back, so it has every kept game."""
    dataset = open_dataset("test", data_dir=data_dir)
    assert dataset.manifest.validation_fraction == 0.0, "this helper needs a build of all games"
    return dataset[TRAIN]


def _game_with(split, matches):
    """The index of the one game in ``split`` that ``matches``."""
    found = [index for index in range(split.games) if matches(split.game(index))]
    assert len(found) == 1, f"expected one matching game, found {len(found)}"
    return found[0]


def test_a_board_read_back_out_of_a_dataset_plays(tmp_path):
    build(tmp_path)
    dataset = open_dataset("test", data_dir=tmp_path)
    split = dataset[TRAIN]

    board = split.board(0)

    assert isinstance(board, chess.Board)
    assert board.is_valid()


def _raising_append(when: int, error):
    """A ``_StreamWriter.append`` that raises ``error`` on the ``when``-th call."""
    original = store._StreamWriter.append
    calls = {"n": 0}

    def append(self, records):
        calls["n"] += 1
        if calls["n"] == when:
            raise error
        return original(self, records)

    return append


def _failing_append(when: int):
    """A ``_StreamWriter.append`` that fails the ``when``-th call, as a full disk would."""
    return _raising_append(when, OSError(errno.ENOSPC, "No space left on device"))


def test_a_write_that_fails_ends_the_build_rather_than_counting_a_broken_game(tmp_path):
    # A shard that cannot be written is not a broken game: counting it as one and carrying on
    # abandons the rest of the file and then reports a clean build over what was lost.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store._StreamWriter, "append", _failing_append(5))

        with pytest.raises(OSError, match="No space left"):
            build(tmp_path, validation_fraction=0.0)

    assert not dataset_path(tmp_path, "test").exists(), "a build that failed leaves no dataset"
    assert list_datasets(tmp_path) == []


def test_a_report_that_cannot_be_made_does_not_lose_the_build(tmp_path, caplog):
    # What a piped build raises when the reader goes away, and what a closed terminal or a
    # dropped ssh session raises at hour four. Reporting is not part of the dataset: a build
    # nobody is watching any more is still a build worth finishing.
    reports = []

    def broken_pipe(progress):
        reports.append(progress)
        raise BrokenPipeError(32, "Broken pipe")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)

        manifest = build(tmp_path, validation_fraction=0.0, progress=broken_pipe)

    assert manifest.games == GOOD_GAMES
    assert open_dataset("test", data_dir=tmp_path).manifest.games == GOOD_GAMES
    assert len(reports) == 1, "having failed once, it stops trying"
    warnings = [record for record in caplog.records if "progress" in record.message]
    assert len(warnings) == 1, "and says so once"


def test_a_build_that_fails_part_way_leaves_nothing_to_trip_over(tmp_path):
    # An interrupted build used to leave a manifest-less directory that list_datasets and
    # stats both denied existed, and that blocked the obvious retry.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store._StreamWriter, "append", _failing_append(5))
        with pytest.raises(OSError, match="No space left"):
            build(tmp_path, validation_fraction=0.0)

    retried = build(tmp_path, validation_fraction=0.0)

    assert retried.games == GOOD_GAMES
    assert list_datasets(tmp_path) == ["test"]


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a file of any mode"
)
def test_a_source_that_cannot_be_read_is_counted_rather_than_fatal(tmp_path):
    # The files are sized when the build starts and opened hours later, so one going away
    # mid-build must not throw away everything read before it.
    unreadable = tmp_path / "locked.pgn"
    unreadable.write_text('[Event "x"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    unreadable.chmod(0o000)
    try:
        manifest = build(
            tmp_path / "data", str(unreadable), fixture("lichess.pgn"), validation_fraction=0.0
        )
    finally:
        # Not left behind at mode 000, which is a trap for anything that later walks the tree.
        unreadable.chmod(0o600)

    assert manifest.games == 4, "the readable source is still read"
    locked, lichess = manifest.sources
    assert locked.games_read == 0
    assert locked.error is not None and "cannot read" in locked.error
    assert lichess.games_kept == 4 and lichess.error is None


def test_an_impossible_date_is_not_stored_as_a_date(tmp_path):
    # Scraped PGN carries these, and a filter or a chart that reads the field as yyyymmdd
    # has no way to tell 20249999 from a date.
    dated = tmp_path / "dates.pgn"
    dated.write_text(
        "".join(
            f'[Event "x"]\n[Site "s{n}"]\n[Date "{date}"]\n[White "a"]\n[Black "b"]\n'
            f'[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
            for n, date in enumerate(
                ("2024.99.99", "2024.02.30", "2024.13.01", "2023.??.??", "2024.05.17")
            )
        )
    )

    build(tmp_path / "data", str(dated), validation_fraction=0.0)
    split = open_dataset("test", data_dir=tmp_path / "data")["train"]

    assert [int(split.game(index)["date"]) for index in range(split.games)] == [
        20240000,  # No month or day it could mean.
        20240200,  # February has no 30th, but the month is real.
        20240000,  # There is no thirteenth month.
        20230000,  # PGN's own way of saying the day is unknown.
        20240517,
    ]


def test_an_overwrite_keeps_both_datasets_until_the_new_one_is_in_place(tmp_path):
    # Replacing a dataset used to delete the old one and then rename the new one over it, so a
    # delete that died part-way — EACCES on one file, a Ctrl-C during the minutes it takes to
    # unlink 50 GB — left a half-deleted old dataset and no new one.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    removed = []
    real_rmtree = builder.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        removed.append(Path(path).name)
        if Path(path).name == "test":
            raise OSError(errno.EACCES, "Permission denied")
        return real_rmtree(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder.shutil, "rmtree", rmtree)

        replaced = build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert "test" not in removed, "the dataset in place is moved aside, never deleted under itself"
    assert replaced.games == 3
    assert open_dataset("test", data_dir=tmp_path).manifest.games == 3
    assert list_datasets(tmp_path) == ["test"]


def test_two_builds_of_one_name_do_not_share_a_working_directory(tmp_path):
    # They used to: the second build wiped the first one's working directory, the first kept
    # writing into files that were no longer there, and whichever finished first published its
    # own manifest over the other's shards. Two builds in one process shared one too, which is
    # why this is per build rather than per process.
    paths = {store.new_partial_path(tmp_path, "test") for _ in range(3)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "getpid", lambda: 4242)
        paths.add(store.new_partial_path(tmp_path, "test"))

    assert len(paths) == 4
    # None of them can be read as a dataset, whatever a reader does with the directory listing.
    assert all(path.name.startswith(".") for path in paths)
    assert dataset_path(tmp_path, "test") not in paths


def test_a_dataset_that_appears_while_a_build_runs_is_not_silently_replaced(tmp_path):
    # Without --overwrite this build said it would not replace a dataset, and that holds however
    # late it finds out — a dataset restored from a backup or copied in while it was reading.
    build(tmp_path / "elsewhere", "lichess.pgn", validation_fraction=0.0)
    arrived = dataset_path(tmp_path / "elsewhere", "test")

    def let_it_appear(progress):
        if not progress.done and not dataset_path(tmp_path, "test").exists():
            shutil.copytree(arrived, dataset_path(tmp_path, "test"))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)

        with pytest.raises(DatasetError, match="appeared"):
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, progress=let_it_appear)

    assert open_dataset("test", data_dir=tmp_path).manifest.games == 4, "the one that appeared"
    assert list_datasets(tmp_path) == ["test"], "and no rubble beside it"


@pytest.mark.skipif(store.fcntl is None, reason="builds are only serialised on POSIX")
def test_a_second_build_of_one_dataset_refuses_to_start(tmp_path):
    # Two builds of one dataset read the same files for hours and one of them then throws the
    # work away. A cron overlap or a retry started too early should hear about it at the start.
    refused = []

    def build_it_again(progress):
        if progress.done or refused:
            return
        try:
            build(tmp_path, "lichess.pgn", validation_fraction=0.0)
        except DatasetError as e:
            refused.append(str(e))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)

        manifest = build(tmp_path, validation_fraction=0.0, progress=build_it_again)

    assert refused, "the second build should have been turned away"
    assert "already running" in refused[0]
    assert manifest.games == GOOD_GAMES, "and the first one finishes"
    assert open_dataset("test", data_dir=tmp_path).manifest.games == GOOD_GAMES


@pytest.mark.skipif(store.fcntl is None, reason="builds are only serialised on POSIX")
def test_a_build_lets_go_of_its_lock_when_it_is_done(tmp_path):
    # Held by the process rather than by a file that exists, so there is nothing to release by
    # hand and nothing left over to block the next build.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    again = build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert again.games == 3


def test_a_filesystem_that_refuses_to_flush_still_gets_a_dataset(tmp_path):
    # Directory fsync answers EINVAL on several network and FUSE filesystems, and opening a
    # directory at all is refused on Windows. A dataset staged on a NAS mount must not be
    # destroyed by the durability that exists to protect it.
    def unsupported(fd):
        raise OSError(errno.EINVAL, "Invalid argument")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", unsupported)

        manifest = build(tmp_path, validation_fraction=0.0)

    assert manifest.games == GOOD_GAMES
    assert len(open_dataset("test", data_dir=tmp_path)[TRAIN]) == manifest.positions


def test_a_shard_that_cannot_be_flushed_ends_the_build(tmp_path):
    # The other side of it: ENOSPC from fsync means the records never reached the disk, which
    # is a lost dataset rather than a filesystem being fussy.
    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", out_of_space)

        with pytest.raises(OSError, match="No space left"):
            build(tmp_path, validation_fraction=0.0)

    assert list_datasets(tmp_path) == []


def test_an_interrupted_build_is_reported_as_interrupted(tmp_path):
    # Closing the shards flushes them, and on a full disk the flush fails too. A failure while
    # closing must not take the place of the exception already on its way out: a Ctrl-C that
    # surfaces as a disk error sends whoever reads it looking for the wrong problem.
    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    def interrupted(self, records):
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as patch:
        # Files are open by the time this bites, so closing them is what fails next.
        patch.setattr(store._StreamWriter, "append", _failing_append(5))
        with pytest.raises(OSError, match="No space left"):
            build(tmp_path, validation_fraction=0.0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", out_of_space)
        patch.setattr(store._StreamWriter, "append", _raising_append(5, KeyboardInterrupt))

        with pytest.raises(KeyboardInterrupt):
            build(tmp_path, validation_fraction=0.0)

    assert list_datasets(tmp_path) == []


def test_a_build_leaves_another_datasets_working_directory_alone(tmp_path):
    # Dataset names may contain dots, so ".d.foo.…partial" is a build of "d.foo" and not a build
    # of "d" with something after it. A build of "d" used to delete it while it was being written.
    live = store.new_partial_path(tmp_path, "test.extra")
    live.mkdir(parents=True)
    (live / "train").mkdir()
    mine = store.new_partial_path(tmp_path, "test")
    mine.mkdir(parents=True)

    build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    assert live.is_dir(), "a build of 'test' must not touch a build of 'test.extra'"
    assert (live / "train").is_dir()
    assert mine.is_dir(), "nor anything it did not create itself"


def test_a_working_directory_left_behind_is_reported(tmp_path, caplog):
    # Whether it belongs to a build that was killed or to one running on another machine cannot
    # be told from here, so it is named and left alone rather than guessed about and deleted.
    left = store.new_partial_path(tmp_path, "test")
    left.mkdir(parents=True)

    build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    assert str(left) in caplog.text
    assert left.is_dir()


def test_a_dataset_an_interrupted_build_set_aside_is_offered_back(tmp_path, caplog):
    # A dataset the tool would otherwise never mention again: list_datasets and stats both skip
    # a dot-prefixed directory, so without this the user has lost it as far as they can tell. A
    # build that failed left what is in place as it found it, so the offer to rename it back is
    # still true when it is made, and stays true until something else builds this dataset.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    aside = store.replaced_path(store.new_partial_path(tmp_path, "test"))
    dataset_path(tmp_path, "test").rename(aside)
    nothing = tmp_path / "empty.pgn"
    nothing.write_text("")
    caplog.clear()

    with pytest.raises(DatasetError):
        build(tmp_path, str(nothing), validation_fraction=0.0)

    assert str(aside) in caplog.text
    assert "rename it to" in caplog.text, "nothing is in place for it to replace"
    assert load_manifest(aside).games == 4, "and it is still the dataset it was"


@contextmanager
def on_one_stream():
    """A terminal: the progress line and the warnings going to the same place, as they do there."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("chess_ai")
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)


def a_dataset_set_aside(tmp_path):
    """A four-game dataset moved out of the way, with nothing of that name left in place."""
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    aside = store.replaced_path(store.new_partial_path(tmp_path, "test"))
    dataset_path(tmp_path, "test").rename(aside)
    return aside


def test_a_leftover_warning_starts_on_a_line_of_its_own(tmp_path):
    # In a terminal the progress line is rewritten and stays open until something ends it, and
    # these warnings go to the same stream. Run on from "0s left", the path in the one message
    # that will ever mention a set-aside dataset cannot be read, let alone copied out of a log.
    aside = a_dataset_set_aside(tmp_path)

    with on_one_stream() as stream, pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)

        build(
            tmp_path,
            "unrated.pgn",
            validation_fraction=0.0,
            progress=ProgressPrinter(stream, interval=0.0, rewrite=True),
        )

    warned = [line for line in stream.getvalue().splitlines() if str(aside) in line]
    assert len(warned) == 1
    assert warned[0].startswith("chess-ai:"), "and not on the end of the progress line"


def test_a_leftover_warning_starts_on_its_own_line_when_the_build_failed(tmp_path):
    # Nothing has ended the line on this path: only a build that finished says that it did, so
    # the summary that ends it above never comes.
    aside = a_dataset_set_aside(tmp_path)
    build(tmp_path, "custom-start.pgn", validation_fraction=0.0)

    with on_one_stream() as stream, pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)

        with pytest.raises(DatasetError, match="kept no games"):
            build(
                tmp_path,
                "malformed.pgn",
                validation_fraction=0.0,
                overwrite=True,
                progress=ProgressPrinter(stream, interval=0.0, rewrite=True),
            )

    warned = [line for line in stream.getvalue().splitlines() if str(aside) in line]
    assert len(warned) == 1
    assert warned[0].startswith("chess-ai:"), "and not on the end of the progress line"


def test_a_dataset_set_aside_is_not_offered_back_by_the_build_that_supersedes_it(tmp_path, caplog):
    # The build that would make the offer wrong is the one that makes it: it starts with nothing
    # in place — the state an interrupt between the two renames leaves — and publishes a dataset
    # hours later. Answering at the start would put "rename it to <datasets/test> to have it
    # again" in the log of a successful run, where following it loses what that run just built.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    aside = store.replaced_path(store.new_partial_path(tmp_path, "test"))
    dataset_path(tmp_path, "test").rename(aside)
    caplog.clear()

    build(tmp_path, "unrated.pgn", validation_fraction=0.0)

    said = caplog.text
    assert str(aside) in said, "it is still reported"
    assert "rename it to" not in said, "doing so would replace the dataset this build just made"
    assert "newer" in said
    assert load_manifest(dataset_path(tmp_path, "test")).games == 3, "and it is the new one"


@pytest.mark.skipif(store.fcntl is None, reason="builds are only serialised on POSIX")
def test_a_filesystem_that_cannot_lock_does_not_block_every_build(tmp_path):
    # ENOLCK is an NFS export with no lock daemon, and EOPNOTSUPP several FUSE mounts. Neither
    # means a build is running, and turning them into that made the dataset unbuildable for good.
    for code in (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS):
        target = tmp_path / str(code)

        def no_locks(fd, operation, code=code):
            raise OSError(code, os.strerror(code))

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(store.fcntl, "flock", no_locks)

            manifest = build(target, "lichess.pgn", validation_fraction=0.0)

        assert manifest.games == 4, f"errno {code} should not stop a build"


@pytest.mark.skipif(store.fcntl is None, reason="builds are only serialised on POSIX")
def test_a_lock_another_build_holds_still_stops_this_one(tmp_path):
    def held(fd, operation):
        raise OSError(errno.EWOULDBLOCK, os.strerror(errno.EWOULDBLOCK))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store.fcntl, "flock", held)

        with pytest.raises(DatasetError, match="already running"):
            build(tmp_path, "lichess.pgn", validation_fraction=0.0)


def test_an_interrupt_between_the_two_renames_keeps_the_dataset(tmp_path):
    # Ctrl-C is aimed at exactly this moment, and the cost of losing the window is a dataset the
    # tool would never mention again: not in place, and hidden under a dot-prefixed name.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_replace = os.replace

    def replace(source, target, **kwargs):
        result = real_replace(source, target, **kwargs)
        if str(target).endswith(store.REPLACED_SUFFIX):
            # Delivered the instant the old dataset has been moved aside and before anything has
            # taken its place, which is the moment a signal has to be survivable in.
            raise KeyboardInterrupt
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder.os, "replace", replace)

        with pytest.raises(KeyboardInterrupt):
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert list_datasets(tmp_path) == ["test"], "the dataset that was there is still there"
    assert open_dataset("test", data_dir=tmp_path).manifest.games == 4


def test_a_build_never_deletes_a_dataset_an_earlier_one_set_aside(tmp_path):
    # The name a build puts the old dataset under used to be a process id and a counter that
    # restarts at 0, so pid reuse made a later build compute the same name and delete what was
    # there — the dataset it had just told the user how to get back.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    stranded = store.datasets_dir(tmp_path) / f".test.stranded{store.REPLACED_SUFFIX}"
    shutil.copytree(dataset_path(tmp_path, "test"), stranded)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store, "replaced_path", lambda partial: stranded)
        patch.setattr(builder, "replaced_path", lambda partial: stranded)

        with pytest.raises(DatasetError):
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert load_manifest(stranded).games == 4, "the dataset set aside is still there, whole"


def test_a_build_refuses_a_working_directory_that_is_already_there(tmp_path):
    # Also a pid-reuse collision: the build used to make its working directory with
    # exist_ok=True and publish whatever it found in it, so a killed build's shards ended up
    # inside a finished dataset that counted none of them.
    occupied = store.datasets_dir(tmp_path) / f".test.occupied{store.PARTIAL_SUFFIX}"
    (occupied / "train" / "positions").mkdir(parents=True)
    stale = occupied / "train" / "positions" / "00007.bin"
    stale.write_bytes(b"\0" * POSITION_DTYPE.itemsize)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "new_partial_path", lambda data_dir, name: occupied)

        with pytest.raises(DatasetError, match="working directory"):
            build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    assert list_datasets(tmp_path) == [], "nothing published out of a directory it did not make"
    assert stale.is_file(), "and what was there is left for whoever it belongs to"


def test_a_working_directory_name_is_not_reused_by_a_later_process(tmp_path):
    # Two fresh interpreters must not agree on a name, which a process id and a per-process
    # counter do as soon as the process id comes round again.
    code = (
        "from pathlib import Path\n"
        "from chess_ai.dataset.store import new_partial_path\n"
        "import os\n"
        "os.getpid = lambda: 4242\n"  # The same process id, as pid reuse gives.
        "print(new_partial_path(Path('/data'), 'test').name)\n"
    )
    names = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }

    assert len(names) == 3, f"the same name came back in another process: {names}"


def test_nothing_says_the_build_is_done_until_it_is(tmp_path):
    # The summary line used to be printed inside the writer's with block, so a build that then
    # failed to flush, publish or rename had already told the user it was built.
    reports = []

    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)
        patch.setattr(os, "fsync", out_of_space)

        with pytest.raises(OSError, match="No space left"):
            build(tmp_path, validation_fraction=0.0, progress=reports.append)

    assert reports, "it should have reported while reading"
    assert not any(report.done for report in reports), "but never that it had finished"


def test_a_build_interrupted_before_it_starts_leaves_no_working_directory(tmp_path):
    # The working directory is made while the writer is constructed, which used to happen just
    # outside the block that removes it again.
    real_init = builder._Build.__init__

    def interrupted(self, **kwargs):
        real_init(self, **kwargs)
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder._Build, "__init__", interrupted)

        with pytest.raises(KeyboardInterrupt):
            build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    assert store.abandoned_partials(tmp_path, "test") == []


@pytest.mark.skipif(store.fcntl is None, reason="builds are only serialised on POSIX")
def test_a_lock_file_that_cannot_be_opened_does_not_stop_the_build(tmp_path):
    # A data directory shared between users: whoever built first owns the lock file, and the
    # next user cannot open it. Locking is a convenience, so this is not the dataset's problem.
    real_open = os.open

    def refuse_the_lock(path, flags, *args, **kwargs):
        if str(path).endswith(store.LOCK_SUFFIX):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, flags, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "open", refuse_the_lock)

        manifest = build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    assert manifest.games == 4


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write a directory of any mode"
)
def test_a_data_directory_that_cannot_be_written_is_said_plainly(tmp_path):
    # Not a traceback: this is a thing the person running the command can fix.
    datasets = store.datasets_dir(tmp_path)
    datasets.mkdir(parents=True)
    datasets.chmod(0o500)
    try:
        with pytest.raises(DatasetError, match="working directory"):
            build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    finally:
        datasets.chmod(0o700)


@pytest.mark.skipif(store.fcntl is None, reason="builds are only serialised on POSIX")
def test_a_build_that_fails_unserialised_is_not_blamed_on_the_lock(tmp_path):
    # The fallback used to yield from inside the handler for the locking error, so every failure
    # in the build came out chained to "No locks available" and sent the reader after that.
    def no_locks(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store.fcntl, "flock", no_locks)
        patch.setattr(os, "fsync", out_of_space)

        with pytest.raises(OSError, match="No space left") as failure:
            build(tmp_path, validation_fraction=0.0)

    chained = []
    cause = failure.value.__context__
    while cause is not None:
        chained.append(str(cause))
        cause = cause.__context__
    assert not any("No locks available" in text for text in chained), chained


def unopenable(directory: Path, name: str = "gone.pgn") -> Path:
    """A PGN file that resolves now and cannot be opened when the build gets to it."""
    path = directory / name
    path.write_text('[Event "x"]\n[Site "s"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    path.chmod(0o000)
    return path


NOT_ROOT = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a file of any mode"
)


@NOT_ROOT
def test_a_build_that_could_read_nothing_at_all_publishes_nothing(tmp_path):
    # A build whose every source went away read no games. Publishing that is publishing an empty
    # dataset, and with --overwrite it would be publishing it over a working one.
    gone = unopenable(tmp_path)
    try:
        with pytest.raises(DatasetError, match="none of"):
            build(tmp_path / "data", str(gone), validation_fraction=0.0)
    finally:
        gone.chmod(0o600)

    assert list_datasets(tmp_path / "data") == []


@NOT_ROOT
def test_a_source_that_went_away_does_not_replace_a_dataset_that_is_there(tmp_path):
    # The race this is all for: the files are sized at second zero and opened hours later, so a
    # nightly "build --overwrite" over a mount that drops must not swap a whole dataset for the
    # part of one it could still read.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    gone = unopenable(tmp_path)
    try:
        with pytest.raises(DatasetError, match="could not be read"):
            build(
                tmp_path,
                "unrated.pgn",
                str(gone),
                validation_fraction=0.0,
                overwrite=True,
            )
    finally:
        gone.chmod(0o600)

    kept = open_dataset("test", data_dir=tmp_path).manifest
    assert kept.games == 4, "the dataset that was there is the one still there"
    assert [Path(source.path).name for source in kept.sources] == ["lichess.pgn"]


@NOT_ROOT
def test_a_source_that_went_away_still_builds_a_dataset_that_is_not_there_yet(tmp_path):
    # Nothing to lose here, so the games that could be read are worth keeping — with the
    # manifest saying the dataset is not the one the sources asked for.
    gone = unopenable(tmp_path)
    try:
        manifest = build(tmp_path / "data", "lichess.pgn", str(gone), validation_fraction=0.0)
    finally:
        gone.chmod(0o600)

    assert manifest.games == 4
    missing = [source for source in manifest.sources if source.went_away]
    assert len(missing) == 1
    assert missing[0].error is not None and "Permission denied" in missing[0].error
    assert missing[0].left_nothing, "it never got as far as a game"
    assert not any(source.went_away for source in manifest.sources if source.error is None)


def test_a_source_that_stops_part_way_through_still_replaces_a_dataset(tmp_path):
    # Not the same thing as a source that could not be opened: bad PGN in the tail of a dump is
    # ordinary, is already counted as skipped games, and must not block every rebuild.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_read_game = chess.pgn.read_game
    calls = {"n": 0}

    def read_game(handle, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("the file stops making sense here")
        return real_read_game(handle, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", read_game)

        replaced = build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert replaced.games == 2, "the games before the file gave up"
    assert replaced.sources[0].error is not None
    assert not replaced.sources[0].went_away, "it was there; its chess stopped making sense"
    assert open_dataset("test", data_dir=tmp_path).manifest.games == 2


def test_a_half_removed_old_dataset_is_reported_as_rubble_not_as_a_dataset(tmp_path):
    # The cleanup past the point of no return ignores errors, so it can stop half way and leave
    # part of the old dataset behind. Advertising that as "rename it back to have it again" would
    # have the user rename a gutted dataset over the good one this build just made.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_rmtree = builder.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if str(path).endswith(store.DISCARDED_SUFFIX):
            # What ignore_errors=True leaves behind when one file will not go: some of the
            # dataset, gone, and no exception to say so.
            (Path(path) / "manifest.json").unlink()
            return None
        return real_rmtree(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder.shutil, "rmtree", rmtree)

        replaced = build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert replaced.games == 3, "the new dataset went in"
    assert store.replaced_datasets(tmp_path, "test") == [], "nothing offers the gutted one back"
    assert store.discarded_datasets(tmp_path, "test"), "it is reported as rubble instead"


def test_a_source_that_goes_away_on_its_first_read_does_not_replace_a_dataset(tmp_path):
    # A mount that drops rarely fails the open — the descriptor is often already cached — it
    # fails the first read. So "could it be opened" is not the question; "did it leave anything
    # behind" is.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_read_game = chess.pgn.read_game

    def stale(handle, **kwargs):
        raise OSError(errno.ESTALE, "Stale file handle")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", stale)

        with pytest.raises(DatasetError, match="nothing to build from"):
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert real_read_game is chess.pgn.read_game, "the patch is undone"
    kept = open_dataset("test", data_dir=tmp_path).manifest
    assert kept.games == 4, "the dataset that was there is the one still there"


def test_a_source_truncated_to_nothing_does_not_replace_a_dataset(tmp_path):
    # No error at all this time: a sync job that truncates a dump mid-build leaves a file the
    # parser reads no games from, which is a build with nothing in it and nothing to complain of.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", lambda handle, **kwargs: None)

        with pytest.raises(DatasetError, match="no games"):
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert open_dataset("test", data_dir=tmp_path).manifest.games == 4


def test_a_build_of_nothing_is_refused_even_where_there_is_nothing_to_lose(tmp_path):
    # The refusals above are about replacing a dataset; this one is not about losing anything.
    # A build that kept no games has not made a dataset, and publishing one with a manifest and
    # an exit status of zero hands the next stage something to open and train on.
    empty = tmp_path / "empty.pgn"
    empty.write_text("")

    with pytest.raises(DatasetError, match="kept no games"):
        build(tmp_path / "data", str(empty), validation_fraction=0.0)

    assert list_datasets(tmp_path / "data") == []
    assert store.abandoned_partials(tmp_path / "data", "test") == [], "and no rubble either"


def test_a_source_that_goes_away_part_way_does_not_replace_a_dataset(tmp_path):
    # A mount can drop after the first game as easily as before it. What tells this from bad PGN
    # is not how many games came back but what failed: the file, or the chess in it.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_read_game = chess.pgn.read_game
    calls = {"n": 0}

    def goes_away(handle, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_read_game(handle, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", goes_away)

        with pytest.raises(DatasetError, match="could not be read"):
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert open_dataset("test", data_dir=tmp_path).manifest.games == 4, "the whole one is kept"


def test_a_source_that_goes_away_is_recorded_as_the_file_and_not_the_chess(tmp_path):
    real_read_game = chess.pgn.read_game
    calls = {"n": 0}

    def goes_away(handle, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_read_game(handle, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", goes_away)

        manifest = build(tmp_path / "data", "unrated.pgn", validation_fraction=0.0)

    source = manifest.sources[0]
    assert manifest.games == 1, "the game it read before the file went"
    assert source.went_away, "the file went away; the PGN in it was fine"
    assert not source.left_nothing, "it did give one before it went"


def test_bad_pgn_part_way_through_is_still_the_files_own_fault(tmp_path):
    # The other side of the same line: a dump whose tail stops making sense is ordinary, its
    # earlier games are in the dataset, and it must not block a rebuild.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_read_game = chess.pgn.read_game
    calls = {"n": 0}

    def nonsense(handle, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("the file stops making sense here")
        return real_read_game(handle, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", nonsense)

        replaced = build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert replaced.games == 2
    assert not replaced.sources[0].went_away
    assert open_dataset("test", data_dir=tmp_path).manifest.games == 2


def test_a_dataset_that_cannot_be_marked_as_discarded_is_left_whole(tmp_path):
    # If it cannot even be renamed within its own parent, deleting it in place would leave a
    # gutted directory still called ".replaced" — the state the rename exists to prevent.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_replace = os.replace

    def replace(source, target, **kwargs):
        if str(target).endswith(store.DISCARDED_SUFFIX):
            raise OSError(errno.EACCES, "Permission denied")
        return real_replace(source, target, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder.os, "replace", replace)

        replaced = build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert replaced.games == 3, "the new dataset went in"
    left = store.replaced_datasets(tmp_path, "test")
    assert left, "the old one is still there"
    assert load_manifest(left[0]).games == 4, "and whole, so renaming it back really would work"


def nonsense_pgn(handle, **kwargs):
    raise ValueError("the first record is not chess")


def test_bad_pgn_in_the_very_first_game_is_not_a_file_that_went_away(tmp_path):
    # The same file and the same corruption as the test above, at the top of the file instead of
    # three games in. Where in the file it happened is not what decides anything: the file was
    # there throughout either way, and those games do not exist to be missed — so it is refused
    # for keeping no games, which is true of it, and not as a file that could not be read.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", nonsense_pgn)

        with pytest.raises(DatasetError, match="kept no games") as refusal:
            build(tmp_path / "data", "unrated.pgn", validation_fraction=0.0)

    assert "could not be read" not in str(refusal.value)
    assert str(FIXTURES / "unrated.pgn") in str(refusal.value), "and it says which source"


def test_bad_pgn_in_the_very_first_game_is_refused_for_the_reason_it_is(tmp_path):
    # Refusing to replace a dataset with an empty one is right; refusing it as a file that
    # "could not be read" was not, because the file was read and had nothing in it.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", nonsense_pgn)

        with pytest.raises(DatasetError, match="kept no games") as refusal:
            build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    assert "could not be read" not in str(refusal.value)
    assert open_dataset("test", data_dir=tmp_path).manifest.games == 4


def test_one_source_of_nothing_but_junk_does_not_block_a_rebuild(tmp_path):
    # The asymmetry that mattered: with another source still giving games, where the junk is in
    # the file decides nothing at all.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_read_game = chess.pgn.read_game
    seen = set()

    def junk_in_one_file(handle, **kwargs):
        if "unrated" in getattr(handle, "name", "") and handle.name not in seen:
            seen.add(handle.name)
            raise ValueError("the first record is not chess")
        return real_read_game(handle, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", junk_in_one_file)

        replaced = build(
            tmp_path, "unrated.pgn", "custom-start.pgn", validation_fraction=0.0, overwrite=True
        )

    assert replaced.games == 1, "the other source's game"
    assert not any(source.went_away for source in replaced.sources)
    assert open_dataset("test", data_dir=tmp_path).manifest.games == 1


def test_a_superseded_dataset_is_not_offered_back_over_a_newer_one(tmp_path, caplog):
    # A dataset left beside a newer one is not one to rename into place, whichever way it got
    # there: doing so would replace the newer one, silently, at the user's own hand.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    real_replace = os.replace

    def replace(source, target, **kwargs):
        if str(target).endswith(store.DISCARDED_SUFFIX):
            raise OSError(errno.EACCES, "Permission denied")
        return real_replace(source, target, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder.os, "replace", replace)

        build(tmp_path, "unrated.pgn", validation_fraction=0.0, overwrite=True)

    caplog.clear()
    build(tmp_path, "custom-start.pgn", validation_fraction=0.0, overwrite=True)

    said = caplog.text
    assert str(store.replaced_datasets(tmp_path, "test")[0]) in said
    assert "interrupted" not in said, "this build finished; it just could not tidy up"
    assert "newer" in said, "and what is in place now is newer than what is being reported"


def test_a_file_cut_into_pieces_gives_up_every_game_exactly_once(tmp_path):
    # Whatever size the pieces are: what a build reads in several processes has to be the games
    # the file holds, not most of them and not one of them twice.
    source = resolve_sources([fixture("lichess.pgn")])[0]
    with source.path.open(encoding="utf-8-sig") as handle:
        whole = []
        while (game := chess.pgn.read_game(handle)) is not None:
            whole.append(str(game))

    for target in (1, 8, 200, 1 << 20):
        pieces = game_ranges(source, 0, target)
        assert sum(piece.bytes for piece in pieces) == source.bytes, f"every byte, at {target}"
        assert [str(game) for piece in pieces for game, _ in games_in_range(piece)] == whole, (
            f"the same games in the same order, at {target}"
        )


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_a_cut_is_never_made_inside_a_game(tmp_path, newline):
    # A comment may hold anything, including something that reads like the start of a game. Cut
    # there and the half before it is kept as a game whose moves stop early, which no count would
    # ever show up: the boundary is a blank line and an [Event, not an [Event.
    #
    # Both line endings, because PGN's own specification says CR/LF and a check that only
    # required the blank line for LF files would be no check at all for the common shape: every
    # line-initial [Event in a CRLF file is preceded by \r\n.
    lines = [
        '[Event "First"]',
        '[Result "1-0"]',
        "",
        "1. e4 { a comment that goes on",
        '[Event "Not really a game"]',
        "and on } e5 2. d4 1-0",
        "",
        '[Event "Second"]',
        '[Result "0-1"]',
        "",
        "1. d4 d5 0-1",
        "",
    ]
    path = tmp_path / "commented.pgn"
    path.write_bytes(newline.join(lines).encode())
    source = resolve_sources([str(path)])[0]

    for target in (1, 8, 32, 200):
        events = [
            game.headers["Event"]
            for piece in game_ranges(source, 0, target)
            for game, _ in games_in_range(piece)
        ]
        assert events == ["First", "Second"], f"two whole games, at {target}"


def test_a_piece_size_of_nothing_is_refused(tmp_path):
    source = resolve_sources([fixture("lichess.pgn")])[0]

    with pytest.raises(ValueError, match="some bytes long"):
        game_ranges(source, 0, 0)


def test_reading_in_several_processes_gives_the_same_dataset_to_the_byte(tmp_path):
    # The whole claim of the parallel path in one test: the shards, not just the counts. A game's
    # ply_offset and a position's game index are the two things order decides, and they are in
    # here, so a piece appended out of turn could not pass this.
    serial, parallel = tmp_path / "one", tmp_path / "many"

    alone = build(serial, workers=1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 200)
        together = build(parallel, workers=4)

    assert shard_bytes(dataset_path(parallel, "test")) == shard_bytes(dataset_path(serial, "test"))
    assert together.model_dump(exclude={"created"}) == alone.model_dump(exclude={"created"}), (
        "and the manifest says the same thing about them, statistics and skip counts and all"
    )


def test_the_split_is_the_same_however_many_processes_read_the_files(tmp_path):
    # split_of is a hash of the game itself for exactly this reason, so it is worth a test that
    # would fail if it ever became a running count.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 120)
        many = build(tmp_path / "many", validation_fraction=0.5, workers=3)
    one = build(tmp_path / "one", validation_fraction=0.5, workers=1)

    with (
        open_dataset("test", data_dir=tmp_path / "many") as parallel,
        open_dataset("test", data_dir=tmp_path / "one") as serial,
    ):
        for split in SPLITS:
            assert move_sequences(parallel[split]) == move_sequences(serial[split])
    assert many.splits == one.splits


def test_a_small_build_reads_in_this_process_whatever_it_was_offered(tmp_path):
    # Starting eight processes to read a directory of exports costs more than reading it does.
    def refuse(*args, **kwargs):
        raise AssertionError("a build this small should not start a process to do it")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "ProcessPoolExecutor", refuse)
        manifest = build(tmp_path, workers=8)

    assert manifest.games == GOOD_GAMES


def test_a_build_needs_at_least_one_worker(tmp_path):
    with pytest.raises(DatasetError, match="at least one worker"):
        build(tmp_path, workers=0)


def test_broken_games_are_counted_the_same_when_workers_read_them(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 100)
        manifest = build(tmp_path / "many", "malformed.pgn", "lichess.pgn", workers=3)
    serial = build(tmp_path / "one", "malformed.pgn", "lichess.pgn", workers=1)

    assert manifest.skipped == serial.skipped
    assert [source.games_read for source in manifest.sources] == [
        source.games_read for source in serial.sources
    ]
    assert [source.games_kept for source in manifest.sources] == [
        source.games_kept for source in serial.sources
    ]


def test_a_source_that_cannot_be_read_is_counted_when_workers_read_them(tmp_path):
    # Cutting the file into pieces is what opens it, so a file that went away fails there rather
    # than in a worker -- and has to be counted exactly as the serial path counts it.
    unreadable = tmp_path / "locked.pgn"
    unreadable.write_text('[Event "x"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    unreadable.chmod(0o000)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
            patch.setattr(builder, "CHUNK_BYTES", 200)
            manifest = build(
                tmp_path / "data",
                str(unreadable),
                fixture("lichess.pgn"),
                validation_fraction=0.0,
                workers=2,
            )
    finally:
        unreadable.chmod(0o600)

    assert manifest.games == 4, "the readable source is still read"
    locked, lichess = manifest.sources
    assert locked.games_read == 0
    assert locked.went_away, "so that an --overwrite does not trade a dataset for this one"
    assert locked.error is not None and "cannot read" in locked.error
    assert lichess.games_kept == 4 and lichess.error is None


def test_a_worker_records_a_failure_nothing_expected_with_its_traceback():
    # The worker half of the contract. Called here rather than through a pool on purpose: a
    # monkeypatch only reaches a worker that inherited this process's memory, which is true
    # under the fork start method and false under spawn (macOS today) and forkserver (Linux
    # from 3.14). A test that needs one of the three is a test that is red on the others.
    real_game_records = builder.game_records

    def explode(record, **kwargs):
        if record.headers.get("White") in ("alice", "erin"):
            raise RuntimeError("something nothing expected")
        return real_game_records(record, **kwargs)

    path = Path(fixture("lichess.pgn"))
    job = builder._Job(
        piece=ByteRange(source=0, path=path, start=0, end=path.stat().st_size),
        rating_source=None,
        validation_fraction=0.0,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "game_records", explode)
        read = builder._read_piece(job)

    assert read.kept == 2, "the games it could still read"
    assert read.tally.skipped[SkipReason.UNREADABLE] == 2
    assert read.tally.unexpected is not None
    assert "something nothing expected" in read.tally.unexpected
    assert "Traceback" in read.tally.unexpected, "so a broken build is recognisable as one"


def test_a_failure_nothing_expected_is_said_once_for_the_whole_build(tmp_path, caplog):
    # The build half: however many pieces carried one, and whichever processes read them, the
    # count is the build's and so is the warning that explains it.
    real_in_order = builder._in_order

    def carrying_failures(pool, jobs, *, in_flight, **rest):
        for read in real_in_order(pool, jobs, in_flight=in_flight, **rest):
            read.tally.note(RuntimeError("something nothing expected"))
            read.tally.skipped[SkipReason.UNREADABLE] += 1
            yield read

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 120)
        patch.setattr(builder, "_in_order", carrying_failures)
        manifest = build(tmp_path, workers=3)

    assert manifest.skipped[SkipReason.UNREADABLE.value] > 1, "several pieces carried one"
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "said once for the build, not once per piece or per process"
    assert "something nothing expected" in caplog.text


def test_a_worker_that_dies_says_what_to_do_about_it(tmp_path):
    # A worker killed for its memory is the one failure of the parallel path a person can act on,
    # and a raw BrokenProcessPool traceback does not tell them how.
    def died(*args, **kwargs):
        raise BrokenProcessPool("a process in the process pool was terminated abruptly")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "_in_order", died)
        with pytest.raises(DatasetError, match="fewer --workers"):
            build(tmp_path, workers=2)

    assert list_datasets(tmp_path) == [], "and nothing is left behind"


def test_progress_is_reported_while_workers_read(tmp_path):
    reports: list[Progress] = []

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 120)
        manifest = build(tmp_path, progress=reports.append, workers=3)

    assert len(reports) > 1, "a piece at a time, not only at the end"
    final = reports[-1]
    assert final.done
    assert final.games_kept == manifest.games
    assert final.games_read == manifest.games + manifest.games_skipped
    assert final.bytes_read == final.bytes_total
    # Monotonic, because it is what a time remaining is worked out from: a piece arriving out of
    # turn would make the estimate walk backwards.
    read = [report.bytes_read for report in reports]
    assert read == sorted(read)


def test_cutting_a_file_with_no_boundaries_reads_it_once(tmp_path, monkeypatch):
    # A file whose games are not separated by a blank line has no boundary to cut at, so every
    # offset's scan runs to the end of it. Scanned from scratch each time that is quadratic in
    # the file's size -- hours of silent reading on a dump, before a single game is parsed.
    path = flat_pgn(tmp_path / "flat.pgn", games=4000)
    source = resolve_sources([str(path)])[0]
    counter = CountingPath(path)
    counter.install(monkeypatch)

    pieces = game_ranges(source, 0, 4096)

    assert len(pieces) == 1, "nothing to cut at, so the file is one piece"
    assert pieces[0].bytes == source.bytes
    # One pass, plus a little overlap. Before this was a single forward pass it read about
    # (size / target) x size, which for this file is fifty times over.
    assert counter.bytes_read < 3 * source.bytes, (
        f"read {counter.bytes_read:,} bytes to cut a {source.bytes:,} byte file"
    )


def test_cutting_a_file_scales_with_its_size_not_its_square(tmp_path, monkeypatch):
    # The shape of the cost, which is what makes the difference between seconds and hours: twice
    # the file should read about twice as much, not four times as much.
    reads = {}
    for games in (2000, 4000):
        path = flat_pgn(tmp_path / f"flat{games}.pgn", games=games)
        source = resolve_sources([str(path)])[0]
        counter = CountingPath(path)
        with pytest.MonkeyPatch.context() as patch:
            counter.install(patch)
            game_ranges(source, 0, 4096)
        reads[games] = counter.bytes_read

    assert reads[4000] < 3 * reads[2000], (
        f"doubling the file multiplied the reading by {reads[4000] / reads[2000]:.1f}x"
    )


def test_reading_a_piece_does_not_pull_the_whole_file_into_memory(tmp_path, monkeypatch):
    # A piece is only as small as the boundaries found in the file. A file with none is one
    # piece spanning all of it, so a piece read in one go is a read of the whole file -- and on
    # a dump that is tens of gigabytes in a worker, which the OOM killer answers.
    path = flat_pgn(tmp_path / "flat.pgn", games=4000)
    size = path.stat().st_size
    piece = ByteRange(source=0, path=path, start=0, end=size)
    counter = CountingPath(path)
    counter.install(monkeypatch)

    games = games_in_range(piece)
    first, _ = next(games)

    assert first.headers["Event"] == "x"
    assert counter.bytes_read < size // 4, (
        f"read {counter.bytes_read:,} of {size:,} bytes to hand back one game"
    )


def test_a_piece_read_as_a_stream_gives_the_same_games(tmp_path):
    # Whatever it reads at a time, a piece is still exactly the games inside it.
    path = flat_pgn(tmp_path / "flat.pgn", games=50)
    size = path.stat().st_size
    whole = [g for g, _ in games_in_range(ByteRange(source=0, path=path, start=0, end=size))]

    assert len(whole) == 50
    assert all(game.headers["Result"] == "1-0" for game in whole)


def test_the_default_worker_count_follows_the_cpus_this_process_may_use(monkeypatch):
    # os.cpu_count() is the machine's, not this process's. A build pinned to two cores of a
    # large host must not default to one worker per core of the host.
    monkeypatch.setattr(os, "cpu_count", lambda: 96)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.delattr(os, "process_cpu_count", raising=False)

    assert builder.default_workers() == 2


def test_the_default_worker_count_is_capped(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 512)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(512)), raising=False)
    monkeypatch.delattr(os, "process_cpu_count", raising=False)

    assert builder.default_workers() == builder.MAX_DEFAULT_WORKERS
    assert builder.most_workers() >= builder.default_workers(), "the ceiling is not below it"


def test_a_worker_count_that_reads_as_a_typo_is_refused(tmp_path):
    # --workers 1000 for 100: 1000 processes, each holding pieces of a file, is not a plan.
    with pytest.raises(DatasetError, match="more than this machine has any use for"):
        build(tmp_path, workers=builder.most_workers() + 1)

    assert list_datasets(tmp_path) == [], "and nothing was built on the way to finding out"


def test_a_worker_count_up_to_the_ceiling_is_allowed(tmp_path):
    # Oversubscribing on purpose still works; only the absurd is refused.
    manifest = build(tmp_path, workers=builder.most_workers())

    assert manifest.games == GOOD_GAMES


def test_a_piece_too_large_for_a_worker_is_not_given_to_one():
    # Streaming the read bounded the bytes, not the records: _read_piece keeps every record of
    # its piece until it has them all, so a piece that is a whole file is a whole file's records
    # in one process however carefully they were read.
    class Refusing:
        def submit(self, *args, **kwargs):
            raise AssertionError("a piece too large to hold must not be given to a worker")

    oversized = builder._Job(
        piece=ByteRange(
            source=0, path=Path("nothing.pgn"), start=0, end=builder.MAX_PIECE_BYTES + 1
        ),
        rating_source=None,
        validation_fraction=0.0,
    )

    assert list(builder._in_order(Refusing(), [oversized], in_flight=2)) == [oversized], (
        "it comes back to be read here instead"
    )


def test_a_file_with_no_boundaries_is_read_without_holding_it(tmp_path, caplog):
    # The file that motivates the whole cap: no blank line between games, so it cuts into one
    # piece spanning all of it. It has to build, and it has to say why it was slow.
    path = flat_pgn(tmp_path / "flat.pgn", games=400)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 2048)
        patch.setattr(builder, "MAX_PIECE_BYTES", 4096)
        manifest = build(tmp_path / "data", str(path), validation_fraction=0.0, workers=3)

    assert manifest.games == 400
    warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "said once for the file, not once per piece"
    assert "no game boundary" in warnings[0] and "flat.pgn" in warnings[0]


def test_a_piece_read_here_gives_the_same_dataset_as_a_worker_would(tmp_path):
    # The fallback is slower, not different: it has to produce the same shards as reading the
    # same file in one process does, which is the property the whole parallel path rests on.
    path = flat_pgn(tmp_path / "flat.pgn", games=400)
    alone = build(tmp_path / "one", str(path), workers=1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 2048)
        patch.setattr(builder, "MAX_PIECE_BYTES", 4096)
        fallen_back = build(tmp_path / "many", str(path), workers=3)

    assert shard_bytes(dataset_path(tmp_path / "many", "test")) == shard_bytes(
        dataset_path(tmp_path / "one", "test")
    )
    assert fallen_back.model_dump(exclude={"created"}) == alone.model_dump(exclude={"created"})


def test_progress_moves_while_a_piece_is_read_here(tmp_path):
    # The fallback is the slow path, so it is the one where somebody most needs to see that the
    # build is getting somewhere. A byte count that never moves makes `fraction` stick at 0 and
    # `seconds_remaining` either None or an estimate that climbs for as long as the read lasts.
    reports: list[Progress] = []
    path = flat_pgn(tmp_path / "flat.pgn", games=2000)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 2048)
        patch.setattr(builder, "MAX_PIECE_BYTES", 4096)
        manifest = build(
            tmp_path / "data",
            str(path),
            validation_fraction=0.0,
            workers=3,
            progress=reports.append,
        )

    assert manifest.games == 2000
    during = [report for report in reports if not report.done]
    assert during, "a read this long reports while it runs"
    assert any(report.bytes_read > 0 for report in during), "and the byte count moves"
    assert any(0 < report.fraction < 1 for report in during), "so a percentage means something"
    read = [report.bytes_read for report in during]
    assert read == sorted(read), "and it never goes backwards"
    assert during[-1].bytes_read <= manifest.sources[0].bytes, "nor past the end of the file"


def test_the_worker_ceiling_is_one_the_pool_will_honour_on_windows(monkeypatch):
    # ProcessPoolExecutor refuses more than 61 workers on Windows, and MIN_WORKER_CEILING is 64,
    # so the ceiling promised a number that platform would not start -- passing the build's own
    # check and then dying inside the pool with a bare ValueError, hours into a real read.
    monkeypatch.setattr(sys, "platform", "win32")

    assert builder.most_workers() <= builder.WINDOWS_MAX_WORKERS
    assert builder.WINDOWS_MAX_WORKERS == 61, "what concurrent.futures says it is"


def test_the_worker_ceiling_is_not_capped_elsewhere(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    assert builder.most_workers() >= builder.MIN_WORKER_CEILING


def test_progress_keeps_moving_after_a_file_gives_up(tmp_path):
    # A file that stops part way still has its remaining pieces read -- they were already in
    # flight -- and their games dropped. Those bytes were read either way, so progress has to
    # account for them: otherwise the fraction sticks and the estimate climbs for the rest of
    # the file, which is exactly what _read_here was fixed for.
    reports: list[Progress] = []
    real_in_order = builder._in_order

    def stopping(pool, jobs, *, in_flight, **rest):
        for seen, outcome in enumerate(real_in_order(pool, jobs, in_flight=in_flight, **rest)):
            if seen == 2 and not isinstance(outcome, builder._Job):
                outcome.error = "stopped reading after 2 games: pretend the file gave up"
            yield outcome

    path = separated_pgn(tmp_path / "many.pgn", games=200)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 1000)
        patch.setattr(builder, "_in_order", stopping)
        manifest = build(
            tmp_path / "data",
            str(path),
            validation_fraction=0.0,
            workers=2,
            progress=reports.append,
        )

    assert manifest.sources[0].error is not None, "the file gave up part way"
    assert 0 < manifest.games < 200, "keeping what it read before that and no more"
    during = [report for report in reports if not report.done]
    read = [report.bytes_read for report in during]
    assert read == sorted(read), "progress never goes backwards"
    assert during[-1].bytes_read > 0.8 * manifest.sources[0].bytes, (
        f"progress stalled at {during[-1].bytes_read:,} of {manifest.sources[0].bytes:,} bytes"
    )


@pytest.mark.parametrize(
    "blank",
    ["", " ", "\t", "   ", " " * 20, "\t \t " * 6],
    ids=["empty", "space", "tab", "spaces", "wide", "wider"],
)
def test_a_separator_line_may_hold_whitespace(tmp_path, blank):
    # Exporters that leave a space or a tab on the line between games are not rare, and the line
    # is still blank. Refusing them costs a file every boundary it has -- which is not a wrong
    # dataset, just a build that silently reads a dump on one core for hours.
    game = (
        '[Event "x"]\r\n[Site "s"]\r\n[Result "1-0"]\r\n[WhiteElo "1500"]\r\n'
        '[BlackElo "1500"]\r\n[TimeControl "600+0"]\r\n\r\n1. e4 e5 1-0\r\n' + blank + "\r\n"
    )
    path = tmp_path / "spaced.pgn"
    path.write_bytes((game * 200).encode())
    source = resolve_sources([str(path)])[0]

    pieces = game_ranges(source, 0, 512)

    assert len(pieces) > 20, f"{len(pieces)} pieces: the separator was not recognised"
    assert sum(piece.bytes for piece in pieces) == source.bytes
    games = [game for piece in pieces for game, _ in games_in_range(piece)]
    assert len(games) == 200
    assert all(len(list(game.mainline_moves())) == 2 for game in games), "and none was cut open"


def test_whitespace_does_not_make_a_comment_line_a_boundary(tmp_path):
    # The reason the blank line is required at all: allowing whitespace must not let a line of
    # text inside a {} comment pass for the start of a game.
    lines = [
        '[Event "First"]',
        '[Result "1-0"]',
        "",
        "1. e4 { a comment that goes on   ",
        '[Event "Not really a game"]',
        "and on } e5 2. d4 1-0",
        " ",
        '[Event "Second"]',
        '[Result "0-1"]',
        "",
        "1. d4 d5 0-1",
        "",
    ]
    path = tmp_path / "commented.pgn"
    path.write_bytes("\r\n".join(lines).encode())
    source = resolve_sources([str(path)])[0]

    for target in (1, 8, 32, 200):
        events = [
            game.headers["Event"]
            for piece in game_ranges(source, 0, target)
            for game, _ in games_in_range(piece)
        ]
        assert events == ["First", "Second"], f"two whole games, at {target}"


def test_no_docstring_about_line_endings_is_mangled_by_its_own_escapes():
    # A docstring explaining \r\n has to survive being printed. Without an r-prefix the escapes
    # become the characters: the sentence breaks apart, and a bare \r sends a terminal's cursor
    # back over what it already wrote.
    source = Path(sources.__file__).read_text()
    mangled = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        written = ast.get_source_segment(source, node.value) or ""
        escaped = "\\r" in written or "\\n" in written
        if escaped and not written.lstrip().startswith(("r'", 'r"')):
            mangled.append(written.splitlines()[0][:60])
    assert not mangled, f"{len(mangled)} docstring(s) need an r-prefix: {mangled}"


def test_a_worker_running_out_of_memory_says_which_file_and_what_to_do():
    # A worker holds every record of its piece and then joins them, so MemoryError is a real way
    # for it to end -- and it arrives as itself rather than as BrokenProcessPool, so it missed
    # the handler that exists for the same failure one step further along, where the kernel does
    # the killing instead. A raw traceback out of concurrent.futures says neither which file it
    # was reading nor that --workers is the knob.
    class OutOfMemory:
        def submit(self, function, job):
            failed = Future()
            failed.set_exception(MemoryError("cannot allocate array"))
            return failed

    job = builder._Job(
        piece=ByteRange(source=0, path=Path("enormous.pgn"), start=0, end=1024),
        rating_source=None,
        validation_fraction=0.0,
    )

    with pytest.raises(DatasetError, match="enormous.pgn") as failure:
        list(builder._in_order(OutOfMemory(), [job], in_flight=2))

    assert "--workers" in str(failure.value)
    assert isinstance(failure.value.__cause__, MemoryError)


def test_a_build_that_has_read_no_games_yet_says_so_and_the_line_moves():
    # Before the first game comes back -- a build cutting its files into pieces, which for a
    # file with no boundaries in it is a scan of the whole thing -- every count is zero and
    # stays zero. A line made of them is one the printer rewrites with itself, which on a
    # terminal cannot be told from a hang.
    lines = [
        format_progress(
            Progress(
                games_read=0,
                games_kept=0,
                positions=0,
                bytes_read=0,
                bytes_total=10**10,
                seconds=seconds,
                done=False,
                scanning=True,
            )
        )
        for seconds in (0.0, 30.0, 90.0)
    ]

    assert len(set(lines)) == 3, f"the line has to move: {lines}"
    assert all("games/s" not in line for line in lines), "and not claim a rate it has not got"


def test_a_piece_that_gives_up_reports_the_files_count_not_its_own(tmp_path):
    # `error` reaches the manifest, and the serial path fills it with the file's running total.
    # A worker only knows its own piece, so composing the sentence there would say "stopped
    # after 7 games" of a file that had read six million.
    real_in_order = builder._in_order

    def stopping(pool, jobs, *, in_flight, **rest):
        for seen, outcome in enumerate(real_in_order(pool, jobs, in_flight=in_flight, **rest)):
            if seen == 5 and not isinstance(outcome, builder._Job):
                outcome.error = "ValueError: pretend the chess stopped making sense"
            yield outcome

    path = separated_pgn(tmp_path / "many.pgn", games=200)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 1000)
        patch.setattr(builder, "_in_order", stopping)
        manifest = build(tmp_path / "data", str(path), validation_fraction=0.0, workers=2)

    source = manifest.sources[0]
    assert source.error is not None
    assert "pretend the chess stopped making sense" in source.error, "the reason survives"
    assert source.games_read > 20, "several pieces had been read by then"
    assert f"after {source.games_read} games" in source.error, (
        f"{source.error!r} should count the file's games, not one piece's"
    )


def test_a_source_that_has_yielded_no_games_still_reports_how_far_it_has_got():
    # "Nothing read yet" is not the same state as "still cutting" once bytes have been read. A
    # source that yields no games at all -- a renamed archive, or a file whose pieces were all
    # dropped after an early one gave up -- has a real percentage and a real estimate, and
    # saying it is still looking for the games is both false and a different-looking hang.
    line = format_progress(
        Progress(
            games_read=0,
            games_kept=0,
            positions=0,
            bytes_read=3_000_000_000,
            bytes_total=10_000_000_000,
            seconds=60.0,
            done=False,
        )
    )

    assert "finding where the games start" not in line
    assert "30%" in line, line


def test_progress_is_reported_while_a_file_is_being_cut(tmp_path):
    # The line moving is format_progress's job; getting it onto the terminal is this one's.
    # game_ranges is a single blocking call per source, so without a report from inside the
    # scan a boundary-free file shows one line at t=0 and nothing until the cut is over.
    reports: list[Progress] = []
    path = flat_pgn(tmp_path / "flat.pgn", games=8000)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 4096)
        patch.setattr(builder, "MAX_PIECE_BYTES", 8192)
        patch.setattr(builder, "SCANS_PER_REPORT", 1)
        patch.setattr(sources, "ALIGN_SCAN", 4096)
        manifest = build(
            tmp_path / "data",
            str(path),
            validation_fraction=0.0,
            workers=2,
            progress=reports.append,
        )

    assert manifest.games == 8000
    cutting = [r for r in reports if not r.games_read and not r.bytes_read]
    assert len(cutting) > 10, f"only {len(cutting)} report(s) while the file was being cut"


def test_no_piece_ends_past_the_size_the_build_recorded(tmp_path):
    # align_to_game reads a whole window from below `size`, so it can accept a boundary above it
    # and hand back a cut past the end the build is measuring against. That happens for a file
    # that grew between resolve_sources' stat() and the cut -- a download or an export still
    # running -- and it makes the last piece reach past bytes_total, so the fraction pins at 100%
    # and the estimate goes negative, then backwards at the next source.
    game = '[Event "x"]\r\n[Result "1-0"]\r\n\r\n1. e4 e5 1-0\r\n\r\n'
    path = tmp_path / "growing.pgn"
    path.write_bytes((game * 100).encode())
    whole = path.stat().st_size

    for recorded in range(whole // 2, whole // 2 + len(game) + 1):
        pieces = game_ranges(Source(path=path, bytes=recorded), 0, 20)
        assert pieces, recorded
        assert pieces[-1].end <= recorded, (
            f"recorded {recorded}, last piece {(pieces[-1].start, pieces[-1].end)}"
        )
        assert sum(piece.bytes for piece in pieces) == recorded, recorded


def test_a_source_that_gives_up_stops_being_handed_to_workers():
    # The `if not state.stopped` guard skips the writing, not the submitting: _in_order pulls the
    # next job every iteration whatever the consumer does. So a dump that gives up at piece 12 of
    # 1,200 had the other 1,188 parsed in full and thrown away -- the whole parallel read of the
    # file, for nothing.
    empty = builder._Read(
        piece=ByteRange(source=0, path=Path("x.pgn"), start=0, end=1),
        tally=builder._Tally(),
        splits={},
    )
    submitted: list[builder._Job] = []

    class Counting:
        def submit(self, function, job):
            submitted.append(job)
            done = Future()
            done.set_result(empty)
            return done

    jobs = [
        builder._Job(
            piece=ByteRange(source=0, path=Path("x.pgn"), start=i * 100, end=(i + 1) * 100),
            rating_source=None,
            validation_fraction=0.0,
        )
        for i in range(40)
    ]
    stopped: set[int] = set()
    taken = 0
    for _ in builder._in_order(
        Counting(), jobs, in_flight=2, wanted=lambda job: job.piece.source not in stopped
    ):
        taken += 1
        stopped.add(0)

    assert taken < 10, f"took {taken} pieces after the source had given up"
    assert len(submitted) < 10, f"submitted {len(submitted)} of 40 pieces of a stopped file"


def test_a_serial_build_says_something_before_its_first_game(tmp_path):
    # Three rounds went into the parallel path's progress and this branch got none of it: no
    # report before the loop, none after a source, and _read_games only reports every
    # REPORT_EVERY games. A file that yields no games therefore passes in silence, and
    # --workers 1 is what the README now recommends for annotated PGN.
    reports: list[Progress] = []
    junk = tmp_path / "junk.pgn"
    junk.write_text("not pgn at all\n" * 100_000)

    manifest = build(
        tmp_path / "data",
        str(junk),
        fixture("lichess.pgn"),
        validation_fraction=0.0,
        workers=1,
        progress=reports.append,
    )

    assert manifest.games == 4, "the readable source is still read"
    assert reports[0].bytes_read == 0, "the first report comes before anything has been read"
    during = [r for r in reports if not r.done]
    assert len(during) >= 2, f"only {len(during)} report(s) before the summary"
    assert any(r.bytes_read >= junk.stat().st_size for r in during), (
        "and one of them accounts for the file that gave nothing"
    )


def test_scanning_for_boundaries_does_not_report_per_window(tmp_path):
    # on_scan fires per ALIGN_SCAN window, which for a boundary-free Lichess month is ~127,000
    # times in a phase where nothing changes but the clock. The printer throttles, but the
    # callback also feeds a test collector and, in time, a socket -- neither of which does.
    reports: list[Progress] = []
    path = flat_pgn(tmp_path / "flat.pgn", games=40000)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 4096)
        patch.setattr(builder, "MAX_PIECE_BYTES", 8192)
        patch.setattr(sources, "ALIGN_SCAN", 1024)
        build(
            tmp_path / "data",
            str(path),
            validation_fraction=0.0,
            workers=2,
            progress=reports.append,
        )

    windows = path.stat().st_size // 1024
    cutting = [r for r in reports if not r.games_read and not r.bytes_read]
    assert cutting, "it still reports while cutting"
    assert len(cutting) < windows // 10, (
        f"{len(cutting)} reports for {windows} windows scanned: no coarser than per-window"
    )


def test_a_stale_total_does_not_make_a_negative_estimate():
    # bytes_total is the size resolve_sources recorded; the serial reader follows the file to
    # whatever its real end is. A file that grew in between makes read exceed total, and
    # `seconds * (total - read) / read` is then negative -- printed as "-9000s left", since
    # format_duration does not guard it either.
    over = Progress(
        games_read=20,
        games_kept=20,
        positions=100,
        bytes_read=2840,
        bytes_total=284,
        seconds=1.0,
        done=False,
    )

    assert over.seconds_remaining is None, "there is no estimate to give once the total is past"
    assert "left" not in format_progress(over)


def test_only_a_build_that_is_scanning_says_it_is_scanning():
    # The serial path has no game_ranges, no align_to_game and no boundary scan: it opens a file
    # and reads games. Inferring the phase from two zeros told it to say otherwise.
    zeros = dict(games_read=0, games_kept=0, positions=0, bytes_read=0, bytes_total=10**10)

    assert "finding where the games start" in format_progress(
        Progress(**zeros, seconds=3.0, done=False, scanning=True)
    )
    reading = format_progress(Progress(**zeros, seconds=3.0, done=False))
    assert "finding where the games start" not in reading
    assert "0%" in reading and "0 games" in reading, reading


def test_a_file_that_yields_no_games_still_reports_while_it_is_read(tmp_path):
    # read_game consumes a whole file looking for a header before it returns None, so a file with
    # no games in it reports nothing for as long as the read takes -- ~73 minutes for 20 GB. Both
    # readers are gated on games, so neither path escapes it.
    prose = tmp_path / "prose.pgn"
    prose.write_text("not pgn at all, just prose that goes on and on\n" * 200_000)

    for workers in (1, 4):
        reports: list[Progress] = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sources, "READ_REPORT_BYTES", 64 << 10)
            patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
            # Above the cap, so the parallel path reads it here rather than in a worker. A
            # worker has no channel to report through, so a piece it reads is silent for as
            # long as it takes -- bounded by MAX_PIECE_BYTES, which is the point of the cap.
            patch.setattr(builder, "MAX_PIECE_BYTES", 1 << 20)
            build(
                tmp_path / f"data{workers}",
                str(prose),
                fixture("lichess.pgn"),
                validation_fraction=0.0,
                workers=workers,
                progress=reports.append,
            )
        read = [r.bytes_read for r in reports if not r.done and r.bytes_read]
        assert len(read) > 10, f"workers={workers}: only {len(read)} report(s) during the read"
        assert read == sorted(read), f"workers={workers}: and it never goes backwards"


def test_closing_a_reader_closes_the_file_it_opened(tmp_path):
    # The reader opens the file and then reads it through a window, and closing the text on top
    # closes the window -- which is not the file. The handle underneath was left to be collected,
    # which is a ResourceWarning and, on an interpreter without refcounting, a descriptor held
    # for as long as it likes.
    path = separated_pgn(tmp_path / "games.pgn", games=3)
    reader = sources.PgnReader(Source(path=path, bytes=path.stat().st_size))
    reader.open()
    handle = reader._handle
    assert next(reader.games()) is not None

    reader.close()

    assert handle is not None and handle.closed
    reader.close()  # and closing twice is closing once


def test_every_report_while_the_files_are_being_cut_says_so(tmp_path):
    # The report between one file's cut and the next's carries the same zeros as the ones from
    # inside the scan, so without the flag it prints "0%, 0 games" in the middle of a phase that
    # is otherwise saying it is looking for boundaries.
    reports: list[Progress] = []
    for name in ("a.pgn", "b.pgn", "c.pgn"):
        separated_pgn(tmp_path / name, games=50)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
        patch.setattr(builder, "CHUNK_BYTES", 2048)
        build(
            tmp_path / "data",
            str(tmp_path / "*.pgn"),
            validation_fraction=0.0,
            workers=2,
            progress=reports.append,
        )

    cutting = [r for r in reports if not r.bytes_read and not r.done]
    assert len(cutting) >= 4, f"only {len(cutting)} report(s) while cutting three files"
    assert all(r.scanning for r in cutting), [r.scanning for r in cutting]
    assert not any(r.scanning for r in reports if r.bytes_read), "and none once reading starts"


def test_a_file_that_grew_does_not_count_into_the_next_files_share(tmp_path):
    # The serial reader follows a file to its real end, and the size the build measures against
    # is the one recorded when it started. Clamped to the build's total rather than to the
    # file's own size, a file that grew reports its extra bytes as the following sources' --
    # and then steps back to where it should have been when it finishes.
    grown = separated_pgn(tmp_path / "a.pgn", games=2000)
    separated_pgn(tmp_path / "b.pgn", games=2000)
    recorded = grown.stat().st_size // 2
    real = sources.resolve_sources

    def stale(patterns):
        return [
            Source(path=s.path, bytes=recorded) if s.path == grown else s for s in real(patterns)
        ]

    reports: list[Progress] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "resolve_sources", stale)
        patch.setattr(sources, "READ_REPORT_BYTES", 8 << 10)
        manifest = build(
            tmp_path / "data",
            str(tmp_path / "*.pgn"),
            validation_fraction=0.0,
            workers=1,
            progress=reports.append,
        )

    assert manifest.games == 4000, "the serial path still reads to the real end"
    read = [r.bytes_read for r in reports if not r.done]
    assert read == sorted(read), "it never goes backwards"
    # Short of its last game, because the reports after that one are of the next file's bytes.
    first = [r.bytes_read for r in reports if r.games_read < 2000 and not r.done]
    assert max(first) <= recorded, f"{max(first)} reported of a file recorded as {recorded}"
