"""Building datasets out of the fixture PGN files: what is kept, what is not, and the split."""

import errno
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import chess
import pytest
from dataset_helpers import FIXTURES, GOOD_GAMES, build, fixture, move_sequences

from chess_ai.dataset import (
    SPLITS,
    TRAIN,
    VALIDATION,
    DatasetError,
    GameFlags,
    PositionFlags,
    Progress,
    RatingSource,
    Result,
    SkipReason,
    TimeControl,
    builder,
    list_datasets,
    load_manifest,
    open_dataset,
    store,
    unpack_board,
)
from chess_ai.dataset.builder import DEFAULT_VALIDATION_FRACTION
from chess_ai.dataset.games import split_of, time_control_class
from chess_ai.dataset.records import POSITION_DTYPE
from chess_ai.dataset.store import dataset_path
from chess_ai.move_codec import VOCABULARY_SIZE, move_at


def test_a_build_keeps_the_games_it_can_read(tmp_path):
    manifest = build(tmp_path)

    assert manifest.games == GOOD_GAMES
    assert manifest.positions == sum(counts.positions for counts in manifest.splits.values())
    assert manifest.positions > 100


def test_broken_games_are_skipped_and_counted_by_reason(tmp_path):
    manifest = build(tmp_path, "malformed.pgn")

    assert manifest.games == 0
    assert manifest.positions == 0
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

    manifest = build(tmp_path / "data", str(junk))

    assert manifest.games == 0
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
    def explode(*args, **kwargs):
        raise RuntimeError("something nothing expected")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "game_records", explode)
        manifest = build(tmp_path, "lichess.pgn")

    assert manifest.games == 0
    assert manifest.skipped == {SkipReason.UNREADABLE.value: 4}
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "said once, not once per game"
    assert "something nothing expected" in caplog.text


def test_a_dataset_of_no_games_is_still_a_dataset(tmp_path):
    empty = tmp_path / "empty.pgn"
    empty.write_text("")

    manifest = build(tmp_path / "data", str(empty))

    assert manifest.games == 0
    dataset = open_dataset("test", data_dir=tmp_path / "data")
    assert len(dataset[TRAIN]) == 0
    assert dataset[TRAIN].games == 0


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


def test_a_dataset_an_interrupted_build_set_aside_is_reported(tmp_path, caplog):
    # A dataset the tool would otherwise never mention again: list_datasets and stats both skip
    # a dot-prefixed directory, so without this the user has lost it as far as they can tell.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    aside = store.replaced_path(store.new_partial_path(tmp_path, "test"))
    dataset_path(tmp_path, "test").rename(aside)

    build(tmp_path, "unrated.pgn", validation_fraction=0.0)

    assert str(aside) in caplog.text
    assert load_manifest(aside).games == 4, "and it is still the dataset it was"


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
