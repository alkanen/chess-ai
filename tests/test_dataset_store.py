"""Sharded storage: that the records come back out the way they went in, in any order."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from dataset_helpers import TINY_SHARDS, build

from chess_ai.dataset import (
    FORMAT_VERSION,
    GAME_DTYPE,
    POSITION_DTYPE,
    SPLITS,
    TRAIN,
    VALIDATION,
    DatasetError,
    ManifestError,
    Shards,
    list_datasets,
    open_dataset,
)
from chess_ai.dataset.store import (
    GAMES,
    MOVES,
    POSITIONS,
    dataset_path,
    shard_path,
)


def built(tmp_path, **options):
    """A dataset built from every fixture, and the reader for it."""
    build(tmp_path, validation_fraction=0.5, **options)
    return open_dataset("test", data_dir=tmp_path)


def test_records_are_written_as_whole_records_in_numbered_shards(tmp_path):
    built(tmp_path, shards=TINY_SHARDS)

    for split in SPLITS:
        directory = dataset_path(tmp_path, "test") / split
        for stream, dtype, per_shard in (
            (POSITIONS, POSITION_DTYPE, TINY_SHARDS.positions_per_shard),
            (GAMES, GAME_DTYPE, TINY_SHARDS.games_per_shard),
            (MOVES, np.dtype("<u2"), TINY_SHARDS.moves_per_shard),
        ):
            shards = sorted((directory / stream).glob("*.bin"))
            assert shards == [
                shard_path(directory / stream, number) for number in range(len(shards))
            ], "shards are numbered from zero with no gaps"
            for shard in shards[:-1]:
                assert shard.stat().st_size == per_shard * dtype.itemsize, (
                    "every shard but the last is full"
                )
            assert shards[-1].stat().st_size % dtype.itemsize == 0, "no half records"


def test_the_fixtures_fill_more_than_one_shard(tmp_path):
    # Otherwise the sharding is never exercised by anything below.
    built(tmp_path, shards=TINY_SHARDS)
    positions = dataset_path(tmp_path, "test") / TRAIN / POSITIONS

    assert len(list(positions.glob("*.bin"))) > 1


def test_a_position_reads_the_same_whatever_shard_it_is_in(tmp_path):
    spread = built(tmp_path / "spread", shards=TINY_SHARDS)[TRAIN]
    together = built(
        tmp_path / "together",
        shards=Shards(positions_per_shard=10_000, games_per_shard=10_000, moves_per_shard=10_000),
    )[TRAIN]

    assert len(spread) == len(together)
    for index in range(len(spread)):
        assert spread.position(index) == together.position(index)
        assert spread.board(index).fen() == together.board(index).fen()


def test_a_batch_comes_back_in_the_order_it_was_asked_for(tmp_path):
    split = built(tmp_path, shards=TINY_SHARDS)[TRAIN]
    wanted = [len(split) - 1, 0, 3, 3, len(split) // 2]

    batch = split.positions(wanted)

    assert batch.dtype == POSITION_DTYPE
    assert len(batch) == len(wanted)
    for slot, index in enumerate(wanted):
        assert batch[slot] == split.position(index)


def test_a_random_batch_matches_reading_one_at_a_time(tmp_path):
    split = built(tmp_path, shards=TINY_SHARDS)[TRAIN]
    wanted = np.random.default_rng(7).integers(0, len(split), size=200)

    batch = split.positions(wanted)

    for slot, index in enumerate(wanted):
        assert batch[slot] == split.position(int(index))


def test_an_empty_batch_is_an_empty_batch(tmp_path):
    split = built(tmp_path, shards=TINY_SHARDS)[TRAIN]

    assert len(split.positions([])) == 0


def test_the_last_record_can_be_asked_for_from_the_end(tmp_path):
    split = built(tmp_path, shards=TINY_SHARDS)[TRAIN]

    assert split.position(-1) == split.position(len(split) - 1)
    assert split.positions([-1])[0] == split.position(len(split) - 1)
    assert split.game(-1) == split.game(split.games - 1)


def test_reading_past_the_end_says_so(tmp_path):
    split = built(tmp_path, shards=TINY_SHARDS)[TRAIN]

    with pytest.raises(IndexError):
        split.position(len(split))
    with pytest.raises(IndexError):
        split.positions([0, len(split)])
    with pytest.raises(IndexError):
        split.game(split.games)
    with pytest.raises(IndexError):
        split.move_sequence(split.games)


def test_a_games_moves_and_positions_line_up_across_shards(tmp_path):
    split = built(tmp_path, shards=TINY_SHARDS)[TRAIN]

    for index in range(split.games):
        game = split.game(index)
        moves = split.move_sequence(index)
        positions = split.game_positions(index)

        assert len(moves) == len(positions) == int(game["ply_count"])
        np.testing.assert_array_equal(moves, positions["move"])


def test_the_splits_are_separate_directories(tmp_path):
    dataset = built(tmp_path)
    root = dataset_path(tmp_path, "test")

    assert sorted(child.name for child in root.iterdir() if child.is_dir()) == sorted(SPLITS)
    assert dataset.splits == list(SPLITS)
    assert len(dataset[TRAIN]) + len(dataset[VALIDATION]) == dataset.manifest.positions


def test_a_split_that_is_not_there_says_which_are(tmp_path):
    dataset = built(tmp_path)

    with pytest.raises(DatasetError, match="no 'test' split"):
        dataset["test"]


def test_a_dataset_that_is_not_there_says_so(tmp_path):
    with pytest.raises(ManifestError, match="manifest.json is missing"):
        open_dataset("nothing", data_dir=tmp_path)


def test_a_dataset_of_another_format_version_is_refused(tmp_path):
    built(tmp_path)
    manifest = dataset_path(tmp_path, "test") / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"format_version": 1', '"format_version": 99'))

    with pytest.raises(ManifestError, match="format version 99"):
        open_dataset("test", data_dir=tmp_path)


def test_a_manifest_that_is_not_json_is_refused(tmp_path):
    built(tmp_path)
    (dataset_path(tmp_path, "test") / "manifest.json").write_text("{not json")

    with pytest.raises(ManifestError, match="invalid JSON"):
        open_dataset("test", data_dir=tmp_path)


def test_a_missing_shard_is_reported_rather_than_read_as_nothing(tmp_path):
    built(tmp_path, shards=TINY_SHARDS)
    shard_path(dataset_path(tmp_path, "test") / TRAIN / POSITIONS, 0).unlink()

    with pytest.raises(DatasetError, match="cannot read dataset shard"):
        open_dataset("test", data_dir=tmp_path)[TRAIN].position(0)


def test_the_datasets_in_a_data_directory_are_listed(tmp_path):
    assert list_datasets(tmp_path) == []

    build(tmp_path, "lichess.pgn")
    (tmp_path / "datasets" / "not-a-dataset").mkdir()
    (tmp_path / "datasets" / "stray.txt").write_text("hello")

    assert list_datasets(tmp_path) == ["test"]


def test_a_truncated_shard_is_reported_as_a_dataset_problem(tmp_path):
    # What an interrupted build, a truncated copy or a full disk leaves behind. numpy raises
    # ValueError rather than OSError for a file that is not a whole number of records, which
    # used to escape as a traceback.
    built(tmp_path)
    shard = shard_path(dataset_path(tmp_path, "test") / TRAIN / POSITIONS, 0)
    shard.write_bytes(shard.read_bytes()[:-7])

    with pytest.raises(DatasetError, match="cannot read dataset shard"):
        open_dataset("test", data_dir=tmp_path)[TRAIN].position(0)


def test_a_manifest_from_a_format_with_new_fields_says_to_rebuild(tmp_path):
    # A newer format is exactly the one carrying fields this code has never heard of, so the
    # version has to be read before the manifest is validated against this version's shape.
    built(tmp_path)
    path = dataset_path(tmp_path, "test") / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format_version"] = FORMAT_VERSION + 1
    manifest["something_added_later"] = {"a": 1}
    path.write_text(json.dumps(manifest))

    with pytest.raises(ManifestError, match=f"format version {FORMAT_VERSION + 1}"):
        open_dataset("test", data_dir=tmp_path)


def test_a_manifest_of_this_format_with_unknown_fields_is_still_refused(tmp_path):
    # Not a version thing: a manifest claiming this format has to match this format.
    built(tmp_path)
    path = dataset_path(tmp_path, "test") / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["something_unexpected"] = 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ManifestError, match="invalid manifest"):
        open_dataset("test", data_dir=tmp_path)


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc to count open files")
def test_a_dataset_gives_back_the_shards_it_mapped(tmp_path):
    # Every mapped shard holds a file descriptor, so a process that opens dataset after
    # dataset — a sweep over configs, or the web server — must be able to hand them back.
    built(tmp_path, shards=TINY_SHARDS)
    before = _open_files()

    with open_dataset("test", data_dir=tmp_path) as dataset:
        split = dataset[TRAIN]
        for index in range(len(split)):
            split.position(index)
        for index in range(split.games):
            split.move_sequence(index)
        while_open = _open_files()

    assert while_open > before, "the shards were mapped at all"
    assert _open_files() == before


def _open_files() -> int:
    return len(os.listdir("/proc/self/fd"))
