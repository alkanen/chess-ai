"""Appending to a dataset: each append is a new version, and every version is a prefix."""

import io
import json
import shutil

import numpy as np
import pytest
from dataset_helpers import (
    TINY_SHARDS,
    build,
    comparable,
    fixture,
    move_sequences,
    separated_pgn,
    shard_bytes,
)

from chess_ai.dataset import (
    SPLITS,
    TRAIN,
    VALIDATION,
    DatasetError,
    ManifestError,
    ProgressPrinter,
    builder,
    load_manifest,
    open_dataset,
)
from chess_ai.dataset.builder import append_dataset
from chess_ai.dataset.manifest import Filters, Manifest
from chess_ai.dataset.store import dataset_lock, dataset_path, records_past

SPREAD = {"shards": TINY_SHARDS, "validation_fraction": 0.3}
"""Small shards and a real validation split, so an append carries on part-way through shards."""


def append(data_dir, *sources, **options) -> Manifest:
    """Append ``sources``, fixture names or paths, to the dataset called "test"."""
    named = [
        fixture(source) if not str(source).startswith("/") else str(source) for source in sources
    ]
    return append_dataset("test", named, data_dir=data_dir, **options)


def test_appending_gives_the_same_dataset_as_building_from_all_the_sources_at_once(tmp_path):
    build(tmp_path / "appended", "lichess.pgn", **SPREAD)
    appended = append(tmp_path / "appended", "unrated.pgn")
    appended = append(tmp_path / "appended", "custom-start.pgn", "malformed.pgn")
    together = build(
        tmp_path / "together",
        "lichess.pgn",
        "unrated.pgn",
        "custom-start.pgn",
        "malformed.pgn",
        **SPREAD,
    )

    assert shard_bytes(dataset_path(tmp_path / "appended", "test")) == shard_bytes(
        dataset_path(tmp_path / "together", "test")
    )
    assert appended.version == 3
    # The totals are the same; how they came to be is what the versions say.
    whole = comparable(appended)
    whole.pop("versions")
    alone = comparable(together)
    alone.pop("versions")
    assert whole == alone
    assert [version.games for version in appended.versions] == [4, 3, together.games - 7]


def test_appending_in_several_processes_gives_the_same_dataset(tmp_path, monkeypatch):
    for name in ("a.pgn", "b.pgn"):
        separated_pgn(tmp_path / name, games=40)
    (tmp_path / "b.pgn").write_text((tmp_path / "b.pgn").read_text().replace("1-0", "0-1"))
    build(tmp_path / "serial", str(tmp_path / "a.pgn"), workers=1, **SPREAD)
    append(tmp_path / "serial", tmp_path / "b.pgn", workers=1)
    monkeypatch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
    monkeypatch.setattr(builder, "CHUNK_BYTES", 600)
    build(tmp_path / "parallel", str(tmp_path / "a.pgn"), workers=3, **SPREAD)
    append(tmp_path / "parallel", tmp_path / "b.pgn", workers=3)

    assert shard_bytes(dataset_path(tmp_path / "parallel", "test")) == shard_bytes(
        dataset_path(tmp_path / "serial", "test")
    )


def test_a_filtered_dataset_appends_through_its_own_filters(tmp_path):
    filters = Filters(min_rating=1600, unknown_rating_passes=False)
    build(tmp_path / "appended", "lichess.pgn", filters=filters, **SPREAD)
    appended = append(tmp_path / "appended", "unrated.pgn", "custom-start.pgn")
    together = build(
        tmp_path / "together",
        "lichess.pgn",
        "unrated.pgn",
        "custom-start.pgn",
        filters=filters,
        **SPREAD,
    )

    assert appended.filters == filters
    assert appended.splits == together.splits
    assert appended.splits[TRAIN].targets is not None, "the target stream was appended to"
    assert shard_bytes(dataset_path(tmp_path / "appended", "test")) == shard_bytes(
        dataset_path(tmp_path / "together", "test")
    )


def test_a_game_names_its_own_source_whichever_version_it_came_in(tmp_path):
    build(tmp_path, "lichess.pgn", validation_fraction=0.0)
    manifest = append(tmp_path, "unrated.pgn")

    split = open_dataset("test", data_dir=tmp_path)[TRAIN]
    sources = [manifest.sources[int(game["source"])].path for game in split.game_records(0, 7)]

    assert sources == [fixture("lichess.pgn")] * 4 + [fixture("unrated.pgn")] * 3


def test_a_version_reads_only_its_own_records_after_an_append(tmp_path):
    build(tmp_path, "lichess.pgn", **SPREAD)
    before = open_dataset("test", data_dir=tmp_path)
    games = {split: move_sequences(before[split]) for split in SPLITS}
    counts = before.manifest.splits

    append(tmp_path, "unrated.pgn", "custom-start.pgn")
    first = open_dataset("test", data_dir=tmp_path, version=1)

    assert first.manifest.version == 1
    assert first.manifest.splits == counts
    for split in SPLITS:
        assert move_sequences(first[split]) == games[split]
        assert move_sequences(before[split]) == games[split], "a reader that was open carries on"
    latest = open_dataset("test", data_dir=tmp_path)
    assert latest.manifest.version == 2
    assert latest.manifest.games > first.manifest.games


def test_a_version_the_dataset_does_not_have_is_refused(tmp_path):
    build(tmp_path, "lichess.pgn")
    append(tmp_path, "unrated.pgn")

    with pytest.raises(ManifestError, match="no version 3; it has versions 1 to 2"):
        open_dataset("test", data_dir=tmp_path, version=3)
    with pytest.raises(ManifestError, match="no version 0"):
        open_dataset("test", data_dir=tmp_path, version=0)


def test_an_earlier_version_has_its_own_statistics(tmp_path):
    first = build(tmp_path, "lichess.pgn")
    latest = append(tmp_path, "unrated.pgn")

    assert latest.at(1) == first
    assert latest.statistics.ratings_unknown > first.statistics.ratings_unknown
    assert latest.sources[0] == first.sources[0]
    assert len(latest.sources) == 2


def test_the_manifest_lists_the_versions(tmp_path):
    build(tmp_path, "lichess.pgn")
    append(tmp_path, "unrated.pgn")

    written = json.loads((dataset_path(tmp_path, "test") / "manifest.json").read_text())

    assert [version["version"] for version in written["versions"]] == [1, 2]
    assert [[source["path"] for source in v["sources"]] for v in written["versions"]] == [
        [fixture("lichess.pgn")],
        [fixture("unrated.pgn")],
    ]
    assert [
        v["splits"][TRAIN]["games"] + v["splits"][VALIDATION]["games"] for v in written["versions"]
    ] == [4, 3]
    assert written["versions"][1]["statistics"]["ratings_unknown"] > 0
    assert len(written["versions"][0]["sources"][0]["sha256"]) == 64
    # And the totals at the top, for whoever reads the file without this code.
    assert written["splits"][TRAIN]["games"] + written["splits"][VALIDATION]["games"] == 7


def test_a_manifest_from_before_versions_reads_as_version_1(tmp_path):
    built = build(tmp_path, "lichess.pgn")
    path = dataset_path(tmp_path, "test") / "manifest.json"
    written = json.loads(path.read_text())
    del written["versions"]
    for source in written["sources"]:
        del source["sha256"], source["month"]
    path.write_text(json.dumps(written))

    old = load_manifest(dataset_path(tmp_path, "test"))

    assert old.version == 1
    assert old.versions[0].created == built.created
    assert old.versions[0].splits == built.splits
    assert old.versions[0].statistics == built.statistics
    assert old.at(1) == old
    assert append(tmp_path, "unrated.pgn").version == 2


# --- Sources already in the dataset -------------------------------------------------------------


def test_a_source_appended_twice_is_refused(tmp_path):
    build(tmp_path, "lichess.pgn")

    with pytest.raises(DatasetError, match="already in dataset 'test'.*in version 1"):
        append(tmp_path, "lichess.pgn")
    assert load_manifest(dataset_path(tmp_path, "test")).version == 1


def test_a_source_is_recognised_by_its_contents_whatever_it_is_called(tmp_path):
    build(tmp_path / "data", "lichess.pgn")
    copy = tmp_path / "renamed.pgn"
    shutil.copy(fixture("lichess.pgn"), copy)

    with pytest.raises(DatasetError, match="renamed.pgn has the same contents as"):
        append(tmp_path / "data", copy)


def test_a_repeat_can_be_asked_for(tmp_path):
    build(tmp_path, "lichess.pgn")

    manifest = append(tmp_path, "lichess.pgn", allow_repeat=True)

    assert manifest.games == 8


def test_one_append_naming_the_same_file_twice_is_refused(tmp_path):
    build(tmp_path / "data", "unrated.pgn")
    copy = tmp_path / "copy.pgn"
    shutil.copy(fixture("lichess.pgn"), copy)

    with pytest.raises(DatasetError, match="copy.pgn has the same contents as .* in this append"):
        append(tmp_path / "data", "lichess.pgn", copy)


def lichess_month(directory, month: str, *, compressed: bool, games: int = 30):
    """A file named as Lichess names the dump of ``month``, holding ``games`` games."""
    directory.mkdir(parents=True, exist_ok=True)
    plain = separated_pgn(directory / f"lichess_db_standard_rated_{month}.pgn", games=games)
    if not compressed:
        return plain
    import zstandard

    packed = plain.with_name(plain.name + ".zst")
    packed.write_bytes(zstandard.ZstdCompressor().compress(plain.read_bytes()))
    plain.unlink()
    return packed


def test_a_lichess_month_is_recognised_compressed_or_not(tmp_path):
    packed = lichess_month(tmp_path / "zst", "2026-01", compressed=True)
    plain = lichess_month(tmp_path / "pgn", "2026-01", compressed=False)
    build(tmp_path / "data", str(packed))

    with pytest.raises(DatasetError, match="the same Lichess month, 2026-01"):
        append(tmp_path / "data", plain)


def test_a_source_recorded_before_checksums_is_recognised_by_its_month(tmp_path):
    first = lichess_month(tmp_path / "a", "2026-01", compressed=False)
    build(tmp_path / "data", str(first))
    forget_checksums(tmp_path / "data")
    first.unlink()
    again = lichess_month(tmp_path / "b", "2026-01", compressed=True)

    with pytest.raises(DatasetError, match="the same Lichess month, 2026-01"):
        append(tmp_path / "data", again)


def test_a_source_recorded_before_checksums_is_hashed_where_it_was(tmp_path):
    original = tmp_path / "games.pgn"
    shutil.copy(fixture("lichess.pgn"), original)
    build(tmp_path / "data", str(original))
    forget_checksums(tmp_path / "data")
    copy = tmp_path / "other.pgn"
    shutil.copy(original, copy)

    with pytest.raises(DatasetError, match="other.pgn has the same contents as"):
        append(tmp_path / "data", copy)


def forget_checksums(data_dir) -> None:
    """Make the dataset "test" look as if it was built before sources had checksums."""
    path = dataset_path(data_dir, "test") / "manifest.json"
    written = json.loads(path.read_text())
    del written["versions"]
    for source in written["sources"]:
        del source["sha256"], source["month"]
    path.write_text(json.dumps(written))


# --- The maximum number of games ----------------------------------------------------------------


def test_the_maximum_caps_the_whole_dataset(tmp_path):
    build(tmp_path, "lichess.pgn", filters=Filters(max_games=3))

    with pytest.raises(DatasetError, match="at its maximum of 3 games; append with a larger"):
        append(tmp_path, "unrated.pgn")


def test_an_append_can_raise_the_maximum(tmp_path):
    build(tmp_path, "lichess.pgn", filters=Filters(max_games=3))

    manifest = append(tmp_path, "unrated.pgn", max_games=5)

    assert manifest.games == 5
    assert manifest.reached_max_games
    assert manifest.filters.max_games == 5
    assert [version.max_games for version in manifest.versions] == [3, 5]
    assert manifest.at(1).filters.max_games == 3


def test_an_append_cannot_lower_the_maximum_below_what_is_there(tmp_path):
    build(tmp_path, "lichess.pgn")

    with pytest.raises(DatasetError, match="already has 4 games"):
        append(tmp_path, "unrated.pgn", max_games=4)


def test_an_append_that_keeps_nothing_adds_no_version(tmp_path):
    build(tmp_path, "lichess.pgn", filters=Filters(time_controls=["rapid"]))
    empty = tmp_path / "bullet.pgn"
    empty.write_text('[Event "x"]\n[Result "1-0"]\n[TimeControl "60+0"]\n\n1. e4 e5 1-0\n\n')

    with pytest.raises(DatasetError, match="kept no games, so there is no version 2 to add"):
        append(tmp_path, empty)
    assert load_manifest(dataset_path(tmp_path, "test")).version == 1
    assert not records_past(
        dataset_path(tmp_path, "test"), load_manifest(dataset_path(tmp_path, "test"))
    )


# --- Interrupted appends ------------------------------------------------------------------------


def interrupted(data_dir, monkeypatch, *sources):
    """Append ``sources`` and kill the append just before its manifest is written."""

    def killed(self, directory):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(Manifest, "save", killed)
        with pytest.raises(KeyboardInterrupt):
            append(data_dir, *sources)


def test_an_interrupted_append_leaves_the_previous_version_readable(tmp_path, monkeypatch):
    build(tmp_path, "lichess.pgn", **SPREAD)
    games = move_sequences(open_dataset("test", data_dir=tmp_path)[TRAIN])

    interrupted(tmp_path, monkeypatch, "unrated.pgn")
    after = open_dataset("test", data_dir=tmp_path)

    assert after.manifest.version == 1
    assert move_sequences(after[TRAIN]) == games
    assert records_past(dataset_path(tmp_path, "test"), after.manifest)


def test_the_next_append_refuses_to_start_over_an_interrupted_one(tmp_path, monkeypatch):
    build(tmp_path, "lichess.pgn", **SPREAD)
    interrupted(tmp_path, monkeypatch, "unrated.pgn")
    left = shard_bytes(dataset_path(tmp_path, "test"))

    with pytest.raises(DatasetError, match="interrupted append.*--resume.*--discard-interrupted"):
        append(tmp_path, "custom-start.pgn")
    assert shard_bytes(dataset_path(tmp_path, "test")) == left, "and touches nothing"


def test_discarding_an_interrupted_append_gives_back_the_previous_version(tmp_path, monkeypatch):
    build(tmp_path / "clean", "lichess.pgn", **SPREAD)
    append(tmp_path / "clean", "custom-start.pgn")
    build(tmp_path / "data", "lichess.pgn", **SPREAD)
    interrupted(tmp_path / "data", monkeypatch, "unrated.pgn")

    manifest = append(tmp_path / "data", "custom-start.pgn", discard_interrupted=True)

    assert manifest.version == 2
    assert shard_bytes(dataset_path(tmp_path / "data", "test")) == shard_bytes(
        dataset_path(tmp_path / "clean", "test")
    )


def test_an_append_holds_the_dataset_lock(tmp_path):
    build(tmp_path, "lichess.pgn")

    with dataset_lock(tmp_path, "test"), pytest.raises(DatasetError, match="already running"):
        append(tmp_path, "unrated.pgn")


def test_appending_to_a_dataset_that_is_not_there_says_to_build_it(tmp_path):
    with pytest.raises(DatasetError, match="no dataset 'test'.*build it"):
        append(tmp_path, "lichess.pgn")


def test_a_build_refuses_the_same_file_twice_under_two_names(tmp_path):
    copy = tmp_path / "copy.pgn"
    shutil.copy(fixture("lichess.pgn"), copy)

    with pytest.raises(DatasetError, match="in this build"):
        build(tmp_path / "data", "lichess.pgn", str(copy))
    assert build(tmp_path / "data", "lichess.pgn", str(copy), allow_repeat=True).games == 8


def test_positions_read_the_same_in_a_version_as_in_the_dataset_it_grew_into(tmp_path):
    build(tmp_path, "lichess.pgn", **SPREAD)
    append(tmp_path, "unrated.pgn")
    first = open_dataset("test", data_dir=tmp_path, version=1)[TRAIN]
    latest = open_dataset("test", data_dir=tmp_path)[TRAIN]

    everything = np.arange(len(first))
    assert (first.positions(everything) == latest.positions(everything)).all()
    with pytest.raises(IndexError):
        first.positions([len(first)])


def test_an_empty_shard_an_interrupted_append_started_counts_as_left_behind(tmp_path):
    # Killed between opening the next shard and writing to it: nothing past the end but a name,
    # which the next append would otherwise trip over when it opens that shard itself.
    build(tmp_path, "lichess.pgn", validation_fraction=0.0, shards=TINY_SHARDS)
    directory = dataset_path(tmp_path, "test")
    manifest = load_manifest(directory)
    games = directory / TRAIN / "games"
    started = games / f"{manifest.splits[TRAIN].games // TINY_SHARDS.games_per_shard:05d}.bin"
    assert manifest.splits[TRAIN].games % TINY_SHARDS.games_per_shard == 0, "the last one is full"
    started.touch()

    assert records_past(directory, manifest) == [started]
    with pytest.raises(DatasetError, match="interrupted append"):
        append(tmp_path, "unrated.pgn")
    assert append(tmp_path, "unrated.pgn", discard_interrupted=True).games == 7


def test_a_source_the_dataset_never_read_from_can_be_appended(tmp_path):
    # A build that filled up before reaching a source recorded it, unread, with a checksum.
    # None of its games are in the dataset, so appending it is not a repeat.
    built = build(tmp_path, "lichess.pgn", "unrated.pgn", filters=Filters(max_games=3))
    assert [source.games_read for source in built.sources][1] == 0

    manifest = append(tmp_path, "unrated.pgn", max_games=10)

    assert manifest.games == 6


def test_a_source_the_dataset_read_part_of_is_still_a_repeat(tmp_path):
    build(tmp_path, "lichess.pgn", "unrated.pgn", filters=Filters(max_games=3))

    with pytest.raises(DatasetError, match="lichess.pgn has the same contents"):
        append(tmp_path, "lichess.pgn", max_games=10)


def slow_checksums(monkeypatch, seconds: float) -> list[float]:
    """A fake clock, and checksums that take ``seconds`` of it per source."""
    clock = [0.0]
    monkeypatch.setattr(builder.time, "monotonic", lambda: clock[0])
    real = builder.sha256_of

    def slow(path, on_read=None):
        clock[0] += seconds
        return real(path, on_read)

    monkeypatch.setattr(builder, "sha256_of", slow)
    return clock


def test_the_reading_is_timed_from_when_it_starts_not_from_the_checksums(tmp_path, monkeypatch):
    # The rate and the time left are worked out from the reading's own time, so the minutes
    # spent taking checksums must not count as minutes spent reading.
    slow_checksums(monkeypatch, 1000)
    reports = []
    build(tmp_path, "lichess.pgn", progress=reports.append)

    reading = [report for report in reports if not report.hashing and not report.done]
    assert reading and all(report.reading_seconds < 1000 for report in reading)
    assert reports[-1].done and reports[-1].seconds >= 1000, "the summary is the whole build's"


def test_the_build_clock_never_goes_backwards(tmp_path, monkeypatch):
    # A printer throttles on it, and a clock that started again after the checksums had every
    # reading report dropped for as long as the checksums had taken: a terminal frozen on
    # "checking the sources, 100%" while the build read.
    clock = slow_checksums(monkeypatch, 300)
    monkeypatch.setattr(builder, "HASH_REPORT_BYTES", 1)  # a report at the end of the checksums
    real_report = builder._Build.report

    def ticking(self, **kwargs):
        clock[0] += 10
        real_report(self, **kwargs)

    monkeypatch.setattr(builder._Build, "report", ticking)
    out = io.StringIO()
    reports = []

    def both(progress):
        reports.append(progress)
        printer(progress)

    printer = ProgressPrinter(out, interval=1.0, rewrite=False)
    build(tmp_path, "lichess.pgn", progress=both)

    seconds = [report.seconds for report in reports]
    assert seconds == sorted(seconds)
    printed = out.getvalue().splitlines()
    assert any(not line.startswith(("checking", "built")) for line in printed), printed
