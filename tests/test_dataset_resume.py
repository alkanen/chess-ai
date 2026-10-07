"""Checkpoints and resuming: a build or append that stops part way is carried on, not redone."""

import io
import shutil
from pathlib import Path

import pytest
from dataset_helpers import BACK_TO_BACK, TINY_SHARDS, build, comparable, fixture, shard_bytes
from test_dataset_compressed import compressed, mixed_pgn, varied_pgn

from chess_ai.dataset import (
    DatasetError,
    ProgressPrinter,
    builder,
    load_manifest,
    open_dataset,
    store,
)
from chess_ai.dataset.builder import append_dataset, build_dataset
from chess_ai.dataset.checkpoint import CHECKPOINT_FILE, load_checkpoint
from chess_ai.dataset.manifest import Filters, Manifest
from chess_ai.dataset.store import dataset_path, records_past

SETTINGS = {
    "shards": TINY_SHARDS,
    "validation_fraction": 0.3,
    # A filter on positions, so that the target stream is written and cut back too.
    "filters": Filters(min_rating=1600, unknown_rating_passes=True),
}


class Killed(BaseException):
    """What kills a build in these tests: not an Exception, so nothing on the way catches it."""


def stream_appends(monkeypatch) -> list[int]:
    """Count the writes to every stream, which is where these tests kill a build."""
    count = [0]
    real = store._StreamWriter.append

    def counting(self, records):
        count[0] += 1
        return real(self, records)

    monkeypatch.setattr(store._StreamWriter, "append", counting)
    return count


def killing(monkeypatch, at: int) -> None:
    """Kill the build at its ``at``-th write to a stream, before it is made.

    A game is three or four writes, so this lands part way through a game's records as often as
    between games, which is what cutting back to a checkpoint has to undo.
    """
    calls = [0]
    real = store._StreamWriter.append

    def append(self, records):
        calls[0] += 1
        if calls[0] == at:
            raise Killed
        return real(self, records)

    monkeypatch.setattr(store._StreamWriter, "append", append)


def checkpoint_every(monkeypatch, opportunities: int) -> None:
    """Write a checkpoint at every ``opportunities``-th place one could be written.

    Every place, rather than once a minute, so that a test of a few hundred games has many; and
    not at all of them, so that a build is killed with records past its last checkpoint.
    """
    monkeypatch.setattr(builder, "CHECKPOINT_SECONDS", 0.0)
    real = builder._Build.checkpoint
    calls = [0]

    def sometimes(self, *, force=False):
        calls[0] += 1
        if force or calls[0] % opportunities == 0:
            real(self, force=True)

    monkeypatch.setattr(builder._Build, "checkpoint", sometimes)


def reading(patch, mode: str) -> None:
    """Read the way ``mode`` says: one process, several, or several with stretches read here."""
    if mode == "serial":
        return
    patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
    patch.setattr(builder, "CHUNK_BYTES", 900)
    if mode == "stretches":
        # Pieces too large for a worker are read in the build's own process, a game at a time,
        # and checkpointed by how many of their games are in.
        patch.setattr(builder, "MAX_PIECE_BYTES", 1200)


def sources(tmp_path: Path, kind: str, mode: str = "serial") -> list[str]:
    """Files of mixed games, broken ones included, plain or compressed.

    For ``mode`` "stretches", also one whose games are not separated by blank lines, which
    cannot be cut and so is read in the build's own process.
    """
    files = [mixed_pgn(tmp_path / "mixed.pgn")]
    if mode == "stretches":
        flat = tmp_path / "flat.pgn"
        flat.write_text(BACK_TO_BACK.replace("1500", "1700") * 40)
        files.append(flat)
    files.append(varied_pgn(tmp_path / "varied.pgn", games=30))
    if kind == "zst":
        return [str(compressed(path, frames=3)) for path in files]
    return [str(path) for path in files]


def workers(mode: str) -> int:
    return 1 if mode == "serial" else 3


def interrupted_then_resumed(run, resume, kills, monkeypatch, where, places: list) -> Manifest:
    """Run ``run``, killed at each write in ``kills`` in turn and resumed after each.

    ``where`` is the directory the checkpoint is in, and what each one said to carry on from is
    put in ``places``: (source, offset, games passed).
    """
    for at in kills:
        with monkeypatch.context() as patch:
            killing(patch, at)
            with pytest.raises(Killed):
                run() if at == kills[0] else resume()
        checkpoint = load_checkpoint(where())
        assert checkpoint is not None
        places.append((checkpoint.source, checkpoint.offset, checkpoint.passing))
    return resume()


@pytest.mark.parametrize("kind", ["plain", "zst"])
@pytest.mark.parametrize("mode", ["serial", "parallel", "stretches"])
def test_a_resumed_build_is_the_build_that_was_never_interrupted(tmp_path, monkeypatch, kind, mode):
    named = sources(tmp_path, kind, mode)
    with monkeypatch.context() as patch:
        reading(patch, mode)
        writes = stream_appends(patch)
        expected = build(tmp_path / "whole", *named, workers=workers(mode), **SETTINGS)
    total = writes[0]
    assert total > 40, "enough writes to kill the build at several places"

    places: list[tuple[int, int, int]] = []
    for at in (1, 7, total // 3, total // 2 + 1, total * 2 // 3, total - 1):
        data_dir = tmp_path / f"killed-{at}"
        with monkeypatch.context() as patch:
            reading(patch, mode)
            checkpoint_every(patch, 3)
            manifest = interrupted_then_resumed(
                lambda data_dir=data_dir: build(
                    data_dir, *named, workers=workers(mode), **SETTINGS
                ),
                lambda data_dir=data_dir: build_dataset(
                    "test", [], data_dir=data_dir, resume=True, workers=workers(mode)
                ),
                # And the resumed build killed in turn, a little further on, where it has that far
                # to go.
                [at, 3] if at <= total // 2 + 1 else [at],
                patch,
                lambda data_dir=data_dir: builder.interrupted_builds(data_dir, "test")[0],
                places,
            )

        assert shard_bytes(dataset_path(data_dir, "test")) == shard_bytes(
            dataset_path(tmp_path / "whole", "test")
        ), f"killed at write {at}"
        assert comparable(manifest) == comparable(expected)
        assert not (dataset_path(data_dir, "test") / CHECKPOINT_FILE).exists()
        assert not builder.interrupted_builds(data_dir, "test")
    # Carried on part way into a file, not from the start of one, and into a stretch read a game
    # at a time where there are any: games passed rather than a piece's start.
    assert any(offset or passing for _, offset, passing in places), places
    if mode != "parallel":
        assert any(passing for _, _, passing in places), places


@pytest.mark.parametrize("kind", ["plain", "zst"])
@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_a_resumed_append_is_the_append_that_was_never_interrupted(
    tmp_path, monkeypatch, kind, mode
):
    # Both files appended, one after the other: the filter leaves out every game of the second,
    # and a version that kept nothing would not be one.
    appended = sources(tmp_path, kind)
    build(tmp_path / "whole", "lichess.pgn", **SETTINGS)
    with monkeypatch.context() as patch:
        reading(patch, mode)
        writes = stream_appends(patch)
        expected = append_dataset(
            "test", appended, data_dir=tmp_path / "whole", workers=workers(mode)
        )
    total = writes[0]

    places: list[tuple[int, int, int]] = []
    for at in (1, total // 2, total - 1):
        data_dir = tmp_path / f"killed-{at}"
        build(data_dir, "lichess.pgn", **SETTINGS)
        with monkeypatch.context() as patch:
            reading(patch, mode)
            checkpoint_every(patch, 2)
            manifest = interrupted_then_resumed(
                lambda data_dir=data_dir: append_dataset(
                    "test", appended, data_dir=data_dir, workers=workers(mode)
                ),
                lambda data_dir=data_dir: append_dataset(
                    "test", [], data_dir=data_dir, resume=True, workers=workers(mode)
                ),
                [at],
                patch,
                lambda data_dir=data_dir: dataset_path(data_dir, "test"),
                places,
            )

        assert shard_bytes(dataset_path(data_dir, "test")) == shard_bytes(
            dataset_path(tmp_path / "whole", "test")
        ), f"killed at write {at}"
        assert comparable(manifest) == comparable(expected)
        assert not (dataset_path(data_dir, "test") / CHECKPOINT_FILE).exists()
    assert any(offset or passing for _, offset, passing in places), places


def interrupted_append(data_dir, monkeypatch, *names, at=5) -> None:
    """Build "test" from lichess.pgn and kill an append of ``names`` at its ``at``-th write."""
    build(data_dir, "lichess.pgn", **SETTINGS)
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)
        killing(patch, at)
        with pytest.raises(Killed):
            append_dataset("test", [fixture(name) for name in names], data_dir=data_dir)


def test_an_interrupted_append_leaves_a_checkpoint_and_the_previous_version_readable(
    tmp_path, monkeypatch
):
    interrupted_append(tmp_path, monkeypatch, "unrated.pgn", at=8)
    directory = dataset_path(tmp_path, "test")
    built = build(tmp_path / "clean", "lichess.pgn", **SETTINGS)

    checkpoint = load_checkpoint(directory)
    assert checkpoint is not None and checkpoint.version == 2
    assert [Path(source.path).name for source in checkpoint.sources] == ["unrated.pgn"]
    dataset = open_dataset("test", data_dir=tmp_path)
    assert dataset.manifest.version == 1
    assert dataset.manifest.games == built.games


def test_an_append_of_other_sources_over_an_interrupted_one_is_refused_without_a_terminal(
    tmp_path, monkeypatch
):
    interrupted_append(tmp_path, monkeypatch, "unrated.pgn")
    left = shard_bytes(dataset_path(tmp_path, "test"))

    with pytest.raises(DatasetError, match="--resume.*--discard-interrupted") as refused:
        append_dataset("test", [fixture("custom-start.pgn")], data_dir=tmp_path)

    assert "unrated.pgn" in str(refused.value), "it says which append is in the way"
    assert shard_bytes(dataset_path(tmp_path, "test")) == left, "and touches nothing"
    assert load_checkpoint(dataset_path(tmp_path, "test")) is not None


def test_an_append_over_an_interrupted_one_aborts_when_told_to(tmp_path, monkeypatch):
    interrupted_append(tmp_path, monkeypatch, "unrated.pgn")
    left = shard_bytes(dataset_path(tmp_path, "test"))
    asked = []

    def no(question):
        asked.append(question)
        return False

    with pytest.raises(DatasetError, match="left the interrupted append.*alone.*--resume"):
        append_dataset("test", [fixture("custom-start.pgn")], data_dir=tmp_path, confirm=no)

    assert len(asked) == 1 and "unrated.pgn" in asked[0] and "Discard" in asked[0]
    assert shard_bytes(dataset_path(tmp_path, "test")) == left
    # And it can still be carried on.
    resumed = append_dataset("test", [], data_dir=tmp_path, resume=True)
    assert resumed.version == 2 and resumed.versions[-1].sources[0].path == fixture("unrated.pgn")


def test_discarding_an_interrupted_append_leaves_the_previous_version_byte_identical(
    tmp_path, monkeypatch
):
    build(tmp_path / "clean", "lichess.pgn", **SETTINGS)
    version_1 = shard_bytes(dataset_path(tmp_path / "clean", "test"))
    expected = append_dataset("test", [fixture("custom-start.pgn")], data_dir=tmp_path / "clean")
    interrupted_append(tmp_path / "data", monkeypatch, "unrated.pgn", at=6)
    directory = dataset_path(tmp_path / "data", "test")
    assert records_past(directory, load_manifest(directory)), "it had written past version 1"

    with monkeypatch.context() as patch:
        # Stopped right after the discarding, to look at what it left.
        def stop(*args, **kwargs):
            raise Killed

        patch.setattr(builder._Build, "run", stop)
        with pytest.raises(Killed):
            append_dataset(
                "test", [fixture("custom-start.pgn")], data_dir=tmp_path / "data", confirm=bool
            )
    assert shard_bytes(directory) == version_1
    assert load_checkpoint(directory) is None

    manifest = append_dataset("test", [fixture("custom-start.pgn")], data_dir=tmp_path / "data")
    assert shard_bytes(directory) == shard_bytes(dataset_path(tmp_path / "clean", "test"))
    assert comparable(manifest) == comparable(expected)


def test_resuming_with_nothing_interrupted_says_so(tmp_path):
    build(tmp_path, "lichess.pgn")

    with pytest.raises(DatasetError, match="no interrupted append"):
        append_dataset("test", [], data_dir=tmp_path, resume=True)
    with pytest.raises(DatasetError, match="no interrupted build"):
        build_dataset("other", [], data_dir=tmp_path, resume=True)


def test_resuming_takes_no_sources(tmp_path, monkeypatch):
    interrupted_append(tmp_path, monkeypatch, "unrated.pgn")

    with pytest.raises(DatasetError, match="name none"):
        append_dataset("test", [fixture("unrated.pgn")], data_dir=tmp_path, resume=True)
    with pytest.raises(DatasetError, match="maximum"):
        append_dataset("test", [], data_dir=tmp_path, resume=True, max_games=100)


def test_a_source_that_changed_is_not_resumed_from_and_nothing_is_lost(tmp_path, monkeypatch):
    source = tmp_path / "games.pgn"
    shutil.copy(fixture("unrated.pgn"), source)
    build(tmp_path / "data", "lichess.pgn", **SETTINGS)
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)
        killing(patch, 5)
        with pytest.raises(Killed):
            append_dataset("test", [str(source)], data_dir=tmp_path / "data")
    directory = dataset_path(tmp_path / "data", "test")
    left = shard_bytes(directory)
    original = source.read_bytes()

    source.write_bytes(original.replace(b"1-0", b"0-1"))  # The same size, other games.
    with pytest.raises(DatasetError, match="contents have changed"):
        append_dataset("test", [], data_dir=tmp_path / "data", resume=True)
    source.write_bytes(original + b"\n")
    with pytest.raises(DatasetError, match="was .* bytes and is"):
        append_dataset("test", [], data_dir=tmp_path / "data", resume=True)

    assert shard_bytes(directory) == left, "refused before anything was cut back"
    source.write_bytes(original)
    resumed = append_dataset("test", [], data_dir=tmp_path / "data", resume=True)
    build(tmp_path / "clean", "lichess.pgn", **SETTINGS)
    expected = append_dataset("test", [str(source)], data_dir=tmp_path / "clean")
    assert comparable(resumed) == comparable(expected)


def test_a_checkpoint_left_by_an_append_that_finished_is_cleared_away(tmp_path, monkeypatch):
    # Killed between writing the manifest and removing the checkpoint: the append is done.
    build(tmp_path, "lichess.pgn")
    with monkeypatch.context() as patch:
        patch.setattr(builder, "remove_checkpoint", lambda directory: None)
        append_dataset("test", [fixture("unrated.pgn")], data_dir=tmp_path)
    assert load_checkpoint(dataset_path(tmp_path, "test")) is not None

    manifest = append_dataset("test", [fixture("custom-start.pgn")], data_dir=tmp_path)

    assert manifest.version == 3
    assert load_checkpoint(dataset_path(tmp_path, "test")) is None


def test_an_append_killed_while_publishing_resumes(tmp_path, monkeypatch):
    build(tmp_path, "lichess.pgn")
    with monkeypatch.context() as patch:

        def killed(self, directory):
            raise Killed

        patch.setattr(Manifest, "save", killed)
        with pytest.raises(Killed):
            append_dataset("test", [fixture("unrated.pgn")], data_dir=tmp_path)

    assert append_dataset("test", [], data_dir=tmp_path, resume=True).games == 7


def test_an_interrupted_build_asks_before_it_is_thrown_away(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)
        killing(patch, 6)
        with pytest.raises(Killed):
            build(tmp_path, "lichess.pgn", **SETTINGS)
    (partial,) = builder.interrupted_builds(tmp_path, "test")
    asked = []

    with pytest.raises(DatasetError, match="left the interrupted build.*alone"):
        build(tmp_path, "unrated.pgn", confirm=lambda question: asked.append(question) or False)
    assert partial.is_dir() and "lichess.pgn" in asked[0]

    manifest = build(tmp_path, "unrated.pgn", confirm=lambda question: True)

    assert manifest.games == 3
    assert not partial.exists()


def test_two_interrupted_builds_are_not_guessed_between(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)
        killing(patch, 6)
        with pytest.raises(Killed):
            build(tmp_path, "lichess.pgn")
    (partial,) = builder.interrupted_builds(tmp_path, "test")
    shutil.copytree(partial, store.new_partial_path(tmp_path, "test"))

    with pytest.raises(DatasetError, match="2 interrupted builds.*remove all but"):
        build_dataset("test", [], data_dir=tmp_path, resume=True)


def test_a_build_interrupted_before_its_first_checkpoint_leaves_nothing(tmp_path, monkeypatch):
    # Killed while taking the checksums: nothing was read, so there is nothing to carry on.
    def killed(path, on_read=None):
        raise Killed

    with monkeypatch.context() as patch:
        patch.setattr(builder, "sha256_of", killed)
        with pytest.raises(Killed):
            build(tmp_path, "lichess.pgn")

    assert not builder.interrupted_builds(tmp_path, "test")
    assert not store.abandoned_partials(tmp_path, "test")


def test_a_resumed_build_reports_its_progress_from_where_it_was(tmp_path, monkeypatch):
    # The line carries on from the checkpoint rather than starting from nothing and racing back,
    # and the rate counts only the games read since it resumed.
    named = sources(tmp_path, "plain")
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 4)
        killing(patch, 60)
        with pytest.raises(Killed):
            build(tmp_path, *named, workers=1, **SETTINGS)
    (partial,) = builder.interrupted_builds(tmp_path, "test")
    checkpoint = load_checkpoint(partial)
    assert checkpoint is not None and checkpoint.bytes_read > 0
    reports = []
    out = io.StringIO()
    printer = ProgressPrinter(out, interval=0.0, rewrite=False)

    def both(progress):
        reports.append(progress)
        printer(progress)

    build_dataset("test", [], data_dir=tmp_path, resume=True, workers=1, progress=both)

    reading_reports = [report for report in reports if not report.hashing]
    assert reading_reports[0].bytes_read == checkpoint.bytes_read
    read = [report.bytes_read for report in reading_reports]
    assert read == sorted(read), "never back to the start of the file it passes over"
    assert all(report.games_before == checkpoint.counts.games_read for report in reports)
    printed = out.getvalue().splitlines()
    assert printed[-1].startswith("built"), printed


def test_a_checkpoint_that_would_cut_into_an_earlier_version_is_refused(tmp_path, monkeypatch):
    interrupted_append(tmp_path, monkeypatch, "unrated.pgn")
    directory = dataset_path(tmp_path, "test")
    checkpoint = load_checkpoint(directory)
    assert checkpoint is not None
    checkpoint.model_copy(update={"splits": {}}).save(directory)
    version_1 = load_manifest(directory)
    before = shard_bytes(directory)

    with pytest.raises(DatasetError, match="fewer records than version 1"):
        append_dataset("test", [], data_dir=tmp_path, resume=True)

    assert shard_bytes(directory) == before
    assert load_manifest(directory) == version_1


# --- The end of a build -------------------------------------------------------------------------


def test_a_build_whose_last_flush_fails_is_kept_to_resume(tmp_path, monkeypatch):
    # The disk fills on the final flush of the shards, after every game has been read. The last
    # checkpoint is at most a minute old, and throwing it away threw away the whole build.
    import errno

    expected = build(tmp_path / "clean", "lichess.pgn", **SETTINGS)
    real = store.DatasetWriter.close

    def close(self):
        # The flushes of every shard still open, which is the last thing the reading does.
        real(self)
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)
        patch.setattr(store.DatasetWriter, "close", close)
        with pytest.raises(OSError, match="No space left"):
            build(tmp_path / "data", "lichess.pgn", **SETTINGS)

    assert builder.interrupted_builds(tmp_path / "data", "test"), "kept, with its checkpoint"
    manifest = build_dataset("test", [], data_dir=tmp_path / "data", resume=True)
    assert comparable(manifest) == comparable(expected)
    assert shard_bytes(dataset_path(tmp_path / "data", "test")) == shard_bytes(
        dataset_path(tmp_path / "clean", "test")
    )


def test_a_build_interrupted_while_its_manifest_is_written_is_kept_to_resume(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)

        def killed(self, directory):
            raise Killed

        patch.setattr(Manifest, "save", killed)
        with pytest.raises(Killed):
            build(tmp_path, "lichess.pgn", **SETTINGS)

    assert builder.interrupted_builds(tmp_path, "test")
    assert build_dataset("test", [], data_dir=tmp_path, resume=True).version == 1


def test_a_build_killed_after_publishing_leaves_a_checkpoint_the_next_append_clears(
    tmp_path, monkeypatch
):
    # Killed between putting the dataset in place and removing the checkpoint that came with it:
    # the build is done, and the checkpoint is of a version the dataset has.
    with monkeypatch.context() as patch:
        real = builder.remove_checkpoint

        def killed(directory):
            if directory == dataset_path(tmp_path, "test"):
                raise Killed
            real(directory)

        patch.setattr(builder, "remove_checkpoint", killed)
        with pytest.raises(Killed):
            build(tmp_path, "lichess.pgn")

    assert open_dataset("test", data_dir=tmp_path).manifest.version == 1
    assert not builder.interrupted_builds(tmp_path, "test")
    assert append_dataset("test", [fixture("unrated.pgn")], data_dir=tmp_path).version == 2
    assert load_checkpoint(dataset_path(tmp_path, "test")) is None


def test_a_build_resumes_from_another_working_directory(tmp_path, monkeypatch):
    # Sources named relative to where the build was started must still be found by a resume
    # started somewhere else: a new shell after a reboot, or cron.
    started = tmp_path / "started"
    started.mkdir()
    shutil.copy(fixture("lichess.pgn"), started / "games.pgn")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(started)
    expected = build(tmp_path / "clean", "games.pgn", **SETTINGS)
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 2)
        killing(patch, 6)
        with pytest.raises(Killed):
            build(tmp_path / "data", "games.pgn", **SETTINGS)

    monkeypatch.chdir(elsewhere)
    manifest = build_dataset("test", [], data_dir=tmp_path / "data", resume=True)

    assert comparable(manifest) == comparable(expected), "the manifest names it as it was given"
    assert manifest.sources[0].path == "games.pgn"


def test_the_time_left_does_not_count_getting_back_to_the_checkpoint(tmp_path, monkeypatch):
    # Getting back to its place can take minutes of decompressing or passing over games, in
    # which nothing is read. Counted as reading time, it made a build resumed at 90% say it had
    # most of an hour left when it had minutes.
    from chess_ai.dataset import sources as sources_module

    named = sources(tmp_path, "plain")
    with monkeypatch.context() as patch:
        checkpoint_every(patch, 4)
        killing(patch, 60)
        with pytest.raises(Killed):
            build(tmp_path, *named, workers=1, **SETTINGS)
    (partial,) = builder.interrupted_builds(tmp_path, "test")
    assert load_checkpoint(partial).passing > 0, "a resume that passes over games"

    clock = [0.0]
    monkeypatch.setattr(builder.time, "monotonic", lambda: clock[0])
    real_pass = sources_module.pass_games

    def slow(text, games):
        clock[0] += 1000
        return real_pass(text, games)

    monkeypatch.setattr(sources_module, "pass_games", slow)
    real_report = builder._Build.report

    def ticking(self, **kwargs):
        clock[0] += 1
        real_report(self, **kwargs)

    monkeypatch.setattr(builder._Build, "report", ticking)
    reports = []
    out = io.StringIO()
    printer = ProgressPrinter(out, interval=0.0, rewrite=False)

    def both(progress):
        reports.append(progress)
        printer(progress)

    build_dataset("test", [], data_dir=tmp_path, resume=True, workers=1, progress=both)

    reading_reports = [
        report
        for report in reports
        if not (report.hashing or report.scanning or report.done)
        and report.games_read > report.games_before
    ]
    assert reading_reports
    assert all(report.reading_seconds < 1000 for report in reading_reports), [
        report.reading_seconds for report in reading_reports
    ]


@pytest.mark.parametrize("kind", ["plain", "zst"])
def test_a_parallel_resume_restarts_its_clock_before_handing_out_a_piece(
    tmp_path, monkeypatch, kind
):
    # Restarted when the first piece came back, the clock had the whole first wave of pieces --
    # one per worker, read alongside it -- counted as read in no time, and the time left started
    # out near zero. A plain file has nothing to catch up at all.
    named = sources(tmp_path, kind)
    with monkeypatch.context() as patch:
        reading(patch, "parallel")
        checkpoint_every(patch, 3)
        killing(patch, 60)
        with pytest.raises(Killed):
            build(tmp_path, *named, workers=3, **SETTINGS)
    (partial,) = builder.interrupted_builds(tmp_path, "test")
    assert load_checkpoint(partial).offset > 0, "resumed part way into a file"

    submitted = [0]
    at_restart = []

    class Counting(builder.ProcessPoolExecutor):
        def submit(self, *args, **kwargs):
            submitted[0] += 1
            return super().submit(*args, **kwargs)

    real = builder._Build._caught_up

    def caught_up(self):
        if self._catching_up:
            at_restart.append(submitted[0])
        real(self)

    monkeypatch.setattr(builder, "ProcessPoolExecutor", Counting)
    monkeypatch.setattr(builder._Build, "_caught_up", caught_up)
    reading(monkeypatch, "parallel")
    build_dataset("test", [], data_dir=tmp_path, resume=True, workers=3)

    assert at_restart == [0]


@pytest.mark.parametrize("kind", ["plain", "zst"])
def test_a_parallel_resume_inside_a_stretch_restarts_its_clock_after_passing_its_games(
    tmp_path, monkeypatch, kind
):
    # The first job is the stretch, read in this process after passing the games it had read;
    # the pieces after it are pulled for handing out before that starts. One of those restarting
    # the clock counted the passing as reading again.
    from chess_ai.dataset import sources as sources_module

    named = sources(tmp_path, kind, "stretches")
    # From where the stretch starts in these files, searching on in case they change.
    for at in range(100, 400, 10):
        data_dir = tmp_path / f"killed-{at}"
        with monkeypatch.context() as patch:
            reading(patch, "stretches")
            checkpoint_every(patch, 2)
            killing(patch, at)
            with pytest.raises(Killed):
                build(data_dir, *named, workers=3, **SETTINGS)
        checkpoint = load_checkpoint(builder.interrupted_builds(data_dir, "test")[0])
        if checkpoint.source == 1 and checkpoint.passing:
            break
    else:
        pytest.fail("no kill point landed inside the stretch")

    events = []
    real_pass = sources_module.pass_games

    def passing(text, games):
        passed = real_pass(text, games)
        if games:
            events.append("passed")
        return passed

    real_caught_up = builder._Build._caught_up

    def caught_up(self):
        if self._catching_up:
            events.append("restarted")
        real_caught_up(self)

    monkeypatch.setattr(sources_module, "pass_games", passing)
    monkeypatch.setattr(builder._Build, "_caught_up", caught_up)
    reading(monkeypatch, "stretches")
    build_dataset("test", [], data_dir=data_dir, resume=True, workers=3)

    assert events == ["passed", "restarted"]


def test_a_resume_at_the_last_game_of_a_stretch_restarts_its_clock_once_it_has_passed_it(
    tmp_path, monkeypatch
):
    # Nothing is left of the stretch past the games it passes, so no game of it restarts the
    # clock. Left to the first piece a worker hands back, the restart counted that piece and the
    # rest of its wave -- read while the games were passed -- as read in no time.
    from chess_ai.dataset import sources as sources_module

    flat = tmp_path / "flat.pgn"
    flat.write_text(BACK_TO_BACK.replace("1500", "1700") * 40)
    named = [str(flat), str(mixed_pgn(tmp_path / "mixed.pgn"))]
    with monkeypatch.context() as patch:
        reading(patch, "stretches")
        # Checkpointed only part way into the stretch, so the last one is on its last game.
        real = builder._Build.checkpoint

        def in_the_stretch(self, *, force=False):
            if force or self._position[2]:
                real(self, force=True)

        patch.setattr(builder, "CHECKPOINT_SECONDS", 0.0)
        patch.setattr(builder._Build, "checkpoint", in_the_stretch)
        build(tmp_path / "whole", *named, workers=3, **SETTINGS)
    expected = shard_bytes(dataset_path(tmp_path / "whole", "test"))
    # Past the stretch's 40 games of four writes each, searching on in case the files change.
    for at in range(160, 400, 5):
        data_dir = tmp_path / f"killed-{at}"
        with monkeypatch.context() as patch:
            reading(patch, "stretches")
            patch.setattr(builder, "CHECKPOINT_SECONDS", 0.0)
            patch.setattr(builder._Build, "checkpoint", in_the_stretch)
            killing(patch, at)
            with pytest.raises(Killed):
                build(data_dir, *named, workers=3, **SETTINGS)
        checkpoint = load_checkpoint(builder.interrupted_builds(data_dir, "test")[0])
        if (checkpoint.source, checkpoint.passing) == (0, 40):
            break
    else:
        pytest.fail("no kill point fell after the stretch's last game")

    events = []
    real_pass = sources_module.pass_games

    def passing(text, games):
        passed = real_pass(text, games)
        if games:
            events.append("passed")
        return passed

    real_caught_up = builder._Build._caught_up

    def caught_up(self):
        if self._catching_up:
            events.append("restarted")
        real_caught_up(self)

    real_read_here = builder._Build._read_here

    def read_here(self, job, state, *, before):
        real_read_here(self, job, state, before=before)
        events.append("stretch done")

    real_take = builder._Build._take

    def take(self, read, state):
        if "took" not in events:
            events.append("took")
        real_take(self, read, state)

    monkeypatch.setattr(sources_module, "pass_games", passing)
    monkeypatch.setattr(builder._Build, "_caught_up", caught_up)
    monkeypatch.setattr(builder._Build, "_read_here", read_here)
    monkeypatch.setattr(builder._Build, "_take", take)
    reading(monkeypatch, "stretches")
    build_dataset("test", [], data_dir=data_dir, resume=True, workers=3)

    # Restarted as the passing ended, not when the first piece of the wave was taken.
    assert events[:4] == ["passed", "restarted", "stretch done", "took"]
    assert shard_bytes(dataset_path(data_dir, "test")) == expected
