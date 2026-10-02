"""The run directory: what it writes, what it keeps, and what a reader sees mid-write."""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import pytest

from chess_ai.encoders import create_encoder
from chess_ai.training import run_store
from chess_ai.training.run_store import (
    CHECKPOINT_INDEX,
    CHECKPOINTS_DIR,
    CONFIG_FILE,
    METRICS_FILE,
    NOTES_FILE,
    STATUS_FILE,
    CheckpointPolicy,
    DatasetReference,
    Lineage,
    ModelReference,
    RunError,
    RunInfo,
    RunNotes,
    RunReader,
    RunStatus,
    RunWriter,
    checkpoint_file,
    choose_checkpoint,
    code_version,
    list_runs,
    open_run,
    read_checkpoints,
    run_path,
    save_notes,
)


def info(name: str = "test", **overrides) -> RunInfo:
    """A plausible ``run.json``, for a run whose contents are not what is being tested."""
    fields = {
        "name": name,
        "created": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "seed": 7,
        "code_version": "git abc1234",
        "device": "cpu (8 threads)",
        "dataset": DatasetReference(
            name="games",
            directory="/data/datasets/games",
            format_version=1,
            games=10,
            positions=500,
            train_positions=450,
            validation_positions=50,
        ),
        "encoder": create_encoder("board-planes").spec,
        "model": ModelReference(architecture="mlp", options={"width": 8}, parameter_count=1234),
        "config": {"name": name, "seed": 7},
        "steps": 100,
        "batch_size": 8,
    }
    return RunInfo(**(fields | overrides))


def writer(tmp_path, name: str = "test", **options) -> RunWriter:
    """A run writer over a fresh directory, with the run info a test does not care about."""
    policy = options.pop("policy", CheckpointPolicy(keep=2))
    return RunWriter.create(
        run_path(tmp_path, name),
        info(name, **options),
        config_text="# the config as written\n",
        policy=policy,
    )


def save(run: RunWriter, step: int, **metrics: float):
    """Save a checkpoint whose contents are just the step, since only the layout is at stake."""
    return run.save_checkpoint(
        step, lambda path: path.write_text(f"weights for step {step}"), metrics=metrics
    )


def test_a_new_run_writes_the_config_and_what_it_resolved_to(tmp_path):
    with writer(tmp_path):
        pass

    reader = open_run(tmp_path, "test")
    assert reader.config_text == "# the config as written\n"
    assert reader.info.seed == 7
    assert reader.info.model.architecture == "mlp"
    assert reader.info.encoder.encoder == "board-planes"
    assert reader.info.dataset.train_positions == 450


def test_a_run_lists_itself_once_it_has_a_run_file(tmp_path):
    assert list_runs(tmp_path) == []
    (tmp_path / "not-a-run").mkdir(parents=True)

    with writer(tmp_path, "second"), writer(tmp_path, "first"):
        pass

    assert list_runs(tmp_path) == ["first", "second"]


def test_a_missing_runs_directory_lists_nothing(tmp_path):
    assert list_runs(tmp_path / "nowhere") == []


def test_a_name_that_is_not_a_directory_name_is_refused(tmp_path):
    for name in ("../escape", "with/slash", ".hidden", ""):
        with pytest.raises(RunError, match="invalid run name"):
            run_path(tmp_path, name)


def test_a_second_run_of_the_same_name_is_refused(tmp_path):
    with writer(tmp_path) as run:
        save(run, 1)

    with pytest.raises(RunError, match="already in .*choose another name"):
        writer(tmp_path)

    assert open_run(tmp_path, "test").checkpoints(), "the first run is left alone"


def test_overwriting_leaves_nothing_of_the_run_it_replaced(tmp_path):
    """Two runs' metrics in one log would be a chart nobody could read."""
    with writer(tmp_path) as first:
        first.log(step=1, loss=9.0)
        save(first, 1, policy_loss=9.0)
        first.heartbeat(RunStatus.RUNNING, steps=100)
    save_notes(open_run(tmp_path, "test"), RunNotes(title="the old one", tags=["best"]))

    replaced = RunWriter.create(
        run_path(tmp_path, "test"),
        info("test"),
        config_text="# newer\n",
        policy=CheckpointPolicy(keep=2),
        overwrite=True,
    )
    with replaced as run:
        run.log(step=1, loss=1.0)

    reader = open_run(tmp_path, "test")
    assert [line["loss"] for line in reader.metrics()] == [1.0]
    assert reader.checkpoints() == []
    assert reader.status is None
    assert reader.notes == RunNotes()
    assert reader.config_text == "# newer\n"


def test_metrics_are_appended_in_the_order_they_were_logged(tmp_path):
    with writer(tmp_path) as run:
        run.log(step=1, split="train", loss=9.0)
        run.log(step=1, split="validation", loss=8.5, top1=0.1)
        run.log(step=2, split="train", loss=7.0)

    lines = open_run(tmp_path, "test").metrics()
    assert [(line["step"], line["split"]) for line in lines] == [
        (1, "train"),
        (1, "validation"),
        (2, "train"),
    ]
    assert lines[1]["top1"] == 0.1


def test_a_loss_that_diverged_is_logged_as_null_rather_than_nan(tmp_path):
    """NaN is not JSON, and a diverged run is exactly when the log is being read.

    ``json.dumps`` writes the bare tokens ``NaN`` and ``Infinity`` by default. Python's own
    parser accepts them, so the text has to be checked rather than the parsed value; a
    browser's ``JSON.parse`` throws, losing the whole line rather than the one field.
    """
    with writer(tmp_path) as run:
        run.log(step=1, split="train", loss=float("nan"), policy_loss=float("inf"), top1=0.5)

    text = (run_path(tmp_path, "test") / METRICS_FILE).read_text()
    assert "NaN" not in text and "Infinity" not in text, text
    line = json.loads(text)
    assert line["loss"] is None and line["policy_loss"] is None
    assert line["top1"] == 0.5, "the fields that are numbers are still numbers"
    assert open_run(tmp_path, "test").metrics() == [line]


def test_a_reader_can_ask_only_for_metrics_it_has_not_seen(tmp_path):
    with writer(tmp_path) as run:
        for step in range(1, 4):
            run.log(step=step, loss=float(step))

        assert [line["step"] for line in open_run(tmp_path, "test").metrics(skip=2)] == [3]
        run.log(step=4, loss=4.0)
        assert [line["step"] for line in open_run(tmp_path, "test").metrics(skip=3)] == [4]


def test_a_half_written_metrics_line_is_skipped_and_read_whole_next_time(tmp_path):
    """What a reader tailing a log sees while the trainer is in the middle of a write."""
    with writer(tmp_path) as run:
        run.log(step=1, loss=9.0)
    path = run_path(tmp_path, "test") / METRICS_FILE
    whole = path.read_text()
    with path.open("a", encoding="utf-8") as f:
        f.write('{"step": 2, "lo')

    reader = RunReader(run_path(tmp_path, "test"))
    assert [line["step"] for line in reader.metrics()] == [1]

    path.write_text(whole + '{"step": 2, "loss": 8.0}\n')
    assert [line["step"] for line in reader.metrics()] == [1, 2]


def test_a_metrics_log_that_is_not_there_yet_reads_as_empty(tmp_path):
    (tmp_path / "test").mkdir()

    assert RunReader(tmp_path / "test").metrics() == []


def steps(records: list[dict]) -> list[int]:
    return [record["step"] for record in records]


def test_a_tail_returns_the_whole_log_first_and_then_only_what_is_new(tmp_path):
    with writer(tmp_path) as run:
        run.log(step=1, loss=9.0)
        run.log(step=2, loss=8.0)
        tail = open_run(tmp_path, "test").tail_metrics()

        records, replace = tail.read()
        assert (steps(records), replace) == ([1, 2], True)
        assert tail.read() == ([], False)

        run.log(step=3, loss=7.0)
        records, replace = tail.read()
        assert (steps(records), replace) == ([3], False)


def test_a_tail_leaves_a_half_written_line_for_the_next_read(tmp_path):
    with writer(tmp_path) as run:
        run.log(step=1, loss=9.0)
    path = run_path(tmp_path, "test") / METRICS_FILE
    tail = RunReader(run_path(tmp_path, "test")).tail_metrics()
    with path.open("a", encoding="utf-8") as f:
        f.write('{"step": 2, "lo')
        f.flush()
        assert steps(tail.read()[0]) == [1]
        f.write('ss": 8.0}\n')
        f.flush()
        assert tail.read() == ([{"step": 2, "loss": 8.0}], False)


def test_a_tail_starts_again_when_the_run_is_overwritten(tmp_path):
    """A new run under the same name is a new history, not more of the old one."""
    with writer(tmp_path) as run:
        for step in (1, 2, 3):
            run.log(step=step, loss=1.0)
    tail = open_run(tmp_path, "test").tail_metrics()
    tail.read()

    replaced = RunWriter.create(
        run_path(tmp_path, "test"),
        info("test"),
        config_text="",
        policy=CheckpointPolicy(keep=2),
        overwrite=True,
    )
    with replaced as run:
        run.log(step=1, loss=5.0)

    records, replace = tail.read()
    assert (steps(records), replace) == ([1], True)
    assert records[0]["loss"] == 5.0


def test_a_tail_starts_again_when_the_log_goes_away(tmp_path):
    with writer(tmp_path) as run:
        run.log(step=1, loss=1.0)
    tail = open_run(tmp_path, "test").tail_metrics()
    tail.read()
    (run_path(tmp_path, "test") / METRICS_FILE).unlink()

    assert tail.read() == ([], True)
    assert tail.read() == ([], False)


def test_a_tail_reads_nan_from_an_old_log_as_null(tmp_path):
    """Logs written before ``log`` wrote ``null`` may hold bare tokens a browser cannot parse."""
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / METRICS_FILE).write_text('{"step": 1, "loss": NaN, "lr": Infinity}\n')

    records, _ = RunReader(tmp_path / "test").tail_metrics().read()

    assert records == [{"step": 1, "loss": None, "lr": None}]


def test_the_latest_metrics_are_the_last_line_of_each_split(tmp_path):
    with writer(tmp_path) as run:
        run.log(step=1, split="train", loss=9.0)
        run.log(step=1, split="validation", loss=9.5)
        run.log(step=2, split="train", loss=8.0)
        run.log(step=3, split="train", loss=7.0)

    latest = open_run(tmp_path, "test").track_latest_metrics().read()

    assert latest["train"]["step"] == 3
    assert latest["validation"]["step"] == 1


def test_the_latest_metrics_keep_up_with_a_growing_log(tmp_path):
    """A split found once is kept until a later line of it replaces it."""
    with writer(tmp_path) as run:
        run.log(step=0, split="validation", top1=0.5)
        run.log(step=1, split="train", loss=9.0)
        latest = open_run(tmp_path, "test").track_latest_metrics()
        assert latest.read()["train"]["step"] == 1
        path = run_path(tmp_path, "test") / METRICS_FILE
        with path.open("a", encoding="utf-8") as f:
            f.write('{"step": 2, "split": "train", "lo')

        assert latest.read() == {
            "train": {"step": 1, "split": "train", "loss": 9.0},
            "validation": {"step": 0, "split": "validation", "top1": 0.5},
        }


def test_the_latest_metrics_start_again_when_the_run_is_overwritten(tmp_path):
    with writer(tmp_path) as run:
        run.log(step=1, split="validation", top1=0.5)
    latest = open_run(tmp_path, "test").track_latest_metrics()
    latest.read()

    replaced = RunWriter.create(
        run_path(tmp_path, "test"),
        info("test"),
        config_text="",
        policy=CheckpointPolicy(keep=2),
        overwrite=True,
    )
    with replaced as run:
        run.log(step=1, split="train", loss=5.0)

    assert latest.read() == {"train": {"step": 1, "split": "train", "loss": 5.0}}


def test_a_run_with_no_metrics_has_no_latest_ones(tmp_path):
    with writer(tmp_path):
        pass

    assert open_run(tmp_path, "test").track_latest_metrics().read() == {}
    assert RunReader(tmp_path / "missing").track_latest_metrics().read() == {}


def test_the_heartbeat_is_replaced_rather_than_added_to(tmp_path):
    with writer(tmp_path) as run:
        run.heartbeat(RunStatus.RUNNING, step=10, steps=100, epoch=0.5)
        run.finish(RunStatus.FINISHED, step=100, steps=100, epoch=5.0)

    status = open_run(tmp_path, "test").status
    assert status is not None
    assert (status.status, status.step, status.epoch) == (RunStatus.FINISHED, 100, 5.0)
    written = json.loads((run_path(tmp_path, "test") / STATUS_FILE).read_text())
    assert written["status"] == "finished"


def test_a_run_that_has_not_started_has_no_heartbeat(tmp_path):
    with writer(tmp_path):
        pass

    assert open_run(tmp_path, "test").status is None


def test_a_crashed_run_says_why_in_its_heartbeat(tmp_path):
    with writer(tmp_path) as run:
        run.heartbeat(RunStatus.CRASHED, steps=100, message="RuntimeError: out of memory")

    status = open_run(tmp_path, "test").status
    assert status is not None and status.status == RunStatus.CRASHED
    assert "out of memory" in status.message


def test_a_checkpoint_is_named_after_its_step_and_sorts_by_it(tmp_path):
    with writer(tmp_path, policy=CheckpointPolicy(keep=10)) as run:
        for step in (5, 40, 1000):
            save(run, step)

    reader = open_run(tmp_path, "test")
    assert [info.step for info in reader.checkpoints()] == [5, 40, 1000]
    assert [info.file for info in reader.checkpoints()] == [
        checkpoint_file(step) for step in (5, 40, 1000)
    ]
    assert sorted(info.file for info in reader.checkpoints()) == [
        info.file for info in reader.checkpoints()
    ], "the names sort the same way the steps do"


def test_retention_keeps_the_most_recent_checkpoints(tmp_path):
    with writer(tmp_path, policy=CheckpointPolicy(keep=2)) as run:
        for step in (1, 2, 3, 4):
            save(run, step)

    reader = open_run(tmp_path, "test")
    assert [info.step for info in reader.checkpoints()] == [3, 4]
    assert not (run_path(tmp_path, "test") / CHECKPOINTS_DIR / checkpoint_file(1)).exists()


def test_retention_keeps_the_best_checkpoint_however_old_it_is(tmp_path):
    """Deleting the best checkpoint would make the metric that chose it pointless."""
    with writer(tmp_path, policy=CheckpointPolicy(keep=2, metric="policy_loss")) as run:
        save(run, 1, policy_loss=1.0)
        for step in (2, 3, 4):
            save(run, step, policy_loss=5.0)

    reader = open_run(tmp_path, "test")
    assert [info.step for info in reader.checkpoints()] == [1, 3, 4]
    assert reader.best_checkpoint().step == 1
    assert reader.latest_checkpoint().step == 4


def test_a_diverged_metric_does_not_cost_the_run_its_best_checkpoint(tmp_path):
    """One NaN must not silently delete the best weights of a run.

    pydantic writes ``null`` for a non-finite float, and a ``null`` failed to validate back
    into ``dict[str, float]``. So the index stopped parsing, every save after it rebuilt from
    the directory listing with no metrics at all, ``_best_step`` could no longer see the best
    checkpoint, and retention stopped pinning it.
    """
    with writer(tmp_path, policy=CheckpointPolicy(keep=1, metric="policy_loss")) as run:
        save(run, 1, policy_loss=1.0)
        save(run, 2, policy_loss=float("nan"))
        save(run, 3, policy_loss=5.0)
        save(run, 4, policy_loss=5.0)

    reader = open_run(tmp_path, "test")
    best = reader.best_checkpoint()
    assert best is not None and best.step == 1, "step 1 measured 1.0 and is still the best"
    assert reader.checkpoint_path(best).is_file(), "and its file was not deleted"
    assert [info.step for info in reader.checkpoints()] == [1, 4], "the newest, plus the best"
    kept = {info.step: info.metrics.get("policy_loss") for info in reader.checkpoints()}
    assert kept[1] == 1.0, "a later divergence does not erase an earlier measurement"


def test_a_diverged_metric_is_recorded_as_null_and_never_wins(tmp_path):
    """``null`` is the honest value, and the same one ``metrics.jsonl`` writes."""
    with writer(tmp_path, policy=CheckpointPolicy(keep=3, metric="policy_loss")) as run:
        save(run, 1, policy_loss=float("nan"))
        save(run, 2, policy_loss=float("inf"))
        save(run, 3, policy_loss=4.0)

    index = (run_path(tmp_path, "test") / CHECKPOINTS_DIR / CHECKPOINT_INDEX).read_text()
    assert '"policy_loss": null' in index, index
    assert "NaN" not in index and "Infinity" not in index
    reader = open_run(tmp_path, "test")
    assert reader.best_checkpoint().step == 3, "a checkpoint that cannot be judged cannot win"
    assert [info.step for info in reader.checkpoints()] == [1, 2, 3]


def test_a_diverged_metric_never_wins_when_higher_is_better_either(tmp_path):
    policy = CheckpointPolicy(keep=3, metric="top1", higher_is_better=True)
    with writer(tmp_path, policy=policy) as run:
        save(run, 1, top1=0.3)
        save(run, 2, top1=float("nan"))

    assert open_run(tmp_path, "test").best_checkpoint().step == 1


def test_the_best_checkpoint_is_the_highest_when_higher_is_better(tmp_path):
    policy = CheckpointPolicy(keep=1, metric="top1", higher_is_better=True)
    with writer(tmp_path, policy=policy) as run:
        save(run, 1, top1=0.3)
        save(run, 2, top1=0.1)

    assert open_run(tmp_path, "test").best_checkpoint().step == 1


def test_the_later_step_wins_a_tie(tmp_path):
    with writer(tmp_path, policy=CheckpointPolicy(keep=3)) as run:
        save(run, 1, policy_loss=2.0)
        save(run, 2, policy_loss=2.0)

    assert open_run(tmp_path, "test").best_checkpoint().step == 2


def test_a_checkpoint_with_no_metrics_is_never_the_best(tmp_path):
    """A checkpoint saved before the first validation has nothing to be judged on."""
    with writer(tmp_path, policy=CheckpointPolicy(keep=3)) as run:
        save(run, 1)
        save(run, 2, policy_loss=4.0)
        save(run, 3)

    reader = open_run(tmp_path, "test")
    assert reader.best_checkpoint().step == 2
    assert [info.step for info in reader.checkpoints()] == [1, 2, 3]


def test_there_is_no_best_checkpoint_before_anything_has_been_measured(tmp_path):
    with writer(tmp_path) as run:
        save(run, 1)

    assert open_run(tmp_path, "test").best_checkpoint() is None
    assert open_run(tmp_path, "test").latest_checkpoint().step == 1


def test_a_checkpoint_the_index_never_heard_of_still_counts(tmp_path):
    """The files decide what exists; a lost index costs the metrics, not the checkpoints."""
    with writer(tmp_path, policy=CheckpointPolicy(keep=5)) as run:
        save(run, 1, policy_loss=3.0)
    directory = run_path(tmp_path, "test") / CHECKPOINTS_DIR
    (directory / checkpoint_file(2)).write_text("weights for step 2")
    (directory / "index.json").unlink()

    reader = open_run(tmp_path, "test")
    assert [info.step for info in reader.checkpoints()] == [1, 2]
    assert reader.checkpoints()[0].metrics == {}


def test_an_index_entry_whose_file_is_gone_does_not_count(tmp_path):
    with writer(tmp_path, policy=CheckpointPolicy(keep=5)) as run:
        save(run, 1, policy_loss=1.0)
        save(run, 2, policy_loss=2.0)
    reader = open_run(tmp_path, "test")
    reader.checkpoint_path(reader.checkpoints()[0]).unlink()

    assert [info.step for info in reader.checkpoints()] == [2]
    assert reader.best_checkpoint().step == 2, "the best is worked out again from what is left"


def test_a_checkpoint_is_only_there_once_it_is_whole(tmp_path):
    """A reader must never be handed a path to a half-written checkpoint."""
    seen = []

    def write(path):
        seen.append(sorted(p.name for p in path.parent.iterdir()))
        path.write_text("weights")

    with writer(tmp_path) as run:
        run.save_checkpoint(1, write, metrics={})

    assert checkpoint_file(1) not in seen[0], "written under another name, then renamed into place"
    assert (run_path(tmp_path, "test") / CHECKPOINTS_DIR / checkpoint_file(1)).is_file()


def test_a_checkpoint_that_cannot_be_written_leaves_nothing_behind(tmp_path):
    def fail(path):
        path.write_text("half")
        raise OSError(28, "No space left on device")

    with (
        writer(tmp_path) as run,
        pytest.raises(RunError, match="cannot save checkpoint.*No space left"),
    ):
        run.save_checkpoint(1, fail, metrics={})

    assert open_run(tmp_path, "test").checkpoints() == []
    assert not list((run_path(tmp_path, "test") / CHECKPOINTS_DIR).glob("*.writing"))


def test_reading_a_directory_that_is_not_a_run_says_so(tmp_path):
    with pytest.raises(RunError, match="cannot read"):
        assert RunReader(tmp_path / "nowhere").info


def test_a_run_file_that_does_not_say_what_it_should_says_which_part(tmp_path):
    with writer(tmp_path):
        pass
    (run_path(tmp_path, "test") / "run.json").write_text('{"name": "test"}')

    with pytest.raises(RunError, match="does not say what it should.*seed"):
        assert open_run(tmp_path, "test").info


def test_the_code_version_says_which_commit_or_which_release():
    version = code_version()

    assert version.startswith(("git ", "version ")), version


def test_opening_a_run_that_is_not_kept_here(tmp_path):
    """Said as a missing run rather than as an empty one: the two mean different things."""
    with pytest.raises(RunError, match="no run called 'ghost'"):
        open_run(tmp_path, "ghost")


def test_opening_a_run_whose_name_could_never_be_one(tmp_path):
    with pytest.raises(RunError, match="invalid run name"):
        open_run(tmp_path, "../elsewhere")


def test_a_run_can_be_opened_before_it_has_written_anything_but_its_directory(tmp_path):
    """A checkpoint is self-contained, so a half-written run is still one worth opening."""
    (tmp_path / "started").mkdir()

    assert open_run(tmp_path, "started").checkpoints() == []


def test_choosing_the_best_the_latest_or_a_particular_checkpoint(tmp_path):
    with writer(tmp_path, policy=CheckpointPolicy(keep=3)) as run:
        save(run, 10, policy_loss=2.0)
        save(run, 20, policy_loss=1.0)
        save(run, 30, policy_loss=3.0)
    reader = open_run(tmp_path, "test")

    assert choose_checkpoint(reader, "best").step == 20
    assert choose_checkpoint(reader, "latest").step == 30
    assert choose_checkpoint(reader, 10).step == 10


def test_the_best_checkpoint_of_a_run_that_has_not_validated_yet(tmp_path):
    """Nothing to judge them by, so the newest is the only answer there is."""
    with writer(tmp_path) as run:
        save(run, 10)
        save(run, 20)

    assert choose_checkpoint(open_run(tmp_path, "test"), "best").step == 20


def test_choosing_a_checkpoint_from_a_step_that_was_never_saved(tmp_path):
    with writer(tmp_path) as run:
        save(run, 10)

    with pytest.raises(RunError, match="no checkpoint from step 15"):
        choose_checkpoint(open_run(tmp_path, "test"), 15)


def test_choosing_a_checkpoint_from_a_run_that_has_saved_none(tmp_path):
    with writer(tmp_path):
        pass

    with pytest.raises(RunError, match="not saved a checkpoint"):
        choose_checkpoint(open_run(tmp_path, "test"), "best")


class Vanished:
    """A directory entry for a checkpoint that goes between the scan and the stat.

    What a reader sees when something outside this module touches the directory: a
    checkpoint deleted by hand, or one the reader is not allowed to read.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def is_file(self) -> bool:
        # os.DirEntry swallows a FileNotFoundError here and answers False for a path that
        # has gone; what it does not swallow is the one in stat().
        return True

    def stat(self):
        raise FileNotFoundError(2, "No such file or directory", self.name)


def test_a_checkpoint_that_goes_while_the_directory_is_read_is_not_one(tmp_path, monkeypatch):
    """A file that is gone is not a checkpoint, and is no reason to fail the whole listing."""
    with writer(tmp_path) as run:
        save(run, 10)
    checkpoints = run_path(tmp_path, "test") / CHECKPOINTS_DIR
    # Held before it is replaced: run_store.os is the os module itself, so the replacement
    # would otherwise call itself.
    scandir = os.scandir
    monkeypatch.setattr(
        run_store.os,
        "scandir",
        lambda directory: [Vanished(checkpoint_file(20)), *scandir(directory)],
    )

    found = read_checkpoints(checkpoints)

    assert [info.step for info in found] == [10]


# Notes.


def test_notes_round_trip_through_the_run_directory(tmp_path):
    with writer(tmp_path):
        pass
    notes = RunNotes(
        title="Wider MLP", tags=["mlp", "width-2048"], notes="Trying width.\nIt helped."
    )

    assert save_notes(open_run(tmp_path, "test"), notes) == notes

    assert open_run(tmp_path, "test").notes == notes
    written = json.loads((run_path(tmp_path, "test") / NOTES_FILE).read_text())
    assert written["format_version"] == run_store.FORMAT_VERSION


def test_a_run_nobody_has_written_notes_for_has_empty_ones(tmp_path):
    with writer(tmp_path):
        pass

    assert open_run(tmp_path, "test").notes == RunNotes(title=None, tags=[], notes="")


def test_notes_are_replaced_whole(tmp_path):
    with writer(tmp_path):
        pass
    run = open_run(tmp_path, "test")
    save_notes(run, RunNotes(title="first", tags=["a"], notes="one"))

    save_notes(run, RunNotes(tags=["b"]))

    assert run.notes == RunNotes(tags=["b"])
    assert not list(run.directory.glob("*.writing"))


def test_notes_that_cannot_be_read_say_so(tmp_path):
    with writer(tmp_path):
        pass
    (run_path(tmp_path, "test") / NOTES_FILE).write_text("{not json")

    with pytest.raises(RunError, match="notes.json"):
        assert open_run(tmp_path, "test").notes


def test_notes_that_cannot_be_written_say_so(tmp_path):
    with writer(tmp_path):
        pass
    (run_path(tmp_path, "test") / NOTES_FILE).mkdir()

    with pytest.raises(RunError, match="cannot write"):
        save_notes(open_run(tmp_path, "test"), RunNotes(title="x"))


def test_a_title_is_trimmed_and_an_empty_one_is_no_title():
    assert RunNotes(title="  Wider MLP ").title == "Wider MLP"
    assert RunNotes(title="   ").title is None


def test_a_title_is_one_line():
    with pytest.raises(ValueError, match="one line"):
        RunNotes(title="two\nlines")


def test_tags_are_trimmed_and_kept_once_each_in_the_order_given():
    assert RunNotes(tags=[" mlp", "baseline", "mlp"]).tags == ["mlp", "baseline"]


@pytest.mark.parametrize("tag", ["", "two words", "a,b", "x" * 41])
def test_a_tag_is_one_short_word(tag):
    with pytest.raises(ValueError, match="tag"):
        RunNotes(tags=[tag])


def test_two_saves_at_once_leave_one_of_them_whole(tmp_path, monkeypatch):
    """Notes have more than one writer: the web server's threads and the command line.

    The second save happens while the first is between writing its temporary file and renaming
    it into place, which is where two writers sharing one temporary name would mix their text.
    """
    with writer(tmp_path):
        pass
    run = open_run(tmp_path, "test")
    first = RunNotes(title="the first save", notes="a much longer text than the second has " * 5)
    second = RunNotes(title="second")
    sync_file = run_store.sync_file
    interrupted = []

    def interleaved(f):
        sync_file(f)
        if not interrupted:
            interrupted.append(True)
            save_notes(run, second)

    monkeypatch.setattr(run_store, "sync_file", interleaved)

    save_notes(run, first)

    assert run.notes == first, "the last to be renamed into place wins, whole"
    assert not list(run.directory.glob("*.writing"))


def test_a_heartbeat_replaces_what_a_crash_mid_write_left_behind(tmp_path):
    """A file with one writer keeps one temporary name, which the next write takes over."""
    with writer(tmp_path) as run:
        (run.directory / f"{STATUS_FILE}.writing").write_text("half a heart")

        run.heartbeat(RunStatus.RUNNING, steps=100)

    assert not list(run.directory.glob("*.writing"))


def test_overwriting_throws_away_what_crashes_mid_write_left_behind(tmp_path):
    with writer(tmp_path):
        pass
    directory = run_path(tmp_path, "test")
    (directory / "notes.json.0123abcd.writing").write_text("{")
    (directory / CHECKPOINTS_DIR / "step-000000005.pt.writing").write_text("half the weights")

    with RunWriter.create(
        directory, info("test"), config_text="", policy=CheckpointPolicy(), overwrite=True
    ):
        pass

    assert not list(directory.rglob("*.writing"))


# One writer at a time, and taking a run up again.


def test_a_run_being_written_cannot_be_written_by_another(tmp_path):
    first = writer(tmp_path)

    with pytest.raises(RunError, match="'test' is being written by another process"):
        RunWriter.create(
            run_path(tmp_path, "test"),
            info(),
            config_text="",
            policy=CheckpointPolicy(),
            overwrite=True,
        )
    with pytest.raises(RunError, match="being written by another process"):
        RunWriter.reopen(run_path(tmp_path, "test"), policy=CheckpointPolicy())

    assert (run_path(tmp_path, "test") / CONFIG_FILE).read_text() == "# the config as written\n"
    first.close()


def test_a_run_is_free_to_write_again_once_its_writer_is_done(tmp_path):
    writer(tmp_path).close()

    with RunWriter.reopen(run_path(tmp_path, "test"), policy=CheckpointPolicy()) as again:
        again.log(step=1, split="train")

    assert steps(RunReader(run_path(tmp_path, "test")).metrics()) == [1]


def test_a_writer_that_died_leaves_no_lock_behind(tmp_path):
    """SIGKILL and the OOM killer give a trainer no chance to tidy up after itself."""
    writer(tmp_path).close()
    directory = run_path(tmp_path, "test")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from chess_ai.training.run_store import CheckpointPolicy, RunWriter\n"
            "RunWriter.reopen(Path(sys.argv[1]), policy=CheckpointPolicy())\n"
            "print('locked', flush=True)\n"
            "time.sleep(60)\n",
            str(directory),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(RunError, match="being written by another process"):
            RunWriter.reopen(directory, policy=CheckpointPolicy())
    finally:
        holder.kill()
        holder.wait()

    RunWriter.reopen(directory, policy=CheckpointPolicy()).close()


def test_a_process_forked_by_the_writer_does_not_keep_its_run_locked(tmp_path):
    """Data-loader workers are forked by the trainer, and can outlive it by a few seconds."""
    import multiprocessing

    first = writer(tmp_path)
    child = multiprocessing.get_context("fork").Process(target=time.sleep, args=(30,))
    child.start()
    try:
        first.close()
        RunWriter.reopen(run_path(tmp_path, "test"), policy=CheckpointPolicy()).close()
    finally:
        child.kill()
        child.join()


def test_reopening_a_directory_that_is_not_a_run_says_so(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(RunError, match="no run in"):
        RunWriter.reopen(tmp_path / "empty", policy=CheckpointPolicy())


def test_rewinding_drops_what_was_logged_after_the_step(tmp_path):
    with writer(tmp_path) as run:
        for step in (1, 2, 3, 4):
            run.log(step=step, split="train")
    directory = run_path(tmp_path, "test")
    with (directory / METRICS_FILE).open("a") as f:
        f.write('{"step": 5, "split": "tr')  # The line a crash cut off.

    with RunWriter.reopen(directory, policy=CheckpointPolicy()) as again:
        again.rewind(2)
        again.log(step=3, split="train", loss=1.0)

    assert steps(RunReader(directory).metrics()) == [1, 2, 3]
    assert (directory / METRICS_FILE).read_text().endswith('"loss": 1.0}\n'), "on a line of its own"


def test_a_tail_starts_again_when_the_run_is_rewound(tmp_path):
    with writer(tmp_path) as run:
        for step in (1, 2, 3):
            run.log(step=step, split="train")
    directory = run_path(tmp_path, "test")
    tail = RunReader(directory).tail_metrics()
    assert steps(tail.read()[0]) == [1, 2, 3]

    with RunWriter.reopen(directory, policy=CheckpointPolicy()) as again:
        again.rewind(1)
        again.log(step=2, split="train")

    records, replace = tail.read()
    assert replace, "the lines it had for steps 2 and 3 are gone"
    assert steps(records) == [1, 2]


def test_rewinding_to_where_the_log_ends_leaves_it_alone(tmp_path):
    """A run that stopped cleanly has nothing to drop, and readers need not start again."""
    with writer(tmp_path) as run:
        run.log(step=1, split="train")
        run.log(step=2, split="validation")
    path = run_path(tmp_path, "test") / METRICS_FILE
    before = os.stat(path).st_ino

    with RunWriter.reopen(run_path(tmp_path, "test"), policy=CheckpointPolicy()) as again:
        again.rewind(2)

    assert os.stat(path).st_ino == before


def test_rewinding_throws_away_a_half_written_checkpoint(tmp_path):
    with writer(tmp_path) as run:
        save(run, 2)
    checkpoints = run_path(tmp_path, "test") / CHECKPOINTS_DIR
    (checkpoints / f"{checkpoint_file(4)}.writing").write_text("half of it")

    with RunWriter.reopen(run_path(tmp_path, "test"), policy=CheckpointPolicy()) as again:
        again.rewind(2)

    assert sorted(path.name for path in checkpoints.iterdir()) == [
        CHECKPOINT_INDEX,
        checkpoint_file(2),
    ]


def test_a_run_says_whose_weights_it_started_from(tmp_path):
    lineage = Lineage(run="pretrained", step=3000, checkpoint="best", dataset="everything")
    writer(tmp_path, initialized_from=lineage).close()

    assert RunReader(run_path(tmp_path, "test")).info.initialized_from == lineage
