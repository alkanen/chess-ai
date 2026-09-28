"""The run directory: what it writes, what it keeps, and what a reader sees mid-write."""

import json
from datetime import UTC, datetime

import pytest

from chess_ai.encoders import create_encoder
from chess_ai.training.run_store import (
    CHECKPOINT_INDEX,
    CHECKPOINTS_DIR,
    METRICS_FILE,
    STATUS_FILE,
    CheckpointPolicy,
    DatasetReference,
    ModelReference,
    RunError,
    RunInfo,
    RunReader,
    RunStatus,
    RunWriter,
    checkpoint_file,
    code_version,
    list_runs,
    open_run,
    run_path,
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
