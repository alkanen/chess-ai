"""A whole training run, small enough to be a test: what it writes, and that it is repeatable."""

import contextlib
import copy
import dataclasses
import json
import math
import os
import pickle
import re
import signal
import tempfile
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import torch
from torch import nn
from training_helpers import dataset, experiment

from chess_ai.dataset import open_dataset
from chess_ai.encoders import create_encoder
from chess_ai.inference import load_engine
from chess_ai.models import create_model
from chess_ai.training import (
    RunStatus,
    TrainingError,
    TrainingStopped,
    load_experiment,
    open_run,
    resume,
    train,
    trainer,
)
from chess_ai.training import checkpoint as checkpoints
from chess_ai.training.experiment import OptimizerSection
from chess_ai.training.run_store import (
    CHECKPOINTS_DIR,
    CheckpointPolicy,
    RunError,
    RunWriter,
    checkpoint_file,
)


@pytest.fixture
def data_dir(tmp_path):
    """A dataset built from the fixture PGN files, with both splits worth training on."""
    dataset(tmp_path / "data")
    return tmp_path / "data"


def run(tmp_path, data_dir, name: str = "tiny", **sections):
    """Train a tiny run and return a reader over what it wrote."""
    config = load_experiment(experiment(tmp_path / f"{name}.toml", **sections))
    directory = train(
        config,
        data_dir=data_dir,
        runs_dir=tmp_path / "runs",
        config_text=(tmp_path / f"{name}.toml").read_text(),
        say=lambda line: None,
    )
    assert directory == tmp_path / "runs" / name
    return open_run(tmp_path / "runs", name)


def said_by(tmp_path, data_dir, name: str = "tiny", **sections) -> list[str]:
    """Train a tiny run and return the lines it printed."""
    lines: list[str] = []
    config = load_experiment(experiment(tmp_path / f"{name}.toml", **sections))
    train(
        config,
        data_dir=data_dir,
        runs_dir=tmp_path / "runs",
        config_text="",
        say=lines.append,
    )
    return lines


def test_a_run_writes_the_config_the_code_and_the_seed(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)

    info = reader.info
    assert info.name == "tiny"
    assert info.seed == 7
    assert info.code_version
    assert info.device.startswith("cpu")
    assert info.config["schedule"]["steps"] == 6, "the config with its defaults filled in"
    assert 'architecture = "mlp"' in reader.config_text, "and the file as it was written"


def test_a_run_records_what_it_trained_on(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)

    with open_dataset("test", data_dir=data_dir) as built:
        assert reader.info.dataset.name == "test"
        assert reader.info.dataset.games == built.manifest.games
        assert reader.info.dataset.train_positions == built.manifest.splits["train"].positions
    assert reader.info.encoder.encoder == "board-planes"
    assert reader.info.model.architecture == "mlp"
    assert reader.info.model.parameter_count > 0
    assert reader.info.positions_per_second_estimate > 0


def test_a_finished_run_says_it_finished(tmp_path, data_dir):
    status = run(tmp_path, data_dir).status

    assert status is not None
    assert status.status == RunStatus.FINISHED
    assert status.step == status.steps == 6
    assert status.epoch > 0
    assert status.positions_per_second > 0
    assert status.message is None


def test_training_metrics_are_logged_as_the_run_goes(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)

    training = [line for line in reader.metrics() if line["split"] == "train"]
    assert [line["step"] for line in training] == [2, 4, 6]
    for line in training:
        assert line["policy_loss"] > 0 and line["value_loss"] > 0
        assert line["loss"] == pytest.approx(line["policy_loss"] + 0.5 * line["value_loss"])
        assert line["positions_per_second"] > 0
        assert line["epoch"] > 0


def test_the_learning_rate_warms_up_then_decays(tmp_path, data_dir):
    reader = run(
        tmp_path,
        data_dir,
        schedule="steps = 8\nwarmup_steps = 4",
        training='device = "cpu"\nbatch_size = 8\ndata_workers = 0\nlog_every_steps = 1',
    )

    rates = [line["learning_rate"] for line in reader.metrics() if line["split"] == "train"]
    assert len(rates) == 8
    assert rates[:4] == sorted(rates[:4]), "warming up"
    assert rates[3] == max(rates)
    assert rates[4:] == sorted(rates[4:], reverse=True), "then decaying"
    assert rates[-1] < rates[0] * 2, "and ending near the floor rather than at the peak"


def test_validation_metrics_are_logged_and_include_the_illegal_move_rate(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)

    validation = [line for line in reader.metrics() if line["split"] == "validation"]
    assert [line["step"] for line in validation] == [3, 6]
    for line in validation:
        assert 0.0 <= line["top1"] <= 1.0
        assert line["top5"] >= line["top1"]
        assert 0.0 <= line["illegal_top_move_rate"] <= 1.0
        assert line["policy_loss"] > 0 and line["value_loss"] > 0
        assert line["positions"] > 0


def test_every_metrics_line_says_how_many_positions_and_seconds_in_it_is(tmp_path, data_dir):
    """What a chart puts runs side by side on, when their batch sizes or speeds differ."""
    reader = run(
        tmp_path,
        data_dir,
        training='device = "cpu"\nbatch_size = 8\ndata_workers = 0\nlog_every_steps = 1',
    )

    lines = reader.metrics()
    train_lines = {line["step"]: line for line in lines if line["split"] == "train"}
    seen = [train_lines[step]["positions_seen"] for step in sorted(train_lines)]
    assert seen == [8 * step for step in sorted(train_lines)]
    elapsed = [train_lines[step]["elapsed"] for step in sorted(train_lines)]
    assert elapsed == sorted(elapsed)
    validation = [line for line in lines if line["split"] == "validation"]
    assert validation, "the run validated"
    for line in validation:
        trained = train_lines[line["step"]]
        assert (line["positions_seen"], line["elapsed"]) == (
            trained["positions_seen"],
            trained["elapsed"],
        ), "placed where the step it measured was done"


def test_a_dataset_with_nothing_held_back_trains_without_validating(tmp_path, capsys):
    """An unsplit dataset is still trainable; it just cannot be measured."""
    dataset(tmp_path / "data", name="unsplit", validation_fraction=0.0)
    config = load_experiment(experiment(tmp_path / "tiny.toml", dataset_name="unsplit"))

    train(
        config,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        config_text="",
    )

    reader = open_run(tmp_path / "runs", "tiny")
    assert all(line["split"] == "train" for line in reader.metrics())
    assert reader.status.status == RunStatus.FINISHED
    assert "no validation positions" in capsys.readouterr().out


def test_checkpoints_are_saved_on_schedule_and_at_the_end(tmp_path, data_dir):
    """Every `every_steps`, and the last step whether or not it lands on one."""
    reader = run(
        tmp_path,
        data_dir,
        schedule="steps = 7\nwarmup_steps = 2",
        checkpoints="every_steps = 3\nkeep = 5",
    )

    assert [info.step for info in reader.checkpoints()] == [3, 6, 7]


def test_checkpoint_retention_keeps_the_most_recent_and_the_best(tmp_path, data_dir):
    reader = run(
        tmp_path,
        data_dir,
        schedule="steps = 9\nwarmup_steps = 2",
        validation="every_steps = 3\npositions = 16",
        checkpoints="every_steps = 3\nkeep = 1",
    )

    kept = [info.step for info in reader.checkpoints()]
    best = reader.best_checkpoint()
    assert kept[-1] == 9, "the newest is always kept"
    assert best is not None and best.step in kept
    assert len(kept) <= 2, f"one recent plus the best, not {kept}"


def test_a_checkpoint_holds_the_weights_the_optimizer_and_what_made_them(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)
    latest = reader.latest_checkpoint()

    payload = checkpoints.load(reader.checkpoint_path(latest))

    assert payload["step"] == 6
    assert payload["run"] == "tiny"
    assert payload["seed"] == 7
    assert payload["architecture"] == "mlp"
    assert payload["model_options"] == {"depth": 1, "width": 8}
    assert payload["config"]["dataset"]["name"] == "test"
    assert checkpoints.spec_of(payload) == reader.info.encoder
    assert payload["model_state"]["policy_head.bias"].shape == (reader.info.encoder.policy_size,)
    assert payload["optimizer_state"]["state"], "the optimizer's moments, so a resume is free"
    assert payload["metrics"]["policy_loss"] > 0


def test_a_checkpoint_that_is_not_one_is_refused(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)
    path = reader.checkpoint_path(reader.latest_checkpoint())
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(checkpoints.CheckpointError, match="cannot read checkpoint"):
        checkpoints.load(path)


def test_two_runs_with_the_same_seed_learn_the_same_thing(tmp_path, data_dir):
    """The whole point of recording the seed."""
    first = run(tmp_path, data_dir, name="first")
    second = run(tmp_path, data_dir, name="second")

    def losses(reader):
        return [line["loss"] for line in reader.metrics() if line["split"] == "train"]

    assert losses(first) == losses(second)
    weights = [
        checkpoints.load(reader.checkpoint_path(reader.latest_checkpoint()))["model_state"]
        for reader in (first, second)
    ]
    for name, value in weights[0].items():
        assert torch.equal(value, weights[1][name]), name


def test_a_different_seed_learns_something_different(tmp_path, data_dir):
    first = run(tmp_path, data_dir, name="first")
    second = run(tmp_path, data_dir, name="second", **{"": 'name = "second"\nseed = 99'})

    def losses(reader):
        return [line["loss"] for line in reader.metrics() if line["split"] == "train"]

    assert losses(first) != losses(second)


def test_the_run_is_left_alone_when_the_name_is_taken(tmp_path, data_dir):
    run(tmp_path, data_dir)

    with pytest.raises(TrainingError, match="already in .*choose another name"):
        run(tmp_path, data_dir)

    assert open_run(tmp_path / "runs", "tiny").status.status == RunStatus.FINISHED


def test_a_dataset_that_is_not_there_lists_the_ones_that_are(tmp_path, data_dir):
    config = load_experiment(experiment(tmp_path / "tiny.toml", dataset_name="nope"))

    with pytest.raises(TrainingError, match="no dataset in .*datasets in .*test"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")


def test_an_architecture_that_is_not_registered_says_what_there_is(tmp_path, data_dir):
    config = load_experiment(
        experiment(tmp_path / "tiny.toml", model='architecture = "transmogrifier"')
    )

    with pytest.raises(TrainingError, match="no architecture called 'transmogrifier'.*resnet"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")


def test_asking_for_cuda_where_there_is_none_says_so(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_experiment(
        experiment(tmp_path / "tiny.toml", training='device = "cuda"\nbatch_size = 8')
    )

    with pytest.raises(TrainingError, match="no CUDA device is available"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")


def test_a_run_that_crashes_stops_claiming_to_be_running(tmp_path, data_dir, monkeypatch):
    """A heartbeat left saying "running" would show up on the dashboard as a live run forever."""
    from chess_ai.training import trainer

    def explode(*args, **kwargs):
        raise RuntimeError("the GPU fell over")

    monkeypatch.setattr(trainer, "_step", explode)
    config = load_experiment(experiment(tmp_path / "tiny.toml"))

    with pytest.raises(RuntimeError, match="the GPU fell over"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")

    status = open_run(tmp_path / "runs", "tiny").status
    assert status.status == RunStatus.CRASHED
    assert "the GPU fell over" in status.message


def test_a_batch_bigger_than_the_split_still_trains(tmp_path, data_dir):
    """A dataset smaller than one batch is what a test dataset is, and it has to work."""
    reader = run(tmp_path, data_dir, training='device = "cpu"\nbatch_size = 4096\ndata_workers = 0')

    assert reader.status.status == RunStatus.FINISHED
    assert reader.metrics()


def test_data_workers_read_batches_in_their_own_processes(tmp_path, data_dir):
    reader = run(tmp_path, data_dir, training='device = "cpu"\nbatch_size = 8\ndata_workers = 2')

    assert reader.status.status == RunStatus.FINISHED
    assert [line["step"] for line in reader.metrics() if line["split"] == "train"] == [6]


def test_the_metrics_log_is_json_lines_one_object_at_a_time(tmp_path, data_dir):
    """What the web server tails, so the shape is part of the contract."""
    reader = run(tmp_path, data_dir)

    text = (reader.directory / "metrics.jsonl").read_text()
    assert text.endswith("\n")
    for line in text.splitlines():
        assert set(json.loads(line)) >= {"step", "epoch", "split"}


def test_nothing_is_left_half_written_in_the_run_directory(tmp_path, data_dir):
    reader = run(tmp_path, data_dir)

    assert not list(reader.directory.rglob("*.writing"))
    assert sorted(path.name for path in reader.directory.iterdir()) == [
        CHECKPOINTS_DIR,
        "config.toml",
        "metrics.jsonl",
        "run.json",
        "status.json",
        "writer.lock",
    ]
    assert (reader.directory / CHECKPOINTS_DIR / checkpoint_file(6)).is_file()


def test_an_interrupted_run_says_where_it_got_to(tmp_path, data_dir, monkeypatch):
    """Ctrl-C is not a crash, and a run that stops has to stop saying it is running."""
    from chess_ai.training import trainer

    real = trainer._step
    # The throughput probe takes real steps of its own before the run starts, and they come
    # through here too; the run's own first step is the one after them.
    before_the_run = trainer.PROBE_STEPS + 1
    calls = {"n": 0}

    def interrupt_after_two_steps(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > before_the_run + 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer, "_step", interrupt_after_two_steps)
    config = load_experiment(experiment(tmp_path / "tiny.toml"))

    with pytest.raises(KeyboardInterrupt):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")

    status = open_run(tmp_path / "runs", "tiny").status
    assert status.status == RunStatus.STOPPED
    assert status.step == 2, "the step it had finished, not zero"
    assert status.epoch > 0
    assert status.message == "interrupted before a checkpoint could be saved"


def test_the_probe_does_not_leave_the_dataset_open_for_the_loader(tmp_path, data_dir, monkeypatch):
    """The object the loader is handed has to still be unopened.

    The throughput probe takes a real batch in the training process, which maps shards. On a
    spawn or forkserver start method the DataLoader pickles the dataset into every worker, and
    a cached `np.memmap` pickles as the whole mapped file — so a probe that left the shards
    open would copy the dataset into each worker at loader startup, every run.
    """
    from chess_ai.training import trainer

    real = trainer.batch_loader
    sizes = []

    def measure(batches, **options):
        sizes.append(len(pickle.dumps(batches)))
        return real(batches, **options)

    monkeypatch.setattr(trainer, "batch_loader", measure)

    run(tmp_path, data_dir)

    shards = (data_dir / "datasets" / "test" / "train" / "positions").iterdir()
    mapped = sum(shard.stat().st_size for shard in shards)
    assert sizes, "the loader was never built"
    assert max(sizes) < mapped, f"the loader was handed {max(sizes)} bytes, shards are {mapped}"


def test_the_heartbeat_keeps_up_between_log_lines(tmp_path, data_dir, monkeypatch):
    """Its cadence has to be HEARTBEAT_SECONDS, not log_every_steps.

    A reader cannot tell a run that stopped reporting from one that is merely between log
    lines, so a heartbeat no more frequent than the logging makes any staleness window
    declare a healthy run dead.
    """
    from chess_ai.training import trainer

    monkeypatch.setattr(trainer, "HEARTBEAT_SECONDS", 0.0)
    beats = []
    real = RunWriter.heartbeat

    def record(self, status, **fields):
        beats.append(fields.get("step"))
        return real(self, status, **fields)

    monkeypatch.setattr(RunWriter, "heartbeat", record)

    reader = run(
        tmp_path,
        data_dir,
        training='device = "cpu"\nbatch_size = 8\ndata_workers = 0\nlog_every_steps = 1000',
    )

    logged = [line["step"] for line in reader.metrics() if line["split"] == "train"]
    assert logged == [6], "one log line, at the last step"
    assert sorted(set(beats)) == [0, 1, 2, 3, 4, 5, 6], f"but a heartbeat every step: {beats}"


def test_a_checkpoint_is_indexed_with_metrics_from_its_own_step(tmp_path, data_dir):
    """Schedules that do not divide each other must not cross-contaminate.

    Otherwise a checkpoint carries an earlier step's numbers, two checkpoints can be indexed
    with the same value, and the tie-break hands `best_step` to one that was never measured.
    """
    reader = run(
        tmp_path,
        data_dir,
        schedule="steps = 6\nwarmup_steps = 1",
        validation="every_steps = 2\npositions = 16",
        checkpoints="every_steps = 3\nkeep = 5",
    )

    measured = {
        line["step"]: line["policy_loss"]
        for line in reader.metrics()
        if line["split"] == "validation"
    }
    saved = reader.checkpoints()
    assert [info.step for info in saved] == [3, 6]
    for info in saved:
        assert info.step in measured, f"checkpoint {info.step} was never validated"
        assert info.metrics["policy_loss"] == pytest.approx(measured[info.step])


def test_the_logged_learning_rate_is_the_one_the_step_was_taken_with(tmp_path, data_dir):
    """The scheduler steps before the log, so reading it there is one step ahead."""
    reader = run(
        tmp_path,
        data_dir,
        schedule="steps = 8\nwarmup_steps = 4",
        optimizer="learning_rate = 0.001",
        training='device = "cpu"\nbatch_size = 8\ndata_workers = 0\nlog_every_steps = 1',
    )

    rates = [line["learning_rate"] for line in reader.metrics() if line["split"] == "train"]
    assert len(rates) == 8
    assert rates[0] == pytest.approx(0.001 / 4), "the first step warms up from a quarter"
    assert rates[3] == pytest.approx(0.001), "and the full rate is reached at the end of warmup"


def test_a_checkpoint_that_cannot_be_written_is_reported_rather_than_traced(
    tmp_path, data_dir, monkeypatch
):
    """A disk filling up an hour into a run is an ordinary accident, not a traceback."""

    def no_space(self, step, write, *, metrics=None):
        raise RunError("cannot save checkpoint: No space left on device")

    monkeypatch.setattr(RunWriter, "save_checkpoint", no_space)
    config = load_experiment(experiment(tmp_path / "tiny.toml"))

    with pytest.raises(TrainingError, match="No space left on device"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")

    status = open_run(tmp_path / "runs", "tiny").status
    assert status.status == RunStatus.CRASHED, "and the run still says what became of it"


def test_a_finished_run_is_not_called_crashed_over_the_closing_summary(
    tmp_path, data_dir, monkeypatch
):
    """Reading back which checkpoint was best is cosmetic; the run is already written."""
    from chess_ai.training import run_store

    def unreadable(self):
        raise RunError("checkpoints/index.json does not say what it should")

    monkeypatch.setattr(run_store.RunReader, "best_checkpoint", unreadable)

    reader = run(tmp_path, data_dir)

    assert reader.status.status == RunStatus.FINISHED
    assert reader.checkpoints(), "and the checkpoints it wrote are still there"


def test_a_checkpoint_schedule_that_forces_extra_validation_says_so(tmp_path, data_dir):
    """Checkpoint steps validate too, so the run has to say when that costs extra passes.

    Otherwise `[validation] every_steps` reads as the knob that decides how often validation
    runs, while the checkpoint schedule quietly overrides it and the run is slower for no
    visible reason.
    """
    lines = said_by(
        tmp_path,
        data_dir,
        schedule="steps = 6\nwarmup_steps = 1",
        validation="every_steps = 6\npositions = 16",
        checkpoints="every_steps = 2\nkeep = 5",
    )

    note = [line for line in lines if "extra validation" in line]
    assert note, lines
    # Checkpoints at 2, 4 and 6 against validation at 6: steps 2 and 4 validate as well.
    assert "2 extra validation" in note[0], note[0]


def test_a_schedule_that_needs_no_extra_validation_says_nothing(tmp_path, data_dir):
    lines = said_by(
        tmp_path,
        data_dir,
        schedule="steps = 6\nwarmup_steps = 1",
        validation="every_steps = 3\npositions = 16",
        checkpoints="every_steps = 3\nkeep = 5",
    )

    assert not [line for line in lines if "extra validation" in line], lines


def test_a_dataset_with_no_validation_split_is_not_told_about_extra_passes(tmp_path):
    """The note exists to make a real cost visible; a run that never validates pays none.

    `_summary` runs before the loop and sees only the two schedules, while whether validation
    happens at all is decided later from the manifest — so the note could contradict the
    warning printed two lines after it.
    """
    dataset(tmp_path / "data", name="unsplit", validation_fraction=0.0)

    lines = said_by(
        tmp_path,
        tmp_path / "data",
        dataset_name="unsplit",
        schedule="steps = 6\nwarmup_steps = 1",
        validation="every_steps = 6\npositions = 16",
        checkpoints="every_steps = 2\nkeep = 5",
    )

    assert not [line for line in lines if "extra validation" in line], lines
    assert [line for line in lines if "no validation positions" in line], "and it says why"


def test_a_null_best_metric_does_not_traceback_after_the_run_finished(
    tmp_path, data_dir, monkeypatch
):
    """A diverged metric reaches the closing summary as None, which cannot be formatted.

    The guard used to ask whether the metric was *present*, which was enough while the only
    two states were absent and a float. `null` is a third, and it went straight through into an
    f-string — after the run store had already recorded FINISHED.
    """
    from chess_ai.training import run_store

    def diverged(self):
        latest = self.checkpoints()[-1]
        return latest.model_copy(update={"metrics": {"policy_loss": None}})

    monkeypatch.setattr(run_store.RunReader, "best_checkpoint", diverged)

    reader = run(tmp_path, data_dir)

    assert reader.status.status == RunStatus.FINISHED
    assert reader.checkpoints(), "and the checkpoints it wrote are still there"


@pytest.mark.parametrize(
    ("steps", "checkpoints", "validation"),
    [(10, c, v) for c in (1, 2, 3, 4, 5, 7, 10, 11) for v in (1, 2, 3, 4, 5, 7, 10, 11)]
    + [(60, c, v) for c in (7, 12, 15, 60, 61) for v in (5, 9, 12, 60, 61)]
    + [(1, 1, 1), (1, 2, 3), (997, 13, 17)],
)
def test_extra_validation_passes_are_counted_exactly(steps, checkpoints, validation):
    """The closed form has to agree with counting the steps out one by one.

    Counting them out is what the summary used to do, which allocated two sets proportional to
    the length of the run just to print one line — before the dataset had been opened.
    """
    from chess_ai.training import trainer

    forced = set(range(checkpoints, steps + 1, checkpoints)) - set(
        range(validation, steps + 1, validation)
    )
    expected = len(forced - {steps})

    config = load_experiment(
        experiment(
            Path(tempfile.mkdtemp()) / "c.toml",
            schedule=f"steps = {steps}\nwarmup_steps = 0",
            validation=f"every_steps = {validation}\npositions = 16",
            checkpoints=f"every_steps = {checkpoints}\nkeep = 1",
        )
    )

    assert trainer._extra_validations(config) == expected


def test_counting_extra_validations_does_not_scale_with_the_run(tmp_path):
    """A ten-million-step schedule checkpointing every step must not build a set of them.

    `checkpoints.every_steps = 1` is an ordinary thing to set while babysitting a short debug
    run and forget to put back, and this is computed before `_prepare` has had the chance to
    fail on the dataset.
    """
    from chess_ai.training import trainer

    config = load_experiment(
        experiment(
            tmp_path / "huge.toml",
            schedule="steps = 10000000\nwarmup_steps = 0",
            validation="every_steps = 1000\npositions = 16",
            checkpoints="every_steps = 1\nkeep = 1",
        )
    )

    tracemalloc.start()
    try:
        answer = trainer._extra_validations(config)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # Every step checkpoints; one in a thousand is a validation step anyway; the last step is
    # both, so it is not extra.
    assert answer == 10_000_000 - 10_000
    assert peak < 1_000_000, f"counting allocated {peak / 1e6:.0f} MB to print one line"


def test_a_run_with_history_and_side_to_move_orientation_trains_validates_and_plays(
    tmp_path, data_dir
):
    reader = run(tmp_path, data_dir, encoder='history = 2\norientation = "side-to-move"')

    assert reader.info.encoder.spatial_channels == 36
    assert reader.info.encoder.options == {
        "rating_scale": 5000.0,
        "history": 2,
        "orientation": "side-to-move",
    }
    validation = [row for row in reader.metrics() if row["split"] == "validation"]
    assert validation and all(0.0 <= row["illegal_top_move_rate"] <= 1.0 for row in validation)
    engine = load_engine(reader.checkpoint_path(reader.latest_checkpoint()))
    assert engine.spec == reader.info.encoder
    board = chess.Board()
    board.push_san("e4")
    assert engine.evaluate(board).best in board.legal_moves


@pytest.mark.parametrize("globals", ["planes", "film"])
def test_a_resnet_trains_validates_and_plays_with_nothing_else_changed(tmp_path, data_dir, globals):
    """The architecture is a config line: the trainer, the checkpoints and the engine take it."""
    reader = run(
        tmp_path,
        data_dir,
        model=f'architecture = "resnet"\nblocks = 1\nchannels = 4\nglobals = "{globals}"',
    )

    assert reader.info.model.architecture == "resnet"
    assert reader.info.model.options == {"blocks": 1, "channels": 4, "globals": globals}
    validation = [row for row in reader.metrics() if row["split"] == "validation"]
    assert validation and all(0.0 <= row["top1"] <= 1.0 for row in validation)
    engine = load_engine(reader.checkpoint_path(reader.latest_checkpoint()))
    board = chess.Board()
    board.push_san("e4")
    assert engine.evaluate(board).best in board.legal_moves


def train_lines(reader) -> list[dict]:
    return [line for line in reader.metrics() if line["split"] == "train"]


def test_training_lines_say_how_large_the_gradient_was_and_how_often_it_was_clipped(
    tmp_path, data_dir
):
    lines = train_lines(run(tmp_path, data_dir))

    assert lines
    for line in lines:
        assert math.isfinite(line["gradient_norm"]) and line["gradient_norm"] > 0
        assert 0.0 <= line["clipped_fraction"] <= 1.0


@pytest.mark.parametrize(("clip", "clipped"), [("0", 0.0), ("1e-9", 1.0)])
def test_the_clipped_fraction_follows_the_threshold(tmp_path, data_dir, clip, clipped):
    """Off clips nothing and still measures the norm; a threshold nothing fits under clips all."""
    lines = train_lines(run(tmp_path, data_dir, optimizer=f"gradient_clip = {clip}"))

    assert lines
    assert all(line["clipped_fraction"] == clipped for line in lines)
    assert all(line["gradient_norm"] > 0 for line in lines)


def test_a_step_measures_and_clips_the_gradient_as_clip_grad_norm_does(tmp_path, data_dir):
    """The norm logged is the one clipping acted on, and the weights move as they always did."""
    config = load_experiment(experiment(tmp_path / "tiny.toml", optimizer="gradient_clip = 0.05"))
    prepared = trainer._prepare(config, data_dir=data_dir)
    batch = prepared.batches[0].to(prepared.device)
    twin = dataclasses.replace(prepared, model=copy.deepcopy(prepared.model))
    twin_optimizer = trainer._optimizer(twin)
    total, *_ = trainer._losses(twin, batch)
    total.backward()
    expected = nn.utils.clip_grad_norm_(twin.model.parameters(), 0.05)
    twin_optimizer.step()

    measured = trainer._step(prepared, batch, trainer._optimizer(prepared))

    assert expected > 0.05, "the test needs a gradient the threshold clips"
    torch.testing.assert_close(measured.gradient_norm, expected)
    assert measured.clipped.item() == 1
    for name, value in prepared.model.state_dict().items():
        torch.testing.assert_close(value, twin.model.state_dict()[name])


DECAY_VARIANTS = [
    pytest.param("mlp", {"depth": 2, "width": 8}, id="mlp"),
    pytest.param("resnet", {"blocks": 1, "channels": 4, "globals": "planes"}, id="resnet-planes"),
    pytest.param("resnet", {"blocks": 1, "channels": 4, "globals": "film"}, id="resnet-film"),
]


def biases_and_norms(model: nn.Module) -> set[int]:
    """The parameters that are a bias, or belong to a normalization layer, by identity."""
    found = {id(p) for name, p in model.named_parameters() if name.endswith("bias")}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._NormBase | nn.LayerNorm):
            found |= {id(p) for p in module.parameters()}
    return found


@pytest.mark.parametrize(("architecture", "options"), DECAY_VARIANTS)
def test_weight_decay_shrinks_weights_and_leaves_biases_and_norms_alone(architecture, options):
    model = create_model(architecture, create_encoder("board-planes").spec, **options)

    decayed, undecayed = trainer._parameter_groups(model, OptimizerSection(weight_decay=0.02))

    assert decayed["weight_decay"] == 0.02 and undecayed["weight_decay"] == 0.0
    grouped = [id(p) for group in (decayed, undecayed) for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters()), "each exactly once"
    assert {id(p) for p in undecayed["params"]} == biases_and_norms(model)


@pytest.mark.parametrize(("architecture", "options"), DECAY_VARIANTS)
def test_decaying_everything_is_still_there_for_the_runs_made_that_way(architecture, options):
    model = create_model(architecture, create_encoder("board-planes").spec, **options)

    (group,) = trainer._parameter_groups(
        model, OptimizerSection(weight_decay=0.02, decay_biases_and_norms=True)
    )

    assert group["weight_decay"] == 0.02
    assert [id(p) for p in group["params"]] == [id(p) for p in model.parameters()]


def test_the_summary_says_what_weight_decay_reaches(tmp_path, data_dir):
    lines = said_by(tmp_path, data_dir, model='architecture = "mlp"\ndepth = 1\nwidth = 8')

    # Weights (12 * 64 + 11) * 8 + 8 * 1968 + 8 * 3, and a bias for each of the 8 + 1968 + 3
    # units, which is all this MLP has.
    (line,) = [line for line in lines if line.startswith("  optimizer")]
    assert line == (
        "  optimizer  AdamW, weight decay 0.01 on 22,000 parameters, "
        "none on 1,979 biases and norms, clip 1"
    )


def test_the_summary_says_when_there_is_no_weight_decay_or_clipping(tmp_path, data_dir):
    lines = said_by(tmp_path, data_dir, optimizer="weight_decay = 0\ngradient_clip = 0")

    assert "  optimizer  AdamW, no weight decay, no clipping" in lines


# Stopping, resuming, and starting from another run's weights.

DROPOUT = 'architecture = "mlp"\ndepth = 2\nwidth = 16\ndropout = 0.2'
"""A model that draws random numbers at every step, so that a resume has to restore them."""

REAL_STEP = trainer._step
"""For putting the step back once a test has stopped a run with it, and wants to resume."""

RESUMABLE = {
    "model": DROPOUT,
    "schedule": "steps = 9\nwarmup_steps = 2",
    "validation": "every_steps = 4\npositions = 16",
    "checkpoints": "every_steps = 4\nkeep = 5",
}


def after_steps(monkeypatch, steps: int, then) -> None:
    """Call ``then`` once the run's own step number ``steps`` is done, and never again."""
    real = trainer._step
    # The throughput probe takes steps of its own first, and they come through here too.
    calls = {"n": -(trainer.PROBE_STEPS + 1)}

    def counted(*args, **kwargs):
        result = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == steps:
            then()
        return result

    monkeypatch.setattr(trainer, "_step", counted)


def signal_after(monkeypatch, steps: int, number: int = signal.SIGTERM) -> None:
    after_steps(monkeypatch, steps, lambda: os.kill(os.getpid(), number))


def resumed(tmp_path, data_dir, name: str, say=lambda line: None):
    resume(name, data_dir=data_dir, runs_dir=tmp_path / "runs", say=say)
    return open_run(tmp_path / "runs", name)


def final_payload(reader) -> dict:
    return checkpoints.load(reader.checkpoint_path(reader.latest_checkpoint()))


def assert_same_state(first: dict, second: dict) -> None:
    for name, value in first["model_state"].items():
        assert torch.equal(value, second["model_state"][name]), name
    moments = [
        (first["optimizer_state"]["state"][key], second["optimizer_state"]["state"][key])
        for key in first["optimizer_state"]["state"]
    ]
    for ours, theirs in moments:
        for name, value in ours.items():
            assert torch.equal(value, theirs[name]), name


def lines_of(reader, split: str) -> list[dict]:
    return [line for line in reader.metrics() if line["split"] == split]


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
def test_a_stop_signal_saves_a_checkpoint_and_says_the_run_stopped(
    tmp_path, data_dir, monkeypatch, number
):
    signal_after(monkeypatch, 5, number)
    said: list[str] = []
    config = load_experiment(experiment(tmp_path / "tiny.toml", **RESUMABLE))

    with pytest.raises(TrainingStopped) as stopped:
        train(
            config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="", say=said.append
        )

    assert (stopped.value.step, stopped.value.signal) == (5, number)
    reader = open_run(tmp_path / "runs", "tiny")
    status = reader.status
    assert status.status == RunStatus.STOPPED
    assert status.step == 5
    assert status.message == f"stopped by {signal.Signals(number).name}"
    assert [info.step for info in reader.checkpoints()] == [4, 5], "saved at the step it stopped"
    assert final_payload(reader)["step"] == 5
    assert any("saving a checkpoint and stopping" in line for line in said), said


def test_a_second_sigterm_still_lets_the_run_save_its_checkpoint(tmp_path, data_dir, monkeypatch):
    """SIGTERM comes from machines, and often twice.

    ``uv run`` forwards a SIGTERM to the trainer, so one sent to the process group — as a
    service manager stopping the run does — arrives twice. Taking the second for impatience
    would throw away the checkpoint the first asked for.
    """

    def twice():
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)

    after_steps(monkeypatch, 5, twice)

    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)

    reader = open_run(tmp_path / "runs", "tiny")
    assert reader.status.message == "stopped by SIGTERM"
    assert reader.latest_checkpoint().step == 5


@pytest.mark.parametrize("first", [signal.SIGINT, signal.SIGTERM])
def test_a_second_ctrl_c_stops_the_run_at_once(tmp_path, data_dir, monkeypatch, first):
    """Somebody who presses it again does not want to wait for the checkpoint."""

    def impatiently():
        os.kill(os.getpid(), first)
        os.kill(os.getpid(), signal.SIGINT)

    # During the sixth step, with the last checkpoint at the fourth.
    after_steps(monkeypatch, 6, impatiently)

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, data_dir, **RESUMABLE)

    reader = open_run(tmp_path / "runs", "tiny")
    assert reader.status.status == RunStatus.STOPPED
    assert reader.status.message == "interrupted before a checkpoint could be saved"
    assert reader.latest_checkpoint().step == 4


def test_a_second_ctrl_c_just_after_a_checkpoint_is_a_stop_at_that_checkpoint(
    tmp_path, data_dir, monkeypatch
):
    """Nothing the run had done was lost, so it says it stopped, and where to carry on from."""

    def impatiently():
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGINT)

    # During the fifth step, which has not counted yet, just after the checkpoint at the fourth.
    after_steps(monkeypatch, 5, impatiently)

    with pytest.raises(TrainingStopped) as stopped:
        run(tmp_path, data_dir, **RESUMABLE)

    assert stopped.value.step == 4
    reader = open_run(tmp_path / "runs", "tiny")
    assert reader.status.message == "stopped by SIGTERM"
    assert reader.latest_checkpoint().step == 4


def test_a_second_ctrl_c_once_the_checkpoint_is_saved_still_says_the_run_stopped_with_it(
    tmp_path, data_dir, monkeypatch
):
    """Interrupting what is left after the stop checkpoint costs nothing, and must not say so."""
    signal_after(monkeypatch, 5, signal.SIGINT)
    real = RunWriter.save_checkpoint

    def impatient(self, step, write, **options):
        saved = real(self, step, write, **options)
        if step == 5:
            os.kill(os.getpid(), signal.SIGINT)
        return saved

    monkeypatch.setattr(RunWriter, "save_checkpoint", impatient)

    with pytest.raises(TrainingStopped) as stopped:
        run(tmp_path, data_dir, **RESUMABLE)

    assert (stopped.value.step, stopped.value.signal) == (5, signal.SIGINT)
    reader = open_run(tmp_path / "runs", "tiny")
    assert reader.status.message == "stopped by SIGINT"
    assert reader.latest_checkpoint().step == 5


def test_a_run_trains_off_the_main_thread_without_stop_signals(tmp_path, data_dir):
    """As a run started from a web request would be: there is no ctrl-c to answer there."""
    import threading

    failures = []

    def training():
        try:
            run(tmp_path, data_dir, training='device = "cpu"\nbatch_size = 8\ndata_workers = 2')
        except Exception as e:  # noqa: BLE001 - reported to the test's own thread
            failures.append(e)

    thread = threading.Thread(target=training)
    thread.start()
    thread.join()

    assert failures == []
    assert open_run(tmp_path / "runs", "tiny").status.status == RunStatus.FINISHED


def test_ctrl_c_twice_at_the_last_step_still_finishes_the_run(tmp_path, data_dir, monkeypatch):
    """A stop asked for at the last step is a run that finished, and a second one after its
    last checkpoint, while it lets go of its loader, interrupts nothing that was left to do."""
    signal_after(monkeypatch, 9, signal.SIGINT)
    real = RunWriter.save_checkpoint

    def impatient(self, step, write, **options):
        saved = real(self, step, write, **options)
        if step == 9:
            os.kill(os.getpid(), signal.SIGINT)
        return saved

    monkeypatch.setattr(RunWriter, "save_checkpoint", impatient)

    reader = run(tmp_path, data_dir, **RESUMABLE)

    assert reader.status.status == RunStatus.FINISHED
    assert reader.status.step == 9


def test_ctrl_c_after_the_run_has_said_it_finished_leaves_it_finished(
    tmp_path, data_dir, monkeypatch
):
    signal_after(monkeypatch, 9, signal.SIGINT)
    real = RunWriter.finish

    def impatient(self, status, **fields):
        real(self, status, **fields)
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(RunWriter, "finish", impatient)

    reader = run(tmp_path, data_dir, **RESUMABLE)

    assert reader.status.status == RunStatus.FINISHED
    assert reader.status.positions_per_second > 0, "the heartbeat finish wrote, not another"


@pytest.mark.parametrize(("first_at", "step", "status"), [(9, 9, "finished"), (5, 5, "stopped")])
def test_a_ctrl_c_while_a_checkpoint_is_indexed_waits_until_it_is(
    tmp_path, data_dir, monkeypatch, first_at, step, status
):
    """The file lands before the index records it, prunes and picks the best; a checkpoint the
    index never heard of has no metrics, so it could never be the run's best."""
    signal_after(monkeypatch, first_at, signal.SIGTERM)
    real = RunWriter._reindex

    def impatient(self, saved, metrics):
        if saved == step:
            os.kill(os.getpid(), signal.SIGINT)
        real(self, saved, metrics)

    monkeypatch.setattr(RunWriter, "_reindex", impatient)

    with contextlib.suppress(TrainingStopped, KeyboardInterrupt):
        run(tmp_path, data_dir, **RESUMABLE)

    reader = open_run(tmp_path / "runs", "tiny")
    index = json.loads((reader.directory / CHECKPOINTS_DIR / "index.json").read_text())
    assert step in [entry["step"] for entry in index["checkpoints"]], "the index has it"
    assert reader.status.status == status


def test_the_handlers_there_were_are_put_back_after_a_run(tmp_path, data_dir):
    before = {number: signal.getsignal(number) for number in trainer.STOP_SIGNALS}

    run(tmp_path, data_dir)

    assert {number: signal.getsignal(number) for number in trainer.STOP_SIGNALS} == before


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [
        (0, "0:00:00"),
        (59.4, "0:00:59"),
        (61, "0:01:01"),
        (11 * 3600 + 7 * 60 + 9, "11:07:09"),
        (24 * 3600 - 1, "23:59:59"),
        (24 * 3600, "1d 00:00:00"),
        (26 * 3600 + 3 * 60 + 9, "1d 02:03:09"),
        (12 * 86400 + 5, "12d 00:00:05"),
    ],
)
def test_a_validation_line_shows_durations_to_the_second(seconds, shown):
    assert trainer._clock(seconds) == shown


def clocked(monkeypatch, seconds_per_step: float) -> dict[str, float]:
    """Fake the trainer's clock so that every step it takes costs ``seconds_per_step``.

    Returns the clock, so that a test can move it on between runs as well.
    """
    clock = {"now": 1000.0}
    monkeypatch.setattr(trainer, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
    real = trainer._step

    def slow(*args, **kwargs):
        clock["now"] += seconds_per_step
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer, "_step", slow)
    return clock


def timings(lines: list[str]) -> list[tuple[str, str, str]]:
    """Each validation line's step, running time and time left."""
    pattern = re.compile(r"^  step (\d+)/9  .*  running (\S+)  left (\S+)$")
    return [match.groups() for line in lines if (match := pattern.match(line))]


def test_a_validation_line_says_how_long_the_run_has_been_going_and_has_left(
    tmp_path, data_dir, monkeypatch
):
    clocked(monkeypatch, 1000)
    said: list[str] = []
    config = load_experiment(experiment(tmp_path / "tiny.toml", **RESUMABLE))

    train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="", say=said.append)

    # Validated at 4, at 8 and at the end, 1000 seconds a step and nothing else costing any.
    assert timings(said) == [
        ("4", "1:06:40", "1:23:20"),
        ("8", "2:13:20", "0:16:40"),
        ("9", "2:30:00", "0:00:00"),
    ]


def test_a_resumed_run_times_itself_from_the_resume(tmp_path, data_dir, monkeypatch):
    """Neither the break before the resume nor the speed before it says anything about now."""
    signal_after(monkeypatch, 1)
    clock = clocked(monkeypatch, 1000)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, name="halves", **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    clock_after_the_break = clock["now"] + 3 * 86400
    clock = clocked(monkeypatch, 3000)
    clock["now"] = clock_after_the_break
    said: list[str] = []

    halves = resumed(tmp_path, data_dir, "halves", say=said.append)

    # From step 1 at 3000 seconds a step: 7 steps by step 8, and one left. Step 4 is too few
    # steps in to estimate from.
    assert timings(said) == [
        ("4", "2:30:00", "?"),
        ("8", "5:50:00", "0:50:00"),
        ("9", "6:40:00", "0:00:00"),
    ]
    # The run's own clock still carries on from where it stopped, breaks left out.
    assert lines_of(halves, "validation")[-1]["elapsed"] == 1 * 1000 + 8 * 3000


def test_a_resume_just_before_a_validation_gives_no_estimate_from_one_step(
    tmp_path, data_dir, monkeypatch
):
    """One step, carrying a validation and the session's start-up, is no rate to go on.

    A stop lands at any step, so one in every ``every_steps`` resumes validates after its first
    step, and dividing all that by 1 promised weeks for a run with hours left.
    """
    signal_after(monkeypatch, 3)
    clocked(monkeypatch, 1000)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, name="halves", **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    clock = clocked(monkeypatch, 1000)
    real_validate = trainer._validate

    def slow_validate(*args, **kwargs):
        clock["now"] += 500
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(trainer, "_validate", slow_validate)
    said: list[str] = []

    resumed(tmp_path, data_dir, "halves", say=said.append)

    # From step 3: step 4 is one step in. Step 8 is five, two validations among them.
    assert timings(said) == [
        ("4", "0:25:00", "?"),
        ("8", "1:40:00", "0:20:00"),
        ("9", "2:05:00", "0:00:00"),
    ]


def test_a_run_stopped_and_resumed_learns_what_it_would_have_in_one_go(
    tmp_path, data_dir, monkeypatch
):
    """Weights, optimizer, schedule, data order and dropout's random draws all carry on."""
    whole = run(tmp_path, data_dir, name="whole", **RESUMABLE)
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, name="halves", **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)

    halves = resumed(tmp_path, data_dir, "halves")

    assert halves.status.status == RunStatus.FINISHED
    assert halves.status.step == 9
    assert_same_state(final_payload(whole), final_payload(halves))
    assert [
        (line["step"], line["policy_loss"], line["top1"]) for line in lines_of(halves, "validation")
    ] == [
        (line["step"], line["policy_loss"], line["top1"]) for line in lines_of(whole, "validation")
    ]
    # The first window after the resume covers fewer steps; the ones after it are the same.
    for ours, theirs in zip(lines_of(halves, "train"), lines_of(whole, "train"), strict=True):
        assert ours["step"] == theirs["step"]
        assert ours["positions_seen"] == theirs["positions_seen"]
        assert ours["learning_rate"] == theirs["learning_rate"]
        if ours["step"] > 6:
            assert ours["loss"] == theirs["loss"], ours["step"]


def test_a_crashed_run_resumes_from_its_last_checkpoint_and_forgets_what_came_after(
    tmp_path, data_dir, monkeypatch
):
    whole = run(tmp_path, data_dir, name="whole", **RESUMABLE)

    def crash():
        raise RuntimeError("the GPU fell over")

    after_steps(monkeypatch, 7, crash)
    with pytest.raises(RuntimeError, match="fell over"):
        run(tmp_path, data_dir, name="crashed", **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    crashed = open_run(tmp_path / "runs", "crashed")
    assert crashed.status.status == RunStatus.CRASHED
    assert max(line["step"] for line in crashed.metrics()) == 6, "it logged past its checkpoint"

    crashed = resumed(tmp_path, data_dir, "crashed")

    assert crashed.status.status == RunStatus.FINISHED
    assert_same_state(final_payload(whole), final_payload(crashed))
    logged = [(line["step"], line["split"]) for line in crashed.metrics()]
    assert logged == [(line["step"], line["split"]) for line in whole.metrics()], (
        "each step logged once, in order"
    )
    elapsed = [line["elapsed"] for line in crashed.metrics()]
    assert elapsed == sorted(elapsed), "and the clock carries on rather than starting again"


def test_a_resumed_run_says_where_it_carries_on_from(tmp_path, data_dir, monkeypatch):
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    said: list[str] = []

    resumed(tmp_path, data_dir, "tiny", say=said.append)

    assert said[0] == "chess-ai: run tiny, seed 7, resuming at step 5", said
    assert not [line for line in said if "predates" in line], said


def test_a_run_can_be_stopped_and_resumed_more_than_once(tmp_path, data_dir, monkeypatch):
    whole = run(tmp_path, data_dir, name="whole", **RESUMABLE)
    signal_after(monkeypatch, 2)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, name="thirds", **RESUMABLE)
    signal_after(monkeypatch, 3)  # Three steps into the resumed run, so at step 5.
    with pytest.raises(TrainingStopped) as stopped:
        resumed(tmp_path, data_dir, "thirds")
    assert stopped.value.step == 5
    monkeypatch.setattr(trainer, "_step", REAL_STEP)

    thirds = resumed(tmp_path, data_dir, "thirds")

    assert_same_state(final_payload(whole), final_payload(thirds))


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_workers_leave_stopping_to_the_trainer(tmp_path, data_dir, monkeypatch, number):
    """Ctrl-C reaches every process in the terminal's group, the loader's workers included,
    and a service manager stopping a run sends SIGTERM to all of its processes.

    A worker that died of it would make the loader raise in the trainer, at whatever it was
    doing — which may be saving the very checkpoint the signal asked for.
    """
    import multiprocessing
    import subprocess

    def signal_everyone():
        # From a process of its own, as a terminal or a service manager would: a worker
        # signalled by its own parent is let off by torch, which would hide what this is about.
        workers = [str(child.pid) for child in multiprocessing.active_children()]
        assert workers, "the loader's workers are running"
        subprocess.run(["kill", "-s", signal.Signals(number).name[3:], *workers], check=True)
        os.kill(os.getpid(), number)
        # As long as saving a large checkpoint takes, which is when the loader would hear of a
        # worker that died, and raise.
        time.sleep(0.5)

    after_steps(monkeypatch, 3, signal_everyone)
    workers = 'device = "cpu"\nbatch_size = 8\ndata_workers = 2\nlog_every_steps = 2'

    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, training=workers, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)

    reader = open_run(tmp_path / "runs", "tiny")
    assert reader.status.status == RunStatus.STOPPED
    assert reader.latest_checkpoint().step == 3
    assert resumed(tmp_path, data_dir, "tiny").status.status == RunStatus.FINISHED


def test_a_finished_run_is_not_resumed(tmp_path, data_dir):
    run(tmp_path, data_dir)

    with pytest.raises(TrainingError, match="already finished all 6 steps"):
        resumed(tmp_path, data_dir, "tiny")


def test_a_run_with_no_checkpoint_cannot_be_resumed(tmp_path, data_dir, monkeypatch):
    def crash():
        raise RuntimeError("the GPU fell over")

    after_steps(monkeypatch, 1, crash)
    with pytest.raises(RuntimeError):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)

    with pytest.raises(TrainingError, match="saved no checkpoint.*--overwrite"):
        resumed(tmp_path, data_dir, "tiny")


def test_a_run_that_is_not_there_cannot_be_resumed(tmp_path, data_dir):
    with pytest.raises(TrainingError, match="no run called 'nope'"):
        resumed(tmp_path, data_dir, "nope")


def test_a_run_still_being_written_is_not_resumed_under_it(tmp_path, data_dir, monkeypatch):
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    reader = open_run(tmp_path / "runs", "tiny")
    before = reader.metrics()

    with (
        RunWriter.reopen(reader.directory, policy=CheckpointPolicy()),
        pytest.raises(TrainingError, match="being written by another process"),
    ):
        resumed(tmp_path, data_dir, "tiny")

    assert reader.metrics() == before
    assert reader.status.status == RunStatus.STOPPED


def test_a_run_whose_dataset_was_rebuilt_is_not_resumed(tmp_path, data_dir, monkeypatch):
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    dataset(data_dir, validation_fraction=0.5, overwrite=True)

    with pytest.raises(
        TrainingError, match=r"not the one run 'tiny' was training on.*initialize_from"
    ):
        resumed(tmp_path, data_dir, "tiny")

    assert open_run(tmp_path / "runs", "tiny").status.status == RunStatus.STOPPED


def grown(data_dir) -> None:
    """Append a version to the dataset "test": the lichess fixture's games a second time."""
    from dataset_helpers import fixture

    from chess_ai.dataset.builder import append_dataset

    append_dataset("test", [fixture("lichess.pgn")], data_dir=data_dir, allow_repeat=True)


def test_a_run_records_the_version_of_the_dataset_it_trained_on(tmp_path, data_dir):
    first = open_dataset("test", data_dir=data_dir).manifest
    grown(data_dir)

    latest = run(tmp_path, data_dir, name="latest")
    pinned = run(tmp_path, data_dir, name="pinned", dataset='name = "test"\nversion = 1')

    assert latest.info.dataset.version == 2
    assert latest.info.dataset.games == first.games + 4
    assert pinned.info.dataset.version == 1
    assert pinned.info.dataset.train_positions == first.splits["train"].positions
    assert pinned.info.dataset.validation_positions == first.splits["validation"].positions


def test_a_run_says_which_version_it_trains_on(tmp_path, data_dir):
    grown(data_dir)

    said = said_by(tmp_path, data_dir, dataset='name = "test"\nversion = 1')

    assert any("dataset    test v1 of 2:" in line for line in said), said


def test_a_run_on_a_version_the_dataset_does_not_have_is_refused(tmp_path, data_dir):
    with pytest.raises(TrainingError, match="no version 2"):
        run(tmp_path, data_dir, dataset='name = "test"\nversion = 2')


def test_a_resumed_run_keeps_the_version_it_started_on(tmp_path, data_dir, monkeypatch):
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    grown(data_dir)

    reader = resumed(tmp_path, data_dir, "tiny")

    assert reader.status.status == RunStatus.FINISHED
    assert reader.info.dataset.version == 1


def test_a_run_from_before_dataset_versions_resumes_on_version_1(tmp_path, data_dir, monkeypatch):
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    path = tmp_path / "runs" / "tiny" / "run.json"
    written = json.loads(path.read_text())
    del written["dataset"]["version"]
    path.write_text(json.dumps(written))
    grown(data_dir)

    assert resumed(tmp_path, data_dir, "tiny").status.status == RunStatus.FINISHED


def test_a_run_from_before_biases_went_undecayed_resumes_with_everything_decayed(
    tmp_path, data_dir, monkeypatch
):
    """Its config does not mention the setting, and its optimizer state has one group."""
    sections = {**RESUMABLE, "optimizer": "decay_biases_and_norms = true"}
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **sections)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    reader = open_run(tmp_path / "runs", "tiny")
    path = reader.checkpoint_path(reader.latest_checkpoint())
    payload = checkpoints.load(path)
    del payload["config"]["optimizer"]["decay_biases_and_norms"]
    checkpoints.save(payload, path)

    assert resumed(tmp_path, data_dir, "tiny").status.status == RunStatus.FINISHED


def test_a_checkpoint_from_before_the_random_state_was_saved_still_resumes(
    tmp_path, data_dir, monkeypatch
):
    signal_after(monkeypatch, 5)
    with pytest.raises(TrainingStopped):
        run(tmp_path, data_dir, **RESUMABLE)
    monkeypatch.setattr(trainer, "_step", REAL_STEP)
    reader = open_run(tmp_path / "runs", "tiny")
    path = reader.checkpoint_path(reader.latest_checkpoint())
    payload = checkpoints.load(path)
    del payload["rng_state"], payload["elapsed"], payload["move_vocabulary"]
    checkpoints.save(payload, path)
    said: list[str] = []

    reader = resumed(tmp_path, data_dir, "tiny", say=said.append)

    assert reader.status.status == RunStatus.FINISHED
    assert [line for line in said if "predates saving the random state" in line], said


def test_a_checkpoint_from_another_move_vocabulary_is_refused(tmp_path, data_dir):
    """A reordered vocabulary has the same size, and would load and play nonsense."""
    reader = run(tmp_path, data_dir)
    path = reader.checkpoint_path(reader.latest_checkpoint())
    payload = checkpoints.load(path)
    payload["move_vocabulary"] = "0123456789abcdef"
    checkpoints.save(payload, path)

    with pytest.raises(checkpoints.CheckpointError, match="another move vocabulary"):
        checkpoints.load(path)


def test_a_new_run_starts_from_another_runs_weights(tmp_path, data_dir):
    source = run(tmp_path, data_dir, name="pretrained")
    config = load_experiment(
        experiment(
            tmp_path / "tuned.toml",
            initialize_from='run = "pretrained"\ncheckpoint = "latest"',
        )
    )
    prepared = trainer._prepare(config, data_dir=data_dir)

    lineage = trainer._initialize(prepared, runs_dir=tmp_path / "runs")

    assert lineage.run == "pretrained" and lineage.step == 6 and lineage.dataset == "test v1"
    for name, value in final_payload(source)["model_state"].items():
        assert torch.equal(value, prepared.model.state_dict()[name]), name


def test_a_fine_tuning_run_records_where_its_weights_came_from(tmp_path, data_dir):
    run(tmp_path, data_dir, name="pretrained")
    from_seed = run(tmp_path, data_dir, name="from-seed")
    said: list[str] = []
    config = load_experiment(
        experiment(tmp_path / "tuned.toml", initialize_from='run = "pretrained"\ncheckpoint = 3')
    )

    train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="", say=said.append)

    tuned = open_run(tmp_path / "runs", "tuned")
    assert tuned.info.initialized_from.model_dump() == {
        "run": "pretrained",
        "step": 3,
        "checkpoint": 3,
        "dataset": "test v1",
    }
    assert "  weights    from run pretrained at step 3, trained on test v1" in said
    assert final_payload(tuned)["step"] == 6, "with a step count of its own"
    assert [line["loss"] for line in lines_of(tuned, "train")] != [
        line["loss"] for line in lines_of(from_seed, "train")
    ], "and it trained from those weights rather than the seed's"


@pytest.mark.parametrize(
    ("sections", "complaint"),
    [
        (
            {"model": 'architecture = "resnet"\nblocks = 1\nchannels = 4'},
            "it is a 'mlp' and this run trains a 'resnet'",
        ),
        ({"encoder": "history = 1"}, "trained on board-planes: spatial 12x8x8.*spatial 24x8x8"),
        ({"encoder": "rating_scale = 3000.0"}, "rating_scale=5000.0.*rating_scale=3000.0"),
        ({"model": 'architecture = "mlp"\ndepth = 1\nwidth = 16'}, "do not fit this model"),
    ],
)
def test_weights_from_another_architecture_or_input_are_refused_with_a_reason(
    tmp_path, data_dir, sections, complaint
):
    run(tmp_path, data_dir, name="pretrained")
    config = load_experiment(
        experiment(tmp_path / "tuned.toml", initialize_from='run = "pretrained"', **sections)
    )

    with pytest.raises(
        TrainingError, match=f"cannot initialize from run 'pretrained'.*{complaint}"
    ):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")

    assert not (tmp_path / "runs" / "tuned").exists(), "refused before anything was written"


def test_dropout_may_change_when_fine_tuning(tmp_path, data_dir):
    """Options that leave the weights' shapes alone are the fine-tuning run's own business."""
    run(tmp_path, data_dir, name="pretrained", model=DROPOUT)
    reader = run(
        tmp_path,
        data_dir,
        name="tuned",
        model='architecture = "mlp"\ndepth = 2\nwidth = 16\ndropout = 0.5',
        initialize_from='run = "pretrained"',
    )

    assert reader.info.initialized_from.run == "pretrained"


def test_initializing_from_a_run_that_is_not_there_says_so(tmp_path, data_dir):
    config = load_experiment(experiment(tmp_path / "tuned.toml", initialize_from='run = "nope"'))

    with pytest.raises(TrainingError, match="cannot initialize from run 'nope': no run called"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")


def test_initializing_from_a_step_that_was_not_kept_says_so(tmp_path, data_dir):
    run(tmp_path, data_dir, name="pretrained")
    config = load_experiment(
        experiment(tmp_path / "tuned.toml", initialize_from='run = "pretrained"\ncheckpoint = 1')
    )

    with pytest.raises(TrainingError, match="no checkpoint from step 1"):
        train(config, data_dir=data_dir, runs_dir=tmp_path / "runs", config_text="")


def test_a_run_cannot_be_initialized_from_itself(tmp_path, data_dir):
    run(tmp_path, data_dir)
    config = load_experiment(experiment(tmp_path / "tiny.toml", initialize_from='run = "tiny"'))

    with pytest.raises(TrainingError, match="cannot be initialized from itself"):
        train(
            config,
            data_dir=data_dir,
            runs_dir=tmp_path / "runs",
            config_text="",
            overwrite=True,
        )
