"""A whole training run, small enough to be a test: what it writes, and that it is repeatable."""

import copy
import dataclasses
import json
import math
import pickle
import tempfile
import tracemalloc
from pathlib import Path

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
    load_experiment,
    open_run,
    train,
    trainer,
)
from chess_ai.training import checkpoint as checkpoints
from chess_ai.training.experiment import OptimizerSection
from chess_ai.training.run_store import CHECKPOINTS_DIR, RunError, RunWriter, checkpoint_file


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
    assert status.message == "interrupted"


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
