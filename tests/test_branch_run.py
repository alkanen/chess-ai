"""scripts/branch-run.py: a run copied at one of its checkpoints, with its decay changed."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from output_helpers import unheard
from training_helpers import dataset, experiment

from chess_ai.training import RunStatus, load_experiment, open_run, resume, train
from chess_ai.training import checkpoint as checkpoints
from chess_ai.training.run_store import RunError

SCRIPT = Path(__file__).parent.parent / "scripts" / "branch-run.py"
_spec = importlib.util.spec_from_file_location("branch_run", SCRIPT)
branch_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(branch_run)

SECTIONS = {
    "model": 'architecture = "mlp"\ndepth = 2\nwidth = 16',
    "schedule": "steps = 12\nwarmup_steps = 2",
    "validation": "every_steps = 4\npositions = 16",
    "checkpoints": "every_steps = 4\nkeep = 5",
}


@pytest.fixture
def source(tmp_path):
    """A finished run of 12 steps with checkpoints at 4, 8 and 12, and an evaluation of each."""
    return trained(tmp_path)


def trained(tmp_path, **sections):
    dataset(tmp_path / "data")
    path = experiment(tmp_path / "source.toml", **{**SECTIONS, **sections})
    train(
        load_experiment(path),
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        config_text=path.read_text(),
        say=unheard,
    )
    reader = open_run(tmp_path / "runs", "source")
    for step in (4, 8, 12):
        directory = reader.evaluation_directory(step)
        directory.mkdir(parents=True)
        result = {"model": {"run": "source", "step": step}}
        (directory / "probe-positions.json").write_text(json.dumps(result))
    return reader


def test_a_copy_has_the_run_up_to_the_step_with_the_change_made(tmp_path, source):
    branch_run.branch(
        tmp_path / "runs", "source", 8, "copy", {"optimizer.weight_decay_schedule": "learning-rate"}
    )

    copy = open_run(tmp_path / "runs", "copy")
    assert [info.step for info in copy.checkpoints()] == [4, 8]
    assert copy.status.status == RunStatus.STOPPED and copy.status.step == 8
    assert max(line["step"] for line in copy.metrics()) == 8
    payload = checkpoints.load(copy.checkpoint_path(copy.latest_checkpoint()))
    assert payload["run"] == "copy" and payload["config"]["name"] == "copy"
    assert payload["config"]["optimizer"]["weight_decay_schedule"] == "learning-rate"
    assert copy.info.name == "copy"
    assert copy.info.config["optimizer"]["weight_decay_schedule"] == "learning-rate"
    assert sorted(path.name for path in (copy.directory / "evaluations").iterdir()) == [
        "step-000000004",
        "step-000000008",
    ]
    probed = json.loads((copy.evaluation_directory(8) / "probe-positions.json").read_text())
    assert probed["model"]["run"] == "copy"
    assert "learning-rate" in copy.notes.title
    assert copy.config_text.startswith("# Copied from run source at step 8, with ")
    assert checkpoints.load(source.checkpoint_path(source.latest_checkpoint()))["run"] == "source"
    assert [info.step for info in source.checkpoints()] == [4, 8, 12], "the source is untouched"


def test_a_copy_resumes_with_its_change_and_can_stop_part_way(tmp_path, source):
    branch_run.branch(tmp_path / "runs", "source", 4, "copy", {"optimizer.weight_decay": 0.0})

    resume("copy", data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", say=unheard, stop_at=8)

    copy = open_run(tmp_path / "runs", "copy")
    assert copy.status.status == RunStatus.STOPPED and copy.status.step == 8
    resume("copy", data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", say=unheard)
    assert open_run(tmp_path / "runs", "copy").status.status == RunStatus.FINISHED
    ours = checkpoints.load(copy.checkpoint_path(copy.latest_checkpoint()))
    theirs = checkpoints.load(source.checkpoint_path(source.latest_checkpoint()))
    assert ours["step"] == theirs["step"] == 12
    assert any(
        not torch.equal(value, theirs["model_state"][name])
        for name, value in ours["model_state"].items()
    ), "trained without weight decay from step 4, so not the same weights"


def test_only_what_a_resumed_run_takes_from_its_config_can_be_changed(tmp_path, source):
    with pytest.raises(ValueError, match="cannot change optimizer.learning_rate"):
        branch_run.branch(tmp_path / "runs", "source", 8, "copy", {"optimizer.learning_rate": 0.1})
    with pytest.raises(ValueError, match="not one a run can have"):
        branch_run.branch(
            tmp_path / "runs", "source", 8, "copy", {"optimizer.weight_decay_schedule": "cosine"}
        )
    assert not (tmp_path / "runs" / "copy").exists()


def test_a_step_with_no_checkpoint_or_a_name_taken_is_refused(tmp_path, source):
    with pytest.raises(RunError, match="no checkpoint of step 6; it has 4, 8, 12"):
        branch_run.branch(tmp_path / "runs", "source", 6, "copy", {})
    with pytest.raises(RunError, match="there already"):
        branch_run.branch(tmp_path / "runs", "source", 8, "source", {})


def test_a_run_from_before_settings_existed_is_copied_with_the_behaviour_it_was_trained_with(
    tmp_path,
):
    """Its config has no ``decay_biases_and_norms`` and no ``compile``, which a resume reads as
    the behaviour it was trained with rather than as today's defaults: one decayed group, and
    not compiled. A copy has to say so, since it names every setting."""
    source = trained(tmp_path, optimizer="decay_biases_and_norms = true")
    for info in source.checkpoints():
        path = source.checkpoint_path(info)
        payload = checkpoints.load(path)
        del payload["config"]["optimizer"]["decay_biases_and_norms"]
        del payload["config"]["training"]["compile"]
        checkpoints.save(payload, path)

    branch_run.branch(tmp_path / "runs", "source", 4, "copy", {"optimizer.weight_decay": 0.0})

    copy = open_run(tmp_path / "runs", "copy")
    config = checkpoints.load(copy.checkpoint_path(copy.latest_checkpoint()))["config"]
    assert config["optimizer"]["decay_biases_and_norms"] is True
    assert config["training"]["compile"] is False
    assert copy.info.config["optimizer"]["decay_biases_and_norms"] is True
    resume("copy", data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", say=unheard, stop_at=8)
    assert open_run(tmp_path / "runs", "copy").latest_checkpoint().step == 8


def test_a_checkpoint_pruned_as_the_source_s_trainer_lets_go_is_not_copied(
    tmp_path, source, monkeypatch
):
    """The checkpoints are listed once the source is locked, so that a trainer still saving its
    last checkpoint, and pruning, cannot delete one of them from under the copy."""
    real = branch_run._WriterLock.acquire.__func__

    def pruned_then_acquired(cls, directory, name, **kwargs):
        # What a trainer finishing just then does: retention deletes a checkpoint, and then
        # the lock is let go of.
        (directory / "checkpoints" / "step-000000004.pt").unlink(missing_ok=True)
        return real(cls, directory, name, **kwargs)

    monkeypatch.setattr(branch_run._WriterLock, "acquire", classmethod(pruned_then_acquired))

    branch_run.branch(tmp_path / "runs", "source", 8, "copy", {})

    assert [info.step for info in open_run(tmp_path / "runs", "copy").checkpoints()] == [8]


def test_a_run_being_trained_is_refused_with_a_sentence(tmp_path, source, capsys, monkeypatch):
    lock = branch_run._WriterLock.acquire(source.directory, "source")
    try:
        monkeypatch.setenv("CHESS_AI_CONFIG", str(config_for(tmp_path)))
        assert branch_run.main(["source", "8", "copy"]) == 2
    finally:
        lock.release()

    assert "is being written by another process" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "copy").exists()


def test_a_lock_that_cannot_be_opened_says_why_rather_than_blaming_a_trainer(
    tmp_path, source, capsys, monkeypatch
):
    """A run copied off another machine can belong to another user, or be read-only."""
    lock = source.directory / "writer.lock"
    lock.chmod(0o000)
    try:
        monkeypatch.setenv("CHESS_AI_CONFIG", str(config_for(tmp_path)))
        assert branch_run.main(["source", "8", "copy"]) == 2
    finally:
        lock.chmod(0o644)

    err = capsys.readouterr().err
    assert "cannot open" in err and "Permission denied" in err
    assert "another process" not in err


def config_for(tmp_path) -> Path:
    path = tmp_path / "chess-ai.toml"
    path.write_text(f'[paths]\nruns = "{tmp_path / "runs"}"\ndata = "{tmp_path / "data"}"\n')
    return path


def test_settings_are_read_as_toml_values(tmp_path):
    assert branch_run._setting("optimizer.weight_decay=0") == ("optimizer.weight_decay", 0)
    assert branch_run._setting("optimizer.weight_decay_schedule=learning-rate") == (
        "optimizer.weight_decay_schedule",
        "learning-rate",
    )
