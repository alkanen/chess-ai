import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

import pytest
import torch
from training_helpers import add_checkpoint, model_run

from chess_ai.cli import main
from chess_ai.evaluator import (
    Evaluator,
    EvaluatorBusy,
    ProbeSuite,
    Stopped,
    configured_suites,
    evaluator_lock,
)
from chess_ai.inference import load_engine
from chess_ai.probes import SUITE, ProbeSetVersion, load_probe_set, read_current_set, read_probes
from chess_ai.sample_games import SUITE as SAMPLE_GAMES
from chess_ai.training.run_store import (
    ARCHIVED_TAG,
    CheckpointPolicy,
    RunNotes,
    RunReader,
    RunWriter,
    save_notes,
)


def evaluator(runs_dir, *, probes=None, said=None, warned=None) -> Evaluator:
    suite = ProbeSuite(probes or load_probe_set("standard"), rating=1500)
    return Evaluator(
        runs_dir,
        {suite.name: suite},
        load=load_engine,
        say=(said.append if said is not None else lambda text: None),
        warn=(warned.append if warned is not None else lambda text: None),
    )


def probed_steps(run: RunReader) -> list[int]:
    """The steps with a probe result, whatever other suites have results about them."""
    return [entry.step for entry in run.evaluations() if entry.suite == SUITE]


def done(jobs) -> list[tuple[str, int]]:
    return [(job.run.name, job.checkpoint.step) for job in jobs]


def small_set(tmp_path, *, version: int):
    path = tmp_path / f"small-v{version}.toml"
    path.write_text(
        f'name = "small"\nversion = {version}\n\n'
        '[[probes]]\nid = "start"\nname = "Start"\ncategory = "opening"\n'
    )
    return load_probe_set(path)


def test_a_new_checkpoint_is_picked_up_exactly_once(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4))
    worker = evaluator(runs)

    first = worker.run_once()
    second = worker.run_once()
    add_checkpoint(run, 6)
    third = worker.run_once()
    fourth = worker.run_once()

    # The newest first, so that a run being trained hears about its latest checkpoint soonest.
    assert done(first) == [("tiny", 4), ("tiny", 2)]
    assert second == []
    assert done(third) == [("tiny", 6)]
    assert fourth == []
    assert [(entry.step, entry.suite) for entry in RunReader(run).evaluations()] == [
        (2, SUITE),
        (4, SUITE),
        (6, SUITE),
    ]


def test_a_new_evaluator_finds_the_results_an_earlier_one_saved(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs)
    evaluator(runs).run_once()

    assert evaluator(runs).run_once() == []


def test_results_round_trip_through_the_run_store(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    said: list[str] = []

    evaluator(runs, said=said).run_once()
    result = read_probes(run, 2)

    assert result is not None
    assert (result.model.run, result.model.checkpoint, result.model.rating) == ("tiny", 2, 1500)
    assert result.checkpoint_written == run.checkpoint_written(run.checkpoints()[0])
    assert (result.probe_set, result.probe_set_version) == ("standard", 1)
    assert len(result.positions) == len(load_probe_set("standard").probes)
    assert result.started <= result.finished
    # What the file says is what reading it back gives, however it was written.
    assert json.loads(run.evaluation(2, SUITE)) == json.loads(result.model_dump_json())
    assert said == [
        f"tiny@2 {SUITE}: solved {result.solved()[0]} of {result.solved()[1]} probes (standard v1)"
    ]


def test_a_new_version_of_the_probe_set_probes_every_checkpoint_on_disk_again(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4))
    evaluator(runs, probes=small_set(tmp_path, version=1)).run_once()
    add_checkpoint(run, 6, keep=2)  # Prunes the checkpoint from step 2.

    again = evaluator(runs, probes=small_set(tmp_path, version=2)).run_once()

    assert done(again) == [("tiny", 6), ("tiny", 4)]
    reader = RunReader(run)
    assert [read_probes(reader, step).probe_set_version for step in (2, 4, 6)] == [1, 2, 2]


def test_a_checkpoint_saved_again_under_its_step_is_probed_again(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2,))
    worker = evaluator(runs)
    worker.run_once()
    checkpoint = run / "checkpoints" / "step-000000002.pt"
    later = checkpoint.stat().st_mtime + 10
    os.utime(checkpoint, (later, later))  # As a checkpoint saved again under its step is.

    assert done(worker.run_once()) == [("tiny", 2)]


def test_a_checkpoint_found_before_its_index_entry_is_not_probed_again_once_it_has_one(tmp_path):
    """A trainer renames the file into place, then indexes it, with its own time, a moment later."""
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2,))
    checkpoints = run / "checkpoints"
    shutil.copy(checkpoints / "step-000000002.pt", checkpoints / "step-000000004.pt")
    worker = evaluator(runs)
    assert done(worker.run_once()) == [("tiny", 4), ("tiny", 2)]

    index = json.loads((checkpoints / "index.json").read_text())
    index["checkpoints"].append(
        {
            **index["checkpoints"][0],
            "step": 4,
            "file": "step-000000004.pt",
            "created": datetime.now(UTC).isoformat(),
        }
    )
    (checkpoints / "index.json").write_text(json.dumps(index))

    assert RunReader(run).checkpoints()[1].file == "step-000000004.pt"
    assert worker.run_once() == []


def test_an_archived_run_and_a_run_that_asks_for_nothing_are_left_alone(tmp_path):
    runs = tmp_path / "runs"
    archived = model_run(runs, "archived")
    save_notes(RunReader(archived), RunNotes(tags=[ARCHIVED_TAG]))
    model_run(runs, "nothing", config={"evaluation": {"suites": []}})
    model_run(runs, "old")

    assert {name for name, _ in done(evaluator(runs).run_once())} == {"old"}


def test_suites_this_code_does_not_know_are_passed_over(tmp_path):
    run = RunReader(
        model_run(tmp_path, config={"evaluation": {"suites": ["probe-positions", "tournament"]}})
    )

    assert configured_suites(run) == ["probe-positions"]


def test_a_checkpoint_that_cannot_be_loaded_is_reported_once_and_not_tried_again(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4))
    (run / "checkpoints" / "step-000000004.pt").write_bytes(b"not a checkpoint")
    warned: list[str] = []
    worker = evaluator(runs, warned=warned)

    first = worker.run_once()
    second = worker.run_once()

    assert done(first) == [("tiny", 4), ("tiny", 2)]
    assert second == []
    assert len(warned) == 1 and warned[0].startswith("tiny@4 cannot be loaded")
    assert read_probes(RunReader(run), 4) is None


def test_a_checkpoint_pruned_before_its_turn_is_passed_over_without_a_word(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4))
    warned: list[str] = []
    worker = evaluator(runs, warned=warned)
    jobs = worker.pending()
    (run / "checkpoints" / "step-000000002.pt").unlink()

    for job in jobs:
        worker.evaluate(job)

    assert warned == []
    assert [entry.step for entry in RunReader(run).evaluations()] == [4]


def test_the_evaluator_stops_waiting_when_told_to(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs)
    stop = threading.Event()
    worker = evaluator(runs)
    thread = threading.Thread(target=worker.run, kwargs={"poll_seconds": 60, "stop": stop})

    thread.start()
    deadline = time.monotonic() + 30
    while len(RunReader(runs / "tiny").evaluations()) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(RunReader(runs / "tiny").evaluations()) == 2


def test_only_one_evaluator_works_on_a_runs_directory_at_a_time(tmp_path):
    with evaluator_lock(tmp_path), pytest.raises(EvaluatorBusy), evaluator_lock(tmp_path):
        pass

    with evaluator_lock(tmp_path):
        pass  # Let go of, it can be held again.


def test_a_run_started_again_under_its_name_loses_the_old_runs_evaluations(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs)
    evaluator(runs).run_once()
    info = RunReader(run).info

    RunWriter.create(run, info, config_text="", policy=CheckpointPolicy(), overwrite=True).close()

    assert RunReader(run).evaluations() == []
    assert not (run / "evaluations").exists()


def test_evaluate_runs_the_suites_again_and_replaces_earlier_results(tmp_path, capsys):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2, 4)))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')
    assert main(["evaluator", "--once"]) == 0
    before = read_probes(run, 4)
    untouched = read_probes(run, 2)

    assert main(["evaluate", "tiny@latest"]) == 0

    after = read_probes(run, 4)
    assert after.finished > before.finished
    assert read_probes(run, 2) == untouched
    out = capsys.readouterr().out
    assert any(line.startswith(f"tiny@4 {SUITE}: solved") for line in out.splitlines())


def test_evaluate_takes_every_checkpoint_of_a_run_and_the_suites_asked_for(tmp_path, capsys):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2, 4), config={"evaluation": {"suites": []}}))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')

    assert main(["evaluate", "tiny", "--suite", "probe-positions"]) == 0

    assert [entry.step for entry in run.evaluations()] == [2, 4]


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        (["evaluate", "tiny"], "run 'tiny' names no evaluation suites; say which with --suite"),
        (["evaluate", "tiny@9"], "run 'tiny' has no checkpoint from step 9"),
        (["evaluate", "missing"], "missing"),
        (["evaluate", "tiny", "--suite", "puzzles"], "there is no suite 'puzzles'"),
        (["evaluate", "tiny@soon"], "a checkpoint"),
    ],
)
def test_evaluate_refuses_what_it_cannot_do(tmp_path, capsys, arguments, complaint):
    runs = tmp_path / "runs"
    model_run(runs, config={"evaluation": {"suites": []}})
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')

    with pytest.raises(SystemExit) as exited:
        main(arguments)

    assert exited.value.code == 2
    assert complaint in capsys.readouterr().err


def test_the_evaluator_records_which_probe_set_it_probes_with(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    small_set(tmp_path, version=3)
    (tmp_path / "chess-ai.toml").write_text(
        f'[paths]\nruns = "{runs}"\n[evaluator]\nprobe_set = "{tmp_path / "small-v3.toml"}"\n'
    )
    assert read_current_set(runs) is None

    assert main(["evaluator", "--once"]) == 0

    assert read_current_set(runs) == ProbeSetVersion(name="small", version=3)


def test_evaluating_on_demand_leaves_the_record_of_the_evaluators_set_alone(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')
    assert main(["evaluator", "--once"]) == 0
    small_set(tmp_path, version=3)
    (tmp_path / "chess-ai.toml").write_text(
        f'[paths]\nruns = "{runs}"\n[evaluator]\nprobe_set = "{tmp_path / "small-v3.toml"}"\n'
    )

    assert main(["evaluate", "tiny"]) == 0

    assert read_current_set(runs) == ProbeSetVersion(name="standard", version=1)


def test_the_evaluator_refuses_to_start_beside_another(tmp_path, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')

    with evaluator_lock(runs), pytest.raises(SystemExit):
        main(["evaluator", "--once"])

    assert "another evaluator is already working on" in capsys.readouterr().err


def test_the_evaluator_command_evaluates_until_sigterm_and_then_stops(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    environment = {**os.environ, "CHESS_AI_PATHS_RUNS": str(runs)}
    environment["CHESS_AI_EVALUATOR_POLL_SECONDS"] = "0.2"
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, chess_ai.cli; sys.exit(chess_ai.cli.main())",
            "evaluator",
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not probed_steps(run) and time.monotonic() < deadline:
            time.sleep(0.1)
        add_checkpoint(run.directory, 4)
        while len(probed_steps(run)) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        worker.send_signal(signal.SIGTERM)
        out, err = worker.communicate(timeout=30)
    finally:
        worker.kill()

    assert worker.returncode == 0, err
    assert probed_steps(run) == [2, 4]
    assert f"tiny@4 {SUITE}" in out
    assert "evaluator stopped" in err


def test_a_run_overwritten_while_a_checkpoint_is_probed_does_not_get_the_old_runs_result(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2,))
    info = RunReader(run).info
    suite = ProbeSuite(load_probe_set("standard"), rating=1500)

    def load_then_overwrite(path):
        engine = load_engine(path)
        # chess-ai train --overwrite, between the checkpoint being loaded and its result saved.
        RunWriter.create(
            run, info, config_text="", policy=CheckpointPolicy(), overwrite=True
        ).close()
        return engine

    worker = Evaluator(runs, {suite.name: suite}, load=load_then_overwrite, warn=pytest.fail)
    worker.run_once()

    assert RunReader(run).evaluations() == []


def test_evaluate_does_not_save_a_result_for_a_checkpoint_replaced_while_it_was_probed(
    tmp_path, monkeypatch, capsys
):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2,))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')
    checkpoint = run / "checkpoints" / "step-000000002.pt"

    def load_then_replace(path):
        engine = load_engine(path)
        later = checkpoint.stat().st_mtime + 10
        os.utime(checkpoint, (later, later))
        return engine

    monkeypatch.setattr("chess_ai.cli._evaluation_loader", lambda config: load_then_replace)

    assert main(["evaluate", "tiny"]) == 0

    assert RunReader(run).evaluations() == []
    assert "tiny@2 changed while it was evaluated, so its result was not saved" in (
        capsys.readouterr().err
    )


def test_a_checkpoint_saved_during_a_backlog_is_evaluated_before_the_rest_of_it(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4, 6))
    order: list[str] = []
    stop = threading.Event()

    def heard(text: str) -> None:
        order.append(text.split()[0])
        if len(order) == 1:
            add_checkpoint(run, 8)
        if len(order) == 4:
            stop.set()

    suite = ProbeSuite(load_probe_set("standard"), rating=1500)
    worker = Evaluator(runs, {suite.name: suite}, load=load_engine, say=heard)

    finishes(lambda: worker.run(poll_seconds=60, stop=stop))

    assert order == ["tiny@6", "tiny@8", "tiny@4", "tiny@2"]


def test_once_evaluates_what_lacked_a_result_when_it_began_and_no_more(tmp_path):
    """So that it ends, however fast the runs save checkpoints meanwhile."""
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4, 6))
    order: list[str] = []

    def heard(text: str) -> None:
        order.append(text.split()[0])
        if len(order) == 1:
            add_checkpoint(run, 8, keep=10)
            (run / "checkpoints" / "step-000000004.pt").unlink()  # Pruned before its turn.

    suite = ProbeSuite(load_probe_set("standard"), rating=1500)
    worker = Evaluator(runs, {suite.name: suite}, load=load_engine, say=heard)

    jobs = finishes(worker.run_once)

    assert order == ["tiny@6", "tiny@2"]
    assert done(jobs) == [("tiny", 6), ("tiny", 2)]
    assert done(worker.pending()) == [("tiny", 8)]


def test_once_stops_between_checkpoints_when_told_to(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2, 4, 6))
    stop = threading.Event()
    suite = ProbeSuite(load_probe_set("standard"), rating=1500)
    worker = Evaluator(runs, {suite.name: suite}, load=load_engine, say=lambda text: stop.set())

    stopped = finishes(lambda: pytest.raises(Stopped, worker.run_once, stop=stop))

    assert done(stopped.value.done) == [("tiny", 6)]
    assert stopped.value.left == 2


def test_once_told_to_stop_after_its_last_checkpoint_has_finished(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    stop = threading.Event()
    suite = ProbeSuite(load_probe_set("standard"), rating=1500)
    worker = Evaluator(runs, {suite.name: suite}, load=load_engine, say=lambda text: stop.set())

    assert done(finishes(lambda: worker.run_once(stop=stop))) == [("tiny", 2)]


@pytest.mark.parametrize(("number", "status"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_evaluator_once_stopped_part_way_says_so_in_its_exit_status(
    tmp_path, monkeypatch, capsys, number, status
):
    """As ``timeout 10m chess-ai evaluator --once && publish`` needs it to."""
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4, 6))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')

    def load_and_get_stopped(path):
        os.kill(os.getpid(), number)  # Handled at once, on this, the main thread.
        return load_engine(path)

    monkeypatch.setattr("chess_ai.cli._evaluation_loader", lambda config: load_and_get_stopped)

    assert main(["evaluator", "--once"]) == status

    # The probes of the checkpoint it was loading are done, its games are not: they are
    # stopped as soon as they look, so the checkpoint is left with the two it never reached.
    assert probed_steps(RunReader(run)) == [6]
    err = capsys.readouterr().err
    assert "evaluator stopped before every checkpoint was evaluated: 3 are left" in err
    # The handlers it put in place are gone with it.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_evaluate_passes_over_a_checkpoint_pruned_while_the_others_were_evaluated(
    tmp_path, monkeypatch, capsys
):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')

    def load_then_prune(path):
        engine = load_engine(path)
        (run / "checkpoints" / "step-000000004.pt").unlink(missing_ok=True)
        return engine

    monkeypatch.setattr("chess_ai.cli._evaluation_loader", lambda config: load_then_prune)

    assert main(["evaluate", "tiny"]) == 0

    assert probed_steps(RunReader(run)) == [2]
    assert "tiny@4 was pruned before it could be evaluated" in capsys.readouterr().err


def finishes(call, timeout: float = 60.0):
    """``call()``'s result, failing the test rather than hanging if it does not return."""
    results: list = []
    thread = threading.Thread(target=lambda: results.append(call()), daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "did not return"
    return results[0]


class NeverStands:
    """A suite that saves nothing that reads back, as a result written with nulls in it does."""

    name = SUITE

    def __init__(self) -> None:
        self.evaluated = 0

    def stands(self, run, checkpoint, written) -> bool:
        return False

    def evaluate(self, run, checkpoint, written, engine, *, stop=None) -> str:
        self.evaluated += 1
        return "done"


def test_a_result_that_does_not_stand_once_saved_is_reported_and_not_tried_again(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2, 4))
    suite = NeverStands()
    warned: list[str] = []
    worker = Evaluator(runs, {suite.name: suite}, load=load_engine, warn=warned.append)

    jobs = finishes(worker.run_once)

    assert done(jobs) == [("tiny", 4), ("tiny", 2)]
    assert suite.evaluated == 2
    assert [text.split()[0] for text in warned] == ["tiny@4", "tiny@2"]
    assert "does not read back" in warned[0]


def test_a_checkpoint_whose_weights_diverged_is_reported_and_saves_nothing(tmp_path):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2,))

    def diverged(path):
        engine = load_engine(path)
        with torch.no_grad():
            for parameter in engine._model.parameters():
                parameter.fill_(float("nan"))
        return engine

    suite = ProbeSuite(load_probe_set("standard"), rating=1500)
    warned: list[str] = []
    worker = Evaluator(runs, {suite.name: suite}, load=diverged, warn=warned.append)

    finishes(worker.run_once)

    assert RunReader(run).evaluations() == []
    assert len(warned) == 1 and "not a number" in warned[0]


def test_evaluate_fails_when_a_named_checkpoint_changes_while_it_is_probed(
    tmp_path, monkeypatch, capsys
):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2,))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')
    checkpoint = run / "checkpoints" / "step-000000002.pt"

    def load_then_replace(path):
        engine = load_engine(path)
        later = checkpoint.stat().st_mtime + 10
        os.utime(checkpoint, (later, later))
        return engine

    monkeypatch.setattr("chess_ai.cli._evaluation_loader", lambda config: load_then_replace)

    with pytest.raises(SystemExit) as exited:
        main(["evaluate", "tiny@2"])

    assert exited.value.code == 2
    assert "tiny@2 changed while it was evaluated, so its result was not saved" in (
        capsys.readouterr().err
    )


def diverging_at(*steps: int):
    """A loader giving the checkpoints from ``steps`` NaN weights, as a run that diverged has."""

    def load(path):
        engine = load_engine(path)
        if engine.step in steps:
            with torch.no_grad():
                for parameter in engine._model.parameters():
                    parameter.fill_(float("nan"))
        return engine

    return load


def test_evaluate_goes_on_past_a_diverged_checkpoint_and_then_fails(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    run = model_run(runs, steps=(2, 4, 6))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')
    monkeypatch.setattr("chess_ai.cli._evaluation_loader", lambda config: diverging_at(4))

    with pytest.raises(SystemExit) as exited:
        main(["evaluate", "tiny"])

    assert exited.value.code == 2
    assert [(entry.step, entry.suite) for entry in RunReader(run).evaluations()] == [
        (2, SUITE),
        (2, SAMPLE_GAMES),
        (6, SUITE),
        (6, SAMPLE_GAMES),
    ]
    err = capsys.readouterr().err
    assert "warning: tiny@4 probe-positions: the network's outputs" in err
    assert "warning: tiny@4 sample-games: the network's outputs" in err
    assert "error: 1 checkpoint could not be evaluated" in err


def test_evaluate_fails_on_a_named_checkpoint_that_diverged(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')
    monkeypatch.setattr("chess_ai.cli._evaluation_loader", lambda config: diverging_at(2))

    with pytest.raises(SystemExit) as exited:
        main(["evaluate", "tiny@2"])

    assert exited.value.code == 2
    assert "error: tiny@2 probe-positions: the network's outputs" in capsys.readouterr().err
