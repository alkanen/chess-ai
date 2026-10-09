"""The runs dashboard's API: the run list, stale runs, and one run followed as it trains.

The runs here are fake run directories written through the run store and appended to while a
viewer is connected, which is all the web server ever sees of a trainer.
"""

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect
from training_helpers import model_run

from chess_ai.config import Config, PathsConfig, ServerConfig
from chess_ai.encoders import create_encoder
from chess_ai.evaluator import ProbeSuite
from chess_ai.inference import load_engine
from chess_ai.probes import SUITE, load_probe_set
from chess_ai.training.run_store import (
    METRICS_FILE,
    NOTES_FILE,
    STATUS_FILE,
    CheckpointPolicy,
    DatasetReference,
    GpuStats,
    Heartbeat,
    ModelReference,
    RunInfo,
    RunNotes,
    RunReader,
    RunStatus,
    RunWriter,
    save_notes,
)
from chess_ai.web import create_app
from chess_ai.web.runs import (
    EvaluationsEvent,
    MetricsEvent,
    RunStateEvent,
    RunStream,
    is_stale,
)

PREFIX = "/chess"

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def info(name: str, created: datetime = NOW) -> RunInfo:
    return RunInfo(
        name=name,
        created=created,
        seed=7,
        code_version="test",
        device="cpu",
        dataset=DatasetReference(
            name="lichess-2024",
            directory="/data/datasets/lichess-2024",
            format_version=1,
            games=10,
            positions=500,
            train_positions=450,
            validation_positions=50,
        ),
        encoder=create_encoder("board-planes").spec,
        model=ModelReference(architecture="mlp", options={"width": 8}, parameter_count=1234),
        config={},
        steps=100,
        batch_size=8,
    )


def new_run(runs: Path, name: str = "live", **options) -> RunWriter:
    """A run directory being written, as a trainer would have just started it."""
    return RunWriter.create(
        runs / name,
        info(name, **options),
        config_text="",
        policy=CheckpointPolicy(),
        overwrite=True,
    )


def beat(runs: Path, name: str, status: RunStatus, *, updated: datetime, step: int = 10) -> None:
    """Write a heartbeat as of ``updated``, which a real writer cannot be told."""
    heartbeat = Heartbeat(
        status=status, pid=1, started=updated, updated=updated, step=step, steps=100
    )
    (runs / name / STATUS_FILE).write_text(heartbeat.model_dump_json())


@pytest.fixture
def runs(tmp_path) -> Path:
    return tmp_path / "runs"


def serve(tmp_path: Path, runs: Path, **server) -> TestClient:
    config = Config(
        server=ServerConfig(path_prefix=PREFIX, **server),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
    )
    return TestClient(create_app(config, static_dir=tmp_path / "static", run_poll_seconds=0.02))


@pytest.fixture
def client(tmp_path, runs) -> Iterator[TestClient]:
    with serve(tmp_path, runs) as serving:
        yield serving


def follow(client: TestClient, name: str = "live"):
    return client.websocket_connect(f"{PREFIX}/api/runs/{name}/ws")


def next_of(websocket: WebSocketTestSession, kind: str) -> dict:
    """The next event of ``kind``, passing over any of the other kind on the way."""
    while (event := websocket.receive_json())["type"] != kind:
        pass
    return event


# The run list.


def test_the_run_list_says_what_each_run_is_where_it_has_got_and_what_it_measured(client, runs):
    with new_run(runs) as run:
        run.log(step=50, split="train", loss=3.0, learning_rate=0.001)
        run.log(step=50, split="validation", loss=3.2, top1=0.25, illegal_top_move_rate=0.1)
        run.log(step=100, split="train", loss=2.5, learning_rate=0.0005)
        run.heartbeat(RunStatus.RUNNING, step=100, steps=100, epoch=0.5, eta_seconds=12.0)

        listed = client.get(f"{PREFIX}/api/runs")

    assert listed.status_code == 200
    [summary] = listed.json()
    assert summary["name"] == "live"
    assert summary["architecture"] == "mlp"
    assert summary["dataset"] == "lichess-2024"
    assert summary["dataset_version"] == 1
    assert summary["status"] == "running"
    assert summary["stale"] is False
    assert (summary["step"], summary["steps"], summary["epoch"]) == (100, 100, 0.5)
    assert summary["eta_seconds"] == 12.0
    assert summary["latest_train"]["loss"] == 2.5
    assert summary["latest_validation"]["top1"] == 0.25


def test_the_run_list_reads_only_what_each_log_has_added_since_it_was_last_asked(client, runs):
    """A run that never validates must not cost a read of its whole log on every refresh.

    The bytes already read are rewritten in place, at the same length, into a validation line:
    a list that went back over them would find it, and one that reads only what was appended
    since cannot.
    """
    with new_run(runs) as run:
        run.log(step=1, split="train", loss=9.0)
        run.log(step=2, split="train", loss=8.0)
        assert client.get(f"{PREFIX}/api/runs").json()[0]["latest_validation"] is None

        log = runs / "live" / METRICS_FILE
        first = log.read_bytes().split(b"\n")[0]
        rewritten = b'{"step": 1, "split": "validation"}'
        with log.open("r+b") as f:
            f.write(rewritten.ljust(len(first)))
        run.log(step=3, split="train", loss=7.0)

        [summary] = client.get(f"{PREFIX}/api/runs").json()

    assert summary["latest_validation"] is None
    assert summary["latest_train"]["step"] == 3


def test_the_run_list_starts_a_runs_metrics_again_when_it_is_overwritten(client, runs):
    with new_run(runs) as run:
        run.log(step=1, split="validation", top1=0.5)
        run.log(step=2, split="train", loss=8.0)
        assert client.get(f"{PREFIX}/api/runs").json()[0]["latest_validation"]["top1"] == 0.5

    with new_run(runs, created=NOW + timedelta(hours=1)) as again:
        again.log(step=1, split="train", loss=5.0)

        [summary] = client.get(f"{PREFIX}/api/runs").json()

    assert summary["latest_validation"] is None
    assert summary["latest_train"]["loss"] == 5.0


def test_the_run_list_starts_again_on_a_new_run_whose_log_reuses_the_old_ones_file(client, runs):
    """An overwrite whose new log the file system gives the old one's inode, and outgrows it.

    The log alone then looks like the same file, grown: the same identity and no shorter than
    what was read. Which inode a new file gets is the file system's choice, so the old one is
    kept alive under another name, given the new run's log, and put back.
    """
    with new_run(runs) as run:
        run.log(step=1, split="validation", top1=0.5)
        run.log(step=2, split="train", loss=8.0)
        assert client.get(f"{PREFIX}/api/runs").json()[0]["latest_validation"]["top1"] == 0.5

    log = runs / "live" / METRICS_FILE
    old_file = runs / "live" / "old-inode"
    os.link(log, old_file)
    with new_run(runs, created=NOW + timedelta(hours=1)) as again:
        for step in range(1, 20):
            again.log(step=step, split="train", loss=5.0)
    with old_file.open("r+b") as f:
        f.truncate(0)
        f.write(log.read_bytes())
    os.replace(old_file, log)

    [summary] = client.get(f"{PREFIX}/api/runs").json()

    assert summary["latest_validation"] is None
    assert summary["latest_train"]["step"] == 19


def test_a_run_that_has_logged_nothing_yet_is_listed_without_metrics(client, runs):
    with new_run(runs):
        [summary] = client.get(f"{PREFIX}/api/runs").json()

    assert summary["status"] is None
    assert summary["latest_train"] is None
    assert summary["latest_validation"] is None


@pytest.mark.parametrize(
    ("status", "age", "stale"),
    [
        (RunStatus.RUNNING, timedelta(seconds=5), False),
        (RunStatus.RUNNING, timedelta(minutes=10), True),
        # A run that ended said so as it ended; an old heartbeat is the truth about it.
        (RunStatus.FINISHED, timedelta(days=30), False),
        (RunStatus.CRASHED, timedelta(days=30), False),
        (RunStatus.STOPPED, timedelta(days=30), False),
    ],
)
def test_a_running_run_whose_heartbeat_has_stopped_is_flagged_stale(
    client, runs, status, age, stale
):
    with new_run(runs):
        beat(runs, "live", status, updated=datetime.now(UTC) - age)

        [summary] = client.get(f"{PREFIX}/api/runs").json()

    assert summary["status"] == status
    assert summary["stale"] is stale


def test_how_old_a_heartbeat_may_be_is_configured(tmp_path, runs):
    with new_run(runs):
        beat(runs, "live", RunStatus.RUNNING, updated=datetime.now(UTC) - timedelta(seconds=30))

        with serve(tmp_path, runs) as patient:
            assert patient.get(f"{PREFIX}/api/runs").json()[0]["stale"] is False
        with serve(tmp_path, runs, stale_after_seconds=10) as impatient:
            assert impatient.get(f"{PREFIX}/api/runs").json()[0]["stale"] is True


def test_staleness_is_decided_on_the_heartbeats_age():
    heartbeat = Heartbeat(status=RunStatus.RUNNING, pid=1, started=NOW, updated=NOW, steps=1)

    assert not is_stale(heartbeat, now=NOW + timedelta(seconds=60), stale_after=60)
    assert is_stale(heartbeat, now=NOW + timedelta(seconds=61), stale_after=60)
    assert not is_stale(None, now=NOW, stale_after=60)


def test_the_run_list_gives_each_runs_title_and_tags(client, runs):
    with new_run(runs, "titled"), new_run(runs, "plain"):
        save_notes(RunReader(runs / "titled"), RunNotes(title="Wide MLP", tags=["mlp", "wide"]))

        listed = {run["name"]: run for run in client.get(f"{PREFIX}/api/runs").json()}

    assert (listed["titled"]["title"], listed["titled"]["tags"]) == ("Wide MLP", ["mlp", "wide"])
    assert (listed["plain"]["title"], listed["plain"]["tags"]) == (None, [])


def test_the_run_list_can_be_filtered_by_tag(client, runs):
    with new_run(runs, "a"), new_run(runs, "b"), new_run(runs, "c"):
        save_notes(RunReader(runs / "a"), RunNotes(tags=["mlp", "wide"]))
        save_notes(RunReader(runs / "b"), RunNotes(tags=["mlp"]))

        def names(*tags: str) -> set[str]:
            found = client.get(f"{PREFIX}/api/runs", params={"tag": list(tags)})
            return {run["name"] for run in found.json()}

        assert names() == {"a", "b", "c"}
        assert names("mlp") == {"a", "b"}
        assert names("mlp", "wide") == {"a"}
        assert names("resnet") == set()


def test_archived_runs_are_left_out_unless_asked_for(client, runs):
    with new_run(runs, "kept"), new_run(runs, "shelved"):
        save_notes(RunReader(runs / "shelved"), RunNotes(tags=["sweep", "archived"]))

        def names(**params) -> set[str]:
            return {run["name"] for run in client.get(f"{PREFIX}/api/runs", params=params).json()}

        assert names() == {"kept"}
        assert names(archived="true") == {"kept", "shelved"}
        # Asking for the tag is asking for the archived runs; nothing else could be meant.
        assert names(tag="archived") == {"shelved"}
        assert names(tag="sweep") == set()
        assert names(tag="sweep", archived="true") == {"shelved"}


def test_a_run_whose_notes_cannot_be_read_is_still_listed(client, runs):
    with new_run(runs):
        (runs / "live" / NOTES_FILE).write_text("{not json")

        [summary] = client.get(f"{PREFIX}/api/runs").json()

    assert (summary["name"], summary["architecture"], summary["tags"]) == ("live", "mlp", [])


# Notes, through the API.


def test_notes_round_trip_through_the_api_and_the_run_directory(client, runs):
    with new_run(runs):
        sent = {"title": "Wide MLP", "tags": ["mlp", "wide"], "notes": "Width 2048.\nBetter."}

        saved = client.put(f"{PREFIX}/api/runs/live/notes", json=sent)
        fetched = client.get(f"{PREFIX}/api/runs/live/notes")

    assert saved.status_code == 200
    assert saved.json() == sent
    assert fetched.json() == sent
    assert fetched.headers["Cache-Control"] == "no-store"
    assert RunReader(runs / "live").notes == RunNotes(**sent)


def test_a_run_nobody_has_annotated_has_empty_notes(client, runs):
    with new_run(runs):
        fetched = client.get(f"{PREFIX}/api/runs/live/notes")

    assert fetched.json() == {"title": None, "tags": [], "notes": ""}


def test_notes_sent_are_tidied_as_they_are_kept(client, runs):
    with new_run(runs):
        saved = client.put(
            f"{PREFIX}/api/runs/live/notes", json={"title": "  ", "tags": ["mlp ", "mlp"]}
        )

    assert saved.json() == {"title": None, "tags": ["mlp"], "notes": ""}


@pytest.mark.parametrize(
    "notes",
    [
        {"tags": ["two words"]},
        {"title": "two\nlines"},
        {"title": "x" * 201},
        {"colour": "red"},
    ],
)
def test_notes_that_cannot_be_kept_are_refused_and_nothing_changes(client, runs, notes):
    with new_run(runs):
        save_notes(RunReader(runs / "live"), RunNotes(title="kept"))

        refused = client.put(f"{PREFIX}/api/runs/live/notes", json=notes)

    assert refused.status_code == 422
    assert RunReader(runs / "live").notes == RunNotes(title="kept")


def test_notes_of_a_run_that_is_not_here_are_not_found(client, runs):
    assert client.get(f"{PREFIX}/api/runs/ghost/notes").status_code == 404
    assert client.put(f"{PREFIX}/api/runs/ghost/notes", json={}).status_code == 404
    assert not (runs / "ghost").exists()


def test_notes_that_cannot_be_read_are_the_servers_fault(client, runs):
    with new_run(runs):
        (runs / "live" / NOTES_FILE).write_text("{not json")

        assert client.get(f"{PREFIX}/api/runs/live/notes").status_code == 500


# One run, followed.


def test_a_viewer_gets_the_run_and_its_whole_log_and_then_what_is_appended(client, runs):
    with new_run(runs) as run:
        run.log(step=1, split="train", loss=9.0)
        run.heartbeat(RunStatus.RUNNING, step=1, steps=100)

        with follow(client) as websocket:
            first = websocket.receive_json()
            assert first["type"] == "run"
            assert first["name"] == "live"
            assert first["info"]["model"]["architecture"] == "mlp"
            assert first["heartbeat"]["status"] == "running"
            assert first["stale"] is False
            history = websocket.receive_json()
            assert history["type"] == "metrics"
            assert history["reset"] is True
            assert [record["step"] for record in history["records"]] == [1]

            run.log(step=2, split="train", loss=8.0)
            run.log(step=2, split="validation", loss=8.5)

            added = next_of(websocket, "metrics")
            records = added["records"]
            # Two lines written in one go may arrive in one event or two.
            if len(records) == 1:
                records += next_of(websocket, "metrics")["records"]
            assert added["reset"] is False
            assert [(record["step"], record["split"]) for record in records] == [
                (2, "train"),
                (2, "validation"),
            ]


def test_a_viewer_hears_when_the_heartbeat_changes(client, runs):
    with new_run(runs) as run:
        run.heartbeat(RunStatus.RUNNING, step=1, steps=100)

        with follow(client) as websocket:
            assert websocket.receive_json()["heartbeat"]["step"] == 1

            run.finish(RunStatus.FINISHED, step=100, steps=100)

            changed = next_of(websocket, "run")
            assert changed["heartbeat"]["status"] == "finished"
            assert changed["heartbeat"]["step"] == 100


def test_a_viewer_is_sent_the_throughput_time_left_and_gpu_statistics(client, runs):
    gpu = GpuStats(
        name="NVIDIA GeForce RTX 4090",
        utilization_percent=87.0,
        memory_used_bytes=20 * 2**30,
        memory_total_bytes=24 * 2**30,
        process_memory_bytes=6 * 2**30,
        temperature_celsius=64.0,
    )
    with new_run(runs) as run:
        run.heartbeat(
            RunStatus.RUNNING,
            step=40,
            steps=100,
            epoch=0.25,
            positions_per_second=344_538.5,
            eta_seconds=3_900.0,
            gpu=gpu,
        )

        with follow(client) as websocket:
            heartbeat = websocket.receive_json()["heartbeat"]

    assert (heartbeat["step"], heartbeat["steps"], heartbeat["epoch"]) == (40, 100, 0.25)
    assert heartbeat["positions_per_second"] == 344_538.5
    assert heartbeat["eta_seconds"] == 3_900.0
    assert heartbeat["gpu"] == gpu.model_dump()


def test_gpu_statistics_nvml_could_not_give_are_sent_as_missing(client, runs):
    with new_run(runs) as run:
        run.heartbeat(
            RunStatus.RUNNING,
            steps=100,
            gpu=GpuStats(memory_used_bytes=20 * 2**30, memory_total_bytes=24 * 2**30),
        )

        with follow(client) as websocket:
            gpu = websocket.receive_json()["heartbeat"]["gpu"]

    assert gpu == {
        "name": None,
        "utilization_percent": None,
        "memory_used_bytes": 20 * 2**30,
        "memory_total_bytes": 24 * 2**30,
        "process_memory_bytes": None,
        "temperature_celsius": None,
    }


def test_a_run_on_the_cpu_is_sent_no_gpu(client, runs):
    with new_run(runs) as run:
        run.heartbeat(RunStatus.RUNNING, steps=100)

        with follow(client) as websocket:
            assert websocket.receive_json()["heartbeat"]["gpu"] is None


def test_a_viewer_hears_when_the_notes_change(client, runs):
    with new_run(runs), follow(client) as websocket:
        assert websocket.receive_json()["notes"] == {"title": None, "tags": [], "notes": ""}

        client.put(f"{PREFIX}/api/runs/live/notes", json={"title": "Renamed", "tags": ["x"]})

        changed = next_of(websocket, "run")

    assert changed["notes"] == {"title": "Renamed", "tags": ["x"], "notes": ""}


def test_a_viewer_hears_when_a_run_goes_stale_though_nothing_on_the_disk_changed(tmp_path, runs):
    with new_run(runs), serve(tmp_path, runs, stale_after_seconds=0.3) as client:
        beat(runs, "live", RunStatus.RUNNING, updated=datetime.now(UTC))

        with follow(client) as websocket:
            assert websocket.receive_json()["stale"] is False

            assert next_of(websocket, "run")["stale"] is True


def test_a_completed_run_is_sent_whole(client, runs):
    with new_run(runs) as run:
        for step in range(1, 201):
            run.log(step=step, split="train", loss=1.0 / step)
        run.finish(RunStatus.FINISHED, step=200, steps=200)

    with follow(client) as websocket:
        assert websocket.receive_json()["heartbeat"]["status"] == "finished"
        history = websocket.receive_json()

    assert len(history["records"]) == 200


def test_a_viewer_starts_again_when_the_run_is_started_again(client, runs):
    with new_run(runs) as run:
        for step in (1, 2, 3):
            run.log(step=step, split="train", loss=9.0)
        with follow(client) as websocket:
            websocket.receive_json()
            assert len(websocket.receive_json()["records"]) == 3

            # The first trainer is gone before the second can have the directory: two at once
            # is what the writer's lock refuses.
            run.close()
            with new_run(runs, created=NOW + timedelta(hours=1)) as again:
                again.log(step=1, split="train", loss=5.0)

                restarted = next_of(websocket, "metrics")
                while not restarted["records"]:
                    restarted = next_of(websocket, "metrics")

    assert restarted["reset"] is True
    assert [record["loss"] for record in restarted["records"]] == [5.0]


def test_a_run_that_is_not_here_is_answered_with_an_error(client):
    with follow(client, "missing") as websocket:
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "missing" in error["message"]
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


def test_the_run_channel_is_only_under_the_prefix(client, runs):
    with (
        new_run(runs),
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/api/runs/live/ws") as websocket,
    ):
        websocket.receive_json()


# The stream itself, against a clock of its own.


def test_a_stream_sends_the_run_state_only_when_it_changes(runs):
    with new_run(runs) as run:
        run.heartbeat(RunStatus.RUNNING, step=1, steps=100)
        stream = RunStream(RunReader(runs / "live"), stale_after=60, clock=lambda: NOW)

        first = stream.poll()
        assert [type(event) for event in first] == [RunStateEvent, MetricsEvent, EvaluationsEvent]
        assert stream.poll() == []

        run.log(step=1, split="train", loss=1.0)
        assert [type(event) for event in stream.poll()] == [MetricsEvent]


def test_a_stream_flags_staleness_as_time_passes(runs):
    now = [NOW]
    with new_run(runs):
        beat(runs, "live", RunStatus.RUNNING, updated=NOW)
        stream = RunStream(RunReader(runs / "live"), stale_after=60, clock=lambda: now[0])
        assert stream.poll()[0].stale is False

        now[0] = NOW + timedelta(seconds=61)
        [event] = stream.poll()

    assert isinstance(event, RunStateEvent)
    assert event.stale is True


def test_a_stream_follows_a_run_whose_files_are_not_written_yet(runs):
    (runs / "early").mkdir(parents=True)
    stream = RunStream(RunReader(runs / "early"), stale_after=60, clock=lambda: NOW)

    state, metrics, evaluations = stream.poll()

    assert isinstance(state, RunStateEvent)
    assert (state.info, state.heartbeat) == (None, None)
    assert isinstance(metrics, MetricsEvent)
    assert (metrics.reset, metrics.records) == (True, [])
    assert evaluations == EvaluationsEvent(results=[])


def test_a_stream_starts_again_on_a_new_run_whose_log_reuses_the_old_ones_file(runs):
    """The stream tells runs apart by when they started, as the run list does; see above."""
    with new_run(runs) as run:
        run.log(step=1, split="validation", top1=0.5)
        stream = RunStream(RunReader(runs / "live"), stale_after=60, clock=lambda: NOW)
        stream.poll()

    log = runs / "live" / METRICS_FILE
    old_file = runs / "live" / "old-inode"
    os.link(log, old_file)
    with new_run(runs, created=NOW + timedelta(hours=1)) as again:
        for step in range(1, 20):
            again.log(step=step, split="train", loss=5.0)
    with old_file.open("r+b") as f:
        f.truncate(0)
        f.write(log.read_bytes())
    os.replace(old_file, log)

    state, metrics = stream.poll()

    assert isinstance(state, RunStateEvent)
    assert isinstance(metrics, MetricsEvent)
    assert metrics.reset is True
    assert [record["step"] for record in metrics.records] == list(range(1, 20))


# Evaluation results.


def probed(runs: Path, name: str = "tiny", *, steps=(2, 4)) -> RunReader:
    """A run with real checkpoints, the first of them probed."""
    run = RunReader(model_run(runs, name, steps=steps))
    suite = ProbeSuite(load_probe_set("standard"), rating=1800)
    first = run.checkpoints()[0]
    suite.evaluate(
        run, first, run.checkpoint_written(first), load_engine(run.checkpoint_path(first))
    )
    return run


def test_a_stream_says_which_checkpoints_have_results_and_when_another_arrives(runs):
    run = probed(runs)
    stream = RunStream(run, stale_after=60, clock=lambda: NOW)

    *_, first = stream.poll()
    nothing_new = stream.poll()
    later = run.checkpoints()[1]
    ProbeSuite(load_probe_set("standard"), rating=1800).evaluate(
        run, later, run.checkpoint_written(later), load_engine(run.checkpoint_path(later))
    )
    [added] = stream.poll()

    assert isinstance(first, EvaluationsEvent)
    assert [(entry.step, entry.suite) for entry in first.results] == [(2, SUITE)]
    assert nothing_new == []
    assert isinstance(added, EvaluationsEvent)
    assert [(entry.step, entry.suite) for entry in added.results] == [(2, SUITE), (4, SUITE)]


def test_a_viewer_hears_of_a_new_evaluation_result(client, runs):
    run = probed(runs)

    with follow(client, "tiny") as websocket:
        first = next_of(websocket, "evaluations")
        later = run.checkpoints()[1]
        ProbeSuite(load_probe_set("standard"), rating=1800).evaluate(
            run, later, run.checkpoint_written(later), load_engine(run.checkpoint_path(later))
        )
        added = next_of(websocket, "evaluations")

    assert [result["step"] for result in first["results"]] == [2]
    assert [result["step"] for result in added["results"]] == [2, 4]


def test_the_evaluation_results_of_a_run_are_listed(client, runs):
    probed(runs)

    response = client.get(f"{PREFIX}/api/runs/tiny/evaluations")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    [entry] = response.json()
    assert (entry["step"], entry["suite"]) == (2, SUITE)


def test_a_probe_result_comes_with_each_position_to_draw(client, runs):
    probed(runs)

    response = client.get(f"{PREFIX}/api/runs/tiny/evaluations/2/probe-positions")

    assert response.status_code == 200
    result = response.json()
    assert (result["model"]["checkpoint"], result["model"]["rating"]) == (2, 1800)
    mate = next(position for position in result["positions"] if position["id"] == "scholars-mate")
    assert mate["best"] == ["h5f7"]
    snapshot = mate["snapshot"]
    assert snapshot["turn"] == "white"
    assert snapshot["last_move"] == {"from_square": "g8", "to_square": "f6"}
    assert snapshot["pieces"]["h5"] == {"color": "white", "type": "queen"}
    assert snapshot["legal_moves"] == {}


@pytest.mark.parametrize(
    ("path", "complaint"),
    [
        ("missing/evaluations/2/probe-positions", "missing"),
        ("tiny/evaluations/4/probe-positions", "no probe result for the checkpoint from step 4"),
        ("tiny/evaluations/2/tournament", "Not Found"),
        ("missing/evaluations", "missing"),
    ],
)
def test_a_result_that_is_not_here_is_not_found(client, runs, path, complaint):
    probed(runs)

    response = client.get(f"{PREFIX}/api/runs/{path}")

    assert response.status_code == 404
    assert complaint in response.json()["detail"]


def test_a_probe_result_that_cannot_be_read_is_the_servers_fault(client, runs):
    run = probed(runs)
    (run.evaluation_directory(2) / f"{SUITE}.json").write_text("{")

    response = client.get(f"{PREFIX}/api/runs/tiny/evaluations/2/probe-positions")

    assert response.status_code == 500
