"""Playing a checkpoint from the browser: picking one, and the game that comes of it.

The runs here are written straight into the runs directory rather than trained, which is all
the web server ever sees of a run anyway: a directory of files it only reads.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import chess
import pytest
from fastapi.testclient import TestClient
from pgn_helpers import read_back
from starlette.testclient import WebSocketTestSession
from training_helpers import model_run

from chess_ai.config import Config, InferenceConfig, PathsConfig, ServerConfig
from chess_ai.inference import DEFAULT_BATCH_SIZE
from chess_ai.training import run_store
from chess_ai.web import app as app_module
from chess_ai.web import create_app

PREFIX = "/chess"

RANDOM = {"kind": "random"}


def model(run: str = "tiny", **settings) -> dict:
    """What the browser posts for a side a checkpoint is to play."""
    return {"kind": "model", "run": run, **settings}


@pytest.fixture
def runs(tmp_path) -> Path:
    """A runs directory with one run in it, with checkpoints at steps 2 and 4."""
    model_run(tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def client(tmp_path, runs) -> Iterator[TestClient]:
    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
    )
    # Entering the client keeps one event loop, in which games play on between requests.
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as serving:
        yield serving


def start(client: TestClient, white: dict, black: dict, **changes):
    return client.post(
        f"{PREFIX}/api/game", json={"white": white, "black": black, "move_delay": 0, **changes}
    )


def play_out(websocket: WebSocketTestSession) -> tuple[list[dict], dict]:
    """Watch a game to its end, and return the moves it was played with and where it ended."""
    first = websocket.receive_json()
    assert first["type"] == "state"
    moves: list[dict] = list(first["game"]["moves"])
    position = first["game"]["position"]
    while position["game_over"] is None:
        event = websocket.receive_json()
        assert event["type"] == "move", f"unexpected {event['type']} event"
        moves.append(event["move"])
        position = event["position"]
    return moves, position


def test_the_runs_there_are_to_play_against_are_listed(client):
    listed = client.get(f"{PREFIX}/api/runs")

    assert listed.status_code == 200
    assert listed.headers["Cache-Control"] == "no-store"
    assert [run["name"] for run in listed.json()] == ["tiny"]
    run = listed.json()[0]
    assert run["architecture"] == "mlp"
    assert run["checkpoints"] == 2
    assert run["status"] == "finished"
    assert (run["step"], run["steps"]) == (4, 4)
    assert run["created"] is not None


def test_a_server_with_no_runs_says_so_rather_than_failing(tmp_path):
    config = Config(paths=PathsConfig(games=tmp_path / "games", runs=tmp_path / "nothing"))
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as empty:
        assert empty.get("/api/runs").json() == []


def test_a_directory_that_is_not_a_run_is_not_offered_as_one(client, runs):
    (runs / "notes").mkdir()

    assert [run["name"] for run in client.get(f"{PREFIX}/api/runs").json()] == ["tiny"]


def test_the_checkpoints_of_a_run_are_listed_newest_first(client):
    listed = client.get(f"{PREFIX}/api/runs/tiny/checkpoints")

    assert listed.status_code == 200
    assert listed.headers["Cache-Control"] == "no-store"
    body = listed.json()
    assert body["run"] == "tiny"
    assert [point["step"] for point in body["checkpoints"]] == [4, 2]
    assert [point["latest"] for point in body["checkpoints"]] == [True, False]
    # The helper's metrics fall as the steps rise, so the first checkpoint is the best one.
    assert [point["best"] for point in body["checkpoints"]] == [False, True]
    assert body["checkpoints"][0]["metrics"]["policy_loss"] == 2.0


def test_the_checkpoints_of_a_run_that_is_not_here(client):
    missing = client.get(f"{PREFIX}/api/runs/other/checkpoints")

    assert missing.status_code == 404
    assert "other" in missing.json()["detail"]


def test_a_name_that_could_never_be_a_run_is_refused_rather_than_looked_for(client):
    refused = client.get(f"{PREFIX}/api/runs/.hidden/checkpoints")

    assert refused.status_code == 404
    assert "invalid run name" in refused.json()["detail"]


def test_a_game_against_a_checkpoint_says_which_one_is_playing(client):
    started = start(client, model(checkpoint="latest", rating=1600), {"kind": "human"})

    assert started.status_code == 200
    white = started.json()["white"]
    assert white["name"] == "tiny step 4"
    assert white["accepts_moves"] is False
    assert white["model"] == {
        "run": "tiny",
        "checkpoint": 4,
        "rating": 1600,
        "strategy": "argmax",
        "temperature": None,
    }
    assert started.json()["black"]["model"] is None


@pytest.mark.parametrize(
    ("choice", "step"),
    [("latest", 4), ("best", 2), (2, 2), (4, 4)],
)
def test_which_checkpoint_of_the_run_plays(client, choice, step):
    started = start(client, model(checkpoint=choice), RANDOM)

    assert started.json()["white"]["model"]["checkpoint"] == step


def test_a_sampling_model_records_the_temperature_it_plays_at(client):
    started = start(client, model(strategy="sample", temperature=1.5, seed=4), RANDOM)

    model_played = started.json()["white"]["model"]
    assert (model_played["strategy"], model_played["temperature"]) == ("sample", 1.5)


def test_a_run_that_is_not_here_is_refused_and_the_game_goes_on(client):
    playing = start(client, RANDOM, RANDOM)

    refused = start(client, model(run="other"), RANDOM)

    assert refused.status_code == 400
    assert "other" in refused.json()["detail"]
    # The game that was being played is still the one on show.
    assert client.get(f"{PREFIX}/api/game/pgn?game={playing.json()['id']}").status_code == 200


def test_a_checkpoint_the_run_never_saved_is_refused(client):
    refused = start(client, model(checkpoint=3), RANDOM)

    assert refused.status_code == 400
    assert "step 3" in refused.json()["detail"]


def test_a_run_with_no_checkpoints_yet_is_refused(client, runs):
    (runs / "starting").mkdir()
    (runs / "starting" / "run.json").write_text("{}", encoding="utf-8")

    refused = start(client, model(run="starting"), RANDOM)

    assert refused.status_code == 400
    assert "not saved a checkpoint" in refused.json()["detail"]


def test_a_checkpoint_that_cannot_be_read_is_refused_with_a_reason(client, runs):
    broken = sorted((runs / "tiny" / "checkpoints").glob("*.pt"))[-1]
    broken.write_bytes(b"not a checkpoint any more")

    refused = start(client, model(checkpoint="latest"), RANDOM)

    assert refused.status_code == 400
    assert "checkpoint" in refused.json()["detail"]


@pytest.mark.parametrize(
    "change",
    [
        {"white": {"kind": "model"}},
        {"white": {"kind": "model", "run": "tiny", "checkpoint": "middling"}},
        {"white": {"kind": "model", "run": "tiny", "strategy": "vibes"}},
        {"white": {"kind": "model", "run": "tiny", "temperature": 0}},
        {"white": {"kind": "model", "run": "tiny", "rating": 9000}},
        {"white": {"kind": "model", "run": "tiny", "depth": 4}},
    ],
)
def test_a_model_player_the_server_cannot_read_is_refused(client, change):
    refused = client.post(f"{PREFIX}/api/game", json={"white": RANDOM, "black": RANDOM, **change})

    assert refused.status_code == 422


def test_a_game_between_two_checkpoints_is_watched_to_its_end(client):
    with client.websocket_connect(f"{PREFIX}/api/game/ws") as websocket:
        assert websocket.receive_json()["type"] == "no_game"
        started = start(
            client,
            model(checkpoint="best", rating=1200),
            model(checkpoint="latest", rating=2000, strategy="sample", seed=1),
        )
        assert started.status_code == 200

        moves, ended = play_out(websocket)

    assert ended["game_over"] is not None
    board = chess.Board()
    for move in moves:
        board.push_uci(move["uci"])
    assert board.fen() == ended["fen"], "every move the viewer saw leads to where it ended"
    # Both sides are models, so every move was chosen by one and says what it was thinking.
    assert all(move["thoughts"]["candidates"] for move in moves)
    assert all(move["thoughts"]["wdl"] is not None for move in moves)


def test_a_finished_game_is_saved_with_the_checkpoints_that_played_it(client, tmp_path):
    with client.websocket_connect(f"{PREFIX}/api/game/ws") as websocket:
        assert websocket.receive_json()["type"] == "no_game"
        start(client, model(checkpoint="best", rating=1200), model(checkpoint="latest"))
        play_out(websocket)

    saved = sorted((tmp_path / "games").glob("*.pgn"))
    assert len(saved) == 1
    headers = read_back(saved[0].read_text(encoding="utf-8")).headers
    assert (headers["WhiteRun"], headers["WhiteCheckpoint"]) == ("tiny", "2")
    assert (headers["BlackRun"], headers["BlackCheckpoint"]) == ("tiny", "4")
    assert headers["WhiteRating"] == "1200"
    assert "BlackRating" not in headers
    assert (headers["WhiteSelection"], headers["BlackSelection"]) == ("argmax", "argmax")


def test_a_game_in_progress_against_a_checkpoint_downloads_with_its_headers(client):
    started = start(client, {"kind": "human"}, model(checkpoint=2, rating=1800))

    pgn = client.get(f"{PREFIX}/api/game/pgn?game={started.json()['id']}")

    headers = read_back(pgn.text).headers
    assert headers["Black"] == "tiny step 2"
    assert headers["BlackRun"] == "tiny"
    assert headers["BlackRating"] == "1800"
    assert "WhiteRun" not in headers


def test_the_inference_batch_the_config_names_is_the_one_the_engine_uses():
    """The config repeats the default rather than importing torch to read it."""
    assert InferenceConfig().batch_size == DEFAULT_BATCH_SIZE


def test_a_player_kind_nothing_here_makes_fails_where_the_mistake_is():
    """A fifth kind of player added without a case here is a mistake worth hearing about.

    Reached by calling the maker directly, because it cannot be reached through the API: a
    kind the request model does not know is refused by the request model. What is being
    checked is that the omission fails at the point of the omission, rather than handing a
    game a player of ``None`` that breaks on its first move, a task away from the cause.
    """

    class OracleSpec(app_module.PlayerSpec):
        kind: Literal["oracle"]

    with pytest.raises(AssertionError, match="oracle"):
        asyncio.run(app_module._player(OracleSpec(kind="oracle"), Config(), {}))


def test_one_checkpoint_playing_itself_is_loaded_once(client, monkeypatch):
    """Two players of one checkpoint are two ratings and one set of weights.

    Loading it twice costs the read twice over and holds the weights twice for the length of
    the game, which on a real model is hundreds of megabytes of nothing.
    """
    import chess_ai.inference as inference

    loaded = []
    real = inference.load_engine

    def counted(path, **settings):
        loaded.append(path)
        return real(path, **settings)

    monkeypatch.setattr(inference, "load_engine", counted)

    started = start(
        client,
        model(checkpoint="latest", rating=1200),
        model(checkpoint="latest", rating=2400, strategy="sample"),
    )

    assert started.status_code == 200
    assert len(loaded) == 1, "the same checkpoint was read twice"
    # Sharing the weights must not share what each side was asked for.
    assert started.json()["white"]["model"]["rating"] == 1200
    assert started.json()["black"]["model"]["rating"] == 2400


def test_two_different_checkpoints_are_two_engines(client, monkeypatch):
    import chess_ai.inference as inference

    loaded = []
    real = inference.load_engine
    monkeypatch.setattr(
        inference, "load_engine", lambda path, **kw: (loaded.append(path), real(path, **kw))[1]
    )

    start(client, model(checkpoint="best"), model(checkpoint="latest"))

    assert len(loaded) == 2


def test_a_device_the_machine_cannot_give_is_the_servers_fault_not_the_viewers(client, monkeypatch):
    """A config asking for a GPU that is not there is nothing the person clicking Start did."""
    from chess_ai.inference import engine as engine_module
    from chess_ai.training.hardware import HardwareError

    def missing(requested: str):
        raise HardwareError(f"the config asks to play on {requested}, and there is no such device")

    monkeypatch.setattr(engine_module, "resolve_device", missing)

    refused = start(client, model(checkpoint="latest"), RANDOM)

    assert refused.status_code == 500
    assert "no such device" in refused.json()["detail"]


def test_one_run_that_cannot_be_read_does_not_sink_the_list(client, runs, monkeypatch):
    """The form offers every other run rather than saying it could not list any of them."""
    model_run(runs, "other")
    real = run_store.read_checkpoints

    def unreadable(directory: Path):
        if directory.parent.name == "tiny":
            raise FileNotFoundError(2, "No such file or directory", str(directory))
        return real(directory)

    monkeypatch.setattr(run_store, "read_checkpoints", unreadable)

    listed = client.get(f"{PREFIX}/api/runs")

    assert listed.status_code == 200
    found = {run["name"]: run for run in listed.json()}
    assert set(found) == {"other", "tiny"}
    assert found["other"]["checkpoints"] == 2
    # Nothing could be counted for the run being written, and the name is still offered.
    assert found["tiny"]["checkpoints"] == 0
