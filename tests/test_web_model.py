"""Playing a checkpoint from the browser: picking one, and the game that comes of it.

The runs here are written straight into the runs directory rather than trained, which is all
the web server ever sees of a run anyway: a directory of files it only reads.
"""

import asyncio
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import chess
import pytest
from fastapi.testclient import TestClient
from pgn_helpers import read_back
from starlette.testclient import WebSocketTestSession
from training_helpers import model_run

from chess_ai.config import Config, GamesConfig, InferenceConfig, PathsConfig, ServerConfig
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
        f"{PREFIX}/api/games", json={"white": white, "black": black, "move_delay": 0, **changes}
    )


def watching(client: TestClient, started):
    """Follow the game that was started, through its watch link."""
    return client.websocket_connect(f"{PREFIX}/api/games/{started.json()['links']['watch']}/ws")


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
    white = started.json()["game"]["white"]
    assert white["name"] == "tiny step 4"
    assert white["accepts_moves"] is False
    assert white["model"] == {
        "run": "tiny",
        "checkpoint": 4,
        "rating": 1600,
        "strategy": "argmax",
        "temperature": None,
    }
    assert started.json()["game"]["black"]["model"] is None


@pytest.mark.parametrize(
    ("choice", "step"),
    [("latest", 4), ("best", 2), (2, 2), (4, 4)],
)
def test_which_checkpoint_of_the_run_plays(client, choice, step):
    started = start(client, model(checkpoint=choice), RANDOM)

    assert started.json()["game"]["white"]["model"]["checkpoint"] == step


def test_a_sampling_model_records_the_temperature_it_plays_at(client):
    started = start(client, model(strategy="sample", temperature=1.5, seed=4), RANDOM)

    model_played = started.json()["game"]["white"]["model"]
    assert (model_played["strategy"], model_played["temperature"]) == ("sample", 1.5)


def test_a_run_that_is_not_here_is_refused_and_the_game_goes_on(client):
    playing = start(client, RANDOM, RANDOM)

    refused = start(client, model(run="other"), RANDOM)

    assert refused.status_code == 400
    assert "other" in refused.json()["detail"]
    # The game that was being played is still there.
    watch = playing.json()["links"]["watch"]
    assert client.get(f"{PREFIX}/api/games/{watch}/pgn").status_code == 200


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
    refused = client.post(f"{PREFIX}/api/games", json={"white": RANDOM, "black": RANDOM, **change})

    assert refused.status_code == 422


def test_a_game_between_two_checkpoints_is_watched_to_its_end(client):
    started = start(
        client,
        model(checkpoint="best", rating=1200),
        model(checkpoint="latest", rating=2000, strategy="sample", seed=1),
    )
    assert started.status_code == 200
    with watching(client, started) as websocket:
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
    started = start(client, model(checkpoint="best", rating=1200), model(checkpoint="latest"))
    with watching(client, started) as websocket:
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

    pgn = client.get(f"{PREFIX}/api/games/{started.json()['links']['watch']}/pgn")

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
    assert started.json()["game"]["white"]["model"]["rating"] == 1200
    assert started.json()["game"]["black"]["model"]["rating"] == 2400


def test_games_playing_the_same_checkpoint_share_one_load(client, monkeypatch):
    """Several friends playing the same model hold it in memory once, not once each."""
    import chess_ai.inference as inference

    loaded = []
    real = inference.load_engine
    monkeypatch.setattr(
        inference, "load_engine", lambda path, **kw: (loaded.append(path), real(path, **kw))[1]
    )

    first = start(client, {"kind": "human"}, model(checkpoint="latest", rating=1200))
    second = start(client, model(checkpoint="latest", rating=1800), {"kind": "human"})

    assert (first.status_code, second.status_code) == (200, 200)
    assert len(loaded) == 1, "the same checkpoint was read for each game"


def test_a_checkpoint_beyond_the_limit_takes_the_place_of_the_one_used_longest_ago(
    tmp_path, runs, monkeypatch
):
    import chess_ai.inference as inference

    loaded = []
    real = inference.load_engine
    monkeypatch.setattr(
        inference, "load_engine", lambda path, **kw: (loaded.append(path.name), real(path, **kw))[1]
    )
    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
        games=GamesConfig(max_loaded_checkpoints=1),
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as small:
        for step in (2, 4, 2):
            assert start(small, {"kind": "human"}, model(checkpoint=step)).status_code == 200

    assert len(loaded) == 3, "with room for one, the first checkpoint had to be read again"


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


def test_a_checkpoint_replaced_while_a_game_waits_is_not_played_in_its_place(
    tmp_path, runs, monkeypatch
):
    """A run restarted under the same name writes new weights at the same steps.

    A game whose checkpoint was let go of in the meantime must not carry on with those as if
    they were the ones it began with, which it would go on naming in its state and its PGN.
    It waits for somebody to choose a checkpoint to carry on with instead.
    """
    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
        games=GamesConfig(max_loaded_checkpoints=1),
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as small:
        first = start(small, {"kind": "human"}, model(checkpoint=2)).json()["links"]
        # Another game's checkpoint takes the only place, so the first one is let go of.
        assert start(small, {"kind": "human"}, model(checkpoint=4)).status_code == 200
        # Restarted with --overwrite: the old run is cleared and new weights saved at the
        # same steps.
        shutil.rmtree(runs / "tiny")
        model_run(runs, seed=99)

        with small.websocket_connect(f"{PREFIX}/api/games/{first['white']}/ws") as websocket:
            assert websocket.receive_json()["type"] == "state"
            websocket.send_json({"type": "move", "uci": "e2e4"})
            assert websocket.receive_json()["type"] == "move"
            paused = websocket.receive_json()

    assert paused["type"] == "paused"
    assert paused["paused"]["side"] == "black"
    assert "has been replaced" in paused["paused"]["reason"]


def pause(client: TestClient, links: dict, runs: Path) -> dict:
    """Delete the checkpoint the model in ``links``' game plays, and have it asked to move."""
    shutil.rmtree(runs / "tiny" / "checkpoints")
    with client.websocket_connect(f"{PREFIX}/api/games/{links['white']}/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "move", "uci": "e2e4"})
        assert websocket.receive_json()["type"] == "move"
        return websocket.receive_json()


def replace(client: TestClient, link: str, **settings):
    return client.post(f"{PREFIX}/api/games/{link}/replace", json=model(**settings))


@pytest.fixture
def small(tmp_path, runs) -> Iterator[TestClient]:
    """A server that holds one checkpoint at a time, so that a deleted one is not still held.

    There is a second run, ``other``, for another game to play, which takes that one place.
    """
    model_run(runs, "other")
    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
        games=GamesConfig(max_loaded_checkpoints=1),
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as serving:
        yield serving


def test_a_checkpoint_deleted_while_a_game_waits_pauses_the_game_for_another(small, runs):
    links = start(small, {"kind": "human"}, model(checkpoint=2)).json()["links"]
    # The game's checkpoint is let go of, so that it has to be read again for its next move.
    assert start(small, {"kind": "human"}, model("other")).status_code == 200

    paused = pause(small, links, runs)
    seen = small.get(f"{PREFIX}/api/games/{links['watch']}").json()

    assert paused["type"] == "paused"
    assert seen["game"]["paused"] == paused["paused"]
    assert seen["game"]["position"]["game_over"] is None


def test_the_person_at_a_paused_game_chooses_another_checkpoint_and_plays_on(small, runs, tmp_path):
    links = start(small, {"kind": "human"}, model(checkpoint=2)).json()["links"]
    assert start(small, {"kind": "human"}, model("other")).status_code == 200
    pause(small, links, runs)
    # Somebody trains the run again, under the same name.
    shutil.rmtree(runs / "tiny")
    model_run(runs, seed=99)

    with small.websocket_connect(f"{PREFIX}/api/games/{links['white']}/ws") as websocket:
        assert websocket.receive_json()["game"]["paused"] is not None
        answer = replace(small, links["white"], checkpoint="latest")
        replaced = websocket.receive_json()
        moved = websocket.receive_json()
        websocket.send_json({"type": "resign"})
        assert websocket.receive_json()["type"] == "game_over"

    assert answer.status_code == 200, answer.text
    assert answer.json()["game"]["black"]["model"]["checkpoint"] == 4
    assert replaced["type"] == "replaced"
    assert replaced["replacement"]["ply"] == 1
    assert replaced["replacement"]["old"]["name"] == "tiny step 2"
    assert replaced["replacement"]["new"]["name"] == "tiny step 4"
    assert moved["type"] == "move" and moved["ply"] == 2
    [saved] = (tmp_path / "games").glob("*.pgn")
    record = read_back(saved.read_text())
    assert record.headers["BlackCheckpoint"] == "4"
    assert record.next() is not None
    assert record.next().comment == "tiny step 2 replaced by tiny step 4"


def test_a_watch_link_cannot_choose_the_checkpoint_a_paused_game_goes_on_with(small, runs):
    links = start(small, {"kind": "human"}, model(checkpoint=2)).json()["links"]
    assert start(small, {"kind": "human"}, model("other")).status_code == 200
    pause(small, links, runs)

    refused = replace(small, links["watch"], run="other")

    assert refused.status_code == 403


def test_a_game_that_is_not_paused_has_no_checkpoint_to_replace(client):
    links = start(client, {"kind": "human"}, model()).json()["links"]

    refused = replace(client, links["white"])

    assert refused.status_code == 409
    assert "not waiting" in refused.json()["detail"]


def test_a_checkpoint_that_cannot_take_over_is_refused_and_the_game_still_waits(small, runs):
    links = start(small, {"kind": "human"}, model(checkpoint=2)).json()["links"]
    assert start(small, {"kind": "human"}, model("other")).status_code == 200
    pause(small, links, runs)

    refused = replace(small, links["white"])

    assert refused.status_code == 400
    assert small.get(f"{PREFIX}/api/games/{links['white']}").json()["game"]["paused"] is not None


def test_a_game_against_a_checkpoint_carries_on_after_a_restart_loading_it_when_needed(
    tmp_path, runs, monkeypatch
):
    import chess_ai.inference as inference

    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as before:
        links = start(before, {"kind": "human"}, model(rating=1500)).json()["links"]

    loaded = []
    real = inference.load_engine
    monkeypatch.setattr(
        inference, "load_engine", lambda path, **kw: (loaded.append(path), real(path, **kw))[1]
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as after:
        seen = after.get(f"{PREFIX}/api/games/{links['white']}").json()
        assert loaded == [], "a checkpoint was loaded before any game needed it"
        with after.websocket_connect(f"{PREFIX}/api/games/{links['white']}/ws") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "move", "uci": "e2e4"})
            assert websocket.receive_json()["type"] == "move"
            answered = websocket.receive_json()

    assert seen["game"]["black"]["model"] == {
        "run": "tiny",
        "checkpoint": 2,
        "rating": 1500,
        "strategy": "argmax",
        "temperature": None,
    }
    assert answered["type"] == "move"
    assert len(loaded) == 1


def test_a_checkpoint_deleted_while_the_server_was_down_pauses_the_game(tmp_path, runs):
    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(games=tmp_path / "games", runs=runs),
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as before:
        links = start(before, {"kind": "human"}, model()).json()["links"]

    with TestClient(create_app(config, static_dir=tmp_path / "static")) as after:
        paused = pause(after, links, runs)

    assert paused["type"] == "paused"
