"""Stockfish as a player in the browser's games: chosen for either side, and gone afterwards.

The engine is the stand-in from ``fixtures/fake_uci.py``, set as ``[stockfish] path`` the way
a real one would be, which logs what it is told and where its processes are. What is under test
is the server's side of it: which sides get an engine, what the game says about them, and that
no engine outlives the game it was started for, however that game ends.
"""

from collections.abc import Iterator
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient
from stockfish_helpers import fake_engine, needs_stockfish, running, started, wait_until_gone

from chess_ai.config import Config, PathsConfig, ServerConfig, StockfishConfig
from chess_ai.web import create_app

GAME = "/chess/api/game"
RANDOM = {"kind": "random"}
HUMAN = {"kind": "human"}

ONLY_MOVE_ENDS_IT = "k7/8/8/8/8/8/1q6/K7 w - - 0 1"
"""White's only move takes the queen, which leaves two bare kings: a draw on the first move."""


def stockfish(elo: int = 1500, **settings) -> dict:
    return {"kind": "stockfish", "elo": elo, "move_time": 0.01, **settings}


def serve(tmp_path: Path, path: Path | str) -> TestClient:
    config = Config(
        server=ServerConfig(path_prefix="/chess"),
        paths=PathsConfig(games=tmp_path / "games", runs=tmp_path / "runs"),
        stockfish=StockfishConfig(path=str(path)),
    )
    return TestClient(create_app(config, static_dir=tmp_path / "static"))


@pytest.fixture
def engine(tmp_path) -> Path:
    return fake_engine(tmp_path / "engine")


@pytest.fixture
def client(tmp_path, engine) -> Iterator[TestClient]:
    with serve(tmp_path, engine) as serving:
        yield serving


def start(client: TestClient, white: dict, black: dict, **changes) -> dict:
    response = client.post(GAME, json={"white": white, "black": black, "move_delay": 0, **changes})
    assert response.status_code == 200, response.text
    return response.json()


def test_stockfish_can_play_white_and_says_how_strong_it_plays(client, engine):
    game = start(client, stockfish(1700, move_time=0.25), HUMAN)

    assert game["white"]["name"] == "Stockfish 1700"
    assert game["white"]["accepts_moves"] is False
    assert game["white"]["stockfish"] == {
        "elo": 1700,
        "requested_elo": 1700,
        "min_elo": 1320,
        "max_elo": 3190,
        "move_time": 0.25,
    }
    assert game["black"]["stockfish"] is None
    assert len(started(engine)) == 1


def test_stockfish_can_play_black_and_moves_when_it_is_its_turn(client, engine):
    game = start(client, HUMAN, stockfish())

    with client.websocket_connect(f"{GAME}/ws") as websocket:
        assert websocket.receive_json()["type"] == "state"
        websocket.send_json({"type": "move", "game": game["id"], "uci": "e2e4"})
        assert websocket.receive_json()["move"]["uci"] == "e2e4"
        reply = websocket.receive_json()

    assert reply["type"] == "move"
    board = chess.Board()
    board.push_uci("e2e4")
    assert board.is_legal(chess.Move.from_uci(reply["move"]["uci"]))


def test_an_elo_out_of_range_is_played_at_the_nearest_level_and_the_game_says_so(client):
    game = start(client, stockfish(800), RANDOM)

    assert game["white"]["name"] == "Stockfish 1320"
    assert game["white"]["stockfish"]["elo"] == 1320
    assert game["white"]["stockfish"]["requested_elo"] == 800


def test_two_stockfish_sides_get_an_engine_each_at_their_own_strength(client, engine):
    game = start(client, stockfish(1400), stockfish(2400))

    assert (game["white"]["name"], game["black"]["name"]) == ("Stockfish 1400", "Stockfish 2400")
    assert len(started(engine)) == 2


def test_a_game_that_ends_on_the_board_takes_its_engine_with_it(client, engine):
    with client.websocket_connect(f"{GAME}/ws") as websocket:
        assert websocket.receive_json()["type"] == "no_game"
        start(client, stockfish(), RANDOM, fen=ONLY_MOVE_ENDS_IT)
        assert websocket.receive_json()["type"] == "state"
        ended = websocket.receive_json()

    assert ended["position"]["game_over"]["reason"] == "insufficient_material"
    wait_until_gone(*started(engine))


def test_an_aborted_game_takes_its_engines_with_it(client, engine):
    game = start(client, stockfish(move_time=30), stockfish())
    pids = started(engine)
    assert all(running(pid) for pid in pids)

    with client.websocket_connect(f"{GAME}/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "abort", "game": game["id"]})
        assert websocket.receive_json()["type"] == "game_over"

    wait_until_gone(*pids)


def test_a_game_replaced_by_another_takes_its_engine_with_it(client, engine):
    start(client, stockfish(move_time=30), HUMAN)
    [first] = started(engine)

    start(client, stockfish(), HUMAN)

    wait_until_gone(first)
    assert running(started(engine)[1])


def test_the_server_going_down_takes_every_engine_with_it(tmp_path, engine):
    with serve(tmp_path, engine) as client:
        start(client, stockfish(move_time=30), stockfish(move_time=30))
        pids = started(engine)
        assert all(running(pid) for pid in pids)

    # Gone by the time the server has finished going down, not some time after.
    assert not any(running(pid) for pid in pids)


def test_a_missing_stockfish_is_the_servers_fault_and_says_how_to_fix_it(tmp_path):
    with serve(tmp_path, tmp_path / "no-such-stockfish") as client:
        response = client.post(GAME, json={"white": stockfish(), "black": RANDOM})

    assert response.status_code == 500
    assert "not found" in response.json()["detail"]
    assert "[stockfish] path" in response.json()["detail"]


def test_a_refused_game_leaves_the_game_before_it_playing(tmp_path):
    with serve(tmp_path, tmp_path / "no-such-stockfish") as client:
        before = start(client, HUMAN, RANDOM)

        client.post(GAME, json={"white": stockfish(), "black": RANDOM})

        with client.websocket_connect(f"{GAME}/ws") as websocket:
            assert websocket.receive_json()["game"]["id"] == before["id"]


def test_an_engine_started_for_a_game_that_is_then_refused_is_not_left_running(client, engine):
    response = client.post(
        GAME, json={"white": stockfish(), "black": {"kind": "model", "run": "no-such-run"}}
    )

    assert response.status_code == 400
    [pid] = started(engine)
    wait_until_gone(pid)


def test_the_server_says_which_stockfish_it_plays_and_the_strengths_it_supports(
    tmp_path,
):
    engine = fake_engine(tmp_path / "engine", "--elo-range", "1350", "2850")
    with serve(tmp_path, engine) as client:
        response = client.get("/chess/api/stockfish")

    assert response.status_code == 200
    assert response.json() == {"name": "Fake UCI", "min_elo": 1350, "max_elo": 2850}
    # Started to be asked, and not left running afterwards.
    wait_until_gone(*started(engine))


def test_the_server_says_why_there_is_no_stockfish_to_play(tmp_path):
    with serve(tmp_path, tmp_path / "no-such-stockfish") as client:
        response = client.get("/chess/api/stockfish")

    assert response.status_code == 500
    assert "not found" in response.json()["detail"]


@pytest.mark.parametrize(
    "player",
    [
        {"kind": "stockfish"},
        stockfish(-1),
        stockfish(4001),
        stockfish(move_time=0),
        stockfish(move_time=61),
        stockfish(depth=10),
    ],
)
def test_a_stockfish_that_cannot_be_asked_for_is_refused(client, engine, player):
    response = client.post(GAME, json={"white": player, "black": RANDOM})

    assert response.status_code == 422
    assert started(engine) == []


def take_back_while_stockfish_thinks(client: TestClient) -> tuple[dict, dict]:
    """Move, take the move back while Stockfish is thinking about it, and move again.

    Returns what the game says after the takeback, and Stockfish's reply to the second move.
    """
    game = start(client, HUMAN, stockfish(move_time=1))
    with client.websocket_connect(f"{GAME}/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "move", "game": game["id"], "uci": "e2e4"})
        assert websocket.receive_json()["move"]["uci"] == "e2e4"
        websocket.send_json({"type": "takeback", "game": game["id"]})
        taken_back = websocket.receive_json()
        websocket.send_json({"type": "move", "game": game["id"], "uci": "d2d4"})
        assert websocket.receive_json()["move"]["uci"] == "d2d4"
        return taken_back, websocket.receive_json()


def test_stockfish_answers_the_move_on_the_board_after_a_takeback_cut_its_thinking_short(
    client,
):
    taken_back, reply = take_back_while_stockfish_thinks(client)

    assert (taken_back["type"], taken_back["ply"]) == ("takeback", 0)
    assert reply["ply"] == 2
    board = chess.Board()
    board.push_uci("d2d4")
    assert board.is_legal(chess.Move.from_uci(reply["move"]["uci"]))


@needs_stockfish
def test_the_installed_stockfish_also_answers_the_move_on_the_board_after_a_takeback(tmp_path):
    with serve(tmp_path, "stockfish") as client:
        _, reply = take_back_while_stockfish_thinks(client)

    board = chess.Board()
    board.push_uci("d2d4")
    assert board.is_legal(chess.Move.from_uci(reply["move"]["uci"]))


@needs_stockfish
def test_the_installed_stockfish_plays_through_the_server(tmp_path):
    with (
        serve(tmp_path, "stockfish") as client,
        client.websocket_connect(f"{GAME}/ws") as websocket,
    ):
        assert websocket.receive_json()["type"] == "no_game"
        game = start(client, stockfish(), RANDOM)
        assert websocket.receive_json()["type"] == "state"
        moved = websocket.receive_json()

    assert game["white"]["stockfish"]["min_elo"] > 0
    assert chess.Board().is_legal(chess.Move.from_uci(moved["move"]["uci"]))
