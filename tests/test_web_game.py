import asyncio
import json
from collections.abc import Iterator

import chess
import httpx2
import pytest
from fastapi.testclient import TestClient
from game_helpers import replay
from pydantic import TypeAdapter
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from chess_ai.config import Config, ServerConfig
from chess_ai.game_session import GameEvent, MoveEvent
from chess_ai.web import create_app
from chess_ai.web.game_channel import ChannelEvent

EVENT = TypeAdapter(ChannelEvent)
RANDOM_GAME = {"white": "random", "black": "random", "move_delay": 0}


def serve(prefix: str, tmp_path) -> TestClient:
    config = Config(server=ServerConfig(path_prefix=prefix))
    return TestClient(create_app(config, static_dir=tmp_path / "static"))


@pytest.fixture
def chess_client(tmp_path) -> Iterator[TestClient]:
    # Entering the client keeps one event loop, in which games play on between requests.
    with serve("/chess", tmp_path) as client:
        yield client


def receive(websocket: WebSocketTestSession) -> ChannelEvent:
    return EVENT.validate_python(websocket.receive_json())


def receive_game(websocket: WebSocketTestSession) -> list[GameEvent]:
    """Receive a game's events, from its state until a move ends the game."""
    first = receive(websocket)
    assert first.type == "state"
    events: list[GameEvent] = [first]
    while replay(events).position.game_over is None:
        event = receive(websocket)
        assert event.type == "move"
        events.append(event)
    return events


def receive_moves(websocket: WebSocketTestSession, count: int) -> list[MoveEvent]:
    events = [receive(websocket) for _ in range(count)]
    assert all(isinstance(event, MoveEvent) for event in events)
    return events


def assert_legal(events: list[GameEvent]) -> None:
    game = replay(events)
    board = chess.Board()
    for record in game.moves:
        board.push_uci(record.uci)
    assert board.fen() == game.position.fen


def test_before_any_game_viewers_see_the_starting_position(chess_client):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        event = receive(websocket)

    assert event.type == "no_game"
    assert event.position.fen == chess.STARTING_FEN


def test_random_game_streams_live_to_the_end(chess_client):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"

        response = chess_client.post("/chess/api/game", json=RANDOM_GAME)
        events = receive_game(websocket)

    assert response.status_code == 200
    started = response.json()
    assert (started["white"], started["black"]) == ("Random mover", "Random mover")
    assert started["position"]["fen"] == chess.STARTING_FEN
    assert events[0].type == "state"
    assert len(events[0].game.moves) < len(replay(events).moves)
    assert_legal(events)


def test_viewers_of_the_same_game_see_the_same_state(chess_client):
    with (
        chess_client.websocket_connect("/chess/api/game/ws") as first,
        chess_client.websocket_connect("/chess/api/game/ws") as second,
    ):
        assert receive(first).type == receive(second).type == "no_game"

        chess_client.post("/chess/api/game", json=RANDOM_GAME)
        first_events = receive_game(first)
        second_events = receive_game(second)

    assert replay(first_events) == replay(second_events)
    assert_legal(first_events)


def test_reconnecting_viewer_gets_the_full_state_including_history(chess_client):
    chess_client.post("/chess/api/game", json={**RANDOM_GAME, "move_delay": 0.02})

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        seen = [receive(websocket), *receive_moves(websocket, 2)]
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        rejoined = [receive(websocket), *receive_moves(websocket, 2)]

    before, after = replay(seen), replay(rejoined)
    assert rejoined[0].type == "state"
    assert rejoined[0].game.moves[: len(before.moves)] == before.moves
    assert len(after.moves) > len(before.moves)
    assert after.position.game_over is None
    assert_legal(rejoined)


def test_new_game_replaces_the_current_one_for_every_viewer(chess_client):
    chess_client.post("/chess/api/game", json={**RANDOM_GAME, "move_delay": 0.02})

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        first_game = [receive(websocket), *receive_moves(websocket, 2)]

        chess_client.post("/chess/api/game", json={**RANDOM_GAME, "move_delay": 0.02})
        while (event := receive(websocket)).type == "move":
            first_game.append(event)
        second_game = [event, *receive_moves(websocket, 2)]

    assert_legal(first_game)
    assert event.type == "state"
    assert event.game.moves == []
    assert_legal(second_game)


@pytest.mark.parametrize(
    "change",
    [
        {"white": "human"},
        {"black": None},
        {"move_delay": -0.1},
        {"move_delay": 10.5},
        {"seed": 1},
    ],
)
def test_new_game_request_is_validated(chess_client, change):
    response = chess_client.post("/chess/api/game", json={**RANDOM_GAME, **change})

    assert response.status_code == 422


def test_game_routes_are_not_served_outside_the_prefix(chess_client):
    assert chess_client.post("/api/game", json=RANDOM_GAME).status_code == 404
    with pytest.raises(WebSocketDisconnect), chess_client.websocket_connect("/api/game/ws"):
        pass


def test_game_websocket_with_empty_prefix(tmp_path):
    with serve("", tmp_path) as client, client.websocket_connect("/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"


@pytest.mark.anyio
async def test_server_shutdown_disconnects_viewers(tmp_path):
    """Viewers are let go by the app itself, not only by the ASGI server closing sockets."""
    app = create_app(Config(server=ServerConfig(path_prefix="/chess")), tmp_path / "static")
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/chess/api/game/ws",
        "raw_path": b"/chess/api/game/ws",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "subprotocols": [],
    }
    to_app: asyncio.Queue[dict] = asyncio.Queue()
    from_app: asyncio.Queue[dict] = asyncio.Queue()
    await to_app.put({"type": "websocket.connect"})

    async with asyncio.timeout(5):
        async with app.router.lifespan_context(app):
            viewer = asyncio.create_task(app(scope, to_app.get, from_app.put))
            assert (await from_app.get())["type"] == "websocket.accept"
            assert json.loads((await from_app.get())["text"])["type"] == "no_game"

        closing = await from_app.get()
        await viewer

    assert (closing["type"], closing["code"]) == ("websocket.close", 1001)


@pytest.mark.anyio
async def test_starting_a_game_while_shutting_down_is_refused(tmp_path):
    app = create_app(Config(server=ServerConfig(path_prefix="/chess")), tmp_path / "static")
    async with app.router.lifespan_context(app):
        pass  # The server starts and shuts down again, closing the game channel.

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post("/chess/api/game", json=RANDOM_GAME)

    assert response.status_code == 503
    assert "shutting down" in response.json()["detail"]
