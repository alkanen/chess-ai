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
from chess_ai.web.app import ViewerEvent

EVENT = TypeAdapter(ViewerEvent)
RANDOM_GAME = {"white": "random", "black": "random", "move_delay": 0}
HUMAN_GAME = {"white": "human", "black": "random", "move_delay": 0}


def serve(prefix: str, tmp_path) -> TestClient:
    config = Config(server=ServerConfig(path_prefix=prefix))
    return TestClient(create_app(config, static_dir=tmp_path / "static"))


@pytest.fixture
def chess_client(tmp_path) -> Iterator[TestClient]:
    # Entering the client keeps one event loop, in which games play on between requests.
    with serve("/chess", tmp_path) as client:
        yield client


def receive(websocket: WebSocketTestSession) -> ViewerEvent:
    return EVENT.validate_python(websocket.receive_json())


def start_game(chess_client, players=HUMAN_GAME, **changes) -> str:
    """Start a game and return the id a viewer quotes back when acting on it."""
    response = chess_client.post("/chess/api/game", json={**players, **changes})
    assert response.status_code == 200
    return response.json()["id"]


def ask(websocket: WebSocketTestSession, game: str, **message) -> None:
    """Send what a viewer asks of ``game``, which every message has to name."""
    websocket.send_json({**message, "game": game})


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
    assert started["white"] == {"name": "Random mover", "accepts_moves": False}
    assert started["black"] == {"name": "Random mover", "accepts_moves": False}
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
        {"white": "stockfish"},
        {"black": None},
        {"move_delay": -0.1},
        {"move_delay": 10.5},
        {"seed": 1},
    ],
)
def test_new_game_request_is_validated(chess_client, change):
    response = chess_client.post("/chess/api/game", json={**RANDOM_GAME, **change})

    assert response.status_code == 422


def test_a_human_move_submitted_over_the_websocket_is_played(chess_client):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"
        response = chess_client.post("/chess/api/game", json=HUMAN_GAME)
        assert receive(websocket).type == "state"

        ask(websocket, response.json()["id"], type="move", uci="e2e4")
        played, answered = receive(websocket), receive(websocket)

    assert response.json()["white"] == {"name": "Human", "accepts_moves": True}
    assert response.json()["black"] == {"name": "Random mover", "accepts_moves": False}
    assert played.type == "move"
    assert (played.ply, played.move.uci) == (1, "e2e4")
    assert answered.type == "move" and answered.ply == 2


def test_human_against_human_takes_both_sides_moves(chess_client):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"
        game = start_game(chess_client, black="human")
        assert receive(websocket).type == "state"

        ask(websocket, game, type="move", uci="e2e4")
        white = receive(websocket)
        ask(websocket, game, type="move", uci="e7e5")
        black = receive(websocket)

    assert white.type == black.type == "move"
    assert [white.move.san, black.move.san] == ["e4", "e5"]


def test_an_illegal_submission_is_rejected_and_the_game_is_unchanged(chess_client):
    game = start_game(chess_client)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        state = receive(websocket)

        ask(websocket, game, type="move", uci="e2e5")
        rejection = receive(websocket)

        # The game went on waiting for a move, so the next legal one is still the first.
        ask(websocket, game, type="move", uci="e2e4")
        played = receive(websocket)

    assert state.type == "state" and state.game.position.fen == chess.STARTING_FEN
    assert rejection.type == "error"
    assert "not a legal move" in rejection.message
    assert played.type == "move"
    assert (played.ply, played.move.uci) == (1, "e2e4")


def test_a_rejection_reaches_only_the_viewer_who_submitted_it(chess_client):
    game = start_game(chess_client)

    with (
        chess_client.websocket_connect("/chess/api/game/ws") as submitter,
        chess_client.websocket_connect("/chess/api/game/ws") as watcher,
    ):
        assert receive(submitter).type == receive(watcher).type == "state"

        ask(submitter, game, type="move", uci="e2e5")
        assert receive(submitter).type == "error"

        ask(submitter, game, type="move", uci="e2e4")
        seen_by_watcher = receive(watcher)

    assert seen_by_watcher.type == "move"
    assert seen_by_watcher.move.uci == "e2e4"


def test_a_move_submitted_for_a_player_that_plays_its_own_is_refused(chess_client):
    game = start_game(chess_client, RANDOM_GAME, move_delay=10)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="move", uci="e2e4")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "Random mover plays this move"


def test_a_move_submitted_before_any_game_is_refused(chess_client):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"
        ask(websocket, "a game that never was", type="move", uci="e2e4")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "no game is in progress"


@pytest.mark.parametrize(
    "message",
    [
        '{"type": "move"}',
        '{"type": "takeback"}',
        '{"uci": "e2e4"}',
        '{"type": "resign"}',
        '{"type": "resign", "color": "green"}',
        '{"type": "abort", "color": "white"}',
        # Every message has to name the game it is meant for.
        '{"type": "move", "uci": "e2e4"}',
        '{"type": "resign", "color": "white"}',
        '{"type": "abort"}',
        "not json at all",
        "",
    ],
)
def test_a_message_the_server_cannot_read_is_reported_back(chess_client, message):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"
        websocket.send_text(message)
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "the server cannot read that message"


def test_a_resignation_ends_the_game_for_every_viewer(chess_client):
    game = start_game(chess_client)

    with (
        chess_client.websocket_connect("/chess/api/game/ws") as resigner,
        chess_client.websocket_connect("/chess/api/game/ws") as watcher,
    ):
        assert receive(resigner).type == receive(watcher).type == "state"

        ask(resigner, game, type="resign", color="white")
        ended, seen_by_watcher = receive(resigner), receive(watcher)

    assert ended.type == seen_by_watcher.type == "game_over"
    assert ended.position == seen_by_watcher.position
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == (
        "0-1",
        "resignation",
    )


def test_black_resigning_hands_the_game_to_white(chess_client):
    game = start_game(chess_client, black="human")

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="resign", color="black")
        ended = receive(websocket)

    assert ended.type == "game_over"
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == (
        "1-0",
        "resignation",
    )


def test_a_resignation_keeps_the_moves_already_played(chess_client):
    game = start_game(chess_client)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        events = [receive(websocket)]
        ask(websocket, game, type="move", uci="e2e4")
        events += receive_moves(websocket, 2)

        ask(websocket, game, type="resign", color="white")
        events.append(receive(websocket))

    game = replay(events)
    assert [move.san for move in game.moves] == [events[1].move.san, events[2].move.san]
    assert game.position.game_over is not None
    assert game.position.game_over.reason == "resignation"
    assert game.position.legal_moves == {}


def test_an_abort_ends_the_game_with_no_result_for_every_viewer(chess_client):
    game = start_game(chess_client, RANDOM_GAME, move_delay=10)

    with (
        chess_client.websocket_connect("/chess/api/game/ws") as aborter,
        chess_client.websocket_connect("/chess/api/game/ws") as watcher,
    ):
        assert receive(aborter).type == receive(watcher).type == "state"

        ask(aborter, game, type="abort")
        ended, seen_by_watcher = receive(aborter), receive(watcher)

    assert ended.type == seen_by_watcher.type == "game_over"
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == ("*", "abort")


def test_a_reconnecting_viewer_is_told_the_game_was_resigned(chess_client):
    game = start_game(chess_client)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="resign", color="white")
        assert receive(websocket).type == "game_over"
    with chess_client.websocket_connect("/chess/api/game/ws") as rejoined:
        arrived = receive(rejoined)

    assert arrived.type == "state"
    assert arrived.game.position.game_over is not None
    assert arrived.game.position.game_over.reason == "resignation"


def test_a_side_that_plays_its_own_moves_cannot_be_resigned(chess_client):
    game = start_game(chess_client, move_delay=10)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="resign", color="black")
        rejection = receive(websocket)

        # The game is untouched, so it still takes White's move.
        ask(websocket, game, type="move", uci="e2e4")
        played = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "Random mover is not yours to resign"
    assert played.type == "move" and played.move.uci == "e2e4"


@pytest.mark.parametrize("ending", [{"type": "resign", "color": "white"}, {"type": "abort"}])
def test_ending_a_game_that_is_already_over_is_refused(chess_client, ending):
    game = start_game(chess_client)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="abort")
        assert receive(websocket).type == "game_over"

        ask(websocket, game, **ending)
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "the game is over"


@pytest.mark.parametrize("ending", [{"type": "resign", "color": "white"}, {"type": "abort"}])
def test_ending_a_game_before_any_has_started_is_refused(chess_client, ending):
    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"
        ask(websocket, "a game that never was", **ending)
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "no game is in progress"


def test_a_new_game_can_be_started_after_a_resignation(chess_client):
    game = start_game(chess_client)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="resign", color="white")
        assert receive(websocket).type == "game_over"

        chess_client.post("/chess/api/game", json=RANDOM_GAME)
        events = receive_game(websocket)

    assert events[0].type == "state" and events[0].game.moves == []
    assert_legal(events)


@pytest.mark.parametrize(
    "stale",
    [{"type": "move", "uci": "e2e4"}, {"type": "resign", "color": "white"}, {"type": "abort"}],
)
def test_what_a_viewer_asks_of_a_replaced_game_never_reaches_its_replacement(chess_client, stale):
    """A click lands on the game the viewer was shown, or on nothing at all.

    Another viewer can start a new game in the moment between the board being drawn and
    the click arriving, and the replacement is a game this viewer has never seen.
    """
    watched = start_game(chess_client, move_delay=10)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        replacement = start_game(chess_client, move_delay=10)
        ask(websocket, watched, **stale)
        seen = [receive(websocket), receive(websocket)]

        # The replacement is untouched, and still takes what is asked of it by name.
        ask(websocket, replacement, type="move", uci="e2e4")
        played = receive(websocket)

    rejection = next(event for event in seen if event.type == "error")
    assert rejection.message == "that game has been replaced"
    assert next(event for event in seen if event.type == "state").game.moves == []
    assert played.type == "move" and played.move.uci == "e2e4"


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


def test_a_takeback_over_the_websocket_reaches_every_viewer(chess_client):
    """The takeback flow under the prefix: everyone watching goes back together."""
    game = start_game(chess_client, black="human")

    with (
        chess_client.websocket_connect("/chess/api/game/ws") as player,
        chess_client.websocket_connect("/chess/api/game/ws") as watcher,
    ):
        assert receive(player).type == receive(watcher).type == "state"
        ask(player, game, type="move", uci="e2e4")
        assert receive(player).type == receive(watcher).type == "move"

        ask(player, game, type="takeback")
        taken_back, seen_by_watcher = receive(player), receive(watcher)

        # Both are looking at the starting position again, and White is on move.
        ask(watcher, game, type="move", uci="d2d4")
        played = receive(player)

    assert taken_back.type == seen_by_watcher.type == "takeback"
    assert taken_back == seen_by_watcher
    assert taken_back.ply == 0
    assert taken_back.position.fen == chess.STARTING_FEN
    assert played.type == "move" and played.move.san == "d4"


def test_a_takeback_against_a_player_that_moves_for_itself_undoes_both_moves(chess_client):
    game = start_game(chess_client, move_delay=0)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, game, type="move", uci="e2e4")
        played = receive_moves(websocket, 2)

        ask(websocket, game, type="takeback")
        taken_back = receive(websocket)

    assert [event.ply for event in played] == [1, 2]
    assert taken_back.type == "takeback"
    assert taken_back.ply == 0
    assert taken_back.position.fen == chess.STARTING_FEN


def test_a_takeback_of_a_game_nobody_plays_by_hand_is_refused(chess_client):
    game = start_game(chess_client, players=RANDOM_GAME, move_delay=0.2)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        assert receive(websocket).type == "move"

        ask(websocket, game, type="takeback")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "neither side is played by hand"


def test_a_takeback_before_a_move_has_been_played_is_refused(chess_client):
    game = start_game(chess_client)

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"

        ask(websocket, game, type="takeback")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "no move has been played yet"


def test_a_takeback_reaches_only_the_game_the_viewer_named(chess_client):
    """A new game in the moment before the click must not be taken back instead."""
    stale = start_game(chess_client, black="human")

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, stale, type="move", uci="e2e4")
        assert receive(websocket).type == "move"
        replacement = start_game(chess_client, black="human")
        assert receive(websocket).type == "state"

        ask(websocket, stale, type="takeback")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "that game has been replaced"
    assert replacement != stale


def test_a_game_starts_from_a_fen_and_says_so(chess_client):
    # Black to move in the Scandinavian, with the queen ready to take on d5.
    fen = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "no_game"
        response = chess_client.post(
            "/chess/api/game", json={"white": "random", "black": "human", "fen": fen}
        )
        state = receive(websocket)

        ask(websocket, response.json()["id"], type="move", uci="d8d5")
        played = receive(websocket)

    assert response.status_code == 200
    assert response.json()["start_fen"] == fen
    assert response.json()["position"]["fen"] == fen
    assert state.type == "state"
    assert state.game.start_fen == fen
    assert state.game.position.turn == "black"
    assert played.type == "move" and played.move.san == "Qxd5"


@pytest.mark.parametrize(
    ("fen", "problem"),
    [
        ("rubbish", "not a FEN"),
        ("8/8/8/8/8/8/8/8 w - - 0 1", "no pieces on the board"),
        ("4k3/8/8/8/8/8/8/8 w - - 0 1", "White has no king"),
    ],
)
def test_a_fen_that_cannot_be_played_from_is_refused_with_a_reason(chess_client, fen, problem):
    response = chess_client.post("/chess/api/game", json={**HUMAN_GAME, "fen": fen})

    assert response.status_code == 400
    assert problem in response.json()["detail"]


def test_a_refused_fen_leaves_the_game_that_is_being_played(chess_client):
    game = start_game(chess_client, black="human")

    with chess_client.websocket_connect("/chess/api/game/ws") as websocket:
        assert receive(websocket).type == "state"
        refused = chess_client.post("/chess/api/game", json={**HUMAN_GAME, "fen": "rubbish"})

        # The game the viewer is watching is still the one they can move in.
        ask(websocket, game, type="move", uci="e2e4")
        played = receive(websocket)

    assert refused.status_code == 400
    assert played.type == "move" and played.move.uci == "e2e4"


def test_a_game_started_without_a_fen_starts_where_games_start(chess_client):
    response = chess_client.post("/chess/api/game", json={**HUMAN_GAME, "fen": None})

    assert response.status_code == 200
    assert response.json()["start_fen"] == chess.STARTING_FEN
