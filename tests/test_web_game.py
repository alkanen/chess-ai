import asyncio
import json
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import httpx2
import pytest
from fastapi.testclient import TestClient
from game_helpers import replay
from pgn_helpers import replayed
from pydantic import TypeAdapter
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from chess_ai.config import Config, GamesConfig, PathsConfig, ServerConfig
from chess_ai.game_session import GameEvent, MoveEvent
from chess_ai.pgn import game_pgn
from chess_ai.web import app as app_module
from chess_ai.web import create_app
from chess_ai.web.app import ViewerEvent

EVENT = TypeAdapter(ViewerEvent)
RANDOM = {"kind": "random"}
HUMAN = {"kind": "human"}
RANDOM_GAME = {"white": RANDOM, "black": RANDOM, "move_delay": 0}
HUMAN_GAME = {"white": HUMAN, "black": RANDOM, "move_delay": 0}


PGN_FILE = re.compile(r'attachment; filename="\d{8}-\d{6}-[0-9a-f]{8}\.pgn"')
"""What the server calls the file it hands the browser: when, and which game."""


def serve(prefix: str, tmp_path, **games) -> TestClient:
    config = Config(
        server=ServerConfig(path_prefix=prefix),
        paths=PathsConfig(games=tmp_path / "games"),
        games=GamesConfig(**games),
    )
    return TestClient(create_app(config, static_dir=tmp_path / "static"))


def saved_games(tmp_path) -> list[Path]:
    """The games the server has saved, in the order it played them."""
    return sorted((tmp_path / "games").glob("*.pgn"))


@pytest.fixture
def chess_client(tmp_path) -> Iterator[TestClient]:
    # Entering the client keeps one event loop, in which games play on between requests.
    with serve("/chess", tmp_path) as client:
        yield client


def receive(websocket: WebSocketTestSession) -> ViewerEvent:
    return EVENT.validate_python(websocket.receive_json())


def start(chess_client, players=HUMAN_GAME, prefix="/chess", **changes) -> dict:
    """Start a game, and return what the server answers: the game and its links."""
    response = chess_client.post(f"{prefix}/api/games", json={**players, **changes})
    assert response.status_code == 200, response.text
    return response.json()


def start_game(chess_client, players=HUMAN_GAME, **changes) -> dict:
    """Start a game and return its links."""
    return start(chess_client, players, **changes)["links"]


def connect(chess_client, link: str, prefix="/chess"):
    return chess_client.websocket_connect(f"{prefix}/api/games/{link}/ws")


def ask(websocket: WebSocketTestSession, **message) -> None:
    """Send what a viewer asks of the game their link reaches."""
    websocket.send_json(message)


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


def test_a_game_is_started_with_a_watch_link_and_a_play_link_for_the_person(chess_client):
    started = start(chess_client)

    links = started["links"]
    assert links["white"] is not None
    assert links["black"] is None
    assert links["control"] is None
    assert links["watch"] not in (links["white"], started["game"]["id"])
    assert started["game"]["position"]["fen"] == chess.STARTING_FEN


def test_a_game_between_two_people_has_a_play_link_for_each(chess_client):
    links = start_game(chess_client, black=HUMAN)

    assert links["white"] is not None and links["black"] is not None
    assert len({links["white"], links["black"], links["watch"]}) == 3


def test_a_link_says_what_it_may_do(chess_client):
    links = start_game(chess_client, black=HUMAN)

    seen = {
        side: chess_client.get(f"/chess/api/games/{links[side]}").json()
        for side in ("white", "black", "watch")
    }

    assert {side: view["access"] for side, view in seen.items()} == {
        "white": "white",
        "black": "black",
        "watch": "watch",
    }
    assert {view["watch"] for view in seen.values()} == {links["watch"]}
    assert seen["watch"]["game"]["moves"] == []
    assert seen["watch"]["updated"]


def test_random_game_streams_live_to_the_end(chess_client):
    started = start(chess_client, RANDOM_GAME, move_delay=0.01)

    with connect(chess_client, started["links"]["watch"]) as websocket:
        events = receive_game(websocket)

    mover = {"name": "Random mover", "accepts_moves": False, "model": None, "stockfish": None}
    assert (started["game"]["white"], started["game"]["black"]) == (mover, mover)
    assert started["links"]["control"] is not None
    assert events[0].type == "state"
    assert events[0].access == "watch"
    assert len(events[0].game.moves) < len(replay(events).moves)
    assert_legal(events)


def test_viewers_of_the_same_game_see_the_same_state(chess_client):
    links = start_game(chess_client, RANDOM_GAME, move_delay=0.01)

    with (
        connect(chess_client, links["watch"]) as first,
        connect(chess_client, links["control"]) as second,
    ):
        first_events = receive_game(first)
        second_events = receive_game(second)

    assert replay(first_events) == replay(second_events)
    assert_legal(first_events)


def test_reconnecting_viewer_gets_the_full_state_including_history(chess_client):
    links = start_game(chess_client, RANDOM_GAME, move_delay=0.02)

    with connect(chess_client, links["watch"]) as websocket:
        seen = [receive(websocket), *receive_moves(websocket, 2)]
    with connect(chess_client, links["watch"]) as websocket:
        rejoined = [receive(websocket), *receive_moves(websocket, 2)]

    before, after = replay(seen), replay(rejoined)
    assert rejoined[0].type == "state"
    assert rejoined[0].game.moves[: len(before.moves)] == before.moves
    assert len(after.moves) > len(before.moves)
    assert after.position.game_over is None
    assert_legal(rejoined)


def test_two_games_are_played_at_once_and_neither_affects_the_other(chess_client):
    first = start_game(chess_client, move_delay=10)
    second = start_game(chess_client, move_delay=10)

    with (
        connect(chess_client, first["white"]) as one,
        connect(chess_client, second["white"]) as other,
    ):
        assert receive(one).type == receive(other).type == "state"
        ask(one, type="move", uci="e2e4")
        ask(other, type="move", uci="d2d4")
        played_here, played_there = receive(one), receive(other)
        ask(other, type="resign")
        assert receive(other).type == "game_over"

    assert played_here.type == played_there.type == "move"
    assert (played_here.move.san, played_there.move.san) == ("e4", "d4")
    still_playing = chess_client.get(f"/chess/api/games/{first['watch']}").json()["game"]
    assert [move["san"] for move in still_playing["moves"]] == ["e4"]
    assert still_playing["position"]["game_over"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"white": {"kind": "stockfish"}},
        {"white": "random"},
        {"black": None},
        {"move_delay": -0.1},
        {"move_delay": 10.5},
        {"seed": 1},
    ],
)
def test_new_game_request_is_validated(chess_client, change):
    response = chess_client.post("/chess/api/games", json={**RANDOM_GAME, **change})

    assert response.status_code == 422


def test_a_human_move_submitted_over_the_websocket_is_played(chess_client):
    started = start(chess_client)

    with connect(chess_client, started["links"]["white"]) as websocket:
        state = receive(websocket)
        ask(websocket, type="move", uci="e2e4")
        played, answered = receive(websocket), receive(websocket)

    assert started["game"]["white"] == {
        "name": "Human",
        "accepts_moves": True,
        "model": None,
        "stockfish": None,
    }
    assert started["game"]["black"] == {
        "name": "Random mover",
        "accepts_moves": False,
        "model": None,
        "stockfish": None,
    }
    assert (state.type, state.access) == ("state", "white")
    assert played.type == "move"
    assert (played.ply, played.move.uci) == (1, "e2e4")
    assert answered.type == "move" and answered.ply == 2


def test_two_people_each_move_their_own_side_through_their_own_link(chess_client):
    links = start_game(chess_client, black=HUMAN)

    with (
        connect(chess_client, links["white"]) as white,
        connect(chess_client, links["black"]) as black,
    ):
        assert receive(white).type == receive(black).type == "state"

        ask(black, type="move", uci="e7e5")
        refused = receive(black)
        ask(white, type="move", uci="e2e4")
        assert receive(white).type == receive(black).type == "move"
        ask(white, type="move", uci="d2d4")
        refused_too = receive(white)
        ask(black, type="move", uci="e7e5")
        played = receive(white)

    assert refused.type == refused_too.type == "error"
    assert refused.message == refused_too.message == "it is not your move"
    assert played.type == "move" and played.move.san == "e5"


def test_an_illegal_submission_is_rejected_and_the_game_is_unchanged(chess_client):
    links = start_game(chess_client)

    with connect(chess_client, links["white"]) as websocket:
        state = receive(websocket)

        ask(websocket, type="move", uci="e2e5")
        rejection = receive(websocket)

        # The game went on waiting for a move, so the next legal one is still the first.
        ask(websocket, type="move", uci="e2e4")
        played = receive(websocket)

    assert state.type == "state" and state.game.position.fen == chess.STARTING_FEN
    assert rejection.type == "error"
    assert "not a legal move" in rejection.message
    assert played.type == "move"
    assert (played.ply, played.move.uci) == (1, "e2e4")


def test_a_rejection_reaches_only_the_viewer_who_submitted_it(chess_client):
    links = start_game(chess_client)

    with (
        connect(chess_client, links["white"]) as submitter,
        connect(chess_client, links["watch"]) as watcher,
    ):
        assert receive(submitter).type == receive(watcher).type == "state"

        ask(submitter, type="move", uci="e2e5")
        assert receive(submitter).type == "error"

        ask(submitter, type="move", uci="e2e4")
        seen_by_watcher = receive(watcher)

    assert seen_by_watcher.type == "move"
    assert seen_by_watcher.move.uci == "e2e4"


def test_a_move_submitted_on_the_other_sides_turn_is_refused(chess_client):
    links = start_game(chess_client, move_delay=10)

    with connect(chess_client, links["white"]) as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, type="move", uci="e2e4")
        assert receive(websocket).type == "move"
        ask(websocket, type="move", uci="d2d4")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "it is not your move"


@pytest.mark.parametrize(
    "action",
    [
        {"type": "move", "uci": "e2e4"},
        {"type": "resign"},
        {"type": "abort"},
        {"type": "takeback"},
        {"type": "answer", "request": 1, "accept": True},
    ],
)
def test_a_watch_link_can_only_watch(chess_client, action):
    links = start_game(chess_client, move_delay=10)

    with (
        connect(chess_client, links["watch"]) as watcher,
        connect(chess_client, links["white"]) as player,
    ):
        assert receive(watcher).type == receive(player).type == "state"
        ask(watcher, **action)
        rejection = receive(watcher)
        # The game is untouched, so it still takes White's first move.
        ask(player, type="move", uci="e2e4")
        played = receive(player)

    assert rejection.type == "error"
    assert rejection.message.startswith("you are watching this game")
    assert played.type == "move" and played.ply == 1


def test_the_control_link_of_a_game_nobody_plays_by_hand_can_only_abort(chess_client):
    links = start_game(chess_client, RANDOM_GAME, move_delay=10)

    with connect(chess_client, links["control"]) as websocket:
        state = receive(websocket)
        ask(websocket, type="move", uci="e2e4")
        rejection = receive(websocket)
        ask(websocket, type="abort")
        ended = receive(websocket)

    assert state.access == "control"
    assert rejection.type == "error"
    assert rejection.message == "nobody plays this game by hand, so nobody can move"
    assert ended.type == "game_over"


@pytest.mark.parametrize(
    "message",
    [
        '{"type": "move"}',
        '{"uci": "e2e4"}',
        '{"type": "answer"}',
        # Which game and which side are the link's to say, not the message's.
        '{"type": "move", "uci": "e2e4", "game": "abc"}',
        '{"type": "resign", "color": "white"}',
        "not json at all",
        "",
    ],
)
def test_a_message_the_server_cannot_read_is_reported_back(chess_client, message):
    links = start_game(chess_client)

    with connect(chess_client, links["white"]) as websocket:
        assert receive(websocket).type == "state"
        websocket.send_text(message)
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "the server cannot read that message"


def test_a_resignation_ends_the_game_for_every_viewer(chess_client):
    links = start_game(chess_client)

    with (
        connect(chess_client, links["white"]) as resigner,
        connect(chess_client, links["watch"]) as watcher,
    ):
        assert receive(resigner).type == receive(watcher).type == "state"

        ask(resigner, type="resign")
        ended, seen_by_watcher = receive(resigner), receive(watcher)

    assert ended.type == seen_by_watcher.type == "game_over"
    assert ended.position == seen_by_watcher.position
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == (
        "0-1",
        "resignation",
    )


def test_black_resigning_hands_the_game_to_white(chess_client):
    links = start_game(chess_client, black=HUMAN)

    with connect(chess_client, links["black"]) as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, type="resign")
        ended = receive(websocket)

    assert ended.type == "game_over"
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == (
        "1-0",
        "resignation",
    )


def test_a_resignation_keeps_the_moves_already_played(chess_client):
    links = start_game(chess_client)

    with connect(chess_client, links["white"]) as websocket:
        events = [receive(websocket)]
        ask(websocket, type="move", uci="e2e4")
        events += receive_moves(websocket, 2)

        ask(websocket, type="resign")
        events.append(receive(websocket))

    game = replay(events)
    assert [move.san for move in game.moves] == [events[1].move.san, events[2].move.san]
    assert game.position.game_over is not None
    assert game.position.game_over.reason == "resignation"
    assert game.position.legal_moves == {}


def test_an_abort_ends_the_game_with_no_result_for_every_viewer(chess_client):
    links = start_game(chess_client, move_delay=10)

    with (
        connect(chess_client, links["white"]) as aborter,
        connect(chess_client, links["watch"]) as watcher,
    ):
        assert receive(aborter).type == receive(watcher).type == "state"

        ask(aborter, type="abort")
        ended, seen_by_watcher = receive(aborter), receive(watcher)

    assert ended.type == seen_by_watcher.type == "game_over"
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == ("*", "abort")


def test_an_aborted_game_is_gone_and_its_links_with_it(chess_client):
    links = start_game(chess_client, move_delay=10)

    with connect(chess_client, links["white"]) as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, type="abort")
        assert receive(websocket).type == "game_over"

    for link in (links["white"], links["watch"]):
        assert chess_client.get(f"/chess/api/games/{link}").status_code == 404
        assert chess_client.get(f"/chess/api/games/{link}/pgn").status_code == 404
        with connect(chess_client, link) as websocket:
            refused = receive(websocket)
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
        assert refused.type == "error"
        assert "no such game" in refused.message
        assert closed.value.code == 1008


def test_a_game_that_has_ended_is_still_there_to_look_at(chess_client):
    links = start_game(chess_client)

    with connect(chess_client, links["white"]) as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, type="resign")
        assert receive(websocket).type == "game_over"
    with connect(chess_client, links["watch"]) as rejoined:
        arrived = receive(rejoined)
        # There is nothing more to come, so the connection is let go of.
        with pytest.raises(WebSocketDisconnect) as closed:
            rejoined.receive_json()

    assert arrived.type == "state"
    assert arrived.game.position.game_over is not None
    assert arrived.game.position.game_over.reason == "resignation"
    assert closed.value.code == 1000


def test_a_link_that_reaches_no_game_is_told_so(chess_client):
    assert chess_client.get("/chess/api/games/nothing-here").status_code == 404
    assert chess_client.get("/chess/api/games/nothing-here/pgn").status_code == 404
    with connect(chess_client, "nothing-here") as websocket:
        refused = receive(websocket)

    assert refused.type == "error"
    assert "no such game" in refused.message


def test_a_new_game_beyond_the_limit_is_refused_with_a_reason(tmp_path):
    with serve("/chess", tmp_path, max_ongoing=2) as client:
        start_game(client, move_delay=10)
        resigning = start_game(client, move_delay=10)

        refused = client.post("/chess/api/games", json=HUMAN_GAME)
        with connect(client, resigning["white"]) as websocket:
            assert receive(websocket).type == "state"
            ask(websocket, type="resign")
            assert receive(websocket).type == "game_over"
        let_in = client.post("/chess/api/games", json=HUMAN_GAME)

    assert refused.status_code == 409
    assert "2 games are already being played" in refused.json()["detail"]
    assert let_in.status_code == 200


def test_game_routes_are_not_served_outside_the_prefix(chess_client):
    links = start_game(chess_client)

    assert chess_client.post("/api/games", json=RANDOM_GAME).status_code == 404
    assert chess_client.get(f"/api/games/{links['watch']}/pgn").status_code == 404
    with pytest.raises(WebSocketDisconnect), connect(chess_client, links["watch"], prefix=""):
        pass


def test_game_websocket_with_empty_prefix(tmp_path):
    with serve("", tmp_path) as client:
        links = start(client, prefix="")["links"]
        with connect(client, links["watch"], prefix="") as websocket:
            assert receive(websocket).type == "state"


@pytest.mark.anyio
async def test_server_shutdown_disconnects_viewers(tmp_path):
    """Viewers are let go by the app itself, not only by the ASGI server closing sockets."""
    app = create_app(Config(server=ServerConfig(path_prefix="/chess")), tmp_path / "static")
    to_app: asyncio.Queue[dict] = asyncio.Queue()
    from_app: asyncio.Queue[dict] = asyncio.Queue()
    await to_app.put({"type": "websocket.connect"})

    async with asyncio.timeout(5):
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app), base_url="http://testserver"
            ) as client:
                started = await client.post(
                    "/chess/api/games", json={**HUMAN_GAME, "move_delay": 10}
                )
            path = f"/chess/api/games/{started.json()['links']['watch']}/ws"
            scope = {
                "type": "websocket",
                "asgi": {"version": "3.0"},
                "scheme": "ws",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "subprotocols": [],
            }
            viewer = asyncio.create_task(app(scope, to_app.get, from_app.put))
            assert (await from_app.get())["type"] == "websocket.accept"
            assert json.loads((await from_app.get())["text"])["type"] == "state"

        closing = await from_app.get()
        await viewer

    # Going away rather than done: the game is not over, and the browser should come back.
    assert (closing["type"], closing["code"]) == ("websocket.close", 1001)


@pytest.mark.anyio
async def test_starting_a_game_while_shutting_down_is_refused(tmp_path):
    app = create_app(Config(server=ServerConfig(path_prefix="/chess")), tmp_path / "static")
    async with app.router.lifespan_context(app):
        pass  # The server starts and shuts down again, closing the games.

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app), base_url="http://testserver"
    ) as client:
        response = await client.post("/chess/api/games", json=RANDOM_GAME)

    assert response.status_code == 503
    assert "shutting down" in response.json()["detail"]


def test_a_takeback_against_a_player_that_moves_for_itself_undoes_both_moves(chess_client):
    links = start_game(chess_client, move_delay=0)

    with (
        connect(chess_client, links["white"]) as websocket,
        connect(chess_client, links["watch"]) as watcher,
    ):
        assert receive(websocket).type == receive(watcher).type == "state"
        ask(websocket, type="move", uci="e2e4")
        played = receive_moves(websocket, 2)
        receive_moves(watcher, 2)

        ask(websocket, type="takeback")
        taken_back, seen_by_watcher = receive(websocket), receive(watcher)

    assert [event.ply for event in played] == [1, 2]
    assert taken_back.type == "takeback"
    assert taken_back == seen_by_watcher
    assert taken_back.ply == 0
    assert taken_back.position.fen == chess.STARTING_FEN


def test_between_two_people_a_takeback_is_asked_for_and_the_other_agrees(chess_client):
    links = start_game(chess_client, black=HUMAN)

    with (
        connect(chess_client, links["white"]) as white,
        connect(chess_client, links["black"]) as black,
    ):
        assert receive(white).type == receive(black).type == "state"
        ask(white, type="move", uci="e2e4")
        assert receive(white).type == receive(black).type == "move"

        ask(white, type="takeback")
        asked, seen_by_black = receive(white), receive(black)
        ask(black, type="answer", request=asked.request.id, accept=True)
        taken_back = receive(white)
        assert receive(black) == taken_back

        # White is on move again, and plays something else.
        ask(white, type="move", uci="d2d4")
        played = receive(black)

    assert asked.type == "request" and asked == seen_by_black
    assert asked.request is not None
    assert (asked.request.kind, asked.request.by) == ("takeback", "white")
    assert taken_back.type == "takeback" and taken_back.ply == 0
    assert played.type == "move" and played.move.san == "d4"


def test_between_two_people_a_declined_takeback_leaves_the_game(chess_client):
    links = start_game(chess_client, black=HUMAN)

    with (
        connect(chess_client, links["white"]) as white,
        connect(chess_client, links["black"]) as black,
    ):
        assert receive(white).type == receive(black).type == "state"
        ask(white, type="move", uci="e2e4")
        assert receive(white).type == receive(black).type == "move"
        ask(white, type="takeback")
        asked = receive(white)
        assert receive(black) == asked

        ask(black, type="answer", request=asked.request.id, accept=False)
        declined = receive(white)
        assert receive(black) == declined
        ask(black, type="move", uci="e7e5")
        played = receive(white)

    assert declined.type == "request" and declined.request is None
    assert played.type == "move" and played.ply == 2


def test_between_two_people_who_have_both_moved_an_abort_is_asked_for(chess_client, tmp_path):
    links = start_game(chess_client, black=HUMAN)

    with (
        connect(chess_client, links["white"]) as white,
        connect(chess_client, links["black"]) as black,
    ):
        assert receive(white).type == receive(black).type == "state"
        for side, uci in ((white, "e2e4"), (black, "e7e5")):
            ask(side, type="move", uci=uci)
            assert receive(white).type == receive(black).type == "move"

        ask(black, type="abort")
        asked = receive(white)
        assert receive(black) == asked
        ask(white, type="answer", request=asked.request.id, accept=True)
        ended = receive(black)

    assert asked.type == "request"
    assert asked.request is not None and asked.request.kind == "abort"
    assert ended.type == "game_over"
    assert chess_client.get(f"/chess/api/games/{links['watch']}").status_code == 404
    assert saved_games(tmp_path) == []


def test_a_takeback_before_a_move_has_been_played_is_refused(chess_client):
    links = start_game(chess_client)

    with connect(chess_client, links["white"]) as websocket:
        assert receive(websocket).type == "state"

        ask(websocket, type="takeback")
        rejection = receive(websocket)

    assert rejection.type == "error"
    assert rejection.message == "no move has been played yet"


def test_a_game_starts_from_a_fen_and_says_so(chess_client):
    # Black to move in the Scandinavian, with the queen ready to take on d5.
    fen = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"

    started = start(chess_client, {"white": RANDOM, "black": HUMAN}, fen=fen)
    with connect(chess_client, started["links"]["black"]) as websocket:
        state = receive(websocket)
        ask(websocket, type="move", uci="d8d5")
        played = receive(websocket)

    assert started["game"]["start_fen"] == fen
    assert started["game"]["position"]["fen"] == fen
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
    response = chess_client.post("/chess/api/games", json={**HUMAN_GAME, "fen": fen})

    assert response.status_code == 400
    assert problem in response.json()["detail"]


def test_a_game_started_without_a_fen_starts_where_games_start(chess_client):
    started = start(chess_client, fen=None)

    assert started["game"]["start_fen"] == chess.STARTING_FEN


def test_a_game_in_progress_is_downloaded_as_far_as_it_has_been_played(chess_client):
    links = start_game(chess_client)
    with connect(chess_client, links["white"]) as websocket:
        state = receive(websocket)
        ask(websocket, type="move", uci="e2e4")
        played = receive_moves(websocket, 2)

        response = chess_client.get(f"/chess/api/games/{links['white']}/pgn")

    assert response.status_code == 200
    # Served as a file to save, not as a page to look at.
    assert response.headers["content-type"] == "application/x-chess-pgn"
    assert PGN_FILE.fullmatch(response.headers["content-disposition"])
    assert '[White "Human"]' in response.text
    assert '[Black "Random mover"]' in response.text
    assert '[Result "*"]' in response.text
    assert replayed(response.text) == replay([state, *played]).position.fen


def test_a_watch_link_downloads_the_same_game(chess_client):
    links = start_game(chess_client)

    by_player = chess_client.get(f"/chess/api/games/{links['white']}/pgn")
    by_watcher = chess_client.get(f"/chess/api/games/{links['watch']}/pgn")

    assert by_player.status_code == by_watcher.status_code == 200
    assert replayed(by_player.text) == replayed(by_watcher.text)


def test_a_finished_game_is_downloaded_with_the_result_it_reached(chess_client):
    links = start_game(chess_client, RANDOM_GAME, move_delay=0.01)
    with connect(chess_client, links["watch"]) as websocket:
        events = receive_game(websocket)

    response = chess_client.get(f"/chess/api/games/{links['watch']}/pgn")

    game = replay(events)
    assert game.position.game_over is not None
    assert response.status_code == 200
    assert f'[Result "{game.position.game_over.result}"]' in response.text
    assert replayed(response.text) == game.position.fen


def test_the_pgn_is_downloaded_under_an_empty_prefix(tmp_path):
    with serve("", tmp_path) as client:
        links = start(client, prefix="")["links"]

        response = client.get(f"/api/games/{links['white']}/pgn")

    assert response.status_code == 200
    assert '[Event "chess-ai game"]' in response.text
    assert PGN_FILE.fullmatch(response.headers["content-disposition"])


def test_a_finished_game_is_saved_in_the_games_directory(tmp_path):
    """Nobody asks for this: a game that reaches a result is kept as it ends."""
    with serve("/chess", tmp_path) as client:
        links = start_game(client, RANDOM_GAME, move_delay=0.01)
        with connect(client, links["watch"]) as websocket:
            events = receive_game(websocket)

    # Shutting the server down waits for the games it was playing, so the file is there.
    [saved] = saved_games(tmp_path)
    game = replay(events)
    assert game.position.game_over is not None
    assert replayed(saved.read_text(encoding="utf-8")) == game.position.fen
    assert f'[Result "{game.position.game_over.result}"]' in saved.read_text(encoding="utf-8")


def test_an_aborted_game_is_not_saved(tmp_path):
    with serve("/chess", tmp_path) as client:
        links = start_game(client)
        with connect(client, links["white"]) as websocket:
            assert receive(websocket).type == "state"
            ask(websocket, type="abort")
            assert receive(websocket).type == "game_over"

    assert saved_games(tmp_path) == []


def test_the_pgn_is_read_on_the_event_loop_the_game_is_played_on(chess_client, monkeypatch):
    """Only a reader on the loop sees a whole game, because that is where moves are made.

    ``GameSession.state`` collects its fields one at a time, and a move appends to the
    moves before it sets the position. A reader preempted between those two reads writes
    out a game whose result is a checkmate that is not among its moves.
    """
    on_the_loop: list[bool] = []

    def note_where_it_runs(game, *, now=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_the_loop.append(False)
        else:
            on_the_loop.append(True)
        return game_pgn(game, now=now)

    monkeypatch.setattr("chess_ai.web.app.game_pgn", note_where_it_runs)
    links = start_game(chess_client)

    assert chess_client.get(f"/chess/api/games/{links['watch']}/pgn").status_code == 200
    assert on_the_loop == [True]


def test_nothing_the_game_routes_answer_is_ever_cached(chess_client):
    """A URL with a different game behind it after every move, and a proxy in front.

    The refusals matter as much as the file: a 404 may be kept by a cache of its own accord.
    """
    links = start_game(chess_client)
    answers = [
        chess_client.get("/chess/api/games/nothing-here"),
        chess_client.get("/chess/api/games/nothing-here/pgn"),
        chess_client.get(f"/chess/api/games/{links['watch']}"),
        chess_client.get(f"/chess/api/games/{links['watch']}/pgn"),
    ]

    assert [answer.status_code for answer in answers] == [404, 404, 200, 200]
    assert [answer.headers.get("cache-control") for answer in answers] == ["no-store"] * 4


@pytest.mark.anyio
async def test_two_games_started_at_once_are_not_both_let_in_past_the_limit(tmp_path, monkeypatch):
    """Making a model player reads a checkpoint off the disk, which takes seconds.

    The handler suspends for that, so two starts can interleave. With room for one more game,
    only one of them may have it.
    """
    app = create_app(
        Config(
            server=ServerConfig(path_prefix="/chess"),
            paths=PathsConfig(games=tmp_path / "games"),
            games=GamesConfig(max_ongoing=1),
        ),
        tmp_path / "static",
    )
    entered, release = threading.Event(), threading.Event()
    build, made, guard = app_module._players, [], threading.Lock()

    async def slowly(request, config, engines):
        """The first game's players take as long as a real checkpoint would."""
        with guard:
            first = not made
            made.append(request)
        if first:
            entered.set()
            released = await asyncio.to_thread(release.wait, 10)
            assert released, "the slow game was never released"
        return await build(request, config, engines)

    monkeypatch.setattr(app_module, "_players", slowly)

    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app), base_url="http://testserver") as c,
    ):
        async with asyncio.timeout(30):
            slow = asyncio.create_task(c.post("/chess/api/games", json=HUMAN_GAME))
            await asyncio.to_thread(entered.wait, 10)
            second = asyncio.create_task(c.post("/chess/api/games", json=HUMAN_GAME))
            # Long enough for the second request to reach the handler, which is all it has
            # to do: whether it gets any further is the thing being tested.
            await asyncio.sleep(0.1)
            release.set()
            first, last = await slow, await second

    assert (first.status_code, last.status_code) == (200, 409)


def resign_through(chess_client, link: str) -> None:
    with connect(chess_client, link) as websocket:
        assert receive(websocket).type == "state"
        ask(websocket, type="resign")
        assert receive(websocket).type == "game_over"


def test_a_finished_game_is_played_again_with_the_same_settings(chess_client):
    fen = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    first = start(chess_client, {"white": RANDOM, "black": HUMAN}, fen=fen, move_delay=10)
    resign_through(chess_client, first["links"]["black"])

    again = chess_client.post(f"/chess/api/games/{first['links']['black']}/rematch")

    assert again.status_code == 200
    game, links = again.json()["game"], again.json()["links"]
    assert (game["white"], game["black"]) == (first["game"]["white"], first["game"]["black"])
    assert game["start_fen"] == fen
    assert game["moves"] == [] and game["position"]["game_over"] is None
    assert game["id"] != first["game"]["id"]
    assert links["black"] not in (None, first["links"]["black"])
    # The first game is still there to look at, as it ended.
    ended = chess_client.get(f"/chess/api/games/{first['links']['watch']}").json()
    assert ended["game"]["position"]["game_over"]["reason"] == "resignation"


def test_a_game_played_again_keeps_the_move_delay(chess_client):
    first = start(chess_client, RANDOM_GAME, move_delay=10)

    again = chess_client.post(f"/chess/api/games/{first['links']['control']}/rematch").json()
    # Long enough for the default delay of half a second to have let a move through.
    time.sleep(1)

    view = chess_client.get(f"/chess/api/games/{again['links']['watch']}").json()
    assert view["game"]["moves"] == []


def test_a_game_between_two_people_played_again_has_a_new_link_for_each(chess_client):
    first = start_game(chess_client, black=HUMAN)
    resign_through(chess_client, first["white"])

    again = chess_client.post(f"/chess/api/games/{first['white']}/rematch").json()["links"]

    assert again["white"] is not None and again["black"] is not None
    assert {again["white"], again["black"]}.isdisjoint({first["white"], first["black"]})


def test_a_watch_link_cannot_start_a_game_again(chess_client):
    links = start_game(chess_client)

    refused = chess_client.post(f"/chess/api/games/{links['watch']}/rematch")

    assert refused.status_code == 403
    assert "watch link" in refused.json()["detail"]


def test_a_game_that_is_gone_cannot_be_played_again(chess_client):
    refused = chess_client.post("/chess/api/games/nothing-here/rematch")

    assert refused.status_code == 404


def test_playing_again_counts_towards_the_limit(tmp_path):
    with serve("/chess", tmp_path, max_ongoing=1) as client:
        links = start_game(client, move_delay=10)

        refused = client.post(f"/chess/api/games/{links['white']}/rematch")

    assert refused.status_code == 409


def test_both_people_playing_again_end_up_in_the_same_new_game(chess_client):
    """Each of two people clicks Play again once their game ends, and expects the other."""
    first = start_game(chess_client, black=HUMAN)
    resign_through(chess_client, first["white"])

    white = chess_client.post(f"/chess/api/games/{first['white']}/rematch").json()
    black = chess_client.post(f"/chess/api/games/{first['black']}/rematch").json()

    assert black["game"]["id"] == white["game"]["id"]
    assert black["links"]["black"] == white["links"]["black"]
    assert black["links"]["watch"] == white["links"]["watch"]
    # Joining hands over your own side's link, not the other person's.
    assert black["links"]["white"] is None


def test_playing_again_twice_from_the_same_game_starts_one_game(chess_client):
    first = start_game(chess_client, move_delay=10)
    resign_through(chess_client, first["white"])

    once = chess_client.post(f"/chess/api/games/{first['white']}/rematch").json()
    twice = chess_client.post(f"/chess/api/games/{first['white']}/rematch").json()

    assert twice["game"]["id"] == once["game"]["id"]
    assert twice["links"]["white"] == once["links"]["white"]


def test_playing_again_after_the_new_game_has_ended_starts_another(chess_client):
    first = start_game(chess_client, move_delay=10)
    resign_through(chess_client, first["white"])
    once = chess_client.post(f"/chess/api/games/{first['white']}/rematch").json()
    resign_through(chess_client, once["links"]["white"])

    again = chess_client.post(f"/chess/api/games/{first['white']}/rematch").json()

    assert again["game"]["id"] != once["game"]["id"]
