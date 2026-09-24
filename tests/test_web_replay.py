"""The replay API: uploaded PGN files, and the games this server has saved."""

import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from chess_ai import replay
from chess_ai.config import Config, PathsConfig, ServerConfig
from chess_ai.replay import MAX_BYTES
from chess_ai.web import app, create_app
from chess_ai.web.app import _read_at_most

FOOLS_MATE = """[Event "chess-ai game"]
[Site "?"]
[Date "2026.09.24"]
[Round "-"]
[White "Human"]
[Black "Random mover"]
[Result "0-1"]
[Termination "checkmate"]

1. f3 e5 2. g4 Qh4# 0-1
"""

SCHOLARS_MATE = """[Event "Kitchen table"]
[Site "Trondheim"]
[Date "2001.04.03"]
[Round "2"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

TWO_GAMES = f"{FOOLS_MATE}\n{SCHOLARS_MATE}"

PGN_TYPE = {"Content-Type": "application/x-chess-pgn"}


@pytest.fixture
def games(tmp_path) -> Path:
    return tmp_path / "games"


def serve(tmp_path, games) -> TestClient:
    config = Config(server=ServerConfig(path_prefix="/chess"), paths=PathsConfig(games=games))
    return TestClient(create_app(config, static_dir=tmp_path / "static"))


@pytest.fixture
def chess_client(tmp_path, games) -> TestClient:
    return serve(tmp_path, games)


def save(games: Path, name: str, pgn: str) -> None:
    games.mkdir(parents=True, exist_ok=True)
    (games / name).write_text(pgn, encoding="utf-8")


def upload(chess_client, pgn: str, **params):
    return chess_client.post(
        "/chess/api/replay/pgn", content=pgn.encode("utf-8"), headers=PGN_TYPE, params=params
    )


def test_a_file_of_several_games_is_read_back_as_every_game_in_it(chess_client):
    response = upload(chess_client, TWO_GAMES)

    assert response.status_code == 200
    file = response.json()
    assert [(game["index"], game["white"], game["black"]) for game in file["games"]] == [
        (0, "Human", "Random mover"),
        (1, "Alice", "Bob"),
    ]


def test_the_game_asked_for_comes_back_as_the_positions_it_passed_through(chess_client):
    file = upload(chess_client, TWO_GAMES, game=1).json()

    played = file["selected"]
    assert played["index"] == 1
    assert [move["san"] for move in played["moves"]] == [
        "e4",
        "e5",
        "Bc4",
        "Nc6",
        "Qh5",
        "Nf6",
        "Qxf7#",
    ]
    mate = played["moves"][-1]["position"]
    assert mate["pieces"]["f7"] == {"color": "white", "type": "queen"}
    assert mate["check_square"] == "e8"
    assert mate["game_over"] == {"result": "1-0", "reason": "checkmate"}
    assert played["start_position"]["turn"] == "white"


def test_a_file_with_no_game_in_it_is_refused_with_something_to_read(chess_client):
    response = upload(chess_client, "I do not know how the pieces move.")

    assert response.status_code == 400
    assert response.json()["detail"] == "no game was found in that file"


def test_a_file_with_a_game_that_cannot_be_read_names_the_game(chess_client):
    broken = f'{FOOLS_MATE}\n[Event "Broken"]\n[White "A"]\n[Black "B"]\n\n1. e4 e5 2. Qh8 *\n'

    response = upload(chess_client, broken)

    assert response.status_code == 400
    assert response.json()["detail"].startswith("game 2 (A – B) cannot be read: illegal san: 'Qh8'")


def test_asking_for_a_game_the_file_has_not_got_is_refused(chess_client):
    response = upload(chess_client, TWO_GAMES, game=7)

    assert response.status_code == 404
    assert response.json()["detail"] == "that file has no game 8: there are 2 games in it"


def test_a_file_larger_than_the_server_will_read_is_turned_away(chess_client):
    response = chess_client.post(
        "/chess/api/replay/pgn", content=b"1. e4 e5\n" * (MAX_BYTES // 9 + 1), headers=PGN_TYPE
    )

    assert response.status_code == 413
    assert "8 MB" in response.json()["detail"]


def test_a_file_too_large_to_read_is_turned_away_without_declaring_its_length(chess_client):
    """A body sent in chunks says no length, so a declared length cannot be the guard."""

    def chunks():
        for _ in range(MAX_BYTES // 1024 + 2):
            yield b"x" * 1024

    response = chess_client.post("/chess/api/replay/pgn", content=chunks(), headers=PGN_TYPE)

    assert response.status_code == 413


@pytest.mark.anyio
async def test_a_body_is_dropped_as_it_arrives_rather_than_read_and_then_refused():
    """What the limit is for: the memory a file too large to read costs is bounded."""
    asked = 0

    async def sent():
        nonlocal asked
        for _ in range(10_000):
            asked += 1
            yield b"x" * 1024

    with pytest.raises(HTTPException) as refused:
        await _read_at_most(sent(), 4096)

    assert refused.value.status_code == 413
    # The four chunks that fill the limit, and the one that passes it.
    assert asked <= 6


UPLOAD_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.3"},
    "http_version": "1.1",
    "method": "POST",
    "scheme": "http",
    "path": "/chess/api/replay/pgn",
    "raw_path": b"/chess/api/replay/pgn",
    "query_string": b"",
    "root_path": "",
    "headers": [(b"host", b"testserver"), (b"content-type", PGN_TYPE["Content-Type"].encode())],
    "client": ("testclient", 50000),
    "server": ("testserver", 80),
}
"""One upload, as the server sees it before anything of this application runs."""


@pytest.mark.anyio
async def test_an_upload_the_viewer_gives_up_on_ends_quietly(tmp_path, games):
    """Giving up on a large file over a slow link is a thing people do on purpose.

    Driven as the server drives it, because the giving up is a message no client of
    ours can send: a piece of the body, and then the connection going away.
    """
    application = create_app(
        Config(server=ServerConfig(path_prefix="/chess"), paths=PathsConfig(games=games)),
        static_dir=tmp_path / "static",
    )
    given_up = [
        {"type": "http.request", "body": b'[Event "x"]', "more_body": True},
        {"type": "http.disconnect"},
    ]
    sent: list[dict] = []

    async def receive():
        return given_up.pop(0)

    async def send(message):
        sent.append(message)

    await application(UPLOAD_SCOPE, receive, send)

    # Answered, rather than raised through the server and logged as a fault of ours.
    assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [
        499
    ]


def test_the_whole_reading_of_a_file_happens_away_from_the_event_loop(chess_client, monkeypatch):
    """Decoding is part of reading it: for a file that is not UTF-8 it is two passes.

    The loop's own thread is asked for from inside the request rather than guessed at,
    since the test itself runs on another thread again and would compare nothing.
    """
    threads: dict[str, str] = {}
    read_body, decode, read = app._read_at_most, replay.decode, replay.read

    async def watched_body(*args, **kwargs):
        # Taking the body is the part that does belong on the loop.
        threads["loop"] = threading.current_thread().name
        return await read_body(*args, **kwargs)

    def watched_decode(data):
        threads["decode"] = threading.current_thread().name
        return decode(data)

    def watched_read(text, **kwargs):
        threads["read"] = threading.current_thread().name
        return read(text, **kwargs)

    monkeypatch.setattr(app, "_read_at_most", watched_body)
    monkeypatch.setattr(replay, "decode", watched_decode)
    monkeypatch.setattr(replay, "read", watched_read)

    assert upload(chess_client, FOOLS_MATE).status_code == 200

    # One thread for the whole reading, and not the one the game is played on.
    assert threads["decode"] == threads["read"] != threads["loop"]


def test_a_game_number_before_the_first_is_refused_in_an_uploaded_file(chess_client):
    """Counting from the end is Python's habit; a viewer asking for game -1 is a mistake."""
    response = upload(chess_client, FOOLS_MATE, game=-1)

    assert response.status_code == 422


def test_a_game_number_before_the_first_is_refused_in_a_saved_game(chess_client, games):
    save(games, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)

    response = chess_client.get(
        "/chess/api/replay/saved/20260924-143005-3f9a1b2c.pgn", params={"game": -1}
    )

    assert response.status_code == 422


def test_saved_games_are_listed_with_the_date_the_players_and_the_result(chess_client, games):
    save(games, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)
    save(games, "20260925-090000-aaaaaaaa.pgn", SCHOLARS_MATE)

    response = chess_client.get("/chess/api/replay/saved")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "20260925-090000-aaaaaaaa.pgn",
            "event": "Kitchen table",
            "date": "2001.04.03",
            "white": "Alice",
            "black": "Bob",
            "result": "1-0",
        },
        {
            "name": "20260924-143005-3f9a1b2c.pgn",
            "event": "chess-ai game",
            "date": "2026.09.24",
            "white": "Human",
            "black": "Random mover",
            "result": "0-1",
        },
    ]
    # A game finishing adds to this list, so a kept copy of it goes out of date at once.
    assert response.headers["cache-control"] == "no-store"


def test_nothing_has_been_saved_before_the_first_game_is_played(chess_client):
    assert chess_client.get("/chess/api/replay/saved").json() == []


def test_a_saved_game_is_opened_by_the_name_the_list_gives_it(chess_client, games):
    save(games, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)

    response = chess_client.get("/chess/api/replay/saved/20260924-143005-3f9a1b2c.pgn")

    assert response.status_code == 200
    file = response.json()
    assert [move["san"] for move in file["selected"]["moves"]] == ["f3", "e5", "g4", "Qh4#"]
    assert file["selected"]["termination"] == "checkmate"


@pytest.mark.parametrize("name", ["secrets.pgn", "../secrets.pgn", "..%2Fsecrets.pgn"])
def test_no_name_opens_a_file_outside_the_games_directory(chess_client, games, tmp_path, name):
    save(games, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)
    (tmp_path / "secrets.pgn").write_text(FOOLS_MATE, encoding="utf-8")

    response = chess_client.get(f"/chess/api/replay/saved/{name}")

    assert response.status_code == 404


def test_a_game_played_here_is_listed_and_replayed_move_for_move(tmp_path, games):
    """The one journey that matters: play a game, then open it again in the viewer."""
    with (
        serve(tmp_path, games) as playing,
        playing.websocket_connect("/chess/api/game/ws") as websocket,
    ):
        assert websocket.receive_json()["type"] == "no_game"
        playing.post(
            "/chess/api/game", json={"white": "random", "black": "random", "move_delay": 0}
        )
        # Played out to its result, which is the point at which a game is saved. A game
        # that ends on the board ends with the move that ended it, and sends no more.
        while websocket.receive_json().get("position", {}).get("game_over") is None:
            pass

    client = serve(tmp_path, games)
    [saved] = client.get("/chess/api/replay/saved").json()
    assert (saved["white"], saved["black"]) == ("Random mover", "Random mover")

    file = client.get(f"/chess/api/replay/saved/{saved['name']}").json()
    played = file["selected"]
    assert played["result"] == saved["result"]
    assert played["plies"] == len(played["moves"])
    # The game ends where its last move leaves it, which is what the viewer will show.
    assert played["moves"][-1]["position"]["game_over"] is not None
