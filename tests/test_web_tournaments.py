"""The tournaments page's API: the list of tournaments, one tournament's leaderboard, crosstable
and games, and its games replayed and downloaded."""

from collections.abc import Iterator

import anyio
import pytest
from fastapi.testclient import TestClient
from pgn_helpers import read_back
from test_tournament import definition, fakes

from chess_ai.config import Config, PathsConfig, ServerConfig
from chess_ai.tournament import create_tournament, open_tournament, play_tournament
from chess_ai.web import create_app

PREFIX = "/chess"


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    config = Config(
        server=ServerConfig(path_prefix=PREFIX),
        paths=PathsConfig(
            games=tmp_path / "games", runs=tmp_path / "runs", tournaments=tmp_path / "tournaments"
        ),
    )
    with TestClient(create_app(config, static_dir=tmp_path / "static")) as serving:
        yield serving


@pytest.fixture
def played(tmp_path):
    """A finished tournament of three entrants, two games a pairing, called ``cup``."""
    create_tournament(tmp_path / "tournaments", definition(games=2))
    anyio.run(play_tournament, open_tournament(tmp_path / "tournaments", "cup"), fakes)
    return open_tournament(tmp_path / "tournaments", "cup")


def test_the_tournaments_are_listed_with_how_far_each_has_got(client, played):
    listed = client.get(f"{PREFIX}/api/tournaments")

    assert listed.status_code == 200
    assert listed.headers["cache-control"] == "no-store"
    (cup,) = listed.json()
    assert cup["name"] == "cup"
    assert cup["entrants"] == ["Stockfish 1350", "Stockfish 1450", "Stockfish 1550"]
    assert (cup["games_played"], cup["games_total"], cup["playing"]) == (6, 6, False)
    assert cup["error"] is None


def test_a_tournament_shows_its_leaderboard_crosstable_and_games(client, played):
    shown = client.get(f"{PREFIX}/api/tournaments/cup")

    assert shown.status_code == 200
    page = shown.json()
    assert page["entrants"] == ["Stockfish 1350", "Stockfish 1450", "Stockfish 1550"]
    assert page["definition"]["games_per_pairing"] == 2
    board = page["leaderboard"]
    assert sorted(s["entrant"] for s in board["standings"]) == [0, 1, 2]
    assert board["crosstable"][0][0] is None and board["crosstable"][0][1]["games"] == 2
    games = page["games"]
    assert [(g["index"], g["round"], g["white"], g["black"]) for g in games][:2] == [
        (0, 1, 1, 0),
        (1, 1, 0, 2),
    ]
    first = read_back(played.games()[0].pgn)
    assert games[0]["opening"] == first.headers["Opening"]
    assert games[0]["result"] == first.headers["Result"]
    assert games[0]["plies"] == len(list(first.mainline_moves()))


def test_a_tournament_game_is_replayed_and_all_of_them_downloaded(client, played):
    third = client.get(f"{PREFIX}/api/tournaments/cup/replay?game=2")
    beyond = client.get(f"{PREFIX}/api/tournaments/cup/replay?game=6")
    pgn = client.get(f"{PREFIX}/api/tournaments/cup/pgn")

    assert third.status_code == 200
    assert len(third.json()["games"]) == 6
    assert third.json()["selected"]["index"] == 2
    assert beyond.status_code == 404
    assert pgn.status_code == 200
    assert 'filename="cup.pgn"' in pgn.headers["content-disposition"]
    assert pgn.text == "\n".join(game.pgn for game in played.games()) + "\n"


@pytest.mark.parametrize("path", ["", "/replay", "/pgn"])
def test_a_tournament_that_is_not_there_is_not_found(client, path):
    assert client.get(f"{PREFIX}/api/tournaments/nothing{path}").status_code == 404


def test_a_tournament_with_no_games_yet_has_every_entrant_level_and_nothing_to_replay(
    client, tmp_path
):
    create_tournament(tmp_path / "tournaments", definition(games=2))

    page = client.get(f"{PREFIX}/api/tournaments/cup").json()
    replayed = client.get(f"{PREFIX}/api/tournaments/cup/replay")

    assert page["games"] == [] and page["leaderboard"]["games_played"] == 0
    assert {standing["rating"] for standing in page["leaderboard"]["standings"]} == {0}
    assert replayed.status_code == 404


def test_a_tournament_being_played_says_so(client, tmp_path):
    import fcntl
    import os

    tournament = create_tournament(tmp_path / "tournaments", definition(games=2))
    # Held as a process playing it holds it.
    descriptor = os.open(tournament.path / "playing.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        listed = client.get(f"{PREFIX}/api/tournaments").json()
        page = client.get(f"{PREFIX}/api/tournaments/cup").json()
    finally:
        os.close(descriptor)

    assert listed[0]["playing"] and page["playing"]
    assert not client.get(f"{PREFIX}/api/tournaments/cup").json()["playing"]


def test_a_tournament_whose_games_cannot_be_read_is_a_server_error(client, played):
    with (played.path / "games.jsonl").open("a") as file:
        file.write("not a game\n")

    for path in ("", "/replay", "/pgn"):
        assert client.get(f"{PREFIX}/api/tournaments/cup{path}").status_code == 500
    (listed,) = client.get(f"{PREFIX}/api/tournaments").json()
    assert listed["name"] == "cup"
    assert "line 7" in listed["error"] and "is not a game" in listed["error"]


@pytest.mark.parametrize("name", [".hidden", "a b", "-dash"])
def test_a_name_no_tournament_can_have_is_not_found(client, name):
    assert client.get(f"{PREFIX}/api/tournaments/{name}").status_code == 404


def test_a_tournament_that_finishes_while_it_is_listed_is_not_shown_as_stopped_short(
    client, played, monkeypatch
):
    import fcntl
    import os

    from chess_ai import tournament

    games = played.path / "games.jsonl"
    lines = games.read_text().splitlines(keepends=True)
    games.write_text("".join(lines[:-1]))
    descriptor = os.open(played.path / "playing.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    read = tournament.Tournament.games

    def finishing(self):
        # The last game ends, and its player with it, just after the games are read.
        found = read(self)
        with games.open("a") as file:
            file.write(lines[-1])
        os.close(descriptor)
        return found

    monkeypatch.setattr(tournament.Tournament, "games", finishing)

    (cup,) = client.get(f"{PREFIX}/api/tournaments").json()

    assert (cup["games_played"], cup["playing"]) in ((5, True), (6, False))
