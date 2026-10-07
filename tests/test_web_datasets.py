"""The datasets API: the list of datasets, a page of one's games, and one game replayed.

The datasets are built from the fixture PGN files into the test's own data directory, with shards
small enough that a page of games and a game's plies both cross from one shard into the next.
"""

import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
from dataset_helpers import GOOD_GAMES, TINY_SHARDS, build, fixture
from fastapi.testclient import TestClient

from chess_ai.config import Config, PathsConfig, ServerConfig
from chess_ai.dataset import FORMAT_VERSION, GAME_DTYPE, POSITION_DTYPE, dataset_path
from chess_ai.dataset.manifest import MANIFEST_FILE
from chess_ai.dataset.records import MOVE_DTYPE
from chess_ai.web import create_app
from chess_ai.web.datasets import pgn_date

PREFIX = "/chess"


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def client(tmp_path, data_dir) -> TestClient:
    config = Config(server=ServerConfig(path_prefix=PREFIX), paths=PathsConfig(data=data_dir))
    return TestClient(create_app(config, static_dir=tmp_path / "static"))


def api(path: str) -> str:
    return f"{PREFIX}/api/{path}"


def every_game(data_dir: Path, split: str = "train") -> list[dict]:
    """A split's games, as the page that lists them says, all on one page."""
    from chess_ai.web.datasets import dataset_games

    return [game.model_dump() for game in dataset_games(data_dir, "test", split, 0, 200).games]


def test_there_are_no_datasets_before_one_is_built(client):
    response = client.get(api("datasets"))

    assert response.status_code == 200
    assert response.json() == []


def test_datasets_are_listed_with_their_sources_filters_counts_and_statistics(client, data_dir):
    manifest = build(data_dir, validation_fraction=0.25, shards=TINY_SHARDS)
    build(data_dir, "lichess.pgn", name="another", validation_fraction=0.0)

    listed = client.get(api("datasets")).json()

    assert [dataset["name"] for dataset in listed] == ["another", "test"]
    test = listed[1]
    assert test["error"] is None
    assert test["manifest"] == manifest.model_dump(mode="json")
    assert test["manifest"]["created"]
    assert {Path(source["path"]).name for source in test["manifest"]["sources"]} == {
        "custom-start.pgn",
        "lichess.pgn",
        "malformed.pgn",
        "unrated.pgn",
    }
    assert test["manifest"]["filters"] == {}
    splits = test["manifest"]["splits"]
    assert splits["train"]["games"] + splits["validation"]["games"] == GOOD_GAMES
    statistics = test["manifest"]["statistics"]
    assert statistics["results"]
    assert statistics["time_controls"]
    assert statistics["ratings"]


def test_a_dataset_that_cannot_be_read_is_listed_with_the_reason(client, data_dir):
    build(data_dir, "lichess.pgn")
    manifest = dataset_path(data_dir, "test") / MANIFEST_FILE
    manifest.write_text(json.dumps({"format_version": FORMAT_VERSION + 1}))

    listed = client.get(api("datasets")).json()

    assert listed[0]["name"] == "test"
    assert listed[0]["manifest"] is None
    assert "rebuild the dataset" in listed[0]["error"]


def test_one_dataset_is_described_by_its_name(client, data_dir):
    manifest = build(data_dir, "lichess.pgn")

    response = client.get(api("datasets/test"))

    assert response.status_code == 200
    assert response.json()["manifest"] == manifest.model_dump(mode="json")


def test_a_dataset_is_described_with_its_versions(client, data_dir):
    from chess_ai.dataset.builder import append_dataset

    build(data_dir, "lichess.pgn", validation_fraction=0.0)
    manifest = append_dataset("test", [fixture("unrated.pgn")], data_dir=data_dir)

    described = client.get(api("datasets/test")).json()["manifest"]

    assert described == manifest.model_dump(mode="json")
    assert [version["version"] for version in described["versions"]] == [1, 2]
    assert [game["source"] for game in every_game(data_dir)] == [fixture("lichess.pgn")] * 4 + [
        fixture("unrated.pgn")
    ] * 3


@pytest.mark.parametrize("name", ["missing", "..", ".test.1-abc.partial"])
def test_a_dataset_that_is_not_there_is_not_found(client, data_dir, name):
    build(data_dir, "lichess.pgn")

    assert client.get(api(f"datasets/{name}")).status_code == 404
    assert client.get(api(f"datasets/{name}/train/games")).status_code == 404
    assert client.get(api(f"datasets/{name}/train/games/0")).status_code == 404


def test_a_page_of_games_says_what_each_game_was(client, data_dir):
    build(data_dir, "lichess.pgn", validation_fraction=0.0)

    page = client.get(api("datasets/test/train/games")).json()

    assert page["dataset"] == "test"
    assert page["split"] == "train"
    assert page["total"] == 4
    assert page["offset"] == 0
    first = next(game for game in page["games"] if game["white_rating"] == 1687)
    assert first == {
        "index": first["index"],
        "plies": 33,
        "white_rating": 1687,
        "black_rating": 1702,
        "result": "1-0",
        "date": "2024.01.05",
        "time_control": "blitz",
        "rating_source": "lichess",
        "source": fixture("lichess.pgn"),
        "custom_start": False,
    }


def test_games_are_paged_through_across_shards(client, data_dir):
    build(data_dir, validation_fraction=0.0, shards=TINY_SHARDS)
    everything = every_game(data_dir)
    assert len(everything) == GOOD_GAMES

    pages = [
        client.get(api("datasets/test/train/games"), params={"offset": offset, "limit": 3}).json()
        for offset in (0, 3, 6)
    ]

    assert [page["offset"] for page in pages] == [0, 3, 6]
    assert [len(page["games"]) for page in pages] == [3, 3, 2]
    assert [game for page in pages for game in page["games"]] == everything
    assert [game["index"] for game in everything] == list(range(GOOD_GAMES))


def test_a_page_past_the_last_game_is_empty(client, data_dir):
    build(data_dir, "lichess.pgn", validation_fraction=0.0)

    page = client.get(api("datasets/test/train/games"), params={"offset": 10}).json()

    assert page["total"] == 4
    assert page["games"] == []


def test_a_game_without_ratings_says_so_rather_than_rating_it_zero(client, data_dir):
    build(data_dir, "unrated.pgn", validation_fraction=0.0)

    games = client.get(api("datasets/test/train/games")).json()["games"]

    # One side unrated ("?"), the other rated: each side is said for itself.
    assert {"white_rating": None, "black_rating": 2100} in [
        {side: game[side] for side in ("white_rating", "black_rating")} for game in games
    ]
    assert not any(game["white_rating"] == 0 for game in games)


@pytest.mark.parametrize(
    "params",
    [{"offset": -1}, {"limit": 0}, {"limit": 201}],
)
def test_a_page_out_of_bounds_is_refused(client, data_dir, params):
    build(data_dir, "lichess.pgn")

    assert client.get(api("datasets/test/train/games"), params=params).status_code == 422


def test_a_split_that_is_not_there_is_not_found(client, data_dir):
    build(data_dir, "lichess.pgn")

    response = client.get(api("datasets/test/everything/games"))

    assert response.status_code == 404
    assert "train, validation" in response.json()["detail"]


def test_a_game_of_a_dataset_is_replayed_move_for_move(client, data_dir):
    build(data_dir, "lichess.pgn", validation_fraction=0.0, shards=TINY_SHARDS)
    listed = next(game for game in every_game(data_dir) if game["white_rating"] == 1687)
    with open(fixture("lichess.pgn"), encoding="utf-8") as f:
        original = chess.pgn.read_game(f)
    assert original is not None

    response = client.get(api(f"datasets/test/train/games/{listed['index']}"))

    assert response.status_code == 200
    game = response.json()
    assert game["index"] == listed["index"]
    assert game["white"] == "rated 1687"
    assert game["black"] == "rated 1702"
    assert game["result"] == "1-0"
    assert game["date"] == "2024.01.05"
    assert game["site"] == "lichess.pgn"
    assert game["event"] == "blitz game"
    assert game["plies"] == 33
    assert game["start_fen"] == chess.STARTING_FEN
    assert [move["uci"] for move in game["moves"]] == [
        move.uci() for move in original.mainline_moves()
    ]
    assert game["moves"][-1]["san"] == "Rd8#"
    assert game["moves"][-1]["position"]["fen"] == original.end().board().fen()


def test_a_game_that_began_somewhere_unusual_is_replayed_from_there(client, data_dir):
    build(data_dir, "custom-start.pgn", validation_fraction=0.0)
    with open(fixture("custom-start.pgn"), encoding="utf-8") as f:
        original = chess.pgn.read_game(f)
    assert original is not None
    [listed] = client.get(api("datasets/test/train/games")).json()["games"]

    game = client.get(api("datasets/test/train/games/0")).json()

    assert listed["custom_start"] is True
    assert game["start_fen"] == original.board().fen()
    assert game["moves"][-1]["position"]["fen"] == original.end().board().fen()


def test_a_game_the_split_has_not_got_is_not_found(client, data_dir):
    build(data_dir, "lichess.pgn", validation_fraction=0.0)

    response = client.get(api("datasets/test/train/games/4"))

    assert response.status_code == 404
    assert "has 4 games" in response.json()["detail"]


def test_a_game_whose_records_do_not_make_a_game_is_reported_as_damaged(client, data_dir):
    build(data_dir, "lichess.pgn", validation_fraction=0.0)
    # Every move of every game replaced by the first move in the vocabulary, which is not
    # legal in the first position of any of them.
    for shard in (dataset_path(data_dir, "test") / "train" / "moves").glob("*.bin"):
        shard.write_bytes(bytes(shard.stat().st_size))

    response = client.get(api("datasets/test/train/games/0"))

    assert response.status_code == 500
    assert "train game 1 of dataset 'test' cannot be read" in response.json()["detail"]


@pytest.mark.parametrize("stream", ["games", "positions", "moves"])
def test_a_dataset_whose_shards_disagree_with_its_manifest_is_reported_as_damaged(
    client, data_dir, stream
):
    # What a rebuild swapped in under a reader holding the old manifest looks like, as well as
    # a copy cut short at a record boundary: every request answers with what is wrong.
    build(data_dir, "lichess.pgn", validation_fraction=0.0, shards=TINY_SHARDS)
    last = sorted((dataset_path(data_dir, "test") / "train" / stream).glob("*.bin"))[-1]
    record = {"games": GAME_DTYPE, "positions": POSITION_DTYPE, "moves": MOVE_DTYPE}[stream]
    last.write_bytes(last.read_bytes()[: -record.itemsize])

    # The last game is the one whose records are in the last shard of every stream.
    response = client.get(api("datasets/test/train/games/3"))

    assert response.status_code == 500
    assert "the manifest says" in response.json()["detail"]
    if stream == "games":
        page = client.get(api("datasets/test/train/games"))
        assert page.status_code == 500
        assert "the manifest says" in page.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"), [("ply_offset", 10**9), ("result", 9), ("time_control", 99)]
)
def test_a_game_record_that_makes_no_sense_is_reported_as_damaged(client, data_dir, field, value):
    build(data_dir, "lichess.pgn", validation_fraction=0.0)
    shard = dataset_path(data_dir, "test") / "train" / "games" / "00000.bin"
    records = np.fromfile(shard, dtype=GAME_DTYPE)
    records[0][field] = value
    records.tofile(shard)

    responses = [
        client.get(api("datasets/test/train/games/0")),
        client.get(api("datasets/test/train/games")),
    ]

    for response in responses:
        if field == "time_control":
            # An enum value this code does not know is shown as unknown rather than refused.
            assert response.status_code == 200
        elif field == "ply_offset" and "games/0" not in str(response.url):
            assert response.status_code == 200, "a page lists games without reading their plies"
        else:
            assert response.status_code == 500
            assert "game 1 of dataset 'test'" in response.json()["detail"]


@pytest.mark.parametrize(
    ("date", "written"),
    [
        (20240105, "2024.01.05"),
        (20240100, "2024.01.??"),
        (20240000, "2024.??.??"),
        (0, "????.??.??"),
    ],
)
def test_a_record_date_is_written_as_pgn_writes_one(date, written):
    assert pgn_date(date) == written
