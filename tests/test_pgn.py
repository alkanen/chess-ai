"""What a game written out as PGN says, and where a finished game is saved."""

from datetime import datetime

import pytest
from game_helpers import ScriptedPlayer, playing, scripted_players, settled
from pgn_helpers import read_back, replayed

from chess_ai.game_session import GameSession, GameState
from chess_ai.pgn import game_pgn, pgn_filename, save_game
from chess_ai.players import HumanPlayer
from chess_ai.position_view import GameOver, GameOverReason

pytestmark = pytest.mark.anyio

FOOLS_MATE = "f2f3 e7e5 g2g4 d8h4"
GAME_ID = "3f9a1b2c4d5e6f70"
"""Stands in for the hex id a real game is given, which is made up afresh every time."""

WHEN = datetime(2026, 9, 24, 14, 30, 5)
"""A fixed moment, so that the date and time a game is written out can be read back."""

MATE_IN_ONE = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 3"
"""Fool's mate with the mate still to come, which is Black's to play and the game's first."""


async def fools_mate() -> GameState:
    """A game played out to checkmate from the usual starting position."""
    session = GameSession(*scripted_players(FOOLS_MATE), id=GAME_ID)
    await session.play()
    return session.state


async def mate_from_a_position_of_its_own() -> GameState:
    """A game that began where ``MATE_IN_ONE`` stands, so Black had the first move."""
    session = GameSession(ScriptedPlayer([]), ScriptedPlayer(["d8h4"]), fen=MATE_IN_ONE, id=GAME_ID)
    await session.play()
    return session.state


async def in_progress() -> GameState:
    """A game one move in, played by a person against a player that moves for itself."""
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]), id=GAME_ID)
    async with playing(session):
        session.submit_move("e2e4")
        await settled()
        return session.state


async def test_a_finished_game_is_written_out_with_the_headers_a_reader_expects():
    game = await fools_mate()

    headers = read_back(game_pgn(game, now=WHEN)).headers

    assert dict(headers) == {
        "Event": "chess-ai game",
        # Nobody has told the server where it stands, and PGN's word for that is "?".
        "Site": "?",
        "Date": "2026.09.24",
        "Time": "14:30:05",
        "Round": "-",
        "White": "Scripted",
        "Black": "Scripted",
        "WhiteType": "program",
        "BlackType": "program",
        "Result": "0-1",
        "Termination": "checkmate",
    }


async def test_a_game_names_each_player_and_says_which_of_them_is_a_person():
    game = await in_progress()

    headers = read_back(game_pgn(game)).headers

    assert (headers["White"], headers["WhiteType"]) == ("Human", "human")
    assert (headers["Black"], headers["BlackType"]) == ("Scripted", "program")


async def test_the_moves_replay_to_the_position_the_game_reached():
    game = await fools_mate()

    pgn = game_pgn(game)

    assert [node.san() for node in read_back(pgn).mainline()] == ["f3", "e5", "g4", "Qh4#"]
    assert replayed(pgn) == game.position.fen


async def test_a_game_in_progress_is_written_out_as_far_as_it_has_been_played():
    game = await in_progress()

    pgn = game_pgn(game)

    headers = read_back(pgn).headers
    # PGN's own "the game has not ended", and no termination, because it has not.
    assert headers["Result"] == "*"
    assert "Termination" not in headers
    assert replayed(pgn) == game.position.fen
    # Black has not answered yet, and the game says so rather than guessing at an end.
    assert pgn.endswith("1. e4 *\n")


async def test_a_game_from_a_position_of_its_own_carries_that_position():
    game = await mate_from_a_position_of_its_own()

    pgn = game_pgn(game)

    headers = read_back(pgn).headers
    assert headers["SetUp"] == "1"
    assert headers["FEN"] == game.start_fen
    # The game opened with Black's move, which is the third move of the game it stands in.
    assert pgn.endswith("3... Qh4# 0-1\n")
    assert replayed(pgn) == game.position.fen


async def test_a_game_from_the_usual_starting_position_carries_no_position_of_its_own():
    """A reader that is told nothing starts a game where games start."""
    game = await fools_mate()

    headers = read_back(game_pgn(game)).headers

    assert "FEN" not in headers
    assert "SetUp" not in headers


async def test_a_resigned_game_is_written_out_as_won_by_the_other_side():
    session = GameSession(HumanPlayer(), ScriptedPlayer([]), id=GAME_ID)
    async with playing(session):
        session.resign("white")
        game = session.state

    headers = read_back(game_pgn(game)).headers

    assert (headers["Result"], headers["Termination"]) == ("0-1", "resignation")


async def test_an_aborted_game_is_written_out_as_given_up_with_no_result():
    session = GameSession(HumanPlayer(), ScriptedPlayer([]), id=GAME_ID)
    async with playing(session):
        session.abort()
        game = session.state

    headers = read_back(game_pgn(game)).headers

    assert (headers["Result"], headers["Termination"]) == ("*", "abandoned")


@pytest.mark.parametrize(
    ("reason", "termination"),
    [
        ("checkmate", "checkmate"),
        ("stalemate", "stalemate"),
        ("insufficient_material", "insufficient material"),
        ("threefold_repetition", "threefold repetition"),
        ("fifty_move_rule", "fifty-move rule"),
        ("resignation", "resignation"),
        ("abort", "abandoned"),
    ],
)
async def test_every_way_a_game_can_end_is_written_out_in_words(
    reason: GameOverReason, termination: str
):
    """A reader looking at ``Termination`` is told how the game ended, not that it did."""
    played = await in_progress()
    over = GameOver(result="1/2-1/2", reason=reason)
    game = played.model_copy(
        update={"position": played.position.model_copy(update={"game_over": over})}
    )

    assert read_back(game_pgn(game)).headers["Termination"] == termination


async def test_a_game_is_saved_as_a_file_of_its_own_in_a_directory_made_for_it(tmp_path):
    game = await fools_mate()
    games = tmp_path / "games" / "played"

    path = save_game(game, games, now=WHEN)

    assert path == games / "20260924-143005-3f9a1b2c.pgn"
    assert path.read_text(encoding="utf-8") == game_pgn(game, now=WHEN)
    assert replayed(path.read_text(encoding="utf-8")) == game.position.fen


async def test_the_file_name_says_when_the_game_was_played_and_which_game_it_was():
    game = await fools_mate()

    assert pgn_filename(game, now=WHEN) == "20260924-143005-3f9a1b2c.pgn"


async def test_a_game_whose_id_is_no_name_for_a_file_is_saved_in_the_directory_anyway(tmp_path):
    """The id names the file, and nothing but a real game's hex id ever should."""
    session = GameSession(*scripted_players(FOOLS_MATE), id="../../../etc/passwd")
    await session.play()

    path = save_game(session.state, tmp_path, now=WHEN)

    assert path.parent == tmp_path
    assert path.name == "20260924-143005-etcpassw.pgn"
