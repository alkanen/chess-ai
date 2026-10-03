import chess
import pytest
from pgn_helpers import read_back
from training_helpers import model_run

from chess_ai import replay
from chess_ai.game_session import GameSession
from chess_ai.match import (
    EVENT,
    MatchGame,
    append_game,
    distinct_games,
    match_filename,
    pairing,
    play_match,
    score,
)
from chess_ai.openings import Opening, OpeningSet, load_opening_set
from chess_ai.players import GameContext, PlayerMove
from chess_ai.position_view import GameOver

pytestmark = pytest.mark.anyio


class FirstLegalPlayer:
    """Plays the first legal move in UCI order: as deterministic as argmax, and much quicker."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def choose_move(self, context: GameContext) -> PlayerMove:
        return PlayerMove(min(context.board.legal_moves, key=lambda move: move.uci()))


class LastLegalPlayer(FirstLegalPlayer):
    """Plays the last legal move in UCI order: deterministic, and not the same player."""

    async def choose_move(self, context: GameContext) -> PlayerMove:
        return PlayerMove(max(context.board.legal_moves, key=lambda move: move.uci()))


def line(name: str, moves: str) -> Opening:
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    return Opening(name=name, moves=tuple(board.move_stack))


TWO_LINES = OpeningSet(
    name="two",
    version=1,
    openings=(line("King's Pawn", "e4 e5 Nf3 Nc6"), line("Queen's Pawn", "d4 d5 c4 e6")),
)


async def test_colours_alternate_and_each_line_is_played_from_both_sides():
    first, second = FirstLegalPlayer("First"), FirstLegalPlayer("Second")

    games = await play_match(first, second, games=4, openings=TWO_LINES)

    assert [game.number for game in games] == [1, 2, 3, 4]
    assert [game.first_plays_white for game in games] == [True, False, True, False]
    assert [(game.game.white.name, game.game.black.name) for game in games] == [
        ("First", "Second"),
        ("Second", "First"),
        ("First", "Second"),
        ("Second", "First"),
    ]
    assert [game.opening.name for game in games] == [
        "King's Pawn",
        "King's Pawn",
        "Queen's Pawn",
        "Queen's Pawn",
    ]


async def test_every_game_starts_with_its_opening_line_as_its_first_moves():
    games = await play_match(
        FirstLegalPlayer("First"), FirstLegalPlayer("Second"), games=3, openings=TWO_LINES
    )

    for game in games:
        played = [chess.Move.from_uci(move.uci) for move in game.game.moves]
        assert tuple(played[: len(game.opening.moves)]) == game.opening.moves
        assert len(played) > len(game.opening.moves), "and the players carried on from there"
        assert game.game.start_fen == chess.STARTING_FEN


async def test_two_deterministic_players_play_distinct_games():
    """Without the openings, two argmax players would play one game over and over."""
    first, second = FirstLegalPlayer("First"), LastLegalPlayer("Second")

    games = await play_match(first, second, games=6, openings=load_opening_set("standard"))

    move_lists = {tuple(move.uci for move in game.game.moves) for game in games}
    assert len(move_lists) == 6


async def test_the_set_starts_again_once_every_line_has_been_played_both_ways():
    assert distinct_games(TWO_LINES) == 4
    assert [pairing(number, TWO_LINES)[0].name for number in range(1, 7)] == [
        "King's Pawn",
        "King's Pawn",
        "Queen's Pawn",
        "Queen's Pawn",
        "King's Pawn",
        "King's Pawn",
    ]


async def test_each_game_is_handed_over_as_soon_as_it_is_over():
    seen: list[MatchGame] = []

    games = await play_match(
        FirstLegalPlayer("First"),
        FirstLegalPlayer("Second"),
        games=3,
        openings=TWO_LINES,
        on_game=seen.append,
    )

    assert seen == games


async def test_a_match_needs_a_game():
    with pytest.raises(ValueError, match="at least one game"):
        await play_match(
            FirstLegalPlayer("First"), FirstLegalPlayer("Second"), games=0, openings=TWO_LINES
        )


async def test_the_pgn_of_a_match_game_says_which_match_game_and_opening_it_was():
    games = await play_match(
        FirstLegalPlayer("First"), FirstLegalPlayer("Second"), games=2, openings=TWO_LINES
    )

    headers = read_back(games[1].pgn).headers
    assert headers["Event"] == EVENT
    assert headers["Round"] == "2"
    assert headers["Opening"] == "King's Pawn"
    assert headers["OpeningSet"] == "two v1"
    assert (headers["White"], headers["Black"]) == ("Second", "First")
    assert "FEN" not in headers, "the game starts where games start, opening moves and all"


async def test_saved_games_replay_to_the_positions_they_ended_in(tmp_path):
    games = await play_match(
        FirstLegalPlayer("First"), FirstLegalPlayer("Second"), games=3, openings=TWO_LINES
    )
    path = tmp_path / "games" / match_filename()
    for game in games:
        append_game(path, game)

    text = path.read_text(encoding="utf-8")
    for index, game in enumerate(games):
        replayed = replay.read(text, selected=index)
        assert len(replayed.games) == 3
        assert replayed.selected.round == str(game.number)
        assert replayed.selected.moves[-1].position.fen == game.game.position.fen
        assert replayed.selected.result == game.game.position.game_over.result


async def test_a_seeded_match_with_temperature_replays_exactly(tmp_path):
    from chess_ai.inference import ModelPlayer, load_engine
    from chess_ai.training.run_store import choose_checkpoint, open_run

    run = open_run(model_run(tmp_path / "runs").parent, "tiny")
    engines = [load_engine(run.checkpoint_path(choose_checkpoint(run, step))) for step in (2, 4)]

    async def match(seed: int) -> list[list[str]]:
        first, second = (
            ModelPlayer(engine, strategy="sample", temperature=1.0, seed=seed + side)
            for side, engine in enumerate(engines)
        )
        games = await play_match(first, second, games=2, openings=TWO_LINES)
        return [[move.uci for move in game.game.moves] for game in games]

    once, again, otherwise = await match(5), await match(5), await match(6)

    assert once == again
    assert once != otherwise, "and the seed is what decides it"


async def test_a_match_between_two_tiny_models_completes(tmp_path):
    from chess_ai.inference import ModelPlayer, load_engine
    from chess_ai.training.run_store import choose_checkpoint, open_run

    run = open_run(model_run(tmp_path / "runs").parent, "tiny")
    first, second = (
        ModelPlayer(load_engine(run.checkpoint_path(choose_checkpoint(run, step))))
        for step in (2, 4)
    )

    games = await play_match(first, second, games=2, openings=TWO_LINES)

    assert all(game.game.position.game_over is not None for game in games)
    assert [read_back(game.pgn).headers["WhiteRun"] for game in games] == ["tiny", "tiny"]
    assert [read_back(game.pgn).headers["WhiteCheckpoint"] for game in games] == ["2", "4"]


def finished(number: int, first_plays_white: bool, result: str) -> MatchGame:
    """A game of a match that ended in ``result``, with no moves needed to get there."""
    state = GameSession(FirstLegalPlayer("W"), FirstLegalPlayer("B")).state
    over = GameOver(result=result, reason="resignation" if result != "1/2-1/2" else "abort")
    state = state.model_copy(
        update={"position": state.position.model_copy(update={"game_over": over})}
    )
    return MatchGame(
        number=number,
        opening=TWO_LINES.openings[0],
        first_plays_white=first_plays_white,
        game=state,
        pgn="",
    )


async def test_the_score_is_counted_from_either_players_point_of_view():
    games = [
        finished(1, True, "1-0"),  # First wins as White.
        finished(2, False, "1-0"),  # Second wins as White.
        finished(3, True, "1/2-1/2"),
        finished(4, False, "0-1"),  # First wins as Black.
    ]

    first, second = score(games), score(games, first=False)

    assert (first.wins, first.draws, first.losses, first.points) == (2, 1, 1, 2.5)
    assert (second.wins, second.draws, second.losses, second.points) == (1, 1, 2, 1.5)
    assert first.fraction == pytest.approx(0.625)
    assert first.margin == pytest.approx(second.margin)


async def test_the_score_is_counted_by_colour_too():
    games = [
        finished(1, True, "1-0"),
        finished(2, False, "1-0"),
        finished(3, True, "1-0"),
        finished(4, False, "1/2-1/2"),
    ]

    as_white = score(games, color=chess.WHITE)
    as_black = score(games, color=chess.BLACK)
    second_as_white = score(games, first=False, color=chess.WHITE)

    assert (as_white.games, as_white.points) == (2, 2.0)
    assert (as_black.games, as_black.points) == (2, 0.5)
    assert (second_as_white.games, second_as_white.points) == (2, 1.5)


async def test_the_range_of_the_score_narrows_with_more_games():
    """Fifty decisive games, half won, give the ±14 points the help promises."""
    split = [finished(n, True, "1-0" if n % 2 else "0-1") for n in range(1, 51)]
    longer = [finished(n, True, "1-0" if n % 2 else "0-1") for n in range(1, 201)]

    assert score(split).margin == pytest.approx(0.14, abs=0.005)
    assert score(longer).margin == pytest.approx(0.07, abs=0.005)


async def test_identical_results_do_not_claim_a_range_of_nothing():
    all_won = score([finished(n, True, "1-0") for n in range(1, 9)])

    assert all_won.fraction == 1.0
    assert all_won.margin == pytest.approx(0.22, abs=0.01)


async def test_one_game_has_no_range():
    assert score([finished(1, True, "1-0")]).margin is None
    assert score([]).fraction is None


async def test_match_files_are_named_to_sort_among_the_saved_games():
    from datetime import datetime

    names = [match_filename(datetime(2026, 10, 3, 12, 0, second)) for second in (1, 1, 2)]

    assert names[0] != names[1], "two matches begun in the same second"
    assert all(name.startswith("20261003-1200") and name.endswith(".pgn") for name in names)
    assert [name[:15] for name in names] == sorted(name[:15] for name in names)
