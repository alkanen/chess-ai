import errno
import threading
from datetime import UTC, datetime

import chess
import chess.pgn
import pytest
from game_helpers import PlayerBroke
from training_helpers import model_run

from chess_ai.ladder import (
    EVENT,
    LadderResult,
    games_path,
    ladder_result,
    play_ladder,
    result_path,
    save_ladder,
    start_games_file,
)
from chess_ai.match import append_game
from chess_ai.openings import Opening, OpeningSet
from chess_ai.players import GameContext, ModelDescription, PlayerMove, StockfishDescription
from chess_ai.rating import Beyond
from chess_ai.training import run_store
from chess_ai.training.run_store import RunError, evaluation_lock, open_run

pytestmark = pytest.mark.anyio

FROM_THE_START = OpeningSet(name="start", version=1, openings=(Opening("Start", ()),))
"""One line of no moves, so that the scripted mates below are played from the first move."""

MATES = {
    # Scholar's mate, against a Black who only pushes rook's pawns.
    "wins": {chess.WHITE: ["e4", "Bc4", "Qh5", "Qxf7#"], chess.BLACK: ["e5", "Qh4#"]},
    # Fool's mate, against a White who only pushes knight's and bishop's pawns.
    "loses": {chess.WHITE: ["f3", "g4"], chess.BLACK: ["a6", "a5", "h6"]},
}
"""Moves that make a game end at once: a player who ``wins`` plays the side that mates and one
who ``loses`` the side that is mated, so that any pair of them plays a game with a known end."""

MODEL = ModelDescription(run="tiny", checkpoint=4, rating=2000)


class MatingPlayer:
    """Plays one of :data:`MATES`, by the colour it has."""

    def __init__(self, name: str, outcome: str) -> None:
        self.name = name
        self.moves = MATES[outcome]

    async def choose_move(self, context: GameContext) -> PlayerMove:
        board = context.board
        san = self.moves[board.turn][len(board.move_stack) // 2]
        return PlayerMove(board.parse_san(san))


class MatingModel(MatingPlayer):
    @property
    def model(self) -> ModelDescription:
        return MODEL


class FakeStockfish(MatingPlayer):
    """A level of the ladder that plays :data:`MATES` and remembers whether it was closed."""

    def __init__(self, elo: int, outcome: str, *, requested: int | None = None) -> None:
        super().__init__(f"Stockfish {elo}", outcome)
        self.stockfish = StockfishDescription(
            elo=elo,
            requested_elo=requested if requested is not None else elo,
            min_elo=1350,
            max_elo=2850,
            move_time=0.1,
        )
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class FailingPlayer:
    name = "Failing"

    async def choose_move(self, context: GameContext) -> PlayerMove:
        raise PlayerBroke("the player is gone")


class Opponents:
    """Makes the levels of a ladder, which win at and above ``strong_from`` and lose below it.

    Each level has ``model`` play the other side of the mate, if it is a :class:`MatingPlayer`.
    """

    def __init__(self, model: object = None, strong_from: int = 10_000) -> None:
        self.model = model
        self.strong_from = strong_from
        self.made: list[FakeStockfish] = []

    async def __call__(self, elo: int) -> FakeStockfish:
        strong = elo >= self.strong_from
        if isinstance(self.model, MatingPlayer):
            self.model.moves = MATES["loses" if strong else "wins"]
        self.made.append(FakeStockfish(elo, "wins" if strong else "loses"))
        return self.made[-1]


async def test_each_level_is_a_match_played_weakest_first_against_an_engine_of_its_own():
    opponents = Opponents()
    handed: list[tuple[int, int]] = []

    played = await play_ladder(
        MatingModel("model", "wins"),
        [1350, 1500, 1700],
        games=2,
        openings=FROM_THE_START,
        opponent=opponents,
        on_game=lambda level, game: handed.append((level.elo, game.number)),
    )

    assert [level.stockfish.elo for level in played] == [1350, 1500, 1700]
    assert [len(level.games) for level in played] == [2, 2, 2]
    assert handed == [(1350, 1), (1350, 2), (1500, 1), (1500, 2), (1700, 1), (1700, 2)]
    assert [opponent.closed for opponent in opponents.made] == [True, True, True]


async def test_ladder_games_say_they_were_played_on_a_ladder():
    (level,) = await play_ladder(
        MatingModel("model", "wins"),
        [1350],
        games=2,
        openings=FROM_THE_START,
        opponent=Opponents(),
    )

    for game in level.games:
        assert game.pgn.startswith(f'[Event "{EVENT}"]')
        assert game.first_score == 1.0


async def test_an_opponent_is_closed_when_its_level_fails():
    opponents = Opponents()

    with pytest.raises(PlayerBroke):
        await play_ladder(
            FailingPlayer(), [1350, 1500], games=2, openings=FROM_THE_START, opponent=opponents
        )

    assert [opponent.closed for opponent in opponents.made] == [True]


async def test_a_level_has_to_be_stockfish():
    async def not_stockfish(elo: int) -> MatingPlayer:
        return MatingPlayer("Random mover", "loses")

    with pytest.raises(TypeError, match="Random mover cannot be a level of a ladder"):
        await play_ladder(
            MatingModel("model", "wins"),
            [1350],
            games=1,
            openings=FROM_THE_START,
            opponent=not_stockfish,
        )


async def test_a_ladder_needs_a_level():
    with pytest.raises(ValueError, match="at least one level"):
        await play_ladder(
            MatingModel("model", "wins"), [], games=1, openings=FROM_THE_START, opponent=Opponents()
        )


async def ladder_of(strong_from: int, levels=(1350, 1500, 1700), games=2) -> LadderResult:
    model = MatingModel("model", "wins")
    played = await play_ladder(
        model,
        list(levels),
        games=games,
        openings=FROM_THE_START,
        opponent=Opponents(model, strong_from),
    )
    now = datetime.now(UTC)
    return ladder_result(
        MODEL,
        played,
        openings=FROM_THE_START,
        games_per_level=games,
        started=now,
        finished=now,
        code_version="test",
    )


async def test_the_result_counts_each_level_and_estimates_between_the_levels():
    # Wins every game against 1350, loses every one against 1700.
    result = await ladder_of(strong_from=1700, levels=(1350, 1700), games=4)

    assert [(level.elo, level.wins, level.losses) for level in result.levels] == [
        (1350, 4, 0),
        (1700, 0, 4),
    ]
    assert result.estimate.rating == pytest.approx(1525, abs=0.01), "halfway, by symmetry"
    assert result.estimate.low < 1525 < result.estimate.high
    assert result.estimate.beyond is None
    assert (result.estimate.games, result.estimate.points) == (8, 4.0)
    assert result.model == MODEL
    assert result.openings == "start v1"


async def test_a_checkpoint_that_loses_every_game_is_below_the_floor():
    result = await ladder_of(strong_from=0)

    assert result.estimate.beyond is Beyond.FLOOR
    assert result.estimate.rating is None
    assert result.estimate.describe().startswith("below 1350")


async def test_a_checkpoint_that_wins_every_game_is_above_the_ceiling():
    result = await ladder_of(strong_from=10_000)

    assert result.estimate.beyond is Beyond.CEILING
    assert result.estimate.describe().startswith("above 1700")


async def test_a_level_stockfish_could_not_play_is_counted_at_the_one_it_did():
    async def clamped(elo: int) -> FakeStockfish:
        return FakeStockfish(max(elo, 1350), "loses", requested=elo)

    played = await play_ladder(
        MatingModel("model", "wins"), [1000], games=1, openings=FROM_THE_START, opponent=clamped
    )
    now = datetime.now(UTC)
    result = ladder_result(
        MODEL,
        played,
        openings=FROM_THE_START,
        games_per_level=1,
        started=now,
        finished=now,
        code_version="test",
    )

    (level,) = result.levels
    assert (level.elo, level.requested_elo) == (1350, 1000)
    assert result.estimate.floor == 1350


async def test_a_saved_ladder_has_its_games_beside_it_and_no_partial_file(tmp_path):
    model_run(tmp_path)
    run = open_run(tmp_path, "tiny")
    played = await play_ladder(
        MatingModel("model", "wins"),
        [1350],
        games=2,
        openings=FROM_THE_START,
        opponent=Opponents(),
    )
    partial = start_games_file(run, 4)
    for game in played[0].games:
        append_game(partial, game)
    now = datetime.now(UTC)
    result = ladder_result(
        MODEL,
        played,
        openings=FROM_THE_START,
        games_per_level=2,
        started=now,
        finished=now,
        code_version="test",
    )

    save_ladder(run, result, partial)

    assert result_path(run, 4) == tmp_path / "tiny/evaluations/step-000000004/stockfish-ladder.json"
    saved = LadderResult.model_validate_json(result_path(run, 4).read_text())
    assert saved == result
    with games_path(run, 4).open() as file:
        rounds = [game.headers["Round"] for game in iter(lambda: chess.pgn.read_game(file), None)]
    assert rounds == ["1", "2"]
    assert not partial.exists()


async def test_each_ladder_writes_its_games_to_a_file_of_its_own(tmp_path):
    model_run(tmp_path)
    run = open_run(tmp_path, "tiny")

    first, second = start_games_file(run, 4), start_games_file(run, 4)

    assert first != second
    assert first.read_text() == second.read_text() == ""
    assert first.name.startswith("stockfish-ladder.")
    assert first.name.endswith(".pgn.partial")


async def saved_ladder_parts(tmp_path) -> tuple:
    """The tiny run, a ladder's result about its step 4, and a partial file of its games."""
    model_run(tmp_path)
    run = open_run(tmp_path, "tiny")
    played = await play_ladder(
        MatingModel("model", "wins"),
        [1350],
        games=1,
        openings=FROM_THE_START,
        opponent=Opponents(),
    )
    partial = start_games_file(run, 4)
    append_game(partial, played[0].games[0])
    now = datetime.now(UTC)
    result = ladder_result(
        MODEL,
        played,
        openings=FROM_THE_START,
        games_per_level=1,
        started=now,
        finished=now,
        code_version="test",
    )
    return run, result, partial


async def test_a_ladder_is_saved_whole_while_no_other_saves_about_the_same_checkpoint(tmp_path):
    """Two ladders finishing together must not leave one's result beside the other's games."""
    run, result, partial = await saved_ladder_parts(tmp_path)
    saving = threading.Thread(target=save_ladder, args=(run, result, partial))

    with evaluation_lock(run.evaluation_directory(4)):
        saving.start()
        saving.join(timeout=0.5)
        assert saving.is_alive(), "it waits for the save already under way"
        assert partial.exists()
        assert not games_path(run, 4).exists()
    saving.join(timeout=5)

    assert not saving.is_alive()
    assert result_path(run, 4).exists()
    assert games_path(run, 4).exists()


async def test_a_result_that_cannot_be_written_moves_no_games(tmp_path, monkeypatch):
    run, result, partial = await saved_ladder_parts(tmp_path)
    games_path(run, 4).write_text("an earlier ladder's games\n")
    result_path(run, 4).write_text("an earlier ladder's result\n")

    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(run_store, "sync_file", disk_full)
    with pytest.raises(RunError, match="No space left on device"):
        save_ladder(run, result, partial)

    assert partial.exists(), "the games are where the ladder said they were"
    assert games_path(run, 4).read_text() == "an earlier ladder's games\n"
    assert result_path(run, 4).read_text() == "an earlier ladder's result\n"
    assert sorted(path.name for path in run.evaluation_directory(4).iterdir()) == sorted(
        [partial.name, "stockfish-ladder.json", "stockfish-ladder.pgn", "evaluation.lock"]
    ), "and no temporary file is left"
