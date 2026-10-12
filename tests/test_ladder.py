import asyncio
import errno
import json
import threading
from datetime import UTC, datetime, timedelta

import chess
import chess.pgn
import pytest
import torch
from game_helpers import PlayerBroke
from output_helpers import heard, unheard
from training_helpers import model_run

from chess_ai import replay
from chess_ai.config import LadderConfig
from chess_ai.evaluator import Evaluator, SuiteStopped, runs_lacking
from chess_ai.inference import load_engine
from chess_ai.ladder import (
    EVENT,
    SUITE,
    LadderResult,
    LadderSuite,
    awaiting_ladder,
    games_path,
    ladder_result,
    play_ladder,
    read_games,
    read_ladder,
    result_path,
    save_ladder,
    start_games_file,
)
from chess_ai.ladder import stands as ladder_stands
from chess_ai.match import append_game
from chess_ai.openings import Opening, OpeningSet
from chess_ai.players import GameContext, ModelDescription, PlayerMove, StockfishDescription
from chess_ai.probes import DivergedError
from chess_ai.rating import Beyond
from chess_ai.sample_games import LIVE_FILE, LiveGame, read_live_game
from chess_ai.training import run_store
from chess_ai.training.run_store import (
    STATUS_FILE,
    RunError,
    RunStatus,
    evaluation_lock,
    open_run,
)

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


class LevelStockfish:
    """A level of a ladder for a real checkpoint to play: the last legal move in UCI order.

    Reads the live file whenever it is asked to move, as a viewer polling it would, and can set
    ``stop`` on its third move and then think for a minute, as a ladder a long way from its end
    is, for the evaluator to be stopped in the middle of.
    """

    def __init__(self, elo: int, *, runs=None, stop: threading.Event | None = None) -> None:
        self.name = f"Stockfish {elo}"
        self.stockfish = StockfishDescription(
            elo=elo, requested_elo=elo, min_elo=1350, max_elo=2850, move_time=0.1
        )
        self.runs = runs
        self.stop = stop
        self.moves = 0
        self.seen: list[LiveGame | None] = []
        self.closed = False

    async def choose_move(self, context: GameContext) -> PlayerMove:
        self.moves += 1
        if self.runs is not None:
            self.seen.append(read_live_game(self.runs, now=datetime.now(UTC), stale_after=60))
        if self.stop is not None and self.moves == 3:
            self.stop.set()
            await asyncio.sleep(60)
        return PlayerMove(max(context.board.legal_moves, key=lambda move: move.uci()))

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


SETTINGS = LadderConfig(levels=(1350, 1500), games_per_level=2, rating=1500, every=4)


def ladder_suite(
    *, runs=None, stop=None, settings=SETTINGS, said=None, load=None, release=None
) -> tuple[LadderSuite, list[LevelStockfish]]:
    made: list[LevelStockfish] = []

    async def start(elo: int, move_time: float) -> LevelStockfish:
        assert move_time == settings.move_time
        made.append(LevelStockfish(elo, runs=runs, stop=stop))
        return made[-1]

    suite = LadderSuite(
        settings,
        FROM_THE_START,
        stockfish=start,
        load=load,
        release=release or (lambda: None),
        live=runs,
        say=(heard(said) if said is not None else unheard),
    )
    return suite, made


def checkpoint_of(run, step: int):
    return next(each for each in run.checkpoints() if each.step == step)


def ladder_checkpoint(suite: LadderSuite, run, step: int, **options) -> str:
    checkpoint = checkpoint_of(run, step)
    return suite.evaluate(
        run,
        checkpoint,
        run.checkpoint_written(checkpoint),
        load_engine(run.checkpoint_path(checkpoint)),
        **options,
    )


def test_the_suite_plays_every_level_and_saves_the_games_with_the_result(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    said: list[str] = []
    suite, made = ladder_suite(said=said)

    summary = ladder_checkpoint(suite, run, 4)

    result = read_ladder(run, 4)
    assert result is not None
    assert [(level.elo, level.games) for level in result.levels] == [(1350, 2), (1500, 2)]
    assert (result.model.run, result.model.checkpoint, result.model.rating) == ("tiny", 4, 1500)
    assert result.checkpoint_written == run.checkpoint_written(checkpoint_of(run, 4))
    assert [(game.index, game.elo, game.model_color) for game in result.games] == [
        (0, 1350, "white"),
        (1, 1350, "black"),
        (2, 1500, "white"),
        (3, 1500, "black"),
    ]
    text = read_games(run, 4)
    assert text is not None
    games = [replay.read(text, selected=index).selected for index in range(4)]
    assert [len(game.moves) for game in games] == [game.plies for game in result.games]
    assert summary.startswith(f"Elo {result.estimate.describe()}, ")
    assert summary.endswith(f"/4 points at {SETTINGS.move_time:g} s a move")
    assert [stockfish.closed for stockfish in made] == [True, True]
    assert not list(run.evaluation_directory(4).glob("*.partial"))
    assert suite.stands(run, checkpoint_of(run, 4), run.checkpoint_written(checkpoint_of(run, 4)))


def test_the_suite_says_when_each_game_begins_and_how_it_ended(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    said: list[str] = []
    suite, _ = ladder_suite(said=said)

    ladder_checkpoint(suite, run, 4)

    assert said[0] == "tiny@4 stockfish-ladder: game 1/4 begins: tiny@4 – Stockfish 1350"
    assert said[1].startswith("tiny@4 stockfish-ladder: game 1/4, Start: tiny@4 – Stockfish 1350 ")
    assert said[2] == "tiny@4 stockfish-ladder: game 2/4 begins: Stockfish 1350 – tiny@4"
    assert said[6] == "tiny@4 stockfish-ladder: game 4/4 begins: Stockfish 1500 – tiny@4"
    assert said[7].startswith("tiny@4 stockfish-ladder: game 4/4, Start: Stockfish 1500 – tiny@4 ")
    assert said[7].endswith(" moves)")
    assert len(said) == 8


def test_the_ladder_games_can_be_watched_as_they_are_played(tmp_path):
    runs = tmp_path / "runs"
    run = open_run(runs, model_run(runs, steps=(4,)).name)
    suite, made = ladder_suite(runs=runs)

    ladder_checkpoint(suite, run, 4)

    first = made[0].seen[0]
    assert first is not None
    assert (first.suite, first.run, first.step, first.index, first.games) == (
        "stockfish-ladder",
        "tiny",
        4,
        0,
        4,
    )
    assert (first.opponent, first.level) == ("stockfish", 1350)
    last = made[1].seen[-1]
    assert (last.index, last.level) == (3, 1500)
    assert not (runs / LIVE_FILE).exists()


def test_the_suite_plays_a_checkpoint_it_loads_itself_and_lets_go_of_it(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    loaded: list = []
    released: list = []

    def load(path):
        loaded.append(path)
        return load_engine(path)

    suite, _ = ladder_suite(load=load, release=lambda: released.append(True))
    checkpoint = checkpoint_of(run, 4)

    class Unused:
        def __getattr__(self, name):
            raise AssertionError("the evaluator's engine is not the one the ladder plays")

    suite.evaluate(run, checkpoint, run.checkpoint_written(checkpoint), Unused())

    assert loaded == [run.checkpoint_path(checkpoint)]
    assert released == [True]
    assert read_ladder(run, 4) is not None


def test_a_ladder_stopped_part_way_saves_nothing_and_leaves_nothing_behind(tmp_path):
    runs = tmp_path / "runs"
    run = open_run(runs, model_run(runs, steps=(4,)).name)
    stop = threading.Event()
    suite, made = ladder_suite(runs=runs, stop=stop)

    with pytest.raises(SuiteStopped):
        ladder_checkpoint(suite, run, 4, stop=stop)

    assert read_ladder(run, 4) is None
    assert not list(run.evaluation_directory(4).glob("*.partial"))
    assert not (runs / LIVE_FILE).exists()
    assert [stockfish.closed for stockfish in made] == [True]


def test_a_diverged_checkpoint_plays_no_ladder(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    suite, made = ladder_suite()
    checkpoint = checkpoint_of(run, 4)
    engine = load_engine(run.checkpoint_path(checkpoint))
    with torch.no_grad():
        for parameter in engine._model.parameters():
            parameter.fill_(float("nan"))

    with pytest.raises(DivergedError):
        suite.evaluate(run, checkpoint, run.checkpoint_written(checkpoint), engine)

    assert made == []
    assert read_ladder(run, 4) is None
    assert not list(run.evaluation_directory(4).glob("*.partial"))


def test_the_evaluator_ladders_every_so_many_steps_and_a_finished_runs_last(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, "done", steps=(2, 4, 6))
    model_run(runs, "going", steps=(2, 4, 6))
    going = open_run(runs, "going")
    beat = going.status
    (going.directory / STATUS_FILE).write_text(
        beat.model_copy(update={"status": RunStatus.RUNNING}).model_dump_json()
    )
    suite, _ = ladder_suite()

    wanted_steps = {
        name: [
            checkpoint.step
            for checkpoint in open_run(runs, name).checkpoints()
            if suite.wants(open_run(runs, name), checkpoint)
        ]
        for name in ("done", "going")
    }

    # Every 4 steps; and the finished run's last, which a run still going has yet to save.
    assert wanted_steps == {"done": [4, 6], "going": [4]}


def test_an_evaluator_with_the_ladder_leaves_the_checkpoints_it_does_not_want(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2, 4), config={"evaluation": {"suites": [SUITE]}})
    going = open_run(runs, "tiny")
    (going.directory / STATUS_FILE).write_text(
        going.status.model_copy(update={"status": RunStatus.RUNNING}).model_dump_json()
    )
    suite, _ = ladder_suite()
    worker = Evaluator(runs, {SUITE: suite}, load=load_engine)

    assert [job.checkpoint.step for job in worker.pending()] == [4]
    assert [job.checkpoint.step for job in worker.run_once()] == [4]
    assert worker.pending() == []
    assert runs_lacking(runs, SUITE, ladder_stands, suite.wants) == []


def test_a_ladder_from_before_results_named_their_file_stands_if_played_after_it(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    suite, _ = ladder_suite()
    ladder_checkpoint(suite, run, 4)
    checkpoint = checkpoint_of(run, 4)
    written = run.checkpoint_written(checkpoint)
    result = read_ladder(run, 4)
    path = result_path(run, 4)

    def older(started: datetime) -> None:
        text = result.model_copy(update={"started": started}).model_dump(mode="json")
        del text["checkpoint_written"], text["games"]
        path.write_text(json.dumps(text))

    older(written + timedelta(seconds=1))
    assert suite.stands(run, checkpoint, written)
    older(written - timedelta(seconds=1))
    assert not suite.stands(run, checkpoint, written)
    # Nor does one about the file a checkpoint of the same step replaced.
    ladder_checkpoint(suite, run, 4)
    assert not suite.stands(run, checkpoint, written + timedelta(seconds=1))


def test_evaluate_ladders_the_checkpoints_the_evaluator_would_unless_one_is_named(tmp_path, capsys):
    from stockfish_helpers import fake_engine

    from chess_ai.cli import main

    runs = tmp_path / "runs"
    run = open_run(runs, model_run(runs, steps=(2, 4)).name)
    engine = fake_engine(tmp_path / "engine")
    (tmp_path / "chess-ai.toml").write_text(
        f'[paths]\nruns = "{runs}"\n[stockfish]\npath = "{engine}"\n'
        "[ladder]\nlevels = [1350]\ngames_per_level = 1\nmove_time = 0.01\nevery = 4\n"
        'device = "cpu"\n'
    )
    config = ["--config", str(tmp_path / "chess-ai.toml")]

    assert main([*config, "evaluate", "tiny", "--suite", SUITE]) == 0

    out = capsys.readouterr().out
    assert f"tiny@2 {SUITE}: skipped" in out
    assert f"tiny@4 {SUITE}: game 1/1 begins: tiny@4 – Stockfish 1350" in out
    assert f"tiny@4 {SUITE}: Elo " in out
    assert read_ladder(run, 2) is None
    assert read_ladder(run, 4) is not None

    assert main([*config, "evaluate", "tiny@2", "--suite", SUITE]) == 0

    assert read_ladder(run, 2) is not None


class PruningStockfish(LevelStockfish):
    """Runs ``on_third_move`` on its third move, as retention pruning the checkpoint or the run
    being started again under its name would happen while a ladder is being played."""

    def __init__(self, elo: int, on_third_move) -> None:
        super().__init__(elo)
        self.on_third_move = on_third_move

    async def choose_move(self, context: GameContext) -> PlayerMove:
        if self.moves == 2:
            self.on_third_move()
        return await super().choose_move(context)


def pruning_suite(on_third_move, **options) -> LadderSuite:
    async def start(elo: int, move_time: float) -> LevelStockfish:
        return PruningStockfish(elo, on_third_move)

    return LadderSuite(SETTINGS, FROM_THE_START, stockfish=start, **options)


def test_a_ladder_of_a_checkpoint_pruned_while_it_was_played_is_kept(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    checkpoint = checkpoint_of(run, 4)
    written = run.checkpoint_written(checkpoint)
    engine = load_engine(run.checkpoint_path(checkpoint))
    # As retention deletes it when the run saves newer ones: the run is the same run.
    suite = pruning_suite(lambda: run.checkpoint_path(checkpoint).unlink(missing_ok=True))

    suite.evaluate(run, checkpoint, written, engine)

    result = read_ladder(run, 4)
    assert result is not None and result.checkpoint_written == written
    assert suite.stands(run, checkpoint, written)


def test_a_ladder_of_a_run_started_again_under_its_name_meanwhile_is_not_saved(tmp_path):
    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    checkpoint = checkpoint_of(run, 4)
    written = run.checkpoint_written(checkpoint)
    engine = load_engine(run.checkpoint_path(checkpoint))

    def started_again() -> None:
        run.checkpoint_path(checkpoint).unlink(missing_ok=True)
        info = json.loads((run.directory / "run.json").read_text())
        info["created"] = (written + timedelta(hours=1)).isoformat()
        (run.directory / "run.json").write_text(json.dumps(info))

    with pytest.raises(run_store.CheckpointChanged):
        pruning_suite(started_again).evaluate(run, checkpoint, written, engine)

    assert read_ladder(run, 4) is None
    assert not list(run.evaluation_directory(4).glob("*.partial"))


def test_a_checkpoint_pruned_before_the_ladder_loads_it_is_passed_over(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(4,), config={"evaluation": {"suites": [SUITE]}})
    run = open_run(runs, "tiny")
    warned: list[str] = []

    def pruned_first(path):
        path.unlink()
        return load_engine(path)

    suite, made = ladder_suite(load=pruned_first)
    worker = Evaluator(runs, {SUITE: suite}, load=load_engine, warn=warned.append)

    assert [job.checkpoint.step for job in worker.run_once()] == [4]

    assert warned == []
    assert made == []
    assert read_ladder(run, 4) is None


@pytest.mark.parametrize("fails", [False, True])
def test_the_suite_lets_go_of_what_it_loaded_before_releasing_it(tmp_path, fails):
    import gc
    import weakref

    run = open_run(tmp_path, model_run(tmp_path, steps=(4,)).name)
    checkpoint = checkpoint_of(run, 4)
    loaded: list = []
    held: list[bool] = []

    def load(path):
        engine = load_engine(path)
        loaded.append(weakref.ref(engine))
        return engine

    def release() -> None:
        gc.collect()
        held.append(loaded[0]() is not None)

    async def start(elo: int, move_time: float) -> LevelStockfish:
        if fails:
            raise RuntimeError("no engine")
        return LevelStockfish(elo)

    suite = LadderSuite(SETTINGS, FROM_THE_START, stockfish=start, load=load, release=release)
    written = run.checkpoint_written(checkpoint)
    engine = load_engine(run.checkpoint_path(checkpoint))

    if fails:
        with pytest.raises(RuntimeError):
            suite.evaluate(run, checkpoint, written, engine)
    else:
        suite.evaluate(run, checkpoint, written, engine)

    assert held == [False]


def test_a_checkpoint_is_awaiting_its_ladder_until_one_stands(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2, 4), config={"evaluation": {"suites": [SUITE]}})
    model_run(runs, "other", steps=(2, 4), config={"evaluation": {"suites": []}})
    run = open_run(runs, "tiny")
    awaiting = awaiting_ladder(run, every=4)

    assert [awaiting(checkpoint) for checkpoint in run.checkpoints()] == [False, True]

    suite, _ = ladder_suite()
    ladder_checkpoint(suite, run, 4)

    assert not awaiting(checkpoint_of(run, 4))
    other = open_run(runs, "other")
    assert not awaiting_ladder(other, every=4)(checkpoint_of(other, 4)), "not asked for"


class FileLooker:
    """A suite listed after the ladder, which finds out whether its checkpoint's file is still
    there when its turn comes, as sample games need it to be."""

    name = "probe-positions"

    def __init__(self) -> None:
        self.found: dict[int, bool] = {}

    def wants(self, run, checkpoint) -> bool:
        return True

    def stands(self, run, checkpoint, written) -> bool:
        return checkpoint.step in self.found

    def evaluate(self, run, checkpoint, written, engine, *, stop=None) -> str:
        self.found[checkpoint.step] = run.checkpoint_written(checkpoint) == written
        return "looked"

    def after(self, run) -> None:
        pass


def test_a_finished_run_lets_go_of_a_laddered_checkpoint_once_its_job_is_over(tmp_path):
    runs = tmp_path / "runs"
    # The ladder before another suite of the same job, as a config may list them.
    model_run(runs, steps=(2, 4, 6), config={"evaluation": {"suites": [SUITE, "probe-positions"]}})
    run = open_run(runs, "tiny")
    index = run.directory / "checkpoints" / "index.json"
    written = json.loads(index.read_text())
    written["policy"]["keep"] = 1
    index.write_text(json.dumps(written))
    suite, _ = ladder_suite(settings=SETTINGS.model_copy(update={"every": 4}))
    looker = FileLooker()
    warned: list[str] = []
    worker = Evaluator(
        runs, {SUITE: suite, looker.name: looker}, load=load_engine, warn=warned.append
    )
    job = next(job for job in worker.pending() if job.checkpoint.step == 4)

    assert worker.evaluate(job)

    assert looker.found[4], "the checkpoint was pruned before the job's other suite"
    # The newest and the best (the first, in these runs) are the run's to keep; 4 was held for
    # its ladder alone, and its result stays.
    assert [checkpoint.step for checkpoint in run.checkpoints()] == [2, 6]
    assert read_ladder(run, 4) is not None
    assert warned == []


def test_a_later_job_without_the_ladder_still_lets_go_of_a_laddered_checkpoint(tmp_path):
    """As after an evaluator stopped between a checkpoint's ladder and its other suites: the
    job that finishes them names no ladder."""
    runs = tmp_path / "runs"
    model_run(runs, steps=(2, 4, 6), config={"evaluation": {"suites": [SUITE, "probe-positions"]}})
    run = open_run(runs, "tiny")
    index = run.directory / "checkpoints" / "index.json"
    written = json.loads(index.read_text())
    written["policy"]["keep"] = 1
    index.write_text(json.dumps(written))
    suite, _ = ladder_suite(settings=SETTINGS.model_copy(update={"every": 4}))
    ladder_checkpoint(suite, run, 4)
    looker = FileLooker()
    worker = Evaluator(runs, {SUITE: suite, looker.name: looker}, load=load_engine)
    job = next(job for job in worker.pending() if job.checkpoint.step == 4)
    assert job.suites == (looker.name,)

    assert worker.evaluate(job)

    assert [checkpoint.step for checkpoint in run.checkpoints()] == [2, 6]
