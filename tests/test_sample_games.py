import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta

import chess
import chess.pgn
import pytest
import torch
from training_helpers import add_checkpoint, model_run

from chess_ai import replay
from chess_ai.config import SampleGamesConfig
from chess_ai.evaluator import Evaluator, Stopped
from chess_ai.game_session import GameState
from chess_ai.inference import load_engine
from chess_ai.match import distinct_games, pairing
from chess_ai.openings import load_opening_set
from chess_ai.players import GameContext, PlayerMove, StockfishDescription
from chess_ai.probes import DivergedError
from chess_ai.sample_games import (
    EVENT,
    LIVE_FILE,
    SUITE,
    LiveGame,
    SampleGamesSuite,
    clear_live_game,
    games_path,
    read_games,
    read_live_game,
    read_sample_games,
    save_live_game,
)
from chess_ai.training.run_store import (
    ARCHIVED_TAG,
    TEMPORARY_SUFFIX,
    RunNotes,
    RunReader,
    save_notes,
)
from chess_ai.web.runs import LiveGameEvent, RunStream

OPENINGS = load_opening_set("standard")


class FakeStockfish:
    """Plays the last legal move in UCI order, says it is Stockfish, and reads the live file
    whenever it is asked to move, as a viewer polling it would."""

    name = "Stockfish 1350"

    def __init__(self, runs=None) -> None:
        self.stockfish = StockfishDescription(
            elo=1350, requested_elo=1300, min_elo=1350, max_elo=2850, move_time=0.1
        )
        self.runs = runs
        self.seen: list[tuple[int, LiveGame | None]] = []
        self.closed = False

    async def choose_move(self, context: GameContext) -> PlayerMove:
        if self.runs is not None:
            live = read_live_game(self.runs, now=datetime.now(UTC), stale_after=60)
            self.seen.append((len(context.board.move_stack), live))
        return PlayerMove(max(context.board.legal_moves, key=lambda move: move.uci()))

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


def suite(*, games=2, stockfish_games=0, opponent=None, live=None) -> SampleGamesSuite:
    async def start(elo: int, move_time: float):
        return opponent

    return SampleGamesSuite(
        SampleGamesConfig(games=games, stockfish_games=stockfish_games, rating=1500),
        OPENINGS,
        stockfish=start if stockfish_games else None,
        live=live,
    )


def play(chosen: SampleGamesSuite, run: RunReader, step: int, *, engine=None) -> str:
    checkpoint = next(each for each in run.checkpoints() if each.step == step)
    return chosen.evaluate(
        run,
        checkpoint,
        run.checkpoint_written(checkpoint),
        engine or load_engine(run.checkpoint_path(checkpoint)),
    )


def test_against_itself_every_game_starts_from_a_line_of_its_own():
    assert [pairing(number, OPENINGS, paired=False) for number in (1, 2, 3)] == [
        (OPENINGS.openings[0], True),
        (OPENINGS.openings[1], True),
        (OPENINGS.openings[2], True),
    ]
    assert distinct_games(OPENINGS, paired=False) == len(OPENINGS.openings)


def test_a_deterministic_checkpoint_plays_distinct_games_against_itself(tmp_path):
    run = RunReader(model_run(tmp_path / "runs", steps=(2,)))

    said = play(suite(games=3), run, 2)

    result = read_sample_games(run, 2)
    assert result is not None
    assert [game.opponent for game in result.games] == ["self"] * 3
    assert [game.model_color for game in result.games] == ["both"] * 3
    # Each from another line, so that argmax does not play one game three times.
    assert [game.opening for game in result.games] == [line.name for line in OPENINGS.openings[:3]]
    text = read_games(run, 2)
    lines = [replay.read(text, selected=index).selected for index in range(3)]
    played = [tuple(move.uci for move in game.moves) for game in lines]
    assert len(set(played)) == 3
    assert said.startswith("3 games against itself (")
    assert (result.model.run, result.model.checkpoint, result.model.rating) == ("tiny", 2, 1500)
    assert result.stockfish is None


def test_against_stockfish_the_colours_alternate_and_each_line_is_played_both_ways(tmp_path):
    run = RunReader(model_run(tmp_path / "runs", steps=(2,)))
    stockfish = FakeStockfish()

    said = play(suite(games=1, stockfish_games=4, opponent=stockfish), run, 2)

    result = read_sample_games(run, 2)
    assert [(game.opponent, game.model_color, game.opening) for game in result.games] == [
        ("self", "both", OPENINGS.openings[0].name),
        ("stockfish", "white", OPENINGS.openings[0].name),
        ("stockfish", "black", OPENINGS.openings[0].name),
        ("stockfish", "white", OPENINGS.openings[1].name),
        ("stockfish", "black", OPENINGS.openings[1].name),
    ]
    assert [game.index for game in result.games] == [0, 1, 2, 3, 4]
    assert [(game.white, game.black) for game in result.games[1:3]] == [
        ("tiny step 2", "Stockfish 1350"),
        ("Stockfish 1350", "tiny step 2"),
    ]
    assert result.stockfish == stockfish.stockfish
    assert stockfish.closed
    assert "4 games against Stockfish 1350 (scored " in said


def test_the_saved_games_replay_to_the_positions_they_ended_in(tmp_path):
    run = RunReader(model_run(tmp_path / "runs", steps=(2,)))

    play(suite(games=2, stockfish_games=2, opponent=FakeStockfish()), run, 2)

    result = read_sample_games(run, 2)
    text = read_games(run, 2)
    assert text == games_path(run, 2).read_text()
    written = io.StringIO(text)
    for index, listed in enumerate(result.games):
        game = replay.read(text, selected=index).selected
        assert game.result == listed.result
        assert len(game.moves) == listed.plies
        # The moves start from the line's moves, and replay in python-chess too.
        line = OPENINGS.openings[[0, 1, 0, 0][index]].moves
        assert [move.uci for move in game.moves[: len(line)]] == [move.uci() for move in line]
        record = chess.pgn.read_game(written)
        assert record is not None and record.errors == []
        assert record.headers["Event"] == EVENT
        assert record.end().board().fen() == game.moves[-1].position.fen
    assert chess.pgn.read_game(written) is None


def test_a_result_stands_for_the_file_it_was_played_with_only(tmp_path):
    directory = model_run(tmp_path / "runs", steps=(2, 4))
    run = RunReader(directory)
    chosen = suite(games=1)
    play(chosen, run, 2)
    first = run.checkpoints()[0]

    assert chosen.stands(run, first, run.checkpoint_written(first))
    assert not chosen.stands(run, run.checkpoints()[1], run.checkpoint_written(first))
    assert not chosen.stands(run, first, run.checkpoint_written(first) + timedelta(seconds=1))


def test_the_evaluator_plays_the_sample_games_of_each_new_checkpoint(tmp_path):
    runs = tmp_path / "runs"
    directory = model_run(runs, steps=(2,))
    said: list[str] = []
    worker = Evaluator(runs, {SUITE: suite(games=1)}, load=load_engine, say=said.append)

    worker.run_once()
    add_checkpoint(directory, 4)
    worker.run_once()

    run = RunReader(directory)
    assert [(entry.step, entry.suite) for entry in run.evaluations()] == [
        (2, SUITE),
        (4, SUITE),
    ]
    assert [line.split(":")[0] for line in said] == [f"tiny@2 {SUITE}", f"tiny@4 {SUITE}"]
    assert worker.run_once() == []


def test_a_diverged_checkpoint_plays_no_games(tmp_path):
    run = RunReader(model_run(tmp_path / "runs", steps=(2,)))
    engine = load_engine(run.checkpoint_path(run.checkpoints()[0]))
    with torch.no_grad():
        for parameter in engine._model.parameters():
            parameter.fill_(float("nan"))

    with pytest.raises(DivergedError, match="weights have most likely diverged"):
        play(suite(games=1), run, 2, engine=engine)

    assert read_sample_games(run, 2) is None
    # Found out before anything was written.
    assert not run.evaluation_directory(2).exists()


def test_stockfish_that_will_not_start_costs_no_games_and_leaves_nothing_behind(tmp_path):
    run = RunReader(model_run(tmp_path / "runs", steps=(2,)))

    async def broken(elo: int, move_time: float):
        raise RuntimeError("no engine here")

    chosen = SampleGamesSuite(
        SampleGamesConfig(games=1, stockfish_games=2), OPENINGS, stockfish=broken
    )

    with pytest.raises(RuntimeError, match="no engine here"):
        play(chosen, run, 2)

    assert read_sample_games(run, 2) is None
    assert not list(run.evaluation_directory(2).glob("*.partial"))


def test_games_cut_short_leave_no_partial_file_and_stockfish_closed(tmp_path):
    run = RunReader(model_run(tmp_path / "runs", steps=(2,)))

    class Failing(FakeStockfish):
        async def choose_move(self, context: GameContext) -> PlayerMove:
            raise RuntimeError("engine died")

    stockfish = Failing()

    with pytest.raises(RuntimeError, match="engine died"):
        play(suite(games=1, stockfish_games=2, opponent=stockfish), run, 2)

    assert stockfish.closed
    assert read_sample_games(run, 2) is None
    assert not list(run.evaluation_directory(2).glob("*.partial"))
    assert not games_path(run, 2).exists()


def test_the_game_being_played_can_be_watched_and_is_gone_once_the_games_are_over(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    stockfish = FakeStockfish(runs)

    play(suite(games=1, stockfish_games=2, opponent=stockfish, live=runs), run, 2)

    # Every time it was Stockfish's turn, the file showed the game up to the move before.
    assert stockfish.seen
    for plies, live in stockfish.seen:
        assert live is not None
        assert (live.run, live.step, live.games, live.opponent) == ("tiny", 2, 3, "stockfish")
        assert len(live.game.moves) == plies
        assert live.game.position.legal_moves == {}
        # The thoughts behind the last move only, which keeps the file small.
        assert all(move.thoughts is None for move in live.game.moves[:-1])
    assert {live.index for _, live in stockfish.seen} == {1, 2}
    assert not (runs / LIVE_FILE).exists()


def test_a_one_off_evaluation_shows_no_game(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    stockfish = FakeStockfish(runs)

    play(suite(games=0, stockfish_games=1, opponent=stockfish), run, 2)

    assert stockfish.seen and all(live is None for _, live in stockfish.seen)


def live_game(run: str, *, updated: datetime) -> LiveGame:
    from chess_ai.position_view import snapshot

    return LiveGame(
        run=run,
        step=2,
        index=0,
        games=2,
        opponent="self",
        updated=updated,
        game=GameState.model_validate(
            {
                "id": "g",
                "white": {"name": "tiny step 2", "accepts_moves": False},
                "black": {"name": "tiny step 2", "accepts_moves": False},
                "start_fen": chess.STARTING_FEN,
                "moves": [],
                "position": snapshot(chess.Board(), legal_moves=False).model_dump(),
            }
        ),
    )


def test_a_game_left_behind_by_a_killed_evaluator_is_not_shown_as_being_played(tmp_path):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    tmp_path.mkdir(exist_ok=True)
    save_live_game(tmp_path, live_game("tiny", updated=now))

    assert read_live_game(tmp_path, now=now + timedelta(seconds=60), stale_after=60) is not None
    assert read_live_game(tmp_path, now=now + timedelta(seconds=61), stale_after=60) is None
    (tmp_path / LIVE_FILE).write_text("{")
    assert read_live_game(tmp_path, now=now, stale_after=60) is None
    clear_live_game(tmp_path)
    clear_live_game(tmp_path)
    assert read_live_game(tmp_path, now=now, stale_after=60) is None


def test_a_run_stream_sends_the_game_its_own_checkpoint_is_playing(tmp_path):
    runs = tmp_path / "runs"
    now = datetime.now(UTC)
    tiny = RunStream(RunReader(model_run(runs, "tiny", steps=(2,))), stale_after=60)
    other = RunStream(RunReader(model_run(runs, "other", steps=(2,))), stale_after=60)
    *_, before = tiny.poll()
    other.poll()

    save_live_game(runs, live_game("tiny", updated=now))
    [playing] = tiny.poll()
    nothing_new = tiny.poll()
    elsewhere = other.poll()
    clear_live_game(runs)
    [over] = tiny.poll()

    assert before == LiveGameEvent(live=None)
    assert isinstance(playing, LiveGameEvent)
    assert playing.live is not None and playing.live.run == "tiny"
    assert nothing_new == []
    assert elsewhere == []
    assert over == LiveGameEvent(live=None)


def test_the_live_file_is_json_a_browser_can_read(tmp_path):
    save_live_game(tmp_path, live_game("tiny", updated=datetime.now(UTC)))

    written = json.loads((tmp_path / LIVE_FILE).read_text())

    assert written["run"] == "tiny"
    assert not list(tmp_path.glob(f"*{TEMPORARY_SUFFIX}"))


def test_settings_that_play_no_games_are_refused():
    with pytest.raises(ValueError, match="cannot both be 0"):
        SampleGamesConfig(games=0, stockfish_games=0)


def test_evaluate_plays_sample_games_against_stockfish_and_closes_it(tmp_path, capsys):
    from stockfish_helpers import fake_engine, started, wait_until_gone

    from chess_ai.cli import main

    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    engine = fake_engine(tmp_path / "engine")
    (tmp_path / "chess-ai.toml").write_text(
        f'[paths]\nruns = "{runs}"\n[stockfish]\npath = "{engine}"\n'
        "[sample_games]\ngames = 1\nstockfish_games = 2\nstockfish_move_time = 0.01\n"
    )

    assert (
        main(["--config", str(tmp_path / "chess-ai.toml"), "evaluate", "tiny", "--suite", SUITE])
        == 0
    )

    result = read_sample_games(run, 2)
    assert [game.opponent for game in result.games] == ["self", "stockfish", "stockfish"]
    assert f"tiny@2 {SUITE}: 1 game against itself" in capsys.readouterr().out
    assert len(started(engine)) == 1
    wait_until_gone(*started(engine))
    # Only the evaluator shows the game being played.
    assert not (runs / LIVE_FILE).exists()


def test_evaluate_says_so_when_stockfish_will_not_start(tmp_path, capsys):
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    (tmp_path / "chess-ai.toml").write_text(
        f'[paths]\nruns = "{runs}"\n[stockfish]\npath = "{tmp_path / "nothing"}"\n'
        "[sample_games]\nstockfish_games = 2\n"
    )

    with pytest.raises(SystemExit) as exited:
        main(["--config", str(tmp_path / "chess-ai.toml"), "evaluate", "tiny", "--suite", SUITE])

    assert exited.value.code == 2
    assert f"tiny@2 {SUITE}: " in capsys.readouterr().err


def test_an_evaluator_starting_clears_a_game_a_killed_one_left_behind(tmp_path):
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    model_run(runs, steps=(2,), config={"evaluation": {"suites": []}})
    save_live_game(runs, live_game("tiny", updated=datetime.now(UTC)))
    (tmp_path / "chess-ai.toml").write_text(f'[paths]\nruns = "{runs}"\n')

    assert main(["--config", str(tmp_path / "chess-ai.toml"), "evaluator", "--once"]) == 0

    assert not (runs / LIVE_FILE).exists()


class StoppingStockfish(FakeStockfish):
    """Sets ``stop`` on its third move and then thinks for a minute, as a game a long way from
    its end is, for the evaluator to be stopped in the middle of."""

    def __init__(self, stop, runs=None) -> None:
        super().__init__(runs)
        self.stop = stop
        self.moves = 0

    async def choose_move(self, context: GameContext) -> PlayerMove:
        self.moves += 1
        if self.moves == 3:
            self.stop.set()
            await asyncio.sleep(60)
        return await super().choose_move(context)


def finishes(call, timeout: float = 20.0):
    """What ``call()`` returns or raises, failing the test if it takes longer than ``timeout``."""
    outcome: list = []

    def run() -> None:
        try:
            outcome.append(("returned", call()))
        except BaseException as e:  # noqa: BLE001 - handed to the test
            outcome.append(("raised", e))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert outcome, f"still running after {timeout} s"
    return outcome[0]


def test_an_evaluator_stopped_in_the_middle_of_the_games_stops_at_once(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    stop = threading.Event()
    stockfish = StoppingStockfish(stop, runs)
    worker = Evaluator(
        runs,
        {SUITE: suite(games=0, stockfish_games=2, opponent=stockfish, live=runs)},
        load=load_engine,
        warn=lambda text: pytest.fail(f"a stop is not a failure: {text}"),
    )

    kind, outcome = finishes(lambda: worker.run_once(stop=stop))

    assert kind == "raised" and isinstance(outcome, Stopped)
    assert (outcome.done, outcome.left) == ([], 1)
    assert stockfish.closed
    assert read_sample_games(run, 2) is None
    assert not list(run.evaluation_directory(2).glob("*.partial"))
    assert not (runs / LIVE_FILE).exists()
    # Not marked as failed: the next evaluator plays them.
    assert [job.checkpoint.step for job in worker.pending()] == [2]


def test_a_running_evaluator_stopped_in_the_middle_of_the_games_stops_at_once(tmp_path):
    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    stop = threading.Event()
    worker = Evaluator(
        runs,
        {SUITE: suite(games=0, stockfish_games=2, opponent=StoppingStockfish(stop))},
        load=load_engine,
    )

    kind, _ = finishes(lambda: worker.run(poll_seconds=60, stop=stop))

    assert kind == "returned"


def bad_openings_config(tmp_path, runs) -> str:
    path = tmp_path / "chess-ai.toml"
    path.write_text(
        f'[paths]\nruns = "{runs}"\n[sample_games]\nopenings = "{tmp_path / "moved.toml"}"\n'
    )
    return str(path)


def test_a_bad_opening_set_does_not_stop_probes_being_evaluated_on_demand(tmp_path):
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    config = bad_openings_config(tmp_path, runs)

    assert main(["--config", config, "evaluate", "tiny", "--suite", "probe-positions"]) == 0

    assert [entry.suite for entry in run.evaluations()] == ["probe-positions"]


def test_a_bad_opening_set_is_an_error_when_sample_games_are_asked_for(tmp_path, capsys):
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    config = bad_openings_config(tmp_path, runs)

    with pytest.raises(SystemExit) as exited:
        main(["--config", config, "evaluate", "tiny", "--suite", SUITE])

    assert exited.value.code == 2
    assert "[sample_games] openings: " in capsys.readouterr().err


def test_an_evaluator_with_a_bad_opening_set_still_probes_and_says_why_it_plays_no_games(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    config = bad_openings_config(tmp_path, runs)
    environment = {**os.environ, "CHESS_AI_EVALUATOR_POLL_SECONDS": "0.2"}
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, chess_ai.cli; sys.exit(chess_ai.cli.main())",
            "--config",
            config,
            "evaluator",
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not run.evaluations() and time.monotonic() < deadline:
            time.sleep(0.1)
        worker.send_signal(signal.SIGTERM)
        _, err = worker.communicate(timeout=30)
    finally:
        worker.kill()

    assert worker.returncode == 0, err
    assert [entry.suite for entry in run.evaluations()] == ["probe-positions"]
    assert err.count("[sample_games] openings: ") == 1
    assert "no sample games are played until the evaluator is restarted" in err


def test_evaluator_once_with_a_bad_opening_set_fails_rather_than_reporting_everything_done(
    tmp_path, capsys
):
    """``--once`` has no later to restart for, and a script checks its exit status."""
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    model_run(runs, steps=(2,))
    config = bad_openings_config(tmp_path, runs)

    with pytest.raises(SystemExit) as exited:
        main(["--config", config, "evaluator", "--once"])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "[sample_games] openings: " in err
    assert "has its results already" not in err


class SlowToClose(FakeStockfish):
    """Takes its time to exit, and the evaluator is told to stop meanwhile: after the games
    have been saved, while they are still being cleaned up after."""

    def __init__(self, stop) -> None:
        super().__init__()
        self.stop = stop

    async def wait_closed(self) -> None:
        self.stop.set()
        await asyncio.sleep(1)


def test_a_stop_after_the_games_were_saved_counts_them_as_done(tmp_path):
    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    stop = threading.Event()
    said: list[str] = []
    worker = Evaluator(
        runs,
        {SUITE: suite(games=0, stockfish_games=1, opponent=SlowToClose(stop))},
        load=load_engine,
        say=said.append,
        warn=lambda text: pytest.fail(text),
    )

    kind, outcome = finishes(lambda: worker.run_once(stop=stop))

    assert kind == "returned", outcome
    assert [job.checkpoint.step for job in outcome] == [2]
    assert read_sample_games(run, 2) is not None
    assert [line.split(":")[0] for line in said] == [f"tiny@2 {SUITE}"]


def test_evaluator_once_with_a_bad_opening_set_probes_runs_that_ask_for_no_games(tmp_path, capsys):
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    run = RunReader(
        model_run(runs, steps=(2,), config={"evaluation": {"suites": ["probe-positions"]}})
    )
    # An archived run asking for games is one the evaluator leaves alone anyway.
    archived = RunReader(model_run(runs, "old", steps=(2,)))
    save_notes(archived, RunNotes(tags=[ARCHIVED_TAG]))
    config = bad_openings_config(tmp_path, runs)

    assert main(["--config", config, "evaluator", "--once"]) == 0

    assert [entry.suite for entry in run.evaluations()] == ["probe-positions"]
    err = capsys.readouterr().err
    assert err.count("[sample_games] openings: ") == 1


def test_evaluator_once_with_a_bad_opening_set_probes_runs_whose_games_are_all_played(
    tmp_path, capsys
):
    """As when the probe set changes long after a run finished, and the openings file moved."""
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    run = RunReader(model_run(runs, steps=(2,)))
    play(suite(games=1), run, 2)
    config = bad_openings_config(tmp_path, runs)

    assert main(["--config", config, "evaluator", "--once"]) == 0

    assert [entry.suite for entry in run.evaluations()] == ["probe-positions", SUITE]
    assert capsys.readouterr().err.count("[sample_games] openings: ") == 1


def test_evaluator_once_with_a_bad_opening_set_fails_for_games_of_a_replaced_checkpoint(
    tmp_path, capsys
):
    from chess_ai.cli import main

    runs = tmp_path / "runs"
    directory = model_run(runs, steps=(2,))
    run = RunReader(directory)
    play(suite(games=1), run, 2)
    add_checkpoint(directory, 2, seed=8)  # Saved again: the games are of the file it replaced.
    config = bad_openings_config(tmp_path, runs)

    with pytest.raises(SystemExit) as exited:
        main(["--config", config, "evaluator", "--once"])

    assert exited.value.code == 2
