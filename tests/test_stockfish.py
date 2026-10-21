"""The Stockfish adapter: starting the engine, holding it to a strength, and letting it go.

Most of these run against a stand-in engine (``fixtures/fake_uci.py``), since what is under test
is what the adapter asks of an engine and what it makes of the answers. The ones marked
``needs_stockfish`` play the real engine, and are skipped where it is not installed.
"""

import asyncio
import time

import chess
import chess.engine
import pytest
from stockfish_helpers import (
    STOCKFISH,
    fake_engine,
    needs_stockfish,
    running,
    started,
    told,
    wait_until_gone,
)

from chess_ai import stockfish as stockfish_module
from chess_ai.game_session import GameSession
from chess_ai.players import GameContext, RandomPlayer
from chess_ai.stockfish import (
    StockfishError,
    StockfishPlayer,
    StockfishTimeout,
    describe_stockfish,
    start_stockfish,
)

pytestmark = pytest.mark.anyio


async def stop(player: StockfishPlayer) -> None:
    """Close ``player`` and wait for its engine to have gone, as a game's end does."""
    player.close()
    await player.wait_closed()


@pytest.fixture
def engine(tmp_path):
    return fake_engine(tmp_path / "engine")


async def test_stockfish_plays_at_the_elo_asked_for_and_thinks_for_the_time_given(engine):
    player = await start_stockfish(str(engine), elo=1700, move_time=0.05)
    try:
        board = chess.Board()
        board.push_uci("e2e4")

        move = (await player.choose_move(GameContext(board))).move

        assert board.is_legal(move)
        assert player.name == "Stockfish 1700"
        assert player.stockfish.elo == player.stockfish.requested_elo == 1700
        assert not player.stockfish.clamped
        assert player.stockfish.move_time == 0.05
        lines = told(engine)
        assert "setoption name UCI_LimitStrength value true" in lines
        assert "setoption name UCI_Elo value 1700" in lines
        # The whole game rather than only where it stands, so repetitions count.
        assert "position startpos moves e2e4" in lines
        assert "go movetime 50" in lines
    finally:
        await stop(player)


async def test_stockfish_held_to_a_number_of_nodes_searches_that_many_whatever_the_time(engine):
    player = await start_stockfish(str(engine), elo=1700, move_time=0.05, nodes=20_000)
    try:
        move = (await player.choose_move(GameContext(chess.Board()))).move

        assert chess.Board().is_legal(move)
        assert (player.stockfish.nodes, player.stockfish.move_time) == (20_000, None)
        assert player.stockfish.pace == "20,000 nodes a move"
        lines = told(engine)
        assert "go nodes 20000" in lines
        assert not [line for line in lines if "movetime" in line]
    finally:
        await stop(player)


@pytest.mark.parametrize(
    ("move_time", "nodes"), [(None, None), (0.1, 1000)], ids=["neither", "both"]
)
def test_stockfish_is_described_as_held_to_a_time_or_to_nodes_and_not_both(move_time, nodes):
    from pydantic import ValidationError

    from chess_ai.players import StockfishDescription

    with pytest.raises(ValidationError, match="a move time or to a number of nodes"):
        StockfishDescription(
            elo=1500,
            requested_elo=1500,
            min_elo=1320,
            max_elo=3190,
            move_time=move_time,
            nodes=nodes,
        )


@pytest.mark.parametrize(("asked", "played"), [(800, 1320), (3500, 3190)])
async def test_an_elo_out_of_stockfishs_range_is_played_at_the_nearer_end_and_says_so(
    engine, asked, played
):
    player = await start_stockfish(str(engine), elo=asked)
    await stop(player)

    assert player.stockfish.elo == played
    assert player.stockfish.requested_elo == asked
    assert (player.stockfish.min_elo, player.stockfish.max_elo) == (1320, 3190)
    assert player.stockfish.clamped
    assert player.name == f"Stockfish {played}"


async def test_the_range_is_the_engines_own_rather_than_one_written_down_here(tmp_path):
    engine = fake_engine(tmp_path / "engine", "--elo-range", "1000", "2000")

    player = await start_stockfish(str(engine), elo=800)
    await stop(player)

    assert (player.stockfish.elo, player.stockfish.min_elo) == (1000, 1000)


async def test_a_missing_binary_says_where_it_looked_and_how_to_point_at_it(tmp_path):
    missing = tmp_path / "no-such-stockfish"

    with pytest.raises(
        StockfishError, match=r"not found at .*no-such-stockfish.*\[stockfish\] path"
    ):
        await start_stockfish(str(missing), elo=1500)


async def test_a_file_that_cannot_be_run_says_so(tmp_path):
    unrunnable = tmp_path / "stockfish"
    unrunnable.write_text("")

    with pytest.raises(StockfishError, match="cannot be run"):
        await start_stockfish(str(unrunnable), elo=1500)


async def test_an_engine_with_no_strength_limit_is_refused_and_not_left_running(tmp_path):
    engine = fake_engine(tmp_path / "engine", "--no-elo")

    with pytest.raises(StockfishError, match="no UCI_Elo option"):
        await start_stockfish(str(engine), elo=1500)

    wait_until_gone(*started(engine))


async def test_a_program_that_does_not_speak_uci_is_given_up_on_and_not_left_running(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    engine = fake_engine(tmp_path / "engine", "--mute")

    with pytest.raises(StockfishError) as silent:
        await start_stockfish(str(engine), elo=1500)

    # Never having answered, it is taken not to be a UCI engine at all, rather than one that is
    # slow to start, and is not tried again.
    assert str(silent.value) == (
        f"{str(engine)!r} did not answer as a UCI engine within 0.5 seconds; is it Stockfish?"
    )
    assert silent.value.__cause__ is None
    wait_until_gone(*started(engine))


async def test_a_start_that_times_out_is_one_that_can_pass_only_if_it_has_answered_before(
    tmp_path, monkeypatch
):
    from chess_ai.match import passing

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    silenced = tmp_path / "silenced"
    engine = fake_engine(tmp_path / "engine", "--mute-if", str(silenced))
    other = fake_engine(tmp_path / "other", "--mute")

    with pytest.raises(StockfishError) as never:
        await start_stockfish(str(other), elo=1500)
    await stop(await start_stockfish(str(engine), elo=1500))
    silenced.touch()
    with pytest.raises(StockfishError) as stalled:
        await start_stockfish(str(engine), elo=1500)

    # A command that has never answered is a wrong path, and would not answer next time either.
    assert not passing(never.value)
    assert str(never.value).endswith("is it Stockfish?")
    # One that has is a machine too busy for it to start, which passes.
    assert passing(stalled.value)
    assert "has answered before" in str(stalled.value)
    wait_until_gone(*started(engine), *started(other))


async def test_an_engine_that_does_not_start_in_time_has_exited_and_been_reaped_by_then(
    tmp_path, monkeypatch
):
    from pathlib import Path

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    engine = fake_engine(tmp_path / "engine", "--mute")

    with pytest.raises(StockfishError):
        await start_stockfish(str(engine), elo=1500)

    [pid] = started(engine)
    assert not Path(f"/proc/{pid}").exists(), "not left for nobody to reap"


async def test_a_command_that_dies_before_it_has_ever_answered_cannot_pass(tmp_path):
    from chess_ai.match import passing

    crashing = tmp_path / "crashing"
    crashing.write_text("#!/bin/sh\necho 'Illegal instruction' >&2\nexit 132\n")
    crashing.chmod(0o755)

    with pytest.raises(StockfishError) as died:
        await start_stockfish(str(crashing), elo=1500)

    assert "could not be started as a UCI engine" in str(died.value)
    assert not passing(died.value), "a wrong [stockfish] path, as a silent one is"


async def test_a_start_this_process_was_stopped_through_most_of_can_pass(tmp_path, monkeypatch):
    from chess_ai.match import passing

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    monkeypatch.setattr(stockfish_module, "STALL_TICK", 0.05)
    # Never answers, and has never answered: only the stall tells it from a wrong path.
    engine = fake_engine(tmp_path / "engine", "--mute")

    starting = asyncio.create_task(start_stockfish(str(engine), elo=1500))
    await asyncio.sleep(0.05)
    time.sleep(1.5)  # As the whole machine stopped, waiting for memory.
    with pytest.raises(StockfishError) as stalled:
        await starting

    assert passing(stalled.value)
    assert "with this process stopped" in str(stalled.value)
    wait_until_gone(*started(engine))


async def test_a_stop_that_comes_while_a_late_move_is_given_up_on_is_not_lost():
    from chess_ai.players import StockfishDescription

    player = object.__new__(StockfishPlayer)
    player._timeout = 0.05
    player._description = StockfishDescription(
        elo=1500, requested_elo=1500, min_elo=1350, max_elo=2850, move_time=0.01
    )
    letting_go = asyncio.Event()

    async def slow_to_let_go():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # As python-chess telling the engine to stop: a moment before it is gone.
            letting_go.set()
            await asyncio.shield(asyncio.sleep(0.2))
            raise

    asking = asyncio.create_task(player._answer(slow_to_let_go()))
    await letting_go.wait()
    asking.cancel()  # The evaluator told to stop, as the move was being given up on.

    with pytest.raises(asyncio.CancelledError):
        await asking


async def test_a_first_start_that_says_what_it_is_and_then_crashes_cannot_pass(tmp_path):
    from chess_ai.match import passing

    # As a Stockfish built for another CPU: its banner comes first, then the instruction it
    # cannot run.
    crashing = tmp_path / "crashing"
    crashing.write_text(
        "#!/bin/sh\necho 'Stockfish 17 by the Stockfish developers'\nkill -ILL $$\n"
    )
    crashing.chmod(0o755)

    with pytest.raises(StockfishError) as died:
        await start_stockfish(str(crashing), elo=1500)

    assert not passing(died.value), "what it died of is its own"


async def test_a_first_start_that_greets_as_another_kind_of_engine_cannot_pass(
    tmp_path, monkeypatch
):
    from chess_ai.match import passing

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    # As an xboard engine, which says hello and then waits to be spoken to in its own way.
    xboard = tmp_path / "xboard"
    xboard.write_text("#!/bin/sh\necho 'GNU Chess 6.2.9'\nexec sleep 30\n")
    xboard.chmod(0o755)

    with pytest.raises(StockfishError) as silent:
        await start_stockfish(str(xboard), elo=1500)

    assert not passing(silent.value)
    assert str(silent.value).endswith("is it Stockfish? It said 'GNU Chess 6.2.9'")


async def test_a_first_start_of_a_stockfish_by_another_name_that_is_slow_can_pass(
    tmp_path, monkeypatch
):
    from chess_ai.match import passing

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    fairy = tmp_path / "fairy"
    fairy.write_text("#!/bin/sh\necho 'Fairy-Stockfish 14 LB by Fabian Fichter'\nexec sleep 30\n")
    fairy.chmod(0o755)

    with pytest.raises(StockfishError) as slow:
        await start_stockfish(str(fairy), elo=1500)

    assert passing(slow.value)


async def test_what_a_program_that_did_not_start_said_first_is_cut_short_in_the_message(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    chatty = tmp_path / "chatty"
    chatty.write_text(f"#!/bin/sh\necho '{'x' * 300}'\nexec sleep 30\n")
    chatty.chmod(0o755)

    with pytest.raises(StockfishError) as silent:
        await start_stockfish(str(chatty), elo=1500)

    said = str(silent.value).split("It said ", 1)[1]
    assert said == repr("x" * 80 + "…")


@pytest.mark.parametrize(
    "said",
    [
        "Waiting for Stockfish lock...",
        "Stockfish not found, downloading...",
        "Stockfish 17: waiting for lock...",
    ],
)
async def test_a_first_start_of_a_script_that_only_mentions_stockfish_cannot_pass(
    tmp_path, monkeypatch, said
):
    from chess_ai.match import passing

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    wrapper = tmp_path / "wrapper"
    wrapper.write_text(f"#!/bin/sh\necho '{said}'\nexec sleep 30\n")
    wrapper.chmod(0o755)

    with pytest.raises(StockfishError) as silent:
        await start_stockfish(str(wrapper), elo=1500)

    assert not passing(silent.value), "not Stockfish saying what it is"


async def test_a_first_start_the_kernel_kills_can_pass(tmp_path):
    from chess_ai.match import passing

    # As the OOM killer does to an engine starting on a machine short of memory.
    killed = tmp_path / "killed"
    killed.write_text("#!/bin/sh\nkill -KILL $$\n")
    killed.chmod(0o755)

    with pytest.raises(StockfishError) as died:
        await start_stockfish(str(killed), elo=1500)

    assert passing(died.value), "a broken binary dies otherwise, as with SIGILL"


async def test_a_first_start_that_said_anything_before_it_timed_out_can_pass(tmp_path, monkeypatch):
    from chess_ai.match import passing

    monkeypatch.setattr(stockfish_module, "START_TIMEOUT", 0.5)
    # Started, said what it is, and then too slow to answer: an engine on a loaded machine.
    engine = fake_engine(tmp_path / "engine", "--banner", "--mute")

    with pytest.raises(StockfishError) as slow:
        await start_stockfish(str(engine), elo=1500)

    assert passing(slow.value)
    assert "though it said what it is" in str(slow.value)
    wait_until_gone(*started(engine))


async def test_an_engine_that_fails_with_an_os_error_after_starting_is_stopped_and_reaped(
    tmp_path, monkeypatch
):
    from pathlib import Path

    engine = fake_engine(tmp_path / "engine")

    async def broken_pipe(self):
        await asyncio.sleep(0.3)  # Started, as far as its wrapper saying which process it is.
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(chess.engine.UciProtocol, "initialize", broken_pipe)

    from chess_ai.match import passing

    with pytest.raises(StockfishError) as broken:
        await start_stockfish(str(engine), elo=1500)

    [pid] = started(engine)
    assert not Path(f"/proc/{pid}").exists(), "stopped, and not left for nobody to reap"
    # Killed by this code once it failed, and not taken for the kernel's doing.
    assert not passing(broken.value)
    assert "killed" not in str(broken.value)


async def test_giving_up_on_a_wait_is_not_held_up_by_what_does_not_let_go(monkeypatch):
    monkeypatch.setattr(stockfish_module, "CLOSE_TIMEOUT", 0.2)

    async def does_not_let_go():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # As asyncio waiting for a killed child on a stalled machine.
            await asyncio.shield(asyncio.sleep(5))
            raise

    began = time.monotonic()
    with pytest.raises(TimeoutError):
        await stockfish_module._within(does_not_let_go(), 0.05)

    assert time.monotonic() - began < 2


async def test_a_start_the_machine_has_no_memory_for_is_one_that_can_pass(monkeypatch):
    import errno

    from chess_ai.match import passing

    async def refused(*args, **kwargs):
        raise OSError(errno.ENOMEM, "Cannot allocate memory")

    monkeypatch.setattr(chess.engine.UciProtocol, "popen", refused)

    with pytest.raises(StockfishError) as short:
        await start_stockfish("stockfish", elo=1500)

    assert passing(short.value)


async def test_closing_stops_the_engine_even_in_the_middle_of_a_move(engine):
    player = await start_stockfish(str(engine), elo=1500, move_time=30)
    [pid] = started(engine)
    thinking = GameContext(chess.Board())
    move = asyncio.create_task(player.choose_move(thinking))
    await asyncio.sleep(0.2)
    assert running(pid)

    player.close()
    player.close()  # A second time does nothing.

    await player.wait_closed()
    assert not running(pid)
    with pytest.raises(chess.engine.EngineTerminatedError):
        await move


async def test_a_move_that_takes_too_long_says_which_level_it_was_and_how_long_it_had(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stockfish_module, "MOVE_GRACE", 0.2)
    engine = fake_engine(tmp_path / "engine", "--hang-after", "0")
    player = await start_stockfish(str(engine), elo=1700, nodes=1000)
    try:
        with pytest.raises(StockfishTimeout) as timed_out:
            await player.choose_move(GameContext(chess.Board()))
    finally:
        await stop(player)

    assert isinstance(timed_out.value, TimeoutError)
    message = str(timed_out.value)
    assert message.startswith("Stockfish 1700 gave no move in 0.")
    assert message.endswith(
        "seconds, with 0.25 allowed for 1,000 nodes, and was taken to have hung"
    )
    assert 0.25 <= timed_out.value.waited < 1.0


async def stalled(player: StockfishPlayer, seconds: float) -> "asyncio.Task":
    """Ask ``player`` for a move, and then stop the event loop for ``seconds`` once it has been
    asked, as a machine stopped waiting for memory stops the whole process."""
    move = asyncio.create_task(player.choose_move(GameContext(chess.Board())))
    await asyncio.sleep(0.05)
    time.sleep(seconds)
    return move


async def test_the_time_this_process_was_stopped_is_not_counted_against_the_engine(
    engine, monkeypatch
):
    monkeypatch.setattr(stockfish_module, "MOVE_GRACE", 0.2)
    monkeypatch.setattr(stockfish_module, "STALL_TICK", 0.05)
    player = await start_stockfish(str(engine), elo=1700, move_time=0.05)
    try:
        # The engine answers within its 0.25 seconds, while this process is stopped for a second.
        move = await (await stalled(player, 1.0))
    finally:
        await stop(player)

    assert chess.Board().is_legal(move.move)


async def test_an_engine_that_hangs_through_a_stall_says_how_long_the_process_was_stopped(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stockfish_module, "MOVE_GRACE", 0.2)
    monkeypatch.setattr(stockfish_module, "STALL_TICK", 0.05)
    engine = fake_engine(tmp_path / "engine", "--hang-after", "0")
    player = await start_stockfish(str(engine), elo=1700, move_time=0.05)
    try:
        with pytest.raises(StockfishTimeout) as timed_out:
            await (await stalled(player, 1.0))
    finally:
        await stop(player)

    assert timed_out.value.waited >= 1.0
    assert timed_out.value.stalled >= 0.8
    assert "of them with this process stopped" in str(timed_out.value)
    assert "for a move time of 0.05 seconds" in str(timed_out.value)


async def test_the_engine_says_what_it_is_and_which_strengths_it_plays(tmp_path):
    engine = fake_engine(tmp_path / "engine", "--elo-range", "1350", "2850")

    info = await describe_stockfish(str(engine))

    assert (info.name, info.min_elo, info.max_elo) == ("Fake UCI", 1350, 2850)
    assert not any(running(pid) for pid in started(engine))


async def test_an_engine_that_cannot_be_described_says_why(tmp_path):
    with pytest.raises(StockfishError, match="no UCI_Elo option"):
        await describe_stockfish(str(fake_engine(tmp_path / "engine", "--no-elo")))


@needs_stockfish
async def test_stockfish_plays_a_whole_legal_game_against_the_random_mover():
    assert STOCKFISH is not None
    stockfish = await start_stockfish(STOCKFISH, elo=1500, move_time=0.01)
    try:
        session = GameSession(stockfish, RandomPlayer(seed=7))
        # Every move is checked as it is made, so a game that ends at all ended legally.
        await session.play()
    finally:
        await stop(stockfish)

    over = session.state.position.game_over
    assert over is not None
    assert over.result in {"1-0", "1/2-1/2"}  # Stockfish does not lose to random moves.
    assert not running(stockfish.pid)


@needs_stockfish
async def test_the_real_engine_plays_a_game_held_to_a_number_of_nodes():
    assert STOCKFISH is not None
    stockfish = await start_stockfish(STOCKFISH, elo=1500, nodes=2_000)
    try:
        session = GameSession(stockfish, RandomPlayer(seed=7))
        await session.play()
    finally:
        await stop(stockfish)

    over = session.state.position.game_over
    assert over is not None
    assert over.result in {"1-0", "1/2-1/2"}


@needs_stockfish
async def test_the_real_engine_is_held_to_its_own_calibrated_range():
    assert STOCKFISH is not None
    stockfish = await start_stockfish(STOCKFISH, elo=100, move_time=0.01)
    await stop(stockfish)

    assert stockfish.stockfish.clamped
    assert stockfish.stockfish.elo == stockfish.stockfish.min_elo > 100


def test_stockfish_starts_on_the_event_loop_the_server_runs_on(engine):
    """``chess-ai serve`` runs on uvloop, which the tests' own loop is not, and which turns down
    some of the ways of starting a process that asyncio's loop takes."""
    import uvloop

    async def play_once() -> chess.Move:
        player = await start_stockfish(str(engine), elo=1500, move_time=0.01)
        try:
            return (await player.choose_move(GameContext(chess.Board()))).move
        finally:
            await stop(player)

    assert chess.Board().is_legal(uvloop.run(play_once()))
