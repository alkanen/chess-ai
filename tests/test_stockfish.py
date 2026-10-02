"""The Stockfish adapter: starting the engine, holding it to a strength, and letting it go.

Most of these run against a stand-in engine (``fixtures/fake_uci.py``), since what is under test
is what the adapter asks of an engine and what it makes of the answers. The ones marked
``needs_stockfish`` play the real engine, and are skipped where it is not installed.
"""

import asyncio

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

    with pytest.raises(StockfishError, match="did not answer as a UCI engine"):
        await start_stockfish(str(engine), elo=1500)

    wait_until_gone(*started(engine))


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
