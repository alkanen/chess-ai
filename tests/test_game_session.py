import asyncio
import json
from itertools import pairwise
from pathlib import Path

import chess
import pytest
from game_helpers import ScriptedPlayer, collect, replay, scripted_players

from chess_ai.game_session import GameSession, IllegalMoveError
from chess_ai.players import CandidateMove, RandomPlayer, Thoughts, WinDrawLoss
from chess_ai.position_view import snapshot

pytestmark = pytest.mark.anyio

FOOLS_MATE = "f2f3 e7e5 g2g4 d8h4"
FRONTEND_FIXTURES = Path(__file__).parents[1] / "frontend" / "src" / "test" / "fixtures"


async def test_scripted_game_plays_to_checkmate():
    session = GameSession(*scripted_players(FOOLS_MATE))

    await session.play()

    state = session.state
    assert [move.san for move in state.moves] == ["f3", "e5", "g4", "Qh4#"]
    assert state.position.last_move is not None
    assert (state.position.last_move.from_square, state.position.last_move.to_square) == (
        "d8",
        "h4",
    )
    assert state.position.game_over is not None
    assert (state.position.game_over.result, state.position.game_over.reason) == (
        "0-1",
        "checkmate",
    )


async def test_frontend_fixture_matches_a_subscribers_events():
    """The Vitest fixture must stay what the server actually sends."""
    session = GameSession(*scripted_players(FOOLS_MATE))

    with session.subscribe() as events:
        await session.play()
        received = [event.model_dump(mode="json") for event in await collect(events)]

    fixture = (FRONTEND_FIXTURES / "fools-mate-events.json").read_text(encoding="utf-8")
    assert received == json.loads(fixture)


async def test_state_names_the_players():
    session = GameSession(RandomPlayer(), ScriptedPlayer([]))

    assert (session.state.white, session.state.black) == ("Random mover", "Scripted")


@pytest.mark.parametrize("seed", range(10))
async def test_random_games_end_legally_at_the_first_game_over(seed):
    session = GameSession(RandomPlayer(seed), RandomPlayer(seed + 1000))

    await session.play()

    state = session.state
    board = chess.Board()
    for record in state.moves:
        assert snapshot(board).game_over is None
        move = chess.Move.from_uci(record.uci)
        assert board.is_legal(move)
        assert board.san(move) == record.san
        board.push(move)
    assert state.position == snapshot(board)
    assert state.position.game_over is not None


async def test_seeded_random_games_are_reproducible():
    games = [GameSession(RandomPlayer(1), RandomPlayer(2)) for _ in range(2)]

    for game in games:
        await game.play()

    assert games[0].state == games[1].state


async def test_subscribers_receive_consistent_event_streams():
    session = GameSession(RandomPlayer(1), RandomPlayer(2))

    with session.subscribe() as early:
        game = asyncio.create_task(session.play())
        early_events = [await anext(early) for _ in range(6)]
        with session.subscribe() as middle:
            early_events += await collect(early)
            middle_events = await collect(middle)
    await game
    with session.subscribe() as late:
        late_events = await collect(late)

    final = session.state
    assert final.position.game_over is not None
    assert early_events[0].type == "state"
    assert early_events[0].game.moves == []
    assert middle_events[0].type == "state"
    assert 0 < len(middle_events[0].game.moves) < len(final.moves)
    assert len(late_events) == 1
    assert replay(early_events) == replay(middle_events) == replay(late_events) == final


async def test_players_thoughts_reach_subscribers():
    thoughts = Thoughts(
        candidates=[
            CandidateMove(uci="f2f3", probability=0.5),
            CandidateMove(uci="e2e4", probability=0.3),
        ],
        wdl=WinDrawLoss(win=0.2, draw=0.3, loss=0.5),
    )
    white = ScriptedPlayer(["f2f3", "g2g4"], thoughts=thoughts)
    black = ScriptedPlayer(["e7e5", "d8h4"])
    session = GameSession(white, black)

    with session.subscribe() as events:
        await session.play()
        received = await collect(events)

    moves = [event.move for event in received if event.type == "move"]
    assert [move.thoughts for move in moves] == [thoughts, None, thoughts, None]
    assert replay(received) == session.state


@pytest.mark.parametrize(
    "illegal",
    [
        "e2e5",  # too far
        "e7e5",  # the opponent's piece
        "e1g1",  # castling through pieces
        "0000",  # a null move
    ],
)
async def test_illegal_move_is_rejected_without_changing_state(illegal):
    white = ScriptedPlayer(["e2e4", illegal])
    black = ScriptedPlayer(["e7e5"])
    session = GameSession(white, black)

    with session.subscribe() as events:
        with pytest.raises(IllegalMoveError):
            await session.play()
        received = await collect(events)

    state = session.state
    assert [move.uci for move in state.moves] == ["e2e4", "e7e5"]
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert state.position == snapshot(board)
    assert replay(received) == state


async def test_moves_are_at_least_the_move_delay_apart():
    delay = 0.05
    session = GameSession(*scripted_players(FOOLS_MATE), move_delay=delay)
    loop = asyncio.get_running_loop()

    times = [loop.time()]
    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        async for event in events:
            if event.type == "move":
                times.append(loop.time())
    await game

    assert len(times) == 5
    gaps = [later - earlier for earlier, later in pairwise(times)]
    # Timer resolution can wake a sleeper a hair early.
    assert min(gaps) >= delay * 0.9, gaps


async def test_close_stops_the_game_and_ends_subscriptions():
    session = GameSession(RandomPlayer(1), RandomPlayer(2))

    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        received = [await anext(events) for _ in range(4)]
        session.close()
        received += await collect(events)
    await game

    state = session.state
    assert state.position.game_over is None
    assert replay(received) == state
    with session.subscribe() as later:
        later_events = await collect(later)
    assert len(later_events) == 1
    assert replay(later_events) == state


async def test_a_session_closed_before_it_starts_makes_no_moves():
    session = GameSession(RandomPlayer(1), RandomPlayer(2))

    with session.subscribe() as events:
        session.close()
        await session.play()
        received = await collect(events)

    assert session.state.moves == []
    assert [event.type for event in received] == ["state"]
