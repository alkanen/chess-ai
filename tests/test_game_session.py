import asyncio
import json
from itertools import pairwise
from pathlib import Path

import chess
import pytest
from game_helpers import ScriptedPlayer, collect, playing, replay, scripted_players

from chess_ai.game_session import GameSession, IllegalMoveError
from chess_ai.players import (
    CandidateMove,
    HumanPlayer,
    MoveRejectedError,
    RandomPlayer,
    Thoughts,
    WinDrawLoss,
)
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


async def test_state_names_the_players_and_says_who_takes_submitted_moves():
    session = GameSession(RandomPlayer(), HumanPlayer())

    state = session.state
    assert state.white.model_dump() == {"name": "Random mover", "accepts_moves": False}
    assert state.black.model_dump() == {"name": "Human", "accepts_moves": True}


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


async def test_a_human_plays_the_move_that_is_submitted():
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        played = [await anext(events), await anext(events)]

    assert [event.move.san for event in played] == ["e4", "e5"]
    assert [move.uci for move in session.state.moves] == ["e2e4", "e7e5"]


async def test_a_submitted_underpromotion_makes_the_piece_it_asks_for():
    """The promotion piece in the UCI is the one that lands on the board.

    The four promotions of a pawn share a destination square, so the last letter of the
    UCI the browser's promotion picker submits is all that tells them apart.
    """
    # 1. h4 g5 2. hxg5 Na6 3. g6 Nb8 4. gxh7 Na6, and the h-pawn takes on g8.
    session = GameSession(HumanPlayer(), ScriptedPlayer(["g7g5", "b8a6", "a6b8", "b8a6"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        for uci in ["h2h4", "h4g5", "g5g6", "g6h7"]:
            session.submit_move(uci)
            played, replied = await anext(events), await anext(events)
            assert (played.type, replied.type) == ("move", "move")

        session.submit_move("h7g8n")
        promotion = await anext(events)

    assert promotion.type == "move"
    assert promotion.move.uci == "h7g8n" and promotion.move.san == "hxg8=N"
    assert session.state.position.pieces["g8"].model_dump() == {
        "color": "white",
        "type": "knight",
    }


async def test_both_sides_can_be_human():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        session.submit_move("e7e5")
        assert (await anext(events)).type == "move"

    assert [move.san for move in session.state.moves] == ["e4", "e5"]


@pytest.mark.parametrize(
    ("uci", "problem"),
    [
        ("e2e5", "not a legal move"),  # too far
        ("e7e5", "not a legal move"),  # the opponent's piece
        ("e1g1", "not a legal move"),  # castling through pieces
        ("0000", "not a legal move"),  # a null move
        ("hello", "is not a move"),
        ("", "is not a move"),
    ],
)
async def test_an_illegal_submission_is_rejected_and_the_game_waits_on(uci, problem):
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"

        with pytest.raises(MoveRejectedError, match=problem):
            session.submit_move(uci)

        assert session.state.moves == []
        assert session.state.position == snapshot(chess.Board())
        session.submit_move("e2e4")
        played = [await anext(events), await anext(events)]

    assert [event.move.uci for event in played] == ["e2e4", "e7e5"]
    assert [move.uci for move in session.state.moves] == ["e2e4", "e7e5"]


async def test_a_move_cannot_be_submitted_for_a_player_that_plays_its_own():
    session = GameSession(ScriptedPlayer(["e2e4"]), HumanPlayer())

    async with playing(session):
        with pytest.raises(MoveRejectedError, match="Scripted plays this move"):
            session.submit_move("e2e4")


async def test_a_second_submission_out_of_turn_is_rejected():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")

        with pytest.raises(MoveRejectedError, match="it is not your turn"):
            session.submit_move("d2d4")

        assert (await anext(events)).type == "move"

    assert [move.uci for move in session.state.moves] == ["e2e4"]


async def test_no_move_can_be_submitted_once_the_game_is_over():
    session = GameSession(*scripted_players(FOOLS_MATE))
    await session.play()

    with pytest.raises(MoveRejectedError, match="the game is over"):
        session.submit_move("e2e4")


async def test_a_submitted_move_is_not_held_back_by_the_move_delay():
    """The delay paces players that move instantly; a person is already slow enough."""
    delay = 0.5
    session = GameSession(HumanPlayer(), HumanPlayer(), move_delay=delay)
    loop = asyncio.get_running_loop()

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        submitted = loop.time()
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        waited = loop.time() - submitted

    assert waited < delay / 2, waited


async def test_a_rejection_does_not_put_the_position_in_its_message():
    """The message reaches the person who submitted the move, so it carries no FEN."""
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session):
        with pytest.raises(MoveRejectedError) as rejected:
            session.submit_move("e2e5")

    assert str(rejected.value) == "e2e5 is not a legal move here"


async def test_the_move_delay_still_paces_the_opponent_that_moves_instantly():
    delay = 0.2
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]), move_delay=delay)
    loop = asyncio.get_running_loop()

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        submitted = loop.time()
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        human = loop.time() - submitted
        assert (await anext(events)).type == "move"
        opponent = loop.time() - submitted

    assert human < delay / 2, human
    # Timer resolution can wake a sleeper a hair early.
    assert opponent >= delay * 0.9, opponent


async def test_two_submitted_moves_are_not_kept_the_move_delay_apart():
    """The delay paces players that move instantly, not the people using the board."""
    delay = 0.5
    session = GameSession(HumanPlayer(), HumanPlayer(), move_delay=delay)
    loop = asyncio.get_running_loop()

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        started = loop.time()
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        session.submit_move("e7e5")
        assert (await anext(events)).type == "move"
        waited = loop.time() - started

    assert [move.san for move in session.state.moves] == ["e4", "e5"]
    assert waited < delay / 2, waited
