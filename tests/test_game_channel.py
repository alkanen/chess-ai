import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from game_helpers import replay, scripted_players

from chess_ai.game_session import ActionRejectedError, GameSession
from chess_ai.players import HumanPlayer, RandomPlayer
from chess_ai.web.game_channel import ChannelEvent, GameChannel, GameChannelClosedError

pytestmark = pytest.mark.anyio


async def test_viewers_follow_each_new_game_after_the_last_one_ended():
    channel = GameChannel()
    first = GameSession(*scripted_players("f2f3 e7e5 g2g4 d8h4"))
    second = GameSession(RandomPlayer(1), RandomPlayer(2), move_delay=10)

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        assert (await anext(events)).type == "no_game"
        channel.start(first)
        first_events = [await anext(events) for _ in range(5)]
        channel.start(second)
        second_state = await anext(events)
        await channel.close()

    assert replay(first_events) == first.state
    assert first.state.position.game_over is not None
    assert second_state.type == "state"
    assert second_state.game == second.state


async def test_a_game_replaced_before_it_began_does_not_stall_viewers():
    channel = GameChannel()
    first = GameSession(RandomPlayer(1), RandomPlayer(2))
    second = GameSession(RandomPlayer(3), RandomPlayer(4), move_delay=10)

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        channel.start(first)
        # Subscribes to the first game before its task has had a chance to run.
        assert (await anext(events)).type == "state"
        channel.start(second)
        event = await anext(events)
        await channel.close()

    assert first.state.moves == []
    assert event.type == "state"
    assert event.game == second.state


async def test_close_stops_the_current_game():
    channel = GameChannel()
    session = GameSession(RandomPlayer(1), RandomPlayer(2), move_delay=0.01)
    channel.start(session)
    await asyncio.sleep(0.05)

    await channel.close()
    moves = len(session.state.moves)
    await asyncio.sleep(0.05)

    assert 0 < moves == len(session.state.moves)
    assert session.state.position.game_over is None


async def close_while_viewer_waits(channel: GameChannel, events: AsyncIterator[ChannelEvent]):
    """Close ``channel`` while a viewer awaits its next event; return what the viewer gets."""
    rest = asyncio.create_task(anext(events, None))
    await asyncio.sleep(0)  # The viewer is now waiting.
    await channel.close()
    return await rest


async def test_close_ends_the_events_of_a_viewer_waiting_for_the_first_game():
    channel = GameChannel()

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        assert (await anext(events)).type == "no_game"

        assert await close_while_viewer_waits(channel, events) is None


async def test_close_ends_the_events_of_a_viewer_waiting_after_a_finished_game():
    channel = GameChannel()
    channel.start(GameSession(*scripted_players("f2f3 e7e5 g2g4 d8h4")))

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        finished_game = [await anext(events) for _ in range(5)]
        assert replay(finished_game).position.game_over is not None

        assert await close_while_viewer_waits(channel, events) is None


async def test_close_ends_the_events_of_a_viewer_of_a_running_game():
    channel = GameChannel()
    channel.start(GameSession(RandomPlayer(1), RandomPlayer(2), move_delay=10))

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        assert (await anext(events)).type == "state"

        assert await close_while_viewer_waits(channel, events) is None


async def test_a_closed_channel_starts_no_games():
    channel = GameChannel()
    await channel.close()

    with pytest.raises(GameChannelClosedError):
        channel.start(GameSession(RandomPlayer(1), RandomPlayer(2)))


async def test_resigning_stops_the_game_that_was_waiting_for_a_move():
    channel = GameChannel()
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)
    channel.start(session)

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        assert (await anext(events)).type == "state"
        channel.resign(session.id, "white")
        ended = await anext(events)

    assert ended.type == "game_over"
    assert ended.position.game_over is not None
    assert (ended.position.game_over.result, ended.position.game_over.reason) == (
        "0-1",
        "resignation",
    )


async def test_aborting_stops_the_game_that_was_waiting_for_a_move():
    channel = GameChannel()
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)
    channel.start(session)

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        assert (await anext(events)).type == "state"
        channel.abort(session.id)
        ended = await anext(events)

    assert ended.type == "game_over"
    assert ended.position.game_over is not None
    assert ended.position.game_over.reason == "abort"


async def test_a_viewer_who_arrives_after_a_resignation_is_told_how_it_ended():
    channel = GameChannel()
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)
    channel.start(session)
    channel.resign(session.id, "white")

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        arrived = await anext(events)

    assert arrived.type == "state"
    assert arrived.game.position.game_over is not None
    assert arrived.game.position.game_over.reason == "resignation"


@pytest.mark.parametrize("ending", ["resign", "abort"])
async def test_nothing_is_resigned_or_aborted_before_any_game(ending):
    channel = GameChannel()

    with pytest.raises(ActionRejectedError, match="no game is in progress"):
        channel.resign("any game", "white") if ending == "resign" else channel.abort("any game")


async def test_a_new_game_starts_after_the_last_one_was_resigned():
    channel = GameChannel()
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)
    channel.start(session)

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        assert (await anext(events)).type == "state"
        channel.resign(session.id, "white")
        assert (await anext(events)).type == "game_over"
        channel.start(GameSession(*scripted_players("f2f3 e7e5 g2g4 d8h4")))
        second = [await anext(events) for _ in range(5)]
        await channel.close()

    assert replay(second).position.game_over is not None
    assert replay(second).position.game_over.reason == "checkmate"


def act_on(channel: GameChannel, game: str, action: str) -> None:
    """Do what a viewer looking at ``game`` asked, whatever the current game is."""
    if action == "move":
        channel.submit_move(game, "e2e4")
    elif action == "resign":
        channel.resign(game, "white")
    else:
        channel.abort(game)


@pytest.mark.parametrize("action", ["move", "resign", "abort"])
async def test_nothing_reaches_the_game_that_replaced_the_one_a_viewer_named(action):
    """A viewer acts on the game their browser is showing, not on whatever is current.

    Someone else can start a new game in the moment between a viewer's last look at the
    board and their click, and the click must not land on a game they never saw.
    """
    channel = GameChannel()
    watched = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)
    replacement = GameSession(HumanPlayer(), RandomPlayer(2), move_delay=10)
    channel.start(watched)
    await asyncio.sleep(0)  # The game is now waiting for its first move.
    channel.start(replacement)
    await asyncio.sleep(0)

    with pytest.raises(ActionRejectedError, match="that game has been replaced"):
        act_on(channel, watched.id, action)

    await channel.close()
    assert replacement.state.moves == []
    assert replacement.state.position.game_over is None


@pytest.mark.parametrize("action", ["move", "resign", "abort"])
async def test_the_game_a_viewer_names_is_the_one_they_are_looking_at(action):
    channel = GameChannel()
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)

    async with asyncio.timeout(5), aclosing(channel.events()) as events:
        channel.start(session)
        state = await anext(events)
        assert state.type == "state"
        await asyncio.sleep(0)

        act_on(channel, state.game.id, action)
        happened = await anext(events)
        await channel.close()

    assert happened.type == ("move" if action == "move" else "game_over")
