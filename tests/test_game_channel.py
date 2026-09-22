import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from game_helpers import replay, scripted_players

from chess_ai.game_session import GameSession
from chess_ai.players import RandomPlayer
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
