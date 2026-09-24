"""Players and event-stream helpers for tests of game sessions."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import chess

from chess_ai.game_session import GameEvent, GameSession, GameState
from chess_ai.players import GameContext, PlayerMove, Thoughts


class ScriptedPlayer:
    """Plays the given moves in order, whether they are legal or not."""

    def __init__(self, moves: Sequence[str], thoughts: Thoughts | None = None) -> None:
        self.name = "Scripted"
        self._moves = iter(moves)
        self._thoughts = thoughts

    async def choose_move(self, context: GameContext) -> PlayerMove:
        return PlayerMove(chess.Move.from_uci(next(self._moves)), self._thoughts)


class PlayerBroke(Exception):
    """A player failed instead of choosing a move, as a lost engine would."""


class BrokenPlayer:
    """Fails when the test says so, in the middle of being asked for a move.

    Stands in for the players that are coming and can fail for real: an engine whose
    process has gone, a model that ran out of memory.
    """

    name = "Broken"

    def __init__(self) -> None:
        self._asked: asyncio.Future[None] | None = None

    async def choose_move(self, context: GameContext) -> PlayerMove:
        self._asked = asyncio.get_running_loop().create_future()
        await self._asked
        raise PlayerBroke("the player is gone")

    def fail(self) -> None:
        """Let the question this player is sitting on fail, on the next turn of the loop."""
        assert self._asked is not None, "the player has not been asked for a move"
        self._asked.set_result(None)


def scripted_players(moves: str) -> tuple[ScriptedPlayer, ScriptedPlayer]:
    """White and Black players for a game given as space-separated UCI moves."""
    ucis = moves.split()
    return ScriptedPlayer(ucis[0::2]), ScriptedPlayer(ucis[1::2])


async def collect(events: AsyncIterator[GameEvent]) -> list[GameEvent]:
    return [event async for event in events]


@asynccontextmanager
async def playing(session: GameSession) -> AsyncIterator[AsyncIterator[GameEvent]]:
    """Play ``session`` in the background, yielding its events, and stop it at the end.

    The game has already asked its first player for a move when the events are yielded,
    so a move can be submitted to a human player straight away.
    """
    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        await asyncio.sleep(0)
        try:
            yield events
        finally:
            game.cancel()
            await asyncio.gather(game, return_exceptions=True)


async def settled() -> None:
    """Let the game's task get as far as asking the player on move for a move.

    A takeback stops the question the game was waiting on, and it takes a few turns of
    the event loop for that to reach the player asked and for the game to ask the next
    one. Yielding more turns than that costs nothing: the game only waits.
    """
    for _ in range(5):
        await asyncio.sleep(0)


def replay(events: Sequence[GameEvent]) -> GameState:
    """The game state a subscriber ends up with, checking that the stream is consistent."""
    first, *rest = events
    assert first.type == "state"
    game = first.game.model_copy(deep=True)
    for index, event in enumerate(rest):
        if event.type == "game_over":
            # Nothing happens in a game after it has been resigned or aborted.
            assert index == len(rest) - 1
            game.position = event.position
            continue
        if event.type == "takeback":
            assert 0 <= event.ply < len(game.moves)
            del game.moves[event.ply :]
            game.position = event.position
            continue
        assert event.type == "move"
        assert event.ply == len(game.moves) + 1
        game.moves.append(event.move)
        game.position = event.position
    return game
