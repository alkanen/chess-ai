"""The game channel: the game on show in the game view, followed by every viewer.

The server holds one current game. Starting a new game replaces it for everyone, so all
viewers always see the same game, and a viewer who (re)connects gets its full state.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Literal

import chess
from pydantic import BaseModel

from chess_ai.game_session import GameEvent, GameSession
from chess_ai.position_view import PositionSnapshot, snapshot

logger = logging.getLogger(__name__)


class GameChannelClosedError(RuntimeError):
    """The channel has been closed, so it plays no further games."""


class NoGameEvent(BaseModel):
    """No game has been started yet."""

    type: Literal["no_game"] = "no_game"
    position: PositionSnapshot
    """The starting position, to show until a game starts."""


ChannelEvent = GameEvent | NoGameEvent


class GameChannel:
    def __init__(self) -> None:
        self._session: GameSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._new_game = asyncio.Event()
        self._closed = False

    def start(self, session: GameSession) -> None:
        """Start playing ``session`` as the current game, stopping the previous one."""
        if self._closed:
            raise GameChannelClosedError("the game channel is closed")
        self._stop_current()
        self._session = session
        self._task = asyncio.create_task(session.play())
        self._task.add_done_callback(_log_failure)
        self._new_game.set()
        self._new_game = asyncio.Event()

    async def close(self) -> None:
        """Stop the current game and end every viewer's events, as when the server shuts down."""
        self._closed = True
        self._new_game.set()  # Wakes the viewers waiting for a game, so they can finish.
        task = self._task
        self._stop_current()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def events(self) -> AsyncIterator[ChannelEvent]:
        """Follow the current game, and every game that replaces it, until the channel closes."""
        followed: GameSession | None = None
        if self._session is None:
            yield NoGameEvent(position=snapshot(chess.Board()))
        while True:
            while self._session is followed:
                if self._closed:
                    return
                await self._new_game.wait()
            followed = self._session
            assert followed is not None
            with followed.subscribe() as events:
                async for event in events:
                    yield event

    def _stop_current(self) -> None:
        if self._session is not None:
            self._session.close()
        if self._task is not None:
            self._task.cancel()


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("The game stopped with an error", exc_info=error)
