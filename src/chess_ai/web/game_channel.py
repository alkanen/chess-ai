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

from chess_ai.game_session import ActionRejectedError, GameEvent, GameSession
from chess_ai.position_view import Color, PositionSnapshot, snapshot

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

    def submit_move(self, game: str, uci: str) -> None:
        """Play ``uci`` in ``game``, for the side to move.

        Raises:
            ActionRejectedError: no game is running, or ``game`` is not the one that is.
            MoveRejectedError: the game has ended, the player to move plays its own
                moves, or the move is not legal. The game is unchanged either way.
        """
        self._named(game).submit_move(uci)

    def resign(self, game: str, color: Color) -> None:
        """End ``game`` as a resignation by ``color``.

        Raises:
            ActionRejectedError: no game is running, ``game`` is not the one that is, it
                has ended, or that side plays its own moves. The game is unchanged.
        """
        self._named(game).resign(color)
        self._stop_current()

    def abort(self, game: str) -> None:
        """End ``game`` with no result.

        Raises:
            ActionRejectedError: no game is running, ``game`` is not the one that is, or
                it has already ended.
        """
        self._named(game).abort()
        self._stop_current()

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

    def _named(self, game: str) -> GameSession:
        """The current game, if it is the one the viewer meant.

        A viewer acts on the game their browser is showing them, which a new game can
        replace in the moment before their click arrives. Acting on the replacement
        instead would end or move a game they have never seen.
        """
        if self._session is None:
            raise ActionRejectedError("no game is in progress")
        if self._session.id != game:
            raise ActionRejectedError("that game has been replaced")
        return self._session

    def _stop_current(self) -> None:
        """Close the current game and let go of the task driving it.

        A game the players did not finish leaves ``play()`` waiting for a move that will
        never come, so closing the session is not enough to stop it.
        """
        if self._session is not None:
            self._session.close()
        if self._task is not None:
            self._task.cancel()


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("The game stopped with an error", exc_info=error)
