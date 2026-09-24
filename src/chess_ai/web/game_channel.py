"""The game channel: the game on show in the game view, followed by every viewer.

The server holds one current game. Starting a new game replaces it for everyone, so all
viewers always see the same game, and a viewer who (re)connects gets its full state.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Literal

import chess
from pydantic import BaseModel

from chess_ai.game_session import ActionRejectedError, GameEvent, GameSession, GameState
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
    def __init__(self, on_finished: Callable[[GameState], None] | None = None) -> None:
        """The channel the game view follows, keeping every game that reaches a result.

        ``on_finished`` is handed each game that reached a result, once it has stopped
        being played, which is where the server saves it as PGN. A game that reached no
        result of its own is not a game that was played to its end, so an aborted game
        and a game replaced by the next one are both dropped rather than kept. Whatever
        keeping a game raises is logged and the channel plays on.
        """
        self._on_finished = on_finished
        self._session: GameSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._new_game = asyncio.Event()
        self._closed = False

    @property
    def current_game(self) -> GameState | None:
        """The game the viewers are looking at, finished or not, or None before the first."""
        return self._session.state if self._session is not None else None

    def start(self, session: GameSession) -> None:
        """Start playing ``session`` as the current game, stopping the previous one."""
        if self._closed:
            raise GameChannelClosedError("the game channel is closed")
        self._stop_current()
        self._session = session
        self._task = asyncio.create_task(self._play(session))
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

    def take_back(self, game: str) -> None:
        """Undo the last move of ``game``, or the last pair of moves.

        Raises:
            ActionRejectedError: no game is running, ``game`` is not the one that is, it
                has ended, no side is played by hand, or no move has been played yet.
                The game is unchanged either way.
        """
        self._named(game).take_back()

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

    async def _play(self, session: GameSession) -> None:
        """Play ``session`` to wherever it stops, and keep the game if it was finished.

        However the game stops, it comes past here: a game played out to a result returns
        from ``play()``, and a game ended off the board, or replaced, has this cancelled.
        """
        try:
            await session.play()
        finally:
            self._keep(session)

    def _keep(self, session: GameSession) -> None:
        """Hand a game that reached a result to whoever keeps the games."""
        game = session.state
        over = game.position.game_over
        # "*" is a game that reached no result at all, which an abort leaves behind.
        if self._on_finished is None or over is None or over.result == "*":
            return
        try:
            self._on_finished(game)
        except Exception:
            # A game that cannot be kept is no reason to stop: the viewers are still
            # looking at it, and the next game has to be playable regardless.
            logger.exception("Could not keep the finished game")

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
