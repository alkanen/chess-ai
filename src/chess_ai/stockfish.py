"""The Stockfish adapter: the engine, run as a process of its own, sat down at the board.

Stockfish is a separately installed binary, found through ``[stockfish] path``, and spoken to
over UCI through python-chess. A player plays at a chosen Elo using the engine's own calibrated
strength limit (``UCI_LimitStrength`` and ``UCI_Elo``), which is what makes it a reference
opponent rather than merely a strong one.

Every player has an engine process of its own, started when the player is made and killed when
it is closed, so that two Stockfish sides of one game can play at two strengths and a game that
ends leaves nothing running behind it.
"""

import asyncio
import logging
import shlex
from typing import Final

import chess
import chess.engine
from pydantic import BaseModel, ConfigDict

from chess_ai.players import GameContext, PlayerMove, StockfishDescription

logger = logging.getLogger(__name__)

DEFAULT_MOVE_TIME: Final = 1.0
"""Seconds a move: quick enough to watch a game at, and not far off what the levels assume.

Stockfish's documentation says its Elo levels were calibrated at two minutes a game plus a
second a move, which is a few seconds a move, and anchored to the CCRL 40/4 engine rating list
rather than to any human pool. So a level is a claim about engine games at about that pace:
much less time a move and Stockfish plays below its level, more and above it. Neither makes a
level a Lichess rating, which is what the ladder estimate will have to keep in mind."""

MIN_MOVE_TIME: Final = 0.01
MAX_MOVE_TIME: Final = 60.0

START_TIMEOUT: Final = 10.0
"""How long the engine has to say it speaks UCI before it is taken not to."""

MOVE_GRACE: Final = 10.0
"""How long past its move time an engine may take before it is taken to have hung.

A move is due when its time is up, but an engine that is loading its network or being starved
of CPU can be late, and a game that fails over a late move is worse than a slow one."""

_LIMIT_STRENGTH: Final = "UCI_LimitStrength"
_ELO: Final = "UCI_Elo"

Command = str | list[str]
"""How the engine is started: the configured path, or a command line a test runs instead."""


class StockfishError(Exception):
    """Stockfish could not be started, or does not do what a player needs of it.

    The message says what to do about it, for the person who chose Stockfish as a player.
    """


class StockfishInfo(BaseModel):
    """Which Stockfish is installed, and the strengths it can be asked to play at."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    """What the engine calls itself, such as "Stockfish 17"."""
    min_elo: int
    """The weakest level it plays, which a lower Elo is raised to."""
    max_elo: int
    """The strongest level it plays, which a higher Elo is lowered to."""


async def describe_stockfish(command: Command) -> StockfishInfo:
    """Which engine ``command`` starts, and the range of strengths it supports.

    The range differs between versions (1350 to 2850 in Stockfish 14, 1320 to 3190 in 19), so
    it is asked of the engine rather than written down here. The engine is started to be asked
    and stopped again at once.

    Raises:
        StockfishError: as :func:`start_stockfish` does.
    """
    transport, engine, shown = await _open(command)
    try:
        low, high = _elo_range(engine, shown)
        return StockfishInfo(name=engine.id.get("name", shown), min_elo=low, max_elo=high)
    finally:
        transport.close()
        await asyncio.shield(engine.returncode)


async def start_stockfish(
    command: Command, *, elo: int, move_time: float = DEFAULT_MOVE_TIME
) -> "StockfishPlayer":
    """A Stockfish player at ``elo``, or as near to it as the engine's strength limit goes.

    An ``elo`` outside the range the engine's ``UCI_Elo`` option allows is moved to the nearer
    end of it, and the player's :attr:`~StockfishPlayer.stockfish` says so.

    Raises:
        StockfishError: there is no engine at ``command``, it cannot be run, it does not speak
            UCI, or it has no calibrated strength limit to play at an Elo with. No process is
            left running either way.
    """
    transport, engine, shown = await _open(command)
    try:
        low, high = _elo_range(engine, shown)
    except StockfishError:
        transport.close()
        await asyncio.shield(engine.returncode)
        raise
    description = StockfishDescription(
        elo=min(max(elo, low), high),
        requested_elo=elo,
        min_elo=low,
        max_elo=high,
        move_time=move_time,
    )
    try:
        await engine.configure({_LIMIT_STRENGTH: True, _ELO: description.elo})
    except BaseException:
        # Not waited for: this may be a cancellation, which is not the moment to wait.
        transport.close()
        raise
    if description.clamped:
        logger.info(
            "Stockfish was asked to play at %d Elo and plays at %d, the nearest it supports",
            description.requested_elo,
            description.elo,
        )
    return StockfishPlayer(transport, engine, description)


async def _open(
    command: Command,
) -> tuple[asyncio.SubprocessTransport, chess.engine.UciProtocol, str]:
    """The engine ``command`` starts, once it has said it speaks UCI, and how to name it."""
    shown = command if isinstance(command, str) else shlex.join(command)
    try:
        # A session of its own, so that the ctrl-c that stops the server is not also delivered
        # to the engine, which would die mid-move before the server could close the game it is
        # in. A server killed outright still takes its engines with it: they quit when their
        # input closes, as every UCI engine does. Not python-chess's `setpgrp`, which asks for
        # a `process_group` that the uvloop `chess-ai serve` runs on refuses to start.
        transport, engine = await asyncio.wait_for(
            chess.engine.popen_uci(command, start_new_session=True), START_TIMEOUT
        )
    except FileNotFoundError as missing:
        raise StockfishError(
            f"Stockfish was not found at {shown!r}: install it, or set [stockfish] path in "
            "chess-ai.toml to where it is"
        ) from missing
    except PermissionError as denied:
        raise StockfishError(f"Stockfish at {shown!r} cannot be run: {denied.strerror}") from denied
    except TimeoutError as silent:
        raise StockfishError(
            f"{shown!r} did not answer as a UCI engine within {START_TIMEOUT:g} seconds; "
            "is it Stockfish?"
        ) from silent
    except (chess.engine.EngineError, OSError) as broken:
        raise StockfishError(f"{shown!r} could not be started as a UCI engine: {broken}") from (
            broken
        )
    return transport, engine, shown


def _elo_range(engine: chess.engine.UciProtocol, shown: str) -> tuple[int, int]:
    """The weakest and strongest Elo the engine's strength limit can be set to."""
    option = engine.options.get(_ELO)
    if _LIMIT_STRENGTH not in engine.options or option is None:
        raise StockfishError(
            f"{shown!r} has no {_ELO} option, so it cannot be asked to play at an Elo; "
            "is it Stockfish?"
        )
    return int(option.min), int(option.max)


class StockfishPlayer:
    """Stockfish at a fixed strength and a fixed time a move. Made by :func:`start_stockfish`."""

    def __init__(
        self,
        transport: asyncio.SubprocessTransport,
        engine: chess.engine.UciProtocol,
        description: StockfishDescription,
    ) -> None:
        self._transport = transport
        self._engine = engine
        self._description = description
        self._limit = chess.engine.Limit(time=description.move_time)

    @property
    def name(self) -> str:
        return f"Stockfish {self._description.elo}"

    @property
    def stockfish(self) -> StockfishDescription:
        """How strong it was asked to be and plays, and how long it thinks."""
        return self._description

    @property
    def pid(self) -> int:
        """The engine's process, which is what tests make sure is gone after a game."""
        return self._transport.get_pid()

    async def choose_move(self, context: GameContext) -> PlayerMove:
        # The whole game rather than only the position, so that the engine knows which
        # positions have already been repeated and does not walk into a draw unawares.
        result = await asyncio.wait_for(
            self._engine.play(context.board, self._limit),
            self._description.move_time + MOVE_GRACE,
        )
        if result.move is None:
            raise StockfishError("Stockfish gave no move")
        return PlayerMove(result.move)

    def close(self) -> None:
        """Stop the engine, at once and whatever it is doing.

        Killed rather than asked to quit: it keeps nothing worth saving, and asking means
        waiting for an answer, which a game being cancelled or a server going down might
        not be there to hear. Closing a closed player does nothing.
        """
        self._transport.close()

    async def wait_closed(self) -> None:
        """Return once the engine's process has exited and been reaped.

        Worth waiting for on the way out of a server: a process that exits after the event
        loop has closed is reaped by nobody and complained about by asyncio.
        """
        # Shielded, so that a wait that is given up on leaves the future for anyone else.
        await asyncio.shield(self._engine.returncode)


class DeferredStockfishPlayer:
    """Stockfish that is started only when it is first asked for a move.

    For a game made again after the server restarted: the game may wait days for a person's
    move, and an engine process held all that while would be held for nothing. Until it is
    started it says it is what it was before the restart, ``description``; once started, what
    the engine says.
    """

    def __init__(
        self,
        command: Command,
        *,
        elo: int,
        move_time: float,
        description: StockfishDescription,
    ) -> None:
        self._command = command
        self._elo = elo
        self._move_time = move_time
        self._description = description
        self._starting: asyncio.Task[StockfishPlayer] | None = None
        self._closed = False

    @property
    def name(self) -> str:
        return f"Stockfish {self.stockfish.elo}"

    @property
    def stockfish(self) -> StockfishDescription:
        started = self._started()
        return started.stockfish if started is not None else self._description

    async def choose_move(self, context: GameContext) -> PlayerMove:
        if self._closed:
            raise StockfishError("Stockfish has been stopped")
        if self._starting is None:
            self._starting = asyncio.ensure_future(
                start_stockfish(self._command, elo=self._elo, move_time=self._move_time)
            )
        # Shielded, so that a question dropped by a takeback while the engine is starting does
        # not leave it half started: the next question waits for the same start.
        engine = await asyncio.shield(self._starting)
        return await engine.choose_move(context)

    def close(self) -> None:
        """Stop the engine, or have it stopped as soon as it has started if it is starting."""
        self._closed = True
        starting = self._starting
        if starting is None:
            return
        if starting.done():
            _close_started(starting)
        else:
            starting.add_done_callback(_close_started)

    async def wait_closed(self) -> None:
        starting = self._starting
        if starting is None:
            return
        try:
            engine = await asyncio.shield(starting)
        except Exception:  # noqa: BLE001 - whoever asked for a move was told why already
            return  # It never started, so there is nothing to wait for.
        await engine.wait_closed()

    def _started(self) -> "StockfishPlayer | None":
        starting = self._starting
        if starting is None or not starting.done() or starting.cancelled():
            return None
        return starting.result() if starting.exception() is None else None


def _close_started(starting: "asyncio.Task[StockfishPlayer]") -> None:
    # Read even when it failed, so that asyncio does not report a failure nobody looked at:
    # whoever asked for a move was told of it already.
    if not starting.cancelled() and starting.exception() is None:
        starting.result().close()
