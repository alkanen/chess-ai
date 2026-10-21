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
import errno
import logging
import re
import shlex
import signal
import time
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, Final

import chess
import chess.engine
from pydantic import BaseModel, ConfigDict

from chess_ai.players import (
    CLOSE_TIMEOUT,
    CanPass,
    GameContext,
    PlayerMove,
    StockfishDescription,
)

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

MIN_NODES: Final = 1
MAX_NODES: Final = 100_000_000

SLOWEST_NODES_PER_SECOND: Final = 20_000
"""The slowest a Stockfish held to a number of nodes is taken to search, to know when it is
late: a tenth of what one thread of a busy desktop CPU does."""

START_TIMEOUT: Final = 10.0
"""How long the engine has to say it speaks UCI before it is taken not to."""

MOVE_GRACE: Final = 10.0
"""How long past its move time an engine may take before it is taken to have hung.

A move is due when its time is up, but an engine that is loading its network or being starved
of CPU can be late, and a game that fails over a late move is worse than a slow one."""

STALL_TICK: Final = 1.0
"""How often a move being waited for looks at the clock, to tell a late engine from a stall.

A look that comes more than twice this late was held up not by the engine but by this process
not running at all, as when the whole machine is stopped waiting for memory: that part of the
wait is not counted against the engine's time."""

STALLED_SHARE: Final = 0.5
"""How much of a start that timed out this process has to have been stopped for, for a command
that has never answered to be taken for one the machine was too busy to start rather than one
that is not a UCI engine at all."""

_LIMIT_STRENGTH: Final = "UCI_LimitStrength"
_ELO: Final = "UCI_Elo"

Command = str | list[str]
"""How the engine is started: the configured path, or a command line a test runs instead."""


def move_timeout(move_time: float | None, nodes: int | None) -> float:
    """How long Stockfish has for a move before it is taken to have hung: its move time and
    :data:`MOVE_GRACE`, or as long as ``nodes`` take at :data:`SLOWEST_NODES_PER_SECOND` and
    :data:`MOVE_GRACE` when it is held to a number of nodes."""
    if nodes is not None:
        return nodes / SLOWEST_NODES_PER_SECOND + MOVE_GRACE
    assert move_time is not None
    return move_time + MOVE_GRACE


class StockfishError(Exception):
    """Stockfish could not be started, or does not do what a player needs of it.

    The message says what to do about it, for the person who chose Stockfish as a player.
    """


class StockfishTimeout(TimeoutError):
    """Stockfish gave no move in the time it had for one, and was taken to have hung."""

    def __init__(self, description: StockfishDescription, *, waited: float, stalled: float) -> None:
        allowed = move_timeout(description.move_time, description.nodes)
        pace = (
            f"{description.nodes:,} nodes"
            if description.nodes is not None
            else f"a move time of {description.move_time:g} seconds"
        )
        stall = f" ({stalled:.1f} of them with this process stopped)" if stalled >= 0.1 else ""
        super().__init__(
            f"Stockfish {description.elo} gave no move in {waited:.1f} seconds{stall}, with "
            f"{allowed:g} allowed for {pace}, and was taken to have hung"
        )
        self.waited = waited
        """Seconds from asking for the move to giving up on it."""
        self.stalled = stalled
        """How much of :attr:`waited` this process was not running, which was not counted."""


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
        await _stop(transport, engine)


async def start_stockfish(
    command: Command,
    *,
    elo: int,
    move_time: float = DEFAULT_MOVE_TIME,
    nodes: int | None = None,
) -> "StockfishPlayer":
    """A Stockfish player at ``elo``, or as near to it as the engine's strength limit goes.

    An ``elo`` outside the range the engine's ``UCI_Elo`` option allows is moved to the nearer
    end of it, and the player's :attr:`~StockfishPlayer.stockfish` says so.

    It thinks for ``move_time`` seconds a move, or, if ``nodes`` is given, searches that many
    positions a move however long that takes. The engine runs on one thread, its default, which
    is what makes a number of nodes the same search every time.

    Raises:
        StockfishError: there is no engine at ``command``, it cannot be run, it does not speak
            UCI, or it has no calibrated strength limit to play at an Elo with. No process is
            left running either way.
    """
    transport, engine, shown = await _open(command)
    try:
        low, high = _elo_range(engine, shown)
    except StockfishError:
        await _stop(transport, engine)
        raise
    description = StockfishDescription(
        elo=min(max(elo, low), high),
        requested_elo=elo,
        min_elo=low,
        max_elo=high,
        move_time=move_time if nodes is None else None,
        nodes=nodes,
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


_ANSWERED: set[str] = set()
"""The engine commands that have answered as UCI engines in this process, as they are shown.

The first of the things that tell a start that fails on a machine too busy for it from a
``[stockfish] path`` that is not a UCI engine, or one that cannot run here: only the first has
ever answered. A first start, such as an evaluator's when a run has just saved a checkpoint and
a training machine is likeliest to stall, is told by the rest (see :func:`_not_started`):
whether the engine said anything at all, how long this process was stopped for, and whether
the kernel killed it."""

_SHORT_OF: Final = frozenset({errno.ENOMEM, errno.EAGAIN})
"""The errors of a process the kernel would not start for want of memory or processes, rather
than for anything about the program, such as its not being there (``ENOENT``), not being
allowed to run (``EACCES``) or not being a program (``ENOEXEC``), which come again every time."""


class StockfishBusy(StockfishError, CanPass):
    """Stockfish could not be started, it seems for the machine's sake rather than its own:
    started again later, it may well start."""


async def _open(
    command: Command,
) -> tuple[asyncio.SubprocessTransport, chess.engine.UciProtocol, str]:
    """The engine ``command`` starts, once it has said it speaks UCI, and how to name it.

    Starting it and its answer have :data:`START_TIMEOUT` between them, counting only the time
    this process was running, as a move does (:func:`_within`). One that does not start has
    been stopped, and its process waited for, before this raises: started again and again on a
    stalled machine, it would otherwise leave a process behind every time for nobody to reap.

    Raises:
        StockfishBusy: it did not start, it seems for the machine's sake: the machine was too
            short to start it, or it has answered before, or this process was stopped for most
            of the wait.
        StockfishError: it did not start, and would not next time either, such as a
            ``[stockfish] path`` that is not a UCI engine.
    """
    shown = command if isinstance(command, str) else shlex.join(command)
    started: list[tuple[asyncio.SubprocessTransport, _Heard]] = []

    async def handshake() -> None:
        # A session of its own, so that the ctrl-c that stops the server is not also delivered
        # to the engine, which would die mid-move before the server could close the game it is
        # in. A server killed outright still takes its engines with it: they quit when their
        # input closes, as every UCI engine does. Not python-chess's `setpgrp`, which asks for
        # a `process_group` that the uvloop `chess-ai serve` runs on refuses to start.
        started.append(await _Heard.popen(command, start_new_session=True))
        await started[0][1].initialize()
        started[0][1].starting = False

    try:
        await _within(handshake(), START_TIMEOUT)
    except Exception as failed:
        # Stopped whatever went wrong and wherever: a process started and left would be one
        # left for nobody to reap.
        returncode = None
        first_line = None
        if started:
            transport, engine = started[0]
            # Before it is stopped, which kills one that is still running: that kill is this
            # code's, and says nothing about the machine. One that ended of itself has been
            # reaped by now: python-chess says an engine died only once asyncio has reaped it.
            returncode = transport.get_returncode()
            first_line = engine.first_line
            await _stop(transport, engine)
        error, cause = _not_started(
            shown,
            failed,
            started=bool(started),
            heard=bool(started) and started[0][1].heard,
            first_line=first_line,
            returncode=returncode,
        )
        raise error from cause
    except BaseException:
        # Not waited for: this is a cancellation, which is not the moment to wait.
        if started:
            started[0][0].close()
        raise
    transport, engine = started[0]
    _ANSWERED.add(shown)
    return transport, engine, shown


_BANNER: Final = re.compile(r"\S*Stockfish \S*\d\S*( \S+)? by ")
"""What Stockfish, or an engine built from it, says first: its name, its version, perhaps a word
more, and who made it, such as ``Stockfish 17 by``, ``Stockfish dev-20240426-1a2b3c by`` or
``Fairy-Stockfish 14 LB by``. Not any line that mentions it, which a script that starts it could
say about it before it hangs, such as ``Waiting for Stockfish lock...``, ``Stockfish not found,
downloading...`` or ``Stockfish 17: waiting for lock...``."""


class _Heard(chess.engine.UciProtocol):
    """A UCI engine that remembers whether it has sounded like one, and what it said first.

    Stockfish, and what is built from it such as Fairy-Stockfish, says what it is the moment it
    starts, before it is asked anything, and any UCI engine answers ``uci`` with ``id`` and
    ``option`` lines before ``uciok``: so one that has said any of that and then not finished in
    time is an engine on a machine too busy for it, however long this process was running
    meanwhile. Only that: an xboard engine, such as GNU Chess, also greets whoever starts it, and
    then waits for ``xboard`` for ever. An engine of another name that says nothing until it is
    asked is heard only once its own start is over, and on a machine stalled before then is
    taken for one that is not a UCI engine at all, as it was before engines were heard.
    """

    heard = False
    """Whether it has said a line that a UCI engine says."""
    first_line: str | None = None
    """The first line it said, whatever it was, for a message about one that did not start."""
    starting = True
    """Whether it is still starting: past that, what it says is moves, and nothing is listened
    for here."""

    def line_received(self, line: str) -> None:
        if self.starting:
            if self.first_line is None:
                self.first_line = line
            if _BANNER.match(line) or line.startswith(("id ", "option ")):
                self.heard = True
        super().line_received(line)


def _not_started(
    shown: str,
    failed: Exception,
    *,
    started: bool,
    heard: bool,
    first_line: str | None,
    returncode: int | None,
) -> tuple[StockfishError, BaseException | None]:
    """What ``failed``, the failure of a start of ``shown``, comes to, and what to raise it from:
    :class:`StockfishBusy` raised from it if it seems the machine's fault, and so can pass, or a
    :class:`StockfishError` raised from nothing that can pass if it seems the engine's.

    ``started`` is whether its process was started; ``heard`` whether it sounded like a UCI
    engine before it failed (:class:`_Heard`), and ``first_line`` what it said first;
    ``returncode`` how its process had ended before it was stopped, ``None`` for one that was
    still running. An engine started by a shell script that does not ``exec`` it ends as the
    script does, with 137 for one the kernel killed, and is not taken to have been killed.
    """
    if not started:
        if isinstance(failed, FileNotFoundError):
            return StockfishError(
                f"Stockfish was not found at {shown!r}: install it, or set [stockfish] path in "
                "chess-ai.toml to where it is"
            ), failed
        if isinstance(failed, PermissionError):
            return StockfishError(
                f"Stockfish at {shown!r} cannot be run: {failed.strerror}"
            ), failed
        if isinstance(failed, OSError) and failed.errno in _SHORT_OF:
            return StockfishBusy(
                f"{shown!r} could not be started: {failed.strerror}; the machine is short of "
                "memory or processes"
            ), failed
        if isinstance(failed, _Late):
            # Not even started in its time: a machine that cannot fork one is a busy one.
            return StockfishBusy(
                f"{shown!r} could not be started within {failed.waited:.1f} seconds: the machine "
                "is likely too busy"
            ), failed
        return StockfishError(f"{shown!r} could not be started as a UCI engine: {failed}"), None
    why = "it has answered before" if shown in _ANSWERED else ""
    if isinstance(failed, _Late):
        if not why and heard:
            why = "it said what it is"
        stalled = (
            f", {failed.stalled:.1f} of them with this process stopped"
            if failed.stalled >= 0.1
            else ""
        )
        said = (
            f"{shown!r} did not answer as a UCI engine within {failed.waited:.1f} seconds{stalled}"
        )
        if not why and failed.stalled >= STALLED_SHARE * failed.waited:
            why = "this process was stopped for most of it"
        if why:
            return StockfishBusy(
                f"{said}, though {why}: the machine is likely too busy for it to start"
            ), failed
        it_said = f" It said {_cut_short(first_line)!r}" if first_line else ""
        return StockfishError(f"{said}; is it Stockfish?{it_said}"), None
    said = f"{shown!r} could not be started as a UCI engine: {failed}"
    # Decided by how it died, and not by whether it said what it is: Stockfish built for another
    # CPU says so before it dies of the instruction it cannot run (SIGILL). Killed (SIGKILL)
    # before this code stopped it is what the kernel's out-of-memory killer does.
    if not why and returncode == -signal.SIGKILL:
        why = "it was killed, as for want of memory"
    if why:
        return StockfishBusy(f"{said}; {why}"), failed
    # Dying before it has ever answered, of its own accord, it would die next time too.
    return StockfishError(said), None


def _cut_short(line: str, length: int = 80) -> str:
    """``line``, or as much of it as fits in ``length`` characters and an ellipsis: what a program
    that is not a UCI engine said first can be a whole usage text, or not text at all."""
    return line if len(line) <= length else f"{line[:length]}…"


async def _stop(transport: asyncio.SubprocessTransport, engine: chess.engine.UciProtocol) -> None:
    """Kill an engine :func:`_open` started, and give its process up to :data:`CLOSE_TIMEOUT`
    to have exited and been reaped: on a stalled machine it can be slow to go, and whatever
    stopped it is not to be held up for longer."""
    transport.close()
    with suppress(TimeoutError):
        await asyncio.wait_for(asyncio.shield(engine.returncode), CLOSE_TIMEOUT)


class _Late(TimeoutError):
    """What :func:`_within` waited for did not come in the time it had."""

    def __init__(self, *, waited: float, stalled: float) -> None:
        super().__init__(f"no answer in {waited:.1f} seconds")
        self.waited = waited
        """Seconds from starting to wait to giving up."""
        self.stalled = stalled
        """How much of :attr:`waited` this process was not running, which was not counted."""


async def _within[T](waiting: Coroutine[Any, Any, T], timeout: float) -> T:
    """What ``waiting`` comes to, unless it takes longer than ``timeout`` seconds.

    Only the time this process was running counts: a machine stopped for longer than the whole
    time, waiting for memory, would otherwise have a timer that fires the moment it starts
    again, ahead of the answer that came in meanwhile. ``waiting`` is cancelled once it is
    given up on, and waited for to have finished, without that wait taking for its own a
    cancellation that comes for whoever is waiting here.

    Raises:
        _Late: it took longer, and has been stopped.
        Whatever ``waiting`` raises.
    """
    task = asyncio.ensure_future(waiting)
    started = last = time.monotonic()
    counted = 0.0
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=min(STALL_TICK, timeout - counted))
            if done:
                return task.result()
            now = time.monotonic()
            counted += min(now - last, 2 * STALL_TICK)
            last = now
            if counted >= timeout:
                raise _Late(waited=now - started, stalled=now - started - counted)
    finally:
        if not task.done():
            task.cancel()
            # Not `await task`: that raises the task's own cancellation, which cannot be told
            # from one of this one's and would be swallowed with it. And not for longer than
            # whatever is given up on takes to go: asyncio waiting for a killed child on a
            # stalled machine can take as long as the stall.
            await asyncio.wait({task}, timeout=CLOSE_TIMEOUT)
        # Looked at, now or once it ends, so that asyncio does not report it as never looked at.
        task.add_done_callback(lambda done: done.cancelled() or done.exception())


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
    """Stockfish at a fixed strength, thinking a fixed time or a fixed number of nodes a move.
    Made by :func:`start_stockfish`."""

    def __init__(
        self,
        transport: asyncio.SubprocessTransport,
        engine: chess.engine.UciProtocol,
        description: StockfishDescription,
    ) -> None:
        self._transport = transport
        self._engine = engine
        self._description = description
        if description.nodes is not None:
            self._limit = chess.engine.Limit(nodes=description.nodes)
        else:
            self._limit = chess.engine.Limit(time=description.move_time)
        self._timeout = move_timeout(description.move_time, description.nodes)

    @property
    def name(self) -> str:
        return f"Stockfish {self._description.elo}"

    @property
    def stockfish(self) -> StockfishDescription:
        """How strong it was asked to be and plays, and how long or how far it thinks."""
        return self._description

    @property
    def pid(self) -> int:
        """The engine's process, which is what tests make sure is gone after a game."""
        return self._transport.get_pid()

    async def choose_move(self, context: GameContext) -> PlayerMove:
        # The whole game rather than only the position, so that the engine knows which
        # positions have already been repeated and does not walk into a draw unawares.
        result = await self._answer(self._engine.play(context.board, self._limit))
        if result.move is None:
            raise StockfishError("Stockfish gave no move")
        return PlayerMove(result.move)

    async def _answer(
        self, playing: Coroutine[Any, Any, chess.engine.PlayResult]
    ) -> chess.engine.PlayResult:
        """What ``playing`` comes to, unless the engine takes longer than it has for a move,
        counting only the time this process was running (:func:`_within`).

        Raises:
            StockfishTimeout: the engine took longer, and its move is no longer waited for.
        """
        try:
            return await _within(playing, self._timeout)
        except _Late as late:
            raise StockfishTimeout(
                self._description, waited=late.waited, stalled=late.stalled
            ) from None

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
