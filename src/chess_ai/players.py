"""The player interface that every kind of player implements, and the random mover.

A game session asks the player to move whenever it is that player's turn. Human,
model, Stockfish and search-based players all implement the same interface, so game
sessions, matches and the UI never need to know which kind they are dealing with.

The two narrower protocols below are what a session does have to tell apart: which sides a
viewer may move for, and which sides are a checkpoint playing. Both are answered by asking the
player, so that adding a kind of player adds nothing here — and so that nothing in this module,
which the web server imports to start any game at all, has to know what a network is.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import chess
from pydantic import BaseModel, ConfigDict


class MoveRejectedError(Exception):
    """A submitted move was not accepted, so the game is left as it was.

    The message is shown to the person who submitted the move, so it says what is wrong
    without naming the position: their browser is already looking at it.
    """


class PlayerUnavailableError(Exception):
    """The player can no longer play, because what it plays with is gone.

    A checkpoint deleted, or replaced by other weights, since the game began with it. Unlike a
    player that broke, nothing is wrong with the game: it waits for another player to take
    this one's place.
    """


@dataclass(frozen=True)
class GameContext:
    """What a player gets to see when it is asked for a move."""

    board: chess.Board
    """The current position with the game's move history. The player's own copy."""


class CandidateMove(BaseModel):
    uci: str
    probability: float


class WinDrawLoss(BaseModel):
    """Probabilities from the point of view of the player who is moving."""

    win: float
    draw: float
    loss: float


class Thoughts(BaseModel):
    """What a player was considering when it chose its move, for the UI to show."""

    candidates: list[CandidateMove] = []
    wdl: WinDrawLoss | None = None


@dataclass(frozen=True)
class PlayerMove:
    move: chess.Move
    thoughts: Thoughts | None = None


class Player(Protocol):
    @property
    def name(self) -> str:
        """Shown to viewers, such as "Random mover"."""
        ...

    async def choose_move(self, context: GameContext) -> PlayerMove:
        """Return a legal move for the side to move in ``context.board``."""
        ...


SelectionStrategy = Literal["argmax", "sample"]
"""How a player turns a distribution over moves into the one move it plays.

Named here rather than with the network, because it is a property of the player: the same
checkpoint plays its best move or samples from its distribution, and a search-based player will
one day choose by another rule again from the same probabilities.
"""

DEFAULT_TEMPERATURE = 1.0
"""Sampling straight from the distribution given, neither flattened nor sharpened."""

MIN_TEMPERATURE = 0.01
"""The lowest temperature worth asking for: below this, sampling is argmax with extra steps.

Here beside the strategy rather than with the sampling, so that whatever takes the setting —
a web request, a config file, a match runner — refuses the same values the sampler would.
"""


class ModelDescription(BaseModel):
    """Which model a player is, and how it was asked to play.

    Everything here is what a game was played with rather than what the network is, so that a
    game watched now and a PGN read in six months both say which weights produced the moves.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    run: str
    """The training run the weights came from."""
    checkpoint: int
    """The step the checkpoint was saved at, which is what names it in the run."""
    rating: int | None = None
    """The rating it was asked to play like, or ``None`` for a position that claims none."""
    strategy: SelectionStrategy = "argmax"
    temperature: float | None = None
    """How flat the distribution is sampled from; ``None`` when it is not sampled at all."""


@runtime_checkable
class ModelBackedPlayer(Protocol):
    """A player whose moves come from a checkpoint, which is what it can say about itself.

    Recognized the way :class:`SubmittedMovePlayer` is, so that a game session can describe a
    model player to viewers and to PGN without knowing what an inference engine is — and without
    the web server importing torch to find out.
    """

    @property
    def model(self) -> ModelDescription:
        """Which checkpoint plays these moves, and how it was asked to choose them."""
        ...


class StockfishDescription(BaseModel):
    """How strong a Stockfish player was asked to be, and how strong it plays.

    The two Elo figures differ when the one asked for was outside the range Stockfish's
    strength limit is calibrated over, and the viewer has to be told so: a game against "800"
    that was really played against Stockfish's floor would teach them the wrong thing.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    elo: int
    """The strength it plays at, within the range Stockfish supports."""
    requested_elo: int
    """The strength that was asked for, which is ``elo`` unless it had to be moved into range."""
    min_elo: int
    """The weakest Stockfish will play, which a request below it was raised to."""
    max_elo: int
    """The strongest calibrated level, which a request above it was lowered to."""
    move_time: float
    """How many seconds it thinks about each move."""

    @property
    def clamped(self) -> bool:
        return self.elo != self.requested_elo


@runtime_checkable
class StockfishBackedPlayer(Protocol):
    """A player whose moves come from Stockfish, recognized as a model player is."""

    @property
    def stockfish(self) -> StockfishDescription: ...


@runtime_checkable
class ClosablePlayer(Protocol):
    """A player holding something that has to be let go of when its game is over.

    An engine process, for instance: whoever made the player closes it once the game it was
    made for has stopped being played, however it stopped. Closing twice is harmless.

    Not a coroutine, deliberately. Closing happens on the way out of a game that may be being
    cancelled, and on the way down of a server that may be cancelling everything; anything
    that had to be awaited could be interrupted half-way and leave a process running. Waiting
    for what was closed to be gone is a step of its own, which can be skipped.
    """

    def close(self) -> None: ...

    async def wait_closed(self) -> None:
        """Return once whatever :meth:`close` let go of is gone, such as an exited process."""
        ...


def close_players(*players: Player) -> None:
    """Let go of whatever each of ``players`` holds, all of them even if one fails to."""
    failures = []
    for player in players:
        if isinstance(player, ClosablePlayer):
            try:
                player.close()
            except Exception as failed:  # noqa: BLE001 - the rest still have to be closed
                failures.append(failed)
    if failures:
        raise ExceptionGroup("players could not be closed", failures)


async def wait_players_closed(*players: Player) -> None:
    """Wait until whatever :func:`close_players` let go of for ``players`` is gone."""
    for player in players:
        if isinstance(player, ClosablePlayer):
            await player.wait_closed()


@runtime_checkable
class SubmittedMovePlayer(Protocol):
    """A player whose moves come from outside the game, rather than being computed.

    Game sessions recognize these players so that viewers can be told which sides they
    may move, and so that a submitted move reaches the player who is waiting for one.
    """

    def submit(self, uci: str) -> None:
        """Play ``uci`` as this player's move.

        Raises:
            MoveRejectedError: it is not this player's turn, or the move is not legal
                in the current position. The game is unchanged either way.
        """
        ...


class RandomPlayer:
    """Plays a uniformly random legal move."""

    name = "Random mover"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def choose_move(self, context: GameContext) -> PlayerMove:
        return PlayerMove(self._rng.choice(list(context.board.legal_moves)))


class HumanPlayer:
    """A person at a browser: the game waits here until a move is submitted.

    Only the move asked for is accepted. A submission that arrives out of turn, or that
    is not legal in the position the player was asked about, is rejected and the game
    goes on waiting, so a mis-click cannot corrupt the game.
    """

    name = "Human"

    def __init__(self) -> None:
        self._position: chess.Board | None = None
        self._move: asyncio.Future[chess.Move] | None = None

    async def choose_move(self, context: GameContext) -> PlayerMove:
        self._position = context.board
        self._move = asyncio.get_running_loop().create_future()
        try:
            return PlayerMove(await self._move)
        finally:
            self._position, self._move = None, None

    def submit(self, uci: str) -> None:
        # A question the game has stopped waiting on, as a takeback makes it do, leaves
        # its move cancelled until the player is asked again. Whoever was on move in the
        # position the game has left is nobody, so their move is nothing to play.
        if self._position is None or self._move is None or self._move.done():
            raise MoveRejectedError("it is not your turn")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as invalid:
            raise MoveRejectedError(f"{uci!r} is not a move") from invalid
        if not self._position.is_legal(move):
            # The person who submitted the move reads this, so it names no position.
            raise MoveRejectedError(f"{uci} is not a legal move here")
        # Cleared before the waiting game is woken, so that a second submission arriving
        # in the meantime is rejected rather than resolving the same move twice.
        awaited, self._position, self._move = self._move, None, None
        awaited.set_result(move)
