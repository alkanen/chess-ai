"""The player interface that every kind of player implements, and the random mover.

A game session asks the player to move whenever it is that player's turn. Human,
model, Stockfish and search-based players all implement the same interface, so game
sessions, matches and the UI never need to know which kind they are dealing with.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import chess
from pydantic import BaseModel


class MoveRejectedError(Exception):
    """A submitted move was not accepted, so the game is left as it was.

    The message is shown to the person who submitted the move, so it says what is wrong
    without naming the position: their browser is already looking at it.
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
        if self._position is None or self._move is None:
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
