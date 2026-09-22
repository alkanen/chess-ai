"""The player interface that every kind of player implements, and the random mover.

A game session asks the player to move whenever it is that player's turn. Human,
model, Stockfish and search-based players all implement the same interface, so game
sessions, matches and the UI never need to know which kind they are dealing with.
"""

import random
from dataclasses import dataclass
from typing import Protocol

import chess
from pydantic import BaseModel


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


class RandomPlayer:
    """Plays a uniformly random legal move."""

    name = "Random mover"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def choose_move(self, context: GameContext) -> PlayerMove:
        return PlayerMove(self._rng.choice(list(context.board.legal_moves)))
