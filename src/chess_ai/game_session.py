"""Game sessions: one game held on the server and driven between two players.

A session asks each player for a move in turn, checks every move through python-chess
and tells any number of subscribers what happens. A subscriber first gets the full
current state and then every move after it, so all viewers see the same game however
late they join.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Literal

import chess
from pydantic import BaseModel

from chess_ai.players import GameContext, Player, PlayerMove, Thoughts
from chess_ai.position_view import PositionSnapshot, snapshot


class IllegalMoveError(Exception):
    """A player chose a move that is not legal in the current position."""


class MoveRecord(BaseModel):
    uci: str
    san: str
    thoughts: Thoughts | None = None


class GameState(BaseModel):
    white: str
    """The name of the player with the white pieces."""
    black: str
    """The name of the player with the black pieces."""
    moves: list[MoveRecord]
    """Every move played so far."""
    position: PositionSnapshot


class GameStateEvent(BaseModel):
    """The full state of the game: the first event every subscriber receives."""

    type: Literal["state"] = "state"
    game: GameState


class MoveEvent(BaseModel):
    type: Literal["move"] = "move"
    ply: int
    """The move's number in ``GameState.moves``, counting from 1."""
    move: MoveRecord
    position: PositionSnapshot
    """The position after the move."""


GameEvent = GameStateEvent | MoveEvent


class GameSession:
    def __init__(self, white: Player, black: Player, *, move_delay: float = 0.0) -> None:
        """A game from the starting position.

        ``move_delay`` is the least number of seconds between two moves, so that viewers
        can follow a game between players that move instantly.
        """
        self._players = {chess.WHITE: white, chess.BLACK: black}
        self._move_delay = move_delay
        self._board = chess.Board()
        self._moves: list[MoveRecord] = []
        self._position = snapshot(self._board)
        self._subscribers: set[asyncio.Queue[GameEvent | None]] = set()
        self._started = False
        self._closed = False

    @property
    def state(self) -> GameState:
        return GameState(
            white=self._players[chess.WHITE].name,
            black=self._players[chess.BLACK].name,
            moves=list(self._moves),
            position=self._position,
        )

    async def play(self) -> None:
        """Play the game to its end, asking each player for a move in turn.

        The session is closed when this returns, fails or is cancelled.

        Raises:
            IllegalMoveError: a player chose an illegal move. The game stays in the
                position before that move.
        """
        if self._started:
            raise RuntimeError("the game has already been played")
        self._started = True
        loop = asyncio.get_running_loop()
        try:
            earliest = loop.time() + self._move_delay
            while not self._closed and self._position.game_over is None:
                player = self._players[self._board.turn]
                choice = await player.choose_move(GameContext(self._board.copy()))
                # Also yields to the event loop when there is no delay left, so that a
                # game between instant players doesn't hold up everything else.
                await asyncio.sleep(max(0.0, earliest - loop.time()))
                if self._closed:
                    break
                self._make_move(choice)
                earliest = loop.time() + self._move_delay
        finally:
            self.close()

    def close(self) -> None:
        """Stop the game where it is and end every subscription.

        A running ``play()`` makes no further moves, but it may still be waiting for a
        player; cancel it to stop it at once.
        """
        if self._closed:
            return
        self._closed = True
        for queue in self._subscribers:
            queue.put_nowait(None)

    @contextmanager
    def subscribe(self) -> Iterator[AsyncIterator[GameEvent]]:
        """Follow the game: its full current state first, then every move as it is made.

        The events end when the session is closed.
        """
        queue: asyncio.Queue[GameEvent | None] = asyncio.Queue()
        queue.put_nowait(GameStateEvent(game=self.state))
        if self._closed:
            queue.put_nowait(None)
        self._subscribers.add(queue)
        try:
            yield _events_until_closed(queue)
        finally:
            self._subscribers.discard(queue)

    def _make_move(self, choice: PlayerMove) -> None:
        move = choice.move
        if not self._board.is_legal(move):
            raise IllegalMoveError(f"{move.uci()} is not a legal move in {self._board.fen()}")
        record = MoveRecord(uci=move.uci(), san=self._board.san(move), thoughts=choice.thoughts)
        self._board.push(move)
        self._moves.append(record)
        self._position = snapshot(self._board)
        event = MoveEvent(ply=len(self._moves), move=record, position=self._position)
        for queue in self._subscribers:
            queue.put_nowait(event)


async def _events_until_closed(
    queue: asyncio.Queue[GameEvent | None],
) -> AsyncIterator[GameEvent]:
    while (event := await queue.get()) is not None:
        yield event
