"""Players and event-stream helpers for tests of game sessions."""

from collections.abc import AsyncIterator, Sequence

import chess

from chess_ai.game_session import GameEvent, GameState
from chess_ai.players import GameContext, PlayerMove, Thoughts


class ScriptedPlayer:
    """Plays the given moves in order, whether they are legal or not."""

    def __init__(self, moves: Sequence[str], thoughts: Thoughts | None = None) -> None:
        self.name = "Scripted"
        self._moves = iter(moves)
        self._thoughts = thoughts

    async def choose_move(self, context: GameContext) -> PlayerMove:
        return PlayerMove(chess.Move.from_uci(next(self._moves)), self._thoughts)


def scripted_players(moves: str) -> tuple[ScriptedPlayer, ScriptedPlayer]:
    """White and Black players for a game given as space-separated UCI moves."""
    ucis = moves.split()
    return ScriptedPlayer(ucis[0::2]), ScriptedPlayer(ucis[1::2])


async def collect(events: AsyncIterator[GameEvent]) -> list[GameEvent]:
    return [event async for event in events]


def replay(events: Sequence[GameEvent]) -> GameState:
    """The game state a subscriber ends up with, checking that the stream is consistent."""
    first, *rest = events
    assert first.type == "state"
    game = first.game.model_copy(deep=True)
    for event in rest:
        assert event.type == "move"
        assert event.ply == len(game.moves) + 1
        game.moves.append(event.move)
        game.position = event.position
    return game
