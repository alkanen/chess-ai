"""Playing with a checkpoint: the network's answer, the move chosen from it, and the player.

Three modules, in the order a move passes through them:

- :mod:`~chess_ai.inference.engine` loads a checkpoint and says what the network makes of a
  position: a distribution over the **legal** moves, a win/draw/loss estimate, and how much
  probability the raw policy put on moves that cannot be played
- :mod:`~chess_ai.inference.selection` turns that distribution into one move, by argmax or by
  sampling at a temperature
- :mod:`~chess_ai.inference.player` is the two of them behind the ordinary player interface,
  so that a game session, a match or the browser never has to know a network is involved

This package is the only thing outside training that imports torch. The web server reaches it
when a game is started with a model in it, and not before.
"""

from chess_ai.inference.engine import (
    DEFAULT_BATCH_SIZE,
    DeviceUnavailableError,
    Evaluation,
    InferenceEngine,
    InferenceError,
    load_engine,
)
from chess_ai.inference.player import TOP_CANDIDATES, ModelPlayer
from chess_ai.inference.selection import DEFAULT_TEMPERATURE, MIN_TEMPERATURE, select_move

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TEMPERATURE",
    "DeviceUnavailableError",
    "MIN_TEMPERATURE",
    "TOP_CANDIDATES",
    "Evaluation",
    "InferenceEngine",
    "InferenceError",
    "ModelPlayer",
    "load_engine",
    "select_move",
]
