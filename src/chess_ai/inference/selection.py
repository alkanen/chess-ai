"""Move selection: a distribution over legal moves in, one move out.

Kept apart from the network on purpose. The engine says how likely each move is; what to do
with that is a separate decision, and the two strategies here are the simple end of it — a
search will one day make the same decision by spending time instead, and will replace nothing
but this module's part.

Both strategies read only the probabilities, never the logits, so they work on anything that
can produce a distribution over moves.
"""

import random

import chess
import numpy as np

from chess_ai.inference.engine import Evaluation
from chess_ai.players import DEFAULT_TEMPERATURE, MIN_TEMPERATURE, SelectionStrategy

__all__ = ["DEFAULT_TEMPERATURE", "MIN_TEMPERATURE", "select_move"]


def select_move(
    evaluation: Evaluation,
    *,
    strategy: SelectionStrategy = "argmax",
    temperature: float = DEFAULT_TEMPERATURE,
    rng: random.Random | None = None,
) -> chess.Move:
    """The move to play from ``evaluation``, by ``strategy``.

    ``argmax`` plays the most probable legal move, which makes the same position give the same
    move every time. ``sample`` draws from the distribution raised to ``1 / temperature``: 1.0
    is the network's own distribution, below it sharpens towards the best move and above it
    flattens towards a coin toss. Sampling needs an ``rng``, and the same seeded one replays the
    same game.

    Raises:
        ValueError: the temperature is not positive, so there is no distribution to sample.
    """
    if strategy == "argmax":
        return evaluation.best
    if temperature < MIN_TEMPERATURE:
        raise ValueError(f"temperature must be at least {MIN_TEMPERATURE}, not {temperature}")
    weights = _tempered(evaluation.probabilities(), temperature)
    moves = [move for move, _ in evaluation.moves]
    return (rng if rng is not None else random).choices(moves, weights=weights.tolist(), k=1)[0]


def _tempered(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """``probabilities`` raised to ``1 / temperature``, as weights to draw one move by.

    Done in logs and shifted by the largest, which is the same thing as taking the softmax of
    the legal logits at this temperature and cannot underflow to all-zero weights: the most
    probable move always keeps a weight of one, however cold the temperature or however little
    the network thought of any of them. Weights need no normalizing; drawing by them does that.
    """
    # A move the network gave nothing at all keeps a weight of zero rather than becoming a
    # warning: log(0) is the -inf that says "never", which is exactly what is meant here.
    with np.errstate(divide="ignore"):
        logs = np.log(probabilities)
    return np.exp((logs - logs.max()) / temperature)
