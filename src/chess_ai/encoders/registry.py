"""The encoder registry: the names an experiment config chooses an input representation by."""

from typing import Any, Final, Protocol

import chess
import numpy as np

from chess_ai.encoders.spec import EncoderSpec, InputBundle
from chess_ai.registry import Registry


class Encoder(Protocol):
    """What every encoder does: say its shapes, and turn positions into model input.

    Two ways in, for the two places positions come from. Training hands over a batch of dataset
    records; a game being played hands over the board itself, which has no move played in it, no
    result and no game around it to fill a record's other fields with. Both come out as the same
    bundle, so a checkpoint sees at play time exactly what it was trained on.
    """

    @property
    def spec(self) -> EncoderSpec:
        """The shapes this encoder produces, and the settings it was made with."""
        ...

    history: int
    """How many earlier positions of its game each position is encoded with, 0 for none.

    Whoever gathers the records gathers this many more behind each; see
    :meth:`~chess_ai.dataset.SplitReader.position_history`.
    """

    def encode(self, positions: np.ndarray) -> InputBundle:
        """Encode a batch of :data:`~chess_ai.dataset.POSITION_DTYPE` records.

        Either (batch,) positions, or (batch, 1 + history): each position and then the ones
        before it in its game, most recent first.
        """
        ...

    def model_moves(self, moves: np.ndarray, white_to_move: np.ndarray | bool) -> np.ndarray:
        """Vocabulary indices of moves on the board, as the indices the model predicts.

        An encoder may show the model a transformed board, and then it has to transform the
        moves to match. Training targets go through this on the way in.
        """
        ...

    def board_moves(self, moves: np.ndarray, white_to_move: np.ndarray | bool) -> np.ndarray:
        """The inverse of :meth:`model_moves`: what the model predicted, as moves on the board."""
        ...

    def encode_board(
        self,
        board: chess.Board,
        *,
        mover_rating: int | None = None,
        opponent_rating: int | None = None,
    ) -> InputBundle:
        """Encode one live position as a batch of one.

        The ratings are the side to move's and their opponent's, which is what a rating-
        conditioned model is asked to play like. ``None`` is the rating a position does not
        claim, flagged as unknown, exactly as a dataset records a game that gave none.
        """
        ...


ENCODERS: Final[Registry[Encoder]] = Registry("encoder")
"""Every registered encoder, by the name a config chooses it with."""


def create_encoder(name: str, /, **options: Any) -> Encoder:
    """The encoder registered as ``name``, made with ``options``.

    Raises :exc:`~chess_ai.registry.RegistryError` for a name that is not registered or an
    option that encoder does not take. ``name`` is positional so that it cannot collide with
    an option a config file names.
    """
    return ENCODERS.create(name, **options)


def encoder_names() -> list[str]:
    """Every registered encoder name, sorted."""
    return ENCODERS.names()


def encoder_defaults(name: str) -> dict[str, Any]:
    """What each of encoder ``name``'s options is when a config does not set it."""
    return ENCODERS.defaults(name)
