"""Encoders: dataset records turned into the input a model sees.

A dataset stores what the games were; an encoder decides what the network is shown. Keeping the
two apart is what lets one dataset feed every architecture and every input representation that
comes later, and it is why the encoder is chosen by name in an experiment config rather than
built into the dataset or the model.

- :mod:`~chess_ai.encoders.spec` is the contract: the shapes, and the arrays themselves
- :mod:`~chess_ai.encoders.planes` is the board-planes encoder
- :mod:`~chess_ai.encoders.registry` maps a config's encoder name to a constructor
"""

from chess_ai.encoders.planes import (
    DEFAULT_RATING_SCALE,
    GLOBAL_FEATURES,
    HALFMOVE_CLOCK_SCALE,
    PIECE_PLANES,
    BoardPlanesEncoder,
)
from chess_ai.encoders.registry import (
    ENCODERS,
    Encoder,
    create_encoder,
    encoder_defaults,
    encoder_names,
)
from chess_ai.encoders.spec import BOARD_SIZE, EncoderSpec, InputBundle

BOARD_PLANES = "board-planes"
"""The name the board-planes encoder is chosen by, which is the default in a config."""

__all__ = [
    "BOARD_PLANES",
    "BOARD_SIZE",
    "DEFAULT_RATING_SCALE",
    "ENCODERS",
    "GLOBAL_FEATURES",
    "HALFMOVE_CLOCK_SCALE",
    "PIECE_PLANES",
    "BoardPlanesEncoder",
    "Encoder",
    "EncoderSpec",
    "InputBundle",
    "create_encoder",
    "encoder_defaults",
    "encoder_names",
]
