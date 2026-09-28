"""The encoder registry: the names an experiment config chooses an input representation by."""

from typing import Any, Final, Protocol

import numpy as np

from chess_ai.encoders.spec import EncoderSpec, InputBundle
from chess_ai.registry import Registry


class Encoder(Protocol):
    """What every encoder does: say its shapes, and turn dataset records into model input."""

    @property
    def spec(self) -> EncoderSpec:
        """The shapes this encoder produces, and the settings it was made with."""
        ...

    def encode(self, positions: np.ndarray) -> InputBundle:
        """Encode a batch of :data:`~chess_ai.dataset.POSITION_DTYPE` records."""
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
