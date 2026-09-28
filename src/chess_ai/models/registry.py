"""The architecture registry: the names an experiment config chooses a model by."""

from typing import Any, Final

from chess_ai.encoders import EncoderSpec
from chess_ai.models.base import ChessModel
from chess_ai.registry import Registry

MODELS: Final[Registry[ChessModel]] = Registry("architecture")
"""Every registered architecture, by the name a config chooses it with."""


def create_model(architecture: str, spec: EncoderSpec, /, **hyperparameters: Any) -> ChessModel:
    """The architecture registered as ``architecture``, built for ``spec``.

    Raises :exc:`~chess_ai.registry.RegistryError` for a name that is not registered or a
    hyperparameter that architecture does not take. Both arguments are positional so that
    neither can collide with a hyperparameter a config file names.
    """
    return MODELS.create(architecture, spec, **hyperparameters)


def architecture_names() -> list[str]:
    """Every registered architecture name, sorted."""
    return MODELS.names()


def architecture_defaults(architecture: str) -> dict[str, Any]:
    """What each of ``architecture``'s hyperparameters is when a config does not set it."""
    return MODELS.defaults(architecture)
