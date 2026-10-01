"""The model zoo: architectures, chosen by name, all with the same input and the same outputs.

Adding one is implementing :class:`~chess_ai.models.base.ChessModel` and registering it under a
name. The trainer, the evaluator, the inference engine and the browser go on unchanged, because
what they depend on is the pair of outputs rather than the architecture: a policy over the shared
move vocabulary, and a win/draw/loss judgement.
"""

from chess_ai.models.base import VALUE_CLASSES, ChessModel, ModelOutput
from chess_ai.models.mlp import MLP
from chess_ai.models.registry import (
    MODELS,
    architecture_defaults,
    architecture_names,
    create_model,
)
from chess_ai.models.resnet import ResNet

__all__ = [
    "MLP",
    "MODELS",
    "VALUE_CLASSES",
    "ChessModel",
    "ModelOutput",
    "ResNet",
    "architecture_defaults",
    "architecture_names",
    "create_model",
]
