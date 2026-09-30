"""What every architecture here is: one input bundle in, a policy and a value out.

The two outputs are the whole reason the trainer, the evaluator, the inference engine and the
browser can treat an MLP and a GPT identically. A model never sees a ``chess.Board`` and never
names a move: it is handed the arrays an encoder produced and answers with a score for every
entry of the shared move vocabulary and a three-way judgement of the position.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn

from chess_ai.dataset import Result
from chess_ai.encoders import EncoderSpec

VALUE_CLASSES = len(Result)
"""Loss, draw and win — the value head's three classes, in :class:`Result`'s order."""


class ModelOutput(NamedTuple):
    """One forward pass's logits, before any softmax or legality masking."""

    policy: torch.Tensor
    """(batch, policy size) logits over the whole move vocabulary, legal or not.

    Unmasked on purpose. The network is trained to put its mass on legal moves by itself, and
    how much it fails to is a metric worth watching; masking happens where a move is actually
    chosen, not here.
    """
    value: torch.Tensor
    """(batch, 3) logits over loss, draw and win **from the point of view of the side to move**,
    which is what the dataset records as each position's result."""


class ChessModel(nn.Module, ABC):
    """A registered architecture, built from an encoder spec and its own hyperparameters.

    Holds the spec it was built from so that a checkpoint can refuse an encoder that would feed
    it the wrong shapes, and so that a run directory can say what the model was trained on
    without the config file being at hand.
    """

    def __init__(self, spec: EncoderSpec) -> None:
        super().__init__()
        self.spec = spec

    @abstractmethod
    def forward(self, spatial: torch.Tensor, globals: torch.Tensor) -> ModelOutput:
        """Score every move and judge the position, for a whole batch.

        ``spatial`` is (batch, channels, board, board) and ``globals`` is (batch, features), as
        the spec describes them, and both are floats. An encoder's planes are bytes, which is
        not what a model takes: whatever moves them to the device converts them, as
        :meth:`~chess_ai.training.batches.Batch.to` does, so a raw
        :class:`~chess_ai.encoders.InputBundle` is never handed over as it is.
        """

    @property
    def parameter_count(self) -> int:
        """How many parameters this model has, which is what a run reports before it starts."""
        return sum(parameter.numel() for parameter in self.parameters())
