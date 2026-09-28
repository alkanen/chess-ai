"""The MLP baseline: flatten everything and put dense layers on it.

It throws away the board's geometry — square e4 is just another input — which makes it the
honest floor for everything that does not: a residual tower or a transformer that cannot beat it
per parameter is not earning its structure. It is also the first thing to train, because a
wiring mistake shows up here in seconds rather than in a two-day run.
"""

from typing import Final

import torch
from torch import nn

from chess_ai.encoders import EncoderSpec
from chess_ai.models.base import VALUE_CLASSES, ChessModel, ModelOutput
from chess_ai.models.registry import MODELS

NAME: Final = "mlp"
"""The name this architecture is chosen by in an experiment config."""


@MODELS.register(NAME)
class MLP(ChessModel):
    """``depth`` hidden layers of ``width`` units, then a policy head and a value head.

    Both heads read the same trunk, so the two losses share every layer but the last: what tells
    a winning position from a losing one is most of what tells a good move from a bad one, and
    the value head is cheap insurance against the policy learning to imitate moves without any
    notion of who is winning.
    """

    def __init__(
        self,
        spec: EncoderSpec,
        *,
        depth: int = 3,
        width: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(spec)
        if depth < 0:
            raise ValueError(f"depth must not be negative, not {depth}")
        if depth and width < 1:
            raise ValueError(f"width must be at least 1, not {width}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), not {dropout}")

        inputs = spec.spatial_features + spec.global_features
        layers: list[nn.Module] = []
        size = inputs
        for _ in range(depth):
            layers.append(nn.Linear(size, width))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            size = width
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(size, spec.policy_size)
        self.value_head = nn.Linear(size, VALUE_CLASSES)

    def forward(self, spatial: torch.Tensor, globals: torch.Tensor) -> ModelOutput:
        # flatten(1) keeps the batch and folds channels, ranks and files into one vector, which
        # is the whole of what this architecture does with the board's shape.
        inputs = torch.cat((spatial.flatten(1), globals), dim=1)
        hidden = self.trunk(inputs)
        return ModelOutput(policy=self.policy_head(hidden), value=self.value_head(hidden))
