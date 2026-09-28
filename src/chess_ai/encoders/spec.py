"""What an encoder produces: the shapes a model is built from, and the arrays it is given.

The spec is the contract between an encoder and every architecture. A model never asks which
encoder it was built for, only how wide its input is, which is what lets one architecture be
trained on several input representations and one encoder feed several architectures. It is
written into the run directory and into every checkpoint, so a checkpoint carries the shape of
the input it expects and can say no to an encoder that would feed it something else.

The bundle is the input itself, as numpy arrays. Encoders stay free of torch on purpose: they
run in data-loader worker processes and in tests, and everything they do is vectorised numpy
over a batch of dataset records.
"""

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

BOARD_SIZE: Final = 8
"""Squares along a rank and a file, which is the shape of the spatial part."""


class EncoderSpec(BaseModel):
    """The shapes one encoder produces, and the settings it produced them with.

    Frozen, because a model and a checkpoint are built against it: a spec that could be
    edited after the fact would be a spec that no longer describes the weights.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoder: str
    """The registered name the encoder was made from; see :mod:`chess_ai.encoders.registry`."""
    spatial_channels: int = Field(ge=0)
    """Planes in the spatial part, 0 for an encoder that produces none."""
    board_size: int = Field(default=BOARD_SIZE, ge=1)
    global_features: int = Field(ge=0)
    """Length of the per-position feature vector that is not laid out on the board."""
    sequence_length: int = Field(default=0, ge=0)
    """Move tokens per example, 0 for an encoder that produces no sequence."""
    policy_size: int = Field(gt=0)
    """How wide a policy output is, which is the size of the shared move vocabulary."""
    options: dict[str, Any] = Field(default_factory=dict)
    """The encoder's own settings, recorded so that a run says what it trained on."""

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """The spatial part's shape per example, without the batch dimension."""
        return (self.spatial_channels, self.board_size, self.board_size)

    @property
    def spatial_features(self) -> int:
        """The spatial part flattened, which is what a dense architecture consumes."""
        return self.spatial_channels * self.board_size * self.board_size

    def describe(self) -> str:
        """A line naming the shapes, for the summary a run prints before it starts."""
        parts = [f"spatial {self.spatial_channels}x{self.board_size}x{self.board_size}"]
        parts.append(f"globals {self.global_features}")
        if self.sequence_length:
            parts.append(f"sequence {self.sequence_length}")
        return f"{self.encoder}: {', '.join(parts)}, policy {self.policy_size}"


@dataclass(frozen=True)
class InputBundle:
    """One batch of model input, in the parts an architecture consumes separately.

    Kept apart rather than concatenated because each architecture wants the global features
    somewhere else: an MLP appends them to the flattened board, a residual tower broadcasts
    them over the planes, a transformer adds them as tokens. Concatenating here would decide
    that for all of them.
    """

    spatial: np.ndarray
    """``float32`` of shape (batch, channels, board, board)."""
    globals: np.ndarray
    """``float32`` of shape (batch, features)."""
    sequence: np.ndarray | None = None
    """``int64`` move tokens of shape (batch, length), or ``None`` from a positional encoder."""

    def __len__(self) -> int:
        return len(self.spatial)
