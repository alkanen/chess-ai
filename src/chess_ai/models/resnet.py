"""The residual tower: the AlphaZero and Maia architecture, convolutions over the 8x8 board.

Where the MLP sees 768 unrelated numbers, a convolution sees a square and its neighbours, with
the same weights on every square: what a knight on e4 attacks is learnt once rather than 64
times. Stacking residual blocks widens what each square can see, two ranks and files per block,
so a handful of blocks already lets every square take in the whole board.

The board is not all of a position, though. Side to move, castling rights, the halfmove clock
and both players' ratings arrive as a separate vector, and a convolution has nowhere to put a
vector. ``globals`` chooses between the two usual answers:

- ``"planes"`` paints each feature over a whole 8x8 plane and stacks those planes on the board's,
  so the first convolution reads them like any other input. Simple, and what AlphaZero did with
  its side-to-move and move-count planes; but after the first layer the features are only what
  that layer made of them.
- ``"film"`` (feature-wise linear modulation) leaves the board alone and lets the vector scale and
  shift every channel of every block instead: each block has a small linear layer that turns the
  features into one multiplier and one offset per channel. A rating can then turn a whole
  channel up or down at every depth of the tower, which is the kind of influence "play like a
  1500" plausibly needs.
"""

from typing import Final, Literal, get_args

import torch
from torch import nn

from chess_ai.encoders import EncoderSpec
from chess_ai.models.base import VALUE_CLASSES, ChessModel, ModelOutput
from chess_ai.models.registry import MODELS

NAME: Final = "resnet"
"""The name this architecture is chosen by in an experiment config."""

Globals = Literal["planes", "film"]
"""How the tower takes in the features that are not on the board; see the module docstring."""

HEAD_CHANNELS: Final = 32
"""Channels each head squeezes the tower down to before its dense layers, as Leela Chess Zero's
policy and value heads do. Fixed rather than configurable: the tower is where the size is."""

VALUE_HIDDEN: Final = 128
"""Units in the value head's one hidden layer."""


class _FiLM(nn.Module):
    """One multiplier and one offset per channel, computed from the global features.

    Starts as the identity, multiplier one and offset zero for every input, so a tower built with
    FiLM begins exactly as a plain one would and the features earn their influence by training.
    The bias is what a batch norm's own scale and shift would otherwise be, which is why the norm
    in front of this has none.
    """

    def __init__(self, features: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(features, 2 * channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(globals)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class _Block(nn.Module):
    """Two 3x3 convolutions and a skip connection around them, conditioned by FiLM if asked."""

    def __init__(self, channels: int, film_features: int | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.film = None if film_features is None else _FiLM(film_features, channels)
        self.norm2 = nn.BatchNorm2d(channels, affine=self.film is None)

    def forward(self, x: torch.Tensor, globals: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.film is not None:
            out = self.film(out, globals)
        return torch.relu(x + out)


@MODELS.register(NAME)
class ResNet(ChessModel):
    """A 3x3 convolution into ``channels`` planes, ``blocks`` residual blocks, then two heads.

    The policy head squeezes the tower to a few planes and reads every move of the vocabulary off
    them with one dense layer, since the vocabulary is a flat list rather than a stack of
    per-square planes. The value head squeezes likewise and goes through one hidden layer to the
    three results. Both read the same tower, for the reason the MLP's heads share a trunk.
    """

    def __init__(
        self,
        spec: EncoderSpec,
        *,
        blocks: int = 6,
        channels: int = 64,
        globals: Globals = "planes",
    ) -> None:
        super().__init__(spec)
        if spec.spatial_channels < 1:
            raise ValueError(f"a residual tower needs board planes, and {spec.encoder} has none")
        if blocks < 0:
            raise ValueError(f"blocks must not be negative, not {blocks}")
        if channels < 1:
            raise ValueError(f"channels must be at least 1, not {channels}")
        if globals not in get_args(Globals):
            raise ValueError(
                f"globals must be one of {', '.join(map(repr, get_args(Globals)))}, not {globals!r}"
            )

        self.globals_as_planes = globals == "planes"
        film_features = None if self.globals_as_planes else spec.global_features
        inputs = spec.spatial_channels + (spec.global_features if self.globals_as_planes else 0)
        self.stem = nn.Conv2d(inputs, channels, 3, padding=1, bias=False)
        self.stem_film = None if film_features is None else _FiLM(film_features, channels)
        self.stem_norm = nn.BatchNorm2d(channels, affine=self.stem_film is None)
        self.tower = nn.ModuleList(_Block(channels, film_features) for _ in range(blocks))

        squares = spec.board_size * spec.board_size
        self.policy_head = nn.Sequential(
            *_squeeze(channels),
            nn.Linear(HEAD_CHANNELS * squares, spec.policy_size),
        )
        self.value_head = nn.Sequential(
            *_squeeze(channels),
            nn.Linear(HEAD_CHANNELS * squares, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, VALUE_CLASSES),
        )

    def forward(self, spatial: torch.Tensor, globals: torch.Tensor) -> ModelOutput:
        x = spatial
        if self.globals_as_planes:
            # expand() is a view, so the constant planes cost nothing until cat copies them in.
            planes = globals[:, :, None, None].expand(-1, -1, *spatial.shape[2:])
            x = torch.cat((spatial, planes.to(spatial.dtype)), dim=1)
        x = self.stem_norm(self.stem(x))
        if self.stem_film is not None:
            x = self.stem_film(x, globals)
        x = torch.relu(x)
        for block in self.tower:
            x = block(x, globals)
        return ModelOutput(policy=self.policy_head(x), value=self.value_head(x))


def _squeeze(channels: int) -> list[nn.Module]:
    """A head's first layers: a 1x1 convolution down to :data:`HEAD_CHANNELS` planes, flattened."""
    return [
        nn.Conv2d(channels, HEAD_CHANNELS, 1, bias=False),
        nn.BatchNorm2d(HEAD_CHANNELS),
        nn.ReLU(),
        nn.Flatten(),
    ]
