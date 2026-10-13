"""The square transformer: the 64 squares as tokens, with self-attention across the board.

A convolution sees a square's neighbours and has to stack layers to see further; attention lets
every square look at every other one in a single layer, which is what a bishop on a1 and a king on
h8 ask of a network. What it gives up is the convolution's built-in sense of which squares are
next to which: a learned embedding per square is all that tells e4 from d5, and the network has to
learn the geometry from it.

Each square becomes one token: every channel the encoder puts on that square (the current board,
and the earlier positions when the encoder shows history) goes through one linear layer shared by
all the squares, and the square's own embedding is added. History stays inside the square's token
rather than becoming tokens of its own, so a position is 64 tokens however far back it looks.

The features that are not on the board (side to move, castling rights, the halfmove clock, the
ratings) become tokens too, one per feature: its value times a learned vector, plus a learned
embedding that says which feature it is. They take part in attention like the squares do, so any
square can read the ratings at any depth, but the heads read only the squares.

The heads are the residual tower's: each squeezes every square to a few features and reads the
whole vocabulary, or the three results, off the flattened board. Keeping them the same keeps a
comparison with the ResNet about the body rather than the heads.
"""

from typing import Final

import torch
from torch import nn

from chess_ai.encoders import EncoderSpec
from chess_ai.models.base import VALUE_CLASSES, ChessModel, ModelOutput
from chess_ai.models.registry import MODELS
from chess_ai.models.resnet import HEAD_CHANNELS, VALUE_HIDDEN

NAME: Final = "square-transformer"
"""The name this architecture is chosen by in an experiment config."""

EMBEDDING_STD: Final = 0.02
"""How far the learned embeddings start from zero, as GPT-2 and BERT start theirs: small next to
the projected squares, so the network begins by reading the board and learns where it is."""


@MODELS.register(NAME)
class SquareTransformer(ChessModel):
    """``layers`` pre-norm transformer layers of ``width`` over the square and feature tokens.

    ``heads`` attention heads split the width between them, so it has to divide it. Each layer's
    feed-forward part is ``ffn_multiplier`` times as wide as the tokens. ``dropout`` is there for
    when a run starts to overfit, which none has yet; it is off by default, as the MLP's is.
    """

    def __init__(
        self,
        spec: EncoderSpec,
        *,
        layers: int = 6,
        width: int = 128,
        heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(spec)
        if spec.spatial_channels < 1:
            raise ValueError(
                f"a square transformer needs board planes, and {spec.encoder} has none"
            )
        if layers < 0:
            raise ValueError(f"layers must not be negative, not {layers}")
        if width < 1:
            raise ValueError(f"width must be at least 1, not {width}")
        if heads < 1:
            raise ValueError(f"heads must be at least 1, not {heads}")
        if width % heads:
            raise ValueError(f"heads must divide width, and {heads} does not divide {width}")
        if ffn_multiplier < 1:
            raise ValueError(f"ffn_multiplier must be at least 1, not {ffn_multiplier}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), not {dropout}")

        squares = spec.board_size * spec.board_size
        self.square_projection = nn.Linear(spec.spatial_channels, width)
        self.square_embedding = nn.Parameter(torch.randn(squares, width) * EMBEDDING_STD)
        self.feature_value = nn.Parameter(torch.randn(spec.global_features, width) * EMBEDDING_STD)
        self.feature_embedding = nn.Parameter(
            torch.randn(spec.global_features, width) * EMBEDDING_STD
        )

        # Pre-norm (norm_first) trains without a careful warmup, and the stock layer is what
        # PyTorch fuses into one attention kernel under bfloat16. A list of layers built one by
        # one rather than nn.TransformerEncoder, which deep-copies a single layer so that every
        # layer starts from the same weights, and which cannot be built with no layers.
        self.body = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                heads,
                dim_feedforward=ffn_multiplier * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        # Pre-norm layers leave their output unnormalized, so the heads get a norm of their own.
        self.final_norm = nn.LayerNorm(width)

        self.policy_head = nn.Sequential(
            *_squeeze(width),
            nn.Linear(HEAD_CHANNELS * squares, spec.policy_size),
        )
        self.value_head = nn.Sequential(
            *_squeeze(width),
            nn.Linear(HEAD_CHANNELS * squares, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, VALUE_CLASSES),
        )

    def forward(self, spatial: torch.Tensor, globals: torch.Tensor) -> ModelOutput:
        # (batch, channels, rank, file) to (batch, square, channels): one row per square.
        squares = self.square_projection(spatial.flatten(2).transpose(1, 2))
        squares = squares + self.square_embedding
        features = globals[:, :, None].to(squares.dtype) * self.feature_value
        features = features + self.feature_embedding
        tokens = torch.cat((squares, features), dim=1)
        for layer in self.body:
            tokens = layer(tokens)
        board = self.final_norm(tokens[:, : squares.shape[1]])
        return ModelOutput(policy=self.policy_head(board), value=self.value_head(board))


def _squeeze(width: int) -> list[nn.Module]:
    """A head's first layers: every square down to :data:`HEAD_CHANNELS` features, flattened."""
    return [nn.Linear(width, HEAD_CHANNELS), nn.ReLU(), nn.Flatten()]
