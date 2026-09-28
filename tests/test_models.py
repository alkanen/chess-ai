"""The model zoo: the shape contract every architecture keeps, and that the wiring works."""

import chess
import pytest
import torch
from torch.nn import functional as F
from training_helpers import records

from chess_ai.dataset import Result
from chess_ai.encoders import EncoderSpec, create_encoder
from chess_ai.models import VALUE_CLASSES, ChessModel, architecture_names, create_model
from chess_ai.move_codec import VOCABULARY_SIZE
from chess_ai.registry import RegistryError

POSITIONS = [
    chess.Board(),
    chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
    chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3"),
]
"""A handful of positions to check shapes on, and to make a tiny model memorize."""


@pytest.fixture
def spec() -> EncoderSpec:
    return create_encoder("board-planes").spec


@pytest.fixture
def batch(spec):
    """The test positions, encoded and as tensors, the way the trainer would hand them over."""
    bundle = create_encoder("board-planes").encode(records(*POSITIONS))
    return torch.from_numpy(bundle.spatial), torch.from_numpy(bundle.globals)


@pytest.mark.parametrize("architecture", architecture_names())
def test_every_architecture_answers_with_a_policy_and_a_value(architecture, spec, batch):
    model = create_model(architecture, spec, width=16)

    out = model(*batch)

    assert out.policy.shape == (len(POSITIONS), spec.policy_size) == (4, VOCABULARY_SIZE)
    assert out.value.shape == (len(POSITIONS), VALUE_CLASSES) == (4, len(Result))
    assert torch.isfinite(out.policy).all() and torch.isfinite(out.value).all()


@pytest.mark.parametrize("architecture", architecture_names())
def test_every_architecture_keeps_the_spec_it_was_built_from(architecture, spec):
    assert create_model(architecture, spec, width=16).spec == spec


@pytest.mark.parametrize("architecture", architecture_names())
def test_every_architecture_is_a_chess_model(architecture, spec):
    model = create_model(architecture, spec, width=16)

    assert isinstance(model, ChessModel)
    assert model.parameter_count == sum(p.numel() for p in model.parameters()) > 0


def test_an_unknown_architecture_says_what_there_is(spec):
    with pytest.raises(RegistryError, match="no architecture called 'mpl'.*mlp"):
        create_model("mpl", spec)


def test_an_unknown_hyperparameter_is_refused_rather_than_ignored(spec):
    with pytest.raises(RegistryError, match="has no option 'widht', did you mean 'width'"):
        create_model("mlp", spec, widht=64)


def test_the_encoder_spec_cannot_be_overridden_as_an_option(spec):
    with pytest.raises(RegistryError, match="has no option 'spec'"):
        create_model("mlp", spec, spec=spec)


def test_a_model_is_built_from_whatever_shapes_the_spec_gives():
    """A model reads its input width off the spec, not off the encoder it happens to come from."""
    wider = EncoderSpec(
        encoder="made-up", spatial_channels=20, global_features=5, policy_size=VOCABULARY_SIZE
    )

    model = create_model("mlp", wider, depth=1, width=16)
    out = model(torch.zeros(2, 20, 8, 8), torch.zeros(2, 5))

    assert out.policy.shape == (2, VOCABULARY_SIZE)


def test_the_mlp_sizes_itself_from_depth_and_width(spec):
    narrow = create_model("mlp", spec, depth=2, width=32)
    wide = create_model("mlp", spec, depth=2, width=64)
    deep = create_model("mlp", spec, depth=4, width=32)

    assert narrow.parameter_count < wide.parameter_count
    assert narrow.parameter_count < deep.parameter_count


def test_a_depthless_mlp_reads_the_input_directly(spec, batch):
    """Depth 0 is a linear model, which is a baseline worth being able to ask for."""
    model = create_model("mlp", spec, depth=0)

    assert model(*batch).policy.shape == (len(POSITIONS), spec.policy_size)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"depth": -1}, "depth must not be negative"),
        ({"width": 0}, "width must be at least 1"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
    ],
)
def test_a_size_that_cannot_be_built_says_so(spec, options, message):
    with pytest.raises(ValueError, match=message):
        create_model("mlp", spec, **options)


@pytest.mark.parametrize("architecture", architecture_names())
def test_every_architecture_can_memorize_a_handful_of_positions(architecture, spec, batch):
    """The cheapest test that catches wiring mistakes.

    A model with more parameters than four positions have information should be able to learn
    those four by heart. If it cannot, something is not connected: a head reading the wrong
    tensor, a loss against the wrong target, gradients not reaching the trunk. None of that shows
    up in a shape check, and all of it shows up here in a second.
    """
    torch.manual_seed(0)
    model = create_model(architecture, spec, width=64)
    moves = torch.arange(len(POSITIONS)) * 37  # Distinct, arbitrary, and inside the vocabulary.
    results = torch.tensor([Result.WIN, Result.LOSS, Result.DRAW, Result.WIN])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for _ in range(200):
        out = model(*batch)
        loss = F.cross_entropy(out.policy, moves) + F.cross_entropy(out.value, results)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    out = model(*batch)
    assert loss.item() < 0.05, "a model this size should learn four positions by heart"
    assert torch.equal(out.policy.argmax(dim=1), moves)
    assert torch.equal(out.value.argmax(dim=1), results)
