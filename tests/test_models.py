"""The model zoo: the shape contract every architecture keeps, and that the wiring works."""

import chess
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from training_helpers import record, records

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


ENCODER_OPTIONS = [
    {},
    {"history": 2},
    {"orientation": "side-to-move"},
    {"history": 3, "orientation": "side-to-move"},
]
"""Every shape of input the encoder can be asked for, which every model has to be built from."""


@pytest.fixture(
    params=ENCODER_OPTIONS, ids=lambda options: ",".join(map(str, options.values())) or "plain"
)
def encoder(request):
    return create_encoder("board-planes", **request.param)


@pytest.fixture
def spec(encoder) -> EncoderSpec:
    return encoder.spec


@pytest.fixture
def batch(encoder):
    """The test positions, encoded and as tensors, the way the trainer would hand them over."""
    bundle = encoder.encode(records(*POSITIONS))
    return torch.from_numpy(bundle.spatial).float(), torch.from_numpy(bundle.globals)


TINY = {
    "mlp": {"width": 16},
    "resnet": {"blocks": 1, "channels": 8},
}
"""Each architecture at a size that builds and runs in milliseconds, by its registered name."""

SMALL = {
    "mlp": {"width": 64},
    "resnet": {"blocks": 2, "channels": 16},
}
"""Each architecture at a size that can learn a handful of positions by heart in a second."""

VARIANTS = [
    pytest.param("mlp", {}, id="mlp"),
    pytest.param("resnet", {"globals": "planes"}, id="resnet-planes"),
    pytest.param("resnet", {"globals": "film"}, id="resnet-film"),
]
"""Every architecture, and every way it has of being wired differently, to check each of them."""


def test_every_architecture_has_a_tiny_size_and_is_checked():
    """A newly registered architecture fails here until the contract tests below cover it."""
    assert sorted(TINY) == sorted(SMALL) == architecture_names()
    assert {variant.values[0] for variant in VARIANTS} == set(architecture_names())


def tiny(architecture: str, spec: EncoderSpec, **options) -> ChessModel:
    return create_model(architecture, spec, **{**TINY[architecture], **options})


def memorize(model, batch, moves, results, *, steps: int = 200) -> float:
    """Train ``model`` on ``batch`` alone until it knows it by heart, or ``steps`` run out.

    Returns the last loss. Stops early because most models get there in a few dozen steps, and
    the dense heads over the whole vocabulary make every step cost something.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(steps):
        out = model(*batch)
        loss = F.cross_entropy(out.policy, moves) + F.cross_entropy(out.value, results)
        if loss.item() < 0.01:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()


def small(architecture: str, spec: EncoderSpec, **options) -> ChessModel:
    return create_model(architecture, spec, **{**SMALL[architecture], **options})


@pytest.mark.parametrize(("architecture", "options"), VARIANTS)
def test_every_architecture_answers_with_a_policy_and_a_value(architecture, options, spec, batch):
    model = tiny(architecture, spec, **options)

    out = model(*batch)

    assert out.policy.shape == (len(POSITIONS), spec.policy_size) == (4, VOCABULARY_SIZE)
    assert out.value.shape == (len(POSITIONS), VALUE_CLASSES) == (4, len(Result))
    assert torch.isfinite(out.policy).all() and torch.isfinite(out.value).all()


@pytest.mark.parametrize(("architecture", "options"), VARIANTS)
def test_every_architecture_answers_one_position_alone_as_it_does_in_a_batch(
    architecture, options, spec, batch
):
    """What the engine does when it plays: one position, the model in eval mode.

    A model with batch norm normalizes by the batch while training, which would make a position's
    answer depend on the others it came with; in eval mode it must not.
    """
    model = tiny(architecture, spec, **options).eval()

    together = model(*batch)
    alone = model(batch[0][:1], batch[1][:1])

    torch.testing.assert_close(alone.policy, together.policy[:1])
    torch.testing.assert_close(alone.value, together.value[:1])


@pytest.mark.parametrize("architecture", architecture_names())
def test_every_architecture_keeps_the_spec_it_was_built_from(architecture, spec):
    assert tiny(architecture, spec).spec == spec


@pytest.mark.parametrize("architecture", architecture_names())
def test_every_architecture_is_a_chess_model(architecture, spec):
    model = tiny(architecture, spec)

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


@pytest.mark.parametrize(("architecture", "options"), VARIANTS)
def test_a_model_is_built_from_whatever_shapes_the_spec_gives(architecture, options):
    """A model reads its input width off the spec, not off the encoder it happens to come from."""
    wider = EncoderSpec(
        encoder="made-up", spatial_channels=20, global_features=5, policy_size=VOCABULARY_SIZE
    )

    model = tiny(architecture, wider, **options)
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


@pytest.mark.parametrize(("architecture", "options"), VARIANTS)
def test_every_architecture_can_memorize_a_handful_of_positions(architecture, options, spec, batch):
    """The cheapest test that catches wiring mistakes.

    A model with more parameters than four positions have information should be able to learn
    those four by heart. If it cannot, something is not connected: a head reading the wrong
    tensor, a loss against the wrong target, gradients not reaching the trunk. None of that shows
    up in a shape check, and all of it shows up here in a second.
    """
    torch.manual_seed(0)
    model = small(architecture, spec, **options)
    moves = torch.arange(len(POSITIONS)) * 37  # Distinct, arbitrary, and inside the vocabulary.
    results = torch.tensor([Result.WIN, Result.LOSS, Result.DRAW, Result.WIN])

    loss = memorize(model, batch, moves, results)

    out = model(*batch)
    assert loss < 0.05, "a model this size should learn four positions by heart"
    assert torch.equal(out.policy.argmax(dim=1), moves)
    assert torch.equal(out.value.argmax(dim=1), results)


@pytest.mark.parametrize("globals", ["planes", "film"])
def test_the_resnet_hears_the_features_that_are_not_on_the_board(globals):
    """Two copies of one board that differ only in the ratings, asked to play different moves.

    Nothing on the planes tells them apart, so a tower that dropped the global features on the
    floor could not learn this however long it trained.
    """
    encoder = create_encoder("board-planes")
    bundle = encoder.encode(
        np.concatenate(
            [record(chess.Board(), mover_rating=800), record(chess.Board(), mover_rating=2800)]
        )
    )
    batch = torch.from_numpy(bundle.spatial).float(), torch.from_numpy(bundle.globals)
    moves = torch.tensor([3, 300])
    results = torch.tensor([Result.LOSS, Result.WIN])
    torch.manual_seed(0)
    model = small("resnet", encoder.spec, globals=globals)

    assert memorize(model, batch, moves, results) < 0.05
    assert torch.equal(model(*batch).policy.argmax(dim=1), moves)


def test_film_starts_out_as_a_plain_tower(spec, batch):
    """Every multiplier one and every offset zero, so the features earn their influence."""
    model = tiny("resnet", spec, globals="film")
    spatial, features = batch

    one = model(spatial, features)
    other = model(spatial, torch.rand_like(features))

    torch.testing.assert_close(one.policy, other.policy)
    torch.testing.assert_close(one.value, other.value)


def test_the_resnet_sizes_itself_from_blocks_and_channels(spec):
    narrow = create_model("resnet", spec, blocks=2, channels=16)
    wide = create_model("resnet", spec, blocks=2, channels=32)
    deep = create_model("resnet", spec, blocks=4, channels=16)

    assert narrow.parameter_count < wide.parameter_count
    assert narrow.parameter_count < deep.parameter_count


def test_a_blockless_resnet_is_a_convolution_and_the_heads(spec, batch):
    model = create_model("resnet", spec, blocks=0, channels=8)

    assert model(*batch).policy.shape == (len(POSITIONS), spec.policy_size)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"blocks": -1}, "blocks must not be negative"),
        ({"channels": 0}, "channels must be at least 1"),
        ({"globals": "tokens"}, "globals must be one of 'planes', 'film', not 'tokens'"),
    ],
)
def test_a_resnet_that_cannot_be_built_says_so(spec, options, message):
    with pytest.raises(ValueError, match=message):
        create_model("resnet", spec, **options)


def test_a_resnet_needs_a_board_to_convolve():
    moves_only = EncoderSpec(
        encoder="moves", spatial_channels=0, global_features=4, sequence_length=16, policy_size=8
    )

    with pytest.raises(ValueError, match="needs board planes, and moves has none"):
        create_model("resnet", moves_only)
