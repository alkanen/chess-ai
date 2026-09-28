"""Validation metrics, against a model whose answers are decided in advance.

A real model's numbers can only be checked for being between 0 and 1. These use a model that
answers what the test tells it to, so top-1, top-5 and the illegal-move rate can be checked
against the arithmetic.
"""

import pytest
import torch
from training_helpers import dataset

from chess_ai.dataset import Result, open_dataset
from chess_ai.dataset.records import unpack_board
from chess_ai.encoders import create_encoder
from chess_ai.models import VALUE_CLASSES, ChessModel, ModelOutput
from chess_ai.move_codec import VOCABULARY_SIZE, move_at, move_index
from chess_ai.training.validate import TOP_K, validate


class Scripted(ChessModel):
    """A model that ranks the moves a test hands it, position by position.

    ``ranking(board, played)`` returns the move indices it should rank first, best first;
    everything else scores lower, in vocabulary order.
    """

    def __init__(self, spec, ranking):
        super().__init__(spec)
        self.ranking = ranking
        self.value = torch.zeros(VALUE_CLASSES)
        self.seen = 0

    def forward(self, spatial, globals):  # noqa: A002 - the bundle's part is called globals
        policy = torch.zeros(len(spatial), VOCABULARY_SIZE)
        for row in range(len(spatial)):
            for rank, index in enumerate(self.ranking(self.seen + row)):
                policy[row, index] = 100.0 - rank
        self.seen += len(spatial)
        return ModelOutput(policy=policy, value=self.value.expand(len(spatial), VALUE_CLASSES))


@pytest.fixture
def split(tmp_path):
    """A validation split to measure on, and the records it holds."""
    dataset(tmp_path / "data")
    with open_dataset("test", data_dir=tmp_path / "data") as built:
        yield built["validation"]


def measure(split, ranking, *, positions=8, **options):
    """Validate a scripted model over ``positions`` positions of ``split``."""
    encoder = create_encoder("board-planes")
    model = Scripted(encoder.spec, ranking)
    return validate(
        model,
        split,
        encoder,
        positions=positions,
        batch_size=options.pop("batch_size", 4),
        device=torch.device("cpu"),
        value_loss_weight=options.pop("value_loss_weight", 0.5),
    )


def played(split, count):
    """The move actually played in each of the first ``count`` positions."""
    return [int(split.position(index)["move"]) for index in range(count)]


def test_a_model_that_always_names_the_played_move_scores_everything(split):
    moves = played(split, 8)

    metrics = measure(split, lambda row: [moves[row]])

    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["illegal_top_move_rate"] == 0.0
    assert metrics["positions"] == 8


def test_a_model_that_ranks_the_played_move_fifth_makes_top_five_and_not_top_one(split):
    moves = played(split, 8)
    legal = [next(iter(unpack_board(split.position(i)).legal_moves)) for i in range(8)]

    # Four other moves ahead of the played one, so it lands exactly at rank five.
    metrics = measure(split, lambda row: [move_index(legal[row]), 1, 2, 3, moves[row]][:TOP_K])

    assert metrics["top1"] == 0.0
    assert metrics["top5"] == pytest.approx(
        sum(move_index(legal[row]) != moves[row] for row in range(8)) / 8
    )


def test_top_one_counts_only_the_positions_it_got_right(split):
    moves = played(split, 8)

    metrics = measure(split, lambda row: [moves[row]] if row % 2 == 0 else [0])

    assert metrics["top1"] == 0.5


def test_a_top_move_that_cannot_be_played_counts_as_illegal(split):
    """Index 0 is a1a2: never legal in a position with a piece of its own on a2, and never the
    played move in these games."""
    metrics = measure(split, lambda row: [0])

    assert metrics["illegal_top_move_rate"] == 1.0
    assert metrics["top1"] == 0.0


def test_a_legal_top_move_that_was_not_played_is_not_counted_as_illegal(split):
    legal = [next(iter(unpack_board(split.position(i)).legal_moves)) for i in range(8)]

    metrics = measure(split, lambda row: [move_index(legal[row])])

    assert metrics["illegal_top_move_rate"] == 0.0


def test_the_combined_loss_weights_the_value_head(split):
    moves = played(split, 8)

    metrics = measure(split, lambda row: [moves[row]], value_loss_weight=0.25)

    assert metrics["loss"] == pytest.approx(metrics["policy_loss"] + 0.25 * metrics["value_loss"])


def test_a_flat_value_head_scores_the_entropy_of_three_equal_classes(split):
    moves = played(split, 8)

    metrics = measure(split, lambda row: [moves[row]])

    assert metrics["value_loss"] == pytest.approx(torch.log(torch.tensor(3.0)).item(), abs=1e-5)
    assert len(Result) == VALUE_CLASSES


def test_the_batch_size_does_not_change_the_answer(split):
    moves = played(split, 8)

    one = measure(split, lambda row: [moves[row]], batch_size=1)
    many = measure(split, lambda row: [moves[row]], batch_size=8)

    assert one == pytest.approx(many)


def test_asking_for_more_positions_than_there_are_uses_what_there_is(split):
    moves = played(split, len(split))

    metrics = measure(split, lambda row: [moves[row]], positions=10_000)

    assert metrics["positions"] == len(split)


def test_validation_leaves_the_model_in_the_mode_it_found_it(split):
    encoder = create_encoder("board-planes")
    model = Scripted(encoder.spec, lambda row: [0])
    model.train()

    validate(
        model,
        split,
        encoder,
        positions=4,
        batch_size=4,
        device=torch.device("cpu"),
        value_loss_weight=0.5,
    )

    assert model.training, "a validation in the middle of a run must not leave dropout off"


def test_validating_on_nothing_is_refused(split):
    encoder = create_encoder("board-planes")

    with pytest.raises(ValueError, match="empty split"):
        validate(
            Scripted(encoder.spec, lambda row: [0]),
            split,
            encoder,
            positions=0,
            batch_size=4,
            device=torch.device("cpu"),
            value_loss_weight=0.5,
        )


def test_the_move_a_record_holds_is_legal_in_the_position_it_holds(split):
    """What the illegal-move check rests on: every played move in the data is itself legal.

    If it were not, the metric would count the model's correct answers as illegal ones, and the
    number would say nothing about the model at all.
    """
    for index in range(len(split)):
        record = split.position(index)
        board = unpack_board(record)

        assert board.is_valid(), board.fen()
        assert board.is_legal(move_at(int(record["move"]))), board.fen()
