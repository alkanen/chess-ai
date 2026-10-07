"""Validation metrics, against a model whose answers are decided in advance.

A real model's numbers can only be checked for being between 0 and 1. These use a model that
answers what the test tells it to, so top-1, top-5 and the illegal-move rate can be checked
against the arithmetic.
"""

import numpy as np
import pytest
import torch
from training_helpers import dataset

from chess_ai.dataset import Result, open_dataset
from chess_ai.dataset.manifest import Filters
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
    encoder = create_encoder("board-planes", **options.pop("encoder", {}))
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


ORIENTED = {"orientation": "side-to-move"}


def test_an_oriented_model_is_scored_on_the_moves_as_it_was_shown_them(split):
    """The model answers in its own terms, mirrored for black, and that is the right answer."""
    encoder = create_encoder("board-planes", **ORIENTED)
    white = [bool(unpack_board(split.position(index)).turn) for index in range(8)]
    assert not all(white), "black is to move in some of these"
    moves = encoder.model_moves(np.array(played(split, 8)), np.array(white)).tolist()

    metrics = measure(split, lambda row: [moves[row]], encoder=ORIENTED)

    assert metrics["top1"] == 1.0
    assert metrics["illegal_top_move_rate"] == 0.0, "turned back before the rules are asked"


def test_an_oriented_model_that_answers_for_the_real_board_is_wrong_for_black(split):
    moves = played(split, 8)
    black = sum(not unpack_board(split.position(index)).turn for index in range(8))

    metrics = measure(split, lambda row: [moves[row]], encoder=ORIENTED)

    assert metrics["top1"] == pytest.approx((8 - black) / 8)


def test_validation_gives_the_encoder_the_history_it_asks_for(split):
    seen = []

    class Watching(Scripted):
        def forward(self, spatial, globals):  # noqa: A002
            seen.append(spatial.clone())
            return super().forward(spatial, globals)

    encoder = create_encoder("board-planes", history=1)
    validate(
        Watching(encoder.spec, lambda row: [0]),
        split,
        encoder,
        positions=8,
        batch_size=8,
        device=torch.device("cpu"),
        value_loss_weight=0.5,
    )

    (spatial,) = seen
    assert spatial.shape == (8, 24, 8, 8)
    plain = create_encoder("board-planes").encode(split.positions(np.arange(8))).spatial
    # The first eight positions are one game's first eight plies, so each one's history is
    # the position before it, and the first has none.
    assert [int(split.position(index)["ply"]) for index in range(8)] == list(range(8))
    assert spatial[0, 12:].sum() == 0
    assert np.array_equal(spatial[1:, 12:].numpy(), plain[:-1])


def test_a_filtered_split_is_measured_on_its_training_targets(tmp_path):
    dataset(tmp_path / "data", filters=Filters(min_rating=1700))
    with open_dataset("test", data_dir=tmp_path / "data") as built:
        split = built["validation"]
        assert 0 < split.targets < len(split), "some positions are stored and not trained on"
        moves = split.positions(split.target_positions(np.arange(split.targets)))["move"]

        metrics = measure(split, lambda n: [int(moves[n])], positions=10_000)

    assert metrics["positions"] == split.targets
    assert metrics["top1"] == 1.0, "every position measured was a target, in order"
