"""Playing with a checkpoint: what the network says, which move comes out, and a whole game.

The models here are untrained, which is the point rather than a shortcut: a network that has
learned nothing puts nearly all of its probability on moves that cannot be played, so masking,
renormalizing and reporting the leftover mass are all being asked to do real work.
"""

import asyncio
import math
import random
from pathlib import Path

import chess
import pytest
import torch
from training_helpers import model_run

from chess_ai.encoders import create_encoder
from chess_ai.game_session import GameSession
from chess_ai.inference import (
    InferenceEngine,
    InferenceError,
    ModelPlayer,
    load_engine,
    select_move,
)
from chess_ai.inference.selection import MIN_TEMPERATURE
from chess_ai.models import ChessModel, ModelOutput, create_model
from chess_ai.move_codec import VOCABULARY_SIZE, move_index
from chess_ai.players import GameContext
from chess_ai.training import checkpoint
from chess_ai.training.run_store import RunReader

POSITIONS = {
    "the opening position": chess.STARTING_FEN,
    "both sides castling": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "an en-passant capture": "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
    "a pawn that must promote": "8/P6k/8/8/8/8/6K1/8 w - - 0 1",
    "a king in check": "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3",
}
"""Positions whose legal moves are nothing like the vocabulary, so masking has to work."""


def engine(seed: int = 0, **settings) -> InferenceEngine:
    """An engine on untrained weights, built without going near a file."""
    encoder = create_encoder("board-planes")
    torch.manual_seed(seed)
    return InferenceEngine(create_model("mlp", encoder.spec, depth=1, width=4), encoder, **settings)


class FixedModel(ChessModel):
    """A model that answers the same logits whatever it is shown, so a test can do the sums."""

    def __init__(self, spec, policy: torch.Tensor, value: torch.Tensor) -> None:
        super().__init__(spec)
        self.register_buffer("policy", policy)
        self.register_buffer("value", value)

    def forward(self, spatial: torch.Tensor, globals: torch.Tensor) -> ModelOutput:
        batch = spatial.shape[0]
        return ModelOutput(policy=self.policy.expand(batch, -1), value=self.value.expand(batch, -1))


def fixed_engine(policy: dict[str, float], elsewhere: float, value=(0.0, 0.0, 0.0)):
    """An engine whose policy is ``elsewhere`` everywhere but the UCI moves named."""
    encoder = create_encoder("board-planes")
    logits = torch.full((1, VOCABULARY_SIZE), elsewhere)
    for uci, logit in policy.items():
        logits[0, move_index(chess.Move.from_uci(uci))] = logit
    return InferenceEngine(
        FixedModel(encoder.spec, logits, torch.tensor([value])),
        encoder,
    )


@pytest.mark.parametrize("fen", POSITIONS.values(), ids=list(POSITIONS))
def test_the_distribution_covers_every_legal_move_and_nothing_else(fen):
    board = chess.Board(fen)

    evaluation = engine().evaluate(board, mover_rating=1600, opponent_rating=1600)

    assert {move for move, _ in evaluation.moves} == set(board.legal_moves)
    assert sum(probability for _, probability in evaluation.moves) == pytest.approx(1.0)
    assert all(probability >= 0.0 for _, probability in evaluation.moves)


@pytest.mark.parametrize("fen", POSITIONS.values(), ids=list(POSITIONS))
@pytest.mark.parametrize("strategy", ["argmax", "sample"])
def test_masking_always_leaves_a_legal_move_to_play(fen, strategy):
    """The one thing an untrained model must not be able to do is play an impossible move."""
    board = chess.Board(fen)
    evaluation = engine().evaluate(board)
    rng = random.Random(1)

    for _ in range(20):
        move = select_move(evaluation, strategy=strategy, rng=rng)
        assert board.is_legal(move)


def test_the_mass_the_raw_policy_put_on_illegal_moves_is_reported():
    """Half the mass on a legal move and half on one that cannot be played here."""
    board = chess.Board()
    legal, illegal = "e2e4", "a1a8"
    assert not board.is_legal(chess.Move.from_uci(illegal))

    evaluation = fixed_engine({legal: 10.0, illegal: 10.0}, elsewhere=-40.0).evaluate(board)

    assert evaluation.illegal_mass == pytest.approx(0.5, abs=1e-6)
    assert dict(evaluation.moves)[chess.Move.from_uci(legal)] == pytest.approx(1.0, abs=1e-6)


def test_an_untrained_model_puts_nearly_everything_on_moves_it_cannot_play():
    """1,800-odd moves in the vocabulary and twenty on the board: the mask is doing the work."""
    evaluation = engine().evaluate(chess.Board())

    assert evaluation.illegal_mass > 0.9
    assert sum(probability for _, probability in evaluation.moves) == pytest.approx(1.0)


def test_masking_before_normalizing_is_not_the_same_as_ranking_the_raw_policy():
    """The probabilities are the legal moves' share of themselves, not of the whole vocabulary.

    Everything illegal is scored far above both legal moves here, so a distribution that had
    only been truncated would give these two almost nothing. What they keep is their odds
    against each other, which one logit apart makes a factor of e.
    """
    evaluation = fixed_engine({"e2e4": 1.0, "e2e3": 0.0}, elsewhere=5.0).evaluate(chess.Board())

    probabilities = dict(evaluation.moves)
    pushed, stepped = (probabilities[chess.Move.from_uci(uci)] for uci in ("e2e4", "e2e3"))
    assert pushed / stepped == pytest.approx(math.e)
    assert evaluation.illegal_mass > 0.99


def test_win_draw_loss_comes_back_in_the_order_the_value_head_was_trained_on():
    """The value head's classes are loss, draw and win, as a dataset records a result."""
    evaluation = fixed_engine({}, elsewhere=0.0, value=(-10.0, 0.0, 10.0)).evaluate(chess.Board())

    wdl = evaluation.wdl
    assert wdl.win > wdl.draw > wdl.loss
    assert wdl.win + wdl.draw + wdl.loss == pytest.approx(1.0)


def test_argmax_is_deterministic():
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    played = engine(seed=3)

    chosen = {select_move(played.evaluate(board), strategy="argmax") for _ in range(10)}

    assert len(chosen) == 1


def test_equally_likely_moves_are_still_ranked_the_same_way_every_time():
    """Nothing about a flat distribution should make the move played depend on the run."""
    flat = fixed_engine({}, elsewhere=1.0)

    first = flat.evaluate(chess.Board()).moves
    again = flat.evaluate(chess.Board()).moves

    assert first == again
    assert [move.uci() for move, _ in first] == sorted(move.uci() for move, _ in first)


def test_seeded_sampling_is_reproducible():
    evaluation = engine(seed=5).evaluate(chess.Board())

    one = [select_move(evaluation, strategy="sample", rng=random.Random(11)) for _ in range(5)]
    other = [select_move(evaluation, strategy="sample", rng=random.Random(11)) for _ in range(5)]

    assert one == other


def test_sampling_spreads_over_the_distribution_rather_than_repeating_the_best_move():
    evaluation = engine(seed=5).evaluate(chess.Board())
    rng = random.Random(2)

    drawn = {
        select_move(evaluation, strategy="sample", temperature=1.5, rng=rng) for _ in range(40)
    }

    assert len(drawn) > 1


def test_a_cold_temperature_samples_the_best_move():
    evaluation = engine(seed=5).evaluate(chess.Board())
    rng = random.Random(3)

    drawn = {
        select_move(evaluation, strategy="sample", temperature=MIN_TEMPERATURE, rng=rng)
        for _ in range(10)
    }

    assert drawn == {evaluation.best}


def test_a_temperature_that_is_not_a_temperature_is_refused():
    evaluation = engine().evaluate(chess.Board())

    with pytest.raises(ValueError, match="temperature"):
        select_move(evaluation, strategy="sample", temperature=0.0, rng=random.Random(0))


def test_argmax_ignores_the_temperature_rather_than_being_refused_one():
    """A form that carries a temperature about sends it whether or not it is sampling."""
    evaluation = engine().evaluate(chess.Board())

    assert select_move(evaluation, strategy="argmax", temperature=0.0) == evaluation.best


def test_a_batch_says_what_the_positions_say_one_at_a_time():
    """Batched for throughput, so a batch must say what the positions say on their own.

    Approximately, because it cannot be exactly: a matrix multiplication of five rows adds up
    in a different order from one of a single row, and the last bits of the two differ.
    """
    boards = [chess.Board(fen) for fen in POSITIONS.values()]
    played = engine(seed=4, batch_size=2)

    together = played.evaluate_many(boards, mover_rating=1500)

    alone = [played.evaluate(board, mover_rating=1500) for board in boards]
    assert len(together) == len(boards)
    for batched, single in zip(together, alone, strict=True):
        assert batched.best == single.best
        assert _named(batched) == pytest.approx(_named(single), rel=1e-5)
        assert batched.illegal_mass == pytest.approx(single.illegal_mass, rel=1e-5)
        assert batched.wdl.model_dump() == pytest.approx(single.wdl.model_dump(), rel=1e-5)


def _named(evaluation) -> dict[str, float]:
    """An evaluation's moves by their UCI, which is how two of them are compared by number."""
    return {move.uci(): probability for move, probability in evaluation.moves}


def test_a_rating_changes_what_the_model_is_asked():
    """Rating conditioning reaches the network, or asking to play like a 1600 means nothing."""
    board = chess.Board()
    played = engine(seed=6)

    assert played.evaluate(board, mover_rating=1200) != played.evaluate(board, mover_rating=2400)


def test_a_position_with_no_move_in_it_is_refused():
    """Fool's mate, which a game session never asks about and a caller could."""
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

    with pytest.raises(ValueError, match="no legal move"):
        engine().evaluate(mated)


def test_a_batch_of_at_most_one_position_is_still_a_batch():
    with pytest.raises(ValueError, match="batch_size"):
        engine(batch_size=0)


def test_a_checkpoint_loads_and_plays_without_the_run_it_came_from(tmp_path):
    """A checkpoint carries its architecture, its options and its encoder spec, and that is all."""
    run = RunReader(model_run(tmp_path))
    path = run.checkpoint_path(run.latest_checkpoint())

    played = load_engine(path)

    assert (played.run, played.step) == ("tiny", 4)
    assert played.spec == create_encoder("board-planes").spec
    assert chess.Board().is_legal(played.evaluate(chess.Board()).best)


def test_two_checkpoints_of_one_run_are_two_different_models(tmp_path):
    run = RunReader(model_run(tmp_path))

    best = load_engine(run.checkpoint_path(run.best_checkpoint()))
    latest = load_engine(run.checkpoint_path(run.latest_checkpoint()))

    assert best.step != latest.step
    assert best.evaluate(chess.Board()) != latest.evaluate(chess.Board())


def test_a_file_that_is_not_a_checkpoint_says_so_rather_than_failing_on_a_move(tmp_path):
    not_a_checkpoint = tmp_path / "weights.pt"
    not_a_checkpoint.write_bytes(b"not a checkpoint")

    with pytest.raises(InferenceError, match="cannot read checkpoint"):
        load_engine(not_a_checkpoint)


def test_a_checkpoint_that_is_not_there_says_where_it_looked(tmp_path):
    with pytest.raises(InferenceError, match="no checkpoint at"):
        load_engine(tmp_path / "step-000000004.pt")


def test_a_checkpoint_of_an_architecture_this_code_does_not_have(tmp_path):
    run = RunReader(model_run(tmp_path))
    path = run.checkpoint_path(run.latest_checkpoint())
    payload = checkpoint.load(path)
    payload["architecture"] = "transmogrifier"
    checkpoint.save(payload, path)

    with pytest.raises(InferenceError, match="transmogrifier"):
        load_engine(path)


def test_a_checkpoint_whose_weights_do_not_fit_its_own_description(tmp_path):
    """The options say one model and the weights are another's; playing it would be nonsense."""
    run = RunReader(model_run(tmp_path))
    path = run.checkpoint_path(run.latest_checkpoint())
    payload = checkpoint.load(path)
    payload["model_options"] = {"depth": 1, "width": 64}
    checkpoint.save(payload, path)

    with pytest.raises(InferenceError, match="do not fit"):
        load_engine(path)


def test_a_checkpoint_trained_on_an_encoder_that_has_since_changed(tmp_path):
    run = RunReader(model_run(tmp_path))
    path = run.checkpoint_path(run.latest_checkpoint())
    payload = checkpoint.load(path)
    # The encoder was built with a rating scale, and this checkpoint now says it was not.
    payload["encoder"] = {**payload["encoder"], "options": {}}
    checkpoint.save(payload, path)

    with pytest.raises(InferenceError, match="rating_scale"):
        load_engine(path)


def player(path: Path, **settings) -> ModelPlayer:
    return ModelPlayer(load_engine(path), **settings)


def checkpoints(tmp_path) -> RunReader:
    return RunReader(model_run(tmp_path))


def test_a_model_player_says_which_weights_it_plays_with(tmp_path):
    run = checkpoints(tmp_path)

    played = player(run.checkpoint_path(run.latest_checkpoint()), rating=1600, strategy="sample")

    assert played.name == "tiny step 4"
    assert played.model.run == "tiny"
    assert played.model.checkpoint == 4
    assert played.model.rating == 1600
    assert (played.model.strategy, played.model.temperature) == ("sample", 1.0)


def test_a_player_that_plays_its_best_move_records_no_temperature(tmp_path):
    """There is no temperature in an argmax game, and one recorded would read as a setting."""
    run = checkpoints(tmp_path)

    played = player(run.checkpoint_path(run.best_checkpoint()), temperature=2.0)

    assert played.model.strategy == "argmax"
    assert played.model.temperature is None


def test_a_model_player_says_what_it_was_considering(tmp_path):
    run = checkpoints(tmp_path)
    played = player(run.checkpoint_path(run.latest_checkpoint()), rating=1600)

    chosen = asyncio.run(played.choose_move(GameContext(chess.Board())))

    assert chess.Board().is_legal(chosen.move)
    assert chosen.thoughts is not None
    assert [candidate.uci for candidate in chosen.thoughts.candidates][0] == chosen.move.uci()
    assert chosen.thoughts.wdl is not None
    considered = sum(candidate.probability for candidate in chosen.thoughts.candidates)
    assert 0.0 < considered <= 1.0


def test_a_game_between_two_checkpoints_is_played_to_its_end(tmp_path):
    """Model against model, which is the game the acceptance criteria ask to be able to watch."""
    run = checkpoints(tmp_path)
    white = player(run.checkpoint_path(run.best_checkpoint()), rating=1200)
    black = player(run.checkpoint_path(run.latest_checkpoint()), rating=2000, strategy="sample")
    session = GameSession(white, black, id="two-models")

    asyncio.run(_play(session))

    state = session.state
    assert state.position.game_over is not None
    assert state.moves, "a finished game has moves in it"
    board = chess.Board()
    for record in state.moves:
        board.push_uci(record.uci)
    assert board.fen() == state.position.fen


def test_two_seeded_samplers_play_the_same_game_twice(tmp_path):
    run = checkpoints(tmp_path)
    path = run.checkpoint_path(run.latest_checkpoint())

    games = []
    for _ in range(2):
        session = GameSession(
            player(path, strategy="sample", seed=17),
            player(path, strategy="sample", seed=23),
            id="seeded",
        )
        asyncio.run(_play(session))
        games.append([record.uci for record in session.state.moves])

    assert games[0] == games[1]


async def _play(session: GameSession) -> None:
    """Play a whole game, under a timeout so that a broken player fails rather than hangs."""
    async with asyncio.timeout(120):
        await session.play()


def oriented_engine(policy: dict[str, float], elsewhere: float = -50.0) -> InferenceEngine:
    """An engine on a side-to-move encoder, whose model scores the UCI moves named."""
    encoder = create_encoder("board-planes", orientation="side-to-move")
    logits = torch.full((1, VOCABULARY_SIZE), elsewhere)
    for uci, logit in policy.items():
        logits[0, move_index(chess.Move.from_uci(uci))] = logit
    return InferenceEngine(FixedModel(encoder.spec, logits, torch.zeros(1, 3)), encoder)


def test_an_oriented_models_predictions_are_turned_back_for_the_real_board():
    """The model says "push the e-pawn two squares" and means the mover's e-pawn."""
    played = oriented_engine({"e2e4": 5.0, "g1f3": 3.0})
    black_to_move = chess.Board()
    black_to_move.push_san("d4")

    as_white = played.evaluate(chess.Board())
    as_black = played.evaluate(black_to_move)

    assert [move.uci() for move, _ in as_white.top(2)] == ["e2e4", "g1f3"]
    assert [move.uci() for move, _ in as_black.top(2)] == ["e7e5", "g8f6"]
    assert {move for move, _ in as_black.moves} == set(black_to_move.legal_moves)
    assert sum(probability for _, probability in as_black.moves) == pytest.approx(1.0)
    assert as_black.illegal_mass == pytest.approx(as_white.illegal_mass)


def test_an_oriented_model_that_answers_for_the_wrong_side_is_playing_illegal_moves():
    """e7e5 in the model's terms is a move from its own seventh rank: not black's pawn push."""
    played = oriented_engine({"e7e5": 20.0})
    black_to_move = chess.Board()
    black_to_move.push_san("d4")

    assert played.evaluate(black_to_move).illegal_mass == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("options", [{"history": 3}, {"history": 2, "orientation": "side-to-move"}])
def test_an_engine_plays_from_a_game_with_its_history_behind_it(options):
    encoder = create_encoder("board-planes", **options)
    torch.manual_seed(0)
    played = InferenceEngine(create_model("mlp", encoder.spec, depth=1, width=4), encoder)
    board = chess.Board()
    from_fen = []
    for san in ("e4", "c5", "Nf3"):
        board.push_san(san)
        from_fen.append(chess.Board(board.fen()))

    with_history = played.evaluate(board)
    without = played.evaluate(from_fen[-1])

    assert {move for move, _ in with_history.moves} == set(board.legal_moves)
    assert _named(with_history) != pytest.approx(_named(without)), "the history is an input"
    assert len(played.evaluate_many([board, *from_fen])) == 4


def test_a_checkpoint_saved_before_the_encoder_had_these_options_still_loads(tmp_path):
    """Its spec names only the rating scale, which is still the spec of that encoder."""
    run = RunReader(model_run(tmp_path))
    path = run.checkpoint_path(run.latest_checkpoint())
    assert checkpoint.load(path)["encoder"]["options"] == {"rating_scale": 5000.0}

    assert load_engine(path).evaluate(chess.Board()).best in chess.Board().legal_moves
