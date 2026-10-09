import hashlib
import textwrap

import chess
import chess.engine
import pytest
from stockfish_helpers import STOCKFISH, needs_stockfish
from training_helpers import model_run

from chess_ai.inference import Evaluation, load_engine
from chess_ai.players import WinDrawLoss
from chess_ai.probes import (
    DEFAULT_PROBE_SET,
    TOP_MOVES,
    DivergedError,
    Probe,
    ProbeSet,
    ProbeSetError,
    bundled_probe_sets,
    load_probe_set,
    probe,
)
from chess_ai.training.run_store import RunReader

STANDARD_V1 = "9ad0af35765377f45fe723af39e15155b47dbafb68cda7f98aad5b62bdfdc2c6"
"""The probes of ``standard`` v1, pinned. A change to the set that keeps its version would show
results measured on different positions side by side as if they were measured on the same."""


def digest(probes) -> str:
    text = "\n".join(
        f"{each.id}|{each.name}|{each.category}|{each.start}|"
        f"{' '.join(move.uci() for move in each.moves)}|"
        f"{' '.join(move.uci() for move in each.best)}|{each.comment}"
        for each in probes.probes
    )
    return hashlib.sha256(text.encode()).hexdigest()


def write_set(path, probes: str, *, version: int = 1):
    path.write_text(f'name = "mine"\nversion = {version}\n\n' + textwrap.dedent(probes))
    return path


def test_the_standard_set_comes_with_the_project_and_is_the_default():
    assert DEFAULT_PROBE_SET == "standard"
    assert "standard" in bundled_probe_sets()

    probes = load_probe_set("standard")

    assert probes.label == "standard v1"
    assert {each.category for each in probes.probes} == {
        "opening",
        "middlegame",
        "tactic",
        "endgame",
    }


def test_the_standard_set_changes_only_with_its_version():
    probes = load_probe_set("standard")

    assert (probes.version, digest(probes)) == (1, STANDARD_V1), (
        "the standard set has changed: raise its version, and pin the new one here"
    )


@needs_stockfish
def test_every_solution_in_the_standard_set_is_clearly_better_than_every_other_move():
    """What the set file promises about ``best``, checked the way it was when it was written."""
    probes = load_probe_set("standard")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    try:
        for each in probes.probes:
            if not each.best:
                continue
            board = each.board()
            others = [move for move in board.legal_moves if move not in each.best]
            weakest = min(score(engine, board, [move]) for move in each.best)
            if others:
                strongest = score(engine, board, others)
                assert clearly_better(weakest, strongest), (each.id, weakest, strongest)
    finally:
        engine.quit()


def score(engine: chess.engine.SimpleEngine, board: chess.Board, moves) -> chess.engine.Score:
    """How good the best of ``moves`` is for the side to move."""
    info = engine.analyse(board, chess.engine.Limit(depth=16), root_moves=list(moves))
    return info["score"].relative


def clearly_better(one: chess.engine.Score, other: chess.engine.Score) -> bool:
    """A faster mate, or a pawn better; a mate in one beats a mate in two, however close."""
    if one.is_mate() or other.is_mate():
        return one > other
    return one.score() > other.score() + 100


def test_a_probe_is_the_position_its_moves_reach_from_its_start(tmp_path):
    path = write_set(
        tmp_path / "mine.toml",
        """
        [[probes]]
        id = "sicilian"
        name = "Sicilian"
        category = "opening"
        moves = "e4 c5"

        [[probes]]
        id = "mate"
        name = "Back rank"
        category = "tactic"
        fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
        best = ["Rd8#"]
        comment = "Mate in one."
        """,
    )

    sicilian, mate = load_probe_set(path).probes

    assert sicilian.start is None
    assert sicilian.board().move_stack == [chess.Move.from_uci("e2e4"), chess.Move.from_uci("c7c5")]
    assert sicilian.best == ()
    assert mate.board().fen() == "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    assert mate.best == (chess.Move.from_uci("d1d8"),)
    assert mate.comment == "Mate in one."


@pytest.mark.parametrize(
    ("probes", "complaint"),
    [
        (
            'id = "a"\nname = "x"\ncategory = "opening"\nmoves = "e4 e4"',
            "cannot play e4 after 1. e4",
        ),
        ('id = "a"\nname = "x"\ncategory = "puzzle"', "category"),
        ('id = "A b"\nname = "x"\ncategory = "opening"', "id"),
        (
            'id = "a"\nname = "x"\ncategory = "opening"\nfen = "8/8/8/8/8/8/8/8 w - - 0 1"',
            "no pieces",
        ),
        ('id = "a"\nname = "x"\ncategory = "tactic"\nmoves = "f3 e5 g4 Qh4#"', "game is over"),
        ('id = "a"\nname = "x"\ncategory = "tactic"\nbest = ["e5"]', "e5 cannot be played"),
        ('id = "a"\nname = "x"\ncategory = "opening"\nlevel = 3', "level"),
    ],
)
def test_a_probe_that_is_not_a_position_to_probe_is_refused(tmp_path, probes, complaint):
    path = write_set(tmp_path / "bad.toml", "[[probes]]\n" + probes + "\n")

    with pytest.raises(ProbeSetError, match=complaint):
        load_probe_set(path)


def test_two_probes_with_one_id_or_one_position_are_refused(tmp_path):
    same_id = write_set(
        tmp_path / "id.toml",
        """
        [[probes]]
        id = "a"
        name = "x"
        category = "opening"

        [[probes]]
        id = "a"
        name = "y"
        category = "opening"
        moves = "e4"
        """,
    )
    same_position = write_set(
        tmp_path / "position.toml",
        """
        [[probes]]
        id = "a"
        name = "x"
        category = "opening"
        moves = "e4 e5 Nf3"

        [[probes]]
        id = "b"
        name = "y"
        category = "opening"
        moves = "Nf3 e5 e4"
        """,
    )

    with pytest.raises(ProbeSetError, match="two probes are called 'a'"):
        load_probe_set(same_id)
    with pytest.raises(ProbeSetError, match="'b' is the same position as 'a'"):
        load_probe_set(same_position)


def test_a_set_that_is_not_there_is_refused_by_name():
    with pytest.raises(ProbeSetError, match="there is no probe set 'nowhere'.*standard"):
        load_probe_set("nowhere")


def test_probing_records_the_most_likely_moves_and_how_likely_the_solutions_were(tmp_path):
    run = RunReader(model_run(tmp_path))
    engine = load_engine(run.checkpoint_path(run.checkpoints()[0]))
    probes = load_probe_set("standard")

    outcomes = probe(engine, probes, rating=2000)

    assert [outcome.id for outcome in outcomes] == [each.id for each in probes.probes]
    for each, outcome in zip(probes.probes, outcomes, strict=True):
        board = each.board()
        assert outcome.fen == board.fen()
        assert len(outcome.top) == min(TOP_MOVES, board.legal_moves.count())
        chances = [move.probability for move in outcome.top]
        assert chances == sorted(chances, reverse=True)
        assert all(board.is_legal(chess.Move.from_uci(move.uci)) for move in outcome.top)
        assert [move.san for move in outcome.top] == [
            board.san(chess.Move.from_uci(move.uci)) for move in outcome.top
        ]
        assert outcome.wdl.win + outcome.wdl.draw + outcome.wdl.loss == pytest.approx(1)
        if each.best:
            assert outcome.best == sorted(move.uci() for move in each.best)
            assert 0 <= outcome.best_probability <= 1
            assert outcome.solved == (outcome.top[0].uci in outcome.best)
        else:
            assert outcome.best_probability is None and outcome.solved is None
    mate = next(outcome for outcome in outcomes if outcome.id == "scholars-mate")
    assert mate.line == "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6"


class Answers:
    """Stands in for an engine, answering every position with the same evaluation."""

    def __init__(self, evaluation) -> None:
        self.evaluation = evaluation

    def evaluate_many(self, boards, **ratings):
        return [self.evaluation] * len(boards)


@pytest.mark.parametrize(
    "broken",
    [
        {"illegal_mass": float("nan")},
        {"wdl": WinDrawLoss(win=float("nan"), draw=0.5, loss=0.5)},
        {"moves": ((chess.Move.from_uci("e2e4"), float("inf")),)},
    ],
    ids=["illegal-mass", "wdl", "probability"],
)
def test_outputs_that_are_not_numbers_are_refused_rather_than_written_as_nulls(broken):
    sane = {
        "moves": ((chess.Move.from_uci("e2e4"), 1.0),),
        "wdl": WinDrawLoss(win=0.4, draw=0.2, loss=0.4),
        "illegal_mass": 0.1,
    }
    single = ProbeSet(
        name="one",
        version=1,
        probes=(Probe("start", "Start", "opening", None, (), (), None),),
    )

    assert probe(Answers(Evaluation(**sane)), single, rating=None)[0].illegal_mass == 0.1
    with pytest.raises(DivergedError, match="not a number"):
        probe(Answers(Evaluation(**{**sane, **broken})), single, rating=None)
