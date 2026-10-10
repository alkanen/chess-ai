"""Probe positions: a fixed set of labelled positions every checkpoint of a run is shown.

What a checkpoint makes of them — its most likely moves, how likely, and who it thinks is
winning — is a picture of what it has learned that a loss curve does not give: whether it
recaptures, whether it sees a mate in one, whether it knows the Lucena position. The same
positions for every checkpoint, so that stepping through a run's checkpoints shows how that
changed as it trained.

A set is a TOML file of probes, with a name and a version, like an opening set. The sets that
come with the project live in ``probe_sets`` next to this module; any other file can be given
by its path. Each probe is reached by playing its moves from its starting position, so that a
model that reads the moves before a position is shown real ones. A probe with a solution says
which moves solve it.

What a checkpoint made of the set is kept in the run directory, beside what any other
evaluation finds out about it, as ``probe-positions.json``. It records which set and version it
was measured on and which checkpoint file it was measured with, which is how the evaluator
tells a result that still stands from one to measure again.
"""

import logging
import math
import tomllib
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chess_ai.players import ModelDescription, WinDrawLoss
from chess_ai.position_view import InvalidFenError, board_from_fen
from chess_ai.training.run_store import (
    CheckpointInfo,
    RunError,
    RunReader,
    _replace_file,
    save_evaluation,
)

if TYPE_CHECKING:
    from chess_ai.inference import InferenceEngine
    from chess_ai.inference.engine import Evaluation

LOGGER = logging.getLogger(__name__)

SUITE: Final = "probe-positions"
"""What probing is called among the evaluation suites, and what its result file is named."""

FORMAT_VERSION: Final = 1

DEFAULT_PROBE_SET: Final = "standard"

TOP_MOVES: Final = 5
"""How many of a checkpoint's most likely moves are kept for each probe."""

_BUNDLED = "probe_sets"
_SUFFIX = ".toml"

Category = Literal["opening", "middlegame", "tactic", "endgame"]


class DivergedError(ValueError):
    """A checkpoint's network gives NaN or infinite outputs, as diverged weights do."""


class ProbeSetError(Exception):
    """A probe set cannot be found or read, or holds a probe that is not a position."""


@dataclass(frozen=True)
class Probe:
    id: str
    """What names the probe across versions of its set; it never changes."""
    name: str
    category: Category
    start: str | None
    """The FEN the moves are played from, or ``None`` for the usual starting position."""
    moves: tuple[chess.Move, ...]
    """The moves that lead to the position probed, every one of them legal where it is played."""
    best: tuple[chess.Move, ...]
    """The moves that solve the probe, or none for a position that has no one solution."""
    comment: str | None

    def board(self) -> chess.Board:
        """The position probed, with the moves that led to it as the board's history."""
        board = chess.Board(self.start) if self.start is not None else chess.Board()
        for move in self.moves:
            board.push(move)
        return board


@dataclass(frozen=True)
class ProbeSet:
    name: str
    version: int
    probes: tuple[Probe, ...]

    @property
    def label(self) -> str:
        """The set and its version, such as "standard v1"."""
        return f"{self.name} v{self.version}"


class _ProbeFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    category: Category
    fen: str | None = None
    moves: str = ""
    best: list[str] = []
    comment: str | None = None


class _ProbeSetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = Field(ge=1)
    probes: list[_ProbeFile] = Field(min_length=1)


def bundled_probe_sets() -> list[str]:
    """The names of the sets that come with the project, sorted."""
    return sorted(
        entry.name.removesuffix(_SUFFIX)
        for entry in resources.files(__package__).joinpath(_BUNDLED).iterdir()
        if entry.name.endswith(_SUFFIX)
    )


def load_probe_set(name_or_path: str | Path) -> ProbeSet:
    """The probe set by the name it comes with, or read from a file of one's own.

    A name of a bundled set is looked up first; anything else is taken to be a path.

    Raises:
        ProbeSetError: there is no such set or file, it is not a probe set, two probes share an
            id or a position, or a probe's position cannot be reached, has no move to make, or
            names a solution that cannot be played in it.
    """
    text, where = _read(str(name_or_path))
    try:
        written = _ProbeSetFile.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as e:
        raise ProbeSetError(f"{where} is not valid TOML: {e}") from e
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in e.errors()
        )
        raise ProbeSetError(f"{where} is not a probe set: {problems}") from e
    probes = tuple(_probe(probe, where) for probe in written.probes)
    ids: set[str] = set()
    reached: dict[str, str] = {}
    for probe in probes:
        if probe.id in ids:
            raise ProbeSetError(f"{where}: two probes are called {probe.id!r}")
        ids.add(probe.id)
        position = probe.board().epd()
        if position in reached:
            raise ProbeSetError(
                f"{where}: {probe.id!r} is the same position as {reached[position]!r}"
            )
        reached[position] = probe.id
    return ProbeSet(name=written.name, version=written.version, probes=probes)


def _read(name_or_path: str) -> tuple[str, str]:
    """The text of the set, and how to name where it came from in a message."""
    if name_or_path in bundled_probe_sets():
        bundled = resources.files(__package__).joinpath(_BUNDLED, f"{name_or_path}{_SUFFIX}")
        return bundled.read_text(encoding="utf-8"), f"probe set {name_or_path!r}"
    path = Path(name_or_path)
    try:
        return path.read_text(encoding="utf-8"), str(path)
    except UnicodeDecodeError as e:
        raise ProbeSetError(f"{path} is not UTF-8 text: {e}") from e
    except FileNotFoundError as e:
        raise ProbeSetError(
            f"there is no probe set {name_or_path!r}: give the path of a file, or one of "
            f"the sets that come with chess-ai ({', '.join(bundled_probe_sets())})"
        ) from e
    except OSError as e:
        raise ProbeSetError(f"cannot read {path}: {e.strerror}") from e


def _probe(written: _ProbeFile, where: str) -> Probe:
    named = f"{where}: {written.id!r}"
    try:
        board = board_from_fen(written.fen) if written.fen is not None else chess.Board()
    except InvalidFenError as e:
        raise ProbeSetError(f"{named}: {e}") from e
    for san in written.moves.split():
        try:
            board.push_san(san)
        except ValueError as e:
            raise ProbeSetError(f"{named} cannot play {san} after {_played(board)}") from e
    if board.is_game_over():
        raise ProbeSetError(f"{named}: the game is over, so there is no move to probe")
    best = []
    for san in written.best:
        try:
            best.append(board.parse_san(san))
        except ValueError as e:
            raise ProbeSetError(f"{named}: {san} cannot be played in the position") from e
    return Probe(
        id=written.id,
        name=written.name,
        category=written.category,
        start=board.root().fen() if written.fen is not None else None,
        moves=tuple(board.move_stack),
        best=tuple(best),
        comment=written.comment,
    )


def _played(board: chess.Board) -> str:
    """The moves so far, for a message about the one that could not follow them."""
    if not board.move_stack:
        return "no moves"
    return board.root().variation_san(board.move_stack)


class ProbeMove(BaseModel):
    """One of the moves a checkpoint thought most likely in a probe."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    uci: str
    san: str
    probability: float
    """Out of the legal moves alone, which sum to one."""


class ProbeOutcome(BaseModel):
    """What a checkpoint made of one probe, with the probe as it was then."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    id: str
    name: str
    category: Category
    comment: str | None
    start: str | None
    """The FEN the moves were played from, or ``None`` for the usual starting position."""
    moves: list[str]
    """The moves that lead to the position, in UCI."""
    line: str
    """The same moves in standard notation, numbered, such as ``1. e4 e5 2. Nf3``."""
    fen: str
    """The position probed."""
    best: list[str]
    """The moves that solve it, in UCI, or none for a position without one solution."""
    top: list[ProbeMove]
    """The checkpoint's most likely moves, most likely first."""
    wdl: WinDrawLoss
    """Its chances, from the side to move's point of view."""
    illegal_mass: float
    """What the raw network put on moves that cannot be played here."""
    best_probability: float | None
    """How likely it thought the moves in :attr:`best` together, or ``None`` without any."""

    @property
    def solved(self) -> bool | None:
        """Whether its most likely move is a solution, or ``None`` for a probe without one."""
        if not self.best:
            return None
        return self.top[0].uci in self.best


class ProbeResult(BaseModel):
    """Everything probing found out about a checkpoint, as ``probe-positions.json`` holds it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    format_version: Literal[1] = FORMAT_VERSION
    suite: Literal["probe-positions"] = SUITE
    model: ModelDescription
    """The run, the checkpoint, and the rating both sides were said to have."""
    checkpoint_written: datetime
    """When the checkpoint file was written, which tells it from another of the same step: a
    run started again under the same name, or resumed from an earlier checkpoint."""
    started: datetime
    finished: datetime
    code_version: str
    probe_set: str
    probe_set_version: int
    positions: list[ProbeOutcome]

    def solved(self) -> tuple[int, int]:
        """How many of the probes that have a solution were solved, and how many have one."""
        marked = [outcome.solved for outcome in self.positions if outcome.solved is not None]
        return sum(marked), len(marked)


def check_finite(evaluation: "Evaluation", what: str) -> None:
    """Make sure ``evaluation``, of the position ``what`` names, is made of numbers.

    Raises:
        DivergedError: some of it is NaN or infinite, as the outputs of diverged weights are.
    """
    # The illegal mass too: it is a softmax over every logit, so a NaN in one the legal moves
    # leave out shows only there.
    numbers = [
        *evaluation.probabilities(),
        *evaluation.wdl.model_dump().values(),
        evaluation.illegal_mass,
    ]
    if not all(math.isfinite(number) for number in numbers):
        raise DivergedError(
            f"the network's outputs for {what} are not a number: its weights have most likely "
            "diverged"
        )


def probe(engine: "InferenceEngine", probes: ProbeSet, *, rating: int | None) -> list[ProbeOutcome]:
    """What ``engine`` makes of every probe in ``probes``, with both sides at ``rating``.

    Raises:
        DivergedError: the network gives NaN or infinite probabilities, as diverged weights do.
    """
    boards = [each.board() for each in probes.probes]
    evaluations = engine.evaluate_many(boards, mover_rating=rating, opponent_rating=rating)
    outcomes = []
    for each, board, evaluation in zip(probes.probes, boards, evaluations, strict=True):
        # Written down, a NaN is a null that no result can be read back from.
        check_finite(evaluation, repr(each.id))
        best = {move.uci() for move in each.best}
        outcomes.append(
            ProbeOutcome(
                id=each.id,
                name=each.name,
                category=each.category,
                comment=each.comment,
                start=each.start,
                moves=[move.uci() for move in each.moves],
                line=board.root().variation_san(board.move_stack),
                fen=board.fen(),
                best=sorted(best),
                top=[
                    ProbeMove(uci=move.uci(), san=board.san(move), probability=probability)
                    for move, probability in evaluation.top(TOP_MOVES)
                ],
                wdl=evaluation.wdl,
                illegal_mass=evaluation.illegal_mass,
                best_probability=(
                    sum(p for move, p in evaluation.moves if move.uci() in best) if best else None
                ),
            )
        )
    return outcomes


def result_path(run: RunReader, step: int) -> Path:
    """Where the probes' result for the checkpoint from ``step`` is kept."""
    return run.evaluation_directory(step) / f"{SUITE}.json"


def save_probes(run: RunReader, result: ProbeResult, checkpoint: CheckpointInfo) -> None:
    """Keep ``result`` for ``checkpoint``, replacing any earlier one, if it is still the file
    the result was measured with.

    Raises:
        CheckpointChanged: the checkpoint's file has been replaced or deleted since; nothing
            was saved.
        RunError: the run directory cannot be written.
    """
    save_evaluation(
        run.evaluation_directory(result.model.checkpoint),
        {f"{SUITE}.json": result.model_dump_json(indent=2) + "\n"},
        current=lambda: run.checkpoint_written(checkpoint) == result.checkpoint_written,
    )


def read_probes(run: RunReader, step: int) -> ProbeResult | None:
    """The probes' result for the checkpoint from ``step``, or ``None`` if there is none.

    Raises:
        RunError: there is one, and it cannot be read or is not a result this code knows.
    """
    text = run.evaluation(step, SUITE)
    if text is None:
        return None
    try:
        return ProbeResult.model_validate_json(text)
    except ValidationError as e:
        raise RunError(f"{result_path(run, step)} is not a probe result: {e}") from e


def stands(result: ProbeResult, step: int, written: datetime, probes: ProbeSet) -> bool:
    """Whether ``result`` is still what probing the checkpoint from ``step``, whose file was
    written at ``written``, with ``probes`` would find.

    Not when it was measured on another set or version of one, nor with another file of the
    same step. The rating and the code it was measured with are not asked about: a result is
    redone for those only on demand.
    """
    return (
        result.probe_set == probes.name
        and result.probe_set_version == probes.version
        and result.model.checkpoint == step
        and result.checkpoint_written == written
    )


CURRENT_SET_FILE: Final = "probe-set.json"
"""In the runs directory: which set the evaluator working on it probes checkpoints with."""


class ProbeSetVersion(BaseModel):
    """A probe set by name and version."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: int


def save_current_set(runs_dir: Path, probes: ProbeSet) -> None:
    """Record that the evaluator working on ``runs_dir`` probes checkpoints with ``probes``.

    Only that evaluator writes it, and only one works on a runs directory at a time, so the
    record has one writer. It says what the evaluator reads when it starts, which is what it
    goes on probing with until it is started again, whatever happens to the set's file.
    """
    record = ProbeSetVersion(name=probes.name, version=probes.version)
    _replace_file(runs_dir / CURRENT_SET_FILE, record.model_dump_json(indent=2) + "\n")


def read_current_set(runs_dir: Path) -> ProbeSetVersion | None:
    """Which set the evaluator working on ``runs_dir`` probes checkpoints with, or ``None`` when
    no evaluator has said, or what it said cannot be read."""
    path = runs_dir / CURRENT_SET_FILE
    try:
        return ProbeSetVersion.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as e:
        # Debug: every viewer of a run asks about once a second.
        LOGGER.debug("Cannot tell which probe set the evaluator uses: %s: %s", path, e)
        return None


def summary(result: ProbeResult) -> str:
    """One line about ``result``, such as ``solved 9 of 17 probes (standard v1)``."""
    solved, out_of = result.solved()
    return f"solved {solved} of {out_of} probes ({result.probe_set} v{result.probe_set_version})"
