"""The run directory: the only thing that knows where a run keeps what it produces.

A run directory is the contract between the three kinds of process this system has. The trainer
writes it, the evaluator adds to it, and the web server only ever reads it and pushes what
changed to browsers. Nothing holds a handle on anything else, so a web server restart neither
kills a run nor loses track of one, and a run from six months ago is still browsable.

::

    <runs>/<name>/config.toml       the experiment config as it was written, comments and all
    <runs>/<name>/run.json          the same config resolved, plus what it resolved against
    <runs>/<name>/metrics.jsonl     append-only, one JSON object per line
    <runs>/<name>/status.json       the heartbeat: replaced whole, never appended to
    <runs>/<name>/checkpoints/      step-<step>.pt, and an index naming the best of them
    <runs>/<name>/notes.json        a title, tags and notes: written by people, never by the trainer

Two write disciplines, because two kinds of reader:

- **Appending** is for the metrics log, which is a history: a reader tails it, and the only
  thing it can catch half-written is the last line, which it skips and sees whole next time.
- **Replacing** is for everything that is a current value rather than a history. Written to a
  temporary name in the same directory and renamed over the old one, so a reader opening it at
  any moment gets one whole version or the other, never a mixture. A crash mid-write leaves the
  previous version intact.

No file here is ever rewritten in place, which is what makes concurrent reading safe without a
lock between processes that do not otherwise know about each other.
"""

import json
import os
import re
import subprocess
import threading
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chess_ai.dataset.files import sync_directory, sync_file
from chess_ai.encoders import EncoderSpec

FORMAT_VERSION: Final = 1
"""The version of this layout, recorded in every file it writes and checked when reading."""

RUN_FILE: Final = "run.json"
CONFIG_FILE: Final = "config.toml"
METRICS_FILE: Final = "metrics.jsonl"
STATUS_FILE: Final = "status.json"
NOTES_FILE: Final = "notes.json"
CHECKPOINTS_DIR: Final = "checkpoints"
CHECKPOINT_INDEX: Final = "index.json"
CHECKPOINT_SUFFIX: Final = ".pt"
CHECKPOINT_STEP_DIGITS: Final = 9
"""Enough that a checkpoint file's name sorts by step as text, up to a billion steps."""

TEMPORARY_SUFFIX: Final = ".writing"
"""What the name of a file being replaced ends in until it is whole; see the module docstring."""

TRAIN: Final = "train"
VALIDATION: Final = "validation"
"""What the ``split`` field of a metrics line says about where the numbers came from."""

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

CheckpointChoice = Literal["latest", "best"] | int
"""Which of a run's checkpoints is meant: its newest, its best by the run's own metric, or the
one saved at a particular step. See :func:`choose_checkpoint`."""


class RunError(Exception):
    """A run directory cannot be read or written, or a name cannot be a run's."""


class RunStatus(StrEnum):
    """Where a run has got to, as its heartbeat says.

    ``RUNNING`` is the only one a reader should distrust: it is what a run says while it is
    alive, so a heartbeat that has stopped being updated means the process is gone rather than
    that it is still going. Telling a stale heartbeat from a live one is a reader's job, because
    only the reader knows what "recently" means to it.
    """

    RUNNING = "running"
    FINISHED = "finished"
    STOPPED = "stopped"
    """Asked to stop and saved a checkpoint on the way out."""
    CRASHED = "crashed"


class GpuStats(BaseModel):
    """What the hardware said about itself when the heartbeat was written.

    Every field is optional. Utilization and temperature come from NVML, which may not be
    installed, may have no driver to talk to, and on some cards declines one question while
    answering the rest; the memory figures come from CUDA. A missing number is missing, not zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = None
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    """In use on the whole device, as CUDA sees it — including whatever else is using the card."""
    memory_total_bytes: int | None = None
    process_memory_bytes: int | None = None
    """What this run itself is holding, which is the number that says whether the batch fits."""
    temperature_celsius: float | None = None


class DatasetReference(BaseModel):
    """Which dataset a run trained on, in enough detail to find it and to notice a change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    directory: str
    format_version: int
    created: datetime | None = None
    games: int
    positions: int
    train_positions: int
    validation_positions: int


class ModelReference(BaseModel):
    """Which architecture a run trained, with the hyperparameters it was built with."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str
    options: dict[str, Any] = Field(default_factory=dict)
    parameter_count: int


class RunInfo(BaseModel):
    """``run.json``: everything about a run that does not change once it has started.

    This is what makes a run reproducible and comparable: the config as it resolved, the code it
    ran, the seed it ran with, and the dataset and encoder spec it resolved against. A checkpoint
    repeats the encoder spec, so a checkpoint can be loaded without this file; this file exists
    so that a run can be understood without loading a checkpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int = FORMAT_VERSION
    name: str
    created: datetime
    seed: int
    code_version: str
    device: str
    dataset: DatasetReference
    encoder: EncoderSpec
    model: ModelReference
    config: dict[str, Any]
    """The experiment config with every default filled in, as JSON-compatible values."""
    steps: int
    batch_size: int
    positions_per_second_estimate: float | None = None
    """What a short probe measured before training started; see the trainer."""


class Heartbeat(BaseModel):
    """``status.json``: where a run is right now, replaced whole every time it is written."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int = FORMAT_VERSION
    status: RunStatus
    pid: int
    started: datetime
    updated: datetime
    step: int = 0
    steps: int
    epoch: float = 0.0
    """How many times the training split has been gone through, as a fraction."""
    positions_per_second: float | None = None
    eta_seconds: float | None = None
    gpu: GpuStats | None = None
    message: str | None = None
    """Why a run crashed, or anything else worth saying about how it ended."""


class CheckpointInfo(BaseModel):
    """One saved checkpoint, as the index records it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    file: str
    created: datetime
    metrics: dict[str, float | None] = Field(default_factory=dict)
    """The validation metrics at this step, which is what "best" is decided on.

    A metric may be ``None``: a diverged run measures a loss of ``nan``, which pydantic writes
    as ``null`` — the same value :meth:`RunWriter.log` writes for it. The type has to admit it,
    because a ``null`` that cannot be read back is worse than the divergence. It stopped this
    file parsing at all, and since every save reads it to apply retention, one bad step erased
    every earlier checkpoint's metrics and let retention delete the best weights of the run.
    """


class CheckpointPolicy(BaseModel):
    """How many checkpoints to keep, and which one counts as the best.

    ``metric`` names a validation metric and ``higher_is_better`` says which way to read it, so
    that accuracy and loss can both decide it. A checkpoint saved before the first validation
    has no metrics and never wins.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    keep: int = Field(default=3, ge=1)
    metric: str = "policy_loss"
    higher_is_better: bool = False


class CheckpointIndex(BaseModel):
    """``checkpoints/index.json``: the checkpoints there are, and which is best.

    The directory is the truth about which checkpoints exist — a file is either there or it is
    not — and this adds what a file name cannot say. A reader reconciles the two, so a crash
    between writing a checkpoint and rewriting this index costs the index entry, not the
    checkpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int = FORMAT_VERSION
    policy: CheckpointPolicy = CheckpointPolicy()
    best_step: int | None = None
    checkpoints: list[CheckpointInfo] = Field(default_factory=list)


MAX_TITLE_LENGTH: Final = 200
MAX_TAG_LENGTH: Final = 40
MAX_TAGS: Final = 50
MAX_NOTES_LENGTH: Final = 100_000

_TAG = re.compile(r"[^\s,]+")


class RunNotes(BaseModel):
    """What people have said about a run: a title to know it by, tags to find it by, and notes.

    The run's name is its directory, which a trainer may be writing into at this moment and a
    checkpoint loaded elsewhere names it by, so it never changes. The title is the name it is
    shown under instead, and can be changed as often as an experiment's meaning becomes clear.

    Tags are single words, so that a command line can take them one at a time and a list of
    them can be written with commas or spaces between: no whitespace and no commas inside one.
    The same tag twice is kept once, in the order first given.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)
    """What to show the run as, instead of its name; ``None`` to show the name."""
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)
    notes: str = Field(default="", max_length=MAX_NOTES_LENGTH)
    """Free text: what the run was for, and what came of it."""

    @field_validator("title")
    @classmethod
    def _one_line(cls, title: str | None) -> str | None:
        if title is None:
            return None
        title = title.strip()
        if "\n" in title or "\r" in title:
            raise ValueError("a title is one line")
        return title or None

    @field_validator("tags")
    @classmethod
    def _words(cls, tags: list[str]) -> list[str]:
        kept: list[str] = []
        for tag in tags:
            tag = tag.strip()
            if not _TAG.fullmatch(tag):
                raise ValueError(f"invalid tag {tag!r}: a tag is one word, with no commas")
            if len(tag) > MAX_TAG_LENGTH:
                raise ValueError(f"tag {tag!r} is longer than {MAX_TAG_LENGTH} characters")
            if tag not in kept:
                kept.append(tag)
        return kept


class _NotesFile(RunNotes):
    """``notes.json``: the notes, and the layout version they were written in."""

    format_version: int = FORMAT_VERSION


def valid_name(name: str) -> str:
    """``name`` if it can be a run's, else raise :exc:`RunError`.

    A run's name is also its directory, so it has to be a plain one: no separators, no leading
    dot, nothing that would walk out of the runs directory.
    """
    if not _NAME.fullmatch(name):
        raise RunError(
            f"invalid run name {name!r}: use letters, digits and . _ -, "
            "starting with a letter or digit"
        )
    return name


def run_path(runs_dir: Path, name: str) -> Path:
    """Where the run called ``name`` lives."""
    return runs_dir / valid_name(name)


def check_available(directory: Path, name: str, *, overwrite: bool) -> None:
    """Raise :exc:`RunError` if ``directory`` already holds a run and may not be replaced.

    The same refusal :meth:`RunWriter.create` makes, offered separately so that it can be made
    first. Everything between choosing a name and writing the first file — loading the dataset,
    building the model, measuring how fast it runs — is work a person should not wait through to
    be told the name was taken.
    """
    if (directory / RUN_FILE).exists() and not overwrite:
        raise RunError(
            f"run {name!r} is already in {directory}; "
            "choose another name or pass --overwrite to replace it"
        )


def list_runs(runs_dir: Path) -> list[str]:
    """Every run in ``runs_dir``, sorted; the ones that have a ``run.json``, at least."""
    try:
        entries = sorted(entry.name for entry in os.scandir(runs_dir) if entry.is_dir())
    except OSError:
        return []
    return [name for name in entries if (runs_dir / name / RUN_FILE).is_file()]


def code_version() -> str:
    """What the code was when a run started: the git commit, and whether it was edited.

    Falls back to the installed version, because a run from an installed package is still a run
    and its results still have to say what produced them.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        from chess_ai import __version__

        return f"version {__version__}"
    return f"git {revision}{'-dirty' if dirty else ''}"


class RunWriter:
    """The trainer's end of a run directory: everything a run produces goes through here."""

    def __init__(self, directory: Path, policy: CheckpointPolicy) -> None:
        self.directory = directory
        self.policy = policy
        self._metrics = (directory / METRICS_FILE).open("a", encoding="utf-8")
        self._started = _now()

    @classmethod
    def create(
        cls,
        directory: Path,
        info: RunInfo,
        *,
        config_text: str,
        policy: CheckpointPolicy,
        overwrite: bool = False,
    ) -> Self:
        """Make ``directory`` and write what does not change, or raise :exc:`RunError`.

        A directory that is already a run is never written into unless ``overwrite`` says so:
        the cost of mixing two runs' metrics and checkpoints together is a day of training that
        cannot be interpreted, and the cost of refusing is typing a different name.
        """
        check_available(directory, info.name, overwrite=overwrite)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / CHECKPOINTS_DIR).mkdir(exist_ok=True)
            if overwrite:
                cls._clear(directory)
            _replace_file(directory / CONFIG_FILE, config_text)
            _replace_file(directory / RUN_FILE, _as_json(info))
            (directory / METRICS_FILE).touch()
        except OSError as e:
            raise RunError(f"cannot create run directory {directory}: {e.strerror or e}") from e
        return cls(directory, policy)

    @staticmethod
    def _clear(directory: Path) -> None:
        """Throw away what a previous run of this name left, so the two cannot be read as one."""
        # The notes too: they were written about the run being replaced, and a title or a
        # "best so far" tag carried over would describe a run that no longer exists.
        for path in (METRICS_FILE, STATUS_FILE, NOTES_FILE):
            (directory / path).unlink(missing_ok=True)
        checkpoints = directory / CHECKPOINTS_DIR
        for path in checkpoints.glob(f"*{CHECKPOINT_SUFFIX}"):
            path.unlink(missing_ok=True)
        (checkpoints / CHECKPOINT_INDEX).unlink(missing_ok=True)
        # And whatever a process killed mid-write left: SIGKILL, the OOM killer and a power cut
        # give it no chance to tidy up, and a notes file's temporary name is never used twice.
        for path in (
            *directory.glob(f"*{TEMPORARY_SUFFIX}"),
            *checkpoints.glob(f"*{TEMPORARY_SUFFIX}"),
        ):
            path.unlink(missing_ok=True)

    def log(self, **fields: Any) -> None:
        """Append one metrics line. Written whole and flushed, so a tail sees it at once."""
        line = json.dumps(_json_safe(fields), default=_json_default, allow_nan=False)
        self._metrics.write(line + "\n")
        self._metrics.flush()

    def heartbeat(
        self,
        status: RunStatus,
        *,
        step: int = 0,
        steps: int,
        epoch: float = 0.0,
        positions_per_second: float | None = None,
        eta_seconds: float | None = None,
        gpu: GpuStats | None = None,
        message: str | None = None,
    ) -> None:
        """Replace ``status.json`` with where the run is now."""
        beat = Heartbeat(
            status=status,
            pid=os.getpid(),
            started=self._started,
            updated=_now(),
            step=step,
            steps=steps,
            epoch=epoch,
            positions_per_second=positions_per_second,
            eta_seconds=eta_seconds,
            gpu=gpu,
            message=message,
        )
        _replace_file(self.directory / STATUS_FILE, _as_json(beat))

    def save_checkpoint(
        self,
        step: int,
        write: Callable[[Path], None],
        *,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save a checkpoint for ``step`` and apply the retention policy.

        ``write`` is handed a path to write the checkpoint to and is not told where it will end
        up: the file is renamed into place once it is whole, so a reader never sees a half-saved
        checkpoint, and this module stays the only one that knows the layout — and the only one
        that does not need to know how a checkpoint is serialised.
        """
        directory = self.directory / CHECKPOINTS_DIR
        final = directory / checkpoint_file(step)
        temporary = final.with_name(final.name + TEMPORARY_SUFFIX)
        try:
            write(temporary)
            with temporary.open("rb") as f:
                sync_file(f)
            os.replace(temporary, final)
            sync_directory(directory)
        except OSError as e:
            temporary.unlink(missing_ok=True)
            raise RunError(f"cannot save checkpoint {final}: {e.strerror or e}") from e
        self._reindex(step, metrics or {})
        return final

    def _reindex(self, step: int, metrics: dict[str, float]) -> None:
        """Record the new checkpoint, work out the best, and delete what is not kept."""
        directory = self.directory / CHECKPOINTS_DIR
        kept = {info.step: info for info in read_checkpoints(directory)}
        kept[step] = CheckpointInfo(
            step=step, file=checkpoint_file(step), created=_now(), metrics=metrics
        )
        best = _best_step(kept.values(), self.policy)
        # The newest `keep`, plus the best wherever it is: a run's best checkpoint is the one
        # worth playing against, and retention that could delete it would make the metric
        # pointless. Kept as a name rather than a second copy of the file.
        surviving = set(sorted(kept, reverse=True)[: self.policy.keep])
        if best is not None:
            surviving.add(best)
        for gone in set(kept) - surviving:
            (directory / kept.pop(gone).file).unlink(missing_ok=True)
        index = CheckpointIndex(
            policy=self.policy,
            best_step=best,
            checkpoints=[kept[step] for step in sorted(kept)],
        )
        _replace_file(directory / CHECKPOINT_INDEX, _as_json(index))
        sync_directory(directory)

    def finish(
        self,
        status: RunStatus,
        *,
        step: int,
        steps: int,
        epoch: float = 0.0,
        positions_per_second: float | None = None,
        message: str | None = None,
    ) -> None:
        """Write the last heartbeat and let go of the metrics log.

        ``positions_per_second`` on a finished run is the whole run's average rather than the
        last interval's, which is the number worth keeping: it is what the next experiment's
        schedule gets sized against.
        """
        self.heartbeat(
            status,
            step=step,
            steps=steps,
            epoch=epoch,
            positions_per_second=positions_per_second,
            message=message,
        )
        self.close()

    def close(self) -> None:
        if not self._metrics.closed:
            self._metrics.flush()
            sync_file(self._metrics)
            self._metrics.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class RunReader:
    """Everything a reader wants from a run directory, including a run being written right now.

    Nothing here holds a file open or caches what can change. The web server reads a run on
    every poll and a trainer may be part-way through writing any of it; re-reading is cheap, and
    a value cached across a write would be a value that is quietly wrong.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def info(self) -> RunInfo:
        """``run.json``, or raise :exc:`RunError` if it is not there or not readable."""
        return _read_model(self.directory / RUN_FILE, RunInfo)

    @property
    def config_text(self) -> str:
        """The experiment config as it was written, comments and all."""
        try:
            return (self.directory / CONFIG_FILE).read_text(encoding="utf-8")
        except OSError as e:
            raise RunError(f"cannot read {self.directory / CONFIG_FILE}: {e.strerror}") from e

    @property
    def status(self) -> Heartbeat | None:
        """The last heartbeat, or ``None`` before the run has written one."""
        if not (self.directory / STATUS_FILE).exists():
            return None
        return _read_model(self.directory / STATUS_FILE, Heartbeat)

    @property
    def notes(self) -> RunNotes:
        """The run's title, tags and notes; empty ones before anybody has written any."""
        path = self.directory / NOTES_FILE
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Asked rather than looked for first, since an overwrite can delete the file
            # between the looking and the reading.
            return RunNotes()
        except OSError as e:
            raise RunError(f"cannot read {path}: {e.strerror or e}") from e
        written = _parse_model(path, text, _NotesFile)
        return RunNotes(title=written.title, tags=written.tags, notes=written.notes)

    def metrics(self, *, skip: int = 0) -> list[dict[str, Any]]:
        """The metrics logged so far, oldest first, skipping the first ``skip`` of them.

        ``skip`` is how a live reader asks for what it has not seen: it says how many lines it
        already has, and gets the rest. Lines that do not parse are skipped, which in practice
        means the last line of a log being appended to at this moment.
        """
        return list(self.stream_metrics(skip=skip))

    def stream_metrics(self, *, skip: int = 0) -> Iterator[dict[str, Any]]:
        """:meth:`metrics`, without holding all of it in memory at once."""
        path = self.directory / METRICS_FILE
        try:
            with path.open("r", encoding="utf-8") as f:
                for number, line in enumerate(f):
                    if number < skip:
                        continue
                    if not line.endswith("\n"):
                        return  # Being written as we read; it will be whole next time.
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except FileNotFoundError:
            return
        except OSError as e:
            raise RunError(f"cannot read {path}: {e.strerror}") from e

    def tail_metrics(self) -> "MetricsTail":
        """A cursor over the metrics log, for a reader that follows it as it grows."""
        return MetricsTail(self.directory / METRICS_FILE)

    def track_latest_metrics(self) -> "LatestMetrics":
        """The last line of each split of the metrics log, for a reader that asks repeatedly."""
        return LatestMetrics(self.tail_metrics())

    def checkpoints(self) -> list[CheckpointInfo]:
        """The checkpoints on disk, oldest first, with whatever the index knows about them."""
        return read_checkpoints(self.directory / CHECKPOINTS_DIR)

    def checkpoint_index(self) -> CheckpointIndex:
        """The checkpoint index, reconciled with the files that are actually there."""
        directory = self.directory / CHECKPOINTS_DIR
        checkpoints = read_checkpoints(directory)
        index = _read_index(directory)
        best = index.best_step
        if best is not None and all(info.step != best for info in checkpoints):
            best = _best_step(checkpoints, index.policy)
        return index.model_copy(update={"checkpoints": checkpoints, "best_step": best})

    def best_checkpoint(self) -> CheckpointInfo | None:
        """The best checkpoint by the run's own metric, or ``None`` if there is none yet."""
        index = self.checkpoint_index()
        return next((info for info in index.checkpoints if info.step == index.best_step), None)

    def latest_checkpoint(self) -> CheckpointInfo | None:
        """The checkpoint from the highest step, which is where a resume starts from."""
        checkpoints = self.checkpoints()
        return checkpoints[-1] if checkpoints else None

    def checkpoint_path(self, info: CheckpointInfo) -> Path:
        """Where ``info``'s file is, which is what a checkpoint is loaded from."""
        return self.directory / CHECKPOINTS_DIR / info.file


class MetricsTail:
    """Where a reader following a metrics log has got to, so that each read returns what is new.

    It keeps a byte offset rather than a line count, so that a read costs what was added and
    not the whole log again. It also notices when the log it was following is no longer the
    one on the disk — a run started again under the same name with ``--overwrite`` deletes the
    log and begins a new one — and says so, because lines appended to what a reader already has
    would join two runs' metrics into one history that neither of them had.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._identity: tuple[int, int] | None = None
        self._started = False

    def read(self) -> tuple[list[dict[str, Any]], bool]:
        """The whole lines added since the last read, and whether they replace what came before.

        The second value is true on the first read, and whenever the log has been replaced or
        cut short since: the records returned are then the log from its beginning, and
        everything read before is to be thrown away. A line still being written is left for
        the next read, which sees it whole.
        """
        replace = not self._started
        self._started = True
        try:
            with self.path.open("rb") as f:
                stat = os.fstat(f.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self._identity or stat.st_size < self._offset:
                    replace = replace or self._identity is not None or self._offset > 0
                    self._identity, self._offset = identity, 0
                f.seek(self._offset)
                added = f.read()
        except FileNotFoundError:
            # Between an overwrite deleting the log and making the new one, or a run whose
            # directory has gone. Either way nothing read before still stands.
            replace = replace or self._offset > 0
            self._identity, self._offset = None, 0
            return [], replace
        except OSError as e:
            raise RunError(f"cannot read {self.path}: {e.strerror}") from e
        whole = added[: added.rfind(b"\n") + 1]
        self._offset += len(whole)
        records = [record for line in whole.splitlines() if (record := _parse_metrics_line(line))]
        return records, replace


def _parse_metrics_line(line: bytes) -> dict[str, Any] | None:
    """One metrics line as a record, or ``None`` for one that is not a JSON object.

    ``NaN`` and ``Infinity`` become ``None``, as :meth:`RunWriter.log` writes them now: a log
    written before it did may still have the bare tokens, which Python reads and a browser's
    ``JSON.parse`` does not.
    """
    try:
        record = json.loads(line, parse_constant=lambda _: None)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


class LatestMetrics:
    """The last line the metrics log has for each split, kept up to date as the log grows.

    The run list shows every run's latest numbers and is asked for every few seconds, while a
    long run's log runs to tens of thousands of lines. A reader that kept one of these per run
    reads each log once, and after that only what has been appended to it — which a search for
    the last line of each split cannot promise, since a split the log has no line for (a run
    with no validation positions, or one that has not validated yet) is only known to be absent
    once the whole log has been read.

    Safe to share between threads, as the web server's worker threads do.
    """

    def __init__(self, tail: MetricsTail, splits: Collection[str] = (TRAIN, VALIDATION)) -> None:
        self._tail = tail
        self._splits = frozenset(splits)
        self._latest: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def read(self) -> dict[str, dict[str, Any]]:
        """The last whole line of each split, keyed by split; a split with no line is left out."""
        with self._lock:
            records, replace = self._tail.read()
            if replace:
                self._latest = {}
            for record in records:
                if (split := record.get("split")) in self._splits:
                    self._latest[split] = record
            return dict(self._latest)


def open_run(runs_dir: Path, name: str) -> RunReader:
    """The run called ``name`` in ``runs_dir``, ready to read.

    The run's own ``run.json`` is not read to decide that it is there. A checkpoint carries
    everything it takes to load it, so a run whose other files are half-written, or were
    written by a version of this code that is no longer here, is still a run worth opening.

    Raises:
        RunError: ``name`` could not be a run's, or no run of that name is kept here.
    """
    run = RunReader(run_path(runs_dir, name))
    if not run.directory.is_dir():
        raise RunError(f"no run called {name!r} is kept here")
    return run


def save_notes(run: RunReader, notes: RunNotes) -> RunNotes:
    """Replace ``run``'s title, tags and notes with ``notes``, and return them as kept.

    Replaced whole, as every current value here is, so that the web server reading them on
    its next poll gets the old notes or the new ones and never half of each. Nothing here
    merges: whoever writes last wins, which for notes edited by one person is what they meant.

    Raises:
        RunError: the notes could not be written.
    """
    path = run.directory / NOTES_FILE
    written = _NotesFile(title=notes.title, tags=notes.tags, notes=notes.notes)
    try:
        _replace_file(path, _as_json(written), writers="many")
    except OSError as e:
        raise RunError(f"cannot write {path}: {e.strerror or e}") from e
    return notes


def choose_checkpoint(run: RunReader, choice: CheckpointChoice) -> CheckpointInfo:
    """The checkpoint of ``run`` that ``choice`` names.

    One definition of "best" and "latest" for everything that plays a checkpoint, so that a
    game started in the browser and a script run against the same run mean the same file.

    Raises:
        RunError: the run has saved no checkpoint yet, or none from the step asked for.
    """
    if choice == "best":
        # A run that has not validated yet has no best checkpoint, and the newest one is the
        # only answer there is to "the one worth playing".
        chosen = run.best_checkpoint() or run.latest_checkpoint()
    elif choice == "latest":
        chosen = run.latest_checkpoint()
    else:
        chosen = next((info for info in run.checkpoints() if info.step == choice), None)
        if chosen is None:
            raise RunError(f"run {run.name!r} has no checkpoint from step {choice}")
    if chosen is None:
        raise RunError(f"run {run.name!r} has not saved a checkpoint yet")
    return chosen


def checkpoint_file(step: int) -> str:
    """What the checkpoint for ``step`` is called: zero-padded, so the names sort by step."""
    return f"step-{step:0{CHECKPOINT_STEP_DIGITS}d}{CHECKPOINT_SUFFIX}"


def checkpoint_step(file: str) -> int | None:
    """The step a checkpoint file name is for, or ``None`` if it is not one of ours."""
    match = re.fullmatch(rf"step-(\d+){re.escape(CHECKPOINT_SUFFIX)}", file)
    return int(match.group(1)) if match else None


def read_checkpoints(directory: Path) -> list[CheckpointInfo]:
    """The checkpoints in ``directory``, oldest first, as the files and the index agree.

    The files decide which checkpoints exist; the index adds when each was written and what the
    validation metrics were. A file with no index entry still counts — it is a checkpoint — and
    an entry with no file does not.
    """
    recorded = {info.step: info for info in _read_index(directory).checkpoints}
    found: list[CheckpointInfo] = []
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return []
    for entry in entries:
        step = checkpoint_step(entry.name)
        if step is None or not entry.is_file():
            continue
        info = recorded.get(step)
        if info is None or info.file != entry.name:
            # Only a checkpoint the index does not describe is asked when it was written,
            # and that question is the one thing in here that can fail on a directory
            # somebody else is touching: the file can go, or refuse to be read, between the
            # scan and the stat. Not through this module's own writing — it unlinks what it
            # prunes before it rewrites the index, so a pruned file is one this reader still
            # has an entry for and never stats — but through anything outside it: a
            # checkpoint deleted by hand, or one this process may not read. A checkpoint that
            # cannot be looked at is not one to offer, which is a plainer answer than a
            # reader raising in the middle of a listing.
            try:
                written = datetime.fromtimestamp(entry.stat().st_mtime, UTC)
            except OSError:
                continue
            info = CheckpointInfo(step=step, file=entry.name, created=written)
        found.append(info)
    return sorted(found, key=lambda info: info.step)


def _read_index(directory: Path) -> CheckpointIndex:
    """The checkpoint index, or an empty one when it is missing or unreadable.

    An index that cannot be read costs the metrics and which checkpoint was best, and must not
    cost the checkpoints themselves: they are on the disk either way, and a run whose index was
    lost is still a run somebody can load a checkpoint from.
    """
    path = directory / CHECKPOINT_INDEX
    try:
        return _read_model(path, CheckpointIndex)
    except RunError:
        return CheckpointIndex()


def _best_step(checkpoints, policy: CheckpointPolicy) -> int | None:
    """Which of ``checkpoints`` is best by ``policy``, or ``None`` if none can be judged.

    A checkpoint whose metric is missing, ``None`` or not finite cannot be compared, so it is
    passed over rather than ordered against real numbers. That is the honest answer about a
    checkpoint nobody could measure, and it keeps a diverged step from winning on a ``nan``.
    """
    scored = [
        (value, info.step)
        for info in checkpoints
        if (value := info.metrics.get(policy.metric)) is not None and isfinite(value)
    ]
    if not scored:
        return None
    # On a tie the later step wins: it is the one trained longer.
    if policy.higher_is_better:
        return max(scored, key=lambda scoring: (scoring[0], scoring[1]))[1]
    return min(scored, key=lambda scoring: (scoring[0], -scoring[1]))[1]


def _read_model(path: Path, model: type[BaseModel]) -> Any:
    """Load and validate one of the JSON files here, or raise :exc:`RunError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RunError(f"cannot read {path}: {e.strerror or e}") from e
    return _parse_model(path, text, model)


def _parse_model(path: Path, text: str, model: type[BaseModel]) -> Any:
    """Validate ``text``, read from ``path``, or raise :exc:`RunError`."""
    try:
        return model.model_validate_json(text)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in e.errors()
        )
        raise RunError(f"{path} does not say what it should: {problems}") from e
    except ValueError as e:
        raise RunError(f"{path} is not valid JSON: {e}") from e


def _replace_file(path: Path, text: str, *, writers: Literal["one", "many"] = "one") -> None:
    """Write ``text`` to ``path`` by renaming it into place; see the module docstring.

    A file with one writer — everything the trainer writes — has one temporary name, so a
    temporary file left by a writer killed mid-write is taken over by the next write rather
    than left lying there. A file with ``"many"`` writers — the notes, which the web server's
    threads and the command line can save at the same moment — gives each write a temporary
    file of its own: two writers sharing one would truncate and write it at once, and rename
    whatever mixture of the two it held into place. With one each, the last rename wins whole.
    What a killed writer of those leaves is thrown away when the run is overwritten.
    """
    unique = f".{uuid.uuid4().hex}" if writers == "many" else ""
    temporary = path.with_name(f"{path.name}{unique}{TEMPORARY_SUFFIX}")
    try:
        # Written through open() rather than tempfile.mkstemp, whose files are readable by their
        # owner only: the web server reading a run need not be the user that trained it.
        with temporary.open("w" if writers == "one" else "x", encoding="utf-8") as f:
            f.write(text)
            sync_file(f)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def _as_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2) + "\n"


def _json_safe(value: Any) -> Any:
    """``value`` with every number JSON can hold, and ``None`` for the ones it cannot.

    A diverged run is exactly when this log is being read, and a loss of ``nan`` is what a
    learning rate one notch too high produces. :func:`json.dumps` would write the bare tokens
    ``NaN`` and ``Infinity``, which Python's own parser accepts and no strict one does — a
    browser's ``JSON.parse`` throws, losing the whole line rather than the one field. ``null``
    is the honest value for a loss that is not a number.
    """
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    """Make the few non-JSON things a metrics line may carry writable."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot write {type(value).__name__} to the metrics log")


def _now() -> datetime:
    return datetime.now(UTC)
