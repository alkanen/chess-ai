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
    <runs>/<name>/evaluations/      step-<step>/, one per checkpoint evaluated, holding what each
                                    evaluation suite wrote about it: <suite>.json and its games
    <runs>/<name>/writer.lock       locked by the one process writing the run, while it does

Two write disciplines, because two kinds of reader:

- **Appending** is for the metrics log, which is a history: a reader tails it, and the only
  thing it can catch half-written is the last line, which it skips and sees whole next time.
- **Replacing** is for everything that is a current value rather than a history. Written to a
  temporary name in the same directory and renamed over the old one, so a reader opening it at
  any moment gets one whole version or the other, never a mixture. A crash mid-write leaves the
  previous version intact.

No file here is ever rewritten in place, which is what makes concurrent reading safe without a
lock between processes that do not otherwise know about each other. Writing is another matter:
two trainers in one run directory would interleave two runs' metrics and prune each other's
checkpoints, so the process writing a run holds a lock on it for as long as it does — and a
process that dies, however it dies, lets go of it with the rest of its files.
"""

import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Self
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chess_ai.dataset.files import sync_directory, sync_file
from chess_ai.encoders import EncoderSpec

LOGGER = logging.getLogger(__name__)

FORMAT_VERSION: Final = 1
"""The version of this layout, recorded in every file it writes and checked when reading."""

RUN_FILE: Final = "run.json"
CONFIG_FILE: Final = "config.toml"
METRICS_FILE: Final = "metrics.jsonl"
STATUS_FILE: Final = "status.json"
NOTES_FILE: Final = "notes.json"
LOCK_FILE: Final = "writer.lock"
CHECKPOINTS_DIR: Final = "checkpoints"
EVALUATIONS_DIR: Final = "evaluations"
EVALUATION_LOCK: Final = "evaluation.lock"
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


class CheckpointChanged(RunError):
    """An evaluation was not saved, because its checkpoint was replaced or deleted meanwhile."""


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
    version: int = 1
    """The version of the dataset the run trained on; runs from before datasets had versions
    trained on the only one there was."""
    directory: str
    format_version: int
    created: datetime | None = None
    games: int
    positions: int
    train_positions: int
    validation_positions: int
    train_targets: int | None = None
    """How many train positions are trained on, when the dataset filtered positions; otherwise
    every one of them is."""


class ModelReference(BaseModel):
    """Which architecture a run trained, with the hyperparameters it was built with."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str
    options: dict[str, Any] = Field(default_factory=dict)
    parameter_count: int


class Lineage(BaseModel):
    """The run and checkpoint a run's weights started from, rather than from the seed.

    This is how a fine-tuning run says what it was fine-tuned from, and following ``run`` back
    through each run's own lineage gives the whole curriculum a model went through.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: str
    step: int
    """The step of the checkpoint the weights came from."""
    checkpoint: CheckpointChoice
    """The checkpoint as the config asked for it: ``best``, ``latest`` or a step."""
    dataset: str | None = None
    """The dataset the run they came from was trained on."""


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
    initialized_from: Lineage | None = None
    """Where the weights started, for a run that did not start from its seed."""


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


_HERE: Final = Path(__file__).resolve()
"""This file, which a checkout has to track for its commit to be this code's."""

_GIT_TIMEOUT: Final = 10
"""Seconds a git command gets before the code version gives up on it."""


def code_version() -> str:
    """What the code was when a run started: the git commit, and whether it was edited.

    Falls back to the installed version when git cannot say (not a checkout, no ``git`` on the
    ``PATH``, or this code is not tracked by the repository it sits in, such as a copy unpacked
    inside another checkout), because a run from an installed package or a copy of the code is
    still a run and its results still have to say what produced them. The fallback says why git
    could not answer, and marks an editable install, whose version stays the same whatever is
    checked out. An install that is not this code gives no version at all: a run whose code
    version is unknown is still worth keeping.
    """
    try:
        revision = _git("rev-parse", "--short", "HEAD")
    except (OSError, subprocess.SubprocessError) as error:
        return f"{_installed_version()} ({_git_failure(error)})"
    try:
        _git("ls-files", "--error-unmatch", _HERE.name)
    except subprocess.CalledProcessError:
        return f"{_installed_version()} (not tracked by the git repository around it)"
    except (OSError, subprocess.SubprocessError) as error:
        return f"{_installed_version()} ({_git_failure(error)})"
    try:
        dirty = _git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError) as error:
        return f"git {revision}, edits unknown ({_git_failure(error)})"
    return f"git {revision}{'-dirty' if dirty else ''}"


def _git(*arguments: str) -> str:
    """What a git command prints, run where this code is."""
    return subprocess.run(
        ["git", *arguments],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=True,
    ).stdout.strip()


def _git_failure(error: OSError | subprocess.SubprocessError) -> str:
    """Why git could not answer, in a few words for the code version."""
    if isinstance(error, FileNotFoundError):
        return "git not found"
    if isinstance(error, subprocess.TimeoutExpired):
        return f"git {error.cmd[1]} timed out"
    if isinstance(error, subprocess.CalledProcessError):
        # Git can warn before it fails (an unreadable config under sudo, say), and hints after.
        lines = (error.stderr or "").strip().splitlines()
        fatal = [line.removeprefix("fatal: ") for line in lines if line.startswith("fatal: ")]
        message = fatal[-1] if fatal else lines[0] if lines else f"exit code {error.returncode}"
        return f"git {error.cmd[1]}: {message}"
    return f"git: {error}"


def _installed_version() -> str:
    """The installed package's version, if the install is this code."""
    try:
        distribution = metadata.distribution("chess-ai")
    except metadata.PackageNotFoundError:
        return "version unknown"
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    except ValueError:
        direct_url = {}
    if direct_url.get("dir_info", {}).get("editable", False):
        url = urlparse(direct_url.get("url", ""))
        if url.scheme != "file" or not _HERE.is_relative_to(Path(url2pathname(url.path)).resolve()):
            return "version unknown"
        return f"version {distribution.version}, editable install"
    installed = Path(distribution.locate_file("chess_ai/training/run_store.py")).resolve()
    if installed != _HERE:
        return "version unknown"
    return f"version {distribution.version}"


class RunWriter:
    """The trainer's end of a run directory: everything a run produces goes through here."""

    def __init__(self, directory: Path, policy: CheckpointPolicy, *, lock: "_WriterLock") -> None:
        self.directory = directory
        self.policy = policy
        self._lock = lock
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
        except OSError as e:
            raise RunError(f"cannot create run directory {directory}: {e.strerror or e}") from e
        lock = _WriterLock.acquire(directory, info.name)
        try:
            # Again, now that nobody else can be writing it: a run of this name may have been
            # started and finished between the first look and the lock.
            check_available(directory, info.name, overwrite=overwrite)
            if overwrite:
                cls._clear(directory)
            _replace_file(directory / CONFIG_FILE, config_text)
            _replace_file(directory / RUN_FILE, _as_json(info))
            (directory / METRICS_FILE).touch()
            return cls(directory, policy, lock=lock)
        except OSError as e:
            lock.release()
            raise RunError(f"cannot create run directory {directory}: {e.strerror or e}") from e
        except BaseException:
            lock.release()
            raise

    @classmethod
    def reopen(cls, directory: Path, *, policy: CheckpointPolicy) -> Self:
        """Take over writing the run in ``directory`` again, which is how a run resumes.

        Locked before anything in it is read, so that nothing read can change underneath:
        a trainer still writing the run would be saving checkpoints past the one a resume
        chooses. See :meth:`rewind` for putting the run back to that checkpoint.
        """
        if not (directory / RUN_FILE).is_file():
            raise RunError(f"no run in {directory}")
        lock = _WriterLock.acquire(directory, directory.name)
        try:
            return cls(directory, policy, lock=lock)
        except OSError as e:
            lock.release()
            raise RunError(f"cannot open the run in {directory}: {e.strerror or e}") from e
        except BaseException:
            lock.release()
            raise

    def rewind(self, step: int) -> None:
        """Put the run back to where it was at ``step``, to carry on from there.

        Whatever the run logged after ``step`` is dropped from the metrics log, since the run
        is about to train those steps again and a chart with both would show two histories
        laid over each other. That is the one time the log is replaced rather than appended to,
        and it is replaced whole, which a reader following it notices as it notices an
        overwrite. A checkpoint a crash left half-written is thrown away too.
        """
        try:
            self._metrics.close()
            _keep_metrics_until(self.directory / METRICS_FILE, step)
            self._metrics = (self.directory / METRICS_FILE).open("a", encoding="utf-8")
            for path in (self.directory / CHECKPOINTS_DIR).glob(f"*{TEMPORARY_SUFFIX}"):
                path.unlink(missing_ok=True)
        except OSError as e:
            raise RunError(f"cannot rewind the run in {self.directory}: {e.strerror or e}") from e

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
        # The evaluations were of the replaced run's checkpoints, and would be taken for this
        # one's: shown on its page, and its own checkpoints of the same steps never evaluated.
        shutil.rmtree(directory / EVALUATIONS_DIR, ignore_errors=True)
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
        try:
            if not self._metrics.closed:
                self._metrics.flush()
                sync_file(self._metrics)
                self._metrics.close()
        finally:
            self._lock.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class _WriterLock:
    """The lock that makes one process the only writer of a run directory.

    An ``flock`` on a file of its own, which the kernel lets go of when the process holding it
    ends — however it ends, SIGKILL and the OOM killer included — so a crashed run never leaves
    a lock behind that somebody has to know to delete. The file itself stays: deleting a lock
    file is a race between one process unlinking it and another locking what it just opened.

    A file system that cannot lock at all is let through with a warning, rather than every run
    on it refused for something it never did.

    A process forked while the lock is held — every data-loader worker is one — shares it, and
    a lock is only let go of once every process sharing it has closed it; a worker orphaned by a
    trainer killed outright would keep the run locked until it noticed. So a forked child closes
    its copy straight away, which leaves the lock with the process that took it.
    """

    held: ClassVar[set[int]] = set()
    """The descriptors this process holds locks through, for a forked child to close."""

    def __init__(self, descriptor: int | None) -> None:
        self._descriptor = descriptor
        if descriptor is not None:
            self.held.add(descriptor)

    @classmethod
    def acquire(cls, directory: Path, name: str) -> "_WriterLock":
        """Lock ``directory`` for this process, or raise :exc:`RunError` if another has it."""
        path = directory / LOCK_FILE
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            raise RunError(f"cannot open {path}: {e.strerror or e}") from e
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise RunError(
                f"run {name!r} is being written by another process; stop that one first"
            ) from None
        except OSError as e:
            os.close(descriptor)
            LOGGER.warning("cannot lock %s, so nothing stops a second writer: %s", path, e)
            return cls(None)
        return cls(descriptor)

    def release(self) -> None:
        if self._descriptor is not None:
            self.held.discard(self._descriptor)
            os.close(self._descriptor)
            self._descriptor = None

    @classmethod
    def _let_go_in_child(cls) -> None:
        for descriptor in cls.held:
            with suppress(OSError):
                os.close(descriptor)
        cls.held = set()


os.register_at_fork(after_in_child=_WriterLock._let_go_in_child)


def _keep_metrics_until(path: Path, step: int) -> None:
    """Drop the lines of the metrics log at ``path`` from after ``step``.

    A line still being written when the run died goes too, since the next line appended would
    otherwise be joined onto it. Left alone when there is nothing to drop, so that resuming a
    run that stopped cleanly does not make every reader start the log again.
    """
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        path.touch()
        return
    kept = [line for line in lines if line.endswith(b"\n") and _logged_at(line) <= step]
    if kept != lines:
        _replace_file(path, b"".join(kept).decode("utf-8", errors="replace"))


def _logged_at(line: bytes) -> int:
    """The step a metrics line was logged at; 0 for one that does not say."""
    step = (_parse_metrics_line(line) or {}).get("step")
    return step if isinstance(step, int) else 0


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

    def checkpoint_written(self, info: CheckpointInfo) -> datetime | None:
        """When ``info``'s file was last written, or ``None`` if it is gone.

        What tells two checkpoints of one step apart, as a run started again under its name or
        resumed from an earlier checkpoint saves: the file's own time rather than the index's,
        which is written a moment after the file and so cannot be read at the same moment.
        """
        try:
            return datetime.fromtimestamp(self.checkpoint_path(info).stat().st_mtime, UTC)
        except FileNotFoundError:
            return None
        except OSError as e:
            raise RunError(f"cannot look at {self.checkpoint_path(info)}: {e.strerror or e}") from e

    def evaluation_directory(self, step: int) -> Path:
        """Where the evaluations of the checkpoint from ``step`` are kept, made or not.

        Kept apart from the checkpoint itself, so that a checkpoint pruned by the run's retention
        policy leaves what was found out about it behind.
        """
        return self.directory / EVALUATIONS_DIR / f"step-{step:0{CHECKPOINT_STEP_DIGITS}d}"

    def evaluations(self) -> list["EvaluationEntry"]:
        """Every suite's result about every checkpoint, by step and then suite.

        Only the results themselves: a save in progress, the games beside a result and the
        lock are left out. A checkpoint pruned since it was evaluated still has its results.
        """
        found: list[EvaluationEntry] = []
        try:
            directories = list(os.scandir(self.directory / EVALUATIONS_DIR))
        except OSError:
            return []
        for directory in directories:
            match = _EVALUATION_DIRECTORY.fullmatch(directory.name)
            if match is None:
                continue
            try:
                files = list(os.scandir(directory.path))
            except OSError:
                # Gone between the two listings, or not a directory after all.
                continue
            for file in files:
                suite = file.name.removesuffix(_RESULT_SUFFIX)
                if suite == file.name or not _NAME.fullmatch(suite):
                    continue
                try:
                    updated = datetime.fromtimestamp(file.stat().st_mtime, UTC)
                except OSError:
                    continue
                found.append(EvaluationEntry(step=int(match[1]), suite=suite, updated=updated))
        return sorted(found, key=lambda entry: (entry.step, entry.suite))

    def evaluation(self, step: int, suite: str) -> str | None:
        """What ``suite`` found out about the checkpoint from ``step``, or ``None`` if nothing.

        The text of the result, as the suite wrote it: what it means is the suite's to say.
        """
        if not _NAME.fullmatch(suite):
            raise RunError(f"invalid suite name {suite!r}")
        path = self.evaluation_directory(step) / f"{suite}{_RESULT_SUFFIX}"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            raise RunError(f"cannot read {path}: {e.strerror or e}") from e


class EvaluationEntry(BaseModel):
    """That a suite has a result about a checkpoint, and when it was last written."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    suite: str
    updated: datetime


_EVALUATION_DIRECTORY: Final = re.compile(r"step-(\d+)")
_RESULT_SUFFIX: Final = ".json"


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


ARCHIVED_TAG: Final = "archived"
"""The tag that sets a run aside without deleting it.

An archived run is left out of every list of runs unless it is asked for, and never offered to
play against, but it is still there to open and compare: a sweep's runs are seldom worth playing
and their curves often worth looking at again. Archiving is adding the tag, so nothing else is
needed to do or undo it.
"""


def listed(tags: Collection[str], *, wanted: Collection[str], archived: bool) -> bool:
    """Whether a run with ``tags`` belongs in a list of runs with all the ``wanted`` tags.

    An archived run belongs in it only when ``archived`` asks for them, or when the archived
    tag is among those wanted, since asking for that tag can mean nothing else.
    """
    if not set(wanted) <= set(tags):
        return False
    return archived or ARCHIVED_TAG in wanted or ARCHIVED_TAG not in tags


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


@contextmanager
def evaluation_lock(directory: Path) -> Iterator[None]:
    """Hold the lock on the evaluation directory ``directory``, making it if need be.

    Whoever saves into an evaluation directory holds it for as long as the save takes, so that
    two saves that finish together cannot interleave their files: the evaluator and a one-off
    evaluation from the command line can both be finishing one suite about one checkpoint. The
    lock goes with the process that holds it, however that process ends.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory / EVALUATION_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        raise RunError(f"cannot lock {directory}: {e.strerror or e}") from e
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


PARTIAL_SUFFIX: Final = ".partial"
"""What the name of a file an evaluation is still writing ends in, such as its games."""


def start_partial_file(directory: Path, name: str) -> Path:
    """A new, empty file in the evaluation directory ``directory`` to write ``name`` into.

    One of its own for every evaluation that asks, so that two evaluations of one checkpoint
    going on at once do not write into each other's; :func:`save_evaluation` moves it to
    ``name`` once the evaluation is over. ``sample-games.pgn`` is started as, for instance,
    ``sample-games.1a2b3c4d.pgn.partial``.

    Raises:
        OSError: the directory or the file cannot be made.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem, dot, suffix = name.partition(".")
    path = directory / f"{stem}.{uuid.uuid4().hex[:8]}{dot}{suffix}{PARTIAL_SUFFIX}"
    path.touch(exist_ok=False)
    return path


def save_evaluation(
    directory: Path,
    texts: Mapping[str, str],
    moves: Mapping[str, Path] | None = None,
    *,
    current: Callable[[], bool] | None = None,
) -> None:
    """Put what a suite found out into the evaluation directory ``directory``, as one set.

    ``texts`` are written whole under their names, and the files in ``moves`` are moved in
    under theirs, all replacing what was there. The texts are written to temporary files
    before anything is moved, which is where a full disk or a missing permission stops a save,
    so that a save that fails changes nothing; and they are renamed into place last, so that a
    result, once it is there, has whatever was moved in with it beside it. The whole save
    holds :func:`evaluation_lock`, so that the last save to finish is the set that stays.

    ``current`` is asked once the lock is held, and a save it says no to is not made: it is
    whether the checkpoint the result is about is still the one on the disk. A run overwritten,
    or a checkpoint saved again, while it was being evaluated would otherwise be given the
    result about the one it replaced.

    Raises:
        CheckpointChanged: ``current`` said no. Nothing was saved or moved.
        RunError: the directory cannot be written. Whatever was to be moved in is where it
            was, unless the failure came after it had been moved.
    """
    with evaluation_lock(directory):
        if current is not None and not current():
            raise CheckpointChanged(f"the checkpoint {directory.name} is about has changed")
        staged: dict[Path, Path] = {}
        try:
            for name, text in texts.items():
                temporary = directory / f"{name}.{uuid.uuid4().hex}{TEMPORARY_SUFFIX}"
                staged[temporary] = directory / name
                with temporary.open("x", encoding="utf-8") as f:
                    f.write(text)
                    sync_file(f)
            for name, source in (moves or {}).items():
                os.replace(source, directory / name)
            for temporary, path in staged.items():
                os.replace(temporary, path)
            sync_directory(directory)
        except OSError as e:
            raise RunError(f"cannot write in {directory}: {e.strerror or e}") from e
        finally:
            for temporary in staged:
                temporary.unlink(missing_ok=True)


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
