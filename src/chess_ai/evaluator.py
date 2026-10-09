"""The evaluator: what each checkpoint of each run can do, found out as the checkpoints appear.

A process of its own, long-lived like the web server, which neither it nor any trainer depends
on. It looks through the runs directory for checkpoints that lack a result from a suite their
run's experiment config names, runs those suites, and writes the results into the run
directory, where the web server finds them and tells whoever is watching the run. Nothing tells
it a checkpoint has been saved: it looks again every so often, and when it finds nothing to do.

Each suite says for itself whether a checkpoint's result still stands, so that a result
measured with a file that has since been replaced, or on a set of probes that has since
changed, is measured again; and a checkpoint is evaluated once per suite, however often it is
looked at. Results outlive their checkpoints: one pruned by its run's retention keeps them,
and is not looked at again.

The newest checkpoints come first, so that a run being trained hears about its latest one
before the evaluator gets on with older ones it has not done yet.
"""

import fcntl
import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import ValidationError

from chess_ai.players import ModelDescription
from chess_ai.probes import (
    ProbeResult,
    ProbeSet,
    probe,
    read_probes,
    save_probes,
    stands,
    summary,
)
from chess_ai.training.experiment import EvaluationSection
from chess_ai.training.run_store import (
    ARCHIVED_TAG,
    CheckpointChanged,
    CheckpointInfo,
    RunError,
    RunReader,
    code_version,
    list_runs,
)

if TYPE_CHECKING:
    from chess_ai.inference import InferenceEngine

LOGGER = logging.getLogger(__name__)

LOCK_FILE: Final = "evaluator.lock"
"""Locked in the runs directory by the evaluator working on it, for as long as it does."""


class Stopped(Exception):
    """:meth:`Evaluator.run_once` was told to stop before it had evaluated all it found."""

    def __init__(self, done: list["Job"], left: int) -> None:
        super().__init__(f"stopped with {left} checkpoints left")
        self.done = done
        """What it evaluated before it stopped, in order."""
        self.left = left
        """How many of the checkpoints it found still lacked a result when it stopped."""


class EvaluatorBusy(Exception):
    """Another evaluator is already working on the runs directory."""


@contextmanager
def evaluator_lock(runs_dir: Path) -> Iterator[None]:
    """Be the one evaluator working on ``runs_dir``, or raise :exc:`EvaluatorBusy`.

    Two would evaluate every checkpoint twice, each replacing the other's results. The lock goes
    with the process that holds it, however that process ends.

    Raises:
        EvaluatorBusy: another process holds it.
        OSError: the lock file cannot be made.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(runs_dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise EvaluatorBusy(f"another evaluator is already working on {runs_dir}") from e
        yield
    finally:
        os.close(descriptor)


class Suite(Protocol):
    """One kind of evaluation, as the evaluator runs it."""

    name: str

    def stands(self, run: RunReader, checkpoint: CheckpointInfo, written: datetime) -> bool:
        """Whether ``checkpoint``, whose file was written at ``written``, has a result from this
        suite that is still what it would find.

        A result that cannot be read does not stand, so that evaluating again replaces it.
        """
        ...

    def evaluate(
        self,
        run: RunReader,
        checkpoint: CheckpointInfo,
        written: datetime,
        engine: "InferenceEngine",
    ) -> str:
        """Evaluate ``checkpoint``, save the result into ``run``, and say in a line how it did.

        ``written`` is when the file ``engine`` was loaded from had been written, looked at
        before it was loaded: a file replaced in between is then evaluated again, rather than
        its result taken for the new one's.

        Raises:
            CheckpointChanged: the checkpoint's file was replaced or deleted after ``written``,
                so the result, which is about the file that was, was not saved.
            RunError: the result cannot be saved.
        """
        ...


class ProbeSuite:
    """The probe positions, as a suite: see :mod:`chess_ai.probes`."""

    name = "probe-positions"

    def __init__(self, probes: ProbeSet, *, rating: int | None) -> None:
        self.probes = probes
        self.rating = rating

    def stands(self, run: RunReader, checkpoint: CheckpointInfo, written: datetime) -> bool:
        try:
            result = read_probes(run, checkpoint.step)
        except RunError:
            return False
        return result is not None and stands(result, checkpoint.step, written, self.probes)

    def evaluate(
        self,
        run: RunReader,
        checkpoint: CheckpointInfo,
        written: datetime,
        engine: "InferenceEngine",
    ) -> str:
        started = datetime.now(UTC)
        positions = probe(engine, self.probes, rating=self.rating)
        result = ProbeResult(
            model=ModelDescription(run=run.name, checkpoint=checkpoint.step, rating=self.rating),
            checkpoint_written=written,
            started=started,
            finished=datetime.now(UTC),
            code_version=code_version(),
            probe_set=self.probes.name,
            probe_set_version=self.probes.version,
            positions=positions,
        )
        save_probes(run, result, checkpoint)
        return summary(result)


@dataclass(frozen=True)
class Job:
    """A checkpoint, and the suites it lacks a result from."""

    run: RunReader
    checkpoint: CheckpointInfo
    written: datetime
    """When the checkpoint's file was written, as it was when the job was found."""
    suites: tuple[str, ...]

    def describe(self) -> str:
        return f"{self.run.name}@{self.checkpoint.step}"


def _key(job: Job) -> tuple[str, int, datetime]:
    """Which checkpoint file ``job`` is about, whatever suites it lacks."""
    return job.run.name, job.checkpoint.step, job.written


Loader = Callable[[Path], "InferenceEngine"]
"""Loads a checkpoint file to evaluate, on whatever device and in whatever batches it is set to."""


def configured_suites(run: RunReader) -> list[str]:
    """The suites ``run``'s experiment config names, or the default for a config that names none.

    Read from what the run recorded of its config, which a run from before there were suites
    recorded without any, and a newer version of this code may have recorded with suites this
    one does not know; those are left out.

    Raises:
        RunError: the run does not say what it is.
    """
    written = run.info.config.get("evaluation")
    if written is None:
        return list(EvaluationSection().suites)
    suites = written.get("suites", []) if isinstance(written, dict) else []
    known = []
    for suite in suites if isinstance(suites, list) else []:
        try:
            EvaluationSection(suites=[suite])
        except ValidationError:
            LOGGER.debug("run %s asks for suite %r, which this code does not know", run.name, suite)
            continue
        known.append(suite)
    return known


class Evaluator:
    """Finds checkpoints that lack results, and evaluates them.

    ``suites`` are the suites it can run, by name. A run asking for one that is not among them
    gets the rest. A checkpoint a suite fails on is not tried again until the evaluator is
    started again, so that a broken file is reported once rather than every few seconds.
    """

    def __init__(
        self,
        runs_dir: Path,
        suites: Mapping[str, Suite],
        *,
        load: Loader,
        say: Callable[[str], None] = LOGGER.info,
        warn: Callable[[str], None] = LOGGER.warning,
    ) -> None:
        self.runs_dir = runs_dir
        self.suites = dict(suites)
        self._load = load
        self._say = say
        self._warn = warn
        self._failed: set[tuple[str, int, datetime, str]] = set()
        """The checkpoints a suite failed on, by run, step, when the file was written, and suite."""

    def pending(self) -> list[Job]:
        """Every checkpoint on disk that lacks a result a suite of its run's would give, newest
        checkpoint first. Archived runs are left alone."""
        jobs: list[Job] = []
        for name in list_runs(self.runs_dir):
            run = RunReader(self.runs_dir / name)
            try:
                if ARCHIVED_TAG in run.notes.tags:
                    continue
                wanted = [suite for suite in configured_suites(run) if suite in self.suites]
            except RunError:
                LOGGER.debug("run %s cannot be read, so it is not evaluated", name, exc_info=True)
                continue
            if not wanted:
                continue
            for checkpoint in run.checkpoints():
                try:
                    written = run.checkpoint_written(checkpoint)
                except RunError:
                    LOGGER.debug("%s@%d cannot be looked at", name, checkpoint.step, exc_info=True)
                    continue
                if written is None:
                    continue  # Pruned since the listing.
                missing = tuple(
                    suite
                    for suite in wanted
                    if (name, checkpoint.step, written, suite) not in self._failed
                    and not self.suites[suite].stands(run, checkpoint, written)
                )
                if missing:
                    jobs.append(
                        Job(run=run, checkpoint=checkpoint, written=written, suites=missing)
                    )
        return sorted(jobs, key=lambda job: job.written, reverse=True)

    def evaluate(self, job: Job) -> None:
        """Run the suites ``job`` names on its checkpoint, loading the checkpoint once for all.

        A checkpoint pruned before its turn came is passed over without a word: it is no longer
        one to evaluate. Any other failure is said, and the checkpoint not tried again.
        """
        run, checkpoint = job.run, job.checkpoint
        path = run.checkpoint_path(checkpoint)
        try:
            engine = self._load(path)
        except Exception as e:
            if not path.exists():
                LOGGER.debug("%s was pruned before it could be evaluated", job.describe())
                return
            self._fail(job, job.suites, f"cannot be loaded: {e}")
            return
        for name in job.suites:
            try:
                said = self.suites[name].evaluate(run, checkpoint, job.written, engine)
            except CheckpointChanged:
                # Replaced or pruned meanwhile: the next look finds the new file, if any.
                LOGGER.debug("%s changed while it was evaluated", job.describe())
                return
            except Exception as e:
                self._fail(job, (name,), f"{name} failed: {e}")
                continue
            if not self.suites[name].stands(run, checkpoint, job.written):
                # Saved, and yet still lacking: evaluating it again would save the same, and
                # since it is the newest it would be next every time, ahead of everything else.
                self._fail(job, (name,), f"{name} saved a result that does not read back")
                continue
            self._say(f"{job.describe()} {name}: {said}")

    def run_once(self, *, stop: threading.Event | None = None) -> list[Job]:
        """Evaluate what lacks a result now, and say what that was, in the order it was done.

        Only the checkpoints the first look finds: one saved meanwhile is left for the next
        time, so that this ends however fast the runs save them. Each turn looks again all the
        same, so that one evaluated, pruned or replaced meanwhile drops out. ``stop`` is looked
        at between checkpoints.

        Raises:
            Stopped: ``stop`` was set while some of them still lacked a result; one set after
                the last of them was done is a pass that finished.
        """
        found = {_key(job) for job in self.pending()}
        done: list[Job] = []
        while found:
            left = [job for job in self.pending() if _key(job) in found]
            if not left:
                break
            if stop is not None and stop.is_set():
                raise Stopped(done, len(left))
            job = left[0]
            found.discard(_key(job))
            self.evaluate(job)
            done.append(job)
        return done

    def run(self, *, poll_seconds: float, stop: threading.Event) -> None:
        """Evaluate checkpoints as they appear, until ``stop`` is set.

        ``stop`` is looked at between checkpoints, and while waiting for more.
        """
        while not stop.is_set():
            job = self._next()
            if job is None:
                stop.wait(poll_seconds)
            else:
                self.evaluate(job)

    def _next(self) -> Job | None:
        """The newest checkpoint lacking a result, looked for afresh every time.

        Afresh rather than from the list one look found, so that a checkpoint saved while a
        backlog is worked through — every checkpoint of every run, when the evaluator first
        starts or the probe set changes — goes ahead of the rest of it instead of waiting, and
        perhaps being pruned by its run before its turn. A look is a few small files a run;
        loading a checkpoint is far more.
        """
        jobs = self.pending()
        return jobs[0] if jobs else None

    def _fail(self, job: Job, suites: Sequence[str], why: str) -> None:
        for suite in suites:
            self._failed.add((job.run.name, job.checkpoint.step, job.written, suite))
        self._warn(
            f"{job.describe()} {why}; it is not tried again until the evaluator is restarted"
        )
