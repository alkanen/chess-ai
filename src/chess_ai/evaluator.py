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

import asyncio
import fcntl
import logging
import os
import threading
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import ValidationError

from chess_ai.output import RESULT, Say
from chess_ai.output import say as said_on_stdout
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

STOP_POLL_SECONDS: Final = 0.1
"""How often a suite that plays games looks at whether the evaluator has been told to stop."""


class Stopped(Exception):
    """:meth:`Evaluator.run_once` was told to stop before it had evaluated all it found."""

    def __init__(self, done: list["Job"], left: int) -> None:
        super().__init__(f"stopped with {left} checkpoints left")
        self.done = done
        """What it evaluated before it stopped, in order."""
        self.left = left
        """How many of the checkpoints it found still lacked a result when it stopped."""


class SuiteStopped(Exception):
    """A suite gave up on a checkpoint because the evaluator was told to stop, saving nothing.

    Not a failure: the checkpoint still lacks the result, and the next evaluator evaluates it.
    """


async def until_stopped[T](
    playing: Coroutine[Any, Any, T], stop: threading.Event | None, saved: list[T]
) -> T:
    """What ``playing`` comes to, unless ``stop`` is set before it has saved it in ``saved``.

    For a suite that plays games, which take minutes: a stop that waited for them would be a
    stop a service manager gives up on and turns into a kill, which leaves the games' files
    behind. So the games are cancelled, in the middle of a move if need be, and their own
    cleaning up — the partial games file, the game shown as being played, Stockfish — is done
    on the way out. ``playing`` adds its result to ``saved`` once it has saved it.

    Raises:
        SuiteStopped: ``stop`` was set before the games were over; nothing was saved.
    """
    task = asyncio.create_task(playing)
    if stop is None:
        return await task
    while not task.done():
        # A threading event, set by a signal handler: there is nothing to await it with.
        await asyncio.wait({task}, timeout=STOP_POLL_SECONDS)
        if stop.is_set() and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            # Finished after all, the moment before it would have been cancelled.
            if not task.cancelled():
                return task.result()
            # Cancelled in its cleaning up, such as waiting for Stockfish to exit, after the
            # games were saved: they are done, and the next evaluator would find them so.
            if saved:
                return saved[0]
            raise SuiteStopped("stopped before the games were over")
    return task.result()


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

    def wants(self, run: RunReader, checkpoint: CheckpointInfo) -> bool:
        """Whether ``checkpoint`` is one this suite evaluates at all, such as every so many
        steps for a suite that takes too long to evaluate every checkpoint a run saves.

        One it does not want is not lacking its result, whatever :meth:`stands` would say.
        """
        ...

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
        *,
        stop: threading.Event | None = None,
    ) -> str:
        """Evaluate ``checkpoint``, save the result into ``run``, and say in a line how it did.

        ``written`` is when the file ``engine`` was loaded from had been written, looked at
        before it was loaded: a file replaced in between is then evaluated again, rather than
        its result taken for the new one's.

        ``stop`` is the evaluator's: a suite that takes long enough for the wait to matter
        looks at it as it goes, and raises :exc:`SuiteStopped` once it is set.

        Raises:
            SuiteStopped: ``stop`` was set before the suite was done; nothing was saved.
            CheckpointChanged: the checkpoint's file was replaced or deleted after ``written``,
                so the result, which is about the file that was, was not saved.
            RunError: the result cannot be saved.
        """
        ...

    def after(self, run: RunReader) -> None:
        """Called once a checkpoint of ``run``, a run that asks for this suite, has been through
        all the suites it was to be evaluated by, whether or not they succeeded and whether or
        not this one was among them; not when they were stopped part-way. For what must wait
        until no suite needs a checkpoint any more, such as pruning checkpoints that were only
        kept for this suite."""
        ...


class ProbeSuite:
    """The probe positions, as a suite: see :mod:`chess_ai.probes`."""

    name = "probe-positions"

    def __init__(self, probes: ProbeSet, *, rating: int | None) -> None:
        self.probes = probes
        self.rating = rating

    def wants(self, run: RunReader, checkpoint: CheckpointInfo) -> bool:
        return True

    def after(self, run: RunReader) -> None:
        pass

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
        *,
        stop: threading.Event | None = None,
    ) -> str:
        # A second or two: done before a stop would be waited for long.
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


def evaluated_runs(runs_dir: Path) -> Iterator[tuple[RunReader, list[str]]]:
    """Every run in ``runs_dir`` the evaluator looks at, with the suites it asks for.

    Archived runs are left alone, and so are runs that cannot be read.
    """
    for name in list_runs(runs_dir):
        run = RunReader(runs_dir / name)
        try:
            if ARCHIVED_TAG in run.notes.tags:
                continue
            suites = configured_suites(run)
        except RunError:
            LOGGER.debug("run %s cannot be read, so it is not evaluated", name, exc_info=True)
            continue
        yield run, suites


def runs_lacking(
    runs_dir: Path,
    suite: str,
    stands: Callable[[RunReader, CheckpointInfo, datetime], bool],
    wants: Callable[[RunReader, CheckpointInfo], bool] = lambda run, checkpoint: True,
) -> list[str]:
    """The runs the evaluator looks at that ask for ``suite`` and have a checkpoint on disk
    it ``wants`` without a result from it that ``stands``, by name: the runs a pass without the
    suite would leave short. ``stands`` and ``wants`` are the suite's own :meth:`Suite.stands`
    and :meth:`Suite.wants`, as one that cannot be set up still has to tell."""
    lacking = []
    for run, suites in evaluated_runs(runs_dir):
        if suite not in suites:
            continue
        for checkpoint in run.checkpoints():
            if not wants(run, checkpoint):
                continue
            try:
                written = run.checkpoint_written(checkpoint)
            except RunError:
                continue  # Not looked at by a pass either.
            if written is not None and not stands(run, checkpoint, written):
                lacking.append(run.name)
                break
    return lacking


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
        say: Say = said_on_stdout,
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
        for run, configured in evaluated_runs(self.runs_dir):
            name = run.name
            wanted = [suite for suite in configured if suite in self.suites]
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
                    and self.suites[suite].wants(run, checkpoint)
                    and not self.suites[suite].stands(run, checkpoint, written)
                )
                if missing:
                    jobs.append(
                        Job(run=run, checkpoint=checkpoint, written=written, suites=missing)
                    )
        return sorted(jobs, key=lambda job: job.written, reverse=True)

    def evaluate(self, job: Job, *, stop: threading.Event | None = None) -> bool:
        """Run the suites ``job`` names on its checkpoint, loading the checkpoint once for all,
        and say whether it got to the end of them.

        A checkpoint pruned before its turn came is passed over without a word: it is no longer
        one to evaluate. Any other failure is said, and the checkpoint not tried again. A suite
        stopped by ``stop`` part-way is neither: the rest are not run, and ``False`` is returned.
        """
        run, checkpoint = job.run, job.checkpoint
        path = run.checkpoint_path(checkpoint)
        try:
            engine = self._load(path)
        except Exception as e:
            if not path.exists():
                LOGGER.debug("%s was pruned before it could be evaluated", job.describe())
                return True
            self._fail(job, job.suites, f"cannot be loaded: {e}")
            return True
        for name in job.suites:
            try:
                said = self.suites[name].evaluate(run, checkpoint, job.written, engine, stop=stop)
            except SuiteStopped:
                LOGGER.debug("%s %s was stopped part-way", job.describe(), name)
                return False
            except CheckpointChanged:
                # Replaced or pruned meanwhile: the next look finds the new file, if any.
                LOGGER.debug("%s changed while it was evaluated", job.describe())
                break
            except Exception as e:
                self._fail(job, (name,), f"{name} failed: {e}")
                continue
            if not self.suites[name].stands(run, checkpoint, job.written):
                # Saved, and yet still lacking: evaluating it again would save the same, and
                # since it is the newest it would be next every time, ahead of everything else.
                self._fail(job, (name,), f"{name} saved a result that does not read back")
                continue
            self._say(f"{job.describe()} {name}: {said}", kind=RESULT)
        self._after(job)
        return True

    def _after(self, job: Job) -> None:
        """Tell the suites of ``job``'s run that a checkpoint of it is done with: every suite the
        run asks for, not only those of the job. A job stopped part-way says nothing, and the
        one that later finishes the checkpoint may name none of the suites that ran before the
        stop, such as a ladder that is already saved and whose checkpoint is to be let go of."""
        try:
            names = [name for name in configured_suites(job.run) if name in self.suites]
        except RunError:
            names = list(job.suites)
        for name in names:
            try:
                self.suites[name].after(job.run)
            except Exception as e:
                self._warn(f"{job.describe()} {name}: {e}")

    def run_once(self, *, stop: threading.Event | None = None) -> list[Job]:
        """Evaluate what lacks a result now, and say what that was, in the order it was done.

        Only the checkpoints the first look finds: one saved meanwhile is left for the next
        time, so that this ends however fast the runs save them. Each turn looks again all the
        same, so that one evaluated, pruned or replaced meanwhile drops out. ``stop`` is looked
        at between checkpoints, and by a suite that takes long, such as the sample games, as it
        goes.

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
            if not self.evaluate(job, stop=stop):
                raise Stopped(done, len(left))
            done.append(job)
        return done

    def run(self, *, poll_seconds: float, stop: threading.Event) -> None:
        """Evaluate checkpoints as they appear, until ``stop`` is set.

        ``stop`` is looked at between checkpoints, while waiting for more, and by a suite that
        takes long as it goes.
        """
        while not stop.is_set():
            job = self._next()
            if job is None:
                stop.wait(poll_seconds)
            else:
                self.evaluate(job, stop=stop)

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
