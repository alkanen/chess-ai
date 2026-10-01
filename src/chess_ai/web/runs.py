"""Training runs as the browser sees them: the run list, and one run followed as it trains.

Everything here only reads run directories, which the trainer may be writing to at this moment;
see :mod:`chess_ai.training.run_store` for why that is safe. Nothing is cached between reads
but where a reader following one run has got to in its metrics log.
"""

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from chess_ai.training.run_store import (
    TRAIN,
    VALIDATION,
    Heartbeat,
    LatestMetrics,
    RunError,
    RunInfo,
    RunReader,
    RunStatus,
)

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _now() -> datetime:
    return datetime.now(UTC)


def is_stale(beat: Heartbeat | None, *, now: datetime, stale_after: float) -> bool:
    """Whether ``beat`` says a run is running and has not said so for ``stale_after`` seconds.

    Only a running run can be stale: a run that finished, stopped or crashed said so as it
    ended, and a heartbeat that has not changed since is the truth about it. A running one whose
    heartbeat has stopped being rewritten has most likely died without the chance to say so —
    killed, out of memory, or the machine gone — and nothing else would ever tell anybody.
    """
    if beat is None or beat.status is not RunStatus.RUNNING:
        return False
    return now - beat.updated > timedelta(seconds=stale_after)


class RunSummary(BaseModel):
    """One training run, as the run list and the new-game form show it.

    Everything but the name is optional: a run being written right now, or one whose
    ``run.json`` cannot be read, is still a run with checkpoints worth playing against.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    architecture: str | None = None
    dataset: str | None = None
    """The name of the dataset the run trains on."""
    created: datetime | None = None
    status: RunStatus | None = None
    """What the run's last heartbeat said, which for a run that is not running may be stale."""
    stale: bool = False
    """Whether the run says it is running and its heartbeat has stopped being updated."""
    updated: datetime | None = None
    """When the last heartbeat was written."""
    step: int | None = None
    """How far the run has got, as its last heartbeat said."""
    steps: int | None = None
    """How far it is going, which with ``step`` says how far through it is."""
    epoch: float | None = None
    """How many times the training split has been gone through, as a fraction."""
    positions_per_second: float | None = None
    eta_seconds: float | None = None
    message: str | None = None
    """Why the run crashed, or anything else its last heartbeat had to say."""
    checkpoints: int = 0
    """How many checkpoints there are to choose between."""
    latest_train: dict[str, Any] | None = None
    """The last training line of the metrics log."""
    latest_validation: dict[str, Any] | None = None
    """The last validation line of the metrics log."""


def describe_run(
    run: RunReader,
    *,
    stale_after: float,
    now: datetime | None = None,
    latest: LatestMetrics | None = None,
) -> RunSummary:
    """One run as the lists show it, as far as its files can be read.

    A run being written right now is a run someone may well want to play against, so anything
    unreadable is left out rather than turned into a refusal: the name and the checkpoints on
    the disk are enough to pick one. Every part is read on its own, so that whatever cannot be
    read costs this run that one line of its description and costs the other runs nothing —
    one run mid-save must not turn the whole list into "could not list the training runs".

    ``latest`` is what follows the run's metrics log between calls; see :class:`LatestMetricsCache`.
    Without one, the whole log is read.
    """
    now = now or _now()
    latest = latest or run.track_latest_metrics()
    summary = RunSummary(name=run.name)
    for part, describe in (
        ("what it has saved", lambda: {"checkpoints": len(run.checkpoints())}),
        ("what it is", lambda: _describes(run.info)),
        ("where it has got to", lambda: _got_to(run.status, now=now, stale_after=stale_after)),
        ("what it has measured", lambda: _measured(latest.read())),
    ):
        try:
            summary = summary.model_copy(update=describe())
        except (RunError, OSError):
            logger.debug("run %s does not say %s", run.name, part, exc_info=True)
    return summary


def _describes(info: RunInfo) -> dict[str, Any]:
    """What ``run.json`` adds to a run's line in the list."""
    return {
        "architecture": info.model.architecture,
        "dataset": info.dataset.name,
        "created": info.created,
        "steps": info.steps,
    }


def _got_to(beat: Heartbeat | None, *, now: datetime, stale_after: float) -> dict[str, Any]:
    """What the last heartbeat adds, or nothing at all before a run has written one."""
    if beat is None:
        return {}
    return {
        "status": beat.status,
        "stale": is_stale(beat, now=now, stale_after=stale_after),
        "updated": beat.updated,
        "step": beat.step,
        "epoch": beat.epoch,
        "positions_per_second": beat.positions_per_second,
        "eta_seconds": beat.eta_seconds,
        "message": beat.message,
    }


def _measured(latest: dict[str, Any]) -> dict[str, Any]:
    return {"latest_train": latest.get(TRAIN), "latest_validation": latest.get(VALIDATION)}


class LatestMetricsCache:
    """Each run's latest metrics, followed from one run list request to the next.

    What makes asking for the list every few seconds cheap: each request reads only what every
    log has had appended since the one before. A run that has gone from the list is forgotten.

    Runs are told apart by directory and by when they started, as ``run.json`` says. The log
    alone cannot tell a run from the one that overwrote it: the new log may be given the old
    one's inode, and by the time the list is next asked for — which may be hours later — have
    outgrown what was read of the old one, and a tail would read on from the middle of it with
    the old run's numbers still standing. :class:`RunStream` tells runs apart the same way.
    """

    def __init__(self) -> None:
        self._runs: dict[Path, tuple[datetime | None, LatestMetrics]] = {}
        self._lock = threading.Lock()

    def follow(self, runs: list[RunReader]) -> list[LatestMetrics]:
        """What follows each of ``runs``, in the same order, and nothing for any other run."""
        started = [_run_started(run) for run in runs]
        with self._lock:
            followed: dict[Path, tuple[datetime | None, LatestMetrics]] = {}
            for run, began in zip(runs, started, strict=True):
                kept = self._runs.get(run.directory)
                if kept is None or kept[0] != began:
                    kept = (began, run.track_latest_metrics())
                followed[run.directory] = kept
            self._runs = followed
            return [followed[run.directory][1] for run in runs]


def _run_started(run: RunReader) -> datetime | None:
    """When ``run`` started, or ``None`` while its ``run.json`` cannot be read."""
    try:
        return run.info.created
    except RunError:
        return None


class CheckpointSummary(BaseModel):
    """One checkpoint of a run, as the form lists it."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    step: int
    created: datetime
    metrics: dict[str, float | None] = {}
    """The validation metrics measured at this step, which is what "best" is decided on."""
    best: bool
    """Whether this is the run's best checkpoint by its own metric."""
    latest: bool
    """Whether this is the newest checkpoint of the run."""


class RunCheckpoints(BaseModel):
    """A run's checkpoints, newest first, which is the order they are offered in."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    run: str
    checkpoints: list[CheckpointSummary]


def describe_checkpoints(run: RunReader) -> RunCheckpoints:
    """A run's checkpoints, newest first, each saying whether it is the best or the newest."""
    index = run.checkpoint_index()
    latest = index.checkpoints[-1].step if index.checkpoints else None
    return RunCheckpoints(
        run=run.name,
        checkpoints=[
            CheckpointSummary(
                step=info.step,
                created=info.created,
                metrics=info.metrics,
                best=info.step == index.best_step,
                latest=info.step == latest,
            )
            for info in reversed(index.checkpoints)
        ],
    )


class RunStateEvent(BaseModel):
    """What a run is and where it has got to; sent first, and again whenever it changes.

    ``info`` and ``heartbeat`` are ``None`` while they cannot be read: before the trainer has
    written them, or for a run whose files a newer or older version of this code wrote.
    """

    type: Literal["run"] = "run"
    name: str
    info: RunInfo | None
    heartbeat: Heartbeat | None
    stale: bool


class MetricsEvent(BaseModel):
    """Lines of the metrics log a viewer has not had yet.

    ``reset`` says that these are the log from its beginning and replace everything sent
    before: on the first event of a connection, and when the run has been started again under
    the same name.
    """

    type: Literal["metrics"] = "metrics"
    reset: bool
    records: list[dict[str, Any]]


RunEvent = RunStateEvent | MetricsEvent


class RunStream:
    """One run, followed: each poll says what has changed in it since the last.

    The web server only ever learns that a run has moved on by looking, since the trainer is a
    process it knows nothing of. Looking is cheap — two small JSON files and whatever has been
    appended to the metrics log — and the run state is compared with what was last sent, so a
    viewer hears about a heartbeat only when something in it changed. That includes the moment
    a heartbeat goes stale, which nothing on the disk marks: only time passing does.
    """

    def __init__(self, run: RunReader, *, stale_after: float, clock: Clock = _now) -> None:
        self._run = run
        self._stale_after = stale_after
        self._clock = clock
        self._tail = run.tail_metrics()
        self._sent: RunStateEvent | None = None

    @property
    def name(self) -> str:
        return self._run.name

    def poll(self) -> list[RunEvent]:
        """The events that bring a viewer up to date; the first poll's say everything."""
        events: list[RunEvent] = []
        state = self._state()
        before = _started(self._sent) if self._sent is not None else None
        if before is not None and _started(state) not in (before, None):
            # A run started again under the same name, which the log alone may not show: the
            # new one can have written as much as the old one had by the time this looks.
            self._tail = self._run.tail_metrics()
        if state != self._sent:
            events.append(state)
            self._sent = state
        records, reset = self._tail.read()
        if records or reset:
            events.append(MetricsEvent(reset=reset, records=records))
        return events

    def _state(self) -> RunStateEvent:
        info = heartbeat = None
        try:
            info = self._run.info
        except RunError:
            logger.debug("run %s does not say what it is", self._run.name, exc_info=True)
        try:
            heartbeat = self._run.status
        except RunError:
            logger.debug("run %s does not say where it has got to", self._run.name, exc_info=True)
        return RunStateEvent(
            name=self._run.name,
            info=info,
            heartbeat=heartbeat,
            stale=is_stale(heartbeat, now=self._clock(), stale_after=self._stale_after),
        )


def _started(state: RunStateEvent) -> datetime | None:
    return state.info.created if state.info is not None else None
