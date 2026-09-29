"""Building a dataset: PGN files in, a versioned directory of records and a manifest out.

The build streams. The input is measured in tens of gigabytes and the output can be larger, so
it writes records as it goes rather than gathering them up, and holds a bounded amount however
large the files are. That is also why nothing here counts games first: the time remaining comes
from bytes read, which is known without a first pass.

It reads in as many processes as it has cores, because parsing SAN and packing records is pure
Python and is all a build does. The files are cut into pieces of whole games, the pieces are
parsed in worker processes, and their records are appended here in the order the pieces came --
which is what makes the dataset identical, to the byte, to one read a game at a time in one
process. Only the offsets depend on the order, and they are decided here, by the writer. See
:meth:`_Build._read_in_parallel`.

It also never stops. Every way one game can be wrong is one game skipped and counted, not a
build abandoned four hours in, and the manifest says afterwards how many were lost to what.

The manifest is written last, which makes it the mark of a finished build: a build that died
partway leaves records nothing will read, because there is no manifest pointing at them.
"""

import logging
import os
import shutil
import sys
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import process as _executor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import chess.pgn
import numpy as np

from chess_ai.dataset.files import sync_directory
from chess_ai.dataset.games import (
    RESULT_NAMES,
    GameRecords,
    GameSkipped,
    SkipReason,
    game_records,
    rating_source_of,
    split_of,
)
from chess_ai.dataset.manifest import (
    SPLITS,
    TRAIN,
    VALIDATION,
    Filters,
    Manifest,
    Shards,
    SourceInfo,
    SplitCounts,
    Statistics,
    rating_bucket,
)
from chess_ai.dataset.progress import Progress
from chess_ai.dataset.records import (
    FORMAT_VERSION,
    GameFlags,
    RatingSource,
    Result,
    TimeControl,
)
from chess_ai.dataset.sources import (
    ByteRange,
    PgnReader,
    Source,
    game_ranges,
    games_in_range,
    quiet_parser,
    resolve_sources,
)
from chess_ai.dataset.store import (
    DEFAULT_SHARDS,
    DatasetError,
    DatasetWriter,
    abandoned_partials,
    dataset_lock,
    dataset_path,
    discarded_datasets,
    discarded_path,
    new_partial_path,
    replaced_datasets,
    replaced_path,
    valid_name,
)
from chess_ai.move_codec import VOCABULARY_SIZE

DEFAULT_VALIDATION_FRACTION: Final = 0.02
"""How much of a dataset is held back by default: enough games to measure, few to lose."""

AUTO_RATING_SOURCE: Final = "auto"
"""What the manifest records when each game's own headers said where its ratings came from."""

REPORT_EVERY: Final = 128
"""Games between progress reports, which a reporter may throttle further."""

SCANS_PER_REPORT: Final = 256
"""Boundary-scan windows between progress reports, which is this path's :data:`REPORT_EVERY`.

About 16 MB of scanning at :data:`~chess_ai.dataset.sources.ALIGN_SCAN`. Reporting per window
instead would be ~127,000 reports for a boundary-free Lichess month, in a phase where nothing
changes but the clock. The printer throttles and would not care, but it is not the only thing
the callback feeds -- a test collects them into a list, and a web server would push each one --
and every other report in this build is already bounded by its own cadence rather than leaning
on the reporter to have one.
"""

CHUNK_BYTES: Final = 8 << 20
"""How much of a PGN file one worker is given at a time.

Small enough that a worker holds its piece and the records made from it -- four times the size
of the PGN, so about 40 MB the pair -- without a build of 32 of them costing real memory, and
that a slow piece at the end of a file holds nothing else up. Large enough that handing it over
costs nothing next to parsing the ten thousand games in it.
"""

PARALLEL_FROM_BYTES: Final = 64 << 20
"""Input below which a build reads in this process however many workers it was offered.

Starting processes, handing pieces to them and pickling records back is worth it for a dump and
not for a directory of exports: below this the whole build is over in about the time the
processes would have taken to start.
"""

MAX_PIECE_BYTES: Final = 2 * CHUNK_BYTES
"""The largest piece a worker is given, past which it is read in the build's own process.

A piece is only as small as the boundaries :func:`~chess_ai.dataset.sources.game_ranges` could
find: a file whose games are not separated by a blank line has none, and cuts into one piece
spanning all of it. A worker keeps every record of its piece until it has them all -- several
times the size of the PGN, briefly doubled while they are joined, and then pickled back in one
message -- so a piece is a memory cost whatever care the *reading* took. Streaming the read
bounded the bytes; this is what bounds the records.

Twice :data:`CHUNK_BYTES`, because a piece is normally one of those plus at most the game that
straddles its end, and no single game comes close to that. A piece past it is a file that is
not shaped the way PGN says, and reading it slowly beats not reading it at all.
"""

IN_FLIGHT_PER_WORKER: Final = 2
"""Pieces given out per worker, so that finishing one does not leave a worker waiting.

This is also what bounds the memory, and the bound is on pieces rather than on bytes, so it is
worth saying where they sit. ``workers * IN_FLIGHT_PER_WORKER`` pieces are outstanding, and a
piece that has been read but not yet written is held *in this process*, not in the worker that
read it -- :func:`_in_order` keeps the futures in order, and a finished one holds its whole
:class:`_Read` until the writing catches up with it.

Measured on a real 8 MiB piece of a Lichess month: 9,170 games, 32.7 MB of records, so about
four times the PGN it came from. At the default of one worker per core on a 32-core machine
that is 64 outstanding, or 2.1 GB if the writing ever fell that far behind -- and 4.2 GB at
:data:`MAX_PIECE_BYTES`. A whole build of that month peaked at 1.0 GB, because the writing does
not fall behind, but the ceiling is the number to reason with rather than the observed peak.
"""

ProgressCallback = Callable[[Progress], None]

LOGGER = logging.getLogger(__name__)


MAX_DEFAULT_WORKERS: Final = 32
"""The most processes a build starts without being asked to.

Not because more cannot help, but because :data:`IN_FLIGHT_PER_WORKER` ties memory to this
number: every worker has pieces of a file outstanding, and a default nobody chose should not
be what fills a 96-core machine's memory. Ask for more with ``--workers`` and it is given.
"""

WORKER_CEILING: Final = 4
"""How far past the usable CPUs ``--workers`` may go before it reads as a typo, not a choice."""

MIN_WORKER_CEILING: Final = 64
"""A floor under that ceiling, so a small machine can still oversubscribe on purpose."""

WINDOWS_MAX_WORKERS: Final = getattr(_executor, "_MAX_WINDOWS_WORKERS", 61)
"""What :class:`~concurrent.futures.ProcessPoolExecutor` refuses to go past on Windows.

Taken from the executor itself where it says, because a ceiling of ours that named a number it
would not honour would be worse than no ceiling: the build would pass its own check and then
die inside the pool with a bare ``ValueError``, hours into reading, rather than with a word
about the argument. 61 is the documented value if a future version stops saying.
"""


def _usable_cpus() -> int:
    """How many CPUs this process may run on, which is not how many the machine has.

    ``os.cpu_count()`` counts the host's, so a build pinned to two cores of a 96-core machine --
    by ``taskset``, a cpuset, or a scheduler handing out part of a node -- would otherwise
    default to 96 processes, and to the memory 96 of them hold.

    A cgroup *quota* (``docker run --cpus=2``) is still not visible here: it throttles rather
    than restricts which CPUs are used, and nothing in the standard library reports it. Such a
    build gets the host's count and runs slowly rather than wrongly.
    """
    available = getattr(os, "process_cpu_count", None)  # 3.13+, and honours PYTHON_CPU_COUNT
    if available is not None and (usable := available()):
        return usable
    affinity = getattr(os, "sched_getaffinity", None)  # POSIX, and what taskset moves
    if affinity is not None:
        try:
            return len(affinity(0)) or 1
        except OSError:  # pragma: no cover - a kernel that refuses to say
            pass
    return os.cpu_count() or 1


def default_workers() -> int:
    """How many processes a build reads with when it is not told: one per usable core, capped.

    Reading is pure Python -- parsing SAN and packing records -- so it is the cores that decide
    how fast a build goes, and there is nothing else for them to be doing while it runs. See
    :data:`MAX_DEFAULT_WORKERS` for why it stops counting at some point.
    """
    return max(1, min(_usable_cpus(), MAX_DEFAULT_WORKERS))


def most_workers() -> int:
    """The most processes ``--workers`` may ask for before the number reads as a mistake.

    ``--workers 1000`` is a typo for ``100`` far more often than it is a plan, and it is not a
    cheap one: each process holds :data:`IN_FLIGHT_PER_WORKER` pieces of a file, so the memory a
    build takes is this number times a constant. Oversubscribing on purpose still works -- the
    ceiling is well above any useful count.

    On Windows it is also whatever the pool will actually start; see
    :data:`WINDOWS_MAX_WORKERS`.
    """
    ceiling = max(MIN_WORKER_CEILING, _usable_cpus() * WORKER_CEILING)
    if sys.platform == "win32":
        return min(ceiling, WINDOWS_MAX_WORKERS)
    return ceiling


def build_dataset(
    name: str,
    patterns: Sequence[str],
    *,
    data_dir: Path,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    rating_source: RatingSource | None = None,
    shards: Shards = DEFAULT_SHARDS,
    progress: ProgressCallback | None = None,
    overwrite: bool = False,
    workers: int | None = None,
    now: datetime | None = None,
) -> Manifest:
    """Build the dataset ``name`` under ``data_dir`` from the PGN files ``patterns`` name.

    ``rating_source`` records that pool for every game; ``None`` reads it from each game's own
    headers, which is what a mixed directory of exports needs. ``validation_fraction`` is how
    much is held back, decided per game by :func:`~chess_ai.dataset.games.split_of`.

    Raises :exc:`~chess_ai.dataset.store.DatasetError` for a name or a source that is not
    usable, or for a dataset that is already there and ``overwrite`` not asked for. Anything
    wrong with a *game* is counted instead, and the manifest that comes back says what was. A
    failure in *writing* the dataset is neither: it ends the build and leaves nothing behind.

    ``workers`` is how many processes parse the PGN files, defaulting to
    :func:`default_workers`. The files are cut into pieces of whole games and the pieces are read
    in parallel, which gives the same dataset as reading every game in this process, to the byte;
    ``1`` reads them here, and so does any build with less than :data:`PARALLEL_FROM_BYTES` to do.

    ``overwrite`` replaces the dataset only once the new one is finished, so the old one survives
    a build that fails — at the cost of both existing at once while it runs. That way round
    because a disk with room for only one copy is exactly the disk that fills up part-way
    through, and deleting first would leave neither.

    A *source* that cannot be opened is counted rather than fatal, but only so far: a build that
    could open none of them, that could not open all of them and would be replacing a dataset
    already there, or that kept no games at all, is refused and publishes nothing. See
    :func:`_check_worth_publishing`.
    """
    valid_name(name)
    if not 0.0 <= validation_fraction <= 1.0:
        raise DatasetError(f"validation fraction {validation_fraction} is not between 0 and 1")
    if workers is not None and workers < 1:
        raise DatasetError(f"a build needs at least one worker to read with, not {workers}")
    if workers is not None and workers > (ceiling := most_workers()):
        raise DatasetError(
            f"{workers} workers is more than this machine has any use for; the most it will "
            f"start is {ceiling}. Each one holds pieces of a PGN file while it reads them, so "
            "the memory a build takes grows with this number"
        )
    sources = resolve_sources(patterns)
    directory = dataset_path(data_dir, name)
    # Held for the whole build, so that a second build of this dataset says so now rather than
    # reading the same files for hours and then throwing the work away.
    with dataset_lock(data_dir, name):
        if directory.exists() and not overwrite:
            raise DatasetError(
                f"dataset {name!r} is already in {directory}; "
                "build it with --overwrite to replace it"
            )
        _report_rubble(data_dir, name)
        # Built beside where it belongs and moved there when it is finished, so that a dataset
        # directory always holds a whole dataset: an interrupted build leaves nothing for a
        # reader to find, for the next build to trip over, or for anyone to wonder about.
        partial = new_partial_path(data_dir, name)
        if partial.exists():
            # Whatever is there belongs to another build, and nothing below may remove it. The
            # name is random, so this is a sanity check rather than something anyone should meet.
            raise DatasetError(f"the working directory {partial} for dataset {name!r} is taken")
        try:
            # Inside the guard, because constructing the writer is what makes the working
            # directory: an interrupt an instant later would otherwise leave it behind for good.
            try:
                build = _Build(
                    directory=partial,
                    shards=shards,
                    sources=sources,
                    rating_source=rating_source,
                    validation_fraction=validation_fraction,
                    progress=progress,
                )
            except OSError as e:
                raise DatasetError(
                    f"cannot make the working directory {partial} for dataset {name!r}: "
                    f"{e.strerror}"
                ) from e
            with build.writer, quiet_parser():
                build.read_sources(
                    sources, workers=default_workers() if workers is None else workers
                )
                manifest = build.manifest(
                    name=name,
                    created=now if now is not None else datetime.now(UTC),
                )
            _check_worth_publishing(manifest, directory)
            manifest.save(partial)
            _publish(partial, directory, name=name, overwrite=overwrite)
            # After everything that can still fail: closing the shards flushes them, and
            # publishing renames them. A summary printed before those would say the build was
            # finished and then be followed by the reason it was not. Before the warnings
            # below, because it is what ends the progress line on this path.
            build.report(done=True)
        except BaseException:
            # Including a Ctrl-C: what was written is unreadable without a manifest, so leaving
            # it behind would only be rubble for the next build to clear up.
            shutil.rmtree(partial, ignore_errors=True)
            raise
        finally:
            # After the attempt rather than before it, because what to say about a dataset an
            # earlier build set aside depends on whether one is in place — and this build is the
            # thing that decides that. Asked first, it would answer for the state it was about
            # to change, and a build that went on to publish would leave "rename it back to have
            # it again" in the log of the very run that made following it lose the new dataset.
            #
            # The line is ended first because these warnings go to the same stream: on the path
            # where the build failed there is no summary to have ended it, and a warning written
            # onto the end of "0s left" is one whose path cannot be read or copied.
            _end_progress_line(progress)
            _report_set_aside(data_dir, name)
    return manifest


def _check_worth_publishing(manifest: Manifest, directory: Path) -> None:
    """Refuse to publish a build that read less than it was asked to, where that would lose.

    The sources are sized when a build starts and read as it reaches them, hours later, so a mount
    that drops or a sync job that rotates a directory of dumps can leave a build with nothing in
    it. Being tolerant of one bad file must not extend to publishing that:

    - if *every* source went away before saying anything, the build has no idea what it is
      missing, and that is what it is told rather than that it kept nothing, which it also did
    - if *some* source went away and a dataset is already there, that source took an unknown
      number of its games with it, the dataset in place was built from more than this one could
      be, and a nightly ``--overwrite`` must not trade it for this. The remedy is to fix the
      source or stop naming it, not a flag that says to carry on anyway
    - if *no games at all* were kept, it does not matter why, and it does not matter whether a
      dataset of this name is already there. A dataset of nothing is not a replacement for a
      dataset of something, and it is not a first dataset either: published with a manifest and
      an exit status of zero, it is one the next stage opens and trains on. This is also the one
      that catches a source truncated between being sized and being read, which fails in no way
      at all — no error, no games, nothing to complain of

    A source that was there throughout and whose *contents* stopped making sense is none of
    these, wherever in the file that happened: those games do not exist to be missed. See
    :attr:`~chess_ai.dataset.manifest.SourceInfo.went_away`.
    """
    silent = [source for source in manifest.sources if source.left_nothing]
    lost = [source for source in manifest.sources if source.went_away]
    in_place = directory.exists()
    if silent and len(silent) == len(manifest.sources):
        raise DatasetError(
            f"none of the {len(silent)} source(s) of dataset {manifest.name!r} could be read, "
            f"so there is nothing to build from: "
            f"{', '.join(str(source.path) for source in silent)}"
        )
    if lost and in_place:
        raise DatasetError(
            f"{len(lost)} of the {len(manifest.sources)} source(s) of dataset "
            f"{manifest.name!r} could not be read whole, and the dataset already in {directory} "
            f"was built from more than this one could be: "
            f"{', '.join(str(source.path) for source in lost)}. It has been left alone; fix "
            "those sources, or leave them out to build from the rest on purpose"
        )
    if manifest.games == 0:
        if in_place:
            raise DatasetError(
                f"this build of dataset {manifest.name!r} kept no games, and the dataset already "
                f"in {directory} has {_games_in(directory)}. It has been left alone; check the "
                "sources are what you meant before replacing it with nothing"
            )
        raise DatasetError(
            f"this build of dataset {manifest.name!r} kept no games, so there is no dataset to "
            f"publish; check the source(s) are what you meant: "
            f"{', '.join(str(source.path) for source in manifest.sources)}"
        )


def _publish(partial: Path, directory: Path, *, name: str, overwrite: bool) -> None:
    """Move the finished dataset into place, letting go of what was there only afterwards.

    Two renames with nothing destructive between them: the dataset in place is moved aside, the
    new one takes its name, and the old one is removed once losing it costs nothing. Deleting
    first meant that a delete which died part-way — one unreadable file, an interruption during
    the minutes it takes to unlink 50 GB — left neither the old dataset nor the new one.
    """
    aside: Path | None = None
    try:
        if directory.exists():
            if not overwrite:
                # Another build of this name finished while this one was reading. This build was
                # told not to replace a dataset, and that holds however late it finds out.
                raise DatasetError(
                    f"dataset {name!r} appeared in {directory} while this build was running; "
                    "build it with --overwrite to replace it"
                )
            # Named after this build, and never cleared first: a directory already at that
            # name would be a dataset an earlier build set aside, and deleting it would throw
            # away the very thing _report_set_aside reports. The rename below refuses
            # instead, which is what it should do for a name that cannot be free.
            aside = replaced_path(partial)
            try:
                os.replace(directory, aside)
            except FileNotFoundError:
                # Gone between the look and the move. Only reachable where builds are not
                # serialised, since dataset_lock is what stops two of them getting here at once.
                aside = None
        # Inside the same guard as the move aside, so that nothing at all can land between them.
        # An interrupt is aimed at exactly this moment, and the cost of losing that race is a
        # dataset sitting under a name no command here would ever mention again.
        os.replace(partial, directory)
    except BaseException as e:
        if aside is not None and not directory.exists():
            # Put it back rather than leave the dataset under a name nothing looks for.
            os.replace(aside, directory)
        if isinstance(e, OSError):
            raise DatasetError(
                f"could not put dataset {name!r} in {directory}: {e.strerror}"
            ) from e
        raise
    sync_directory(directory.parent)
    if aside is not None:
        # Renamed before it is deleted, so that a delete which stops half way leaves something
        # that reads as rubble. A half-deleted dataset still called ".replaced" would be offered
        # back to whoever reads the next build's warning, and renaming it over the dataset this
        # build just made would lose them both.
        discarded = discarded_path(aside)
        try:
            os.replace(aside, discarded)
            # Made durable before a single file of it is unlinked: a crash part-way through the
            # minutes it takes to delete 50 GB would otherwise leave the old name over gutted
            # contents, which is the state the rename exists to prevent.
            sync_directory(discarded.parent)
        except OSError:
            # Left whole rather than deleted where it is: a delete that stopped half way would
            # leave a gutted dataset still called ".replaced", which reads as a whole one set
            # aside. Reported as a leftover instead; it costs disk, and _report_set_aside
            # exists to make that somebody's decision.
            return
        # Past the point of no return: failing here leaves rubble rather than losing a dataset.
        shutil.rmtree(discarded, ignore_errors=True)


def _games_in(directory: Path) -> str:
    """How many games the dataset in ``directory`` holds, for saying what would have been lost."""
    from chess_ai.dataset.manifest import ManifestError, load_manifest

    try:
        return f"{load_manifest(directory).games:,} games"
    except ManifestError:
        return "games in it"


def _end_progress_line(progress: ProgressCallback | None) -> None:
    """End a progress line left open, so that what is printed next starts on a line of its own.

    A progress report is a plain callable and most have nothing to end. The one that rewrites a
    single line in a terminal leaves it unterminated until the build says it is done, which a
    build that failed never does, and says so by having a ``finish``. Ending it twice is not a
    thing to guard against here: the one that has it also knows whether its line is still open.
    """
    finish = getattr(progress, "finish", None)
    if finish is not None:
        finish()


def _report_rubble(data_dir: Path, name: str) -> None:
    """Say what earlier builds of this dataset left lying about, since nothing else will.

    Neither kind is a dataset and neither is removed: see
    :func:`~chess_ai.dataset.store.abandoned_partials` for why a working directory cannot safely
    be told from a live build's, and half of a dataset is still somebody's disk to free.

    Said before the build rather than after it, which is where the advice about a *whole* dataset
    set aside has to go (:func:`_report_set_aside`). Nothing here depends on how this build turns
    out, and a build about to spend four hours filling a disk that already holds the rubble of
    three others is worth telling first.
    """
    for path in abandoned_partials(data_dir, name):
        LOGGER.warning(
            "chess-ai: %s is a working directory of another build of %r — one that was killed, or "
            "one running elsewhere. It is not a dataset and nothing will read it; remove it once "
            "no build of %r is running.",
            path,
            name,
            name,
        )
    for path in discarded_datasets(data_dir, name):
        LOGGER.warning(
            "chess-ai: %s is what is left of a dataset an earlier build of %r replaced and could "
            "not finish deleting. It is part of a dataset without the rest of it; remove it.",
            path,
            name,
        )


def _report_set_aside(data_dir: Path, name: str) -> None:
    """Say that a whole dataset an earlier build moved out of the way is still there.

    Only its owner should decide what happens to it, so it is named and left alone. What to say
    about it depends entirely on whether a dataset of this name is in place, which is why this
    is asked once the build it is being said to has finished or failed: at that point the answer
    is settled, and the advice keeps for as long as nothing else builds this dataset.
    """
    in_place = dataset_path(data_dir, name)
    for path in replaced_datasets(data_dir, name):
        if in_place.exists():
            # Either a build that could not mark it as discarded, or one interrupted between the
            # two renames and followed by a later build. Which it was does not change the advice:
            # what is in place now is newer, and renaming this over it would lose that.
            LOGGER.warning(
                "chess-ai: %s is a dataset %r that a build set aside and did not remove. What is "
                "in %s now is newer than it, so renaming it back would replace the newer one; "
                "keep it only if you want the older dataset, and remove it otherwise.",
                path,
                name,
                in_place,
            )
            continue
        LOGGER.warning(
            "chess-ai: %s is the dataset %r that an interrupted build set aside and never put "
            "back. Nothing else will mention it: rename it to %s to have it again, or remove it.",
            path,
            name,
            dataset_path(data_dir, name),
        )


@dataclass
class _Tally:
    """How many games a build read, and what was in the ones it kept.

    Separate from the build because a parallel build counts in each worker and adds the counts
    up here: this is both a build's running total and one piece of a file's contribution to
    it, and :meth:`add` is what turns the second into the first.
    """

    games_read: int = 0
    results: Counter[Result] = field(default_factory=Counter)
    time_controls: Counter[TimeControl] = field(default_factory=Counter)
    rating_sources: Counter[RatingSource] = field(default_factory=Counter)
    ratings: Counter[str] = field(default_factory=Counter)
    ratings_unknown: int = 0
    skipped: Counter[SkipReason] = field(default_factory=Counter)
    unexpected: str | None = None
    """The first failure nothing expected, written out, for the build to say once.

    Carried as text rather than as the exception, because it travels back from a worker process:
    an arbitrary exception may not survive being pickled, and what is wanted from it is the
    traceback anyway.
    """

    def count(self, records: GameRecords) -> None:
        """Take one kept game into the statistics."""
        game = records.game[0]
        self.results[Result(int(game["result"]))] += 1
        self.time_controls[TimeControl(int(game["time_control"]))] += 1
        self.rating_sources[RatingSource(int(game["rating_source"]))] += 1
        flags = int(game["flags"])
        for rating, unknown in (
            (int(game["white_rating"]), GameFlags.WHITE_RATING_UNKNOWN),
            (int(game["black_rating"]), GameFlags.BLACK_RATING_UNKNOWN),
        ):
            if flags & unknown:
                self.ratings_unknown += 1
            else:
                self.ratings[rating_bucket(rating)] += 1

    def note(self, error: BaseException) -> None:
        """Keep the first failure nothing expected, to be said out loud once."""
        if self.unexpected is None:
            self.unexpected = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )

    def add(self, other: "_Tally") -> None:
        """Take another tally's counts into this one."""
        self.games_read += other.games_read
        self.results += other.results
        self.time_controls += other.time_controls
        self.rating_sources += other.rating_sources
        self.ratings += other.ratings
        self.ratings_unknown += other.ratings_unknown
        self.skipped += other.skipped
        if self.unexpected is None:
            self.unexpected = other.unexpected


@dataclass(frozen=True)
class _Job:
    """One piece of one PGN file to read, and everything reading it has to be told."""

    piece: ByteRange
    rating_source: RatingSource | None
    validation_fraction: float


@dataclass
class _Records:
    """One split's share of the games out of one piece, ready to be appended in one go."""

    games: np.ndarray
    positions: np.ndarray
    moves: np.ndarray
    ply_counts: np.ndarray
    """How many plies each game has, which is what the writer turns into offsets."""


@dataclass
class _Read:
    """What reading one piece produced: its records, its counts, and how it ended."""

    piece: ByteRange
    tally: _Tally
    splits: dict[str, _Records]
    error: str | None = None
    went_away: bool = False
    """Whether the file went away, rather than its contents having stopped making sense."""

    @property
    def kept(self) -> int:
        return sum(len(records.games) for records in self.splits.values())


@dataclass
class _SourceTally:
    """What a build has made of one source file: what the manifest will say about it."""

    read: int = 0
    kept: int = 0
    error: str | None = None
    went_away: bool = False
    stopped: bool = False
    """Whether a piece of this file gave up, so the rest of it is not this dataset's."""
    explained: bool = False
    """Whether this file has already been named as one being read the slow way."""


def _read_piece(job: _Job) -> _Read:
    """Read one piece of one PGN file into records. This is what runs in a worker process.

    Everything here is per game and keeps no state between games, which is what makes reading a
    file in several processes give the same dataset as reading it in one. In particular the split
    a game lands in is a hash of the game itself -- see
    :func:`~chess_ai.dataset.games.split_of` -- so it cannot depend on which piece the game was
    in or on which process read it. Where a game's records *go* is not decided here at all: that
    is the writer's, in order, in the parent.

    Nothing in here raises for a game. A piece that cannot be read stops, says why, and hands
    back what it had; the caller decides what that means for the file it came from.
    """
    tally = _Tally()
    bins: dict[str, tuple[list, list, list, list]] = {split: ([], [], [], []) for split in SPLITS}
    error: str | None = None
    went_away = False
    # Applied here as well as in the parent: a worker started by spawn or forkserver rather than
    # fork inherits nothing, and a dump of millions of games has thousands of unreadable ones.
    with quiet_parser():
        reading = games_in_range(job.piece)
        while True:
            try:
                # The offset is for a caller reading a piece itself; a worker reports one figure
                # for the whole piece when it hands it back.
                record, _ = next(reading)
            except StopIteration:
                break
            except Exception as e:
                # As in _read_games: the piece stopped mid-game. What was read is kept and the
                # rest is not, and whether the file went away or the chess in it stopped making
                # sense is the difference between an unknown number of games missing and none.
                tally.note(e)
                tally.skipped[SkipReason.UNREADABLE] += 1
                # The reason only. How many games the *file* had read by then is not known
                # here -- `tally` counts this piece -- and it is what the manifest wants, so
                # the sentence is composed in _take where the file's running total is.
                error = _described(e)
                went_away = isinstance(e, OSError)
                break
            tally.games_read += 1
            try:
                records = game_records(
                    record,
                    source=job.piece.source,
                    rating_source=(
                        rating_source_of(record) if job.rating_source is None else job.rating_source
                    ),
                )
            except GameSkipped as skipped:
                tally.skipped[skipped.reason] += 1
                continue
            except Exception as e:
                tally.note(e)
                tally.skipped[SkipReason.UNREADABLE] += 1
                continue
            split = VALIDATION if split_of(records.identity, job.validation_fraction) else TRAIN
            games, positions, moves, counts = bins[split]
            games.append(records.game)
            positions.append(records.positions)
            moves.append(records.moves)
            counts.append(len(records.positions))
            tally.count(records)
    return _Read(
        piece=job.piece,
        tally=tally,
        splits={
            split: _Records(
                games=np.concatenate(games),
                positions=np.concatenate(positions),
                moves=np.concatenate(moves),
                ply_counts=np.asarray(counts, dtype=np.int64),
            )
            for split, (games, positions, moves, counts) in bins.items()
            if games
        },
        error=error,
        went_away=went_away,
    )


def _in_order(
    pool: ProcessPoolExecutor,
    jobs: Sequence[_Job],
    *,
    in_flight: int,
    wanted: Callable[[_Job], bool] | None = None,
) -> Iterator["_Read | _Job"]:
    """Every job's outcome, in the order the jobs were given, ``in_flight`` of them at a time.

    A job comes back as its :class:`_Read`, or as the job itself when its piece is larger than
    :data:`MAX_PIECE_BYTES` -- too large for a worker to hold the records of, and so for the
    caller to read where nothing has to be held. See :meth:`_Build._read_here`.

    In order, because that is what makes a parallel build's dataset the same as a serial one's:
    the records reach the shards in the order the games appear in the files, so every offset and
    every game index comes out where it would have.

    Bounded, because reading is several times faster than writing and a piece's records are
    several times the size of the PGN they came from. Handing the pool every job at once would
    put the whole dataset in this process's memory on its way to the disk -- hundreds of
    gigabytes for a Lichess month -- so one more is given out only as one comes back.

    ``wanted`` is asked about each job before it is given out, and one it refuses is neither read
    nor yielded. That is what lets a file which gave up stop costing anything: the caller backs
    it with whether the piece's source has stopped, and the pieces after the failure are skipped
    rather than parsed in full and thrown away. Asked at the moment of handing out rather than
    up front, because whether a source has stopped is only known once an earlier piece of it has
    come back.
    """

    def given_out(job: _Job) -> tuple[_Job, "Future[_Read] | None"]:
        """``job`` with the worker reading it, or with ``None`` if it is too large for one."""
        if job.piece.bytes > MAX_PIECE_BYTES:
            return job, None
        return job, pool.submit(_read_piece, job)

    queued: deque[tuple[_Job, Future[_Read] | None]] = deque()
    rest = iter(jobs)

    def fill(count: int) -> None:
        """Hand out up to ``count`` more jobs, passing over any ``wanted`` refuses."""
        added = 0
        while added < count:
            job = next(rest, None)
            if job is None:
                return
            if wanted is not None and not wanted(job):
                continue
            queued.append(given_out(job))
            added += 1

    fill(in_flight)
    while queued:
        job, running = queued.popleft()
        # Given out before waiting on the one that is due, so a worker is never idle while this
        # process is blocked on a result.
        fill(1)
        if running is None:
            yield job
            continue
        try:
            read = running.result()
        except MemoryError as e:
            # A worker holds every record of its piece and then joins them, so this is a real
            # way for one to end. It arrives as itself rather than as BrokenProcessPool, so it
            # would otherwise miss the handler for the same failure one step further along --
            # where the kernel does the killing -- and reach the user as a traceback out of
            # concurrent.futures naming neither the file nor the knob.
            raise DatasetError(
                f"a process ran out of memory reading {job.piece.bytes:,} bytes of "
                f"{job.piece.path}. Each one holds every record of its piece at once, so "
                "fewer --workers would avoid it"
            ) from e
        yield read


class _Build:
    """One build in progress: what has been read, what has been kept, and what went wrong.

    This exists so that the reading, the counting and the reporting can see each other without
    being threaded through arguments: a progress report is a snapshot of all three at once.
    """

    def __init__(
        self,
        *,
        directory: Path,
        shards: Shards,
        sources: Sequence[Source],
        rating_source: RatingSource | None,
        validation_fraction: float,
        progress: ProgressCallback | None,
    ) -> None:
        self.writer = DatasetWriter(directory, shards)
        self.shards = shards
        self.rating_source = rating_source
        self.validation_fraction = validation_fraction
        self._progress = progress
        self._started = time.monotonic()
        self._bytes_total = sum(source.bytes for source in sources)
        self._bytes_before = 0
        """Bytes of the sources already finished, which the current file's count adds to."""
        self._bytes_read = 0
        self.sources: list[SourceInfo] = []
        self.tally = _Tally()
        self._said_unexpected = False

    def read_sources(self, sources: Sequence[Source], *, workers: int) -> None:
        """Read every source into the dataset, in this process or in ``workers`` of them.

        One process below :data:`PARALLEL_FROM_BYTES` of input whatever was asked for, because
        starting processes for a directory of exports costs more than reading it does.
        """
        if workers > 1 and self._bytes_total >= PARALLEL_FROM_BYTES:
            self._read_in_parallel(sources, workers=workers)
            return
        # Before the first source, as the parallel path does, so there is a line on the screen
        # from the first moment rather than from the 128th game: _read_games reports on its own
        # cadence, and a source that gives no games never reaches it at all.
        self.report()
        for index, source in enumerate(sources):
            self.read_source(source, index)

    def _read_in_parallel(self, sources: Sequence[Source], *, workers: int) -> None:
        """Read the sources with ``workers`` processes parsing pieces of them at once.

        Parsing a game and turning it into records is what a build spends its time on, and both
        are per game. The only part that has to happen in order is the writing, and that is
        arithmetic over an array. So the files are cut into pieces of whole games, other processes
        read the pieces, and their records are appended here in the order the pieces came in --
        which gives the same dataset, to the byte, as reading every game in this one.

        With one difference worth knowing: the pieces end where the file ended when the build
        started, since that is the size :func:`~chess_ai.dataset.sources.resolve_sources`
        recorded and what the progress is measured against. The serial path reads to whatever
        the end is when it gets there, so a file still being written -- a download or an export
        that has not finished -- gives the two paths different games. Stopping at the recorded
        size is the deliberate choice: it is the one that makes a build reproducible and its
        percentage mean something. A file that grows underneath a build is a broken input
        either way, and the serial path's answer to it is a truncated last game rather than a
        better one.

        The counting is the same either way: each piece brings a :class:`_Tally` of its own and
        they are added up here, which is why the statistics cannot depend on how many workers
        read the files.
        """
        before: dict[int, int] = {}
        running = 0
        for index, source in enumerate(sources):
            before[index] = running
            running += source.bytes
        found = {index: _SourceTally() for index in range(len(sources))}
        jobs: list[_Job] = []
        # Before the cutting rather than after it, so there is a line on the screen from the
        # first moment. It says zeros until a piece comes back, which is what is true: the
        # bytes it counts are bytes games have been read from, and cutting has read none.
        self.report(scanning=True)
        for index, source in enumerate(sources):
            try:
                pieces = game_ranges(source, index, CHUNK_BYTES, on_scan=self._scanning())
            except DatasetError as e:
                # The file cannot be read at all, which is what PgnReader.open failing means in
                # the serial path, and gets the same answer: counted, and not the end of a build.
                found[index].error = str(e)
                found[index].went_away = True
                LOGGER.warning("chess-ai: %s; its games are not in this dataset", e)
                continue
            jobs.extend(
                _Job(
                    piece=piece,
                    rating_source=self.rating_source,
                    validation_fraction=self.validation_fraction,
                )
                for piece in pieces
            )
            # Cutting a well-formed file reads under a percent of it and is over in
            # milliseconds, but a file with no boundary in it is scanned to the end before it
            # yields its single piece -- 300 MB/s, so half a minute for a Lichess month and
            # longer for a directory of them. Nothing has been read at this point, so what the
            # line says is the clock -- which it only does for a report that says it is
            # scanning; see format_progress.
            self.report(scanning=True)
        # Nothing of this build's is handed to a worker but the jobs. No shard is open yet --
        # _StreamWriter opens its file on the first append, which is the first result coming
        # back, after the pool exists -- and under spawn or forkserver a worker inherits nothing
        # regardless. Under fork, a worker started later does inherit the shards this process
        # has open by then: _adjust_process_count runs per submit and returns early when a
        # worker is idle, so the pool can grow at any point in the loop, not only to replace one
        # that died. That is safe because such a worker neither writes to those descriptors nor
        # flushes them -- multiprocessing ends a child with os._exit, which does not flush what
        # Python has buffered -- and not because of anything the pool promises about when it
        # starts processes, which is nothing.
        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            for outcome in _in_order(
                pool,
                jobs,
                in_flight=workers * IN_FLIGHT_PER_WORKER,
                wanted=lambda job: not found[job.piece.source].stopped,
            ):
                index = outcome.piece.source
                state = found[index]
                # An earlier piece of this file giving up gives up the rest of it, as the serial
                # path does. `wanted` above keeps those pieces from being handed out at all --
                # without it a dump that failed at piece 12 of 1,200 had the other 1,188 parsed
                # in full and thrown away -- but the few already in flight when it gave up still
                # arrive here, and their games are dropped rather than written.
                #
                # The progress below then jumps to the end of the file rather than to this
                # piece's end, because the rest of it is not going to be read: left counting
                # pieces, the fraction would stall for whatever share of the build this file was.
                if not state.stopped:
                    if isinstance(outcome, _Job):
                        self._read_here(outcome, state, before=before[index])
                    else:
                        self._take(outcome, state)
                self._bytes_read = before[index] + (
                    sources[index].bytes if state.stopped else outcome.piece.end
                )
                self.report()
        except BrokenProcessPool as e:
            # Every way a worker can stop answering arrives here, and the common one is not the
            # one worth advising about: a Ctrl-C goes to the whole process group, so the workers
            # can die before this process's own KeyboardInterrupt is delivered to it. Telling
            # somebody who just interrupted their build to use fewer workers would be nonsense,
            # so the reason comes before the advice and the advice is conditional.
            raise DatasetError(
                f"a process reading the PGN files stopped without answering: {e}. If you "
                "interrupted the build, that is why. Otherwise it was most likely killed for "
                "using too much memory, and fewer --workers would avoid it"
            ) from e
        finally:
            # Cancelled so that a build being torn down is not waiting on a queue of pieces
            # nobody will write, and waited for so that no worker is still reading a file while
            # this build is being cleared up after.
            pool.shutdown(wait=True, cancel_futures=True)
        for index, source in enumerate(sources):
            state = found[index]
            self.sources.append(
                SourceInfo(
                    path=str(source.path),
                    bytes=source.bytes,
                    games_read=state.read,
                    games_kept=state.kept,
                    error=state.error,
                    went_away=state.went_away,
                )
            )
        self._bytes_read = self._bytes_total

    def _take(self, read: _Read, state: _SourceTally) -> None:
        """Write one worker's piece into the dataset, and take its counts into the build's."""
        state.read += read.tally.games_read
        state.kept += read.kept
        self.tally.add(read.tally)
        self._say_unexpected()
        for split, records in read.splits.items():
            self.writer.splits[split].add_games(
                records.games, records.positions, records.moves, records.ply_counts
            )
        if read.error is not None:
            # Counted after the increment above, so this is the file's total and reads the same
            # as the sentences _read_games and _read_here build from theirs. A worker's own
            # count would say "after 7 games" of a file that had read six million.
            state.error = f"stopped reading after {state.read} games: {read.error}"
            state.went_away = read.went_away
            state.stopped = True

    def _reading(self, before: int, size: int) -> Callable[[int], None]:
        """A callback that moves the progress along as bytes are read, not only as games are.

        Both readers only reach their own loop when the parser returns, and on a file with no
        games in it that is once, at the end. This is called from under the parser instead, so
        the one number that is moving -- how far into the file the reading has got -- is the one
        the line is built from. ``before`` is the bytes of the sources already finished, and
        ``size`` is how many the source being read was recorded as having: the serial reader
        follows a file to its real end, and what a file that grew has past that size is not
        progress through the next one.
        """

        def at(position: int) -> None:
            self._bytes_read = before + min(position, size)
            self.report()

        return at

    def _scanning(self) -> Callable[[], None]:
        """A callback for :func:`~chess_ai.dataset.sources.align_to_game` to report through.

        The cadence lives here rather than in the scan because how often a build says what it is
        doing is the build's business; the scan's is finding boundaries. It reports on the first
        window as well as every :data:`SCANS_PER_REPORT` after it, so a file whose cut is short
        still says it has started.
        """
        seen = 0

        def scanned() -> None:
            nonlocal seen
            seen += 1
            # Counted from the first window rather than to the last, so that it fires on window
            # one whatever the cadence is -- `seen % 1` is never 1, which made a cadence of one
            # report nothing at all.
            if (seen - 1) % SCANS_PER_REPORT == 0:
                self.report(scanning=True)

        return scanned

    def _read_here(self, job: _Job, state: _SourceTally, *, before: int) -> None:
        """Read one piece in this process, a game at a time, writing each as it is made.

        For a piece past :data:`MAX_PIECE_BYTES`, which a worker must not be given: it would hold
        every record of it at once, and the piece this happens for is typically a whole file.
        Read here, each game reaches the shards as soon as it exists and nothing accumulates --
        the same bound the serial path has always had, and the same games in the same order, so
        the dataset is the one a worker would have produced.

        Slower than a worker, and said out loud once per file, because a build that quietly took
        an hour longer than it should have is a build nobody can fix the cause of.

        Slower than that, in fact, and worth being exact about: this runs on the consumer side of
        :func:`_in_order`, so while it runs the generator is suspended and nothing more is handed
        out. The pool finishes the ``workers * IN_FLIGHT_PER_WORKER`` pieces already with it and
        then idles for the rest of the read -- so the build is *single-threaded* for the duration,
        not merely slower on this stretch, and the parent holds those finished results the whole
        time rather than transiently. For a 20 GB file of this shape that is hours with one core
        busy.

        Fixing that means reading the piece somewhere the loop can keep consuming past it -- a
        thread, or the pool returning bounded batches -- and neither is worth it for an input this
        malformed while the alternative is a correct dataset either way. What is not acceptable is
        being quiet about it, which is why it is here and in the warning below. ``before`` is
        how many bytes of earlier sources there are, so that the progress this reports is the
        whole build's and keeps moving throughout -- this being the read where it matters most,
        and the one where a worker's single figure at the end would say nothing for hours.
        """
        if not state.explained:
            state.explained = True
            LOGGER.warning(
                "chess-ai: %s has %s bytes with no game boundary in them, which is more than a "
                "worker should hold at once, so that stretch is being read in this process, on "
                "one core, with the others idle until it is done. Its games are most likely not "
                "separated by a blank line.",
                job.piece.path,
                f"{job.piece.bytes:,}",
            )
        reading = games_in_range(job.piece, on_read=self._reading(before, job.piece.end))
        while True:
            try:
                record, position = next(reading)
            except StopIteration:
                return
            except Exception as e:
                # The same answer _read_games and _read_piece give: what was read is kept, the
                # rest of the file is not, and whether it went away decides what that costs.
                self._note_failure(e)
                self.tally.skipped[SkipReason.UNREADABLE] += 1
                state.error = f"stopped reading after {state.read} games: {_described(e)}"
                state.went_away = isinstance(e, OSError)
                state.stopped = True
                return
            state.read += 1
            self.tally.games_read += 1
            state.kept += self._add_game(record, job.piece.source)
            if state.read % REPORT_EVERY == 0:
                self._bytes_read = before + position
                self.report()

    def read_source(self, source: Source, index: int) -> None:
        """Read every game in one file into the dataset, whatever the file turns out to hold.

        What goes wrong in the *file* is counted and the rest of the file given up on. What goes
        wrong in writing the dataset — a full disk, a write error — is not a broken game and is
        not this method's to forgive: it ends the build, because counting it as one skipped game
        would abandon the rest of the file and then report a clean build over what was lost.
        """
        read = 0
        kept = 0
        error: str | None = None
        went_away = False
        reader = PgnReader(source, on_read=self._reading(self._bytes_before, source.bytes))
        try:
            # The open is guarded on its own, and nothing else is: a DatasetError out of the
            # reading below would be a write failure recorded as a bad file, which is the shape
            # of bug the reading was split up to make impossible.
            reader.open()
        except DatasetError as e:
            # The file could not be opened at all. Its size was taken when the build started and
            # it is opened hours later, so this is a file that went away or changed hands rather
            # than one that was never there: resolve_sources refuses those up front. Hours of
            # reading is not worth throwing away over one file of a hundred.
            error = str(e)
            went_away = True
            LOGGER.warning("chess-ai: %s; its games are not in this dataset", e)
        else:
            try:
                read, kept, error, went_away = self._read_games(reader, index)
            finally:
                reader.close()
        self._bytes_before += source.bytes
        self._bytes_read = self._bytes_before
        # The file is behind us either way, however few of its games were kept, and those bytes
        # are read bytes. Without this a source that gave nothing passes in silence -- which for
        # a 20 GB file that is not PGN is the whole read with a blank terminal.
        self.report()
        self.sources.append(
            SourceInfo(
                path=str(source.path),
                bytes=source.bytes,
                games_read=read,
                games_kept=kept,
                error=error,
                went_away=went_away,
            )
        )

    def _read_games(self, reader: PgnReader, index: int) -> tuple[int, int, str | None, bool]:
        """Read an open file's games: how many it held, how many were kept, why not, and whose
        fault it was.

        Only the parsing of each game is forgiven here. Everything the loop body does — writing
        the records, counting them — is outside that, because a write failure is not a broken
        game: counted as one it would abandon the rest of the file and then report a clean build
        over what was lost.
        """
        read = 0
        kept = 0
        games = reader.games()
        while True:
            try:
                record = next(games)
            except StopIteration:
                return read, kept, None, False
            except Exception as e:
                # The file stopped, mid-game. What was read is kept, the rest of the file is not,
                # and the count says a game was lost. Whether the file went away or the chess in
                # it stopped making sense is the difference between a dataset missing an unknown
                # number of games and one that has all the games there were: an OSError is the
                # file, and anything else is what the parser found in it.
                self._note_failure(e)
                self.tally.skipped[SkipReason.UNREADABLE] += 1
                why = f"stopped reading after {read} games: {_described(e)}"
                return read, kept, why, isinstance(e, OSError)
            read += 1
            self.tally.games_read += 1
            # Clamped for the same reason the cut is: the reader follows the file to its real end
            # and the source's size is the one recorded when the build started, so a file that
            # grew in between would count into the next source's share -- past the total, for
            # the last one, and the estimate below zero -- and then step back when it finished.
            self._bytes_read = self._bytes_before + min(reader.bytes_read, reader.source.bytes)
            kept += self._add_game(record, index)
            if read % REPORT_EVERY == 0:
                self.report()

    def _add_game(self, record: chess.pgn.Game, index: int) -> int:
        """Add one game, or count why it could not be added. Returns 1 if it was kept."""
        try:
            records = game_records(
                record,
                source=index,
                rating_source=(
                    rating_source_of(record) if self.rating_source is None else self.rating_source
                ),
            )
        except GameSkipped as skipped:
            self.tally.skipped[skipped.reason] += 1
            return 0
        except Exception as e:
            # Not a shape of broken game anyone has seen yet. One game is not worth abandoning
            # a build for, so it goes in the same place as the rest.
            self._note_failure(e)
            self.tally.skipped[SkipReason.UNREADABLE] += 1
            return 0
        split = VALIDATION if split_of(records.identity, self.validation_fraction) else TRAIN
        self.writer.splits[split].add_game(records.game, records.positions, records.moves)
        self.tally.count(records)
        return 1

    def _note_failure(self, error: BaseException) -> None:
        """Keep what went wrong in a way nothing expected, and say so once."""
        self.tally.note(error)
        self._say_unexpected()

    def _say_unexpected(self) -> None:
        """Say once that something went wrong in a way nothing here expected.

        Counting a game as unreadable is right for a broken game and hides a broken build: a
        bug in this code would skip every game in every file and report a clean dataset of
        nothing. The first one is logged, with its traceback, so that a build going wrong
        systematically says so instead of looking like a directory of bad PGN.

        Said from whichever path found it, so that a build reading in several processes says it
        once for the whole build rather than once per process -- the count is the build's, and
        so is the warning that explains it.
        """
        if self._said_unexpected or self.tally.unexpected is None:
            return
        self._said_unexpected = True
        LOGGER.warning(
            "chess-ai: a game could not be read for a reason nothing expected, and is counted "
            "as %s; further ones are counted silently\n%s",
            SkipReason.UNREADABLE.value,
            self.tally.unexpected,
        )

    def report(self, *, done: bool = False, scanning: bool = False) -> None:
        """Tell the progress callback, if there is one, where the build has got to.

        A report that cannot be made is not a build that cannot be finished. The callback writes
        somewhere this has no opinion about — a terminal that can close, an ssh session that can
        drop, a pipe into ``head`` that can go away — and none of that is worth a day's reading.
        The first failure is said out loud and is the last one attempted.
        """
        if self._progress is None:
            return
        try:
            self._progress(
                Progress(
                    games_read=self.tally.games_read,
                    games_kept=sum(split.games for split in self.writer.splits.values()),
                    positions=sum(split.positions for split in self.writer.splits.values()),
                    bytes_read=self._bytes_total if done else self._bytes_read,
                    bytes_total=self._bytes_total,
                    seconds=time.monotonic() - self._started,
                    done=done,
                    scanning=scanning,
                )
            )
        except Exception as e:
            self._progress = None
            LOGGER.warning(
                "chess-ai: progress reporting stopped, and the build carries on: %s",
                _described(e),
            )

    def manifest(self, *, name: str, created: datetime) -> Manifest:
        """Everything the build now knows about the dataset it has written."""
        return Manifest(
            format_version=FORMAT_VERSION,
            name=name,
            created=created,
            move_vocabulary_size=VOCABULARY_SIZE,
            validation_fraction=self.validation_fraction,
            rating_source=(
                AUTO_RATING_SOURCE
                if self.rating_source is None
                else self.rating_source.name.lower()
            ),
            sources=self.sources,
            filters=Filters(),
            shards=self.shards,
            splits={
                split: SplitCounts(
                    games=self.writer.splits[split].games,
                    positions=self.writer.splits[split].positions,
                )
                for split in SPLITS
            },
            skipped={
                reason.value: self.tally.skipped[reason]
                for reason in SkipReason
                if self.tally.skipped[reason]
            },
            statistics=Statistics(
                results={
                    text: self.tally.results[result]
                    for result, text in RESULT_NAMES.items()
                    if self.tally.results[result]
                },
                time_controls=_by_name(self.tally.time_controls),
                rating_sources=_by_name(self.tally.rating_sources),
                ratings=dict(sorted(self.tally.ratings.items(), key=lambda item: int(item[0]))),
                ratings_unknown=self.tally.ratings_unknown,
            ),
        )


def _described(error: Exception) -> str:
    """One line naming what went wrong, for the manifest to record against a source."""
    return f"{type(error).__name__}: {error}"


def _by_name(counts: Counter) -> dict[str, int]:
    """Counted enum members as lowercase names, in the enum's own order, dropping zeroes."""
    return {member.name.lower(): counts[member] for member in sorted(counts) if counts[member]}
