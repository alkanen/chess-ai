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

What it does leave is a checkpoint, written about once a minute once the records it counts are on
the disk: where the reading had got to, and the counts so far. A build or append that stopped part
way is carried on from there by cutting the records back to what the checkpoint counts and reading
on; see :mod:`~chess_ai.dataset.checkpoint`. Where it can be carried on from is wherever
everything read is in the dataset and nothing else is -- between pieces, or between games of a
stretch read in this process -- which is where :meth:`_Build.checkpoint` is called.

A build is an append to an empty dataset. :func:`append_dataset` reads more sources into a dataset
that is already there, through the filters it was built with, and adds what they kept to the end
of it as a new version; :func:`build_dataset` does the same into a directory of its own and moves
it into place when it is done. One path, so that a dataset appended to month by month holds the
same records as one built from all the months at once.
"""

import logging
import os
import shutil
import sys
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import process as _executor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Final

import chess.pgn
import numpy as np

from chess_ai.dataset.checkpoint import (
    Checkpoint,
    Counts,
    PlannedSource,
    SourceState,
    has_checkpoint,
    load_checkpoint,
    remove_checkpoint,
)
from chess_ai.dataset.download import dump_month
from chess_ai.dataset.files import sha256_of, sync_directory
from chess_ai.dataset.filters import FilterReason, Screen
from chess_ai.dataset.games import (
    RESULT_NAMES,
    GameSkipped,
    SkipReason,
    game_records,
    rating_source_of,
    skips_moves,
    split_of,
)
from chess_ai.dataset.manifest import (
    SPLITS,
    TRAIN,
    VALIDATION,
    Filters,
    Manifest,
    ManifestError,
    Shards,
    SourceInfo,
    SplitCounts,
    Statistics,
    Version,
    load_manifest,
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
    CompressedTail,
    PgnReader,
    Piece,
    Source,
    TextPiece,
    game_ranges,
    games_in_piece,
    quiet_parser,
    resolve_sources,
    text_pieces,
)
from chess_ai.dataset.store import (
    DEFAULT_SHARDS,
    DatasetError,
    DatasetWriter,
    SplitWriter,
    abandoned_partials,
    cut_back,
    cut_back_to,
    dataset_lock,
    dataset_path,
    discarded_datasets,
    discarded_path,
    new_partial_path,
    records_missing,
    records_past,
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

A piece of a compressed file is its text rather than a pair of offsets, so each outstanding one
also holds up to :data:`MAX_PIECE_BYTES` of PGN here until its worker is done with it: another
0.5 GB at 64 pieces of :data:`CHUNK_BYTES`, on top of the records.
"""

HASH_REPORT_BYTES: Final = 64 << 20
"""Bytes checksummed between progress reports while the sources are being fingerprinted."""

CHECKPOINT_SECONDS: Final = 60.0
"""How often a build writes down where it has got to, so that it can be carried on from there.

Each one flushes every open shard to the disk, which costs a fraction of a second; a minute of
reading is what an interrupted build loses at most.
"""

ProgressCallback = Callable[[Progress], None]

Confirm = Callable[[str], bool]
"""Asks a person a yes-or-no question, and says whether they answered yes."""

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
    filters: Filters | None = None,
    allow_repeat: bool = False,
    resume: bool = False,
    discard_interrupted: bool = False,
    confirm: Confirm | None = None,
) -> Manifest:
    """Build the dataset ``name`` under ``data_dir`` from the PGN files ``patterns`` name.

    ``rating_source`` records that pool for every game; ``None`` reads it from each game's own
    headers, which is what a mixed directory of exports needs. ``validation_fraction`` is how
    much is held back, decided per game by :func:`~chess_ai.dataset.games.split_of`.

    Raises :exc:`~chess_ai.dataset.store.DatasetError` for a name or a source that is not
    usable, or for a dataset that is already there and ``overwrite`` not asked for. Anything
    wrong with a *game* is counted instead, and the manifest that comes back says what was. A
    failure in *writing* the dataset is neither: it ends the build.

    ``workers`` is how many processes parse the PGN files, defaulting to
    :func:`default_workers`. The files are cut into pieces of whole games and the pieces are read
    in parallel, which gives the same dataset as reading every game in this process, to the byte;
    ``1`` reads them here, and so does any build with less than :data:`PARALLEL_FROM_BYTES` to do.

    ``filters`` says which games are kept and which of their positions are trained on; see
    :class:`~chess_ai.dataset.manifest.Filters`. Without them every game is kept, less those no
    build keeps: the broken ones and those ended for cheating.

    ``overwrite`` replaces the dataset only once the new one is finished, so the old one survives
    a build that fails — at the cost of both existing at once while it runs. That way round
    because a disk with room for only one copy is exactly the disk that fills up part-way
    through, and deleting first would leave neither.

    A *source* that cannot be opened is counted rather than fatal, but only so far: a build that
    could open none of them, that could not open all of them and would be replacing a dataset
    already there, or that kept no games at all, is refused and publishes nothing. See
    :func:`_check_worth_publishing`.

    The same file named twice under different names, or a Lichess month named both compressed
    and not, is refused unless ``allow_repeat`` says it is meant; see :func:`_check_repeats`.

    The dataset this makes is version 1 of it; :func:`append_dataset` adds the next.

    A build that stops part way through reading -- interrupted, killed, out of disk or memory --
    keeps its working directory and the checkpoint in it, and ``resume`` carries it on from there
    with the sources and settings it was started with, which are then not given again; see
    :mod:`~chess_ai.dataset.checkpoint`. A new build of a dataset that has an interrupted one
    asks ``confirm`` whether to discard it, or discards it without asking if
    ``discard_interrupted`` says to. With neither it is refused: that build's hours are not this
    one's to throw away.
    """
    valid_name(name)
    if not 0.0 <= validation_fraction <= 1.0:
        raise DatasetError(f"validation fraction {validation_fraction} is not between 0 and 1")
    _check_workers(workers)
    if resume and patterns:
        raise DatasetError(
            "a resumed build reads the sources the interrupted one was started with; name none"
        )
    if resume and discard_interrupted:
        raise DatasetError("an interrupted build can be resumed or discarded, not both")
    sources = [] if resume else resolve_sources(patterns)
    directory = dataset_path(data_dir, name)
    # Held for the whole build, so that a second build of this dataset says so now rather than
    # reading the same files for hours and then throwing the work away.
    with dataset_lock(data_dir, name):
        if directory.exists() and not overwrite:
            raise DatasetError(
                f"dataset {name!r} is already in {directory}; "
                "build it with --overwrite to replace it"
            )
        interrupted = interrupted_builds(data_dir, name)
        checkpoint: Checkpoint | None = None
        if resume:
            partial, checkpoint = _build_to_resume(name, interrupted)
        else:
            if interrupted:
                _settle_interrupted_builds(
                    name, interrupted, discard=discard_interrupted, confirm=confirm
                )
            # Built beside where it belongs and moved there when it is finished, so that a
            # dataset directory always holds a whole dataset: an interrupted build leaves
            # nothing for a reader to find.
            partial = new_partial_path(data_dir, name)
            if partial.exists():
                # Whatever is there belongs to another build, and nothing below may remove it.
                # The name is random, so this is a sanity check rather than something anyone
                # should meet.
                raise DatasetError(f"the working directory {partial} for dataset {name!r} is taken")
        _report_rubble(data_dir, name)
        build: _Build | None = None
        try:
            # Inside the guard, because constructing the writer is what makes the working
            # directory: an interrupt an instant later would otherwise leave it behind for good.
            try:
                build = (
                    _Build.resuming(partial, checkpoint, progress=progress)
                    if checkpoint is not None
                    else _Build(
                        directory=partial,
                        shards=shards,
                        sources=sources,
                        rating_source=rating_source,
                        validation_fraction=validation_fraction,
                        progress=progress,
                        filters=filters if filters is not None else Filters(),
                    )
                )
            except OSError as e:
                raise DatasetError(
                    f"cannot make the working directory {partial} for dataset {name!r}: "
                    f"{e.strerror}"
                ) from e
            manifest = build.run(
                name=name,
                workers=workers,
                now=now,
                allow_repeat=allow_repeat,
            )
            try:
                _check_worth_publishing(manifest, directory, replacing=directory.exists())
            except DatasetError:
                # Read to the end and found nothing worth publishing, which reading it again
                # would find too: there is nothing to carry on, and nothing worth keeping.
                remove_checkpoint(partial)
                raise
            # The checkpoint stays until the dataset is in place, so that a disk filling up or a
            # Ctrl-C while the manifest is written or the dataset moved costs the last minute of
            # reading rather than all of it. Removed from where the dataset now is: one left
            # behind there by a crash is of a version the dataset has, which the next append
            # recognises and clears away.
            manifest.save(partial)
            _publish(partial, directory, name=name, overwrite=overwrite)
            remove_checkpoint(directory)
            # After everything that can still fail: closing the shards flushes them, and
            # publishing renames them. A summary printed before those would say the build was
            # finished and then be followed by the reason it was not. Before the warnings
            # below, because it is what ends the progress line on this path.
            build.report(done=True)
        except BaseException:
            # Whatever failed, the working directory is gone once the dataset has been put in
            # place, and has no checkpoint if nothing was worth publishing.
            if has_checkpoint(partial):
                # Hours of reading, which a resume carries on from; including after a Ctrl-C,
                # which is as likely to be "not now" as "not this", and including at the very
                # end, when the last flush of the shards finds the disk full.
                _end_progress_line(progress)
                LOGGER.warning(
                    "chess-ai: the build of dataset %r stopped part way; what it had read is "
                    "kept in %s. Carry it on with 'chess-ai dataset build %s --resume', or "
                    "start again with --discard-interrupted.",
                    name,
                    partial,
                    name,
                )
            else:
                # What was written is unreadable without a manifest, and there is no checkpoint
                # to carry on from, so leaving it behind would only be rubble for the next
                # build to clear up.
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


def append_dataset(
    name: str,
    patterns: Sequence[str],
    *,
    data_dir: Path,
    max_games: int | None = None,
    progress: ProgressCallback | None = None,
    workers: int | None = None,
    now: datetime | None = None,
    allow_repeat: bool = False,
    discard_interrupted: bool = False,
    resume: bool = False,
    confirm: Confirm | None = None,
) -> Manifest:
    """Read the PGN files ``patterns`` name into dataset ``name``, as its next version.

    The sources are read through the filters, validation fraction and rating source the dataset
    was built with -- a dataset's name means one definition of data -- and what they keep is
    added after the records already there. ``max_games`` raises the dataset's cap on its total
    games, for a dataset that has reached it; it cannot lower it.

    A source the dataset already holds is refused, found by the SHA-256 of its bytes or by being
    the same Lichess month, unless ``allow_repeat`` says it is meant; see :func:`_check_repeats`.

    The manifest is written last, as a build's is, so a run reading an earlier version carries on
    undisturbed while this runs, and an append that dies leaves records past the last version that
    nothing reads, with the checkpoint it last wrote. ``resume`` carries that append on from the
    checkpoint, with the sources and maximum it was started with, which are then not given again.
    A new append on top of one asks ``confirm`` whether to cut its records off, or cuts them off
    without asking if ``discard_interrupted`` says to; with neither it is refused. Records are
    never left in the middle of a dataset: they are at its end, or they are gone.

    Raises :exc:`~chess_ai.dataset.store.DatasetError` for anything that stops the append from
    starting, and for an append that kept no games.
    """
    valid_name(name)
    _check_workers(workers)
    if resume:
        if patterns:
            raise DatasetError(
                "a resumed append reads the sources the interrupted one was started with; name none"
            )
        if discard_interrupted:
            raise DatasetError("an interrupted append can be resumed or discarded, not both")
        if max_games is not None:
            raise DatasetError("a resumed append keeps the maximum the interrupted one was given")
    sources = [] if resume else resolve_sources(patterns)
    directory = dataset_path(data_dir, name)
    with dataset_lock(data_dir, name):
        if not directory.is_dir():
            raise DatasetError(
                f"there is no dataset {name!r} in {directory} to append to; "
                "build it with 'chess-ai dataset build' first"
            )
        try:
            base = load_manifest(directory)
        except ManifestError as e:
            raise DatasetError(f"cannot append to dataset {name!r}: {e}") from e
        _check_appendable(base)
        checkpoint, unreadable = _append_checkpoint(directory, base)
        past = records_past(directory, base)
        if resume:
            if unreadable is not None:
                raise unreadable
            if checkpoint is None:
                raise DatasetError(
                    f"dataset {name!r} has records past its version {base.version} that an "
                    "interrupted append left behind, but no checkpoint to carry them on from; "
                    "append with --discard-interrupted to cut them off"
                    if past
                    else f"there is no interrupted append to dataset {name!r} to resume"
                )
            build = _Build.resuming(directory, checkpoint, progress=progress, base=base)
        else:
            if past or checkpoint is not None or unreadable is not None:
                _settle_interrupted_append(
                    base,
                    directory,
                    checkpoint,
                    past,
                    discard=discard_interrupted,
                    confirm=confirm,
                )
            filters = base.filters.model_copy(
                update={"max_games": _cap(base, max_games)},
            )
            build = _Build(
                directory=directory,
                shards=base.shards,
                sources=sources,
                rating_source=(
                    None
                    if base.rating_source == AUTO_RATING_SOURCE
                    else RatingSource[base.rating_source.upper()]
                ),
                validation_fraction=base.validation_fraction,
                progress=progress,
                filters=filters,
                base=base,
            )
        try:
            manifest = build.run(name=name, workers=workers, now=now, allow_repeat=allow_repeat)
            try:
                _check_worth_publishing(manifest, directory, replacing=False)
            except DatasetError:
                # Read to the end and found nothing worth a version, which reading it again
                # would find too. Nothing was kept, so nothing was written past the version.
                remove_checkpoint(directory)
                raise
            manifest.save(directory)
            # After the manifest: a checkpoint left behind by a crash between the two is one for
            # a version the dataset has, which the next append recognises and removes.
            remove_checkpoint(directory)
            build.report(done=True)
        except BaseException:
            if has_checkpoint(directory):
                _end_progress_line(progress)
                LOGGER.warning(
                    "chess-ai: the append to dataset %r stopped part way; what it had read is "
                    "kept past version %d, where nothing reads it. Carry it on with "
                    "'chess-ai dataset append %s --resume', or cut it off with "
                    "--discard-interrupted on the next append.",
                    name,
                    base.version,
                    name,
                )
            raise
        finally:
            _end_progress_line(progress)
    return manifest


def interrupted_builds(data_dir: Path, name: str) -> list[Path]:
    """Working directories of builds of ``name`` that stopped part way and can be carried on.

    Those with a checkpoint in them; the rest are rubble, as they always were.
    """
    return [path for path in abandoned_partials(data_dir, name) if has_checkpoint(path)]


def _build_to_resume(name: str, interrupted: Sequence[Path]) -> tuple[Path, Checkpoint]:
    """The interrupted build of ``name`` to carry on, and its checkpoint, if there is just one."""
    if not interrupted:
        raise DatasetError(f"there is no interrupted build of dataset {name!r} to resume")
    if len(interrupted) > 1:
        raise DatasetError(
            f"there are {len(interrupted)} interrupted builds of dataset {name!r}, in "
            f"{', '.join(str(path) for path in interrupted)}; remove all but the one to resume"
        )
    (partial,) = interrupted
    checkpoint = load_checkpoint(partial)
    assert checkpoint is not None, "interrupted_builds only finds directories with one"
    if checkpoint.version != 1 or checkpoint.base_created is not None:
        raise DatasetError(f"the checkpoint in {partial} is not one of a build")
    return partial, checkpoint


def _settle_interrupted_builds(
    name: str, interrupted: Sequence[Path], *, discard: bool, confirm: Confirm | None
) -> None:
    """Discard interrupted builds of ``name`` if that is what was asked for, or refuse."""
    described = "; ".join(_described_build(path) for path in interrupted)
    if not discard:
        if confirm is None:
            raise DatasetError(
                f"an interrupted build of dataset {name!r} is in {described}. Carry it on with "
                f"'chess-ai dataset build {name} --resume', or build with --discard-interrupted "
                "to throw it away and start this one"
            )
        if not confirm(
            f"An interrupted build of dataset {name!r} is in {described}.\n"
            "Discard it and start this build from scratch?"
        ):
            raise DatasetError(
                f"left the interrupted build of dataset {name!r} alone; carry it on with "
                f"'chess-ai dataset build {name} --resume'"
            )
    for path in interrupted:
        LOGGER.warning("chess-ai: discarding the interrupted build of %r in %s", name, path)
        # The checkpoint first, so that a delete which stops part way leaves rubble that is
        # reported as such, and never a checkpoint over records that are no longer all there.
        remove_checkpoint(path)
        shutil.rmtree(path, ignore_errors=True)


def _described_build(path: Path) -> str:
    """Which interrupted build is in ``path``, for a person deciding whether to keep it."""
    try:
        checkpoint = load_checkpoint(path)
    except DatasetError:
        return f"{path} (its checkpoint cannot be read)"
    assert checkpoint is not None
    return f"{path} ({_described_checkpoint(checkpoint)})"


def _described_checkpoint(checkpoint: Checkpoint) -> str:
    """What an interrupted build or append was doing, in a line."""
    names = [Path(source.path).name for source in checkpoint.sources]
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    return (
        f"started {checkpoint.started.astimezone():%Y-%m-%d %H:%M} on {shown}, "
        f"{checkpoint.games:,} games kept by its last checkpoint"
    )


def _append_checkpoint(
    directory: Path, base: Manifest
) -> tuple[Checkpoint | None, DatasetError | None]:
    """The checkpoint of an interrupted append to ``base``, or why it cannot be read.

    One for a version the dataset already has is of an append that finished and was stopped
    before it could remove it. It is of no use to anyone, and is removed here.
    """
    try:
        checkpoint = load_checkpoint(directory)
    except DatasetError as e:
        return None, e
    if checkpoint is not None and checkpoint.version <= base.version:
        remove_checkpoint(directory)
        return None, None
    return checkpoint, None


def _settle_interrupted_append(
    base: Manifest,
    directory: Path,
    checkpoint: Checkpoint | None,
    past: Sequence[Path],
    *,
    discard: bool,
    confirm: Confirm | None,
) -> None:
    """Cut off what an interrupted append left if that is what was asked for, or refuse.

    Cut off rather than written after: a dataset whose interrupted records were left in the middle
    of it would have them in every version from then on, and nothing to say which they were.
    """
    name = base.name
    if checkpoint is not None:
        what = f"an interrupted append ({_described_checkpoint(checkpoint)})"
    elif past:
        what = (
            f"records past its version {base.version} that an interrupted append left behind, "
            f"in {len(past)} shard(s) such as {past[0]}"
        )
    else:
        what = "an interrupted append whose checkpoint cannot be read"
    resumable = checkpoint is not None
    if not discard:
        if confirm is None:
            raise DatasetError(
                f"dataset {name!r} has {what}. Nothing reads its records. "
                + (
                    f"Carry it on with 'chess-ai dataset append {name} --resume', or append"
                    if resumable
                    else "Append"
                )
                + " with --discard-interrupted to cut them off and start this one from version "
                f"{base.version}"
            )
        if not confirm(
            f"Dataset {name!r} has {what}.\n"
            f"Discard its records and append these sources after version {base.version}?"
        ):
            raise DatasetError(
                f"left the interrupted append to dataset {name!r} alone"
                + (
                    f"; carry it on with 'chess-ai dataset append {name} --resume'"
                    if resumable
                    else ""
                )
            )
    LOGGER.warning(
        "chess-ai: discarding the records an interrupted append left past version %d of dataset %r",
        base.version,
        name,
    )
    # The checkpoint first, so that a cut that stops part way leaves records that can only be
    # discarded, and never a checkpoint counting records that are no longer there.
    remove_checkpoint(directory)
    cut_back(directory, base)


def _check_workers(workers: int | None) -> None:
    """Refuse a number of worker processes that cannot be, or that reads as a typo."""
    if workers is not None and workers < 1:
        raise DatasetError(f"a build needs at least one worker to read with, not {workers}")
    if workers is not None and workers > (ceiling := most_workers()):
        raise DatasetError(
            f"{workers} workers is more than this machine has any use for; the most it will "
            f"start is {ceiling}. Each one holds pieces of a PGN file while it reads them, so "
            "the memory a build takes grows with this number"
        )


def _check_appendable(base: Manifest) -> None:
    """Refuse to append to a dataset whose records this code would write differently."""
    if base.format_version != FORMAT_VERSION:
        raise DatasetError(
            f"dataset {base.name!r} is format version {base.format_version}, and this code "
            f"appends version {FORMAT_VERSION}; rebuild it to append to it"
        )
    if base.move_vocabulary_size != VOCABULARY_SIZE:
        raise DatasetError(
            f"dataset {base.name!r} was built against a move vocabulary of "
            f"{base.move_vocabulary_size} moves and this code has {VOCABULARY_SIZE}; "
            "rebuild it to append to it"
        )


def _cap(base: Manifest, max_games: int | None) -> int | None:
    """The cap on the dataset's total games an append works to: the dataset's, or a raised one.

    Refuses one that leaves no room, since an append that can keep nothing would read every source
    to the end to find that out.
    """
    if max_games is not None:
        if max_games <= base.games:
            raise DatasetError(
                f"dataset {base.name!r} already has {base.games:,} games, so a maximum of "
                f"{max_games:,} leaves no room to append any; the maximum is on the dataset's "
                "total, so give one above that"
            )
        return max_games
    cap = base.filters.max_games
    if cap is not None and base.games >= cap:
        raise DatasetError(
            f"dataset {base.name!r} is at its maximum of {cap:,} games; append with a larger "
            "--max-games to add more"
        )
    return cap


def _check_repeats(
    name: str, known: Sequence[tuple[int, SourceInfo]], new: Sequence[SourceInfo]
) -> None:
    """Refuse sources whose games are already in the dataset, or named twice in this append.

    A source is the same as another if it has the same bytes, by SHA-256, or if both are dumps of
    the same Lichess month -- which is what catches a dump appended after its own decompressed
    copy, whose bytes differ and whose games do not. ``known`` is the dataset's sources with the
    version each came in.

    A source recorded before datasets had versions has no checksum. It is recognised by its month
    when its name has one, and otherwise by hashing it where it was, if it is still there and the
    same size as the new one: identical bytes are at least the same number of them.
    """
    repeats: list[str] = []
    seen: list[tuple[str, SourceInfo]] = [
        (f"version {version}", source) for version, source in known
    ]
    for source in new:
        for where, other in seen:
            why = _same_source(source, other)
            if why is not None:
                repeats.append(f"{source.path} has {why} as {other.path}, in {where}")
                break
        seen.append(("this append" if known else "this build", source))
    if repeats:
        raise DatasetError(
            f"these sources are already in dataset {name!r}, and their games would be in it "
            f"twice: {'; '.join(repeats)}. Leave them out, or say --allow-repeat if that is meant"
        )


def _same_source(source: SourceInfo, other: SourceInfo) -> str | None:
    """How ``source`` is the same file as ``other``, or ``None`` if it is not."""
    if source.sha256 is not None and source.sha256 == other.sha256:
        return "the same contents"
    month = other.month if other.month is not None else dump_month(Path(other.path).name)
    if source.month is not None and source.month == month:
        return f"the same Lichess month, {month},"
    if other.sha256 is None and source.sha256 is not None and other.bytes == source.bytes:
        try:
            if sha256_of(Path(other.path)) == source.sha256:
                return "the same contents"
        except OSError:
            pass
    return None


def _check_worth_publishing(manifest: Manifest, directory: Path, *, replacing: bool) -> None:
    """Refuse to publish a build or append that read less than it was asked to, where that would
    lose.

    The sources are sized when a build starts and read as it reaches them, hours later, so a mount
    that drops or a sync job that rotates a directory of dumps can leave a build with nothing in
    it. Being tolerant of one bad file must not extend to publishing that:

    - if *every* source went away before saying anything, the build has no idea what it is
      missing, and that is what it is told rather than that it kept nothing, which it also did
    - if *some* source went away and the build is ``replacing`` a dataset already there, that
      source took an unknown number of its games with it, the dataset in place was built from
      more than this one could be, and a nightly ``--overwrite`` must not trade it for this. The
      remedy is to fix the source or stop naming it, not a flag that says to carry on anyway
    - if *no games at all* were kept, it does not matter why, and it does not matter whether a
      dataset of this name is already there. A dataset of nothing is not a replacement for a
      dataset of something, and it is not a first dataset either: published with a manifest and
      an exit status of zero, it is one the next stage opens and trains on. Nor is it a version:
      a run asked to train on the latest would be told it was new data. This is also the one
      that catches a source truncated between being sized and being read, which fails in no way
      at all — no error, no games, nothing to complain of

    A source that was there throughout and whose *contents* stopped making sense is none of
    these, wherever in the file that happened: those games do not exist to be missed. See
    :attr:`~chess_ai.dataset.manifest.SourceInfo.went_away`.

    Only the latest version is judged, which is what this build or append read.
    """
    latest = manifest.versions[-1]
    appending = latest.version > 1
    doing = (
        f"this append to dataset {manifest.name!r}"
        if appending
        else f"this build of dataset {manifest.name!r}"
    )
    silent = [source for source in latest.sources if source.left_nothing]
    lost = [source for source in latest.sources if source.went_away]
    if silent and len(silent) == len(latest.sources):
        raise DatasetError(
            f"none of the {len(silent)} source(s) "
            + ("appended to" if appending else "of")
            + f" dataset {manifest.name!r} could be read, so there is nothing to "
            + ("append" if appending else "build from")
            + f": {', '.join(str(source.path) for source in silent)}"
        )
    if lost and replacing:
        raise DatasetError(
            f"{len(lost)} of the {len(latest.sources)} source(s) of dataset "
            f"{manifest.name!r} could not be read whole, and the dataset already in {directory} "
            f"was built from more than this one could be: "
            f"{', '.join(str(source.path) for source in lost)}. It has been left alone; fix "
            "those sources, or leave them out to build from the rest on purpose"
        )
    if latest.games == 0:
        # The filters are the likelier culprit when they left out games, and the advice says so.
        filtered = sum(latest.filtered.values())
        what = (
            f"the filters left out all {filtered:,} games they were shown"
            if filtered
            else "the source(s) are what you meant"
        )
        if appending:
            raise DatasetError(
                f"{doing} kept no games, so there is no version {latest.version} to add; check "
                + ("whether " if filtered else "")
                + (
                    what
                    if filtered
                    else f"{what}: {', '.join(str(source.path) for source in latest.sources)}"
                )
            )
        if replacing:
            raise DatasetError(
                f"{doing} kept no games, and the dataset already "
                f"in {directory} has {_games_in(directory)}. It has been left alone; check "
                + ("whether " if filtered else "")
                + f"{what} before replacing it with nothing"
            )
        raise DatasetError(
            f"{doing} kept no games, so there is no dataset to "
            + (
                f"publish: {what}"
                if filtered
                else f"publish; check {what}: "
                f"{', '.join(str(source.path) for source in latest.sources)}"
            )
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
        if has_checkpoint(path):
            # An interrupted build, which resuming or discarding is the answer to, not this.
            continue
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


OUTCOMES: Final[tuple[str, ...]] = (
    TRAIN,
    VALIDATION,
    *(reason for reason in SkipReason),
    *(reason for reason in FilterReason),
)
"""What can become of one game that was read, numbered by place: kept in a split, or left out
for one reason or another. A worker hands back one of these per game, in the order it read them;
see :class:`_Read`."""

_OUTCOME: Final = {outcome: code for code, outcome in enumerate(OUTCOMES)}

KEPT: Final = 2
"""Outcomes below this are games kept, in :data:`TRAIN` and then :data:`VALIDATION`."""


@dataclass
class _Tally:
    """How many games a build read, what became of them, and what was in the ones it kept.

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
    filtered: Counter[FilterReason] = field(default_factory=Counter)
    not_targets: Counter[FilterReason] = field(default_factory=Counter)
    """Positions of kept games that are not trained on, by which filter left them out."""
    unexpected: str | None = None
    """The first failure nothing expected, written out, for the build to say once.

    Carried as text rather than as the exception, because it travels back from a worker process:
    an arbitrary exception may not survive being pickled, and what is wanted from it is the
    traceback anyway.
    """

    def count(self, games: np.ndarray, not_targets: np.ndarray) -> None:
        """Take kept games into the statistics: their records, and per game how many of their
        positions the mover's rating and the clock left out of training.

        Counted over arrays rather than a game at a time, because the parent counts every game
        a worker read, in one go per piece, after deciding how many of them the build keeps.
        """
        if not len(games):
            return
        for counter, column, kind in (
            (self.results, "result", Result),
            (self.time_controls, "time_control", TimeControl),
            (self.rating_sources, "rating_source", RatingSource),
        ):
            for value, count in enumerate(np.bincount(games[column])):
                if count:
                    counter[kind(value)] += int(count)
        flags = games["flags"]
        for column, unknown in (
            ("white_rating", GameFlags.WHITE_RATING_UNKNOWN),
            ("black_rating", GameFlags.BLACK_RATING_UNKNOWN),
        ):
            missing = (flags & unknown) != 0
            self.ratings_unknown += int(np.count_nonzero(missing))
            buckets, counts = np.unique(games[column][~missing], return_counts=True)
            for rating, count in zip(buckets.tolist(), counts.tolist(), strict=True):
                self.ratings[rating_bucket(rating)] += count
        rating, clock = np.asarray(not_targets, dtype=np.int64).reshape(-1, 2).sum(axis=0)
        if rating:
            self.not_targets[FilterReason.RATING] += int(rating)
        if clock:
            self.not_targets[FilterReason.CLOCK] += int(clock)

    def left_out(self, reason: SkipReason | FilterReason, count: int = 1) -> None:
        """Count games left out for ``reason``, which says whether they were broken or filtered."""
        if isinstance(reason, SkipReason):
            self.skipped[reason] += count
        else:
            self.filtered[reason] += count

    def note(self, error: BaseException) -> None:
        """Keep the first failure nothing expected, to be said out loud once."""
        if self.unexpected is None:
            self.unexpected = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )

    def to_counts(self) -> Counts:
        """This, as a checkpoint writes it down: every count by the name of what it counts."""
        return Counts(
            games_read=self.games_read,
            results=_named(self.results),
            time_controls=_named(self.time_controls),
            rating_sources=_named(self.rating_sources),
            ratings=dict(self.ratings),
            ratings_unknown=self.ratings_unknown,
            skipped=_named(self.skipped),
            filtered=_named(self.filtered),
            not_targets=_named(self.not_targets),
            unexpected=self.unexpected,
        )

    @classmethod
    def from_counts(cls, counts: Counts) -> "_Tally":
        """The tally a checkpoint wrote down."""
        return cls(
            games_read=counts.games_read,
            results=Counter({Result[key]: value for key, value in counts.results.items()}),
            time_controls=Counter(
                {TimeControl[key]: value for key, value in counts.time_controls.items()}
            ),
            rating_sources=Counter(
                {RatingSource[key]: value for key, value in counts.rating_sources.items()}
            ),
            ratings=Counter(counts.ratings),
            ratings_unknown=counts.ratings_unknown,
            skipped=Counter({SkipReason[key]: value for key, value in counts.skipped.items()}),
            filtered=Counter({FilterReason[key]: value for key, value in counts.filtered.items()}),
            not_targets=Counter(
                {FilterReason[key]: value for key, value in counts.not_targets.items()}
            ),
            unexpected=counts.unexpected,
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
        self.filtered += other.filtered
        self.not_targets += other.not_targets
        if self.unexpected is None:
            self.unexpected = other.unexpected


@dataclass(frozen=True)
class _Job:
    """One piece of one PGN file to read, and everything reading it has to be told."""

    piece: Piece
    rating_source: RatingSource | None
    validation_fraction: float
    screen: Screen = Screen()
    source_offset: int = 0
    """How many sources the dataset had before this build's, which its games' source indices
    count on from."""
    passing: int = 0
    """How many games at the start of the piece a resumed build had read already."""


@dataclass
class _Records:
    """One split's share of the games out of one piece, ready to be appended in one go."""

    games: np.ndarray
    positions: np.ndarray
    moves: np.ndarray
    ply_counts: np.ndarray
    """How many plies each game has, which is what the writer turns into offsets."""
    targets: np.ndarray
    """Which positions are training targets, one boolean each."""
    not_targets: np.ndarray
    """Per game, how many positions the rating and the clock left out of training: (games, 2)."""

    def first(self, games: int) -> "_Records":
        """The first ``games`` games of these, which is where a build that is full stops."""
        plies = int(self.ply_counts[:games].sum())
        return _Records(
            games=self.games[:games],
            positions=self.positions[:plies],
            moves=self.moves[:plies],
            ply_counts=self.ply_counts[:games],
            targets=self.targets[:plies],
            not_targets=self.not_targets[:games],
        )


@dataclass
class _Read:
    """What reading one piece produced: its records, what became of each game, and how it ended.

    ``outcomes`` is one :data:`OUTCOMES` code per game read, in the order they were read, rather
    than counts. That is what lets the parent stop at exactly the game that fills a build with a
    maximum: the games of the piece past it were read by the worker and are not this build's, so
    they are neither written nor counted, and the dataset and its manifest come out as a build
    reading one game at a time would have left them.
    """

    piece: Piece
    """The piece read, less its text if it had any: that has been read, and is not sent back."""
    outcomes: np.ndarray
    splits: dict[str, _Records]
    unexpected: str | None = None
    error: str | None = None
    went_away: bool = False
    """Whether the file went away, rather than its contents having stopped making sense."""


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

    def saved(self) -> SourceState:
        """This, as a checkpoint writes it down."""
        return SourceState(
            read=self.read,
            kept=self.kept,
            error=self.error,
            went_away=self.went_away,
            stopped=self.stopped,
        )


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
    bins: dict[str, tuple[list, ...]] = {split: ([], [], [], [], [], []) for split in SPLITS}
    outcomes: list[int] = []
    error: str | None = None
    went_away = False
    screen = job.screen
    # Applied here as well as in the parent: a worker started by spawn or forkserver rather than
    # fork inherits nothing, and a dump of millions of games has thousands of unreadable ones.
    with quiet_parser():
        reading = games_in_piece(
            job.piece, skip=lambda headers: skips_moves(headers, screen), passing=job.passing
        )
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
                # Not an outcome: the game it stopped in was never read, and is counted as
                # unreadable by whoever takes this piece, if the build still wants it.
                tally.note(e)
                # The reason only. How many games the *file* had read by then is not known
                # here -- this counts the piece -- and it is what the manifest wants, so the
                # sentence is composed in _take where the file's running total is.
                error = _described(e)
                went_away = isinstance(e, OSError)
                break
            try:
                records = game_records(
                    record,
                    source=job.piece.source + job.source_offset,
                    rating_source=(
                        rating_source_of(record) if job.rating_source is None else job.rating_source
                    ),
                    screen=screen,
                )
            except GameSkipped as skipped:
                outcomes.append(_OUTCOME[skipped.reason])
                continue
            except Exception as e:
                tally.note(e)
                outcomes.append(_OUTCOME[SkipReason.UNREADABLE])
                continue
            split = VALIDATION if split_of(records.identity, job.validation_fraction) else TRAIN
            outcomes.append(_OUTCOME[split])
            games, positions, moves, counts, targets, not_targets = bins[split]
            games.append(records.game)
            positions.append(records.positions)
            moves.append(records.moves)
            counts.append(len(records.positions))
            targets.append(records.targets)
            not_targets.append(records.not_targets)
    return _Read(
        piece=(
            replace(job.piece, text=b"", error=None)
            if isinstance(job.piece, TextPiece)
            else job.piece
        ),
        outcomes=np.asarray(outcomes, dtype=np.uint8),
        splits={
            split: _Records(
                games=np.concatenate(games),
                positions=np.concatenate(positions),
                moves=np.concatenate(moves),
                ply_counts=np.asarray(counts, dtype=np.int64),
                targets=np.concatenate(targets),
                not_targets=np.asarray(not_targets, dtype=np.int64),
            )
            for split, (games, positions, moves, counts, targets, not_targets) in bins.items()
            if games
        },
        unexpected=tally.unexpected,
        error=error,
        went_away=went_away,
    )


def _in_order(
    pool: ProcessPoolExecutor,
    jobs: Iterable[_Job],
    *,
    in_flight: int,
    wanted: Callable[[_Job], bool] | None = None,
) -> Iterator["_Read | _Job"]:
    """Every job's outcome, in the order the jobs were given, ``in_flight`` of them at a time.

    A job comes back as its :class:`_Read`, or as the job itself when its piece is larger than
    :data:`MAX_PIECE_BYTES` -- too large for a worker to hold the records of, and so for the
    caller to read where nothing has to be held. See :meth:`_Build._read_here`. The tail of a
    compressed file that could not be cut is always such a piece.

    ``jobs`` may be a generator, and for a compressed file it is one that decompresses the file
    as it goes: it is only advanced as jobs are handed out, so it holds no more than this does.

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
        if isinstance(job.piece, CompressedTail) or job.piece.bytes > MAX_PIECE_BYTES:
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
        filters: Filters,
        base: Manifest | None = None,
        resume: Checkpoint | None = None,
    ) -> None:
        """``base`` is the dataset being appended to, or ``None`` for a build of a new one.

        ``resume`` is the checkpoint of an interrupted build or append to carry on from, in
        ``directory``; see :meth:`resuming`.
        """
        self.filters = filters
        self.screen = filters.screen()
        self.base = base
        self.resume = resume
        self.writer = DatasetWriter(
            directory,
            shards,
            targets=self.screen.per_position,
            counts=(
                resume.splits if resume is not None else base.splits if base is not None else None
            ),
        )
        self._source_offset = len(base.sources) if base is not None else 0
        self._games_before = base.games if base is not None else 0
        self._positions_before = base.positions if base is not None else 0
        self._sources = list(sources)
        self._fingerprints: list[tuple[str | None, str | None]] = (
            [(source.sha256, source.month) for source in resume.sources]
            if resume is not None
            else [(None, dump_month(source.path.name)) for source in sources]
        )
        """Each source's SHA-256 and Lichess month; see :meth:`fingerprint`."""
        self._named = (
            [source.path for source in resume.sources]
            if resume is not None
            else [str(source.path) for source in sources]
        )
        """Each source's path as it was given, which the manifest records whichever directory a
        resumed build runs in; the sources themselves are read by their absolute paths."""
        self._started_at = resume.started if resume is not None else datetime.now(UTC)
        """When the build started, which a checkpoint says to tell one interrupted build from
        another; a resumed one is the build it carries on."""
        self.shards = shards
        self.rating_source = rating_source
        self.validation_fraction = validation_fraction
        self._progress = progress
        self._began = time.monotonic()
        """When the build started, which every report's ``seconds`` counts from."""
        self._started = self._began
        """When the current phase started: the checksums, then the reading. A rate and a time
        left are worked out from this, so the checksums do not count as time spent reading."""
        self._bytes_total = sum(source.bytes for source in sources)
        self._hash_total = self._bytes_total
        """The bytes the checksums are taken of, which a resumed build takes of fewer sources."""
        self._bytes_before = 0
        """Bytes of the sources already finished, which the current file's count adds to."""
        self._bytes_read = 0
        self.sources: list[SourceInfo] = []
        self.tally = _Tally()
        self._said_unexpected = False
        self.full = False
        """Whether the build has kept its ``max_games`` and reads no further."""
        self.states: dict[int, _SourceTally] = {}
        """What has become of each source the reading has reached."""
        self._position = (0, 0, 0)
        """Where the reading is: the source, the start of the piece of it being read, and how many
        games of that piece are in the dataset. What a checkpoint says to carry on from."""
        self.parallel = False
        """Whether the sources are read a piece at a time in several processes."""
        self._checkpointed_at = 0.0
        self._catching_up = resume is not None
        """Whether a resumed build is still getting back to where it was, which is not reading;
        see :meth:`_caught_up`."""
        self._games_resumed = 0
        self._bytes_resumed = 0
        if resume is not None:
            self.tally = _Tally.from_counts(resume.counts)
            self._said_unexpected = self.tally.unexpected is not None
            self.full = resume.full
            self.states = {
                index: _SourceTally(
                    read=state.read,
                    kept=state.kept,
                    error=state.error,
                    went_away=state.went_away,
                    stopped=state.stopped,
                )
                for index, state in enumerate(resume.states)
            }
            self._position = (resume.source, resume.offset, resume.passing)
            self.parallel = resume.parallel
            self._bytes_read = self._bytes_resumed = resume.bytes_read
            self._games_resumed = self.tally.games_read
            self._bytes_before = sum(source.bytes for source in sources[: resume.source])

    @classmethod
    def resuming(
        cls,
        directory: Path,
        checkpoint: Checkpoint,
        *,
        progress: ProgressCallback | None,
        base: Manifest | None = None,
    ) -> "_Build":
        """A build carrying on from ``checkpoint``, with the sources and settings it was started
        with. ``base`` is the dataset an append was appending to, or ``None`` for a build.

        Raises :exc:`~chess_ai.dataset.store.DatasetError` for a checkpoint that cannot be carried
        on from: one of another dataset, one whose records are not all on the disk, or one whose
        sources are not the size they were. The checksums are compared when the build runs.
        """
        _check_resumable(directory, checkpoint, base)
        return cls(
            directory=directory,
            shards=checkpoint.shards,
            sources=[
                Source(path=Path(source.absolute), bytes=source.bytes)
                for source in checkpoint.sources
            ],
            rating_source=(
                None
                if checkpoint.rating_source == AUTO_RATING_SOURCE
                else RatingSource[checkpoint.rating_source.upper()]
            ),
            validation_fraction=checkpoint.validation_fraction,
            progress=progress,
            filters=checkpoint.filters,
            base=base,
            resume=checkpoint,
        )

    @property
    def kept(self) -> int:
        """How many games the dataset has so far, in both splits: what it had, and what this kept.

        The dataset's total, because that is what ``max_games`` caps.
        """
        return sum(split.games for split in self.writer.splits.values())

    def run(
        self, *, name: str, workers: int | None, now: datetime | None, allow_repeat: bool
    ) -> Manifest:
        """Check the sources, read them into the dataset, and say what the dataset now is.

        The manifest that comes back is the dataset with this build's version on the end. Writing
        it is the caller's, since only the caller knows where it goes.

        A resumed build checks that its sources are the files it was reading, cuts the dataset
        back to what its checkpoint counts, and reads on from there. Its sources were checked
        for repeats when it started.
        """
        self.fingerprint()
        workers = default_workers() if workers is None else workers
        if self.resume is not None:
            assert self.writer.directory.is_dir()
            # Only now, after the checksums: a build refused for a source that changed has not
            # touched the records it would have been carried on from.
            cut_back_to(self.writer.directory, self.resume.shards, self.resume.splits)
        else:
            self.parallel = workers > 1 and self._bytes_total >= PARALLEL_FROM_BYTES
        if self.resume is None and not allow_repeat:
            # Only sources something was read from. One a build never opened, because it was full
            # before getting there, or one that went away before giving a game, has none of its
            # games in the dataset, and appending it is how they get there. One that was read and
            # whose games the filters left out does count: they would be left out again.
            known = [
                (version.version, source)
                for version in (self.base.versions if self.base is not None else [])
                for source in version.sources
                if source.games_read > 0
            ]
            _check_repeats(
                name,
                known,
                [
                    self._source_info(index, source, None)
                    for index, source in enumerate(self._sources)
                ],
            )
        with self.writer, quiet_parser():
            if self.resume is None:
                # Before a record is written, so that whatever an interrupted build leaves past
                # the end of a dataset has a checkpoint saying what it is.
                self.states[0] = _SourceTally()
                self.checkpoint(force=True)
            self.read_sources(self._sources, workers=workers)
            return self.manifest(name=name, created=now if now is not None else datetime.now(UTC))

    def fingerprint(self) -> None:
        """Take the SHA-256 of every source, which is what recognises one appended before.

        Taken before a game is read, so that a source already in the dataset is refused at the
        start rather than hours into reading it. A pass over every byte, at the speed of the disk:
        tens of seconds for a Lichess month. A file that cannot be read has none, and is counted
        as unreadable when the reading gets to it, as it always was.

        A resumed build takes them again of the sources it has still to read, and refuses to go on
        if any is not the file it was: reading on from an offset into other bytes would put games
        in the dataset that are not in any file, or the same games twice.
        """
        indices = list(range(len(self._sources)))
        if self.resume is not None:
            indices = [
                index
                for index in indices[self.resume.source :]
                if not (index == self.resume.source and self.states[index].stopped)
            ]
        if not indices:
            return
        self._hash_total = sum(self._sources[index].bytes for index in indices)
        hashed = 0
        reported = 0

        def on_read(count: int) -> None:
            nonlocal hashed, reported
            hashed += count
            if hashed - reported >= HASH_REPORT_BYTES:
                reported = hashed
                self._bytes_read = hashed
                self.report(hashing=True)

        self.report(hashing=True)
        for index in indices:
            source = self._sources[index]
            before = hashed
            try:
                digest = sha256_of(source.path, on_read)
            except OSError:
                digest = None
            hashed = before + source.bytes
            if self.resume is None:
                self._fingerprints[index] = (digest, self._fingerprints[index][1])
            elif digest != self._fingerprints[index][0]:
                raise DatasetError(
                    f"{source.path} is not the file the interrupted "
                    f"{'append' if self.base is not None else 'build'} was reading: "
                    + ("it cannot be read" if digest is None else "its contents have changed")
                    + ". Put that file back to carry it on, or discard it with "
                    "--discard-interrupted"
                )
        self._bytes_read = self._bytes_resumed
        self._hash_total = self._bytes_total
        self._started = time.monotonic()

    def _source_info(self, index: int, source: Source, state: "_SourceTally | None") -> SourceInfo:
        """What the manifest says about source ``index`` of this build, which made ``state`` of
        it; ``None`` for one it never reached."""
        state = state if state is not None else _SourceTally()
        sha256, month = self._fingerprints[index]
        return SourceInfo(
            path=self._named[index],
            bytes=source.bytes,
            games_read=state.read,
            games_kept=state.kept,
            error=state.error,
            sha256=sha256,
            month=month,
            went_away=state.went_away,
        )

    def _skip(self, headers: chess.pgn.Headers) -> bool:
        return skips_moves(headers, self.screen)

    def read_sources(self, sources: Sequence[Source], *, workers: int) -> None:
        """Read every source into the dataset, in this process or in ``workers`` of them.

        One process below :data:`PARALLEL_FROM_BYTES` of input whatever was asked for, because
        starting processes for a directory of exports costs more than reading it does. Which of
        the two was decided when the build started, and a resumed build keeps to it.
        """
        if self.parallel:
            self._read_in_parallel(sources, workers=workers)
            return
        # Before the first source, as the parallel path does, so there is a line on the screen
        # from the first moment rather than from the 128th game: _read_games reports on its own
        # cadence, and a source that gives no games never reaches it at all.
        self.report()
        resumed_at = self.resume.source if self.resume is not None else 0
        for index, source in enumerate(sources):
            state = self.states.get(index)
            if index < resumed_at or self.full:
                # Finished by the build this one resumes, or never opened and not missing
                # anything: the build had what it was asked for.
                self.sources.append(self._source_info(index, source, state))
                continue
            if state is not None and state.stopped:
                # Given up on by the build this one resumes.
                self._bytes_before += source.bytes
                self.sources.append(self._source_info(index, source, state))
                continue
            if state is None:
                state = self.states[index] = _SourceTally()
            self._position = (index, 0, state.read)
            self.checkpoint()
            self.read_source(source, index, state)

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
        found = self.states
        for index in range(len(sources)):
            found.setdefault(index, _SourceTally())
        resumed_at, offset, passing = self._position if self.resume is not None else (0, 0, 0)
        jobs: list[Iterable[_Job]] = []
        # Before the cutting rather than after it, so there is a line on the screen from the
        # first moment. It says zeros until a piece comes back, which is what is true: the
        # bytes it counts are bytes games have been read from, and cutting has read none.
        self.report(scanning=True)
        for index, source in enumerate(sources):
            if index < resumed_at or found[index].stopped:
                # Read by the build this one resumes, or given up on by it.
                continue
            at, past = (offset, passing) if index == resumed_at else (0, 0)
            try:
                if source.compressed:
                    # Cut as it is read, since it cannot be cut any other way; see text_pieces.
                    # Opened here all the same, so that a file that cannot be read at all is
                    # counted as one, just as cutting a plain file counts it.
                    _try_opening(source)
                    jobs.append(self._text_jobs(source, index, found[index], at, past))
                    continue
                pieces = _from(
                    game_ranges(source, index, CHUNK_BYTES, on_scan=self._scanning()), at, source
                )
            except DatasetError as e:
                # The file cannot be read at all, which is what PgnReader.open failing means in
                # the serial path, and gets the same answer: counted, and not the end of a build.
                found[index].error = str(e)
                found[index].went_away = True
                LOGGER.warning("chess-ai: %s; its games are not in this dataset", e)
                continue
            jobs.append(
                [self._job(piece, passing=past if piece.start == at else 0) for piece in pieces]
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
                self._handing_out(chain.from_iterable(jobs)),
                in_flight=workers * IN_FLIGHT_PER_WORKER,
                wanted=lambda job: not found[job.piece.source].stopped and not self.full,
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
                # And a build that is full drops what was still in flight when it filled up, the
                # same way: `wanted` stops anything more being handed out.
                if not state.stopped and not self.full:
                    if isinstance(outcome, _Job):
                        self._read_here(outcome, state, before=before[index])
                    else:
                        # Caught up already: as this piece was handed out, or by _read_here once
                        # the stretch before it had passed its games, whether or not any game
                        # followed them. A no-op unless the first piece had games to pass and went
                        # to a worker, which only piece sizes changed between the two runs can do.
                        self._caught_up()
                        self._take(outcome, state)
                self._bytes_read = (
                    self._bytes_total
                    if self.full
                    else before[index]
                    + (sources[index].bytes if state.stopped else outcome.piece.end)
                )
                self.report()
                if not isinstance(outcome.piece, CompressedTail):
                    # The next piece of the file starts where this one stopped. A tail has no
                    # next piece, and _read_here left the place in it as it went.
                    self._position = (index, _after(outcome.piece), 0)
                self.checkpoint()
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
            self.sources.append(self._source_info(index, source, found[index]))
        self._bytes_read = self._bytes_total

    def _handing_out(self, jobs: Iterator[_Job]) -> Iterator[_Job]:
        """``jobs``, with the reading's clock restarted as the first is handed out.

        At that moment a resumed build is back at its place: a compressed file has been
        decompressed up to it, and a plain one had nothing to catch up. Restarted any later, when
        the first piece comes back, the whole first wave of pieces -- one per worker, read in the
        same seconds -- counted as read in no time, and the time left started out near zero.

        Not for a first piece that has games to pass first, which is a stretch read in this
        process: :meth:`_read_here` restarts it once they are passed. Decided by the first job
        alone, because :func:`_in_order` pulls a batch before it reads any: a later piece of the
        batch restarting it would count the passing as reading again. The workers start on those
        pieces while the games are passed, so the restart is a little late, which is the smaller
        error.
        """
        first = True
        for job in jobs:
            if first and not job.passing:
                self._caught_up()
            first = False
            yield job

    def _job(self, piece: Piece, passing: int = 0) -> _Job:
        """What a worker is told to read ``piece``, past its first ``passing`` games."""
        return _Job(
            piece=piece,
            rating_source=self.rating_source,
            validation_fraction=self.validation_fraction,
            screen=self.screen,
            source_offset=self._source_offset,
            passing=passing,
        )

    def _text_jobs(
        self, source: Source, index: int, state: _SourceTally, at: int = 0, passing: int = 0
    ) -> Iterator[_Job]:
        """The jobs of a compressed file, cut as its text is decompressed in this process.

        Stops decompressing when the build is full or an earlier piece of the file gave up, as
        ``wanted`` stops handing out the pieces of a plain file: neither has a use for the rest.

        ``at`` is where in the text a resumed build carries on, which has to be where a piece
        starts, and ``passing`` how many games of that piece it had read. The text before it is
        decompressed and cut again, without being read; see :func:`text_pieces`.
        """
        first = True
        for piece in text_pieces(
            source,
            index,
            CHUNK_BYTES,
            MAX_PIECE_BYTES,
            stop=lambda: state.stopped or self.full,
            resume_at=at,
            on_passing=lambda: self.report(scanning=True),
        ):
            if first and at and piece.start != at:
                raise DatasetError(
                    f"{source.path} no longer cuts where the interrupted build's checkpoint says "
                    f"it was, at {at:,} bytes into its text; discard it with "
                    "--discard-interrupted"
                )
            yield self._job(piece, passing=passing if first else 0)
            first = False

    def _take(self, read: _Read, state: _SourceTally) -> None:
        """Write one worker's piece into the dataset, and take its counts into the build's.

        Only as much of it as the build still has room for: a build with a maximum stops at the
        game that fills it, and what the worker read past that is neither written nor counted.
        """
        outcomes = read.outcomes
        error = read.error
        max_games = self.filters.max_games
        if max_games is not None:
            kept_at = np.flatnonzero(outcomes < KEPT)
            room = max_games - self.kept
            if len(kept_at) >= room:
                outcomes = outcomes[: kept_at[room - 1] + 1]
                # Whatever stopped the worker came after the last game this build wanted.
                error = None
                self.full = True
        codes = np.bincount(outcomes, minlength=len(OUTCOMES))
        state.read += len(outcomes)
        state.kept += int(codes[_OUTCOME[TRAIN]] + codes[_OUTCOME[VALIDATION]])
        self.tally.games_read += len(outcomes)
        for code, count in enumerate(codes.tolist()):
            if code >= KEPT and count:
                self.tally.left_out(OUTCOMES[code], count)
        if self.tally.unexpected is None:
            self.tally.unexpected = read.unexpected
        self._say_unexpected()
        for split, records in read.splits.items():
            records = records.first(int(codes[_OUTCOME[split]]))
            self.writer.splits[split].add_games(
                records.games,
                records.positions,
                records.moves,
                records.ply_counts,
                records.targets,
            )
            self.tally.count(records.games, records.not_targets)
        if error is not None:
            # The game the piece stopped in, which the worker could not count as read.
            self.tally.skipped[SkipReason.UNREADABLE] += 1
            # Counted after the increment above, so this is the file's total and reads the same
            # as the sentences _read_games and _read_here build from theirs. A worker's own
            # count would say "after 7 games" of a file that had read six million.
            state.error = f"stopped reading after {state.read} games: {error}"
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
                "worker should hold at once, so %s is being read in this process, on one core, "
                "with the others idle until it is done. Its games are most likely not separated "
                "by a blank line.",
                job.piece.path,
                f"{job.piece.bytes:,}",
                # A compressed file cannot be cut again further on without holding its text up
                # to the next boundary, which is the thing there is no room for.
                "the rest of the file" if isinstance(job.piece, CompressedTail) else "that stretch",
            )
        reading = games_in_piece(
            job.piece,
            on_read=self._reading(before, job.piece.end),
            skip=self._skip,
            passing=job.passing,
            on_passing=lambda _: self.report(scanning=True),
        )
        here = job.passing
        while True:
            if self.full:
                return
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
            finally:
                # The first of these is what passes the games a resumed build had read, and
                # however it ends -- a game, the end of the stretch, a failure -- the build is
                # back at its place. Left to the first game, a stretch with none past its place
                # left the restart to the first piece a worker handed back, and the wave of
                # pieces read meanwhile counted as read in no time.
                self._caught_up()
            state.read += 1
            here += 1
            self.tally.games_read += 1
            state.kept += self._add_game(record, job.piece.source)
            if state.read % REPORT_EVERY == 0:
                self._bytes_read = before + position
                self.report()
            # A stretch like this can be a whole file read for hours, so it is checkpointed as it
            # goes, by how many of its games are in.
            self._position = (job.piece.source, job.piece.start, here)
            self.checkpoint()

    def read_source(self, source: Source, index: int, state: _SourceTally) -> None:
        """Read every game in one file into the dataset, whatever the file turns out to hold.

        What goes wrong in the *file* is counted and the rest of the file given up on. What goes
        wrong in writing the dataset — a full disk, a write error — is not a broken game and is
        not this method's to forgive: it ends the build, because counting it as one skipped game
        would abandon the rest of the file and then report a clean build over what was lost.
        """
        reader = PgnReader(
            source, on_read=self._reading(self._bytes_before, source.bytes), skip=self._skip
        )
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
            state.error = str(e)
            state.went_away = True
            state.stopped = True
            LOGGER.warning("chess-ai: %s; its games are not in this dataset", e)
        else:
            try:
                self._read_games(reader, index, state)
            finally:
                reader.close()
        self._bytes_before += source.bytes
        self._bytes_read = self._bytes_before
        # The file is behind us either way, however few of its games were kept, and those bytes
        # are read bytes. Without this a source that gave nothing passes in silence -- which for
        # a 20 GB file that is not PGN is the whole read with a blank terminal.
        self.report()
        self.sources.append(self._source_info(index, source, state))

    def _read_games(self, reader: PgnReader, index: int, state: _SourceTally) -> None:
        """Read an open file's games into ``state``: how many it held, how many were kept, why
        not, and whose fault it was.

        A resumed build passes over the games of the file it had read already, which is
        ``state.read`` of them.

        Only the parsing of each game is forgiven here. Everything the loop body does — writing
        the records, counting them — is outside that, because a write failure is not a broken
        game: counted as one it would abandon the rest of the file and then report a clean build
        over what was lost.
        """
        games = reader.games(state.read, on_passing=lambda _: self.report(scanning=True))
        while True:
            if self.full:
                return
            try:
                record = next(games)
            except StopIteration:
                return
            except Exception as e:
                # The file stopped, mid-game. What was read is kept, the rest of the file is not,
                # and the count says a game was lost. Whether the file went away or the chess in
                # it stopped making sense is the difference between a dataset missing an unknown
                # number of games and one that has all the games there were: an OSError is the
                # file, and anything else is what the parser found in it.
                self._note_failure(e)
                self.tally.skipped[SkipReason.UNREADABLE] += 1
                state.error = f"stopped reading after {state.read} games: {_described(e)}"
                state.went_away = isinstance(e, OSError)
                state.stopped = True
                return
            finally:
                # As in _read_here: back at its place however the passing ended.
                self._caught_up()
            state.read += 1
            self.tally.games_read += 1
            # Clamped for the same reason the cut is: the reader follows the file to its real end
            # and the source's size is the one recorded when the build started, so a file that
            # grew in between would count into the next source's share -- past the total, for
            # the last one, and the estimate below zero -- and then step back when it finished.
            self._bytes_read = self._bytes_before + min(reader.bytes_read, reader.source.bytes)
            state.kept += self._add_game(record, index)
            if state.read % REPORT_EVERY == 0:
                self.report()
            self._position = (index, 0, state.read)
            self.checkpoint()

    def _add_game(self, record: chess.pgn.Game, index: int) -> int:
        """Add one game, or count why it could not be added. Returns 1 if it was kept."""
        try:
            records = game_records(
                record,
                source=index + self._source_offset,
                rating_source=(
                    rating_source_of(record) if self.rating_source is None else self.rating_source
                ),
                screen=self.screen,
            )
        except GameSkipped as skipped:
            self.tally.left_out(skipped.reason)
            return 0
        except Exception as e:
            # Not a shape of broken game anyone has seen yet. One game is not worth abandoning
            # a build for, so it goes in the same place as the rest.
            self._note_failure(e)
            self.tally.skipped[SkipReason.UNREADABLE] += 1
            return 0
        split = VALIDATION if split_of(records.identity, self.validation_fraction) else TRAIN
        self.writer.splits[split].add_game(
            records.game, records.positions, records.moves, records.targets
        )
        self.tally.count(records.game, np.array([records.not_targets]))
        if self.filters.max_games is not None and self.kept >= self.filters.max_games:
            self.full = True
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

    def report(self, *, done: bool = False, scanning: bool = False, hashing: bool = False) -> None:
        """Tell the progress callback, if there is one, where the build has got to.

        A report that cannot be made is not a build that cannot be finished. The callback writes
        somewhere this has no opinion about — a terminal that can close, an ssh session that can
        drop, a pipe into ``head`` that can go away — and none of that is worth a day's reading.
        The first failure is said out loud and is the last one attempted.
        """
        if self._progress is None:
            return
        now = time.monotonic()
        try:
            self._progress(
                Progress(
                    games_read=self.tally.games_read,
                    # This build's, not the dataset's: what is being reported is the reading.
                    games_kept=self.kept - self._games_before,
                    positions=sum(split.positions for split in self.writer.splits.values())
                    - self._positions_before,
                    bytes_read=self._bytes_total if done else self._bytes_read,
                    bytes_total=self._hash_total if hashing else self._bytes_total,
                    # Always the whole build's, which a printer throttles on and so must never go
                    # back; the phase's own time is beside it, for the rate and the time left.
                    seconds=now - self._began,
                    reading_seconds=now - self._started if self._started != self._began else None,
                    done=done,
                    scanning=scanning,
                    hashing=hashing,
                    games_before=self._games_resumed,
                    bytes_before=0 if hashing else self._bytes_resumed,
                )
            )
        except Exception as e:
            self._progress = None
            LOGGER.warning(
                "chess-ai: progress reporting stopped, and the build carries on: %s",
                _described(e),
            )

    def _caught_up(self) -> None:
        """Start the reading's clock again, the first time a resumed build reads past its place.

        Getting back there is minutes of decompressing a dump or passing over games, in which
        nothing new is read. Timed as reading, it made the time left for a build resumed at 90%
        come out at about six times what it was. Called before the first game past the place
        is counted, so the counts the rate leaves out are still the ones it was resumed with.
        """
        if self._catching_up:
            self._catching_up = False
            self._started = time.monotonic()

    def checkpoint(self, *, force: bool = False) -> None:
        """Write down where the build has got to, if it has been :data:`CHECKPOINT_SECONDS` since
        it last did, so that it can be carried on from here if it stops.

        Called only where everything the reading has done is in the dataset, and nothing it has
        not: between pieces, between games. The records are flushed to the disk first, since a
        checkpoint counting records the disk lost is one that cannot be carried on from.
        """
        now = time.monotonic()
        if not force and now - self._checkpointed_at < CHECKPOINT_SECONDS:
            return
        self._checkpointed_at = now
        self.writer.flush()
        source, offset, passing = self._position
        Checkpoint(
            format_version=FORMAT_VERSION,
            version=self.base.version + 1 if self.base is not None else 1,
            base_created=self.base.created if self.base is not None else None,
            started=self._started_at,
            validation_fraction=self.validation_fraction,
            rating_source=self._rating_source_name,
            filters=self.filters,
            shards=self.shards,
            sources=[
                PlannedSource(
                    path=named,
                    absolute=str(item.path.absolute()),
                    bytes=item.bytes,
                    sha256=sha256,
                    month=month,
                )
                for item, named, (sha256, month) in zip(
                    self._sources, self._named, self._fingerprints, strict=True
                )
            ],
            parallel=self.parallel,
            splits={split: writer.counts for split, writer in self.writer.splits.items()},
            source=source,
            offset=offset,
            passing=passing,
            states=[
                self.states.get(index, _SourceTally()).saved()
                for index in range(min(source + 1, len(self._sources)))
            ],
            counts=self.tally.to_counts(),
            bytes_read=self._bytes_read,
            full=self.full,
        ).save(self.writer.directory)

    @property
    def _rating_source_name(self) -> str:
        """The rating source as a manifest and a checkpoint write it."""
        return AUTO_RATING_SOURCE if self.rating_source is None else self.rating_source.name.lower()

    def manifest(self, *, name: str, created: datetime) -> Manifest:
        """Everything now known about the dataset: what it was, and this build's version of it."""
        base_splits = self.base.splits if self.base is not None else {}
        version = Version(
            version=self.base.version + 1 if self.base is not None else 1,
            created=created,
            max_games=self.filters.max_games,
            sources=self.sources,
            splits={
                split: _added(self.writer.splits[split], base_splits.get(split)) for split in SPLITS
            },
            skipped={
                reason.value: self.tally.skipped[reason]
                for reason in SkipReason
                if self.tally.skipped[reason]
            },
            filtered={
                reason.value: self.tally.filtered[reason]
                for reason in FilterReason
                if self.tally.filtered[reason]
            },
            not_targets={
                reason.value: self.tally.not_targets[reason]
                for reason in FilterReason
                if self.tally.not_targets[reason]
            },
            reached_max_games=self.full,
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
        return Manifest(
            format_version=FORMAT_VERSION,
            name=name,
            created=created if self.base is None else self.base.created,
            move_vocabulary_size=VOCABULARY_SIZE,
            validation_fraction=self.validation_fraction,
            rating_source=self._rating_source_name,
            filters=self.filters,
            shards=self.shards,
            versions=[*(self.base.versions if self.base is not None else []), version],
        )


def _added(writer: SplitWriter, before: SplitCounts | None) -> SplitCounts:
    """What ``writer`` has added to a split that held ``before``."""
    before = before if before is not None else SplitCounts()
    targets = writer.targets
    return SplitCounts(
        games=writer.games - before.games,
        positions=writer.positions - before.positions,
        targets=None if targets is None else targets - before.trained_on,
    )


def _named(counts: Counter) -> dict[str, int]:
    """Counted enum members by name, as a checkpoint writes them; see :meth:`_Tally.to_counts`."""
    return {member.name: count for member, count in counts.items() if count}


def _after(piece: ByteRange | TextPiece) -> int:
    """Where the piece of a file after ``piece`` starts, in the units a checkpoint counts in."""
    return piece.end if isinstance(piece, ByteRange) else piece.stop


def _from(pieces: list[ByteRange], at: int, source: Source) -> list[ByteRange]:
    """The pieces of a plain file from ``at`` on, where a resumed build carries on reading it.

    ``at`` was where a piece started when the build was interrupted, and a file cut the same way
    is cut the same again. If no piece starts there now, something about the cutting has changed
    and carrying on would leave games out or read some twice, so it is refused.
    """
    if not at:
        return pieces
    if at < (pieces[-1].end if pieces else 0) and not any(piece.start == at for piece in pieces):
        raise DatasetError(
            f"{source.path} no longer cuts where the interrupted build's checkpoint says it was, "
            f"at byte {at:,}; discard it with --discard-interrupted"
        )
    return [piece for piece in pieces if piece.start >= at]


def _check_resumable(directory: Path, checkpoint: Checkpoint, base: Manifest | None) -> None:
    """Refuse a checkpoint that cannot be carried on from in ``directory``, before anything is
    read or changed. Whether its sources are the same files is checked by their checksums later.
    """
    doing = "append" if base is not None else "build"
    discard = f"discard the interrupted {doing} with --discard-interrupted"
    if checkpoint.format_version != FORMAT_VERSION:
        raise DatasetError(
            f"the interrupted {doing} wrote records of format version "
            f"{checkpoint.format_version}, and this code writes version {FORMAT_VERSION}; "
            + discard
        )
    if base is not None and (
        checkpoint.version != base.version + 1 or checkpoint.base_created != base.created
    ):
        raise DatasetError(
            f"the checkpoint in {directory} is of an append to another dataset {base.name!r} "
            f"than the one there now; {discard}"
        )
    sources = len(checkpoint.sources)
    if checkpoint.source > sources or len(checkpoint.states) != min(checkpoint.source + 1, sources):
        raise DatasetError(f"the checkpoint in {directory} does not add up; {discard}")
    if base is not None and any(
        counts.games < before.games
        or counts.positions < before.positions
        or counts.trained_on < before.trained_on
        for split, before in base.splits.items()
        for counts in [checkpoint.splits.get(split, SplitCounts())]
    ):
        # Cutting back to it would cut into the versions before it, which are other runs' data.
        raise DatasetError(
            f"the checkpoint in {directory} counts fewer records than version {base.version} "
            f"has; {discard}"
        )
    missing = records_missing(directory, checkpoint.shards, checkpoint.splits)
    if missing:
        raise DatasetError(
            f"the records the interrupted {doing} wrote are not all there any more, in "
            f"{missing[0]} and {len(missing) - 1} other stream(s); {discard}"
            if len(missing) > 1
            else f"the records the interrupted {doing} wrote are not all there any more, in "
            f"{missing[0]}; {discard}"
        )
    for index, source in enumerate(checkpoint.sources):
        if index < checkpoint.source or (
            index == checkpoint.source and index < sources and checkpoint.states[index].stopped
        ):
            continue
        try:
            size = Path(source.absolute).stat().st_size
        except OSError as e:
            raise DatasetError(
                f"cannot read {source.absolute}, which the interrupted {doing} had still to read: "
                f"{e.strerror}. Put it back to carry the {doing} on, or {discard}"
            ) from e
        if size != source.bytes:
            raise DatasetError(
                f"{source.absolute} is not the file the interrupted {doing} was reading: it was "
                f"{source.bytes:,} bytes and is {size:,}. Put that file back to carry the "
                f"{doing} on, or {discard}"
            )


def _try_opening(source: Source) -> None:
    """Raise :exc:`DatasetError` if ``source`` cannot be opened, as cutting a plain file would."""
    try:
        source.path.open("rb").close()
    except OSError as e:
        raise DatasetError(f"cannot read {source.path}: {e.strerror}") from e


def _described(error: Exception) -> str:
    """One line naming what went wrong, for the manifest to record against a source."""
    return f"{type(error).__name__}: {error}"


def _by_name(counts: Counter) -> dict[str, int]:
    """Counted enum members as lowercase names, in the enum's own order, dropping zeroes."""
    return {member.name.lower(): counts[member] for member in sorted(counts) if counts[member]}
