"""Sharded record files: how a dataset is written to disk and read back at random.

A dataset is three streams per split — positions, games and moves — each an array of
fixed-size records cut into numbered shards. Shards keep any one file to a size an
operating system, a backup and a file browser are all comfortable with, and because every
shard but the last holds exactly the number of records the manifest names, finding record
*i* stays arithmetic: divide by that number for the shard, take the remainder for the
place in it. No index needs to be written, loaded or kept in step.

Reading maps the shards rather than loading them. The trainer takes random batches from a
dataset far larger than memory, and asks for tens of thousands of scattered records a
second; both are what memory mapping is for.

::

    <dataset>/manifest.json
    <dataset>/train/positions/00000.bin
    <dataset>/train/games/00000.bin
    <dataset>/train/moves/00000.bin
    <dataset>/train/targets/00000.bin
    <dataset>/validation/...

``targets`` is there only when the build filtered positions: it lists, in order, the positions
of the split that are training targets. Without it every position is one.

Appending to a dataset adds records to the end of every stream, so each version of a dataset is
a prefix of it: a reader of an earlier version reads the first so many records and never looks
past them, which is what lets it carry on while an append writes beyond. Records past the last
version are what an interrupted append leaves; nothing reads them, and :func:`records_past` is
what finds them.

The two splits are separate directories rather than a flag on each record, which is what
makes a leak impossible to write rather than merely tested for: nothing the trainer reads
from a split's directory can have come from the other.
"""

import errno
import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final
from uuid import uuid4

import chess
import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX only, and this project runs on Linux
    fcntl = None  # type: ignore[assignment]

from chess_ai.dataset.files import sync_directory, sync_file
from chess_ai.dataset.manifest import (
    SPLITS,
    Manifest,
    ManifestError,
    Shards,
    SplitCounts,
    load_manifest,
)
from chess_ai.dataset.records import (
    GAME_DTYPE,
    MOVE_DTYPE,
    POSITION_DTYPE,
    TARGET_DTYPE,
    unpack_board,
)

LOGGER = logging.getLogger(__name__)

DATASETS_DIR: Final = "datasets"
"""Where datasets live inside the data directory."""

POSITIONS: Final = "positions"
GAMES: Final = "games"
MOVES: Final = "moves"
TARGETS: Final = "targets"

PARTIAL_SUFFIX: Final = ".partial"
"""What a dataset being built is called until it is finished; see :func:`new_partial_path`."""

REPLACED_SUFFIX: Final = ".replaced"
"""What the dataset being replaced is called for the moment between two renames."""

DISCARDED_SUFFIX: Final = ".discarded"
"""What a dataset being thrown away is called once nothing wants it back.

The dataset a build replaces is renamed to this before it is deleted, so that a delete which
stops half way leaves something that reads as rubble rather than as a dataset somebody could
still have. See :func:`replaced_datasets` and :func:`discarded_datasets`.
"""

BUILD_TOKEN: Final = 8
"""Hex digits of randomness naming one build, which is plenty to never see twice."""

LOCK_SUFFIX: Final = ".lock"
"""What the file a build holds while it runs is called; see :func:`dataset_lock`."""

HELD_BY_ANOTHER: Final = frozenset(
    code
    for code in (getattr(errno, name, None) for name in ("EWOULDBLOCK", "EAGAIN", "EACCES"))
    if code is not None
)
"""What locking answers when somebody else holds the lock, rather than when it cannot be done."""

SHARD_SUFFIX: Final = ".bin"
SHARD_DIGITS: Final = 5
"""Enough for 99,999 shards, which at a million positions each is far past any dump."""

DEFAULT_SHARDS: Final = Shards(
    positions_per_shard=1_000_000,
    games_per_shard=100_000,
    moves_per_shard=8_000_000,
)
"""Shard sizes chosen so that each file lands in the tens of megabytes."""

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class DatasetError(Exception):
    """A dataset cannot be read, or a name cannot be a dataset's."""


def valid_name(name: str) -> str:
    """``name`` if it can be a dataset's, else raise :exc:`DatasetError`.

    A dataset's name is also its directory, so it has to be a plain one: no separators, no
    leading dot, nothing that would walk out of the data directory.
    """
    if not _NAME.fullmatch(name):
        raise DatasetError(
            f"invalid dataset name {name!r}: use letters, digits and . _ -, "
            "starting with a letter or digit"
        )
    return name


def datasets_dir(data_dir: Path) -> Path:
    """Where datasets live under ``data_dir``."""
    return data_dir / DATASETS_DIR


def dataset_path(data_dir: Path, name: str) -> Path:
    """The directory dataset ``name`` lives in."""
    return datasets_dir(data_dir) / valid_name(name)


@contextmanager
def dataset_lock(data_dir: Path, name: str) -> Iterator[None]:
    """Hold the right to build dataset ``name``, or refuse to start at all.

    Two builds of one dataset are not worth running: they read the same files for hours and one
    of them then throws its work away. A cron overlap, a retry started before the first was
    really dead, or two shells should hear about it at the start rather than at the end.

    The lock is a file the operating system holds for as long as the build's process lives, so a
    build killed outright leaves nothing to clean up and nothing to explain: the unlock is the
    process ending. That is why it is not a file whose existence means "locked", which would need
    a rule for telling a live build from a lock nobody released, and would leave a dataset
    unbuildable until someone deleted a file by hand.

    Where this cannot be had — anything that is not POSIX, or a filesystem that does not do
    locks — builds are not serialised, and :func:`new_partial_path` is what keeps two of them from
    writing over each other.
    """
    if fcntl is None:
        yield
        return
    root = datasets_dir(data_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        # For reading only: flock does not mind, and a lock file that somebody else created in a
        # data directory two people share can then still be locked by anyone who can read it.
        fd = os.open(root / f".{valid_name(name)}{LOCK_SUFFIX}", os.O_CREAT | os.O_RDONLY, 0o644)
    except OSError as e:
        # No lock file, for the same kind of reason as no lock: a directory somebody else owns, or
        # one mounted read-only. Whether the build itself can write is its own question to answer.
        LOGGER.warning("chess-ai: cannot lock builds in %s: %s", root, e.strerror)
        yield
        return
    unlocked: OSError | None = None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in HELD_BY_ANOTHER:
                # The filesystem does not do locks at all: an NFS export with no lock daemon, or
                # one of several FUSE mounts. Taken at its word, for the same reason the flushing
                # in files.py is — serialising builds is a convenience, not the dataset, and a
                # mount that cannot do it must not make a dataset unbuildable for good.
                unlocked = e
            else:
                raise DatasetError(
                    f"a build of dataset {name!r} is already running in {root}; "
                    "wait for it to finish"
                ) from e
        if unlocked is not None:
            LOGGER.warning(
                "chess-ai: %s cannot lock, so two builds of one dataset are not kept apart "
                "there: %s",
                root,
                unlocked.strerror,
            )
        # Outside the handler above, so that whatever the build raises is not chained onto the
        # locking error and read by whoever sees the traceback as having been caused by it.
        yield
    finally:
        # Which releases the lock: it belongs to this open file, not to the file on disk.
        os.close(fd)


def new_partial_path(data_dir: Path, name: str) -> Path:
    """A working directory for a build of dataset ``name``, to be moved into place when finished.

    Beside the finished dataset, so that moving it there is a rename within one filesystem, and
    named as no dataset can be — :func:`valid_name` refuses a leading dot — so that a build in
    progress can never be mistaken for a dataset.

    A fresh path every time, because two builds of one dataset must not share a working
    directory. They did once, and the result was not that one of them lost: each kept writing
    into files the other had deleted, and whichever finished first published its own manifest
    over the other's shards, with the counts and the records disagreeing and nothing raising.

    Random rather than counted, because a count restarts with the process that holds it: a process
    id and a counter come back together as soon as the operating system reuses the id, which on a
    small ``pid_max`` or in a container takes minutes. A name that comes back is a name a later
    build can walk into — a dead build's shards published inside a finished dataset, or a dataset
    an interrupted build set aside deleted by the build after it.

    The process id is in the name for whoever is reading it, and for nothing else: what it means is
    a question only the machine that wrote it could answer, so no decision is taken on it. See
    :func:`abandoned_partials`.
    """
    build = f"{os.getpid()}-{uuid4().hex[:BUILD_TOKEN]}"
    return datasets_dir(data_dir) / f".{valid_name(name)}.{build}{PARTIAL_SUFFIX}"


def replaced_path(partial: Path) -> Path:
    """Where the dataset being replaced by the build working in ``partial`` waits.

    Named after that build, so that two builds replacing one dataset cannot choose the same place
    to put the dataset they are replacing.
    """
    return partial.with_name(partial.name[: -len(PARTIAL_SUFFIX)] + REPLACED_SUFFIX)


def abandoned_partials(data_dir: Path, name: str) -> list[Path]:
    """Working directories of builds of ``name`` that are lying about, to be reported not removed.

    A build clears up after itself, so one of these belongs either to a build that was killed
    outright or to a build running somewhere this cannot see: another machine sharing the data
    directory, or a filesystem where :func:`dataset_lock` could not serialise anything.

    Which of the two it is cannot be told from here, and guessing wrong is expensive. A process id
    means nothing on a machine other than the one that wrote it, and a build filling one open
    shard for an hour does not touch its directory's timestamp, so neither a liveness check nor an
    age check can answer it. Deleting one that turned out to be live is the worst outcome
    available: the build carries on writing into files that are no longer there and then publishes
    a manifest counting shards that have gone. So they are named for whoever is reading and left
    exactly where they are.
    """
    return _leftovers(data_dir, name, PARTIAL_SUFFIX)


def replaced_datasets(data_dir: Path, name: str) -> list[Path]:
    """Datasets of this name set aside by a build that was interrupted while publishing.

    A dataset in one of these is a dataset nothing else will ever mention: :func:`list_datasets`
    and everything built on it skip dot-prefixed directories. Renaming one back is all it takes to
    have it again, which is worth saying to whoever lost it.
    """
    return _leftovers(data_dir, name, REPLACED_SUFFIX)


def discarded_datasets(data_dir: Path, name: str) -> list[Path]:
    """Remains of datasets this dataset's builds replaced and then failed to finish deleting.

    Nothing wants these: whatever is left is part of a dataset, without the rest of it. They are
    reported so that the disk they are taking up is somebody's decision rather than a mystery.
    """
    return _leftovers(data_dir, name, DISCARDED_SUFFIX)


def discarded_path(aside: Path) -> Path:
    """What the dataset waiting in ``aside`` is called once it is being thrown away."""
    return aside.with_name(aside.name[: -len(REPLACED_SUFFIX)] + DISCARDED_SUFFIX)


def _leftovers(data_dir: Path, name: str, suffix: str) -> list[Path]:
    """Directories of ``name``'s that a build left behind, by what they are called.

    Matched rather than sliced apart, because a dataset name may contain dots: ".d.foo.1-0.partial"
    is a build of "d.foo", not a build of "d" with something on the end, and reading it as the
    latter is how a build of one dataset came to delete the working directory of another.
    """
    root = datasets_dir(data_dir)
    if not root.is_dir():
        return []
    named = re.compile(rf"\.{re.escape(valid_name(name))}\.\d+-[0-9a-f]+{re.escape(suffix)}\Z")
    return sorted(
        entry
        for entry in root.glob(f".{name}.*{suffix}")
        if entry.is_dir() and named.fullmatch(entry.name)
    )


def list_datasets(data_dir: Path) -> list[str]:
    """The names of the datasets in ``data_dir``, in order, ignoring anything else there."""
    root = datasets_dir(data_dir)
    if not root.is_dir():
        return []
    from chess_ai.dataset.manifest import MANIFEST_FILE

    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and (entry / MANIFEST_FILE).is_file()
    )


def shard_path(directory: Path, number: int) -> Path:
    """The file holding shard ``number`` of the stream in ``directory``."""
    return directory / f"{number:0{SHARD_DIGITS}d}{SHARD_SUFFIX}"


class _StreamWriter:
    """One stream of records, written out as numbered shards of ``per_shard`` records.

    ``count`` is how many records the stream already holds, which is where an append starts: the
    last shard is carried on from its end if it is not full, and a new one started if it is.
    """

    def __init__(self, directory: Path, dtype: np.dtype, per_shard: int, count: int = 0) -> None:
        self._directory = directory
        self._dtype = dtype
        self._per_shard = per_shard
        self._count = count
        self._file: BinaryIO | None = None

    @property
    def count(self) -> int:
        """How many records have been written."""
        return self._count

    def append(self, records: np.ndarray) -> None:
        """Write ``records``, starting a new shard whenever the current one fills up."""
        assert records.dtype == self._dtype, f"expected {self._dtype}, got {records.dtype}"
        written = 0
        while written < len(records):
            in_shard = self._count % self._per_shard
            if in_shard == 0:
                self._next_shard()
            elif self._file is None:
                self._carry_on(in_shard)
            assert self._file is not None
            chunk = records[written : written + self._per_shard - in_shard]
            self._file.write(chunk.tobytes())
            self._count += len(chunk)
            written += len(chunk)

    def _next_shard(self) -> None:
        self.close()
        self._directory.mkdir(parents=True, exist_ok=True)
        # Never over a file that is there: one that is holds records past the end of the dataset,
        # and an append refuses to start over those rather than writing on top of them.
        self._file = shard_path(self._directory, self._count // self._per_shard).open("xb")

    def _carry_on(self, in_shard: int) -> None:
        """Open the last shard to append to it, after the ``in_shard`` records it holds."""
        path = shard_path(self._directory, self._count // self._per_shard)
        file = path.open("ab")
        size = file.tell()
        if size != in_shard * self._dtype.itemsize:
            file.close()
            # The caller checks for records past the end before appending anything, so this is
            # the invariant said where it matters rather than something anyone should meet.
            raise DatasetError(
                f"cannot append to dataset shard {path}: it holds {size:,} bytes and the "
                f"manifest says {in_shard * self._dtype.itemsize:,}"
            )
        self._file = file

    def flush(self) -> None:
        """Put what has been written so far on the disk, keeping the shard open to write more.

        What a checkpoint is written after: the records it counts have to be there to be resumed
        from. The directory is flushed as well, for the name of a shard started since the last.
        """
        if self._file is None:
            return
        sync_file(self._file)
        sync_directory(self._directory)

    def close(self) -> None:
        """Finish the current shard, with what was written actually on the disk.

        Flushed all the way down rather than left to the operating system, because the manifest
        written after this is what says the build finished: a manifest that survives a power cut
        while the records it counts do not would be a dataset that reads as corrupt. The shard's
        directory is flushed as well, since a file's contents surviving is no use if its name
        does not.

        The file is let go of before anything that can fail, so that a shard is never left half
        closed for a later close to trip over again — and flushing is exactly what fails on the
        full disk this exists to protect against.
        """
        if self._file is None:
            return
        file, self._file = self._file, None
        try:
            sync_file(file)
        finally:
            file.close()
        sync_directory(self._directory)


class SplitWriter:
    """One split being written: games go in whole, and it keeps track of where they went.

    The caller brings the chess — a game's positions, the moves played in them, and what
    the headers said — and this fills in the two things only a growing file knows: which
    game index the positions belong to, and where in the split the game's plies start.
    """

    def __init__(
        self,
        directory: Path,
        shards: Shards,
        *,
        targets: bool = False,
        counts: SplitCounts | None = None,
    ) -> None:
        """``counts`` is what the split already holds, for an append to carry on from."""
        counts = counts if counts is not None else SplitCounts()
        self._directory = directory
        self._positions = _StreamWriter(
            directory / POSITIONS, POSITION_DTYPE, shards.positions_per_shard, counts.positions
        )
        self._games = _StreamWriter(
            directory / GAMES, GAME_DTYPE, shards.games_per_shard, counts.games
        )
        self._moves = _StreamWriter(
            directory / MOVES, MOVE_DTYPE, shards.moves_per_shard, counts.positions
        )
        self._targets = (
            _StreamWriter(
                directory / TARGETS, TARGET_DTYPE, shards.targets_per_shard, counts.trained_on
            )
            if targets
            else None
        )

    @property
    def games(self) -> int:
        return self._games.count

    @property
    def positions(self) -> int:
        return self._positions.count

    @property
    def targets(self) -> int | None:
        """How many positions are training targets, or ``None`` if this split does not say."""
        return None if self._targets is None else self._targets.count

    def add_game(
        self,
        game: np.ndarray,
        positions: np.ndarray,
        moves: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> None:
        """Append one whole game: its record, its positions, and the moves played in them."""
        assert len(game) == 1, "a game is one record"
        self.add_games(game, positions, moves, np.array([len(positions)], dtype=np.int64), targets)

    def add_games(
        self,
        games: np.ndarray,
        positions: np.ndarray,
        moves: np.ndarray,
        ply_counts: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> None:
        """Append several whole games at once, ``ply_counts`` saying how long each one is.

        What :meth:`add_game` does, worked out for the whole lot in one go: arithmetic over an
        array rather than a Python loop per game. That is what lets a build which parsed its
        games in other processes write them here without the writing becoming the slow part.

        This is the only place the two things only a growing file knows are decided, so a caller
        can hand over what it worked out for itself -- the chess -- and never where it went.

        ``targets`` is one boolean per position, saying which are trained on, and is written only
        by a split that keeps a target stream; ``None`` means every position is one.
        """
        assert len(games) == len(ply_counts), "a count for every game"
        assert len(positions) == len(moves), "every position has the move played in it"
        assert int(np.sum(ply_counts)) == len(positions), "the counts have to add up"
        # An exclusive running total: where each game's plies start, counted from the first of them.
        games["ply_offset"] = self._positions.count + (np.cumsum(ply_counts) - ply_counts)
        games["ply_count"] = ply_counts
        positions["game"] = self._games.count + np.repeat(np.arange(len(games)), ply_counts)
        if self._targets is not None:
            chosen = np.flatnonzero(targets) if targets is not None else np.arange(len(positions))
            self._targets.append((self._positions.count + chosen).astype(TARGET_DTYPE))
        self._positions.append(positions)
        self._moves.append(moves)
        self._games.append(games)

    def flush(self) -> None:
        """Put every stream's records so far on the disk; see :meth:`_StreamWriter.flush`."""
        for stream in (self._positions, self._games, self._moves, self._targets):
            if stream is not None:
                stream.flush()

    @property
    def counts(self) -> SplitCounts:
        """How much this split holds now, in the form a manifest or a checkpoint says it."""
        return SplitCounts(games=self.games, positions=self.positions, targets=self.targets)

    def close(self) -> None:
        """Close all three streams, whatever any one of them does on the way.

        Closing flushes, and flushing is what fails when the disk fills up. One stream failing
        must not leave the other two with their files open and unflushed.
        """
        with ExitStack() as stack:
            for stream in (self._positions, self._games, self._moves, self._targets):
                if stream is not None:
                    stack.callback(stream.close)
        sync_directory(self._directory)


class DatasetWriter:
    """A dataset being built or appended to: both splits, and the manifest written when it is done.

    ``counts`` is how much each split of the directory already holds, which this writes after: the
    dataset being appended to, or what an interrupted build or append had written when it was
    last checkpointed. Without them this is a new build, into a directory of its own.
    """

    def __init__(
        self,
        directory: Path,
        shards: Shards = DEFAULT_SHARDS,
        *,
        targets: bool = False,
        counts: Mapping[str, SplitCounts] | None = None,
    ) -> None:
        self.directory = directory
        self.shards = shards
        if counts is None:
            # Never a directory that is already there: one that is belongs to another build, and
            # writing into it publishes its shards inside this dataset. The builder checks first,
            # so this is the invariant said where it belongs rather than a condition anyone should
            # hit.
            directory.mkdir(parents=True, exist_ok=False)
        elif not directory.is_dir():
            raise DatasetError(f"no dataset to append to in {directory}")
        self.splits = {
            split: SplitWriter(
                directory / split,
                shards,
                targets=targets,
                counts=counts.get(split) if counts is not None else None,
            )
            for split in SPLITS
        }

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
            return
        # Already failing. What closing has to say now is noise beside what went wrong, and
        # letting it raise would put a disk error in the place of the Ctrl-C that caused it.
        with suppress(Exception):
            self.close()

    def flush(self) -> None:
        """Put every record written so far on the disk, to checkpoint them."""
        for writer in self.splits.values():
            writer.flush()
        sync_directory(self.directory)

    def close(self) -> None:
        """Close every split, whatever any one of them does; see :meth:`SplitWriter.close`."""
        with ExitStack() as stack:
            for writer in self.splits.values():
                stack.callback(writer.close)
        sync_directory(self.directory)


class _StreamReader:
    """One stream of records, read out of its shards without loading them."""

    def __init__(
        self,
        directory: Path,
        dtype: np.dtype,
        per_shard: int,
        count: int,
        still_ours: Callable[[], bool] | None = None,
    ) -> None:
        self._directory = directory
        self._dtype = dtype
        self._per_shard = per_shard
        self._count = count
        self._still_ours = still_ours
        """Asked whether the dataset on the disk is still the one being read, when a shard holds
        more than this reader expected; see :meth:`_shard`."""
        self._shards: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self._count

    def record(self, index: int) -> np.void:
        """Record ``index``, as a view into the mapped shard it lives in."""
        index = self._checked(index)
        shard, offset = divmod(index, self._per_shard)
        return self._shard(shard)[offset]

    def gather(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """Records ``indices``, in the order asked for, as one array.

        This is the trainer's batch: the indices are scattered, so they are taken shard by
        shard, and what comes back is a copy rather than a view into several files.
        """
        wanted = np.asarray(indices, dtype=np.int64).reshape(-1)
        wanted = np.where(wanted < 0, wanted + self._count, wanted)
        if len(wanted) and (wanted.min() < 0 or wanted.max() >= self._count):
            raise IndexError(
                f"{self._directory.name} index out of range: "
                f"{len(self)} records in {self._directory}"
            )
        out = np.empty(len(wanted), dtype=self._dtype)
        shards, offsets = np.divmod(wanted, self._per_shard)
        for shard in np.unique(shards):
            picked = shards == shard
            out[picked] = self._shard(int(shard))[offsets[picked]]
        return out

    def span(self, start: int, count: int) -> np.ndarray:
        """``count`` records from ``start`` on, which is how a game's own plies are read."""
        if start < 0 or count < 0 or start + count > self._count:
            raise IndexError(
                f"{self._directory.name} span {start}..{start + count} is outside "
                f"the {self._count} records in {self._directory}"
            )
        out = np.empty(count, dtype=self._dtype)
        taken = 0
        while taken < count:
            shard, offset = divmod(start + taken, self._per_shard)
            chunk = min(count - taken, self._per_shard - offset)
            out[taken : taken + chunk] = self._shard(shard)[offset : offset + chunk]
            taken += chunk
        return out

    def _checked(self, index: int) -> int:
        if -self._count <= index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(
                f"{self._directory.name} index {index} is outside the {self._count} "
                f"records in {self._directory}"
            )
        return index

    def _shard(self, number: int) -> np.memmap:
        mapped = self._shards.get(number)
        if mapped is None:
            path = shard_path(self._directory, number)
            # Every shard but the last is full, and the last holds at least the rest of the count.
            # It may hold more: what later versions appended, or what an append is writing right
            # now, and neither is this reader's. Only the records it counts are mapped, so what
            # lies past them -- a half-written record included -- is never looked at.
            expected = min(self._per_shard, self._count - number * self._per_shard)
            try:
                size = path.stat().st_size
            except OSError as e:
                raise DatasetError(f"cannot read dataset shard {path}: {e.strerror}") from e
            held = size // self._dtype.itemsize
            # A shard holding fewer is not the one the manifest describes: a copy cut short, a
            # build that ran out of disk, or a rebuild swapped in under a reader still holding the
            # old manifest. Left to numpy, a short one either raises a bare IndexError or, worse,
            # repeats the records it has into the places of the ones it has not. One holding more
            # is what appending looks like, and also what a rebuild can look like, which is told
            # apart by asking whether the dataset on the disk is still this one.
            if held < expected or (
                held > expected and self._still_ours is not None and not self._still_ours()
            ):
                raise DatasetError(
                    f"cannot read dataset shard {path}: it holds {held} records and "
                    f"the manifest says {expected}; the dataset is damaged, or was rebuilt "
                    "while being read"
                )
            try:
                mapped = np.memmap(path, dtype=self._dtype, mode="r", shape=(expected,))
            except OSError as e:
                raise DatasetError(f"cannot read dataset shard {path}: {e.strerror}") from e
            except ValueError as e:
                # The file shrank between being measured and being mapped. numpy raises
                # ValueError for it rather than OSError, and a traceback is no way to report it.
                raise DatasetError(f"cannot read dataset shard {path}: {e}") from e
            self._shards[number] = mapped
        return mapped

    def close(self) -> None:
        """Let go of every shard this has mapped.

        Each mapping holds a file descriptor for as long as it is kept, so a process that reads
        one dataset after another — a sweep over configs, or the web server — would otherwise
        collect one per shard it ever touched and never give any back.

        The mappings are dropped rather than closed, because a record handed out is a view into
        one: closing a mapping out from under a caller's record would be a way to crash the
        process, while dropping the last reference to it cannot be.
        """
        self._shards.clear()


class SplitReader:
    """Random access to one split's records, for the trainer and for looking at games.

    ``len`` is the number of positions, because a position is what a training example is
    made of. A position record names its game, and a game record names the run of plies
    that are its own, so either can be followed to the other.
    """

    def __init__(
        self,
        directory: Path,
        shards: Shards,
        counts_games: int,
        counts_positions: int,
        counts_targets: int | None = None,
        still_ours: Callable[[], bool] | None = None,
    ):
        self._positions = _StreamReader(
            directory / POSITIONS,
            POSITION_DTYPE,
            shards.positions_per_shard,
            counts_positions,
            still_ours,
        )
        self._games = _StreamReader(
            directory / GAMES, GAME_DTYPE, shards.games_per_shard, counts_games, still_ours
        )
        self._moves = _StreamReader(
            directory / MOVES, MOVE_DTYPE, shards.moves_per_shard, counts_positions, still_ours
        )
        self._targets = (
            None
            if counts_targets is None
            else _StreamReader(
                directory / TARGETS,
                TARGET_DTYPE,
                shards.targets_per_shard,
                counts_targets,
                still_ours,
            )
        )

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def targets(self) -> int:
        """How many positions are training targets: all of them, unless the build filtered some."""
        return len(self._positions) if self._targets is None else len(self._targets)

    def target_positions(self, ordinals: np.ndarray | list[int]) -> np.ndarray:
        """The position indices of training targets ``ordinals``, counting targets from 0.

        What training and validation draw from: the n-th target is the n-th position when every
        position is one, and the n-th entry of the target stream when the build filtered some.
        """
        wanted = np.asarray(ordinals, dtype=np.int64).reshape(-1)
        if self._targets is None:
            if len(wanted) and (wanted.min() < 0 or wanted.max() >= len(self._positions)):
                raise IndexError(f"target index out of range: {len(self._positions)} targets")
            return wanted
        return self._targets.gather(wanted).astype(np.int64)

    @property
    def games(self) -> int:
        """How many games this split holds."""
        return len(self._games)

    def position(self, index: int) -> np.void:
        """Position ``index``, as a :data:`~chess_ai.dataset.records.POSITION_DTYPE` record."""
        return self._positions.record(index)

    def positions(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """Positions ``indices`` as one array, which is what a training batch is."""
        return self._positions.gather(indices)

    def position_history(self, indices: np.ndarray | list[int], history: int) -> np.ndarray:
        """Positions ``indices``, each followed by the ``history`` positions before it in its game.

        Shaped (len(indices), 1 + history): column 0 is what :meth:`positions` returns, and
        column ``k`` is the position ``k`` plies earlier. Where a game had not been going that
        long the record is blank — zero bytes, which is an empty board — rather than the tail
        of whichever game happens to be stored in front of it.

        A game's positions are stored end to end in the order they were played, so going back a
        ply is going back a record, and a position's own ``ply`` says how far back its game goes.
        """
        wanted = np.asarray(indices, dtype=np.int64).reshape(-1)
        wanted = np.where(wanted < 0, wanted + len(self), wanted)
        out = np.zeros((len(wanted), history + 1), dtype=POSITION_DTYPE)
        out[:, 0] = self._positions.gather(wanted)
        if history:
            back = np.arange(1, history + 1)
            played = out[:, 0]["ply"][:, None] >= back
            earlier = out[:, 1:]
            earlier[played] = self._positions.gather((wanted[:, None] - back)[played])
        return out

    def game(self, index: int) -> np.void:
        """Game ``index``, as a :data:`~chess_ai.dataset.records.GAME_DTYPE` record."""
        return self._games.record(index)

    def game_records(self, start: int, count: int) -> np.ndarray:
        """``count`` game records from game ``start`` on, which is one page of a list of games."""
        return self._games.span(start, count)

    def game_positions(self, index: int) -> np.ndarray:
        """Every position of game ``index``, in the order it was played."""
        game = self.game(index)
        return self._positions.span(int(game["ply_offset"]), int(game["ply_count"]))

    def move_sequence(self, index: int) -> np.ndarray:
        """Game ``index``'s moves as vocabulary indices, which is what a sequence model reads."""
        game = self.game(index)
        return self._moves.span(int(game["ply_offset"]), int(game["ply_count"]))

    def board(self, index: int) -> chess.Board:
        """Position ``index`` as a board, for looking at a dataset rather than training on it."""
        return unpack_board(self.position(index))

    def close(self) -> None:
        """Let go of the shards this split has mapped; see :meth:`_StreamReader.close`."""
        for stream in (self._positions, self._games, self._moves, self._targets):
            if stream is not None:
                stream.close()


class Dataset:
    """A built dataset at one of its versions: its manifest, and its splits to read records from.

    ``version`` is the latest when it is not given. The manifest is the one of that version, so
    everything read through this -- counts, statistics, records -- is what that version holds,
    whatever has been appended since.
    """

    def __init__(self, directory: Path, version: int | None = None) -> None:
        self.directory = directory
        latest = load_manifest(directory)
        self.manifest = latest if version is None else latest.at(version)
        self.latest_version = latest.version
        """The newest version the dataset had when this was opened."""
        self._splits = {
            split: SplitReader(
                directory / split,
                self.manifest.shards,
                counts.games,
                counts.positions,
                counts.targets,
                still_ours=self._still_ours,
            )
            for split, counts in self.manifest.splits.items()
        }

    def _still_ours(self) -> bool:
        """Whether the dataset in this directory is still this one, with this version in it.

        A rebuild under the same name was created at another time, and one that has fewer
        versions than this reads has lost the one being read.
        """
        try:
            now = load_manifest(self.directory)
        except ManifestError:
            return False
        return now.created == self.manifest.created and now.version >= self.manifest.version

    @property
    def splits(self) -> list[str]:
        return list(self._splits)

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Let go of every shard of every split, and of the file descriptors they hold.

        A dataset read to the end of a training run does not need this — the process is ending —
        but anything long-lived that opens datasets in turn does.
        """
        for split in self._splits.values():
            split.close()

    def __getitem__(self, split: str) -> SplitReader:
        try:
            return self._splits[split]
        except KeyError:
            raise DatasetError(
                f"{self.directory} has no {split!r} split; it has {', '.join(self.splits)}"
            ) from None


def open_dataset(name: str, *, data_dir: Path, version: int | None = None) -> Dataset:
    """The dataset called ``name`` in ``data_dir`` at ``version``, ready to read records from.

    The latest version when ``version`` is not given.
    """
    return Dataset(dataset_path(data_dir, name), version)


_STREAMS: Final = (
    (POSITIONS, POSITION_DTYPE, "positions_per_shard", "positions"),
    (GAMES, GAME_DTYPE, "games_per_shard", "games"),
    (MOVES, MOVE_DTYPE, "moves_per_shard", "positions"),
    (TARGETS, TARGET_DTYPE, "targets_per_shard", "trained_on"),
)
"""Every stream a split can have: its directory, its records, its shard size, and its count."""


def _stream_ends(
    directory: Path, shards: Shards, splits: Mapping[str, SplitCounts]
) -> Iterator[tuple[Path, np.dtype, int, int]]:
    """Every stream of the dataset in ``directory``, with where it ends after ``splits``.

    Each comes with its record type, its shard size and how many records ``splits`` counts in
    it. A target stream is counted only when the dataset keeps one; otherwise none of it is the
    dataset's, and anything in it is past the end.
    """
    for split in SPLITS:
        counts = splits.get(split, SplitCounts())
        for stream, dtype, per_shard, counted in _STREAMS:
            count = 0 if stream == TARGETS and counts.targets is None else getattr(counts, counted)
            yield directory / split / stream, dtype, getattr(shards, per_shard), count


def _shards_in(stream: Path) -> dict[int, Path]:
    """The shard files of the stream in ``stream``, by number."""
    if not stream.is_dir():
        return {}
    named = re.compile(rf"(\d{{{SHARD_DIGITS}}}){re.escape(SHARD_SUFFIX)}")
    return {
        int(found[1]): entry
        for entry in stream.iterdir()
        if (found := named.fullmatch(entry.name)) is not None
    }


def records_past(directory: Path, manifest: Manifest) -> list[Path]:
    """The shards of the dataset in ``directory`` that hold more than ``manifest`` counts.

    See :func:`records_past_counts`, which this is for a manifest's counts.
    """
    return records_past_counts(directory, manifest.shards, manifest.splits)


def records_past_counts(
    directory: Path, shards: Shards, splits: Mapping[str, SplitCounts]
) -> list[Path]:
    """The shards of the dataset in ``directory`` that hold more than ``splits`` counts.

    Which is what an append that was interrupted leaves: what it wrote is on the disk and its
    manifest is not, so nothing reads it. Only sizes are looked at, which is a few hundred
    ``stat`` calls for a large dataset.

    A shard past the last one counted is past the end however little it holds, an empty one
    included: an append killed between starting a shard and writing to it leaves just the name,
    and the next append starting that shard would find it taken.
    """
    past = []
    for stream, dtype, per_shard, count in _stream_ends(directory, shards, splits):
        last, in_last = divmod(count, per_shard)
        for number, path in sorted(_shards_in(stream).items()):
            if number < last:
                continue
            if number > last or in_last == 0 or path.stat().st_size > in_last * dtype.itemsize:
                past.append(path)
    return past


def records_missing(
    directory: Path, shards: Shards, splits: Mapping[str, SplitCounts]
) -> list[Path]:
    """The streams of the dataset in ``directory`` that hold fewer records than ``splits`` counts.

    Which a checkpoint's counts never should: they are written after the records are flushed. One
    that does is a dataset that was damaged or copied short since, and carrying on from it would
    write records after a gap. Only sizes are looked at, as :func:`records_past_counts` does.
    """
    short = []
    for stream, dtype, per_shard, count in _stream_ends(directory, shards, splits):
        shards_in = _shards_in(stream)
        for number in range(-(-count // per_shard)):
            expected = min(per_shard, count - number * per_shard) * dtype.itemsize
            path = shards_in.get(number)
            if path is None or path.stat().st_size < expected:
                short.append(stream)
                break
    return short


def cut_back(directory: Path, manifest: Manifest) -> None:
    """Throw away every record of the dataset in ``directory`` that ``manifest`` does not count.

    See :func:`cut_back_to`, which this is for a manifest's counts.
    """
    cut_back_to(directory, manifest.shards, manifest.splits)


def cut_back_to(directory: Path, shards: Shards, splits: Mapping[str, SplitCounts]) -> None:
    """Throw away every record of the dataset in ``directory`` that ``splits`` does not count.

    Shards past the end are removed and the last one counted is cut down to its records, so the
    dataset is exactly what ``splits`` describes again, to the byte. Every change is flushed
    before this returns, so that an append which starts next never finds them back.
    """
    for stream, dtype, per_shard, count in _stream_ends(directory, shards, splits):
        last, in_last = divmod(count, per_shard)
        for number, path in sorted(_shards_in(stream).items(), reverse=True):
            if number > last or (number == last and in_last == 0):
                path.unlink()
            elif number == last and path.stat().st_size > in_last * dtype.itemsize:
                with path.open("r+b") as file:
                    file.truncate(in_last * dtype.itemsize)
                    sync_file(file)
        if stream.is_dir():
            sync_directory(stream)
