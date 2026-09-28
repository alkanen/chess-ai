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
    <dataset>/validation/...

The two splits are separate directories rather than a flag on each record, which is what
makes a leak impossible to write rather than merely tested for: nothing the trainer reads
from a split's directory can have come from the other.
"""

import errno
import logging
import os
import re
from collections.abc import Iterator
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
from chess_ai.dataset.manifest import SPLITS, Shards, load_manifest
from chess_ai.dataset.records import (
    GAME_DTYPE,
    MOVE_DTYPE,
    POSITION_DTYPE,
    unpack_board,
)

LOGGER = logging.getLogger(__name__)

DATASETS_DIR: Final = "datasets"
"""Where datasets live inside the data directory."""

POSITIONS: Final = "positions"
GAMES: Final = "games"
MOVES: Final = "moves"

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
    """One stream of records, written out as numbered shards of ``per_shard`` records."""

    def __init__(self, directory: Path, dtype: np.dtype, per_shard: int) -> None:
        self._directory = directory
        self._dtype = dtype
        self._per_shard = per_shard
        self._count = 0
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
            assert self._file is not None
            chunk = records[written : written + self._per_shard - in_shard]
            self._file.write(chunk.tobytes())
            self._count += len(chunk)
            written += len(chunk)

    def _next_shard(self) -> None:
        self.close()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._file = shard_path(self._directory, self._count // self._per_shard).open("wb")

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

    def __init__(self, directory: Path, shards: Shards) -> None:
        self._directory = directory
        self._positions = _StreamWriter(
            directory / POSITIONS, POSITION_DTYPE, shards.positions_per_shard
        )
        self._games = _StreamWriter(directory / GAMES, GAME_DTYPE, shards.games_per_shard)
        self._moves = _StreamWriter(directory / MOVES, MOVE_DTYPE, shards.moves_per_shard)

    @property
    def games(self) -> int:
        return self._games.count

    @property
    def positions(self) -> int:
        return self._positions.count

    def add_game(self, game: np.ndarray, positions: np.ndarray, moves: np.ndarray) -> None:
        """Append one whole game: its record, its positions, and the moves played in them."""
        assert len(game) == 1, "a game is one record"
        assert len(positions) == len(moves), "every position has the move played in it"
        game["ply_offset"] = self._positions.count
        game["ply_count"] = len(positions)
        positions["game"] = self._games.count
        self._positions.append(positions)
        self._moves.append(moves)
        self._games.append(game)

    def close(self) -> None:
        """Close all three streams, whatever any one of them does on the way.

        Closing flushes, and flushing is what fails when the disk fills up. One stream failing
        must not leave the other two with their files open and unflushed.
        """
        with ExitStack() as stack:
            for stream in (self._positions, self._games, self._moves):
                stack.callback(stream.close)
        sync_directory(self._directory)


class DatasetWriter:
    """A dataset being built: both splits, and the manifest written when it is done."""

    def __init__(self, directory: Path, shards: Shards = DEFAULT_SHARDS) -> None:
        self.directory = directory
        self.shards = shards
        # Never a directory that is already there: one that is belongs to another build, and
        # writing into it publishes its shards inside this dataset. The builder checks first, so
        # this is the invariant said where it belongs rather than a condition anyone should hit.
        directory.mkdir(parents=True, exist_ok=False)
        self.splits = {split: SplitWriter(directory / split, shards) for split in SPLITS}

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

    def close(self) -> None:
        """Close every split, whatever any one of them does; see :meth:`SplitWriter.close`."""
        with ExitStack() as stack:
            for writer in self.splits.values():
                stack.callback(writer.close)
        sync_directory(self.directory)


class _StreamReader:
    """One stream of records, read out of its shards without loading them."""

    def __init__(self, directory: Path, dtype: np.dtype, per_shard: int, count: int) -> None:
        self._directory = directory
        self._dtype = dtype
        self._per_shard = per_shard
        self._count = count
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
            try:
                mapped = np.memmap(path, dtype=self._dtype, mode="r")
            except OSError as e:
                raise DatasetError(f"cannot read dataset shard {path}: {e.strerror}") from e
            except ValueError as e:
                # A file that is not a whole number of records, which is what a truncated copy
                # or a build that ran out of disk leaves behind. numpy raises ValueError for it
                # rather than OSError, and a traceback is no way to report a damaged dataset.
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

    def __init__(self, directory: Path, shards: Shards, counts_games: int, counts_positions: int):
        self._positions = _StreamReader(
            directory / POSITIONS, POSITION_DTYPE, shards.positions_per_shard, counts_positions
        )
        self._games = _StreamReader(
            directory / GAMES, GAME_DTYPE, shards.games_per_shard, counts_games
        )
        self._moves = _StreamReader(
            directory / MOVES, MOVE_DTYPE, shards.moves_per_shard, counts_positions
        )

    def __len__(self) -> int:
        return len(self._positions)

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

    def game(self, index: int) -> np.void:
        """Game ``index``, as a :data:`~chess_ai.dataset.records.GAME_DTYPE` record."""
        return self._games.record(index)

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
        for stream in (self._positions, self._games, self._moves):
            stream.close()


class Dataset:
    """A built dataset: its manifest, and its splits to read records from."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = load_manifest(directory)
        self._splits = {
            split: SplitReader(
                directory / split,
                self.manifest.shards,
                counts.games,
                counts.positions,
            )
            for split, counts in self.manifest.splits.items()
        }

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


def open_dataset(name: str, *, data_dir: Path) -> Dataset:
    """The dataset called ``name`` in ``data_dir``, ready to read records from."""
    return Dataset(dataset_path(data_dir, name))
