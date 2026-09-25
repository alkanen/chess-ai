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

import os
import re
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final

import chess
import numpy as np

from chess_ai.dataset.manifest import SPLITS, Shards, load_manifest
from chess_ai.dataset.records import (
    GAME_DTYPE,
    MOVE_DTYPE,
    POSITION_DTYPE,
    unpack_board,
)

DATASETS_DIR: Final = "datasets"
"""Where datasets live inside the data directory."""

POSITIONS: Final = "positions"
GAMES: Final = "games"
MOVES: Final = "moves"

PARTIAL_SUFFIX: Final = ".partial"
"""What a dataset being built is called until it is finished; see :func:`partial_path`."""

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


def partial_path(data_dir: Path, name: str) -> Path:
    """Where dataset ``name`` is built before it is moved into place.

    Beside the finished dataset, so that moving it there is a rename within one filesystem, and
    named as no dataset can be — :func:`valid_name` refuses a leading dot — so that a build in
    progress can never be mistaken for a dataset.
    """
    return datasets_dir(data_dir) / f".{valid_name(name)}{PARTIAL_SUFFIX}"


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
        while the records it counts do not would be a dataset that reads as corrupt.
        """
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None


class SplitWriter:
    """One split being written: games go in whole, and it keeps track of where they went.

    The caller brings the chess — a game's positions, the moves played in them, and what
    the headers said — and this fills in the two things only a growing file knows: which
    game index the positions belong to, and where in the split the game's plies start.
    """

    def __init__(self, directory: Path, shards: Shards) -> None:
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
        self._positions.close()
        self._games.close()
        self._moves.close()


class DatasetWriter:
    """A dataset being built: both splits, and the manifest written when it is done."""

    def __init__(self, directory: Path, shards: Shards = DEFAULT_SHARDS) -> None:
        self.directory = directory
        self.shards = shards
        directory.mkdir(parents=True, exist_ok=True)
        self.splits = {split: SplitWriter(directory / split, shards) for split in SPLITS}

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for writer in self.splits.values():
            writer.close()


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
