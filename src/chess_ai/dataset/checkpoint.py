"""Checkpoints: how far an interrupted build or append got, so that it can carry on from there.

A build of a Lichess month runs for hours, and what ends one part way is rarely the build's own
fault: a reboot, a full disk, a process killed for its memory, a Ctrl-C. Starting again from
nothing costs all of those hours again. So a build or append writes one of these about once a
minute, after the records it counts are flushed to the disk, and ``--resume`` cuts the dataset
back to what it counts and reads on from where it says.

It says everything the reading had in hand at that moment: the settings it was reading with, the
sources and their checksums, how many records of each split were written, which source it was
in and where in it -- always at a game boundary, or a number of games past one -- and the running
counts the manifest will be made of. Read on from there, the dataset comes out the same, to the
byte, as one that was never interrupted.

It sits beside the manifest (or where the manifest will be, for a build), and nothing that reads
a dataset looks at it. It is removed once the manifest it was leading up to is written.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chess_ai.dataset.files import sync_directory, sync_file
from chess_ai.dataset.manifest import Filters, Shards, SplitCounts
from chess_ai.dataset.store import DatasetError

CHECKPOINT_FILE: Final = "checkpoint.json"

CHECKPOINT_VERSION: Final = 1
"""The layout of the file, which a checkpoint written by other code may not share."""


class PlannedSource(BaseModel):
    """One source of the interrupted build or append, as it was when the build started."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    """The path as it was given, which is what the manifest records."""
    absolute: str
    """The same path made absolute, which is what a resume reads: it may be started from any
    directory, and a relative path would name another file or none."""
    bytes: int
    sha256: str | None = None
    month: str | None = None


class SourceState(BaseModel):
    """What the build had made of one source by the time of the checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read: int = 0
    kept: int = 0
    error: str | None = None
    went_away: bool = False
    stopped: bool = False
    """Whether the rest of the source is not this dataset's: it gave up part way."""


class Counts(BaseModel):
    """The running counts a version's statistics are made of, keyed by name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games_read: int = 0
    results: dict[str, int] = Field(default_factory=dict)
    time_controls: dict[str, int] = Field(default_factory=dict)
    rating_sources: dict[str, int] = Field(default_factory=dict)
    ratings: dict[str, int] = Field(default_factory=dict)
    ratings_unknown: int = 0
    skipped: dict[str, int] = Field(default_factory=dict)
    filtered: dict[str, int] = Field(default_factory=dict)
    not_targets: dict[str, int] = Field(default_factory=dict)
    unexpected: str | None = None


class Checkpoint(BaseModel):
    """Where an interrupted build or append got to; see the module docstring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_version: int = CHECKPOINT_VERSION
    format_version: int
    """The record format the records so far were written in."""
    version: int = Field(ge=1)
    """The version of the dataset being written: 1 for a build."""
    base_created: datetime | None = None
    """When the dataset being appended to was built, which tells it from a rebuild of the name;
    ``None`` for a build."""
    started: datetime
    """When the interrupted build or append started, for saying which one it was."""
    validation_fraction: float
    rating_source: str
    filters: Filters
    shards: Shards
    sources: list[PlannedSource]
    parallel: bool
    """Whether the sources were being read a piece at a time in several processes.

    A resumed build reads the same way whatever it is given, because the two ways cut a file at
    different places, and a file whose comments hold whole games reads differently cut elsewhere.
    """
    splits: dict[str, SplitCounts]
    """How many records each split held, all of them flushed to the disk."""
    source: int = Field(ge=0)
    """The source the reading was in: every one before it is finished."""
    offset: int = Field(ge=0)
    """Where in that source the next piece starts: a byte offset into a plain file, or into the
    decompressed text of a compressed one. Always the start of a piece, which is a game boundary."""
    passing: int = Field(default=0, ge=0)
    """How many games after ``offset`` were read already, for a stretch read a game at a time."""
    states: list[SourceState]
    """What became of every source up to and including ``source``."""
    counts: Counts
    bytes_read: int = 0
    full: bool = False
    """Whether the dataset had reached its maximum, so that nothing more is read."""

    @property
    def games(self) -> int:
        """How many games this build or append had kept."""
        return sum(state.kept for state in self.states)

    def save(self, directory: Path) -> None:
        """Write the checkpoint into ``directory`` in place of the last one, all or nothing.

        On a fixed temporary name: only the build holding the dataset's lock writes one.
        """
        path = directory / CHECKPOINT_FILE
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2) + "\n")
            sync_file(f)
        os.replace(temporary, path)
        sync_directory(directory)


def load_checkpoint(directory: Path) -> Checkpoint | None:
    """The checkpoint in ``directory``, or ``None`` if there is none.

    Raises :exc:`~chess_ai.dataset.store.DatasetError` for one that cannot be read, which is a
    reason to stop rather than to guess: what it says is what decides which records are kept.
    """
    path = directory / CHECKPOINT_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except OSError as e:
        raise DatasetError(f"cannot read {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise DatasetError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict) or data.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise DatasetError(
            f"{path} is not a checkpoint this version of chess-ai can carry on from; discard "
            "the interrupted build with --discard-interrupted"
        )
    try:
        return Checkpoint.model_validate(data)
    except ValidationError as e:
        raise DatasetError(f"invalid checkpoint {path}: {e}") from e


def remove_checkpoint(directory: Path) -> None:
    """Remove the checkpoint in ``directory``, if there is one, for good."""
    try:
        (directory / CHECKPOINT_FILE).unlink()
    except FileNotFoundError:
        return
    sync_directory(directory)


def has_checkpoint(directory: Path) -> bool:
    """Whether ``directory`` holds a checkpoint, readable or not."""
    return (directory / CHECKPOINT_FILE).is_file()
