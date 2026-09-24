"""The manifest: everything about a dataset that is not one of its records.

A dataset is the thing a run trained on, so months later the only honest answer to "what
did this model learn from?" is a file that was written when the data was. The manifest is
that file: the PGN files it came from, the filters that were applied, what was kept, what
was skipped and why, when it was built, and the format and vocabulary it was built
against. It is JSON rather than anything cleverer so that it can be read without this
code, and it is written once, when the build finishes.

The statistics live here too, computed while the games streamed past. Recomputing them
means reading the whole dataset again, which for tens of gigabytes is minutes of work to
answer a question the build already knew the answer to.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_FILE: Final = "manifest.json"

TRAIN: Final = "train"
VALIDATION: Final = "validation"
SPLITS: Final = (TRAIN, VALIDATION)
"""The two splits, which are directories inside a dataset as well as names."""

RATING_BUCKET: Final = 100
"""How wide a bar of the rating histogram is."""


class ManifestError(Exception):
    """A dataset's manifest is missing, unreadable, or of a format this cannot read."""


class SourceInfo(BaseModel):
    """One PGN file a dataset was built from, and what came out of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    bytes: int
    games_read: int
    games_kept: int


class Filters(BaseModel):
    """Which games a build let through.

    Empty so far: every game in every source is kept. The filters themselves are a slice
    of their own, and this is where they will be recorded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Shards(BaseModel):
    """How many records go in one file of each stream.

    A reader needs these to find a record: every shard but the last holds exactly this
    many, so a record's shard and its place in that shard are one division away.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions_per_shard: int = Field(gt=0)
    games_per_shard: int = Field(gt=0)
    moves_per_shard: int = Field(gt=0)


class SplitCounts(BaseModel):
    """How much of a dataset one split holds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games: int = 0
    positions: int = 0


class Statistics(BaseModel):
    """What a dataset is made of, counted as it was built.

    The distributions are counted per game, except the ratings, which are counted per
    player and so have two entries per game — a game between 1500 and 2000 says
    something about both.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    results: dict[str, int] = Field(default_factory=dict)
    """Games by result, as PGN writes it: "1-0", "0-1", "1/2-1/2"."""
    time_controls: dict[str, int] = Field(default_factory=dict)
    rating_sources: dict[str, int] = Field(default_factory=dict)
    ratings: dict[str, int] = Field(default_factory=dict)
    """Players by rating, keyed by the low end of their :data:`RATING_BUCKET`-wide bucket."""
    ratings_unknown: int = 0
    """Players whose rating the file did not give."""


class Manifest(BaseModel):
    """A dataset, described. See :func:`load_manifest` and :meth:`save`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int
    name: str
    created: datetime
    move_vocabulary_size: int
    """The vocabulary the move indices are in, which a model has to agree with."""
    validation_fraction: float
    rating_source: str
    """What the build recorded as the ratings' pool, or "auto" when each game said."""
    sources: list[SourceInfo] = Field(default_factory=list)
    filters: Filters = Filters()
    shards: Shards
    splits: dict[str, SplitCounts] = Field(default_factory=dict)
    skipped: dict[str, int] = Field(default_factory=dict)
    """Games left out, counted by why; see :class:`~chess_ai.dataset.builder.SkipReason`."""
    statistics: Statistics = Statistics()

    @property
    def games(self) -> int:
        return sum(counts.games for counts in self.splits.values())

    @property
    def positions(self) -> int:
        return sum(counts.positions for counts in self.splits.values())

    @property
    def games_skipped(self) -> int:
        return sum(self.skipped.values())

    def save(self, directory: Path) -> Path:
        """Write the manifest into ``directory``, replacing any manifest already there.

        Written beside itself and moved into place, so a reader either sees the whole
        manifest of the build before or the whole manifest of this one, never half of
        either.
        """
        path = directory / MANIFEST_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path


def load_manifest(directory: Path) -> Manifest:
    """The manifest of the dataset in ``directory``.

    Raises :exc:`ManifestError` if it is not there, not readable, or written by a version
    of the format this code does not know, which is a clearer answer than records read as
    the wrong shape.
    """
    from chess_ai.dataset.records import FORMAT_VERSION

    path = directory / MANIFEST_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ManifestError(f"no dataset in {directory}: {MANIFEST_FILE} is missing") from e
    except OSError as e:
        raise ManifestError(f"cannot read {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise ManifestError(f"invalid JSON in {path}: {e}") from e
    try:
        manifest = Manifest.model_validate(data)
    except ValueError as e:
        raise ManifestError(f"invalid manifest {path}: {e}") from e
    if manifest.format_version != FORMAT_VERSION:
        raise ManifestError(
            f"{path} is dataset format version {manifest.format_version}, "
            f"and this is version {FORMAT_VERSION}; rebuild the dataset"
        )
    return manifest


def rating_bucket(rating: int) -> str:
    """The histogram bucket ``rating`` falls in, keyed by the bucket's low end."""
    return str(rating // RATING_BUCKET * RATING_BUCKET)
