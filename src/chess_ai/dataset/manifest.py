"""The manifest: everything about a dataset that is not one of its records.

A dataset is the thing a run trained on, so months later the only honest answer to "what
did this model learn from?" is a file that was written when the data was. The manifest is
that file: the PGN files it came from, the filters that were applied, what was kept, what
was skipped and why, when it was built, and the format and vocabulary it was built
against. It is JSON rather than anything cleverer so that it can be read without this
code, and it is written when a build or an append finishes, never part-way through one.

A dataset grows by appending, and each append is a :class:`Version`: what it read, what it
kept, and the statistics of what it kept. A version is a prefix of the dataset -- version *n*
is the first so many records of every stream, and appending only ever adds to the end -- so
the manifest of an earlier version is the same file read up to that version; see
:meth:`Manifest.at`. The totals at the top of the file are those of the latest version, worked
out from the versions rather than kept beside them, so the two can never disagree.

The statistics live here too, computed while the games streamed past. Recomputing them
means reading the whole dataset again, which for tens of gigabytes is minutes of work to
answer a question the build already knew the answer to.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from chess_ai.dataset.files import sync_directory, sync_file
from chess_ai.dataset.filters import Screen, Termination, date_bounds
from chess_ai.dataset.records import TimeControl

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
    error: str | None = None
    """What stopped this file being read whole, if anything did.

    A file that could not be opened, or that stopped making sense part-way through, leaves the
    dataset short of its games. The dataset is still usable, and this is what says it is not
    the dataset the sources asked for.
    """
    sha256: str | None = None
    """The SHA-256 of the file's bytes, taken before it was read; ``None`` before datasets had
    versions.

    What recognises a source already in a dataset, whatever it is called now. For a Lichess dump
    it is the checksum Lichess publishes, so a month streamed from Lichess and a downloaded copy
    of it are known to be the same file.
    """
    month: str | None = None
    """The Lichess month the file is a dump of, as ``YYYY-MM``, when its name says it is one.

    A second way of recognising a source: a dump's decompressed copy has other bytes and the
    same games.
    """
    went_away: bool = False
    """Whether what failed was the file rather than the PGN in it.

    A file that could not be opened, or that stopped answering while it was being read, took an
    unknown number of its games with it: a mount that drops can do so before the first game or
    after the thousandth, and how many came back first says nothing about how many were left. A
    file that is *there* and stops making sense is the opposite — that is the file's own content,
    it is counted among the skipped games, and what came before it is all there was to have.
    """

    @property
    def left_nothing(self) -> bool:
        """Whether the file went away before a single game came back out of it.

        Not "no games came back", which would be a count standing in for a reason: a file that is
        there and whose every record is junk gave nothing either, and those games do not exist to
        be missed. This is the file that went before it said anything, so what was in it is
        unknown and the build has no idea what it is missing.
        """
        return self.went_away and self.games_read == 0


class Filters(BaseModel):
    """Which games a build let through, and which of their positions it trains on.

    Every field is optional and means no filtering of its kind when it is not set, so an empty
    one is a dataset of every game in every source -- less the games no filter can let through,
    which are skipped whatever this says: the broken ones, and those ended for cheating. Fields
    that are not set are left out of the manifest, so that an empty one reads as ``{}``.

    These belong to the dataset rather than to a build, which is what lets its name mean one
    definition of data: anything filtered differently is a different dataset.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_by_name=True,
        validate_by_alias=True,
        # Written as "from", which is what a definition file calls it, in every dump.
        serialize_by_alias=True,
        # JSON writes an infinity as null, which would read back as no filter at all.
        allow_inf_nan=False,
    )

    min_rating: int | None = Field(default=None, ge=0)
    """The lowest rating of the player to move whose positions are trained on."""
    max_rating: int | None = Field(default=None, ge=0)
    """The highest rating of the player to move whose positions are trained on."""
    unknown_rating_passes: bool = False
    """Whether a player without a rating passes the rating limits. Only means anything with one."""
    time_controls: list[str] | None = None
    """The time-control classes kept, "unknown" among them if wanted; every class if not set."""
    exclude_terminations: list[str] = Field(default_factory=list)
    """The ways a game ended that leave it out; see :class:`~.filters.Termination`."""
    from_date: str | None = Field(default=None, alias="from")
    """The first day kept, as YYYY, YYYY-MM or YYYY-MM-DD; a year or month from its first day."""
    until: str | None = None
    """The last day kept, the same way; a year or month ends on its last day."""
    min_clock: float | None = Field(default=None, ge=0)
    """Seconds the player to move must have left for their position to be trained on."""
    sample: float | None = Field(default=None, gt=0, le=1)
    """The fraction of the games passing the filters that is kept, chosen by a hash of each."""
    max_games: int | None = Field(default=None, gt=0)
    """The most games kept: the first so many, in the order the sources give them."""

    @field_validator("time_controls")
    @classmethod
    def _known_time_controls(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        names = [member.name.lower() for member in TimeControl]
        unknown = [name for name in value if name not in names]
        if unknown:
            raise ValueError(f"no such time control {unknown[0]!r}; choose from {', '.join(names)}")
        if not value:
            raise ValueError("an empty list of time controls would keep no games at all")
        return [name for name in names if name in value]

    @field_validator("exclude_terminations")
    @classmethod
    def _known_terminations(cls, value: list[str]) -> list[str]:
        names = [member.value for member in Termination]
        unknown = [name for name in value if name not in names]
        if unknown:
            raise ValueError(f"no such termination {unknown[0]!r}; choose from {', '.join(names)}")
        if Termination.RULES_INFRACTION in value:
            raise ValueError(
                "games ended for a rules infraction are always left out, so there is no need to "
                "exclude them"
            )
        return [name for name in names if name in value]

    @field_validator("from_date", "until")
    @classmethod
    def _a_date(cls, value: str | None) -> str | None:
        if value is not None:
            date_bounds(value)
        return value

    @model_validator(mode="after")
    def _in_order(self) -> "Filters":
        if (
            self.min_rating is not None
            and self.max_rating is not None
            and self.min_rating > self.max_rating
        ):
            raise ValueError(
                f"the minimum rating {self.min_rating} is above the maximum {self.max_rating}"
            )
        if (
            self.from_date is not None
            and self.until is not None
            and date_bounds(self.from_date)[0] > date_bounds(self.until)[1]
        ):
            raise ValueError(f"the range from {self.from_date} until {self.until} is empty")
        return self

    @model_serializer(mode="wrap")
    def _only_what_is_set(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Against each field's own default, so that what counts as "not set" is said once, where
        # the field is. Every limit defaults to None, which a limit of 0 ("rated players only")
        # is not equal to. Pydantic decides the keys: the field's name, or its alias.
        defaults = {}
        for name, info in type(self).model_fields.items():
            default = info.get_default(call_default_factory=True)
            defaults[name] = defaults[info.serialization_alias or name] = default
        return {key: value for key, value in handler(self).items() if value != defaults[key]}

    def screen(self) -> Screen:
        """These filters in the form the reading checks them in."""
        return Screen(
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            unknown_rating_passes=self.unknown_rating_passes,
            time_controls=(
                None
                if self.time_controls is None
                else frozenset(TimeControl[name.upper()] for name in self.time_controls)
            ),
            excluded_terminations=frozenset(
                Termination(name) for name in self.exclude_terminations
            ),
            first_day=None if self.from_date is None else date_bounds(self.from_date)[0],
            last_day=None if self.until is None else date_bounds(self.until)[1],
            min_clock=self.min_clock,
            sample=self.sample,
        )


class Shards(BaseModel):
    """How many records go in one file of each stream.

    A reader needs these to find a record: every shard but the last holds exactly this
    many, so a record's shard and its place in that shard are one division away.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions_per_shard: int = Field(gt=0)
    games_per_shard: int = Field(gt=0)
    moves_per_shard: int = Field(gt=0)
    targets_per_shard: int = Field(default=8_000_000, gt=0)
    """Defaulted, because a dataset built before there were targets says nothing about them."""


class SplitCounts(BaseModel):
    """How much of a dataset one split holds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games: int = 0
    positions: int = 0
    targets: int | None = None
    """How many of the positions are training targets, or ``None`` when every one of them is.

    Set when the build had a filter on positions, and then the split has a stream of their
    indices; see :class:`~chess_ai.dataset.filters.Screen`.
    """

    @property
    def trained_on(self) -> int:
        """How many positions training draws from, which is what an epoch is."""
        return self.positions if self.targets is None else self.targets


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


class Version(BaseModel):
    """One build or append: what it read, and what it added to the end of the dataset.

    Everything here is this version's own share, not the dataset's total up to it: the counts are
    what it appended, and the statistics are of the games it kept. A version's records are the
    ones after every earlier version's, so where version *n* ends is the sum of versions 1 to *n*.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    created: datetime
    max_games: int | None = Field(default=None, gt=0)
    """The cap on the dataset's total games this version was added under."""
    sources: list[SourceInfo] = Field(default_factory=list)
    splits: dict[str, SplitCounts] = Field(default_factory=dict)
    skipped: dict[str, int] = Field(default_factory=dict)
    filtered: dict[str, int] = Field(default_factory=dict)
    not_targets: dict[str, int] = Field(default_factory=dict)
    reached_max_games: bool = False
    statistics: Statistics = Statistics()

    @property
    def games(self) -> int:
        return sum(counts.games for counts in self.splits.values())

    @property
    def positions(self) -> int:
        return sum(counts.positions for counts in self.splits.values())


_DERIVED: Final = (
    "created",
    "sources",
    "splits",
    "skipped",
    "filtered",
    "not_targets",
    "reached_max_games",
    "statistics",
)
"""The manifest's fields that are worked out from its versions, and only read from a manifest
written before there were versions."""


class Manifest(BaseModel):
    """A dataset up to one of its versions, described. See :func:`load_manifest` and :meth:`save`.

    The fields after ``shards`` are the totals of every version in ``versions``, so a manifest cut
    back to an earlier version with :meth:`at` describes exactly what that version holds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int
    name: str
    created: datetime
    """When the dataset was built, which is when its first version was."""
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
    filtered: dict[str, int] = Field(default_factory=dict)
    """Games the filters left out, counted by which; see
    :class:`~chess_ai.dataset.filters.FilterReason`."""
    not_targets: dict[str, int] = Field(default_factory=dict)
    """Positions stored but not trained on, counted by which filter left them out."""
    reached_max_games: bool = False
    """Whether the latest version stopped at ``max_games`` rather than at the end of its sources."""
    statistics: Statistics = Statistics()
    versions: list[Version] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _totals_of_the_versions(cls, data: Any) -> Any:
        """Fill in the totals from the versions, or read a manifest from before versions as one.

        A manifest written before datasets had versions is the description of one build, which is
        what version 1 is. One written since has its totals worked out here rather than trusted,
        so that cutting the list of versions short is all :meth:`at` has to do.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        filters = data.get("filters", {})
        if isinstance(filters, Filters):
            filters = filters.model_dump(by_alias=True)
        if not data.get("versions"):
            if "created" not in data:
                return data  # Not a manifest at all, which validating says better than this would.
            data["versions"] = [
                {
                    "version": 1,
                    "created": data["created"],
                    "max_games": dict(filters).get("max_games"),
                    **{field: data[field] for field in _DERIVED[1:] if field in data},
                }
            ]
            return data
        versions = [
            version if isinstance(version, Version) else Version.model_validate(version)
            for version in data["versions"]
        ]
        expected = list(range(1, len(versions) + 1))
        if [version.version for version in versions] != expected:
            raise ValueError(
                f"the versions are numbered {[version.version for version in versions]}, "
                f"not {expected}"
            )
        latest = versions[-1]
        data["versions"] = versions
        data["created"] = versions[0].created
        data["sources"] = [source for version in versions for source in version.sources]
        data["splits"] = _summed_splits(versions)
        for field in ("skipped", "filtered", "not_targets"):
            data[field] = _summed([getattr(version, field) for version in versions])
        data["reached_max_games"] = latest.reached_max_games
        data["statistics"] = _summed_statistics([version.statistics for version in versions])
        # The cap is the one version that counts: each append may raise it.
        filters = dict(filters)
        filters.pop("max_games", None)
        if latest.max_games is not None:
            filters["max_games"] = latest.max_games
        data["filters"] = filters
        return data

    @property
    def version(self) -> int:
        """Which version of the dataset this describes, counting from 1."""
        return len(self.versions)

    def at(self, version: int) -> "Manifest":
        """The dataset as it was at ``version``: its first so many records, and their totals.

        Raises :exc:`ManifestError` for a version the dataset does not have.
        """
        if not 1 <= version <= self.version:
            raise ManifestError(
                f"dataset {self.name!r} has no version {version}; it has versions 1 to "
                f"{self.version}"
                if self.version > 1
                else f"dataset {self.name!r} has no version {version}; it has only version 1"
            )
        if version == self.version:
            return self
        return Manifest.model_validate(
            {
                **self.model_dump(exclude=set(_DERIVED) | {"versions"}),
                "versions": self.versions[:version],
            }
        )

    @property
    def games(self) -> int:
        return sum(counts.games for counts in self.splits.values())

    @property
    def positions(self) -> int:
        return sum(counts.positions for counts in self.splits.values())

    @property
    def games_skipped(self) -> int:
        return sum(self.skipped.values())

    @property
    def games_filtered(self) -> int:
        return sum(self.filtered.values())

    @property
    def trained_on(self) -> int:
        """How many positions of both splits are training targets."""
        return sum(counts.trained_on for counts in self.splits.values())

    def save(self, directory: Path) -> Path:
        """Write the manifest into ``directory``, replacing any manifest already there.

        Written beside itself and moved into place, so a reader either sees the whole manifest
        of the build before or the whole manifest of this one, never half of either. Both the
        file and the directory entry are flushed to the disk, because the manifest is what says
        a build finished: a manifest that survives a power cut while the records it counts do
        not is worse than no manifest at all.
        """
        path = directory / MANIFEST_FILE
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2) + "\n")
            sync_file(f)
        os.replace(temporary, path)
        sync_directory(directory)
        return path


def load_manifest(directory: Path) -> Manifest:
    """The manifest of the dataset in ``directory``.

    Raises :exc:`ManifestError` if it is not there, not readable, or written by a version
    of the format this code does not know, which is a clearer answer than records read as
    the wrong shape.
    """
    from chess_ai.dataset.records import FORMAT_VERSION, READABLE_FORMATS

    path = directory / MANIFEST_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ManifestError(f"no dataset in {directory}: {MANIFEST_FILE} is missing") from e
    except OSError as e:
        raise ManifestError(f"cannot read {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise ManifestError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ManifestError(f"invalid manifest {path}: it is not a JSON object")
    # Before validating, because a manifest of a later format is exactly the one carrying
    # fields this version has never heard of, and "extra inputs are not permitted" is not the
    # answer to "why can this not be read?".
    version = data.get("format_version")
    if version not in READABLE_FORMATS:
        raise ManifestError(
            f"{path} is dataset format version {version!r}, and this is version "
            f"{FORMAT_VERSION}; rebuild the dataset"
        )
    try:
        return Manifest.model_validate(data)
    except ValueError as e:
        raise ManifestError(f"invalid manifest {path}: {e}") from e


def rating_bucket(rating: int) -> str:
    """The histogram bucket ``rating`` falls in, keyed by the bucket's low end."""
    return str(rating // RATING_BUCKET * RATING_BUCKET)


def _summed(counts: list[dict[str, int]]) -> dict[str, int]:
    """Counts by name added up, in the order the names first appear."""
    total: dict[str, int] = {}
    for each in counts:
        for key, count in each.items():
            total[key] = total.get(key, 0) + count
    return total


def _summed_splits(versions: list[Version]) -> dict[str, SplitCounts]:
    """Where each split ends after ``versions``: the sum of what each of them appended.

    A split's targets are counted only when some version counted them, which is all of them or
    none: whether positions are filtered is the dataset's, and no append can change it.
    """
    splits: dict[str, SplitCounts] = {}
    for version in versions:
        for split, counts in version.splits.items():
            before = splits.get(split, SplitCounts())
            targets = (
                None
                if before.targets is None and counts.targets is None
                else before.trained_on + counts.trained_on
            )
            splits[split] = SplitCounts(
                games=before.games + counts.games,
                positions=before.positions + counts.positions,
                targets=targets,
            )
    return splits


def _summed_statistics(statistics: list[Statistics]) -> Statistics:
    """The statistics of several versions' games, as if they had been counted together."""
    if len(statistics) == 1:
        return statistics[0]
    ratings = _summed([each.ratings for each in statistics])
    return Statistics(
        results=_summed([each.results for each in statistics]),
        time_controls=_summed([each.time_controls for each in statistics]),
        rating_sources=_summed([each.rating_sources for each in statistics]),
        ratings=dict(sorted(ratings.items(), key=lambda item: int(item[0]))),
        ratings_unknown=sum(each.ratings_unknown for each in statistics),
    )
