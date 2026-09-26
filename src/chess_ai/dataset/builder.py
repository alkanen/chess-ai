"""Building a dataset: PGN files in, a versioned directory of records and a manifest out.

The build streams. It never holds more than one game, because the input is measured in tens
of gigabytes and the output can be larger, and it writes records as it goes rather than
gathering them up. That is also why nothing here counts games first: the time remaining comes
from bytes read, which is known without a first pass.

It also never stops. Every way one game can be wrong is one game skipped and counted, not a
build abandoned four hours in, and the manifest says afterwards how many were lost to what.

The manifest is written last, which makes it the mark of a finished build: a build that died
partway leaves records nothing will read, because there is no manifest pointing at them.
"""

import logging
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import chess.pgn

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
from chess_ai.dataset.sources import PgnReader, Source, quiet_parser, resolve_sources
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

ProgressCallback = Callable[[Progress], None]

LOGGER = logging.getLogger(__name__)


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

    ``overwrite`` replaces the dataset only once the new one is finished, so the old one survives
    a build that fails — at the cost of both existing at once while it runs. That way round
    because a disk with room for only one copy is exactly the disk that fills up part-way
    through, and deleting first would leave neither.

    A *source* that cannot be opened is counted rather than fatal, but only so far: a build that
    could open none of them, or that could not open all of them and would be replacing a dataset
    already there, is refused. See :func:`_check_sources_were_read`.
    """
    valid_name(name)
    if not 0.0 <= validation_fraction <= 1.0:
        raise DatasetError(f"validation fraction {validation_fraction} is not between 0 and 1")
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
        _report_leftovers(data_dir, name)
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
                for index, source in enumerate(sources):
                    build.read_source(source, index)
                manifest = build.manifest(
                    name=name,
                    created=now if now is not None else datetime.now(UTC),
                )
            _check_sources_were_read(manifest, directory)
            manifest.save(partial)
            _publish(partial, directory, name=name, overwrite=overwrite)
        except BaseException:
            # Including a Ctrl-C: what was written is unreadable without a manifest, so leaving
            # it behind would only be rubble for the next build to clear up.
            shutil.rmtree(partial, ignore_errors=True)
            raise
        # Last of all, because everything above it can still fail: closing the shards flushes
        # them, and publishing renames them. A summary printed before those would say the build
        # was finished and then be followed by the reason it was not.
        build.report(done=True)
    return manifest


def _check_sources_were_read(manifest: Manifest, directory: Path) -> None:
    """Refuse to publish a build that did not read the files it was given, where that would lose.

    The sources are sized when a build starts and opened as it reaches them, hours later, so a
    mount that drops or a sync job that rotates a directory of dumps can leave a build reading
    nothing at all. Such a build has no business being published:

    - if *no* source could be opened, there is nothing to publish, whether or not a dataset of
      this name is already there
    - if *some* source could not be opened and a dataset is already there, the dataset in place
      is more complete than this one, and a nightly ``--overwrite`` must not trade it for this.
      The remedy is to fix the source or to stop naming it, not a flag that says to carry on

    A source that opened and then stopped making sense part-way through is not this: that is
    ordinary bad PGN, its earlier games are in the dataset, and it is counted as skipped games.
    """
    missing = [source for source in manifest.sources if not source.opened]
    if not missing:
        return
    named = ", ".join(str(source.path) for source in missing)
    if len(missing) == len(manifest.sources):
        raise DatasetError(
            f"none of the {len(missing)} source(s) of dataset {manifest.name!r} could be read, "
            f"so there is nothing to build from: {named}"
        )
    if directory.exists():
        raise DatasetError(
            f"{len(missing)} of the {len(manifest.sources)} source(s) of dataset "
            f"{manifest.name!r} could not be read, and the dataset already in {directory} was "
            f"built from more than this one could be: {named}. It has been left alone; fix those "
            "sources, or leave them out to build from the rest on purpose"
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
            # away the very thing _report_leftovers offers back. The rename below refuses
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
        except OSError:
            # Nothing else can be done about it here, and the dataset is already in place.
            discarded = aside
        # Past the point of no return: failing here leaves rubble rather than losing a dataset.
        shutil.rmtree(discarded, ignore_errors=True)


def _report_leftovers(data_dir: Path, name: str) -> None:
    """Say what earlier builds of this dataset left lying about, since nothing else will.

    Neither kind is removed: see :func:`~chess_ai.dataset.store.abandoned_partials` for why a
    working directory cannot safely be told from a live build's, and a set-aside dataset is real
    data that only its owner should decide about.
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
    for path in replaced_datasets(data_dir, name):
        LOGGER.warning(
            "chess-ai: %s is the dataset %r that an interrupted build set aside and never put "
            "back. Nothing else will mention it: rename it to %s to have it again, or remove it.",
            path,
            name,
            dataset_path(data_dir, name),
        )


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
        self.games_read = 0
        self.sources: list[SourceInfo] = []
        self.results: Counter[Result] = Counter()
        self.time_controls: Counter[TimeControl] = Counter()
        self.rating_sources: Counter[RatingSource] = Counter()
        self.ratings: Counter[str] = Counter()
        self.ratings_unknown = 0
        self.skipped: Counter[SkipReason] = Counter()
        self._noted_failure = False

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
        opened = True
        reader = PgnReader(source)
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
            opened = False
            LOGGER.warning("chess-ai: %s; its games are not in this dataset", e)
        else:
            try:
                read, kept, error = self._read_games(reader, index)
            finally:
                reader.close()
        self._bytes_before += source.bytes
        self._bytes_read = self._bytes_before
        self.sources.append(
            SourceInfo(
                path=str(source.path),
                bytes=source.bytes,
                games_read=read,
                games_kept=kept,
                error=error,
                opened=opened,
            )
        )

    def _read_games(self, reader: PgnReader, index: int) -> tuple[int, int, str | None]:
        """Read an open file's games, and say how many it held, how many were kept, and why not.

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
                return read, kept, None
            except Exception as e:
                # The file stopped making sense, mid-game. What was read is kept, the rest of the
                # file is not, and the count says a game was lost.
                self._note_failure(e)
                self.skipped[SkipReason.UNREADABLE] += 1
                return read, kept, f"stopped reading after {read} games: {_described(e)}"
            read += 1
            self.games_read += 1
            self._bytes_read = self._bytes_before + reader.bytes_read
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
            self.skipped[skipped.reason] += 1
            return 0
        except Exception as e:
            # Not a shape of broken game anyone has seen yet. One game is not worth abandoning
            # a build for, so it goes in the same place as the rest.
            self._note_failure(e)
            self.skipped[SkipReason.UNREADABLE] += 1
            return 0
        split = VALIDATION if split_of(records.identity, self.validation_fraction) else TRAIN
        self.writer.splits[split].add_game(records.game, records.positions, records.moves)
        self._count(records)
        return 1

    def _note_failure(self, error: Exception) -> None:
        """Say once that something went wrong in a way nothing here expected.

        Counting a game as unreadable is right for a broken game and hides a broken build: a
        bug in this code would skip every game in every file and report a clean dataset of
        nothing. The first one is logged, with its traceback, so that a build going wrong
        systematically says so instead of looking like a directory of bad PGN.
        """
        if self._noted_failure:
            return
        self._noted_failure = True
        LOGGER.warning(
            "chess-ai: a game could not be read for a reason nothing expected, and is counted "
            "as %s; further ones are counted silently",
            SkipReason.UNREADABLE.value,
            exc_info=error,
        )

    def _count(self, records: GameRecords) -> None:
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

    def report(self, *, done: bool = False) -> None:
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
                    games_read=self.games_read,
                    games_kept=sum(split.games for split in self.writer.splits.values()),
                    positions=sum(split.positions for split in self.writer.splits.values()),
                    bytes_read=self._bytes_total if done else self._bytes_read,
                    bytes_total=self._bytes_total,
                    seconds=time.monotonic() - self._started,
                    done=done,
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
                reason.value: self.skipped[reason] for reason in SkipReason if self.skipped[reason]
            },
            statistics=Statistics(
                results={
                    text: self.results[result]
                    for result, text in RESULT_NAMES.items()
                    if self.results[result]
                },
                time_controls=_by_name(self.time_controls),
                rating_sources=_by_name(self.rating_sources),
                ratings=dict(sorted(self.ratings.items(), key=lambda item: int(item[0]))),
                ratings_unknown=self.ratings_unknown,
            ),
        )


def _described(error: Exception) -> str:
    """One line naming what went wrong, for the manifest to record against a source."""
    return f"{type(error).__name__}: {error}"


def _by_name(counts: Counter) -> dict[str, int]:
    """Counted enum members as lowercase names, in the enum's own order, dropping zeroes."""
    return {member.name.lower(): counts[member] for member in sorted(counts) if counts[member]}
