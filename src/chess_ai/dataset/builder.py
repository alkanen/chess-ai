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
import shutil
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import chess.pgn

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
    dataset_path,
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
    wrong with a *game* is counted instead, and the manifest that comes back says what was.
    """
    valid_name(name)
    if not 0.0 <= validation_fraction <= 1.0:
        raise DatasetError(f"validation fraction {validation_fraction} is not between 0 and 1")
    sources = resolve_sources(patterns)
    directory = dataset_path(data_dir, name)
    if directory.exists():
        if not overwrite:
            raise DatasetError(f"dataset {name!r} is already in {directory}")
        shutil.rmtree(directory)

    build = _Build(
        directory=directory,
        shards=shards,
        sources=sources,
        rating_source=rating_source,
        validation_fraction=validation_fraction,
        progress=progress,
    )
    with build.writer, quiet_parser():
        for index, source in enumerate(sources):
            build.read_source(source, index)
        manifest = build.manifest(
            name=name,
            created=now if now is not None else datetime.now(UTC),
        )
        build.report(done=True)
    manifest.save(directory)
    return manifest


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
        """Read every game in one file into the dataset, whatever the file turns out to hold."""
        read = 0
        kept = 0
        with PgnReader(source) as reader:
            try:
                for record in reader.games():
                    read += 1
                    self.games_read += 1
                    self._bytes_read = self._bytes_before + reader.bytes_read
                    kept += self._add_game(record, index)
                    if read % REPORT_EVERY == 0:
                        self.report()
            except Exception as e:
                # The file itself stopped making sense, mid-game. What was read is kept, the
                # rest of the file is not, and the count says a game was lost.
                self._note_failure(e)
                self.skipped[SkipReason.UNREADABLE] += 1
        self._bytes_before += source.bytes
        self._bytes_read = self._bytes_before
        self.sources.append(
            SourceInfo(path=str(source.path), bytes=source.bytes, games_read=read, games_kept=kept)
        )

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
        """Tell the progress callback, if there is one, where the build has got to."""
        if self._progress is None:
            return
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


def _by_name(counts: Counter) -> dict[str, int]:
    """Counted enum members as lowercase names, in the enum's own order, dropping zeroes."""
    return {member.name.lower(): counts[member] for member in sorted(counts) if counts[member]}
