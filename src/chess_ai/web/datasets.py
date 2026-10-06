"""Datasets as the browser sees them: what each one is, and the games it holds.

Everything here reads what a build left behind and nothing else. The manifest already says what
a dataset is made of — it counted as the games went past — so the list and the statistics are
the manifest as it is, and only a page of games or one game being looked at touches the records.

A dataset stores what a game was, not what its file said about it: no player names, no event,
no termination. A game opened from one is therefore described by what the records do keep — the
ratings, the date, the result and the file it came from — and replayed from the position it
started in, which is the first of its own positions.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict

from chess_ai import replay
from chess_ai.dataset import (
    SPLITS,
    Dataset,
    DatasetError,
    GameFlags,
    Manifest,
    ManifestError,
    RatingSource,
    SplitReader,
    TimeControl,
    dataset_path,
    list_datasets,
    load_manifest,
    open_dataset,
    unpack_board,
)
from chess_ai.dataset.games import RESULT_NAMES
from chess_ai.dataset.manifest import MANIFEST_FILE
from chess_ai.dataset.records import Result as RecordResult
from chess_ai.move_codec import move_at
from chess_ai.position_view import Result

MAX_PAGE: Final = 200
"""The most games one page of a dataset's games can hold."""

DEFAULT_PAGE: Final = 50


class NoSuchDatasetError(LookupError):
    """There is no dataset of that name, or nothing in it at the place asked for."""


class DatasetSummary(BaseModel):
    """One dataset, as the list of them shows it: its manifest, or why that cannot be read.

    A dataset whose manifest is damaged, or of a format this code does not read, is still listed:
    it is taking up a directory and quite possibly a lot of disk, and leaving it out would make a
    dataset somebody built simply vanish.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    manifest: Manifest | None = None
    error: str | None = None
    """Why the manifest cannot be read, when it cannot."""


class DatasetGame(BaseModel):
    """One game of a dataset, as a page of its games lists it."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    index: int
    """Which game of its split this is, counting from zero; how it is opened."""
    plies: int
    white_rating: int | None
    """None where the file gave no rating."""
    black_rating: int | None
    result: Result
    date: str
    """As PGN writes it ("2024.01.05"), with "??" for what the file did not say."""
    time_control: str
    """The class the game's clock falls in: "bullet", "blitz", ..., or "unknown"."""
    rating_source: str
    """The pool the ratings come from: "lichess", "chesscom", ..., or "unknown"."""
    source: str | None
    """The PGN file the game came from, as the manifest names it."""
    custom_start: bool
    """Whether the game began somewhere other than where games begin."""


class DatasetGames(BaseModel):
    """One page of a split's games, in the order the build stored them."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    dataset: str
    split: str
    total: int
    """How many games the split holds, which says how many pages there are."""
    offset: int
    games: list[DatasetGame]


def describe_datasets(data_dir: Path) -> list[DatasetSummary]:
    """Every dataset in ``data_dir``, by name."""
    return [_summary(data_dir, name) for name in list_datasets(data_dir)]


def describe_dataset(data_dir: Path, name: str) -> DatasetSummary:
    """The dataset called ``name``, or :exc:`NoSuchDatasetError`."""
    _existing(data_dir, name)
    return _summary(data_dir, name)


def dataset_games(data_dir: Path, name: str, split: str, offset: int, limit: int) -> DatasetGames:
    """Up to ``limit`` games of ``split`` from game ``offset`` on.

    A page past the last game is empty rather than missing: the split is there, and has no
    more games in it.

    Raises:
        NoSuchDatasetError: there is no such dataset, or it has no such split.
        DatasetError: the dataset is there and cannot be read.
    """
    with _opened(data_dir, name) as dataset:
        reader = _split(dataset, split)
        count = max(0, min(limit, reader.games - offset))
        records = reader.game_records(offset, count) if count else np.empty(0)
        games = []
        for n, record in enumerate(records):
            with _damaged(name, split, offset + n):
                games.append(_listed(dataset.manifest, offset + n, record))
        return DatasetGames(
            dataset=name, split=split, total=reader.games, offset=offset, games=games
        )


def dataset_game(data_dir: Path, name: str, split: str, index: int) -> replay.ReplayGame:
    """Game ``index`` of ``split``, with the position before every move and after it.

    Raises:
        NoSuchDatasetError: there is no such dataset, split or game.
        DatasetError: the game's records cannot be read, or do not make a game.
    """
    with _opened(data_dir, name) as dataset:
        reader = _split(dataset, split)
        if not 0 <= index < reader.games:
            raise NoSuchDatasetError(
                f"dataset {name!r} has no {split} game {index + 1}: "
                f"its {split} split has {reader.games:,} games"
            )
        # Everything below reads the game's records and takes them at their word, so a record
        # that does not make sense — a ply offset past the end, a result that is not one, a move
        # that cannot be played — is reported as the damage it is rather than as a traceback.
        with _damaged(name, split, index):
            listed = _listed(dataset.manifest, index, reader.game(index))
            positions = reader.game_positions(index)
            moves = reader.move_sequence(index)
            summary = replay.GameSummary(
                index=index,
                # Which dataset and game this is, the viewer already knows from how it was
                # opened; the clock is what the records keep that nothing else on the page says.
                event="?" if listed.time_control == "unknown" else f"{listed.time_control} game",
                site=Path(listed.source).name if listed.source else "?",
                date=listed.date,
                round="-",
                white=_player(listed.white_rating),
                black=_player(listed.black_rating),
                result=listed.result,
                termination=None,
                plies=listed.plies,
            )
            # The game's first position is where it started, which is all a game that began
            # somewhere unusual needs to be played out again.
            return replay.replayed(
                summary, unpack_board(positions[0]), (move_at(int(m)) for m in moves)
            )


@contextmanager
def _damaged(name: str, split: str, index: int) -> Iterator[None]:
    """Report a game whose records do not make a game as a :exc:`DatasetError` naming it."""
    try:
        yield
    except (ValueError, IndexError, KeyError) as damaged:
        raise DatasetError(
            f"{split} game {index + 1} of dataset {name!r} cannot be read: {damaged}"
        ) from damaged


def _existing(data_dir: Path, name: str) -> Path:
    """The directory of dataset ``name``, which has to be there and finished."""
    try:
        directory = dataset_path(data_dir, name)
    except DatasetError as invalid:
        raise NoSuchDatasetError(f"there is no dataset called {name!r}") from invalid
    if not (directory / MANIFEST_FILE).is_file():
        raise NoSuchDatasetError(f"there is no dataset called {name!r}")
    return directory


def _summary(data_dir: Path, name: str) -> DatasetSummary:
    try:
        return DatasetSummary(name=name, manifest=load_manifest(dataset_path(data_dir, name)))
    except ManifestError as unreadable:
        return DatasetSummary(name=name, error=str(unreadable))


@contextmanager
def _opened(data_dir: Path, name: str) -> Iterator[Dataset]:
    """A dataset opened for one request, and let go of at the end of it.

    Let go of because the server lives for weeks and each mapped shard holds a file descriptor;
    see :meth:`~chess_ai.dataset.Dataset.close`.
    """
    _existing(data_dir, name)
    with open_dataset(name, data_dir=data_dir) as dataset:
        yield dataset


def _split(dataset: Dataset, split: str) -> SplitReader:
    if split not in SPLITS or split not in dataset.splits:
        raise NoSuchDatasetError(
            f"dataset {dataset.manifest.name!r} has no {split!r} split; "
            f"it has {', '.join(dataset.splits)}"
        )
    return dataset[split]


def _listed(manifest: Manifest, index: int, record: np.void) -> DatasetGame:
    flags = int(record["flags"])
    source = int(record["source"])
    return DatasetGame(
        index=index,
        plies=int(record["ply_count"]),
        white_rating=None
        if flags & GameFlags.WHITE_RATING_UNKNOWN
        else int(record["white_rating"]),
        black_rating=None
        if flags & GameFlags.BLACK_RATING_UNKNOWN
        else int(record["black_rating"]),
        result=RESULT_NAMES[RecordResult(int(record["result"]))],  # type: ignore[arg-type]
        date=pgn_date(int(record["date"])),
        time_control=_name(TimeControl, int(record["time_control"])),
        rating_source=_name(RatingSource, int(record["rating_source"])),
        source=manifest.sources[source].path if source < len(manifest.sources) else None,
        custom_start=bool(flags & GameFlags.CUSTOM_START),
    )


def pgn_date(date: int) -> str:
    """A record's ``yyyymmdd`` as PGN writes a date, "??" standing in for any part that is 0."""
    year, rest = divmod(date, 10000)
    month, day = divmod(rest, 100)
    return ".".join(
        [
            f"{year:04d}" if year else "????",
            f"{month:02d}" if month else "??",
            f"{day:02d}" if day else "??",
        ]
    )


def _name(kind: type[TimeControl] | type[RatingSource], value: int) -> str:
    """An enum field as the manifest's statistics name it, lowercase, or "unknown"."""
    try:
        return kind(value).name.lower()
    except ValueError:
        return "unknown"


def _player(rating: int | None) -> str:
    """A side of a dataset game, which has a rating and no name."""
    return "rating unknown" if rating is None else f"rated {rating}"
