"""Datasets: PGN files turned into records a model can be trained on.

A dataset is a directory of fixed-size records with a manifest describing them, versioned and
independent of any encoder: it stores what the games were, not what a network should see, so
one dataset feeds every architecture and every encoder variant that comes later.

The pieces, in the order a build uses them:

- :mod:`~chess_ai.dataset.sources` finds the PGN files and reads games out of them
- :mod:`~chess_ai.dataset.games` turns one game into records, or says why it cannot
- :mod:`~chess_ai.dataset.records` is what those records are, down to the byte
- :mod:`~chess_ai.dataset.store` writes and reads them as sharded, memory-mapped files
- :mod:`~chess_ai.dataset.manifest` is everything about a dataset that is not a record
- :mod:`~chess_ai.dataset.builder` runs the build; :mod:`~chess_ai.dataset.summary` reads it back

Training reads a dataset through :func:`open_dataset`.
"""

from chess_ai.dataset.builder import DEFAULT_VALIDATION_FRACTION, build_dataset
from chess_ai.dataset.games import SkipReason
from chess_ai.dataset.manifest import (
    SPLITS,
    TRAIN,
    VALIDATION,
    Manifest,
    ManifestError,
    Shards,
    load_manifest,
)
from chess_ai.dataset.progress import Progress, ProgressPrinter
from chess_ai.dataset.records import (
    FORMAT_VERSION,
    GAME_DTYPE,
    POSITION_DTYPE,
    GameFlags,
    PositionFlags,
    RatingSource,
    Result,
    TimeControl,
    unpack_board,
)
from chess_ai.dataset.store import (
    Dataset,
    DatasetError,
    SplitReader,
    dataset_path,
    list_datasets,
    open_dataset,
)
from chess_ai.dataset.summary import summarize

__all__ = [
    "DEFAULT_VALIDATION_FRACTION",
    "FORMAT_VERSION",
    "GAME_DTYPE",
    "POSITION_DTYPE",
    "SPLITS",
    "TRAIN",
    "VALIDATION",
    "Dataset",
    "DatasetError",
    "GameFlags",
    "Manifest",
    "ManifestError",
    "PositionFlags",
    "Progress",
    "ProgressPrinter",
    "RatingSource",
    "Result",
    "Shards",
    "SkipReason",
    "SplitReader",
    "TimeControl",
    "build_dataset",
    "dataset_path",
    "list_datasets",
    "load_manifest",
    "open_dataset",
    "summarize",
    "unpack_board",
]
