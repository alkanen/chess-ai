"""A dataset definition: a TOML file saying what to build a dataset from and what to keep.

::

    sources = ["../data/lichess/lichess_db_standard_rated_2024-0[1-6].pgn"]
    validation_fraction = 0.02
    rating_source = "auto"

    [filters]
    min_rating = 2200
    time_controls = ["rapid", "classical"]

It is read when a dataset is built and not afterwards: what the build actually used, flags and
all, is in the manifest. The name is not in it, so one file can build several datasets -- a full
one and a quick ``-test`` with ``--max-games`` -- under names given on the command line.

Sources that are not absolute are relative to the file, not to wherever the build is started
from, so that the file means the same dataset from anywhere.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chess_ai.dataset.manifest import Filters
from chess_ai.dataset.store import DatasetError


class DatasetDefinition(BaseModel):
    """What a definition file says; everything in it can be overridden on the command line."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: list[str] = Field(default_factory=list)
    validation_fraction: float | None = None
    rating_source: str | None = None
    filters: Filters = Filters()


def load_definition(path: Path) -> DatasetDefinition:
    """The definition in ``path``, with its sources made relative to where the file is.

    Raises :exc:`~chess_ai.dataset.store.DatasetError` for a file that is not there, is not TOML,
    or says something a definition cannot.
    """
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise DatasetError(f"no dataset definition {path}") from e
    except OSError as e:
        raise DatasetError(f"cannot read dataset definition {path}: {e.strerror}") from e
    except tomllib.TOMLDecodeError as e:
        raise DatasetError(f"invalid TOML in dataset definition {path}: {e}") from e
    try:
        definition = DatasetDefinition.model_validate(data)
    except ValidationError as e:
        raise DatasetError(f"invalid dataset definition {path}: {e}") from e
    base = path.parent
    return definition.model_copy(
        update={
            "sources": [
                source if Path(source).is_absolute() else str(base / source)
                for source in definition.sources
            ]
        }
    )
