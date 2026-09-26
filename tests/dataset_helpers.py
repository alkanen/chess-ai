"""Fixture PGN files, and building a dataset out of them in a test's own directory."""

from pathlib import Path

from chess_ai.dataset import Manifest, Shards, build_dataset

FIXTURES = Path(__file__).parent / "fixtures" / "pgn"
"""Small PGN files: real-shaped games, games with no ratings, and games that are broken.

Referred to by absolute path because the test suite runs in a directory of its own.
"""

GOOD_GAMES = 8
"""Games in the fixtures that a build keeps; the rest are broken in a way of their own."""

TINY_SHARDS = Shards(positions_per_shard=7, games_per_shard=2, moves_per_shard=5)
"""Shard sizes small enough that the fixtures fill several of each, which is the point."""


def fixture(name: str) -> str:
    """The path of one fixture PGN file, as a build takes it."""
    path = FIXTURES / name
    assert path.is_file(), f"no fixture PGN called {name}"
    return str(path)


def build(data_dir: Path, *sources: str, **options) -> Manifest:
    """Build a dataset called "test" from ``sources``, or from every fixture file.

    A bare fixture file name becomes its path; anything else is passed to the build as it is,
    so a test can hand it a glob, a directory or a file it wrote itself.
    """
    named = [fixture(source) if (FIXTURES / source).is_file() else source for source in sources]
    return build_dataset("test", named or [str(FIXTURES)], data_dir=data_dir, **options)


def move_sequences(split) -> list[tuple[int, ...]]:
    """Every game in ``split`` as the moves it is made of, which is what tells games apart."""
    return [tuple(split.move_sequence(game).tolist()) for game in range(split.games)]
