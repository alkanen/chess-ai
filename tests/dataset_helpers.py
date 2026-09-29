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


def build(data_dir: Path, *sources: str, name: str = "test", **options) -> Manifest:
    """Build a dataset called "test" from ``sources``, or from every fixture file.

    A bare fixture file name becomes its path; anything else is passed to the build as it is,
    so a test can hand it a glob, a directory or a file it wrote itself.
    """
    named = [fixture(source) if (FIXTURES / source).is_file() else source for source in sources]
    return build_dataset(name, named or [str(FIXTURES)], data_dir=data_dir, **options)


def move_sequences(split) -> list[tuple[int, ...]]:
    """Every game in ``split`` as the moves it is made of, which is what tells games apart."""
    return [tuple(split.move_sequence(game).tolist()) for game in range(split.games)]


def shard_bytes(directory: Path) -> dict[str, bytes]:
    """Every shard of a built dataset, by its path inside it.

    What a build produced, to the byte, which is how two builds are compared without trusting
    either one's manifest to describe it.
    """
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*.bin"))
    }


BACK_TO_BACK = (
    '[Event "x"]\n[Site "https://lichess.org/a"]\n[Result "1-0"]\n'
    '[WhiteElo "1500"]\n[BlackElo "1500"]\n[TimeControl "600+0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n'
)
"""One game written so that the next one follows it with no blank line in between.

PGN's export format puts a blank line between games and most writers do, but nothing enforces
it, and a file that does not is the shape that makes cutting one hard. See
:func:`~chess_ai.dataset.sources.align_to_game`.
"""


def flat_pgn(path: Path, games: int) -> Path:
    """``games`` games at ``path``, none of them separated from the next by a blank line."""
    path.write_text(BACK_TO_BACK * games)
    return path


class CountingPath:
    """Wraps one path so that every byte read through :meth:`open` is counted.

    Reading is what a boundary scan costs, and unlike a stopwatch it is the same number on
    every machine -- which is what makes "this does not read the file over and over" a test
    rather than a timing guess.

    Every way of reading has to be counted, not just :meth:`read`. A buffered reader fills
    itself with :meth:`readinto`, and a counter that only knew about ``read`` scored zero for
    the whole streamed path -- which made the test that used it pass against an implementation
    that swallowed the file whole, which is the one thing it existed to catch.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.bytes_read = 0

    def install(self, monkeypatch) -> None:
        real_open = Path.open
        counter = self

        class Counted:
            def __init__(self, handle):
                self._handle = handle

            def read(self, *args, **kwargs):
                data = self._handle.read(*args, **kwargs)
                counter.bytes_read += len(data)
                return data

            def read1(self, *args, **kwargs):
                data = self._handle.read1(*args, **kwargs)
                counter.bytes_read += len(data)
                return data

            def readline(self, *args, **kwargs):
                data = self._handle.readline(*args, **kwargs)
                counter.bytes_read += len(data)
                return data

            def readinto(self, buffer, *args, **kwargs):
                got = self._handle.readinto(buffer, *args, **kwargs)
                counter.bytes_read += got or 0
                return got

            def readinto1(self, buffer, *args, **kwargs):
                got = self._handle.readinto1(buffer, *args, **kwargs)
                counter.bytes_read += got or 0
                return got

            def __getattr__(self, name):
                return getattr(self._handle, name)

            def __enter__(self):
                self._handle.__enter__()
                return self

            def __exit__(self, *exc):
                return self._handle.__exit__(*exc)

        def opener(self, *args, **kwargs):
            handle = real_open(self, *args, **kwargs)
            return Counted(handle) if self == counter.path else handle

        monkeypatch.setattr(Path, "open", opener)


SEPARATED = (
    '[Event "x"]\n[Site "https://lichess.org/a"]\n[Result "1-0"]\n'
    '[WhiteElo "1500"]\n[BlackElo "1500"]\n[TimeControl "600+0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n\n'
)
"""One game with a blank line after it, so a file of these cuts into as many pieces as asked."""


def separated_pgn(path: Path, games: int) -> Path:
    """``games`` games at ``path``, separated the way PGN says, so they can be cut between."""
    path.write_text(SEPARATED * games)
    return path
