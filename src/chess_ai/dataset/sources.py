"""Which PGN files a build reads, and reading the games out of one of them.

A build is given paths on a command line, which may be files, directories or globs — and a
glob the shell did not expand, because a pattern matching a year of monthly dumps is easier
to type than the twelve names, and quoting it is easier than remembering not to.

Reading a file says how far through it it has got, which is the only honest basis for a time
remaining: games differ in length, and a dump's own game count is not written anywhere.
"""

import glob
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, TextIO

import chess.pgn

from chess_ai.dataset.store import DatasetError

SUFFIX: Final = ".pgn"
"""What a PGN file is called, and all that is picked up from a directory."""

ENCODING: Final = "utf-8-sig"
"""How a PGN file is read: UTF-8, past a byte-order mark some exporters write.

PGN's own specification says Latin-1 and the world writes UTF-8, so bytes that are neither
are replaced rather than raised over. A player's name with one broken character in it is not
a reason to abandon an import.
"""

_GLOB = "*?["


@dataclass(frozen=True)
class Source:
    """One PGN file to read, and how large it is, which is what makes progress an estimate."""

    path: Path
    bytes: int


def resolve_sources(patterns: Sequence[str]) -> list[Source]:
    """The PGN files ``patterns`` name, in order, without repeats.

    Each pattern is a file, a directory (its own ``*.pgn`` files, in name order), or a glob.
    A pattern matching nothing is an error: a mistyped path that quietly built an empty
    dataset would only be noticed by the run that trained on it.
    """
    sources: list[Source] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matched = _matches(pattern)
        if not matched:
            raise DatasetError(f"no PGN files match {pattern!r}")
        for path in matched:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            sources.append(Source(path=path, bytes=path.stat().st_size))
    return sources


def _matches(pattern: str) -> list[Path]:
    """The files one pattern names, in the order they will be read."""
    path = Path(pattern)
    if path.is_dir():
        return sorted(child for child in path.iterdir() if _is_pgn(child))
    if any(character in pattern for character in _GLOB):
        return sorted(Path(match) for match in glob.glob(pattern) if _is_pgn(Path(match)))
    if not path.is_file():
        raise DatasetError(f"no such PGN file: {pattern}")
    return [path]


def _is_pgn(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == SUFFIX


class PgnReader:
    """The games in one PGN file, one at a time, with the file's read position to hand.

    A game the parser could not read comes back all the same, carrying what went wrong in its
    ``errors``; deciding what to do about that belongs to the caller, which counts it.
    """

    def __init__(self, source: Source) -> None:
        self.source = source
        self._file: TextIO | None = None
        self._bytes_read = 0

    def open(self) -> None:
        """Open the file, or raise :exc:`DatasetError` saying why it cannot be read.

        Separate from the ``with`` block so that a caller can guard the opening of a file without
        also guarding what it then does with the games: they fail for unrelated reasons and
        deserve unrelated answers.
        """
        try:
            self._file = self.source.path.open(encoding=ENCODING, errors="replace")
        except OSError as e:
            raise DatasetError(f"cannot read {self.source.path}: {e.strerror}") from e

    def close(self) -> None:
        """Let go of the file, whether or not it was read to the end."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "PgnReader":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def bytes_read(self) -> int:
        """How far into the file the last game ended."""
        return self._bytes_read

    def games(self) -> Iterator[chess.pgn.Game]:
        """Every record in the file, in order, until there are no more."""
        assert self._file is not None, "read a PgnReader inside its with block"
        while True:
            record = chess.pgn.read_game(self._file)
            self._bytes_read = self._file.tell()
            if record is None:
                return
            yield record


@contextmanager
def quiet_parser() -> Iterator[None]:
    """Stop python-chess logging every unreadable game while a build is running.

    It logs one line per bad move, and a dump of millions of games has thousands of them.
    The counts per reason in the manifest say the same thing in a form that fits on a screen.
    """
    logger = logging.getLogger("chess.pgn")
    was = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(was)
