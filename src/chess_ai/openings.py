"""Opening sets: the curated lines the games of a match start from.

A set is a TOML file of named lines in standard notation, with a name and a version. The sets
that come with the project live in ``opening_sets`` next to this module; any other file can be
given by its path, for a match that wants openings of its own.

Every line is checked when the set is read, so that a mistyped move is refused before a match
starts rather than in its forty-first game, and two lines that reach the same position are
refused too: a match between two deterministic players would play them as one game twice.
"""

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import chess
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_OPENING_SET = "standard"
"""The set a match plays unless told otherwise."""

_BUNDLED = "opening_sets"
_SUFFIX = ".toml"


class OpeningSetError(Exception):
    """An opening set cannot be found or read, or holds a line that cannot be played."""


@dataclass(frozen=True)
class Opening:
    name: str
    """What the line is called, such as "Ruy Lopez", which a PGN's ``Opening`` tag carries."""
    moves: tuple[chess.Move, ...]
    """The line from the starting position, every move of it legal where it is played."""

    def board(self) -> chess.Board:
        """The position the line reaches, with its moves as the board's history."""
        board = chess.Board()
        for move in self.moves:
            board.push(move)
        return board


@dataclass(frozen=True)
class OpeningSet:
    name: str
    version: int
    openings: tuple[Opening, ...]

    @property
    def label(self) -> str:
        """The set and its version, as a match's PGN records it: "standard v1"."""
        return f"{self.name} v{self.version}"


class _OpeningFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    moves: str = Field(min_length=1)


class _OpeningSetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = Field(ge=1)
    openings: list[_OpeningFile] = Field(min_length=1)


def bundled_opening_sets() -> list[str]:
    """The names of the sets that come with the project, sorted."""
    return sorted(
        entry.name.removesuffix(_SUFFIX)
        for entry in resources.files(__package__).joinpath(_BUNDLED).iterdir()
        if entry.name.endswith(_SUFFIX)
    )


def load_opening_set(name_or_path: str | Path) -> OpeningSet:
    """The opening set by the name it comes with, or read from a file of one's own.

    A name of a bundled set is looked up first; anything else is taken to be a path.

    Raises:
        OpeningSetError: there is no such set or file, it is not an opening set, or one of its
            lines has a move that cannot be played, or reaches the position another one does.
    """
    text, where = _read(str(name_or_path))
    try:
        written = _OpeningSetFile.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as e:
        raise OpeningSetError(f"{where} is not valid TOML: {e}") from e
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in e.errors()
        )
        raise OpeningSetError(f"{where} is not an opening set: {problems}") from e
    openings = tuple(_opening(line, where) for line in written.openings)
    reached: dict[str, str] = {}
    for opening in openings:
        position = opening.board().epd()
        if position in reached:
            raise OpeningSetError(
                f"{where}: {opening.name!r} reaches the same position as {reached[position]!r}, "
                "so the two would be played as one game"
            )
        reached[position] = opening.name
    return OpeningSet(name=written.name, version=written.version, openings=openings)


def _read(name_or_path: str) -> tuple[str, str]:
    """The text of the set, and how to name where it came from in a message."""
    if name_or_path in bundled_opening_sets():
        bundled = resources.files(__package__).joinpath(_BUNDLED, f"{name_or_path}{_SUFFIX}")
        return bundled.read_text(encoding="utf-8"), f"opening set {name_or_path!r}"
    path = Path(name_or_path)
    try:
        return path.read_text(encoding="utf-8"), str(path)
    except FileNotFoundError as e:
        raise OpeningSetError(
            f"there is no opening set {name_or_path!r}: give the path of a file, or one of "
            f"the sets that come with chess-ai ({', '.join(bundled_opening_sets())})"
        ) from e
    except OSError as e:
        raise OpeningSetError(f"cannot read {path}: {e.strerror}") from e


def _opening(line: _OpeningFile, where: str) -> Opening:
    board = chess.Board()
    for san in line.moves.split():
        try:
            board.push_san(san)
        except ValueError as e:
            raise OpeningSetError(
                f"{where}: {line.name!r} cannot play {san} after {_played(board)}"
            ) from e
    return Opening(name=line.name, moves=tuple(board.move_stack))


def _played(board: chess.Board) -> str:
    """The moves so far, for a message about the one that could not follow them."""
    if not board.move_stack:
        return "no moves"
    return chess.Board().variation_san(board.move_stack)
