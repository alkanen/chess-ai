"""The games the server holds, written to disk so that they outlive a restart.

The server is restarted for every update, and a game between friends can take a week, so every
game is written down as it changes: one JSON file per game, named by its id, in the directory
``[paths] ongoing_games`` names. The server reads them all back when it starts.

A file holds what it takes to carry on with the game: its links, its moves and how far it has
got, and for each side what it takes to sit the same player down again. The players themselves
are not in it: a checkpoint is loaded again, and Stockfish started again, only when the game
next needs them.

Every write happens on the event loop the games are played on, one after another, so a game's
file has one writer. It is written next to itself under a fixed temporary name and then moved
into place, so that a server that dies mid-write leaves the file as it was rather than half of
it; the temporary file it may leave behind is overwritten by the next write, and never read.
"""

import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chess_ai.game_session import SessionRecord
from chess_ai.players import SelectionStrategy, StockfishDescription

logger = logging.getLogger(__name__)

SUFFIX = ".json"
_WRITING = ".json.tmp"

_FILE_NAME = re.compile(r"[A-Za-z0-9_-]+")
"""What a game's id has to be to name a file: nothing that reaches outside the directory."""


class GameLinks(BaseModel):
    """The links to one game, as the person who started it is given them."""

    watch: str
    """Follows the game and can do nothing to it; for anybody."""
    white: str | None = None
    """Plays White, for a game in which a person plays White."""
    black: str | None = None
    """Plays Black, for a game in which a person plays Black."""
    control: str | None = None
    """Aborts the game, for a game in which nobody plays either side by hand."""


class _PlayerRecord(BaseModel):
    """What it takes to sit the same player down again, after a restart.

    The player as it was made, rather than as it was asked for: a game started against a run's
    latest checkpoint goes on against the checkpoint it began with, not a newer one.
    """

    model_config = ConfigDict(extra="forbid")


class HumanRecord(_PlayerRecord):
    kind: Literal["human"] = "human"


class RandomRecord(_PlayerRecord):
    kind: Literal["random"] = "random"


class ModelRecord(_PlayerRecord):
    kind: Literal["model"] = "model"
    run: str
    step: int
    fingerprint: tuple[int, ...]
    """Which file the checkpoint was, so that one written in its place since is not taken for
    it; see :class:`chess_ai.web.engine_cache.CheckpointKey`."""
    rating: int | None
    strategy: SelectionStrategy
    temperature: float
    seed: int | None
    """The seed the player was made with. A seeded player made again draws the same numbers
    from the start, not from where it had got to, so its moves after a restart are not the
    ones an unbroken game would have had."""


class StockfishRecord(_PlayerRecord):
    kind: Literal["stockfish"] = "stockfish"
    elo: int
    """The strength that was asked for, which Stockfish is asked for again."""
    move_time: float
    description: StockfishDescription
    """How it played, which is what the game shows until the engine is started again."""


PlayerRecord = Annotated[
    HumanRecord | RandomRecord | ModelRecord | StockfishRecord, Field(discriminator="kind")
]


class StoredGame(BaseModel):
    """One game as it is kept on disk."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    """Which shape the file has, so that a later server can tell an older file from its own."""
    links: GameLinks
    white: PlayerRecord
    black: PlayerRecord
    session: SessionRecord
    settings: dict[str, Any] | None = None
    """What the game was started with, as whoever started it gave it, to start another the
    same way."""
    next: str | None = None
    """The id of the game started from this one once it ended, if one has been."""

    @property
    def id(self) -> str:
        return self.session.id


class GameStore:
    """The directory the games are kept in, one file per game."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, game: StoredGame) -> None:
        """Keep ``game``, in place of whatever was kept of it before.

        Raises:
            OSError: the file could not be written. What was kept before is still there.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        writing = self._path(game.id, _WRITING)
        writing.write_text(game.model_dump_json(), encoding="utf-8")
        os.replace(writing, self._path(game.id, SUFFIX))

    def remove(self, game_id: str) -> None:
        """Delete what is kept of game ``game_id``, if anything is.

        Raises:
            OSError: the file is there and could not be deleted.
        """
        for suffix in (SUFFIX, _WRITING):
            self._path(game_id, suffix).unlink(missing_ok=True)

    def read_all(self) -> list[StoredGame]:
        """Every game kept here, in the order of their ids.

        A file that cannot be read is reported and left where it is, for somebody to look at;
        the games in the other files are read regardless.
        """
        try:
            paths = sorted(self.directory.glob(f"*{SUFFIX}"))
        except OSError:
            logger.exception("Cannot look for games in %s", self.directory)
            return []
        games = []
        for path in paths:
            try:
                game = StoredGame.model_validate_json(path.read_bytes())
            except (OSError, ValidationError):
                logger.exception("Cannot read the game in %s, so it is left out", path)
                continue
            if f"{game.id}{SUFFIX}" != path.name:
                logger.error(
                    "%s holds game %s, which is not its own, so it is left out", path, game.id
                )
                continue
            games.append(game)
        return games

    def _path(self, game_id: str, suffix: str) -> Path:
        if not _FILE_NAME.fullmatch(game_id):
            raise ValueError(f"a game's id names its file, and {game_id!r} cannot")
        return self.directory / f"{game_id}{suffix}"
