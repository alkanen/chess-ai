"""Global configuration: a TOML file whose settings environment variables can override.

The file is found through, in order: an explicit path (the CLI's ``--config``), the
``CHESS_AI_CONFIG`` environment variable, or ``chess-ai.toml`` in the current directory.
Without a file every setting keeps its default.

Each setting ``<key>`` in section ``[<section>]`` can be overridden by the environment
variable ``CHESS_AI_<SECTION>_<KEY>``, for example ``CHESS_AI_SERVER_PORT=9000``.
"""

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Device = Literal["auto", "cuda", "cpu"]
"""What a config may ask to run a network on. ``auto`` is the GPU when there is one, else the
CPU. Named here rather than with the training config, because two configs choose a device and
this is the module that neither of them, and nothing heavier, has to import to say so."""

DEFAULT_INFERENCE_BATCH = 32
"""Positions per forward pass when a checkpoint is asked about several at once.

Repeated from :mod:`chess_ai.inference` rather than imported, because importing it here would
put torch behind every ``chess-ai`` command; the inference tests check that the two agree.
"""

ENV_PREFIX = "CHESS_AI_"
CONFIG_PATH_ENV = "CHESS_AI_CONFIG"
DEFAULT_CONFIG_FILE = Path("chess-ai.toml")

_PREFIX_SEGMENT = re.compile(r"[A-Za-z0-9._~-]+")


class ConfigError(Exception):
    """The configuration file or an environment override is missing or invalid."""


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    path_prefix: str = ""
    """URL path everything is served under, such as "/chess". Empty serves at the root."""
    stale_after_seconds: float = Field(default=120.0, gt=0)
    """How old a running run's heartbeat may be before the dashboard flags the run as stale.

    A trainer rewrites its heartbeat every couple of seconds while it steps, but not while it is
    starting up, compiling, validating or saving a checkpoint, so this is well above that."""

    @field_validator("path_prefix")
    @classmethod
    def _normalize_path_prefix(cls, value: str) -> str:
        """Normalize to "" or "/segment[/segment...]" with no trailing slash."""
        stripped = value.strip().strip("/")
        if not stripped:
            return ""
        segments = stripped.split("/")
        for segment in segments:
            if not _PREFIX_SEGMENT.fullmatch(segment) or segment in {".", ".."}:
                raise ValueError(
                    f"invalid path prefix {value!r}: use slash-separated segments of "
                    "letters, digits and . _ ~ -"
                )
        return "/" + "/".join(segments)


class PathsConfig(BaseModel):
    """Where the things the system keeps on disk live."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    games: Path = Path("games")
    """Directory the games played here are saved in, one PGN file per game."""
    data: Path = Path("data")
    """Directory the datasets built here are kept in, one directory per dataset."""
    runs: Path = Path("runs")
    """Directory training runs are kept in, one directory per run."""
    ongoing_games: Path = Path("ongoing-games")
    """Directory the games the server holds are kept in while they last, one file per game, so
    that they outlive a restart. Not under ``games``: those are the PGN files of games that
    ended, which are kept for good, and these are deleted once a game has expired."""

    @field_validator("games", "data", "runs", "ongoing_games")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        """Expand a leading ``~``, which a path in a config file may well be written with."""
        return value.expanduser()


class InferenceConfig(BaseModel):
    """How a checkpoint is played with, when a game has one in it.

    The defaults are what it takes to play against a model on the machine that is training one:
    the CPU, and batches small enough that nothing here is what makes a run run out of memory.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Device = "cpu"
    """Where to run the network: "cpu", "cuda", or "auto" for a GPU if there is one.

    The CPU by default, deliberately. A game asks about one position at a time, which even a
    large model answers fast enough to play against, and a training run has the card.

    Typed, so that a misspelled device is refused when the config is read rather than when
    somebody first tries to play a model — which is a request away from anything that could
    explain it, and which nobody who clicked Start had a hand in."""
    batch_size: int = Field(default=DEFAULT_INFERENCE_BATCH, ge=1, le=1024)
    """How many positions go through the network at once when several are asked about."""


class StockfishConfig(BaseModel):
    """Where the Stockfish engine is, for the games and evaluations that play against it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = "stockfish"
    """The Stockfish binary: a path, or a bare name looked up on ``PATH`` as a shell would.

    A string rather than a ``Path``, because the default is a command to look up rather than a
    file here, and a ``Path`` would turn it into ``./stockfish``."""

    @field_validator("path")
    @classmethod
    def _expand_user(cls, value: str) -> str:
        return os.path.expanduser(value)


class GamesConfig(BaseModel):
    """How many games the server holds at once, and what it keeps loaded for them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_ongoing: int = Field(default=20, ge=1)
    """How many games may be in progress at once. A new game beyond that is refused; a game
    that has ended, or been aborted, no longer counts."""
    max_loaded_checkpoints: int = Field(default=10, ge=1)
    """How many checkpoints the games may hold loaded at once, all games together. Games that
    play the same checkpoint share it. Beyond this, the one used longest ago is let go of, and
    loaded again when a game next needs it."""
    checkpoint_idle_hours: float = Field(default=24.0, gt=0)
    """How long a loaded checkpoint nobody has played with is kept before it is let go of."""
    expire_after_days: float = Field(default=7.0, gt=0)
    """How long a game is kept after its last move, finished or not, before it is deleted along
    with its links. Starting a game, every move, and the game ending start the count again."""


class LadderConfig(BaseModel):
    """The Stockfish ladder a checkpoint's Elo is measured on, unless a ladder is told otherwise.

    The defaults play a hundred games, about ten minutes' worth for a mid-sized network on a
    GPU, and span Stockfish's lowest levels, which is where a network that only predicts moves
    plays.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    levels: tuple[int, ...] = (1350, 1500, 1700, 1900, 2100)
    """The Elo levels Stockfish plays the checkpoint at, weakest first. A level outside the range
    Stockfish supports is played at the nearest end of it (1350 to 2850 for Stockfish 14)."""
    games_per_level: int = Field(default=20, ge=1)
    """How many games the checkpoint plays against each level, the colours swapped every game."""
    rating: int = Field(default=2000, ge=0, le=4000)
    """The rating the checkpoint is asked to play like. Always one: asked for none, a model
    plays like a game that claims no rating, which is far below what it can do."""
    move_time: float = Field(default=0.1, ge=0.01, le=60.0)
    """Seconds Stockfish has for each move. Its levels were calibrated at a few seconds a move,
    so at this one it plays below them and the ratings are higher than they would be there:
    compare ladders played at one move time only."""
    openings: str = "standard"
    """The opening set the games start from: one that comes with chess-ai, or a file's path."""
    device: Device = "auto"
    """Where the checkpoint's network runs: "cpu", "cuda", or "auto" for a GPU if there is one.

    Not ``[inference] device``, which is the CPU so that games played in the browser leave a
    training run the card. A ladder asks about thousands of positions one after another, which
    takes a mid-sized network over a minute a game on a busy CPU, and a GPU a fraction of that
    at a share of the card a training run does not notice."""

    @field_validator("levels", mode="before")
    @classmethod
    def _split_levels(cls, value: Any) -> Any:
        """Read ``"1350,1500"``, as an environment variable or the command line says it."""
        if isinstance(value, str):
            return [level.strip() for level in value.split(",") if level.strip()]
        return value

    @field_validator("levels")
    @classmethod
    def _sorted_levels(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("a ladder needs at least one level")
        if any(not 0 <= level <= 4000 for level in value):
            raise ValueError("a level is an Elo from 0 to 4000")
        if len(set(value)) != len(value):
            raise ValueError("each level can only be on the ladder once")
        return tuple(sorted(value))


class EvaluatorConfig(BaseModel):
    """How the evaluator works through checkpoints, and what the probes ask about them.

    The defaults keep it out of a training run's way: the CPU, two threads of it, and small
    batches. Probing a checkpoint is a few dozen positions, which that does in a second or two.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Device = "cpu"
    """Where a checkpoint's network runs: "cpu", "cuda", or "auto" for a GPU if there is one."""
    batch_size: int = Field(default=16, ge=1, le=1024)
    """How many positions go through the network at once."""
    threads: int = Field(default=2, ge=1)
    """How many CPU threads the network may use. Torch would otherwise take every core, which
    a training run's data loaders are using."""
    poll_seconds: float = Field(default=30.0, gt=0)
    """How long the evaluator waits before looking for new checkpoints again, once it has
    evaluated every checkpoint it found."""
    rating: int = Field(default=2000, ge=0, le=4000)
    """The rating a checkpoint is asked about the probe positions at, for both sides. Results
    already saved keep the rating they were measured at; ``chess-ai evaluate`` redoes them."""
    probe_set: str = "standard"
    """The probe positions: a set that comes with chess-ai, or a file's path. Probing with a
    different set, or a new version of this one, probes every checkpoint on disk again."""


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: ServerConfig = ServerConfig()
    paths: PathsConfig = PathsConfig()
    inference: InferenceConfig = InferenceConfig()
    stockfish: StockfishConfig = StockfishConfig()
    games: GamesConfig = GamesConfig()
    ladder: LadderConfig = LadderConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()


def load_config(path: Path | None = None, environ: Mapping[str, str] | None = None) -> Config:
    """Load the configuration file (if any) and apply environment variable overrides."""
    if environ is None:
        environ = os.environ
    if path is None and environ.get(CONFIG_PATH_ENV):
        path = Path(environ[CONFIG_PATH_ENV])
    if path is None and DEFAULT_CONFIG_FILE.is_file():
        path = DEFAULT_CONFIG_FILE

    data: dict[str, Any] = {}
    if path is not None:
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except OSError as e:
            raise ConfigError(f"cannot read config file {path}: {e.strerror}") from e
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML in config file {path}: {e}") from e

    for section_name, section_field in Config.model_fields.items():
        section_model = section_field.annotation
        assert section_model is not None
        for key in section_model.model_fields:
            env_name = f"{ENV_PREFIX}{section_name}_{key}".upper()
            if env_name in environ:
                section = data.setdefault(section_name, {})
                if not isinstance(section, dict):
                    raise ConfigError(f"[{section_name}] in {path} must be a table")
                section[key] = environ[env_name]

    try:
        return Config.model_validate(data)
    except ValidationError as e:
        source = f"config file {path} or environment" if path else "environment"
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in e.errors()
        )
        raise ConfigError(f"invalid configuration in {source}: {problems}") from e
