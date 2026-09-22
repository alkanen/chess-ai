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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: ServerConfig = ServerConfig()


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
