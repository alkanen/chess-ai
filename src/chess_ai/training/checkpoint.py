"""What a checkpoint holds, and how to read one back.

A checkpoint has to be enough on its own. Six months from now a run directory may be the only
thing left of an experiment, and loading a checkpoint to play against has to work without the
config file that produced it still being on the disk at the same path — so the config and the
encoder spec travel inside the file, next to the weights.

The optimizer state is in there too, which roughly doubles the size for AdamW and is what makes
resuming a crashed run free rather than a fresh start with warm weights.

Written as plain dicts, tensors and numbers only, so that loading can refuse to execute anything:
``torch.load`` is told ``weights_only=True``, which is the difference between reading a file and
running it.
"""

from pathlib import Path
from typing import Any, Final

import torch

from chess_ai.encoders import EncoderSpec

FORMAT_VERSION: Final = 1
"""The version of a checkpoint's contents, checked when one is loaded."""


class CheckpointError(Exception):
    """A checkpoint cannot be read, or is not one this code understands."""


def build(
    *,
    step: int,
    run: str,
    seed: int,
    architecture: str,
    model_options: dict[str, Any],
    spec: EncoderSpec,
    config: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Everything that goes into one checkpoint file."""
    return {
        "format_version": FORMAT_VERSION,
        "step": step,
        "run": run,
        "seed": seed,
        "architecture": architecture,
        "model_options": dict(model_options),
        "encoder": spec.model_dump(mode="json"),
        "config": config,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "metrics": dict(metrics or {}),
    }


def save(payload: dict[str, Any], path: Path) -> None:
    """Write ``payload`` to ``path``. The run store decides where, and renames it into place."""
    torch.save(payload, path)


def load(path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Read a checkpoint, or raise :exc:`CheckpointError` saying why it cannot be read."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except FileNotFoundError as e:
        raise CheckpointError(f"no checkpoint at {path}") from e
    except OSError as e:
        raise CheckpointError(f"cannot read checkpoint {path}: {e.strerror or e}") from e
    except Exception as e:  # noqa: BLE001 - torch raises several unrelated types for bad files
        raise CheckpointError(f"cannot read checkpoint {path}: {e}") from e
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path} is not a checkpoint")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"{path} is checkpoint format version {version!r}, and this is version {FORMAT_VERSION}"
        )
    return payload


def spec_of(payload: dict[str, Any]) -> EncoderSpec:
    """The encoder spec a checkpoint was trained against, as a spec again."""
    try:
        return EncoderSpec.model_validate(payload["encoder"])
    except (KeyError, ValueError) as e:
        raise CheckpointError(f"checkpoint does not say what encoder it was trained on: {e}") from e
