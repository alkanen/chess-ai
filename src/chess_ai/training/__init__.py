"""Training: an experiment config in, a run directory out.

- :mod:`~chess_ai.training.experiment` is the config file: everything a run depends on
- :mod:`~chess_ai.training.run_store` is the run directory, and the only thing that knows its
  layout; the web server reads runs through it and never through the trainer
- :mod:`~chess_ai.training.batches` reads and encodes batches, in worker processes
- :mod:`~chess_ai.training.trainer` is the loop
- :mod:`~chess_ai.training.validate` measures a model on held-back games
- :mod:`~chess_ai.training.checkpoint` is what a checkpoint holds
- :mod:`~chess_ai.training.hardware` picks the device and reports on it
"""

from chess_ai.training.batches import Batch, PositionBatches, batch_loader
from chess_ai.training.experiment import (
    VALIDATION_METRICS,
    ExperimentConfig,
    ExperimentError,
    load_experiment,
)
from chess_ai.training.hardware import HardwareError, describe_device, resolve_device
from chess_ai.training.run_store import (
    CheckpointChoice,
    CheckpointInfo,
    CheckpointPolicy,
    GpuStats,
    Heartbeat,
    RunError,
    RunInfo,
    RunNotes,
    RunReader,
    RunStatus,
    RunWriter,
    choose_checkpoint,
    list_runs,
    open_run,
    run_path,
    save_notes,
)
from chess_ai.training.trainer import TrainingError, train
from chess_ai.training.validate import validate

__all__ = [
    "VALIDATION_METRICS",
    "Batch",
    "CheckpointChoice",
    "CheckpointInfo",
    "CheckpointPolicy",
    "ExperimentConfig",
    "ExperimentError",
    "GpuStats",
    "HardwareError",
    "Heartbeat",
    "PositionBatches",
    "RunError",
    "RunInfo",
    "RunNotes",
    "RunReader",
    "RunStatus",
    "RunWriter",
    "TrainingError",
    "batch_loader",
    "choose_checkpoint",
    "describe_device",
    "list_runs",
    "load_experiment",
    "open_run",
    "resolve_device",
    "run_path",
    "save_notes",
    "train",
    "validate",
]
