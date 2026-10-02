"""The experiment config: one file that says everything about a run.

A run is started from a config file and nothing else, so that a run is reproducible from what is
on disk rather than from what was typed. Everything a run depends on is in here — the dataset,
the input representation, the architecture and its size, the optimizer and the schedule, what to
validate and how often, what to keep, and the seed — and the file is copied into the run
directory verbatim, comments and all.

The encoder and model sections accept whatever options the thing they name accepts. That is what
makes adding an architecture free: register it, and its hyperparameters are config keys, with no
schema here to edit. Unknown keys are caught against the constructor's own signature, so a typo
stops the run instead of quietly training at a default; see :mod:`chess_ai.registry`.

::

    name = "mlp-baseline"
    seed = 1234

    [dataset]
    name = "carlsen"

    [encoder]
    name = "board-planes"
    rating_scale = 5000.0

    [model]
    architecture = "mlp"
    depth = 3
    width = 1024
"""

import re
import tomllib
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chess_ai.config import Device
from chess_ai.encoders import BOARD_PLANES
from chess_ai.training.run_store import CheckpointPolicy

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

VALIDATION_METRICS: Final = {
    "loss": False,
    "policy_loss": False,
    "value_loss": False,
    "top1": True,
    "top5": True,
    "illegal_top_move_rate": False,
}
"""The validation metrics a checkpoint policy may be judged on, and whether higher is better."""


class ExperimentError(Exception):
    """An experiment config is missing, unreadable, or does not say what it must."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _OpenSection(BaseModel):
    """A section whose extra keys are options for whatever it names."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @property
    def options(self) -> dict[str, Any]:
        """The keys beyond this section's own, which are passed on as options."""
        return dict(self.model_extra or {})


class DatasetSection(_Section):
    """Which dataset to train on. It is found by name in the configured data directory."""

    name: str


class EncoderSection(_OpenSection):
    """Which input representation to use, and its options."""

    name: str = BOARD_PLANES


class ModelSection(_OpenSection):
    """Which architecture to train, and its size hyperparameters."""

    architecture: str = "mlp"


class TrainingSection(_Section):
    """How the loop runs: batch size, what to train on, and how the two losses are weighted."""

    device: Device = "auto"
    batch_size: int = Field(default=1024, gt=0)
    value_loss_weight: float = Field(default=0.5, ge=0.0)
    """How much the win/draw/loss head counts next to the policy. The policy is the point;
    the value head is there so the network has some notion of who is winning."""
    mixed_precision: bool = True
    """Run the forward and backward passes in bfloat16 on a GPU that supports it."""
    data_workers: int = Field(default=4, ge=0)
    """Processes that read and encode batches. 0 does it in the training process."""
    log_every_steps: int = Field(default=50, gt=0)


class OptimizerSection(_Section):
    """AdamW, and the gradient clipping that goes with it."""

    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    decay_biases_and_norms: bool = False
    """Whether weight decay also shrinks biases and normalization scales and shifts.

    Off by default, as is usual: decay is for the weights that multiply an input. Runs made
    before the trainer told the two apart decayed everything, and this brings that back."""
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.95, ge=0.0, lt=1.0)
    epsilon: float = Field(default=1e-8, gt=0.0)
    gradient_clip: float = Field(default=1.0, ge=0.0)
    """Maximum gradient norm; 0 turns clipping off."""


class ScheduleSection(_Section):
    """How long to train, and how the learning rate moves over that time.

    Linear warmup, then cosine decay to a floor. The floor is a fraction of the learning rate
    rather than an absolute value so that changing the learning rate does not silently change
    the shape of the schedule.
    """

    steps: int = Field(default=10_000, gt=0)
    warmup_steps: int = Field(default=500, ge=0)
    min_learning_rate_fraction: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("warmup_steps")
    @classmethod
    def _not_longer_than_the_run(cls, value: int, info) -> int:
        steps = info.data.get("steps")
        if steps is not None and value > steps:
            raise ValueError(f"warmup_steps {value} is longer than the whole run of {steps} steps")
        return value


class ValidationSection(_Section):
    """How often to validate at least, and on how much of the validation split.

    The same positions every time, taken from the front of the split: comparing a curve across
    steps only means something if the numbers are measured on the same thing. Raising
    ``positions`` costs time per validation and buys a less noisy curve.

    ``every_steps`` is a floor rather than the whole story. A checkpoint step validates as well,
    so that the metrics indexed with a checkpoint were measured on that checkpoint; when the two
    schedules do not divide each other that is more validation than this asks for, and the run
    says how much before it starts.
    """

    every_steps: int = Field(default=500, gt=0)
    positions: int = Field(default=16_384, gt=0)


class CheckpointSection(_Section):
    """What to save and what to keep.

    ``metric`` names one of the validation metrics and decides which checkpoint is the best one;
    it is kept whatever ``keep`` says, since it is the one worth playing against.
    """

    every_steps: int = Field(default=1000, gt=0)
    keep: int = Field(default=3, ge=1)
    metric: str = "policy_loss"

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in VALIDATION_METRICS:
            raise ValueError(
                f"unknown metric {value!r}; choose one of {', '.join(sorted(VALIDATION_METRICS))}"
            )
        return value

    def policy(self) -> CheckpointPolicy:
        """This section as the run store's retention policy."""
        return CheckpointPolicy(
            keep=self.keep, metric=self.metric, higher_is_better=VALIDATION_METRICS[self.metric]
        )


class InitializeSection(_Section):
    """Another run's checkpoint to start the weights from, instead of drawing them from the seed.

    Weights only: the optimizer, the schedule and the step all start afresh, on whatever dataset
    this config names. That is fine-tuning, and it is how a curriculum goes from everything to
    the strongest players. The architecture, its options and the encoder have to be the ones the
    checkpoint was trained with, because the weights only mean anything in the shapes they were
    trained in.
    """

    run: str
    """The run to take the weights from, by name, in the runs directory."""
    checkpoint: Literal["best", "latest"] | int = "best"
    """Which of its checkpoints: its best by its own metric, its newest, or the one from a step."""

    @field_validator("run")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(f"invalid run name {value!r}")
        return value

    @field_validator("checkpoint")
    @classmethod
    def _a_step(cls, value: str | int) -> str | int:
        if isinstance(value, int) and value < 0:
            raise ValueError(f"a checkpoint step is not negative, and {value} is")
        return value


class ExperimentConfig(BaseModel):
    """A whole experiment: what to train, on what, how, and for how long."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """Also the run directory's name, so it has to be a plain one."""
    seed: int = Field(default=0, ge=0, lt=2**64)
    """Draws the initial weights and the order positions are visited in.

    Bounded at both ends, because the two generators it is given to disagree about what they
    will take and each objects only at one end. numpy's ``SeedSequence`` rejects negative
    entropy and accepts integers of any size; ``torch.manual_seed`` takes a negative number
    happily and raises above ``0xffff_ffff_ffff_ffff``. Either way the config would load, and
    the failure would come later as a traceback rather than as a word about the file — so the
    field is bounded by what both accept. ``tomllib`` does not enforce TOML's own 64-bit
    integer range, so a config really can carry a number this large.
    """
    dataset: DatasetSection
    encoder: EncoderSection = EncoderSection()
    model: ModelSection = ModelSection()
    training: TrainingSection = TrainingSection()
    optimizer: OptimizerSection = OptimizerSection()
    schedule: ScheduleSection = ScheduleSection()
    validation: ValidationSection = ValidationSection()
    checkpoints: CheckpointSection = CheckpointSection()
    initialize_from: InitializeSection | None = None

    @field_validator("name")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(
                f"invalid run name {value!r}: use letters, digits and . _ -, "
                "starting with a letter or digit"
            )
        return value

    def resolved(self) -> dict[str, Any]:
        """This config with every default filled in, for the run directory to record."""
        return self.model_dump(mode="json")


def load_experiment(path: Path, *, name: str | None = None) -> ExperimentConfig:
    """Read an experiment config, or raise :exc:`ExperimentError` saying what is wrong with it.

    A config that does not name the run takes the file's own name, so that a config file and the
    run it produces are called the same thing without saying so twice. ``name`` overrides both,
    which is how the same config is run more than once.
    """
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except OSError as e:
        raise ExperimentError(f"cannot read experiment config {path}: {e.strerror or e}") from e
    except tomllib.TOMLDecodeError as e:
        raise ExperimentError(f"invalid TOML in experiment config {path}: {e}") from e
    if not isinstance(data, dict):  # pragma: no cover - tomllib always gives a table
        raise ExperimentError(f"experiment config {path} is not a table")
    if name is not None:
        data["name"] = name
    data.setdefault("name", path.stem)
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in e.errors()
        )
        raise ExperimentError(f"invalid experiment config {path}: {problems}") from e
