"""The training loop: one experiment config in, one run directory out.

Supervised next-move prediction, exactly as the PRD frames it: show the network a position, have
it predict the move the human played, repeat. The loss is cross-entropy over the shared move
vocabulary, plus a weighted cross-entropy on the win/draw/loss head, whose target is how the game
the position came from actually ended, from the mover's point of view.

Everything the run produces goes through the run store, and nothing is held in memory that a
crash would lose: metrics are appended as they are measured, the heartbeat is replaced as the run
moves, and checkpoints are written as they are made. A run is therefore worth exactly as much
after it crashes as it was a moment before.

A run can be stopped and carried on. SIGINT or SIGTERM asks it to stop: it finishes the step it
is on, saves a checkpoint and says it stopped, and :func:`resume` later carries on from that
checkpoint — or from the last one a crashed run saved — exactly where it was, weights, optimizer,
schedule, data order and random state and all, so that a run stopped and resumed learns what it
would have learned in one go. A new run can also start from another run's weights instead of
from its seed (``[initialize_from]``), which is how fine-tuning works: weights only, into a run
with a dataset, an optimizer and a schedule of its own.

The loop keeps two things honest about speed. It never reads a loss back from the GPU except when
it is about to log one, because every read is a stall; and before it starts it measures a few real
steps and says how fast they were, so a two-day run can be recognised as a two-day run before it
is two days in.
"""

import copy
import logging
import math
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from math import lcm
from pathlib import Path
from typing import Any, Final, NamedTuple

import torch
from pydantic import ValidationError
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset

from chess_ai.dataset import TRAIN, VALIDATION, Dataset, DatasetError, ManifestError, open_dataset
from chess_ai.encoders import Encoder, EncoderSpec, create_encoder
from chess_ai.models import ChessModel, create_model
from chess_ai.move_codec import VOCABULARY_SIZE
from chess_ai.registry import RegistryError
from chess_ai.training import checkpoint
from chess_ai.training.batches import (
    STOP_SIGNALS,
    Batch,
    PositionBatches,
    batch_loader,
    holding_stop_signals,
    iterate,
)
from chess_ai.training.checkpoint import CheckpointError
from chess_ai.training.experiment import ExperimentConfig, OptimizerSection
from chess_ai.training.hardware import (
    HardwareError,
    autocast_dtype,
    describe_device,
    gpu_stats,
    resolve_device,
)
from chess_ai.training.run_store import TRAIN as TRAIN_SPLIT
from chess_ai.training.run_store import VALIDATION as VALIDATION_SPLIT
from chess_ai.training.run_store import (
    CheckpointPolicy,
    DatasetReference,
    Lineage,
    ModelReference,
    RunError,
    RunInfo,
    RunReader,
    RunStatus,
    RunWriter,
    check_available,
    choose_checkpoint,
    code_version,
    open_run,
    run_path,
)
from chess_ai.training.validate import validate

LOGGER = logging.getLogger(__name__)

HEARTBEAT_SECONDS: Final = 2.0
"""How often at most the heartbeat is rewritten, however fast the steps go by."""

PROBE_STEPS: Final = 3
"""Steps the throughput probe times. Enough to be past the first step's setup costs."""


class TrainingError(Exception):
    """Something about this run cannot work, and the person who asked for it can fix it."""


class TrainingStopped(Exception):  # noqa: N818 - it is not an error, and saying so is the point
    """A run was asked to stop by a signal, and did: its checkpoint is saved, and it says so.

    Raised rather than returned, so that nothing that waits for a run can take a run that
    stopped half-way for one that finished.
    """

    def __init__(self, directory: Path, step: int, signal_number: int) -> None:
        super().__init__(f"stopped by {signal.Signals(signal_number).name} at step {step:,}")
        self.directory = directory
        self.step = step
        self.signal = signal_number


@dataclass
class _Progress:
    """Where the loop has got to, for a heartbeat written by whoever catches the way it ended.

    A run that stopped or crashed has to say at which step, and the place that knows is the loop
    while the place that handles it is outside the loop. One mutable object between them is less
    to go wrong than two copies of the handling.
    """

    step: int = 0
    epoch: float = 0.0


@dataclass
class _Start:
    """Where the loop starts: at the beginning, or where a resumed run's checkpoint left off."""

    step: int = 0
    elapsed: float = 0.0
    """Seconds the run had trained for by ``step``, so that its clock carries on from there."""
    rng_state: dict[str, torch.Tensor] | None = None
    """The random state to carry on from; ``None`` to keep what the seed gave."""


@dataclass
class _Run:
    """What the loop needs, once the config has been resolved against the actual machine."""

    config: ExperimentConfig
    device: torch.device
    dtype: torch.dtype | None
    dataset: Dataset
    encoder: Encoder
    spec: EncoderSpec
    model: ChessModel
    batches: PositionBatches


def train(
    config: ExperimentConfig,
    *,
    data_dir: Path,
    runs_dir: Path,
    config_text: str,
    overwrite: bool = False,
    say: Callable[[str], None] = print,
) -> Path:
    """Train what ``config`` describes, and return the run directory it wrote.

    Raises :exc:`TrainingError` for anything the config or the machine is wrong about, which is
    everything a person running this can do something about.
    """
    directory = run_path(runs_dir, config.name)
    try:
        # Before anything expensive: being told the name is taken is worth nothing once the
        # dataset is open and the throughput probe has run.
        check_available(directory, config.name, overwrite=overwrite)
    except RunError as e:
        raise TrainingError(e) from e

    run = _prepare(config, data_dir=data_dir)
    # Before the probe, which puts back whatever weights it found: these are the ones to keep.
    lineage = _initialize(run, runs_dir=runs_dir) if config.initialize_from else None
    estimate = _probe(run)
    for line in _summary(run, estimate, lineage=lineage):
        say(line)

    try:
        writer = RunWriter.create(
            directory,
            _info(run, estimate=estimate, lineage=lineage),
            config_text=config_text,
            policy=config.checkpoints.policy(),
            overwrite=overwrite,
        )
    except RunError as e:
        raise TrainingError(e) from e
    say(f"chess-ai: run {config.name} in {directory}")
    _train_to_end(run, writer, _optimizer(run), _Start(), say=say)
    return directory


def resume(
    name: str, *, data_dir: Path, runs_dir: Path, say: Callable[[str], None] = print
) -> Path:
    """Carry on training the run called ``name`` from its latest checkpoint, to the end.

    Everything comes from the checkpoint — the config, the weights, the optimizer's moments, the
    step, the random state and how long the run had been going — and the dataset is opened by
    name again and checked to be the one the run was trained on, since the order positions are
    visited in is a function of the dataset, the seed and the step. The learning rate schedule is
    a function of the step too, so it carries on as it was: a run resumes the schedule it was
    started with, and cannot be made longer by resuming it.

    Raises:
        TrainingError: the run cannot be resumed, and the message says why.
        TrainingStopped: it was stopped again.
    """
    try:
        directory = open_run(runs_dir, name).directory
        # Taken before anything is read, so that what is read cannot change underneath: a
        # trainer still writing this run would be saving checkpoints past the one chosen here.
        writer = RunWriter.reopen(directory, policy=CheckpointPolicy())
    except RunError as e:
        raise TrainingError(e) from e
    try:
        run, optimizer, start = _resumable(RunReader(directory), data_dir=data_dir, say=say)
        writer.policy = run.config.checkpoints.policy()
        writer.rewind(start.step)
    except RunError as e:
        writer.close()
        raise TrainingError(e) from e
    except BaseException:
        writer.close()
        raise
    say(f"chess-ai: resuming run {name} in {directory}")
    _train_to_end(run, writer, optimizer, start, say=say)
    return directory


def _resumable(
    reader: RunReader, *, data_dir: Path, say: Callable[[str], None]
) -> tuple[_Run, torch.optim.Optimizer, _Start]:
    """The run in ``reader`` put back as its latest checkpoint left it, ready to carry on."""
    name = reader.name
    status = reader.status
    if status is not None and status.status == RunStatus.FINISHED:
        raise TrainingError(f"run {name!r} has already finished all {status.steps:,} steps")
    latest = reader.latest_checkpoint()
    if latest is None:
        raise TrainingError(
            f"run {name!r} saved no checkpoint, so there is nothing to resume it from; "
            "start it again with 'chess-ai train --overwrite'"
        )
    payload = _load(reader.checkpoint_path(latest), doing=f"resume run {name!r}")
    config = _config_of(payload, name)
    run = _prepare(config, data_dir=data_dir)
    _same_dataset(run, reader)
    doing = f"resume run {name!r} from step {latest.step:,}"
    _check_compatible(run, payload, doing=doing)
    estimate = _probe(run)
    # After the probe, which puts back the weights it found rather than the ones it left.
    _load_weights(run, payload, doing=doing)
    optimizer = _optimizer(run)
    try:
        optimizer.load_state_dict(payload["optimizer_state"])
    except (KeyError, ValueError) as e:
        raise TrainingError(
            f"cannot {doing}: its optimizer state does not fit this run's optimizer: {e}"
        ) from e
    start = _Start(
        step=latest.step,
        elapsed=_elapsed_at(payload, reader, latest.step),
        rng_state=payload.get("rng_state") or None,
    )
    for line in _summary(run, estimate, start=start):
        say(line)
    return run, optimizer, start


def _config_of(payload: dict[str, Any], name: str) -> ExperimentConfig:
    """The config a checkpoint was trained with, as this code reads configs.

    Runs made before weight decay left biases and norms alone decayed everything, and their
    optimizer state has the one parameter group to show for it; their config does not say so,
    because the setting did not exist. The new default would build two groups and fail to load
    the state, so a config without the setting gets the behaviour it was trained with.
    """
    data = copy.deepcopy(payload.get("config") or {})
    if isinstance(data.get("optimizer"), dict):
        data["optimizer"].setdefault("decay_biases_and_norms", True)
    data["name"] = name
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in e.errors()
        )
        raise TrainingError(
            f"run {name!r} was trained with a config this code no longer accepts: {problems}"
        ) from e


def _same_dataset(run: _Run, reader: RunReader) -> None:
    """Refuse to resume on a dataset other than the one the run was training on.

    A dataset rebuilt under the same name has other positions in its train split, or the same
    ones in another order, and a resumed run would visit them in an order the steps before it
    never saw — revisiting some and skipping others, which is not the run that was stopped.
    """
    recorded = reader.info.dataset
    manifest = run.dataset.manifest
    there = (recorded.train_positions, recorded.created)
    here = (_split_positions(manifest, TRAIN), manifest.created)
    if there != here:
        raise TrainingError(
            f"dataset {manifest.name!r} is not the one run {reader.name!r} was training on "
            f"({there[0]:,} train positions, built {there[1]}; now {here[0]:,}, built {here[1]}), "
            "so a resumed run would not carry on where it was. Start a new run with "
            "[initialize_from] to train its weights on this dataset instead"
        )


def _elapsed_at(payload: dict[str, Any], reader: RunReader, step: int) -> float:
    """How long the run had trained for by ``step``, so that its clock carries on from there.

    Checkpoints written before they recorded it leave the metrics log to say, which has the
    time of every line it wrote; the last one up to ``step`` is close enough for a chart.
    """
    elapsed = payload.get("elapsed")
    if isinstance(elapsed, int | float):
        return float(elapsed)
    times = [
        line["elapsed"]
        for line in reader.stream_metrics()
        if isinstance(line.get("step"), int)
        and line["step"] <= step
        and isinstance(line.get("elapsed"), int | float)
    ]
    return float(max(times, default=0.0))


def _initialize(run: _Run, *, runs_dir: Path) -> Lineage:
    """Load the weights ``[initialize_from]`` names into ``run``, and say where they came from."""
    source = run.config.initialize_from
    assert source is not None
    if source.run == run.config.name:
        raise TrainingError(
            f"run {run.config.name!r} cannot be initialized from itself; give it another name"
        )
    try:
        reader = open_run(runs_dir, source.run)
        chosen = choose_checkpoint(reader, source.checkpoint)
    except RunError as e:
        raise TrainingError(f"cannot initialize from run {source.run!r}: {e}") from e
    doing = f"initialize from run {source.run!r} at step {chosen.step:,}"
    payload = _load(reader.checkpoint_path(chosen), doing=doing)
    _check_compatible(run, payload, doing=doing)
    _load_weights(run, payload, doing=doing)
    trained_on = (payload.get("config") or {}).get("dataset") or {}
    return Lineage(
        run=source.run,
        step=chosen.step,
        checkpoint=source.checkpoint,
        dataset=trained_on.get("name") if isinstance(trained_on, dict) else None,
    )


def _load(path: Path, *, doing: str) -> dict[str, Any]:
    try:
        return checkpoint.load(path)
    except CheckpointError as e:
        raise TrainingError(f"cannot {doing}: {e}") from e


def _check_compatible(run: _Run, payload: dict[str, Any], *, doing: str) -> None:
    """Refuse a checkpoint whose weights were trained for another architecture or input.

    Asked before the weights are loaded, because the answer is worth more than the shape error
    loading them would give — and because an encoder with the same shapes and other options
    (another rating scale, another orientation) would load without one, into a model that then
    reads every input differently from how it learned to.
    """
    problems = []
    architecture = payload.get("architecture")
    if architecture != run.config.model.architecture:
        problems.append(
            f"it is a {architecture!r} and this run trains a {run.config.model.architecture!r}"
        )
    try:
        spec = checkpoint.spec_of(payload)
    except CheckpointError as e:
        problems.append(str(e))
    else:
        if spec != run.spec:
            problems.append(
                f"it was trained on {_describe_spec(spec)} and this run's encoder gives "
                f"{_describe_spec(run.spec)}"
            )
    if problems:
        raise TrainingError(f"cannot {doing}: {'; '.join(problems)}")


def _load_weights(run: _Run, payload: dict[str, Any], *, doing: str) -> None:
    """Load a checkpoint's weights, or say which model options they were trained with.

    Options that leave the weights' shapes alone, such as dropout, may differ: fine-tuning with
    more of it is an ordinary thing to want.
    """
    try:
        run.model.load_state_dict(payload["model_state"])
    except (KeyError, RuntimeError) as e:
        raise TrainingError(
            f"cannot {doing}: its weights do not fit this model "
            f"(trained with{_options(payload.get('model_options') or {}) or ' no options'}, "
            f"and this run has{_options(run.config.model.options) or ' none'}): {e}"
        ) from e


def _describe_spec(spec: EncoderSpec) -> str:
    return f"{spec.describe()}{_options(spec.options)}"


def _train_to_end(
    run: _Run,
    writer: RunWriter,
    optimizer: torch.optim.Optimizer,
    start: _Start,
    *,
    say: Callable[[str], None],
) -> None:
    """Run the loop with ``writer``, and have the run say how it ended whichever way it does."""
    progress = _Progress(step=start.step, epoch=_epoch(run, start.step))
    with writer, _stop_signals() as stop:
        try:
            stopped_by = _loop(run, writer, progress, optimizer, start, stop=stop, say=say)
        except KeyboardInterrupt:
            # Only a second ctrl-c, or one that came before there was a loop to stop: the
            # first is caught and stops the run at the end of its step, with a checkpoint.
            steps = run.config.schedule.steps
            if progress.step == steps and _saved_at(writer, steps):
                # The last step's checkpoint is saved, so the run had finished, and all that
                # was interrupted was tidying up after it.
                _finished_anyway(writer, run, progress, say=say)
                return
            if not _saved_at(writer, progress.step):
                message = "interrupted before a checkpoint could be saved"
                _ended(writer, RunStatus.STOPPED, run, progress, message)
                raise
            # A checkpoint of the step the run had got to is on the disk, so whatever was
            # interrupted, such as letting go of the loader after the stop checkpoint, cost
            # nothing: the run stopped there, and a resume carries on from it.
            stopped_by = stop.signal or signal.SIGINT
        except RunError as e:
            # A disk filling up an hour into a run is an ordinary accident: the run says it
            # crashed, and the person running it gets a sentence rather than a traceback.
            _ended(writer, RunStatus.CRASHED, run, progress, f"{type(e).__name__}: {e}")
            raise TrainingError(e) from e
        except Exception as e:
            _ended(writer, RunStatus.CRASHED, run, progress, f"{type(e).__name__}: {e}")
            raise
        if stopped_by is not None:
            message = f"stopped by {signal.Signals(stopped_by).name}"
            _ended(writer, RunStatus.STOPPED, run, progress, message)
            raise TrainingStopped(writer.directory, progress.step, stopped_by)


def _finished_anyway(
    writer: RunWriter, run: _Run, progress: _Progress, *, say: Callable[[str], None]
) -> None:
    """Say a run that was interrupted after its last checkpoint finished, unless it already has.

    Asked of the heartbeat rather than remembered, for the same reason as :func:`_saved_at`: the
    interruption can fall after the run said it finished and before anything here noted it,
    and the heartbeat it wrote then has the whole run's throughput, which this one cannot.
    """
    try:
        status = RunReader(writer.directory).status
    except RunError:
        status = None
    if status is None or status.status != RunStatus.FINISHED:
        steps = run.config.schedule.steps
        writer.finish(RunStatus.FINISHED, step=steps, steps=steps, epoch=round(progress.epoch, 4))
    say(f"chess-ai: finished {run.config.schedule.steps:,} steps")


def _saved_at(writer: RunWriter, step: int) -> bool:
    """Whether the run has a checkpoint of ``step`` on the disk.

    Asked of the disk rather than remembered, because what it answers is how a run that was
    interrupted ended, and an interruption can fall between a checkpoint landing and anything
    here noting that it did. A checkpoint file is only there once it is whole, and the loop
    holds stop signals back until its save is done, index and all, so a file that is there
    is a checkpoint that was saved.
    """
    latest = RunReader(writer.directory).latest_checkpoint()
    return latest is not None and latest.step == step


class _StopRequest:
    """Which signal asked the run to stop, once one has."""

    def __init__(self) -> None:
        self.signal: int | None = None


@contextmanager
def _stop_signals() -> Iterator[_StopRequest]:
    """Turn SIGINT and SIGTERM into a request the loop answers at the end of its step.

    A step interrupted half-way would leave the optimizer half-updated, which is no state to
    save. So the first signal only says the run should stop, and the loop stops when the step is
    whole. A second ctrl-c is somebody who will not wait, and interrupts at once, as ctrl-c
    always did. A second SIGTERM is not: it comes from a machine, and often twice — ``uv run``
    forwards it to the trainer, so one sent to the process group, as a service manager stopping
    the run sends it, arrives twice — and SIGKILL is there for a stop that cannot wait.

    Nothing is printed from the handler, which can run in the middle of another print. Python
    only lets the main thread handle signals, so anywhere else this does nothing, and the
    handlers there were before are put back afterwards.
    """
    request = _StopRequest()
    if threading.current_thread() is not threading.main_thread():
        yield request
        return

    def now(signal_number: int, frame: Any) -> None:
        raise KeyboardInterrupt

    def stop(signal_number: int, frame: Any) -> None:
        if request.signal is None:
            request.signal = signal_number
        signal.signal(signal.SIGINT, now)

    previous = {each: signal.signal(each, stop) for each in STOP_SIGNALS}
    try:
        yield request
    finally:
        for each, handler in previous.items():
            signal.signal(each, handler)


def _ended(
    writer: RunWriter, status: RunStatus, run: _Run, progress: _Progress, message: str
) -> None:
    """Say how a run ended, and where it had got to when it did."""
    writer.heartbeat(
        status,
        step=progress.step,
        steps=run.config.schedule.steps,
        epoch=round(progress.epoch, 4),
        message=message,
    )


def _prepare(config: ExperimentConfig, *, data_dir: Path) -> _Run:
    """Resolve the config against this machine and this dataset, or say why it cannot be."""
    try:
        device = resolve_device(config.training.device)
    except HardwareError as e:
        raise TrainingError(e) from e
    dtype = autocast_dtype(device, mixed_precision=config.training.mixed_precision)

    try:
        dataset = open_dataset(config.dataset.name, data_dir=data_dir)
    except (DatasetError, ManifestError) as e:
        raise TrainingError(_with_available(e, data_dir)) from e
    if dataset.manifest.move_vocabulary_size != VOCABULARY_SIZE:
        raise TrainingError(
            f"dataset {config.dataset.name!r} was built against a move vocabulary of "
            f"{dataset.manifest.move_vocabulary_size} moves and this code has "
            f"{VOCABULARY_SIZE}; its move indices mean something else, so rebuild it"
        )
    if not dataset.manifest.splits.get(TRAIN, None) or not dataset.manifest.splits[TRAIN].positions:
        raise TrainingError(f"dataset {config.dataset.name!r} has nothing in its train split")

    # Seeded before the model is built, because building it is what draws the initial weights.
    torch.manual_seed(config.seed)
    try:
        encoder = create_encoder(config.encoder.name, **config.encoder.options)
        model = create_model(config.model.architecture, encoder.spec, **config.model.options)
    except (RegistryError, ValueError) as e:
        raise TrainingError(e) from e
    try:
        batches = PositionBatches(
            dataset.directory,
            TRAIN,
            encoder=config.encoder.name,
            encoder_options=config.encoder.options,
            batch_size=config.training.batch_size,
            batches=config.schedule.steps,
            seed=config.seed,
        )
    except (DatasetError, ValueError) as e:
        raise TrainingError(e) from e
    return _Run(
        config=config,
        device=device,
        dtype=dtype,
        dataset=dataset,
        encoder=encoder,
        spec=encoder.spec,
        model=model.to(device),
        batches=batches,
    )


def _optimizer(run: _Run) -> torch.optim.Optimizer:
    settings = run.config.optimizer
    return torch.optim.AdamW(
        _parameter_groups(run.model, settings),
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
    )


def _parameter_groups(model: nn.Module, settings: OptimizerSection) -> list[dict[str, Any]]:
    """The model's parameters, split into those weight decay shrinks and those it leaves alone.

    Decay is for weight matrices and convolution kernels, the parameters that multiply an input.
    A bias or a normalization layer's scale and shift only moves or rescales what comes out, and
    pulling those towards zero is not regularization but a bias towards an arbitrary output; for
    a scale it also undoes the normalization's own say over how loud a channel is. They are the
    one-dimensional parameters, in every architecture here, which is the rule used to find them.

    ``decay_biases_and_norms`` puts everything back in one decayed group, as runs before this
    rule were trained, so that those runs can still be reproduced.
    """
    if settings.decay_biases_and_norms:
        return [{"params": list(model.parameters()), "weight_decay": settings.weight_decay}]
    decayed = [p for p in model.parameters() if p.ndim > 1]
    undecayed = [p for p in model.parameters() if p.ndim <= 1]
    return [
        {"params": decayed, "weight_decay": settings.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def _schedule(
    run: _Run, optimizer: torch.optim.Optimizer, *, completed: int = 0
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to a fraction of the learning rate.

    ``completed`` is how many steps have already been taken, for a resumed run. The schedule is
    a function of the step and nothing else, so starting it there is restoring it exactly; it
    reads each group's starting rate from ``initial_lr``, which the optimizer state it was
    loaded from carries.
    """
    schedule = run.config.schedule
    warmup = schedule.warmup_steps
    floor = schedule.min_learning_rate_fraction
    decaying = max(1, schedule.steps - warmup)

    def factor(completed: int) -> float:
        if completed < warmup:
            return (completed + 1) / warmup
        progress = min(1.0, (completed - warmup) / decaying)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor, last_epoch=completed - 1)


def _losses(run: _Run, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The combined loss and its two halves, for one batch."""
    with torch.autocast(run.device.type, dtype=run.dtype, enabled=run.dtype is not None):
        out = run.model(batch.spatial, batch.globals)
        policy_loss = F.cross_entropy(out.policy, batch.move)
        value_loss = F.cross_entropy(out.value, batch.result)
    total = policy_loss + run.config.training.value_loss_weight * value_loss
    return total, policy_loss.detach(), value_loss.detach()


class _StepResult(NamedTuple):
    """What one step measured, as tensors still on the device."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    gradient_norm: torch.Tensor
    """The total norm of the gradient before any clipping."""
    clipped: torch.Tensor
    """1 if clipping scaled this step's gradient down, else 0."""


def _step(run: _Run, batch: Batch, optimizer: torch.optim.Optimizer) -> _StepResult:
    """One optimizer step. Returns what it measured as tensors, unread, so as not to stall."""
    total, policy_loss, value_loss = _losses(run, batch)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    parameters = [p for p in run.model.parameters() if p.grad is not None]
    # Measured whether or not anything is clipped: how large the gradient is says as much about
    # the learning rate as the loss does, and it is the number that says whether a clip
    # threshold is doing anything at all.
    norm = nn.utils.get_total_norm([p.grad for p in parameters])
    clip = run.config.optimizer.gradient_clip
    if clip:
        # The parameters, not their gradients: handed the gradients, this looks for *their*
        # gradients, finds none, and clips nothing without a word.
        nn.utils.clip_grads_with_norm_(parameters, clip, norm)
        clipped = norm > clip
    else:
        clipped = torch.zeros((), device=norm.device)
    optimizer.step()
    return _StepResult(total.detach(), policy_loss, value_loss, norm, clipped)


def _probe(run: _Run) -> float | None:
    """Positions per second over a few real steps, measured before the run starts.

    Real steps, with a real optimizer, on a real batch: a forward pass alone would flatter the
    estimate by the backward pass and the weight update, which together are most of a step. The
    weights are put back afterwards, so the run still starts from exactly the seeded
    initialization, and the throwaway optimizer never sees the run's own.
    """
    weights = {name: value.detach().clone() for name, value in run.model.state_dict().items()}
    optimizer = _optimizer(run)
    try:
        batch = run.batches[0].to(run.device)
        for index in range(PROBE_STEPS + 1):
            if index == 1:
                # The first step pays for CUDA's lazy setup and cuBLAS picking kernels, which a
                # run of thousands of steps should not be judged by.
                _synchronize(run.device)
                started = time.perf_counter()
            _step(run, batch, optimizer)
        _synchronize(run.device)
        elapsed = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError as e:
        raise TrainingError(
            f"the GPU ran out of memory on a batch of {run.config.training.batch_size}: {e}. "
            "Lower batch_size, or train on a smaller model"
        ) from e
    except Exception as e:  # noqa: BLE001 - the probe is the first thing to run any of this
        LOGGER.warning("could not measure throughput: %s", e)
        return None
    finally:
        run.model.load_state_dict(weights)
        # The probe took a real batch, which mapped shards in this process. The loader pickles
        # this object into each worker on a spawn or forkserver start method, and a mapped
        # shard pickles as the whole file, so it has to be handed over unopened.
        run.batches.close()
    return PROBE_STEPS * run.config.training.batch_size / elapsed if elapsed > 0 else None


def _loop(
    run: _Run,
    writer: RunWriter,
    progress: _Progress,
    optimizer: torch.optim.Optimizer,
    start: _Start,
    *,
    stop: _StopRequest,
    say: Callable[[str], None],
) -> int | None:
    """Train to the end of the schedule, logging, validating and checkpointing as it goes.

    Returns ``None`` at the end of the schedule, or the signal that stopped it before then,
    once the checkpoint of the step it stopped at is saved.
    """
    config = run.config
    steps = config.schedule.steps
    scheduler = _schedule(run, optimizer, completed=start.step)
    validation = _validation_split(run, say=say)

    run.model.train()
    session_started = time.perf_counter()
    """When this process started stepping, which the console's running time and estimate count
    from: a resumed run may be on another machine, or sharing it differently, than before."""
    started = session_started - start.elapsed
    window = _Window(run.device)
    latest: dict[str, float] = {}
    measured_at: int | None = None
    """The step ``latest`` was measured at, so a checkpoint is never indexed with another's."""
    throughput: float | None = None
    step = start.step
    positions_seen = step * run.batches.positions_per_batch
    """How many training positions the weights have been moved by, which is what a chart of
    runs with different batch sizes compares them on. Not ``positions``: a validation line
    already says how many positions it measured under that name."""
    beat = _Beat(writer, steps=steps)
    beat.write(RunStatus.RUNNING, step=step, epoch=progress.epoch, device=run.device, force=True)
    if start.rng_state is not None:
        # Last, so that nothing between here and the first step draws from it.
        _restore_rng(start.rng_state, run.device)

    for batch in _stream(run, first=start.step):
        batch = batch.to(run.device, non_blocking=run.device.type == "cuda")
        # Read before the scheduler moves on: this is the rate the weights just moved by, and
        # get_last_lr() after scheduler.step() is already the next step's.
        learning_rate = scheduler.get_last_lr()[0]
        window.add(_step(run, batch, optimizer), positions=len(batch))
        scheduler.step()
        step += 1
        positions_seen += len(batch)
        progress.step, progress.epoch = step, _epoch(run, step)
        elapsed = time.perf_counter() - started

        if step % config.training.log_every_steps == 0 or step == steps:
            speed = window.take()
            throughput = speed["positions_per_second"]
            writer.log(
                step=step,
                epoch=round(progress.epoch, 4),
                split=TRAIN_SPLIT,
                learning_rate=learning_rate,
                elapsed=round(elapsed, 3),
                positions_seen=positions_seen,
                **speed,
            )

        # Outside the logging branch, so that the heartbeat's cadence is HEARTBEAT_SECONDS
        # rather than log_every_steps: a reader cannot tell a run that has stopped reporting
        # from one that is merely between log lines. _Beat does the throttling.
        beat.write(
            RunStatus.RUNNING,
            step=step,
            epoch=progress.epoch,
            device=run.device,
            positions_per_second=throughput,
            eta_seconds=_eta(step, steps, elapsed),
        )

        saving = step % config.checkpoints.every_steps == 0 or step == steps
        # Validated whenever a checkpoint is about to be saved, as well as on its own schedule,
        # so that the metrics indexed with a checkpoint are measured on that checkpoint. The two
        # schedules need not divide each other, and metrics from an earlier step would make
        # "best" a lie — including handing it to a checkpoint that was never measured.
        if validation is not None and (
            saving or step % config.validation.every_steps == 0 or step == steps
        ):
            with window.paused():
                latest = _validate(run, validation)
            # Placed by the time and the positions the step was done at, as the training line
            # is, rather than after the validation: it measures the weights of that step, and
            # a chart against either lines the two up there.
            writer.log(
                step=step,
                epoch=round(progress.epoch, 4),
                split=VALIDATION_SPLIT,
                elapsed=round(elapsed, 3),
                positions_seen=positions_seen,
                **latest,
            )
            say(
                _validation_line(
                    step,
                    steps,
                    latest,
                    first=start.step,
                    running=time.perf_counter() - session_started,
                    # The shortest gap between two validations, checkpoints validating too.
                    settled=min(config.validation.every_steps, config.checkpoints.every_steps),
                )
            )
            measured_at = step

        # A stop asked for at the last step is a run that finished.
        stopping = stop.signal is not None and step < steps
        if stop.signal is not None and step == steps:
            say(f"chess-ai: {signal.Signals(stop.signal).name} at the last step: finishing the run")
        if stopping:
            say(
                f"chess-ai: {signal.Signals(stop.signal).name} at step {step:,}: "
                "saving a checkpoint and stopping"
            )
            saving = True

        if saving:
            metrics = latest if measured_at == step else {}
            with window.paused():
                save = _writer(
                    run,
                    optimizer,
                    step=step,
                    metrics=metrics,
                    elapsed=time.perf_counter() - started,
                )
                # Whole or not at all: the file lands before the index records it, prunes
                # and picks the best, and a checkpoint the index never heard of has no
                # metrics and can never be the best. A ctrl-c waits the seconds it takes.
                with holding_stop_signals():
                    path = writer.save_checkpoint(step, save, metrics=metrics)
            LOGGER.info("saved %s", path)

        if stopping:
            return stop.signal

    elapsed = time.perf_counter() - started
    writer.finish(
        RunStatus.FINISHED,
        step=step,
        steps=steps,
        epoch=round(_epoch(run, step), 4),
        positions_per_second=step * config.training.batch_size / elapsed if elapsed else None,
    )
    say(f"chess-ai: finished {steps:,} steps in {_duration(elapsed)}{_best(writer, config)}")
    return None


def _best(writer: RunWriter, config: ExperimentConfig) -> str:
    """Which checkpoint came out best, read back from the run rather than from the last one.

    The last validation is not the best one — a run that overfits is *worst* at the end — and the
    checkpoint worth playing against is the one the run store kept, so the answer comes from
    there.
    """
    try:
        best = RunReader(writer.directory).best_checkpoint()
    except RunError as e:
        # Cosmetic, and the run has already finished and written everything it produced.
        # Failing here would report a successful run as crashed.
        LOGGER.warning("could not read back which checkpoint was best: %s", e)
        return ""
    # Whether the metric is a number, not merely whether it is there: a diverged validation
    # records null, and None goes straight through a format spec into a TypeError — raised
    # after the run store has already recorded FINISHED.
    value = best.metrics.get(config.checkpoints.metric) if best else None
    if value is None:
        return ""
    return f", best {config.checkpoints.metric} {value:.4f} at step {best.step:,}"


def _stream(run: _Run, *, first: int = 0) -> Iterator[Batch]:
    """Every batch of the run from ``first`` on, in one pass over one loader.

    A batch is numbered over the whole run, and which positions it holds is a function of that
    number and the seed, so a resumed run starts at the batch it would have trained on next and
    sees exactly what it would have seen. See :mod:`chess_ai.training.batches`.
    """
    batches = run.batches if not first else Subset(run.batches, range(first, len(run.batches)))
    loader = batch_loader(batches, workers=run.config.training.data_workers, device=run.device)
    try:
        yield from iterate(loader)
    finally:
        # Lets the worker processes go rather than leaving them to a finalizer, which matters
        # when the run is ending because something went wrong.
        del loader


class _Window:
    """What the steps since the last log line measured, summed on the device.

    Kept as tensors and added to without being read, because reading one back from a GPU waits
    for every kernel queued behind it. One read per log line costs nothing; one per step costs a
    measurable share of the run.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._reset()

    def _reset(self) -> None:
        self._sums = torch.zeros(len(_StepResult._fields), dtype=torch.float64, device=self._device)
        self._steps = 0
        self._positions = 0
        self._since = time.perf_counter()
        self._paused = 0.0

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Keep what happens inside out of the throughput figure.

        Validating and writing a checkpoint are not training, and counting their seconds as
        training seconds would make the interval after each validation report a throughput the
        hardware never had — and make a slower ``every_steps`` look like a faster GPU.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._paused += time.perf_counter() - started

    def add(self, measured: _StepResult, *, positions: int) -> None:
        self._sums += torch.stack([value.double() for value in measured])
        self._steps += 1
        self._positions += positions

    def take(self) -> dict[str, float]:
        """The means since the last call, and how fast training produced them."""
        elapsed = time.perf_counter() - self._since - self._paused
        loss, policy_loss, value_loss, gradient_norm, clipped = (
            self._sums / max(1, self._steps)
        ).tolist()
        means = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "gradient_norm": gradient_norm,
            "clipped_fraction": clipped,
            "positions_per_second": self._positions / elapsed if elapsed > 0 else 0.0,
        }
        self._reset()
        return means


class _Beat:
    """The heartbeat, written no more often than :data:`HEARTBEAT_SECONDS`."""

    def __init__(self, writer: RunWriter, *, steps: int) -> None:
        self._writer = writer
        self._steps = steps
        self._last = 0.0

    def write(
        self,
        status: RunStatus,
        *,
        step: int,
        epoch: float,
        device: torch.device,
        positions_per_second: float | None = None,
        eta_seconds: float | None = None,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force and now - self._last < HEARTBEAT_SECONDS:
            return
        self._last = now
        self._writer.heartbeat(
            status,
            step=step,
            steps=self._steps,
            epoch=round(epoch, 4),
            positions_per_second=positions_per_second,
            eta_seconds=eta_seconds,
            gpu=gpu_stats(device),
        )


def _validation_split(run: _Run, *, say: Callable[[str], None]):
    """The validation split to measure on, or ``None`` with a word about why there is none."""
    counts = run.dataset.manifest.splits.get(VALIDATION)
    if counts is None or not counts.positions:
        say(
            "chess-ai: warning: this dataset has no validation positions, so no validation "
            "metrics will be logged; rebuild it with a larger --validation-fraction"
        )
        return None
    return run.dataset[VALIDATION]


def _validate(run: _Run, split) -> dict[str, float]:
    return validate(
        run.model,
        split,
        run.encoder,
        positions=run.config.validation.positions,
        batch_size=run.config.training.batch_size,
        device=run.device,
        value_loss_weight=run.config.training.value_loss_weight,
        autocast_dtype=run.dtype,
    )


def _writer(
    run: _Run,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: dict[str, float],
    elapsed: float,
) -> Callable[[Path], None]:
    """A function that writes this step's checkpoint wherever the run store puts it."""
    payload = checkpoint.build(
        step=step,
        run=run.config.name,
        seed=run.config.seed,
        architecture=run.config.model.architecture,
        model_options=run.config.model.options,
        spec=run.spec,
        config=run.config.resolved(),
        model_state=run.model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        metrics=metrics,
        rng_state=_rng_state(run.device),
        elapsed=round(elapsed, 3),
    )
    return lambda path: checkpoint.save(payload, path)


def _rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    """The state of every random generator training draws from, which is dropout's.

    The order positions are visited in is not among them: it is a function of the seed and the
    epoch, and is worked out again rather than restored. The loader has a generator of its own.
    """
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict[str, torch.Tensor], device: torch.device) -> None:
    if "cpu" in state:
        torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _info(run: _Run, *, estimate: float | None, lineage: Lineage | None) -> RunInfo:
    """What ``run.json`` records: the config, and everything it resolved against."""
    manifest = run.dataset.manifest
    return RunInfo(
        name=run.config.name,
        created=datetime.now(UTC),
        seed=run.config.seed,
        code_version=code_version(),
        device=describe_device(run.device),
        dataset=DatasetReference(
            name=manifest.name,
            directory=str(run.dataset.directory),
            format_version=manifest.format_version,
            created=manifest.created,
            games=manifest.games,
            positions=manifest.positions,
            train_positions=_split_positions(manifest, TRAIN),
            validation_positions=_split_positions(manifest, VALIDATION),
        ),
        encoder=run.spec,
        model=ModelReference(
            architecture=run.config.model.architecture,
            options=run.config.model.options,
            parameter_count=run.model.parameter_count,
        ),
        config=run.config.resolved(),
        steps=run.config.schedule.steps,
        batch_size=run.config.training.batch_size,
        positions_per_second_estimate=estimate,
        initialized_from=lineage,
    )


def _summary(
    run: _Run,
    estimate: float | None,
    *,
    lineage: Lineage | None = None,
    start: _Start | None = None,
) -> list[str]:
    """What a run says about itself before it starts, so it can be stopped before it is too late."""
    config = run.config
    manifest = run.dataset.manifest
    positions = run.batches.positions
    steps = config.schedule.steps
    first = start.step if start else 0
    epochs = steps * config.training.batch_size / positions
    precision = "bf16" if run.dtype is torch.bfloat16 else "fp32"
    lines = [
        f"chess-ai: run {config.name}, seed {config.seed}"
        + (f", resuming at step {first:,}" if start else ""),
        f"  device     {describe_device(run.device)}, {precision}",
        f"  dataset    {manifest.name}: {manifest.games:,} games, {positions:,} train positions, "
        f"{_split_positions(manifest, VALIDATION):,} validation",
        f"  encoder    {run.spec.describe()}",
        f"  model      {config.model.architecture}, "
        f"{run.model.parameter_count:,} parameters{_options(config.model.options)}",
        f"  schedule   {steps:,} steps of {config.training.batch_size} "
        f"({epochs:.1f} epochs), lr {config.optimizer.learning_rate:g} "
        f"warmup {config.schedule.warmup_steps:,} then cosine",
        f"  optimizer  {_describe_optimizer(run)}",
    ]
    if lineage is not None:
        trained_on = f", trained on {lineage.dataset}" if lineage.dataset else ""
        lines.append(f"  weights    from run {lineage.run} at step {lineage.step:,}{trained_on}")
    if start is not None and start.rng_state is None:
        lines.append(
            "  note       the checkpoint predates saving the random state, so dropout will not "
            "draw what it would have"
        )
    if estimate:
        seconds = (steps - first) * config.training.batch_size / estimate
        lines.append(
            f"  throughput {estimate:,.0f} positions/s measured, "
            f"about {_duration(seconds)} for {'the rest of ' if start else ''}the run"
        )
    # Only when this run will validate at all: whether it does is decided from the manifest,
    # not from the schedules, and a note about extra passes two lines above the warning that
    # none will happen invents a cost rather than revealing one.
    extra = _extra_validations(config) if _split_positions(manifest, VALIDATION) else 0
    if extra:
        lines.append(
            f"  note       {extra:,} extra validation pass{'es' if extra != 1 else ''}: "
            f"checkpoints every {config.checkpoints.every_steps:,} do not all land on"
        )
        lines.append(
            f"             validation every {config.validation.every_steps:,}, and each "
            "checkpoint validates to be judged on its own"
        )
    return lines


def _describe_optimizer(run: _Run) -> str:
    """AdamW's settings, and how many parameters its weight decay reaches."""
    settings = run.config.optimizer
    clip = f"clip {settings.gradient_clip:g}" if settings.gradient_clip else "no clipping"
    if not settings.weight_decay:
        return f"AdamW, no weight decay, {clip}"
    decayed, *undecayed = (
        sum(p.numel() for p in group["params"]) for group in _parameter_groups(run.model, settings)
    )
    decay = f"weight decay {settings.weight_decay:g} on {decayed:,} parameters"
    if undecayed:
        decay += f", none on {undecayed[0]:,} biases and norms"
    return f"AdamW, {decay}, {clip}"


def _extra_validations(config: ExperimentConfig) -> int:
    """How many validations the checkpoint schedule adds beyond what ``[validation]`` asks for.

    A checkpoint step validates whether or not it is a validation step, so that the metrics
    indexed with a checkpoint were measured on that checkpoint. When the two schedules do not
    divide each other that costs passes nobody asked for, and a run that is slower for an
    invisible reason is worse than one that says why — so the summary says how many, and says
    nothing when the answer is none.

    Counts the schedules only. Whether the run validates at all is a property of the dataset,
    so the caller checks that first.
    """
    steps = config.schedule.steps
    checkpoints = config.checkpoints.every_steps
    validations = config.validation.every_steps

    # Arithmetic rather than enumeration: both schedules are arithmetic progressions, so the
    # answer is a count, and building the two sets of step numbers would allocate in proportion
    # to the length of the run — before the dataset has even been opened — to print one line.
    forced = steps // checkpoints - steps // lcm(checkpoints, validations)
    # The last step is left out: it validates and checkpoints whatever the schedules say.
    return forced - (steps % checkpoints == 0 and steps % validations != 0)


def _epoch(run: _Run, step: int) -> float:
    """How many times over the training split the run has been, as a fraction."""
    return step * run.config.training.batch_size / run.batches.positions


def _eta(step: int, steps: int, elapsed: float) -> float | None:
    """How long the rest of the run will take, from how long the run so far took.

    From the whole run rather than from the last interval's throughput, because everything a run
    spends time on counts towards when it finishes: the slow first step, every validation, and
    every checkpoint written to disk. A figure derived from training throughput alone would
    promise an end time the run then misses by every validation it does.
    """
    if step <= 0 or elapsed <= 0:
        return None
    return (steps - step) * elapsed / step


def _validation_line(
    step: int,
    steps: int,
    metrics: dict[str, float],
    *,
    first: int,
    running: float,
    settled: int,
) -> str:
    """One validation's results, and how long this session has been running and has left.

    ``first`` is the step this session started from and ``running`` how long it has been going.
    The estimate is ``_eta``'s, from this session alone: nothing from before a resume says how
    fast the run goes now. It waits until the session is ``settled`` steps in, a whole gap
    between validations, so that the validation just done and the session's start-up are
    spread over at least as many steps as a validation is in a steady run. Before that, a
    resume that stopped a step short of a validation would divide both by a single step.
    """
    if step == steps:
        left = _clock(0)
    elif step - first < settled:
        left = "?"
    else:
        left = _clock((steps - step) * running / (step - first))
    return (
        f"  step {step:>{len(f'{steps:,}')},}/{steps:,}  "
        f"policy {metrics['policy_loss']:.4f}  value {metrics['value_loss']:.4f}  "
        f"top1 {metrics['top1']:.1%}  top5 {metrics['top5']:.1%}  "
        f"illegal {metrics['illegal_top_move_rate']:.1%}  "
        f"running {_clock(running)}  left {left}"
    )


def _options(options: dict[str, Any]) -> str:
    return (
        f" ({', '.join(f'{key}={value}' for key, value in sorted(options.items()))})"
        if options
        else ""
    )


def _split_positions(manifest, split: str) -> int:
    counts = manifest.splits.get(split)
    return counts.positions if counts else 0


def _duration(seconds: float) -> str:
    """A rough human duration, which is all anyone reads an estimate to two digits for."""
    if seconds < 1:
        return "<1s"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f} days"


def _clock(seconds: float) -> str:
    """A duration to the second, as ``H:MM:SS``, with whole days in front from 24 hours on."""
    days, rest = divmod(round(seconds), 24 * 3600)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02}:{minutes:02}:{seconds:02}"
    return f"{hours}:{minutes:02}:{seconds:02}"


def _synchronize(device: torch.device) -> None:
    """Wait for the device to actually finish, which timing anything on a GPU depends on."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _with_available(error: Exception, data_dir: Path) -> str:
    """``error``, plus which datasets there are, which is usually the next question."""
    from chess_ai.dataset import list_datasets

    names = list_datasets(data_dir)
    if not names:
        return f"{error} (no datasets in {data_dir}; build one with 'chess-ai dataset build')"
    return f"{error} (datasets in {data_dir}: {', '.join(names)})"
