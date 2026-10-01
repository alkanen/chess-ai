"""The training loop: one experiment config in, one run directory out.

Supervised next-move prediction, exactly as the PRD frames it: show the network a position, have
it predict the move the human played, repeat. The loss is cross-entropy over the shared move
vocabulary, plus a weighted cross-entropy on the win/draw/loss head, whose target is how the game
the position came from actually ended, from the mover's point of view.

Everything the run produces goes through the run store, and nothing is held in memory that a
crash would lose: metrics are appended as they are measured, the heartbeat is replaced as the run
moves, and checkpoints are written as they are made. A run is therefore worth exactly as much
after it crashes as it was a moment before.

The loop keeps two things honest about speed. It never reads a loss back from the GPU except when
it is about to log one, because every read is a stall; and before it starts it measures a few real
steps and says how fast they were, so a two-day run can be recognised as a two-day run before it
is two days in.
"""

import logging
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from math import lcm
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.nn import functional as F

from chess_ai.dataset import TRAIN, VALIDATION, Dataset, DatasetError, ManifestError, open_dataset
from chess_ai.encoders import Encoder, EncoderSpec, create_encoder
from chess_ai.models import ChessModel, create_model
from chess_ai.move_codec import VOCABULARY_SIZE
from chess_ai.registry import RegistryError
from chess_ai.training import checkpoint
from chess_ai.training.batches import Batch, PositionBatches, batch_loader
from chess_ai.training.experiment import ExperimentConfig
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
    DatasetReference,
    ModelReference,
    RunError,
    RunInfo,
    RunReader,
    RunStatus,
    RunWriter,
    check_available,
    code_version,
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
    estimate = _probe(run)
    for line in _summary(run, estimate):
        say(line)

    try:
        writer = RunWriter.create(
            directory,
            _info(run, estimate=estimate),
            config_text=config_text,
            policy=config.checkpoints.policy(),
            overwrite=overwrite,
        )
    except RunError as e:
        raise TrainingError(e) from e
    say(f"chess-ai: run {config.name} in {directory}")

    progress = _Progress()
    with writer:
        try:
            _loop(run, writer, progress, say=say)
        except KeyboardInterrupt:
            # A checkpoint on the way out, and a resume that picks it up, is its own slice of
            # work; all this can promise is that the run does not go on claiming to be running.
            _ended(writer, RunStatus.STOPPED, run, progress, "interrupted")
            raise
        except RunError as e:
            # A disk filling up an hour into a run is an ordinary accident: the run says it
            # crashed, and the person running it gets a sentence rather than a traceback.
            _ended(writer, RunStatus.CRASHED, run, progress, f"{type(e).__name__}: {e}")
            raise TrainingError(e) from e
        except Exception as e:
            _ended(writer, RunStatus.CRASHED, run, progress, f"{type(e).__name__}: {e}")
            raise
    return directory


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
        run.model.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
    )


def _schedule(run: _Run, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to a fraction of the learning rate."""
    schedule = run.config.schedule
    warmup = schedule.warmup_steps
    floor = schedule.min_learning_rate_fraction
    decaying = max(1, schedule.steps - warmup)

    def factor(completed: int) -> float:
        if completed < warmup:
            return (completed + 1) / warmup
        progress = min(1.0, (completed - warmup) / decaying)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _losses(run: _Run, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The combined loss and its two halves, for one batch."""
    with torch.autocast(run.device.type, dtype=run.dtype, enabled=run.dtype is not None):
        out = run.model(batch.spatial, batch.globals)
        policy_loss = F.cross_entropy(out.policy, batch.move)
        value_loss = F.cross_entropy(out.value, batch.result)
    total = policy_loss + run.config.training.value_loss_weight * value_loss
    return total, policy_loss.detach(), value_loss.detach()


def _step(run: _Run, batch: Batch, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, ...]:
    """One optimizer step. Returns the losses as tensors, unread, so as not to stall on them."""
    total, policy_loss, value_loss = _losses(run, batch)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    clip = run.config.optimizer.gradient_clip
    if clip:
        nn.utils.clip_grad_norm_(run.model.parameters(), clip)
    optimizer.step()
    return total.detach(), policy_loss, value_loss


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


def _loop(run: _Run, writer: RunWriter, progress: _Progress, *, say: Callable[[str], None]) -> None:
    """Train to the end of the schedule, logging, validating and checkpointing as it goes."""
    config = run.config
    steps = config.schedule.steps
    optimizer = _optimizer(run)
    scheduler = _schedule(run, optimizer)
    validation = _validation_split(run, say=say)

    run.model.train()
    started = time.perf_counter()
    window = _Window(run.device)
    latest: dict[str, float] = {}
    measured_at: int | None = None
    """The step ``latest`` was measured at, so a checkpoint is never indexed with another's."""
    throughput: float | None = None
    step = 0
    positions_seen = 0
    """How many training positions the weights have been moved by, which is what a chart of
    runs with different batch sizes compares them on. Not ``positions``: a validation line
    already says how many positions it measured under that name."""
    beat = _Beat(writer, steps=steps)
    beat.write(RunStatus.RUNNING, step=0, epoch=0.0, device=run.device, force=True)

    for batch in _stream(run):
        batch = batch.to(run.device, non_blocking=run.device.type == "cuda")
        # Read before the scheduler moves on: this is the rate the weights just moved by, and
        # get_last_lr() after scheduler.step() is already the next step's.
        learning_rate = scheduler.get_last_lr()[0]
        window.add(*_step(run, batch, optimizer), positions=len(batch))
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
            say(_validation_line(step, steps, latest))
            measured_at = step

        if saving:
            metrics = latest if measured_at == step else {}
            with window.paused():
                path = writer.save_checkpoint(
                    step, _writer(run, optimizer, step=step, metrics=metrics), metrics=metrics
                )
            LOGGER.info("saved %s", path)

    elapsed = time.perf_counter() - started
    writer.finish(
        RunStatus.FINISHED,
        step=step,
        steps=steps,
        epoch=round(_epoch(run, step), 4),
        positions_per_second=step * config.training.batch_size / elapsed if elapsed else None,
    )
    say(f"chess-ai: finished {steps:,} steps in {_duration(elapsed)}{_best(writer, config)}")


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


def _stream(run: _Run) -> Iterator[Batch]:
    """Every batch of the run, in one pass over one loader; see :mod:`chess_ai.training.batches`."""
    loader = batch_loader(run.batches, workers=run.config.training.data_workers, device=run.device)
    try:
        yield from loader
    finally:
        # Lets the worker processes go rather than leaving them to a finalizer, which matters
        # when the run is ending because something went wrong.
        del loader


class _Window:
    """The losses and positions since the last log line, summed on the device.

    Kept as tensors and added to without being read, because reading one back from a GPU waits
    for every kernel queued behind it. One read per log line costs nothing; one per step costs a
    measurable share of the run.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._reset()

    def _reset(self) -> None:
        self._sums = torch.zeros(3, dtype=torch.float64, device=self._device)
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

    def add(
        self,
        total: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        *,
        positions: int,
    ) -> None:
        self._sums += torch.stack((total, policy_loss, value_loss)).double()
        self._steps += 1
        self._positions += positions

    def take(self) -> dict[str, float]:
        """The means since the last call, and how fast training produced them."""
        elapsed = time.perf_counter() - self._since - self._paused
        loss, policy_loss, value_loss = (self._sums / max(1, self._steps)).tolist()
        means = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
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
    run: _Run, optimizer: torch.optim.Optimizer, *, step: int, metrics: dict[str, float]
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
    )
    return lambda path: checkpoint.save(payload, path)


def _info(run: _Run, *, estimate: float | None) -> RunInfo:
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
    )


def _summary(run: _Run, estimate: float | None) -> list[str]:
    """What a run says about itself before it starts, so it can be stopped before it is too late."""
    config = run.config
    manifest = run.dataset.manifest
    positions = run.batches.positions
    steps = config.schedule.steps
    epochs = steps * config.training.batch_size / positions
    precision = "bf16" if run.dtype is torch.bfloat16 else "fp32"
    lines = [
        f"chess-ai: run {config.name}, seed {config.seed}",
        f"  device     {describe_device(run.device)}, {precision}",
        f"  dataset    {manifest.name}: {manifest.games:,} games, {positions:,} train positions, "
        f"{_split_positions(manifest, VALIDATION):,} validation",
        f"  encoder    {run.spec.describe()}",
        f"  model      {config.model.architecture}, "
        f"{run.model.parameter_count:,} parameters{_options(config.model.options)}",
        f"  schedule   {steps:,} steps of {config.training.batch_size} "
        f"({epochs:.1f} epochs), lr {config.optimizer.learning_rate:g} "
        f"warmup {config.schedule.warmup_steps:,} then cosine",
    ]
    if estimate:
        seconds = steps * config.training.batch_size / estimate
        lines.append(
            f"  throughput {estimate:,.0f} positions/s measured, "
            f"about {_duration(seconds)} for the run"
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


def _validation_line(step: int, steps: int, metrics: dict[str, float]) -> str:
    return (
        f"  step {step:>{len(f'{steps:,}')},}/{steps:,}  "
        f"policy {metrics['policy_loss']:.4f}  value {metrics['value_loss']:.4f}  "
        f"top1 {metrics['top1']:.1%}  top5 {metrics['top5']:.1%}  "
        f"illegal {metrics['illegal_top_move_rate']:.1%}"
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
