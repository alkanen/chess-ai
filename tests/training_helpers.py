"""Tiny datasets, position records and experiment configs, for the training tests.

A run that takes a second is a run a test can afford: two hidden units, a handful of steps, and
the fixture PGN files the dataset tests already use. What the tests check is that the pieces are
wired together and write what they promise, which a small model shows as well as a large one.
"""

import textwrap
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import chess
import numpy as np
import torch
from dataset_helpers import build

from chess_ai.dataset import POSITION_DTYPE, RatingSource, Result, TimeControl
from chess_ai.dataset.records import position_record
from chess_ai.encoders import create_encoder
from chess_ai.models import create_model
from chess_ai.move_codec import move_index
from chess_ai.training import checkpoint
from chess_ai.training.run_store import (
    CheckpointPolicy,
    DatasetReference,
    ModelReference,
    RunInfo,
    RunStatus,
    RunWriter,
)

VALIDATION_FRACTION = 0.3
"""Enough of the fixture games held back that the validation split is worth measuring on."""


def dataset(data_dir: Path, name: str = "test", **options):
    """Build a dataset from every fixture PGN, with a validation split big enough to use."""
    options.setdefault("validation_fraction", VALIDATION_FRACTION)
    return build(data_dir, name=name, **options)


def record(
    board: chess.Board,
    *,
    move: str | None = None,
    result: Result = Result.DRAW,
    mover_rating: int = 1500,
    opponent_rating: int = 1600,
    mover_rating_known: bool = True,
    opponent_rating_known: bool = True,
    rating_source: RatingSource = RatingSource.LICHESS,
    time_control: TimeControl = TimeControl.BLITZ,
    ply: int = 0,
) -> np.ndarray:
    """One position record for ``board``, as a batch of one, the way a dataset stores it.

    ``move`` is a UCI move that has to be legal in the position, defaulting to whichever move
    python-chess generates first: which move was played does not matter to an encoder, and a
    legal one keeps the record the same shape as a real one.
    """
    played = chess.Move.from_uci(move) if move else next(iter(board.legal_moves))
    assert board.is_legal(played), f"{played} is not legal in {board.fen()}"
    return np.array(
        [
            position_record(
                board,
                move=move_index(played),
                result=result,
                mover_rating=mover_rating,
                opponent_rating=opponent_rating,
                mover_rating_known=mover_rating_known,
                opponent_rating_known=opponent_rating_known,
                rating_source=rating_source,
                time_control=time_control,
                ply=ply,
            )
        ],
        dtype=POSITION_DTYPE,
    )


def records(*boards: chess.Board) -> np.ndarray:
    """One record per board, as one batch."""
    return np.concatenate([record(board) for board in boards])


def experiment(path: Path, dataset_name: str = "test", **settings) -> Path:
    """Write an experiment config small enough to train in a test, and return its path.

    ``settings`` are TOML fragments keyed by section, appended to what is written here, so a
    test can say what it is about — ``checkpoints="every_steps = 2\\nkeep = 1"`` — without
    repeating the rest.
    """
    sections = {
        "": f'name = "{path.stem}"\nseed = 7',
        "dataset": f'name = "{dataset_name}"',
        "model": 'architecture = "mlp"\ndepth = 1\nwidth = 8',
        "training": 'device = "cpu"\nbatch_size = 8\ndata_workers = 0\nlog_every_steps = 2',
        "schedule": "steps = 6\nwarmup_steps = 2",
        "validation": "every_steps = 3\npositions = 16",
        "checkpoints": "every_steps = 3\nkeep = 2",
    }
    for section, extra in settings.items():
        sections[section] = textwrap.dedent(extra).strip()
    text = "\n\n".join(
        (body if not section else f"[{section}]\n{body}") for section, body in sections.items()
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def model_run(
    runs_dir: Path,
    name: str = "tiny",
    *,
    steps: Sequence[int] = (2, 4),
    depth: int = 1,
    width: int = 4,
    seed: int = 7,
) -> Path:
    """A run directory with real checkpoints in it, written without training anything.

    What the inference and web tests need is a file the engine can load, not a model that has
    learned something: untrained weights make a legal move as surely as trained ones do. Each
    step gets weights of its own, so a test can tell two checkpoints apart by the moves they
    play, and the validation metrics fall as the steps rise — which makes the *first* step the
    best one, so that "best" and "latest" are never the same checkpoint.
    """
    encoder = create_encoder("board-planes")
    options = {"depth": depth, "width": width}
    directory = runs_dir / name
    info = RunInfo(
        name=name,
        created=datetime.now(UTC),
        seed=seed,
        code_version="test",
        device="cpu",
        dataset=DatasetReference(
            name="test",
            directory=str(runs_dir / "data" / "test"),
            format_version=1,
            games=2,
            positions=40,
            train_positions=30,
            validation_positions=10,
        ),
        encoder=encoder.spec,
        model=ModelReference(
            architecture="mlp",
            options=options,
            parameter_count=create_model("mlp", encoder.spec, **options).parameter_count,
        ),
        config={},
        steps=max(steps),
        batch_size=8,
    )
    with RunWriter.create(
        directory, info, config_text="", policy=CheckpointPolicy(keep=len(steps))
    ) as writer:
        for index, step in enumerate(steps):
            # A seed per step, so that two checkpoints of one run are two different models.
            torch.manual_seed(seed + step)
            model = create_model("mlp", encoder.spec, **options)
            metrics = {"policy_loss": 1.0 + index}
            payload = checkpoint.build(
                step=step,
                run=name,
                seed=seed,
                architecture="mlp",
                model_options=options,
                spec=encoder.spec,
                config={},
                model_state=model.state_dict(),
                optimizer_state={},
                metrics=metrics,
            )
            writer.save_checkpoint(
                step, lambda path, saved=payload: checkpoint.save(saved, path), metrics=metrics
            )
        writer.finish(RunStatus.FINISHED, step=max(steps), steps=max(steps))
    return directory
