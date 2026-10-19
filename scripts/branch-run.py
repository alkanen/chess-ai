#!/usr/bin/env python3
"""Copy a run as it was at one of its checkpoints, with some optimizer settings changed.

For trying a change part way through a long run without training the run again from the start:
each copy carries on from the same checkpoint, with the same data order and the same learning
rate schedule, and differs only in what was changed. ``chess-ai resume COPY --stop-at STEP``
then trains a stretch of it, and resuming it again carries it on to the end of its schedule.

The copy is a run of its own, named as runs are, with everything the source had up to the
step: the metrics log, the checkpoints up to it (the rest are left behind, since resuming
starts from the latest one), and the evaluations of those checkpoints, so that the copy's
charts and Stockfish ladder go back to the start. Its title says what it was copied from and
what changed, and its ``config.toml`` starts with a comment saying the same; the settings it
trains with are in its checkpoint, which is where ``chess-ai resume`` reads them from.

Only the settings that a resumed run takes from its config can be changed:
``optimizer.weight_decay``, ``optimizer.weight_decay_schedule`` and
``optimizer.gradient_clip``. The learning rate and Adam's betas come from the optimizer state
the checkpoint holds, and anything else would change what the run is.

::

    uv run scripts/branch-run.py RUN 524000 RUN-wd-sched \\
        --set optimizer.weight_decay_schedule=learning-rate
    uv run scripts/branch-run.py RUN 524000 RUN-wd0 --set optimizer.weight_decay=0
    uv run chess-ai resume RUN-wd-sched --stop-at 564000
"""

import argparse
import shutil
import sys
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from chess_ai.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_FILE, ConfigError, load_config
from chess_ai.run_rename import Renamed, _Rename, _replace_checkpoint
from chess_ai.training import checkpoint
from chess_ai.training.checkpoint import CheckpointError
from chess_ai.training.experiment import ExperimentConfig
from chess_ai.training.run_store import (
    CHECKPOINT_INDEX,
    CHECKPOINTS_DIR,
    CONFIG_FILE,
    EVALUATIONS_DIR,
    MAX_TITLE_LENGTH,
    METRICS_FILE,
    NOTES_FILE,
    PARTIAL_SUFFIX,
    RUN_FILE,
    STATUS_FILE,
    TEMPORARY_SUFFIX,
    CheckpointIndex,
    Heartbeat,
    RunError,
    RunNotes,
    RunStatus,
    _as_json,
    _keep_metrics_until,
    _replace_file,
    _WriterLock,
    checkpoint_file,
    open_run,
    run_path,
    valid_name,
)
from chess_ai.training.trainer import _as_trained

CHANGEABLE = frozenset(
    {"optimizer.weight_decay", "optimizer.weight_decay_schedule", "optimizer.gradient_clip"}
)
"""The settings a resumed run takes from its config, rather than from its optimizer state."""

LOCKS = ("*.lock", f"*{TEMPORARY_SUFFIX}", f"*{PARTIAL_SUFFIX}")
"""What is not copied from an evaluation: locks, and files still being written."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        target = branch(
            config.paths.runs, args.source, args.step, args.name, dict(args.set), title=args.title
        )
    except (ConfigError, RunError, CheckpointError, ValueError) as e:
        print(f"{parser.prog}: error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"{parser.prog}: error: {e}", file=sys.stderr)
        return 2
    print(f"copied run {args.source} at step {args.step:,} to {target}")
    print(f"train it with: uv run chess-ai resume {args.name} --stop-at STEP")
    return 0


def branch(
    runs_dir: Path,
    source_name: str,
    step: int,
    name: str,
    changes: dict[str, Any],
    *,
    title: str | None = None,
) -> Path:
    """Copy run ``source_name`` as it was at ``step`` to a run called ``name``, with ``changes``
    made to its config, and return the copy's directory."""
    for key in changes:
        if key not in CHANGEABLE:
            raise ValueError(f"cannot change {key}: only {', '.join(sorted(CHANGEABLE))} can be")
    source = open_run(runs_dir, source_name)
    target = run_path(runs_dir, valid_name(name))
    if target.exists():
        raise RunError(f"{target} is there already")
    # Before the checkpoints are listed, and held while they are copied, so that no trainer
    # saves or prunes any meanwhile: one letting go of the run as this began could otherwise
    # delete a checkpoint between the listing and the copy.
    lock = _WriterLock.acquire(source.directory, source_name)
    try:
        kept = [info for info in source.checkpoints() if info.step <= step]
        if not kept or kept[-1].step != step:
            steps = ", ".join(f"{info.step:,}" for info in source.checkpoints())
            raise RunError(
                f"run {source_name!r} has no checkpoint of step {step:,}; it has {steps}"
            )
        payload = checkpoint.load(source.checkpoint_path(kept[-1]))
        config = _changed(payload["config"], changes, name)
        target.mkdir()
    except BaseException:
        lock.release()
        raise
    try:
        _copy(source.directory, target, step, [info.file for info in kept])
        index = CheckpointIndex.model_validate_json(
            (source.directory / CHECKPOINTS_DIR / CHECKPOINT_INDEX).read_text()
        )
        _replace_file(
            target / CHECKPOINTS_DIR / CHECKPOINT_INDEX, _as_json(_kept_index(index, step))
        )
        _replace_checkpoint(
            target / CHECKPOINTS_DIR / checkpoint_file(step),
            {**payload, "run": name, "config": config},
        )
        renaming = _Rename({source_name}, name, dry_run=False, say=lambda *a, **k: None)
        renaming.evaluations(target, Renamed())
        described = (
            ", ".join(f"{key} = {value!r}" for key, value in changes.items()) or "no changes"
        )
        what = f"Copied from run {source_name} at step {step:,}, with {described}"
        _write_files(source, target, step, config, what, title)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        lock.release()
    return target


def _changed(config: dict[str, Any], changes: dict[str, Any], name: str) -> dict[str, Any]:
    """The checkpoint's ``config`` with ``changes`` made and the copy's name, checked.

    A setting the source's config predates is written as the source was trained without it,
    as resuming the source would read it: the copy names every setting, and today's default
    there would be another run's behaviour, or optimizer state that no longer fits.
    """
    edited = _as_trained(config)
    for key, value in changes.items():
        section, setting = key.split(".")
        edited[section][setting] = value
    edited["name"] = name
    try:
        return ExperimentConfig.model_validate(edited).resolved()
    except ValidationError as e:
        raise ValueError(f"the changed config is not one a run can have: {e}") from e


def _copy(source: Path, target: Path, step: int, checkpoints: list[str]) -> None:
    """The source's files up to ``step``, but for the ones that say what it is and how it is."""
    shutil.copy2(source / METRICS_FILE, target / METRICS_FILE)
    _keep_metrics_until(target / METRICS_FILE, step)
    (target / CHECKPOINTS_DIR).mkdir()
    for file in checkpoints:
        # With its time, which is what tells the evaluator the results copied with it stand.
        shutil.copy2(source / CHECKPOINTS_DIR / file, target / CHECKPOINTS_DIR / file)
    (target / EVALUATIONS_DIR).mkdir()
    for evaluation in sorted((source / EVALUATIONS_DIR).glob("step-*")):
        if evaluation.is_dir() and int(evaluation.name.removeprefix("step-")) <= step:
            shutil.copytree(
                evaluation,
                target / EVALUATIONS_DIR / evaluation.name,
                ignore=shutil.ignore_patterns(*LOCKS),
            )


def _kept_index(index: CheckpointIndex, step: int) -> CheckpointIndex:
    """``index`` with only the checkpoints up to ``step``, and the best of those."""
    kept = [info for info in index.checkpoints if info.step <= step]
    policy = index.policy
    measured = [info for info in kept if info.metrics.get(policy.metric) is not None]
    best = None
    if measured:
        pick = max if policy.higher_is_better else min
        best = pick(measured, key=lambda info: info.metrics[policy.metric]).step
    return index.model_copy(update={"checkpoints": kept, "best_step": best})


def _write_files(
    source, target: Path, step: int, config: dict[str, Any], what: str, title: str | None
) -> None:
    """The copy's own config, notes and status, and last its ``run.json``, which is what makes
    the directory a run."""
    text = (source.directory / CONFIG_FILE).read_text(encoding="utf-8")
    original = tomllib.loads(text).get("name")
    if original is not None:
        text = text.replace(f'name = "{original}"', f'name = "{config["name"]}"', 1)
    _replace_file(target / CONFIG_FILE, f"# {what} (scripts/branch-run.py).\n{text}")

    notes = source.notes
    shown = title or f"{notes.title or source.name} — {what[0].lower()}{what[1:]}"
    _replace_file(
        target / NOTES_FILE,
        _as_json(
            RunNotes(title=shown[:MAX_TITLE_LENGTH], tags=list(notes.tags), notes=notes.notes)
        ),
    )

    epochs = [
        line["epoch"] for line in source.metrics() if line["step"] <= step and "epoch" in line
    ]
    now = datetime.now(UTC)
    _replace_file(
        target / STATUS_FILE,
        _as_json(
            Heartbeat(
                status=RunStatus.STOPPED,
                pid=0,
                started=now,
                updated=now,
                step=step,
                steps=config["schedule"]["steps"],
                epoch=epochs[-1] if epochs else 0.0,
                message=what,
            )
        ),
    )
    info = source.info.model_copy(update={"name": config["name"], "config": config})
    _replace_file(target / RUN_FILE, _as_json(info))


def _setting(text: str) -> tuple[str, Any]:
    """``section.key=value``, with the value read as TOML reads it, or as a string."""
    key, equals, value = text.partition("=")
    if not equals or key.count(".") != 1:
        raise argparse.ArgumentTypeError(f"section.key=value, not {text!r}")
    try:
        return key.strip(), tomllib.loads(f"value = {value.strip()}")["value"]
    except tomllib.TOMLDecodeError:
        return key.strip(), value.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="branch-run", description=__doc__.splitlines()[0], allow_abbrev=False
    )
    parser.add_argument("source", help="the run to copy")
    parser.add_argument("step", type=int, help="the step of the checkpoint to copy it at")
    parser.add_argument("name", help="what to call the copy")
    parser.add_argument(
        "--set",
        type=_setting,
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help=f"a setting to change, one of {', '.join(sorted(CHANGEABLE))}; repeatable",
    )
    parser.add_argument(
        "--title", help="the copy's title (default: the source's, and what changed)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"configuration file (default: ${CONFIG_PATH_ENV}, else ./{DEFAULT_CONFIG_FILE} "
        "if it exists), which is what says where runs are kept",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
