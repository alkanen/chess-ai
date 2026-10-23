"""The limit on open files, which reading a dataset of many shards at random runs into.

Each test that changes the limit does so in a process of its own: a lowered limit cannot always
be raised again, and the test process would carry it into every test after.
"""

import logging
import resource
import subprocess
import sys
import textwrap

import pytest
from dataset_helpers import build
from training_helpers import VALIDATION_FRACTION, experiment

from chess_ai.dataset import Shards, open_dataset
from chess_ai.dataset.open_files import MARGIN, allow_open_files

ONE_PER_SHARD = Shards(
    positions_per_shard=1, games_per_shard=1, moves_per_shard=1, targets_per_shard=1
)
"""A shard for every record, so that the fixtures need more files open than a lowered limit."""

LOW = 48
"""A soft limit the fixtures in :data:`ONE_PER_SHARD` shards need more descriptors than."""


def limited(script: str, *, soft: int, hard: int | None = None, cwd) -> subprocess.CompletedProcess:
    """Run ``script`` in a Python process whose limit on open files is ``soft`` (and ``hard``)."""
    _, current = resource.getrlimit(resource.RLIMIT_NOFILE)
    prelude = (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_NOFILE, ({soft}, {hard or current}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(script)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_split_counts_the_shards_training_reads_from(tmp_path):
    build(tmp_path, shards=ONE_PER_SHARD, validation_fraction=VALIDATION_FRACTION)
    dataset = open_dataset("test", data_dir=tmp_path)

    train = dataset.manifest.splits["train"]
    assert dataset["train"].training_shards == train.positions, "one a position, no targets"


def test_running_out_of_files_on_a_shard_says_how_to_raise_the_limit(tmp_path):
    build(tmp_path / "data", shards=ONE_PER_SHARD)

    ran = limited(
        """
        from pathlib import Path
        import numpy as np
        from chess_ai.dataset import open_dataset
        split = open_dataset("test", data_dir=Path("data"))["train"]
        split.positions(np.arange(len(split)))
        """,
        soft=LOW,
        cwd=tmp_path,
    )

    assert ran.returncode != 0
    last = ran.stderr.strip().splitlines()[-1]
    assert last.startswith("chess_ai.dataset.store.DatasetError: cannot read dataset shard ")
    assert "Too many open files" in last
    assert f"this process may open {LOW}" in last
    assert "ulimit -n 65536" in last


def test_the_soft_limit_is_raised_to_the_hard_one(tmp_path):
    ran = limited(
        """
        import resource
        from chess_ai.dataset.open_files import allow_open_files
        allow_open_files(1000)
        print(*resource.getrlimit(resource.RLIMIT_NOFILE))
        """,
        soft=LOW,
        hard=4096,
        cwd=tmp_path,
    )

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.split() == ["4096", "4096"]
    assert ran.stderr == "", "it is enough, so nothing to say"


def test_a_hard_limit_too_low_for_the_shards_says_so_with_the_fix(tmp_path):
    ran = limited(
        """
        import logging, resource
        from chess_ai.dataset.open_files import allow_open_files
        logging.basicConfig(format="%(levelname)s %(message)s")
        allow_open_files(5000)
        print(*resource.getrlimit(resource.RLIMIT_NOFILE))
        """,
        soft=LOW,
        hard=1024,
        cwd=tmp_path,
    )

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.split() == ["1024", "1024"], "raised as far as it goes all the same"
    assert ran.stderr.startswith("WARNING reading this dataset needs about ")
    assert f"{5000 + MARGIN:,} open files" in ran.stderr
    assert "5,000 shards" in ran.stderr
    assert "the limit is 1,024" in ran.stderr
    assert "ulimit -Hn 65536" in ran.stderr


def test_a_limit_already_high_enough_is_left_alone(monkeypatch, caplog):
    monkeypatch.setattr(resource, "getrlimit", lambda which: (100_000, 100_000))
    monkeypatch.setattr(resource, "setrlimit", lambda *args: pytest.fail("set again"))

    with caplog.at_level(logging.DEBUG):
        allow_open_files(5000)

    assert caplog.records == []


def test_training_with_a_soft_limit_below_what_the_dataset_needs_trains(tmp_path):
    # The limit a login shell, screen or a systemd service starts with, scaled down to the
    # fixtures: lowered below what the shards need, with the hard limit left where it was.
    build(tmp_path / "data", shards=ONE_PER_SHARD, validation_fraction=VALIDATION_FRACTION)
    config = experiment(
        tmp_path / "tiny.toml",
        training='device = "cpu"\nbatch_size = 8\ndata_workers = 0\nlog_every_steps = 2',
        schedule="steps = 12\nwarmup_steps = 2",
    )

    ran = limited(
        f"""
        import sys
        from chess_ai.cli import main
        sys.exit(main(["train", {str(config)!r}, "-vv"]))
        """,
        soft=LOW,
        cwd=tmp_path,
    )

    assert ran.returncode == 0, ran.stderr[-3000:]
    assert f"raised the limit on open files from {LOW} to " in ran.stderr
    assert "chess-ai: finished 12 steps" in ran.stdout
