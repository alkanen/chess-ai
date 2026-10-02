"""Engines to play against in tests: the real Stockfish when it is installed, and a stand-in."""

import shutil
import sys
import time
from pathlib import Path

import pytest

FAKE_UCI = Path(__file__).parent / "fixtures" / "fake_uci.py"

STOCKFISH = shutil.which("stockfish")
"""The real engine, found as the default ``[stockfish] path`` finds it, or ``None``."""

needs_stockfish = pytest.mark.skipif(STOCKFISH is None, reason="Stockfish is not installed")


def fake_engine(directory: Path, *options: str) -> Path:
    """An executable that runs the stand-in engine with ``options``, and logs what it is told.

    A file of its own rather than a command line, because the configured path is one: this is
    what a test sets ``[stockfish] path`` to. Each process it starts adds its id to ``pids``
    next to it, which is how a test finds the processes it has to make sure are gone.
    """
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake-stockfish"
    script.write_text(
        "#!/bin/sh\n"
        f'echo $$ >> "{directory / "pids"}"\n'
        f'exec "{sys.executable}" "{FAKE_UCI}" --log "{directory / "log"}" {" ".join(options)}\n'
    )
    script.chmod(0o755)
    return script


def started(engine: Path) -> list[int]:
    """The processes ``engine`` (from :func:`fake_engine`) has been started as, oldest first."""
    pids = engine.parent / "pids"
    return [int(line) for line in pids.read_text().split()] if pids.exists() else []


def told(engine: Path) -> list[str]:
    """Every line the processes of ``engine`` have been sent."""
    log = engine.parent / "log"
    return log.read_text().splitlines() if log.exists() else []


def running(pid: int) -> bool:
    """Whether ``pid`` is a live process; one that has exited and not yet been reaped is not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        # Not there, or there when it was opened and gone by the time it was read.
        return False
    # The state follows the command name, which is in parentheses and may contain spaces.
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def wait_until_gone(*pids: int, timeout: float = 5.0) -> None:
    """Wait for every one of ``pids`` to stop, and fail if any of them does not."""
    deadline = time.monotonic() + timeout
    while any(running(pid) for pid in pids):
        assert time.monotonic() < deadline, f"still running: {[p for p in pids if running(p)]}"
        time.sleep(0.02)
