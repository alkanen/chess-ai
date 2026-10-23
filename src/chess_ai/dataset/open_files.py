"""How many files a process may hold open, which reading a large dataset at random runs into.

Every shard a reader has mapped holds a file descriptor for as long as it stays mapped: Python's
``mmap`` keeps a duplicate of the file's. A training batch of a couple of thousand random
positions touches most of a large split's shards, so a process reading one soon holds one
descriptor a shard: about 2,000 for lichess-2026 version 3. A login shell, ``screen`` and a
systemd service start with a soft limit of 1024, and the reader dies with "Too many open files"
at its first batch.

The soft limit is only a default, which any process may raise as far as its hard limit, and the
hard limit is usually far higher (a million on a typical Linux machine). So a command about to
read a dataset at random raises its own soft limit before it starts any worker, which inherits
it whether forked or spawned. The limit counts each process's own descriptors, so a worker
needs room for the shards it maps, however many workers there are.
"""

import logging
from typing import Final

try:
    import resource
except ImportError:  # Windows, which has no such limit to raise.
    resource = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

MARGIN: Final = 256
"""Descriptors a reading process needs besides its shards: the interpreter, torch, the run's
files, a loader's pipes and the epoch's order."""

SUGGESTED_LIMIT: Final = 65536
"""What the hint for running out suggests raising the limit to: far more than any dataset here."""


def allow_open_files(shards: int) -> None:
    """Let this process hold as many files open as it may, before reading ``shards`` shards.

    Raises the soft limit on open files to the hard limit, and warns when even that is less
    than ``shards`` and a margin: reading would then die part way into the first batches, and
    this says so before minutes of start-up rather than after.
    """
    if resource is None:
        return
    needed = shards + MARGIN
    unlimited = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # macOS answers an unlimited hard limit and then refuses an unlimited soft one, so there
    # it is raised only as far as needed.
    wanted = hard if hard != unlimited else max(soft, needed)
    if soft != unlimited and soft < wanted:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (ValueError, OSError) as e:
            LOGGER.warning(
                "cannot raise the limit on open files from %d to %d: %s", soft, wanted, e
            )
        else:
            LOGGER.debug("raised the limit on open files from %d to %d", soft, wanted)
            soft = wanted
    if soft != unlimited and soft < needed:
        LOGGER.warning(
            "reading this dataset needs about %s open files in each process (%s shards and %s "
            'for the rest), and the limit is %s; it will stop with "Too many open files" '
            "unless the hard limit is raised: `ulimit -Hn %d` as root, LimitNOFILE= in a "
            "systemd unit, or nofile in /etc/security/limits.conf",
            f"{needed:,}",
            f"{shards:,}",
            MARGIN,
            f"{soft:,}",
            max(needed, SUGGESTED_LIMIT),
        )


def raise_the_limit_hint() -> str:
    """What to say after "Too many open files": what the limit is, and how to raise it."""
    if resource is None:
        return ""
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = "unlimited" if soft == resource.RLIM_INFINITY else f"{soft:,}"
    return (
        f" (this process may open {limit}, and every shard it reads holds one open; "
        f"raise the limit before starting, for example with `ulimit -n {SUGGESTED_LIMIT}`)"
    )
