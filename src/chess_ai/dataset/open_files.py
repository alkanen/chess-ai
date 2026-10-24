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


HARD_LIMIT_ADVICE: Final = (
    "`ulimit -Hn {limit}` as root, LimitNOFILE= in a systemd unit, or nofile in "
    "/etc/security/limits.conf"
)
"""How to raise the hard limit, which only root can, with ``{limit}`` to fill in."""


def allow_open_files(shards: int) -> None:
    """Let this process hold open the ``shards`` shards it is about to read, and the rest.

    Raises the soft limit on open files as far as that needs, and to at least
    :data:`SUGGESTED_LIMIT` for room to spare, but no further: every process this one starts
    inherits the limit, and some software sizes a table by it or closes every descriptor up to
    it. Warns when the hard limit is less than that: reading would then die part way into the
    first batches, and this says so before minutes of start-up rather than after.
    """
    if resource is None:
        return
    needed = shards + MARGIN
    unlimited = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = max(needed, SUGGESTED_LIMIT)
    if hard != unlimited:
        wanted = min(wanted, hard)
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
            'for the rest), and the limit is %s; it is likely to stop with "Too many open '
            'files" unless the hard limit is raised: %s',
            f"{needed:,}",
            f"{shards:,}",
            MARGIN,
            f"{soft:,}",
            HARD_LIMIT_ADVICE.format(limit=max(needed, SUGGESTED_LIMIT)),
        )


def raise_the_limit_hint() -> str:
    """What to say after "Too many open files": what the limit is, and how to raise it.

    A soft limit below the hard one is raised with ``ulimit -n``. One already at the hard
    limit, as a training process's is after :func:`allow_open_files`, can only be raised by
    raising the hard limit, which ``ulimit -n`` would either be refused or not change.
    """
    if resource is None:
        return ""
    unlimited = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == unlimited:
        return ""
    if hard != unlimited and soft >= hard:
        advice = HARD_LIMIT_ADVICE.format(limit=max(SUGGESTED_LIMIT, 2 * hard))
        return (
            f" (this process may open {soft:,}, as many as its hard limit allows, and every "
            f"shard it reads holds one open; raise the hard limit before starting: {advice})"
        )
    raised = SUGGESTED_LIMIT if hard == unlimited else min(SUGGESTED_LIMIT, hard)
    return (
        f" (this process may open {soft:,}, and every shard it reads holds one open; "
        f"raise the limit before starting, for example with `ulimit -n {raised}`)"
    )
