"""Getting what a build wrote onto the disk, on filesystems that may not agree to say.

A dataset's manifest is what marks it finished, so the records it counts have to be durable
before it is published: a manifest that survives a power cut while its shards do not is a
dataset that reads as corrupt. That takes an ``fsync`` on each file *and* one on the directory
holding it, because flushing a file says nothing about whether its name survives.

Neither call is available everywhere. A directory cannot be opened as a file on Windows at all,
and ``fsync`` on a directory answers ``EINVAL`` on several network and FUSE filesystems — which
is not exotic here, since the data directory is configurable and an 80 GB dump is exactly the
thing people stage on a NAS mount. So the two are told apart:

- a file's contents are the dataset, and a failure to flush them ends the build, because a full
  disk surfaces as ``ENOSPC`` at exactly that point
- a directory entry is only a name, and a filesystem that will not flush one is taken at its
  word rather than allowed to destroy a finished build
"""

import errno
import os
from pathlib import Path
from typing import IO, Final

UNSUPPORTED: Final = frozenset(
    code
    for code in (
        getattr(errno, name, None) for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS")
    )
    if code is not None
)
"""What a filesystem answers when it does not implement flushing, rather than when it failed."""


def sync_file(file: IO) -> None:
    """Flush ``file`` and everything behind it onto the disk.

    Raises whatever the disk says, because buffered data that never reached the device shows up
    here and nowhere else. The exception is a filesystem that does not implement the call at
    all, which is taken at its word: the write itself succeeded, and there is nothing else to
    ask of it.
    """
    file.flush()
    try:
        os.fsync(file.fileno())
    except OSError as e:
        if e.errno not in UNSUPPORTED:
            raise


def sync_directory(directory: Path) -> None:
    """Flush ``directory``'s own entries onto the disk, which syncing a file in it does not do.

    Best effort, and silent when it cannot be done at all; see the module docstring. What is at
    stake is only whether the names of files already written survive a power cut, which is not
    worth failing a finished build over.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
