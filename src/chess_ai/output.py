"""What the command-line tools print, and where.

Two channels, set up once for every command by :func:`configure`:

- :func:`say` is the command's own account of what it is doing — the stage it is starting, the
  game it is playing, the result it got — on stdout. Each line has a :class:`Kind`, which is
  what ``--quiet`` thins out.
- :mod:`logging` is for what went wrong or is worth a second look, on stderr: warnings and errors
  by default, ``--verbose`` adding info and then debug records. Messages are written without a
  ``chess-ai:`` prefix or a level; the one handler here adds both.

Every line starts with the time it was written, so that a detached run's log, or a ``screen``
window looked at hours later, says when things happened. Not where the line goes to the journal,
which stamps each line itself, and not for the commands that only answer a question
(``runs list``), whose output is read at once or handed to a script.
"""

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Final, Protocol, TextIO


class Kind(IntEnum):
    """How much a line matters, which is what decides whether ``--quiet`` hides it."""

    DETAIL = 0
    """One step of many: a stage starting, a game, a progress line, a checkpoint saved."""
    MILESTONE = 1
    """Where a long job has got to: a validation, a header naming what is being played."""
    RESULT = 2
    """What the command was run for: a summary, a finished run, a suite's result."""


DETAIL: Final = Kind.DETAIL
MILESTONE: Final = Kind.MILESTONE
RESULT: Final = Kind.RESULT


class Say(Protocol):
    """Somewhere to say a line, of the kind it is: :func:`say`, or a test's list of them."""

    def __call__(self, text: str, *, kind: Kind = MILESTONE) -> None: ...


class Timestamps(StrEnum):
    """``--timestamps``: whether lines start with the time they were written."""

    AUTO = "auto"
    """Unless the line goes to the journal, or the command only answers a question."""
    ALWAYS = "always"
    NEVER = "never"


QUIETEST: Final = -2
LOUDEST: Final = 2
"""How far ``--quiet`` and ``--verbose`` go; more of either changes nothing."""

_LOG_LEVELS: Final = {
    -2: logging.ERROR,
    -1: logging.WARNING,
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}
"""Which records reach stderr at each verbosity. Warnings stay down to a single ``--quiet``:
they are the lines that would be missed."""

_LEAST_SHOWN: Final = {-2: RESULT, -1: MILESTONE}
"""The least a line must matter to be said at a quiet verbosity; everything is said otherwise."""


@dataclass
class _Settings:
    verbosity: int = 0
    timestamps: Timestamps = Timestamps.AUTO
    query: bool = False
    """Whether the command only answers a question, which :attr:`Timestamps.AUTO` leaves
    unstamped."""


_settings = _Settings()
_handler: logging.Handler | None = None


def configure(
    *, verbosity: int = 0, timestamps: Timestamps = Timestamps.AUTO, query: bool = False
) -> None:
    """Set up both channels for a command: how much is said, and whether lines are stamped.

    ``verbosity`` is ``--verbose`` counted up and ``--quiet`` counted down. Called again, as a
    test or a program running several commands in one process does, it replaces what the call
    before set up rather than adding to it.
    """
    global _handler
    _settings.verbosity = max(QUIETEST, min(LOUDEST, verbosity))
    _settings.timestamps = timestamps
    _settings.query = query
    root = logging.getLogger()
    if _handler is None:
        _handler = _StderrHandler()
        root.addHandler(_handler)
    root.setLevel(_LOG_LEVELS[_settings.verbosity])


def reset() -> None:
    """Undo :func:`configure`, leaving logging as it was before any command set it up."""
    global _handler
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler = None
    logging.getLogger().setLevel(logging.WARNING)
    _settings.verbosity, _settings.timestamps, _settings.query = 0, Timestamps.AUTO, False


def shown(kind: Kind) -> bool:
    """Whether a line of ``kind`` is said at this verbosity."""
    return kind >= _LEAST_SHOWN.get(_settings.verbosity, DETAIL)


def say(text: str, *, kind: Kind = MILESTONE, err: bool = False) -> None:
    """Print ``text`` on stdout, or on stderr with ``err``, if a line of ``kind`` is said.

    Or stop trying to, if there is nobody left to print it to: every stream a command writes to
    can go away while it is running — a closed terminal, a dropped ssh session, a pipe into
    ``head`` — and none of that is worth turning finished work into a traceback.
    """
    if shown(kind):
        write(sys.stderr if err else sys.stdout, text)


def write(stream: TextIO, text: str) -> None:
    """Write ``text`` on ``stream``, every line of it stamped if lines there are, and flush it.

    Every line, not just the first: a summary of several lines, or a record with a traceback,
    is otherwise the one place in a log where a line does not start with the time, which is what
    anything reading the log goes by.

    A stream that has gone is passed over, whichever way it went: one whose other end closed
    raises :exc:`OSError`, and one closed or detached by whoever called ``main`` raises
    :exc:`ValueError`. Neither may reach the code that was saying something, least of all from
    a logging handler, which is never supposed to raise.

    A stream that is still there is not passed over. One that cannot encode a character — not
    UTF-8, with ``PYTHONIOENCODING=ascii`` or a locale that is not — gets the line with that
    character escaped (``3\\u201322``), since a dash is no reason to lose a result line or to
    stop a run; any other refusal is raised, as it would be anywhere else.
    """
    prefix = stamp(stream)
    if prefix:
        text = "\n".join(prefix + line for line in text.split("\n"))
    try:
        try:
            print(text, file=stream, flush=True)
        except UnicodeEncodeError:
            # Raised before anything is written: the whole line is encoded first.
            print(_encodable(text, stream), file=stream, flush=True)
    except OSError:
        pass
    except ValueError:
        if not _gone(stream):
            raise


def _encodable(text: str, stream: TextIO) -> str:
    """``text`` with what ``stream``'s encoding has no character for written as an escape."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return text.encode(encoding, "backslashreplace").decode(encoding)


def _gone(stream: TextIO) -> bool:
    """Whether ``stream`` was closed or detached, which is what a write to it raising
    :exc:`ValueError` means when it does not mean something else."""
    try:
        return bool(stream.closed)
    except ValueError:
        return True  # Detached: even asking whether it is closed raises.
    except AttributeError:
        return False


def stamp(stream: TextIO) -> str:
    """What a line written to ``stream`` now starts with: the time and a space, or nothing."""
    if not _stamped(stream):
        return ""
    return f"{_now():%Y-%m-%d %H:%M:%S} "


def _stamped(stream: TextIO) -> bool:
    if _settings.timestamps is Timestamps.ALWAYS:
        return True
    if _settings.timestamps is Timestamps.NEVER or _settings.query:
        return False
    return not to_the_journal(stream)


def to_the_journal(stream: TextIO) -> bool:
    """Whether ``stream`` is systemd's journal, which stamps every line it is given.

    ``JOURNAL_STREAM`` is the device and inode of the journal's socket, and is handed down to
    every process started under the service, so it is compared with what the stream really is:
    a command whose output the service redirected to a file is not writing to the journal.
    """
    value = os.environ.get("JOURNAL_STREAM")
    if not value:
        return False
    try:
        device, inode = (int(part) for part in value.split(":"))
        status = os.fstat(stream.fileno())
    except (ValueError, OSError, AttributeError):
        # io.UnsupportedOperation, for a stream with no descriptor, is both of the first two.
        return False
    return (status.st_dev, status.st_ino) == (device, inode)


_now: Callable[[], datetime] = datetime.now
"""The time a line is stamped with; a test fixes it."""


class _StderrHandler(logging.Handler):
    """Writes every record as one line on whatever ``sys.stderr`` is when it is written.

    Looked up each time rather than held, so that a test capturing stderr, or a program that
    points it elsewhere, gets the records. A stream that has gone is passed over as :func:`say`
    passes it over, rather than with logging's own report of a failed write, which is a traceback.
    One that is still there and refuses the record gets that report, through :meth:`handleError`,
    as any handler's would: a handler never raises into the code that was logging.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(_Formatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:  # noqa: BLE001 - logging's own way of reporting a record it cannot format
            self.handleError(record)
            return
        if sys.stderr is None:
            return
        try:
            write(sys.stderr, text)
        except Exception:  # noqa: BLE001 - a stream that refuses the record, for its own reason
            self.handleError(record)


class _Formatter(logging.Formatter):
    """``chess-ai: warning: ...``; below warnings, with the logger's name, since info and debug
    records are read to find out where something happened."""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname.lower()
        where = f" [{record.name}]" if record.levelno < logging.WARNING else ""
        text = f"chess-ai: {level}{where}: {record.getMessage()}"
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            text += "\n" + self.formatStack(record.stack_info)
        return text
