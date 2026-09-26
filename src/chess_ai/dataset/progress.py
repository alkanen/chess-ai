"""Progress while a dataset is being built, which for millions of games is the whole day.

A build of a Lichess monthly dump runs for hours, and the only question anyone has while it
does is whether to wait. So it reports what it has done and, where it can, what is left: the
sources' sizes are known up front, so how far into them the reading has got answers it.

The reporting is a callback rather than printing, so that the build has no opinion about
where it is running — the CLI prints, a test collects, and a web server could push.
"""

import sys
from dataclasses import dataclass
from typing import Final, TextIO

INTERVAL: Final = 1.0
"""How often a reporter that throttles will actually report, in seconds."""


@dataclass(frozen=True)
class Progress:
    """How far a build has got. The last one of a build is its summary."""

    games_read: int
    games_kept: int
    positions: int
    bytes_read: int
    bytes_total: int
    """The sources' total size, which is known before a byte is read."""
    seconds: float
    """How long the build has been running."""
    done: bool
    """Whether this is the final report of the build."""

    @property
    def games_per_second(self) -> float:
        return self.games_read / self.seconds if self.seconds > 0 else 0.0

    @property
    def fraction(self) -> float:
        """How much of the input has been read, from 0 to 1."""
        if self.bytes_total <= 0:
            return 1.0 if self.done else 0.0
        return min(self.bytes_read / self.bytes_total, 1.0)

    @property
    def seconds_remaining(self) -> float | None:
        """How much longer at this rate, or ``None`` when there is no way to tell yet."""
        if self.done or self.bytes_total <= 0 or self.bytes_read <= 0 or self.seconds <= 0:
            return None
        return self.seconds * (self.bytes_total - self.bytes_read) / self.bytes_read


class ProgressPrinter:
    """Prints progress on one line that rewrites itself, at most every :data:`INTERVAL`.

    Rewriting one line keeps hours of progress out of a terminal's scrollback; the summary
    line at the end is the one that stays. Anywhere that is not a terminal — a log file, a
    test — gets a line at a time instead, because a carriage return is not readable there.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        interval: float = INTERVAL,
        rewrite: bool | None = None,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._interval = interval
        self._last = -interval
        self._rewrite = rewrite if rewrite is not None else _is_terminal(self._stream)
        self._unfinished = False
        """Whether a line has been written that nothing has ended yet."""

    def __call__(self, progress: Progress) -> None:
        if not progress.done and progress.seconds - self._last < self._interval:
            return
        self._last = progress.seconds
        line = format_progress(progress)
        if not self._rewrite:
            # Redirected into a log file, where a rewritten line is one unreadable line.
            self._stream.write(f"{line}\n")
        else:
            # Clear to the end of the line: the line before may have been longer than this one.
            self._stream.write(f"\r\x1b[K{line}" + ("\n" if progress.done else ""))
            self._unfinished = not progress.done
        self._stream.flush()

    def finish(self) -> None:
        """End the line being rewritten, if one is still open, without claiming anything.

        A build that fails never reports itself done, so the line it left in a terminal has no
        newline on it and whatever is printed next — the error, on the same stream — lands on the
        end of it. This closes the line and says nothing else, which is the point: only a build
        that finished may print that it did.
        """
        if not self._unfinished:
            return
        self._unfinished = False
        self._stream.write("\n")
        self._stream.flush()


def format_progress(progress: Progress) -> str:
    """One line saying what the build has done and how much longer it has to go."""
    parts = [
        f"{progress.games_kept:,} games",
        f"{progress.positions:,} positions",
        f"{progress.games_per_second:,.0f} games/s",
    ]
    skipped = progress.games_read - progress.games_kept
    if skipped:
        parts.append(f"{skipped:,} skipped")
    if progress.done:
        return f"built {', '.join(parts)} in {format_duration(progress.seconds)}"
    if progress.bytes_total > 0:
        parts.insert(0, f"{progress.fraction:.0%}")
        remaining = progress.seconds_remaining
        if remaining is not None:
            parts.append(f"{format_duration(remaining)} left")
    return ", ".join(parts)


def format_duration(seconds: float) -> str:
    """A length of time as something readable at a glance: "12s", "4m 03s", "2h 07m"."""
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m {whole % 60:02d}s"
    return f"{whole // 3600}h {whole % 3600 // 60:02d}m"


def _is_terminal(stream: TextIO) -> bool:
    """Whether ``stream`` is something a carriage return means anything to."""
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False
