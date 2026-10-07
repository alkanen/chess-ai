"""Downloading Lichess's monthly dumps of rated standard games, as they are published.

A month is tens of gigabytes and takes an hour or more to fetch, so a download that stops part
way is the normal case rather than the exception: it is kept as a ``.part`` file and carried on
from where it stopped the next time it is asked for. Only a file whose checksum matches the one
Lichess publishes is given its real name, so a dump under its own name is always a whole one --
which is what lets a build trust it, and lets this skip one that is already there.

The files are kept compressed, because a build reads them that way; see
:mod:`chess_ai.dataset.sources`.
"""

import http.client
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import BinaryIO, Final, TextIO

from chess_ai.dataset.files import sha256_of, sync_directory, sync_file
from chess_ai.dataset.progress import INTERVAL, format_duration
from chess_ai.dataset.store import HELD_BY_ANOTHER, DatasetError

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None

LICHESS_URL: Final = "https://database.lichess.org/standard/"
"""Where Lichess publishes its dumps of rated standard games, and their checksums beside them."""

CHECKSUMS: Final = "sha256sums.txt"
"""The file that lists every published dump's SHA-256, as ``sha256sum`` writes them."""

DOWNLOADS_DIR: Final = "lichess"
"""Where the dumps are kept under the data directory, beside the datasets built from them."""

PART_SUFFIX: Final = ".part"
"""What a download is called until it is whole and checked."""

READ_BYTES: Final = 1 << 20
"""How much is read from the connection at a time."""

TIMEOUT: Final = 60.0
"""Seconds a connection may go without sending anything before the download gives up on it.

Not a limit on how long a download takes, which is an hour or more: a stalled one stops, keeps
what it has, and carries on from there when it is asked for again.
"""

_MONTH = re.compile(r"(\d{4})-(\d{2})")

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int | None], None]
"""Told the file's name, the bytes of it there are so far, and how many there will be if known."""


class DownloadError(DatasetError):
    """A dump could not be downloaded, or what was downloaded is not what was published."""


def parse_month(text: str) -> str:
    """``text`` if it is a month as the dumps are named, ``YYYY-MM``; else raise ``ValueError``."""
    found = _MONTH.fullmatch(text)
    if found is None:
        raise ValueError(f"{text!r} is not a month written YYYY-MM")
    try:
        date(int(found[1]), int(found[2]), 1)
    except ValueError as e:
        raise ValueError(f"{text!r} is not a month written YYYY-MM") from e
    return text


def dump_name(month: str) -> str:
    """What Lichess calls the dump of ``month``."""
    return f"lichess_db_standard_rated_{month}.pgn.zst"


_DUMP = re.compile(r"lichess_db_standard_rated_(\d{4}-\d{2})\.pgn(?:\.zst)?", re.IGNORECASE)


def dump_month(name: str) -> str | None:
    """The month a file called ``name`` is the Lichess dump of, or ``None`` if it is not one.

    Compressed or not: a dump decompressed with ``zstd -d`` keeps its name less the ``.zst``,
    and its games are the same month's.
    """
    found = _DUMP.fullmatch(name)
    if found is None:
        return None
    try:
        return parse_month(found[1])
    except ValueError:
        return None


def downloads_dir(data_dir: Path) -> Path:
    """Where downloaded dumps are kept under ``data_dir``."""
    return data_dir / DOWNLOADS_DIR


def download_months(
    months: Sequence[str],
    *,
    directory: Path,
    base_url: str | None = None,
    progress: ProgressCallback | None = None,
) -> list[Path]:
    """Download the dumps of ``months`` into ``directory``, and say where each one is.

    A dump already there is left alone: it was checked before it was given its name. One that
    was partly downloaded is carried on from where it stopped. Every month asked for has to be in
    the published checksums, which are fetched first, so that a mistyped month is refused before
    anything is downloaded rather than after the months before it.
    """
    base_url = LICHESS_URL if base_url is None else base_url
    checksums = _published_checksums(base_url)
    wanted = [dump_name(month) for month in months]
    missing = [name for name in wanted if name not in checksums]
    if missing:
        raise DownloadError(
            f"Lichess has not published {', '.join(missing)}; the months there are run from "
            f"{min(checksums, default='none')} to {max(checksums, default='none')}"
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise DownloadError(f"cannot make {directory}: {e.strerror}") from e
    return [
        _download(base_url + name, directory / name, checksums[name], progress)
        for name in dict.fromkeys(wanted)
    ]


def _published_checksums(base_url: str) -> dict[str, str]:
    """Every published dump's name, with its SHA-256 in lowercase hex."""
    url = base_url + CHECKSUMS
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise DownloadError(f"cannot fetch the published checksums from {url}: {_why(e)}") from e
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            # sha256sum marks a file hashed in binary mode with a leading asterisk.
            checksums[parts[1].lstrip("*")] = parts[0].lower()
    if not checksums:
        raise DownloadError(f"{url} lists no checksums, so nothing it names can be checked")
    return checksums


def _download(url: str, path: Path, expected: str, progress: ProgressCallback | None) -> Path:
    """Fetch ``url`` to ``path`` by way of a ``.part`` file, checked against ``expected``."""
    with _held(path):
        # Asked only once the lock is held: a download of the same month that finished while
        # this one was waiting to start has renamed its part into place by now.
        if path.exists():
            return path
        part = path.with_name(path.name + PART_SUFFIX)
        try:
            with part.open("ab") as out:
                _fetch(url, out, path.name, progress)
                sync_file(out)
        except OSError as e:
            raise DownloadError(
                f"could not write {part}: {e.strerror}; what was downloaded is kept, and asking "
                "for this month again carries on from there"
            ) from e
        actual = _sha256(part)
        if actual != expected:
            # Nothing in it can be trusted, including the part a resume would keep: it is the
            # file as a whole that does not match, and no one can say which bytes are wrong.
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"{path.name} does not match the checksum Lichess published for it (it hashes "
                f"to {actual}, not {expected}). The download has been deleted; ask for this "
                "month again to download it from the start"
            )
        os.replace(part, path)
        sync_directory(path.parent)
    return path


def _fetch(url: str, out: BinaryIO, name: str, progress: ProgressCallback | None) -> None:
    """Append what is left of ``url`` to ``out``, which holds however much of it is there.

    Asks for the rest with a ``Range`` request, and starts again from nothing if the server
    sends the whole file instead -- or a range that does not start where ``out`` ends, which
    would otherwise be appended as if it did.
    """
    have = out.seek(0, os.SEEK_END)
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:
            # Nothing past what is there already: the download was whole when it was stopped,
            # before it was checked. The checksum decides whether it really is.
            return
        raise DownloadError(f"cannot download {url}: {_why(e)}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise DownloadError(f"cannot download {url}: {_why(e)}") from e
    with response:
        if have and not _continues(response, have):
            out.truncate(0)
            have = 0
        length = response.headers.get("Content-Length")
        total = have + int(length) if length is not None and length.isdigit() else None
        if progress is not None:
            progress(name, have, total)
        while True:
            # Only the reading is the connection's to fail: an error writing is the disk's, and
            # is said as one by the caller.
            try:
                data = response.read(READ_BYTES)
            except (OSError, http.client.HTTPException) as e:
                out.flush()
                raise DownloadError(
                    f"the download of {url} stopped after {have:,} bytes: {_why(e)}. What was "
                    "downloaded is kept, and asking for this month again carries on from there"
                ) from e
            if not data:
                break
            out.write(data)
            have += len(data)
            if progress is not None:
                progress(name, have, total)
    if total is not None and have < total:
        raise DownloadError(
            f"the download of {url} ended after {have:,} of its {total:,} bytes. What was "
            "downloaded is kept, and asking for this month again carries on from there"
        )


def _continues(response, have: int) -> bool:
    """Whether ``response`` is the rest of a file of which ``have`` bytes are already here."""
    if response.status != 206:
        return False
    found = re.fullmatch(r"bytes (\d+)-\d+/(\d+|\*)", response.headers.get("Content-Range", ""))
    return found is not None and int(found[1]) == have


def _sha256(path: Path) -> str:
    try:
        return sha256_of(path)
    except OSError as e:
        raise DownloadError(f"cannot read {path} to check it: {e.strerror}") from e


@contextmanager
def _held(path: Path) -> Iterator[None]:
    """Hold the right to download ``path``, or refuse because another download has it.

    Two downloads of one month into one ``.part`` would each append to it, and the checksum
    would only say so an hour later. Held as :func:`~chess_ai.dataset.store.dataset_lock` holds
    a build's: a lock the operating system lets go of when the process ends, so a download that
    was killed leaves nothing to clear up. Where locks cannot be had, downloads go unguarded.
    """
    if fcntl is None:
        yield
        return
    lock = path.with_name(f".{path.name}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_RDONLY, 0o644)
    except OSError as e:
        LOGGER.warning("chess-ai: cannot lock the download of %s: %s", path.name, e.strerror)
        yield
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in HELD_BY_ANOTHER:
                raise DownloadError(
                    f"{path.name} is already being downloaded into {path.parent}; "
                    "wait for that to finish"
                ) from e
            LOGGER.warning("chess-ai: cannot lock the download of %s: %s", path.name, e.strerror)
        yield
    finally:
        os.close(fd)


def _why(error: BaseException) -> str:
    """What went wrong with a connection, in a few words."""
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code} {error.reason}"
    if isinstance(error, urllib.error.URLError):
        return str(error.reason)
    return getattr(error, "strerror", None) or str(error) or type(error).__name__


class DownloadPrinter:
    """Prints a download's progress on one line that rewrites itself, at most every
    :data:`~chess_ai.dataset.progress.INTERVAL`, as :class:`~chess_ai.dataset.ProgressPrinter`
    does a build's. A line at a time where the stream is not a terminal.

    The rate is of this session's bytes only: a resumed download starts with most of the file
    already there, and counting that in would promise a minute for what takes an hour.
    """

    def __init__(self, stream: TextIO | None = None, interval: float = INTERVAL) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._interval = interval
        self._rewrite = _is_terminal(self._stream)
        self._name: str | None = None
        self._started = 0.0
        self._first = 0
        self._last = 0.0
        self._unfinished = False

    def __call__(self, name: str, have: int, total: int | None) -> None:
        now = time.monotonic()
        if name != self._name:
            self.finish()
            self._name, self._started, self._first, self._last = name, now, have, -self._interval
        done = total is not None and have >= total
        if not done and now - self._last < self._interval:
            return
        self._last = now
        line = _download_line(name, have, total, have - self._first, now - self._started)
        try:
            if self._rewrite:
                self._stream.write(f"\r\x1b[K{line}" + ("\n" if done else ""))
                self._unfinished = not done
            else:
                self._stream.write(f"{line}\n")
            self._stream.flush()
        except Exception:  # noqa: BLE001 - a report that cannot be made is not a failed download
            pass

    def finish(self) -> None:
        """End the line being rewritten, if one is still open, without claiming anything."""
        if not self._unfinished:
            return
        self._unfinished = False
        try:
            self._stream.write("\n")
            self._stream.flush()
        except Exception:  # noqa: BLE001 - see above
            pass


def _download_line(name: str, have: int, total: int | None, fetched: int, seconds: float) -> str:
    """One line saying how far a download has got and how much longer it has to go."""
    parts = [name]
    if total:
        parts.append(f"{have / total:.0%} of {_size(total)}")
    else:
        parts.append(_size(have))
    if seconds > 0 and fetched > 0:
        rate = fetched / seconds
        parts.append(f"{_size(rate)}/s")
        if total and have < total:
            parts.append(f"{format_duration((total - have) / rate)} left")
    return ", ".join(parts)


def _size(count: float) -> str:
    """A number of bytes as something readable at a glance: "512 B", "3.4 MB", "29.1 GB"."""
    for unit in ("B", "kB", "MB", "GB"):
        if count < 1000 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1000
    raise AssertionError("unreachable")


def _is_terminal(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False
