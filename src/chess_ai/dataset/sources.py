"""Which PGN files a build reads, and reading the games out of one of them.

A build is given paths on a command line, which may be files, directories or globs — and a
glob the shell did not expand, because a pattern matching a year of monthly dumps is easier
to type than the twelve names, and quoting it is easier than remembering not to.

Reading a file says how far through it it has got, which is the only honest basis for a time
remaining: games differ in length, and a dump's own game count is not written anywhere.

A file can also be cut into :class:`ByteRange` pieces of whole games, which is what lets a
build read one file in several processes at once. The cutting is a cheap scan for game
boundaries rather than a parse, so it costs a few reads per piece and not a pass over the
file.

A Lichess dump comes compressed with zstd, as ``.pgn.zst``, and is read that way rather than
decompressed to disk first: a recent month is about 30 GB compressed and seven times that as
text. A compressed stream cannot be entered part way through, so such a file is not cut by
offset; the process that reads it decompresses it and cuts the *text* instead, into
:class:`TextPiece` pieces it hands over whole. See :func:`text_pieces`.
"""

import glob
import io
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Final, TextIO

import chess.pgn
import zstandard

from chess_ai.dataset.store import DatasetError

SUFFIX: Final = ".pgn"
"""What a PGN file is called, and one of the two things picked up from a directory."""

COMPRESSED_SUFFIX: Final = ".pgn.zst"
"""What a zstd-compressed PGN file is called, which is how Lichess publishes its dumps."""

ENCODING: Final = "utf-8-sig"
"""How a PGN file is read: UTF-8, past a byte-order mark some exporters write.

PGN's own specification says Latin-1 and the world writes UTF-8, so bytes that are neither
are replaced rather than raised over. A player's name with one broken character in it is not
a reason to abandon an import.
"""

_GLOB = "*?["

SEPARATOR: Final = b"[Event "
"""What the first line of a game looks like, which is what a boundary is found by."""

ALIGN_SCAN: Final = 64 << 10
"""Bytes read at a time while looking for the next boundary; a game is a fraction of this."""

READ_REPORT_BYTES: Final = 4 << 20
"""Bytes read between calls to a reader's ``on_read``, which is how a long read says it is alive.

Counted under the parser rather than around it, because ``chess.pgn.read_game`` consumes a whole
file looking for a header before it gives up on finding one: a file with no games in it is a
single call, and nothing above it is asked anything for as long as that takes. At the ~4.7 MB/s
such a file is read at, four megabytes is about a report a second.
"""


@dataclass(frozen=True)
class Source:
    """One PGN file to read, and how large it is, which is what makes progress an estimate.

    For a compressed file the size is the compressed one, and so is every count of bytes read
    from it: that is the number known before the file is read, and the only one a percentage
    can be taken of.
    """

    path: Path
    bytes: int

    @property
    def compressed(self) -> bool:
        """Whether the file is zstd-compressed, which is decided by its name ending in ``.zst``."""
        return self.path.name.lower().endswith(".zst")


def resolve_sources(patterns: Sequence[str]) -> list[Source]:
    """The PGN files ``patterns`` name, in order, without repeats.

    Each pattern is a file, a directory (its own ``*.pgn`` and ``*.pgn.zst`` files, in name
    order), or a glob.
    A pattern matching nothing is an error: a mistyped path that quietly built an empty
    dataset would only be noticed by the run that trained on it.
    """
    sources: list[Source] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matched = _matches(pattern)
        if not matched:
            raise DatasetError(f"no PGN files match {pattern!r}")
        for path in matched:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            sources.append(Source(path=path, bytes=path.stat().st_size))
    _refuse_both_copies(sources, seen)
    return sources


def _refuse_both_copies(sources: Sequence[Source], seen: set[Path]) -> None:
    """Refuse a dump named together with its own decompressed copy, which is its games twice.

    ``zstd -d`` keeps the file it decompressed, so a dump looked at where it was downloaded
    leaves ``x.pgn`` beside ``x.pgn.zst``, and a build of the directory would read the month
    twice: every game in the same split both times, and nothing to say so.

    Refused rather than one of them skipped, because neither is safe to pick for the person.
    The ``.pgn`` is written in place while ``zstd`` runs, so it may be half a month; and it may
    as well be a copy somebody edited on purpose. Naming the files, or moving one, says which.
    """
    pairs = [
        (source.path, counterpart)
        for source in sources
        if source.compressed
        # Looked for beside the link and beside its target, since a dump is often a link to
        # another disk and either can hold the copy. Beside the target matters even when the
        # build names that disk too: the dump there is dropped as a duplicate of the link if the
        # link came first, which leaves the link the only source to find the copy from.
        for counterpart in sorted(
            {source.path.with_suffix("").resolve(), source.path.resolve().with_suffix("")}
        )
        if counterpart != source.path.resolve() and counterpart in seen
    ]
    if pairs:
        raise DatasetError(
            "these dumps are named together with their decompressed copies, which would read "
            "their games twice: "
            + "; ".join(f"{packed} and {plain}" for packed, plain in pairs)
            + ". Name the files to read, or move one copy of each out of the way"
        )


def _matches(pattern: str) -> list[Path]:
    """The files one pattern names, in the order they will be read."""
    path = Path(pattern)
    if path.is_dir():
        return sorted(child for child in path.iterdir() if _is_pgn(child))
    if any(character in pattern for character in _GLOB):
        return sorted(Path(match) for match in glob.glob(pattern) if _is_pgn(Path(match)))
    if not path.is_file():
        raise DatasetError(f"no such PGN file: {pattern}")
    return [path]


def _is_pgn(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(SUFFIX) or name.endswith(COMPRESSED_SUFFIX))


@dataclass(frozen=True)
class ByteRange:
    """A run of whole games inside one PGN file, as byte offsets into it.

    ``source`` is the index of the file in the build's own list of them, which is what a game
    record stores and what the manifest counts against.
    """

    source: int
    path: Path
    start: int
    end: int

    @property
    def bytes(self) -> int:
        """How much of the file this piece is."""
        return self.end - self.start


def align_to_game(
    handle: BinaryIO, offset: int, size: int, on_scan: Callable[[], None] | None = None
) -> int:
    """The first byte of the first whole game at or after ``offset``, or ``size`` if none is left.

    A boundary is an ``[Event `` tag at the start of a line with a blank line in front of it,
    which is how PGN separates one game from the next, and is as close to safe as a cut can be
    made without parsing the file.

    Not entirely safe, and the exception is worth stating rather than hoping for. A ``{}``
    comment may contain anything, including a blank line followed by ``[Event `` -- which is
    what an annotated game looks like when a model game has been pasted into the notes. That
    matches here exactly as a real separator does, and the cut lands inside the comment: the
    game before it is kept with its moves stopping early, and the remainder comes back as a
    game that will not parse and is counted as one ``unreadable`` skip. Both are quiet.
    Measured on a file where one game in six is annotated that way, cutting at 64 KiB produced
    12,004 games where reading it in one process produced 12,000.

    Nothing bounded can tell the two apart, because the pasted game carries headers of its own;
    only tracking ``{}`` depth from a known point outside one could, and that is a parse of
    everything before the cut, which is the thing being avoided. So the cost is named here and
    in the README rather than claimed away. It scales with the number of cuts rather than the
    size of the file, so a larger ``target_bytes`` makes it rarer and never absent.

    The blank line is required rather than only the tag because a ``{}`` comment may contain
    any text at all, including something that looks like the start of a game. A cut inside a
    game would not fail loudly -- it would keep the first half as a game whose moves stop
    early, and count the second half as rubble -- so the test for a boundary is the stricter
    one.

    A boundary that is *missed*, in a file whose games are not separated by a blank line, only
    makes the pieces uneven: every byte still falls inside exactly one of them, and every cut
    is still at some game's first byte, so no game is read twice or left out.
    """
    if offset <= 0:
        return 0
    # Reading starts before `offset` so that a boundary beginning exactly there is still found
    # with the blank line in front of it to recognise it by.
    at = max(0, offset - _LEAD)
    while at < size:
        handle.seek(at)
        window = handle.read(ALIGN_SCAN)
        if on_scan is not None:
            # A file with no boundary in it is scanned to the end in one call, which for a dump
            # is half a minute inside a function that otherwise says nothing until it returns.
            on_scan()
        if not window:
            return size
        searched = 0
        while True:
            found = window.find(SEPARATOR, searched)
            if found < 0:
                break
            start = at + found
            if start >= offset and _separated(window, found):
                return start
            searched = found + 1
        if len(window) < ALIGN_SCAN:
            # Read to the end of the file with no boundary left in it. Without this the step
            # below is negative on a short final read, and the scan walks backwards for ever.
            return size
        # Overlapping the last few bytes, so that a boundary straddling the end of this window
        # is found by the next one, lead-in and all.
        at += len(window) - (len(SEPARATOR) + _LEAD)
    return size


_LEAD: Final = 16
r"""Bytes read before an offset, and the overlap between one scan window and the next.

Not a bound on how far :func:`_separated` looks behind a candidate -- it walks back over
whatever whitespace is actually on the blank line, so a separator of any width is recognised.
This is the read-behind that gives a boundary sitting exactly at an offset something to be
recognised *by* (``at = max(0, offset - _LEAD)``), and the overlap that keeps one straddling
the end of a window from being missed by both.

So the only separator this constant can hide is one whose blank line is wider than it and that
falls at a window edge, and hiding is the harmless direction: the scan takes the next boundary
and the pieces come out a little uneven. It was two bytes when the test was two bytes, and four
when the test needed ``\r\n\r\n``; sixteen is simply a comfortable overlap now that the test
sizes itself.
"""

_BLANK: Final = frozenset(b" \t")
"""What a blank line may hold and still be blank, which is what exporters leave behind."""


def _separated(window: bytes, found: int) -> bool:
    r"""Whether the ``[Event `` at ``found`` in ``window`` has a blank line in front of it.

    A line ending, then a line with nothing on it but whitespace, then another line ending.

    Tested as whole endings rather than as two bytes, because the two-byte version was not the
    strict test it was meant to be: ``\n`` preceded by ``\n`` or ``\r`` accepts ``\r\n``, which
    is how *every* line begins in a CRLF file. The blank line was therefore only ever required
    of files written with bare ``\n``, and a CRLF game whose comment held a line starting with
    ``[Event `` was cut in half -- the first part kept as a game whose moves stop early, the
    second counted as rubble.

    Whitespace on the blank line is allowed because exporters leave it there and the line is
    still blank. Refusing it is not a wrong dataset but a worse failure to notice: one exporter
    writes every separator in a file, so a trailing space is not one boundary missed but all of
    them, and the file cuts into a single piece that is read on one core for hours. A line of
    *text* still fails, which is the strictness this is for.

    Walked backwards a byte at a time rather than sliced, for two reasons. Slicing the window
    prefix cost a copy per candidate, which is invisible where a boundary is found quickly and
    quadratic on a file where none is -- the one where every candidate is tested. And any fixed
    bound on how far back to look is a cliff: a separator one byte wider than the bound fails,
    and by the argument above it fails for the whole file at once. Stepping over the whitespace
    that is actually there has neither cost.

    Reading less than there is can only answer ``False`` where a longer look would have said
    ``True``: a boundary hidden this way makes the pieces uneven, where a boundary imagined
    would cut a game in half. Only a candidate within a line or so of the window's start can be
    hidden, and the scan simply takes the next one.
    """
    at = _before_ending(window, found)
    if at is None:
        return False
    while at > 0 and window[at - 1] in _BLANK:
        at -= 1
    return _before_ending(window, at) is not None


def _before_ending(window: bytes, at: int) -> int | None:
    r"""Where the line ending that finishes at ``at`` starts, or ``None`` if there is not one.

    ``\r\n`` before ``\n`` so that a CRLF ending is consumed whole rather than leaving the
    ``\r`` behind to be mistaken for the blank line's content.
    """
    if at >= 2 and window[at - 2] == 0x0D and window[at - 1] == 0x0A:
        return at - 2
    if at >= 1 and window[at - 1] == 0x0A:
        return at - 1
    return None


def game_ranges(
    source: Source,
    index: int,
    target_bytes: int,
    on_scan: Callable[[], None] | None = None,
) -> list[ByteRange]:
    """``source`` cut into pieces of whole games, each about ``target_bytes`` long.

    A target rather than a rule: a piece runs on to the next boundary. Pieces smaller than the
    file divided by the number of workers on purpose -- with one piece each, a worker that drew
    the slow end of a file holds up the whole build, and with many each of them stays fed to the
    last one.

    One pass over the file, whatever its games look like. Each offset's scan starts where the
    last one stopped rather than at the offset itself, which matters far more than it sounds:
    :func:`align_to_game` runs to the end of the file when it finds no boundary, so scanning
    from each offset independently reads the whole tail once per offset. That is quadratic in
    the file's size -- for a dump whose games are not separated by a blank line it is hours of
    silent reading before a single game is parsed. Keeping the scan monotonic makes the whole
    cutting cost one read of the file.

    Raises :exc:`~chess_ai.dataset.store.DatasetError` if the file cannot be read, which is the
    same answer :meth:`PgnReader.open` gives for the same thing.
    """
    if target_bytes <= 0:
        raise ValueError(f"a piece has to be some bytes long, not {target_bytes}")
    size = source.bytes
    if size <= 0:
        return []
    cuts: list[int] = []
    try:
        with source.path.open("rb") as handle:
            found = 0
            for offset in range(0, size, target_bytes):
                # Never below the last cut: everything under it has been scanned already, and a
                # boundary exactly at it is the piece that is already being measured, not the
                # start of the next one.
                # Clamped, because align_to_game reads a whole window from below `size` and so
                # can accept a boundary above it. That only happens for a file which grew after
                # resolve_sources measured it, and a piece reaching past the size everything
                # else is counted against puts the progress past 100% and then backwards.
                start = min(
                    align_to_game(
                        handle, max(offset, found + 1) if cuts else offset, size, on_scan
                    ),
                    size,
                )
                cuts.append(start)
                found = start
                if start >= size:
                    # Nothing left to find. Every later offset would rescan the tail to say so.
                    break
    except OSError as e:
        raise DatasetError(f"cannot read {source.path}: {e.strerror}") from e
    cuts.append(size)
    # Empty pieces dropped: two offsets can align to one boundary, and everything past the last
    # boundary in the file aligns to its end.
    return [
        ByteRange(source=index, path=source.path, start=start, end=end)
        # Not strict: the ends are the starts shifted along by one, so they are one shorter.
        for start, end in zip(cuts, cuts[1:], strict=False)
        if end > start
    ]


class _Window(io.RawIOBase):
    """A file read through, counting as it goes, and stopping early if asked to.

    Two jobs that are really one. :func:`games_in_range` needs a file that ends where its piece
    does, so the parser stops without being told; and both readers need something under the
    parser counting bytes, because the parser is not asked how far it has got until it returns --
    and on a file with no games in it, it does not return until the file is finished.

    ``end`` of ``None`` reads to the file's own end, which is what a whole-file read wants, and
    deliberately not the size the build recorded: reading to the real end is what the serial path
    has always done.
    """

    def __init__(
        self,
        handle: BinaryIO,
        start: int,
        end: int | None,
        on_read: Callable[[int], None] | None = None,
    ) -> None:
        self._handle = handle
        self._handle.seek(start)
        self._start = start
        self._left = None if end is None else max(0, end - start)
        self._read = 0
        self.on_read = on_read
        """Told the position every :data:`READ_REPORT_BYTES`; may be set part way through."""
        self._since = 0

    @property
    def position(self) -> int:
        """How far into the *file* this has handed bytes out, as an absolute offset.

        What a build reports progress from. It runs ahead of the parser by whatever the buffers
        above it are holding -- tens of kilobytes -- because it counts what was handed up rather
        than what has been parsed out of it. That is the right trade for a progress bar: it costs
        nothing, and while the parser is inside one long call it is the only number there is.
        """
        return self._start + self._read

    @property
    def name(self) -> str:
        """The wrapped file's name, so that reading through this looks like reading the file.

        ``TextIOWrapper`` and ``BufferedReader`` both delegate ``name`` downwards, so without
        this the object handed to the parser has no name at all -- and something asking which
        file it is reading gets nothing.
        """
        return self._handle.name

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self._left == 0:
            return 0
        view = memoryview(buffer)
        want = len(view) if self._left is None else min(len(view), self._left)
        got = self._handle.readinto(view[:want]) or 0
        if self._left is not None:
            self._left -= got
        self._read += got
        if self.on_read is not None and got:
            self._since += got
            if self._since >= READ_REPORT_BYTES:
                self._since = 0
                self.on_read(self.position)
        return got


def reading(
    handle: BinaryIO,
    start: int,
    end: int | None,
    on_read: Callable[[int], None] | None = None,
    *,
    compressed: bool = False,
) -> tuple[TextIO, _Window]:
    """``handle`` from ``start`` to ``end`` as text, and the window counting underneath it.

    ``compressed`` decompresses what the window hands up, so the window goes on counting bytes
    of the *file*: those are what a compressed source's size is measured in.
    """
    window = _Window(handle, start, end, on_read)
    raw: io.RawIOBase = _Decompressing(window) if compressed else window
    text = io.TextIOWrapper(io.BufferedReader(raw), encoding=ENCODING, errors="replace")
    return text, window


DECOMPRESS_READ: Final = 128 << 10
"""Compressed bytes fed to the decompressor at a time: about a megabyte of PGN out of it."""


class _Decompressing(io.RawIOBase):
    """A zstd-compressed file read through as the text it holds.

    Written on a decompression object rather than taken from ``zstandard``'s own stream reader,
    because that one ends quietly where the file does: a dump cut off part way through -- a
    download that stopped, a copy onto a full disk -- reads as a shorter dump with nothing to
    say it is one. Here a file that ends inside a frame raises :exc:`zstandard.ZstdError`, as a
    corrupt one does, and a build counts it as a file whose contents stopped making sense.

    Frames follow one another, as ``zstd`` writes them for concatenated input, and each is
    decompressed in turn.
    """

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self._frame = zstandard.ZstdDecompressor().decompressobj()
        self._in_frame = False
        """Whether the current frame has been given any input, so that ending now cuts it."""
        self._pending = memoryview(b"")
        self._ended = False

    @property
    def name(self) -> str:
        return self._raw.name

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending:
            if self._ended:
                return 0
            data = self._raw.read(DECOMPRESS_READ)
            if not data:
                self._ended = True
                if self._in_frame:
                    raise zstandard.ZstdError("the file ends part way through a zstd frame")
                return 0
            self._pending = memoryview(self._decompressed(data))
        size = min(len(buffer), len(self._pending))
        buffer[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size

    def _decompressed(self, data: bytes) -> bytes:
        """What ``data`` decompresses to, starting the next frame wherever one ends."""
        out = []
        while data:
            out.append(self._frame.decompress(data))
            self._in_frame = True
            if not self._frame.eof:
                break
            data = self._frame.unused_data
            self._frame = zstandard.ZstdDecompressor().decompressobj()
            self._in_frame = False
        return b"".join(out)


@dataclass(frozen=True)
class TextPiece:
    """A run of whole games out of a compressed file, as the text itself.

    What a compressed file is cut into in place of a :class:`ByteRange`, by the process that
    decompresses it; see :func:`text_pieces`. ``end`` is how far into the *compressed* file the
    decompression had got when the piece was cut, which is what a build counts its progress in.

    ``error`` is what stopped the decompression, carried by the last piece it produced: the games
    in ``text`` are read first, and the error is raised where the next game would have been.

    ``start`` and ``stop`` are where ``text`` lies in the file's decompressed text, which is how a
    resumed build finds its place again; see :func:`text_pieces`.
    """

    source: int
    path: Path
    text: bytes
    end: int
    error: Exception | None = None
    start: int = 0
    stop: int = 0

    @property
    def bytes(self) -> int:
        """How much text this piece is, which is what a worker holds while it reads it."""
        return len(self.text)


@dataclass(frozen=True)
class CompressedTail:
    """The rest of a compressed file from a point no boundary could be found after.

    The compressed counterpart of a :class:`ByteRange` too large for a worker: the text from
    ``start`` -- an offset into the *decompressed* file -- has run past the largest piece without
    a game boundary in it, so it cannot be cut, and is read in the build's own process to the end
    of the file. Not to the next boundary: nothing here can say where that is without holding
    the text up to it, which is what the piece size bounds. ``end`` is the file's size, which is
    where the progress stands once it is read.

    Carries no open file or decompressor, so it can wait in a queue for as long as it likes: it
    is read by decompressing the file again from the beginning and passing over ``start`` bytes,
    which costs a few seconds of a file shaped this way and saves holding one open meanwhile.
    """

    source: int
    path: Path
    start: int
    end: int
    bytes: int
    """How much text had gone by without a boundary, which is what made this a tail."""


type Piece = ByteRange | TextPiece | CompressedTail
"""Anything a file is cut into for reading."""


def text_pieces(
    source: Source,
    index: int,
    target_bytes: int,
    most_bytes: int,
    stop: Callable[[], bool] = lambda: False,
    resume_at: int = 0,
    on_passing: Callable[[], None] | None = None,
) -> Iterator[TextPiece | CompressedTail]:
    """``source``, a compressed file, decompressed and cut into pieces of whole games.

    The counterpart of :func:`game_ranges` for a file that cannot be entered part way through:
    this reads it from the start, and each piece is the text itself, about ``target_bytes`` of
    it, ending at the first game boundary past that. A boundary is what :func:`align_to_game`
    takes one to be, so the same file read compressed or not is cut at the same kind of place,
    and gives the same games.

    A generator, so that the file is decompressed only as fast as the pieces are wanted: the
    caller holds a bounded number of them, whatever the size of the file. ``stop`` is asked
    before each piece and ends the reading when it says so -- for a build that has its games,
    or a file an earlier piece of which gave up.

    Text that runs past ``most_bytes`` without a boundary ends the cutting with a
    :class:`CompressedTail` for the rest of the file, rather than growing a piece without limit.

    A failure to read or decompress the file is not raised: it ends the cutting with a piece of
    whatever whole games were in hand and the error to raise after them. An :exc:`OSError`
    means the file went away; anything else, that its contents stopped making sense.

    ``resume_at`` passes over the pieces that start before that offset into the text, which is
    where a resumed build carries on from: one of the pieces' starts. The file is still cut from
    its beginning, and not from there, because where a cut falls depends on how the text came out
    of the decompressor before it -- and a cut that falls elsewhere can, in a file whose comments
    hold whole games, read other games. Cut the same way, it is the same pieces after it.
    ``on_passing`` is told of each piece passed over, which for a month is minutes of saying
    nothing otherwise.
    """
    if target_bytes <= 0:
        raise ValueError(f"a piece has to be some bytes long, not {target_bytes}")
    text = bytearray()
    # The text before `text`, which is where its first game starts in the decompressed file.
    taken = 0
    searched = 0
    window: _Window | None = None

    def read() -> int:
        return 0 if window is None else min(window.position, source.bytes)

    try:
        with source.path.open("rb") as handle:
            window = _Window(handle, 0, None)
            stream = _Decompressing(window)
            ended = False
            while not stop():
                cut = None
                while cut is None:
                    if len(text) >= target_bytes:
                        cut = _boundary_after(text, max(target_bytes, searched))
                        if cut is not None:
                            break
                        # Overlapping the end, so that a separator split across two reads is
                        # found once the rest of it arrives.
                        searched = max(target_bytes, len(text) - len(SEPARATOR) + 1)
                        if len(text) > most_bytes:
                            # A tail always starts at or past a resumed build's place: the place is
                            # a piece's start, and nothing is cut after a tail.
                            yield CompressedTail(
                                source=index,
                                path=source.path,
                                start=taken,
                                end=source.bytes,
                                bytes=len(text),
                            )
                            return
                    if ended:
                        cut = len(text)
                        break
                    data = stream.read(DECOMPRESS_READ * 8)
                    if data:
                        text += data
                    else:
                        ended = True
                if cut == 0:
                    return
                if taken < resume_at:
                    if on_passing is not None:
                        on_passing()
                else:
                    yield TextPiece(
                        source=index,
                        path=source.path,
                        text=bytes(text[:cut]),
                        end=read(),
                        start=taken,
                        stop=taken + cut,
                    )
                del text[:cut]
                taken += cut
                searched = 0
    except (OSError, zstandard.ZstdError) as e:
        # The games before the last boundary in hand are whole and are read; the one it stopped
        # in is not, and is counted as unreadable by whoever reads the error.
        whole = _last_boundary(text)
        yield TextPiece(
            source=index,
            path=source.path,
            # Only what comes after the place a resumed build carries on from. Whatever came
            # before it was read by the build that stopped, and the error is still this file's.
            text=bytes(text[max(0, resume_at - taken) : whole]),
            end=read(),
            error=e,
            start=max(taken, resume_at),
            stop=max(taken + whole, resume_at),
        )


def _boundary_after(text: bytearray, offset: int) -> int | None:
    """Where the first game boundary at or after ``offset`` in ``text`` starts, if there is one.

    ``text`` starts at the start of a game, so a blank line in front of a candidate is always in
    it to be recognised by; see :func:`_separated`.
    """
    searched = offset
    while True:
        found = text.find(SEPARATOR, searched)
        if found < 0:
            return None
        if found > 0 and _separated(text, found):
            return found
        searched = found + 1


def _last_boundary(text: bytearray) -> int:
    """Where the last game in ``text`` starts, which is where the whole games in it end."""
    searched = len(text)
    while True:
        found = text.rfind(SEPARATOR, 0, searched)
        if found <= 0:
            return 0
        if _separated(text, found):
            return found
        searched = found


class _Screened(chess.pgn.GameBuilder):
    """A game builder that does not parse the moves of a game its headers have already decided.

    The moves are what parsing a game costs, and a build filtered down to a few percent of a dump
    would otherwise parse every one of them only to throw the game away. python-chess reads a game
    it is told to skip as fast as it can find the end of it; what comes back has the headers and
    no moves, and :func:`~chess_ai.dataset.games.game_records` reaches the same verdict on it from
    the headers before it looks for any.
    """

    def __init__(self, skip: Callable[[chess.pgn.Headers], bool]) -> None:
        super().__init__()
        self._skip = skip

    def end_headers(self) -> chess.pgn.SkipType | None:
        return chess.pgn.SKIP if self._skip(self.game.headers) else None


def read_game(
    text: TextIO, skip: Callable[[chess.pgn.Headers], bool] | None = None
) -> chess.pgn.Game | None:
    """The next game in ``text``, without its moves where ``skip`` says so of its headers."""
    if skip is None:
        return chess.pgn.read_game(text)
    return chess.pgn.read_game(text, Visitor=lambda: _Screened(skip))


class _Passing(chess.pgn.BaseVisitor[bool]):
    """A visitor that skips a whole game, headers and all, and says only that there was one."""

    def begin_game(self) -> chess.pgn.SkipType:
        return chess.pgn.SKIP

    def result(self) -> bool:
        return True


def pass_games(text: TextIO, games: int) -> int:
    """Read past the next ``games`` games in ``text``, and say how many there were.

    How a resumed build gets back to the game it stopped before, when it was reading a stretch of
    a file in its own process and not a piece at a time. python-chess reads a game it skips to the
    same line it reads one it parses to, the blank line after it: a build already skips the moves
    of every game its filters leave out, and the games after those are the games they always were.
    This skips the headers as well, which is as fast as python-chess finds the end of a game.
    """
    passed = 0
    while passed < games and chess.pgn.read_game(text, Visitor=_Passing) is not None:
        passed += 1
    return passed


def games_in_range(
    piece: ByteRange,
    on_read: Callable[[int], None] | None = None,
    skip: Callable[[chess.pgn.Headers], bool] | None = None,
    passing: int = 0,
    on_passing: Callable[[int], None] | None = None,
) -> Iterator[tuple[chess.pgn.Game, int]]:
    """Every game in ``piece``, with where the reading had got to, as a stream.

    The offset comes with the game because a caller reading a piece itself -- rather than in a
    worker, which reports one figure when it is done -- has no other way to say how far it has
    got, and :meth:`io.TextIOWrapper.tell` is not one: see below. It is an absolute offset into
    the file, and :attr:`_Window.position` says how exact.

    Streamed rather than read in one go, because a piece is only as small as the boundaries
    :func:`game_ranges` could find in the file: one whose games are not separated by a blank
    line has none, and is a single piece spanning all of it. Read whole, that is a read of the
    entire file into one worker -- tens of gigabytes for a dump, which ends with the worker
    killed for its memory and the build reporting something else entirely.

    Nothing here asks the file how far through it it is. :meth:`io.TextIOWrapper.tell` re-syncs
    the decoder to answer, where :class:`_Window` already knows -- and it knows part way through
    a game, which ``tell`` cannot, since the parser does not return until the game is finished.
    The piece's end is enforced underneath the decoder for the same reason.

    An :exc:`OSError` is left as one rather than dressed up, because a file that goes away half
    way through a build is the one failure whose *kind* the build has to act on: it means the
    dataset is missing an unknown number of games, which is not the same as a file whose
    contents stopped making sense. See
    :func:`~chess_ai.dataset.builder._check_worth_publishing`.
    """
    with piece.path.open("rb") as handle:
        text, window = reading(handle, piece.start, piece.end, on_passing)
        pass_games(text, passing)
        window.on_read = on_read
        while True:
            record = read_game(text, skip)
            if record is None:
                return
            yield record, window.position


def games_in_piece(
    piece: Piece,
    on_read: Callable[[int], None] | None = None,
    skip: Callable[[chess.pgn.Headers], bool] | None = None,
    passing: int = 0,
    on_passing: Callable[[int], None] | None = None,
) -> Iterator[tuple[chess.pgn.Game, int]]:
    """Every game in a piece of any kind, with where the reading had got to, as a stream.

    :func:`games_in_range` for a :class:`ByteRange`, and the same for the pieces of a compressed
    file: the offsets are into the compressed file, which is what its size is counted in. A
    :class:`TextPiece` is in memory and reports its ``end`` throughout; ``on_read`` only matters
    to a :class:`CompressedTail`, which is a long read in the build's own process.

    ``passing`` is how many games at the start of the piece to read past without yielding them,
    which is where a resumed build that stopped part way through the piece carries on; see
    :func:`pass_games`. They are behind the progress already, so ``on_read`` is not told of them;
    ``on_passing`` is, in its place, as the reading goes.
    """
    if isinstance(piece, ByteRange):
        yield from games_in_range(piece, on_read, skip, passing, on_passing)
    elif isinstance(piece, TextPiece):
        text = io.TextIOWrapper(io.BytesIO(piece.text), encoding=ENCODING, errors="replace")
        pass_games(text, passing)
        while True:
            record = read_game(text, skip)
            if record is None:
                break
            yield record, piece.end
        if piece.error is not None:
            raise piece.error
    else:
        yield from _games_in_tail(piece, on_read, skip, passing, on_passing)


def _games_in_tail(
    piece: CompressedTail,
    on_read: Callable[[int], None] | None,
    skip: Callable[[chess.pgn.Headers], bool] | None,
    passing: int = 0,
    on_passing: Callable[[int], None] | None = None,
) -> Iterator[tuple[chess.pgn.Game, int]]:
    """Every game in a compressed file from ``piece.start`` of its text to its end."""
    with piece.path.open("rb") as handle:
        window = _Window(handle, 0, None, on_passing)
        stream = _Decompressing(window)
        passed = 0
        while passed < piece.start:
            data = stream.read(min(DECOMPRESS_READ * 8, piece.start - passed))
            if not data:
                # The file is shorter than when it was cut: there is no rest of it to read.
                return
            passed += len(data)
        text = io.TextIOWrapper(io.BufferedReader(stream), encoding=ENCODING, errors="replace")
        pass_games(text, passing)
        # Only from here: what was passed over is behind the progress already, and reporting it
        # again would send the line back to the start of the file and then forward.
        window.on_read = on_read
        while True:
            record = read_game(text, skip)
            if record is None:
                return
            yield record, window.position


class PgnReader:
    """The games in one PGN file, one at a time, with the file's read position to hand.

    A game the parser could not read comes back all the same, carrying what went wrong in its
    ``errors``; deciding what to do about that belongs to the caller, which counts it.
    """

    def __init__(
        self,
        source: Source,
        on_read: Callable[[int], None] | None = None,
        skip: Callable[[chess.pgn.Headers], bool] | None = None,
    ) -> None:
        self.source = source
        self._on_read = on_read
        self._skip = skip
        self._handle: BinaryIO | None = None
        self._file: TextIO | None = None
        self._window: _Window | None = None

    def open(self) -> None:
        """Open the file, or raise :exc:`DatasetError` saying why it cannot be read.

        Separate from the ``with`` block so that a caller can guard the opening of a file without
        also guarding what it then does with the games: they fail for unrelated reasons and
        deserve unrelated answers.
        """
        try:
            self._handle = self.source.path.open("rb")
        except OSError as e:
            raise DatasetError(f"cannot read {self.source.path}: {e.strerror}") from e
        # Through a window, for the counting rather than for the bound: `None` reads to the
        # file's real end, as this has always done. It is what lets a read say it is alive while
        # the parser is inside a single call that swallows a whole file.
        self._file, self._window = reading(
            self._handle, 0, None, self._on_read, compressed=self.source.compressed
        )

    def close(self) -> None:
        """Let go of the file, whether or not it was read to the end."""
        if self._file is not None:
            self._file.close()  # which closes the buffer and the window under it
            self._file = None
        if self._handle is not None:
            # And not the file: the window reads through the handle without owning it, so that
            # games_in_range can hand it one a `with` block closes. This opened it, so this does.
            self._handle.close()
        self._handle = None
        self._window = None

    def __enter__(self) -> "PgnReader":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def bytes_read(self) -> int:
        """How far into the file the reading has got.

        The window's count rather than :meth:`io.TextIOWrapper.tell`, so that it is the same
        number the callback fires on and is available part way through a game. It runs ahead of
        the parser by what the buffers hold; see :attr:`_Window.position`.
        """
        return self._window.position if self._window is not None else 0

    def games(
        self, passing: int = 0, on_passing: Callable[[int], None] | None = None
    ) -> Iterator[chess.pgn.Game]:
        """Every record in the file, in order, until there are no more.

        ``passing`` is how many to read past first without yielding them, which is where a
        resumed build carries on from. They are behind the progress already, so ``on_passing``
        is told how far the reading has got while they are passed, in place of ``on_read``. See
        :func:`pass_games`.
        """
        assert self._file is not None, "read a PgnReader inside its with block"
        if passing:
            assert self._window is not None
            on_read, self._window.on_read = self._window.on_read, on_passing
            try:
                pass_games(self._file, passing)
            finally:
                self._window.on_read = on_read
        while True:
            record = read_game(self._file, self._skip)
            if record is None:
                return
            yield record


@contextmanager
def quiet_parser() -> Iterator[None]:
    """Stop python-chess logging every unreadable game while a build is running.

    It logs one line per bad move, and a dump of millions of games has thousands of them.
    The counts per reason in the manifest say the same thing in a form that fits on a screen.
    """
    logger = logging.getLogger("chess.pgn")
    was = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(was)
