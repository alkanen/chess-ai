"""Downloading Lichess dumps, against a server in the test that plays Lichess's part."""

import hashlib
import io
import os
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chess_ai.cli import main
from chess_ai.dataset import download
from chess_ai.dataset.download import (
    DownloadError,
    DownloadPrinter,
    download_months,
    dump_name,
    parse_month,
)

JANUARY = dump_name("2017-01")
FEBRUARY = dump_name("2017-02")


@dataclass
class Lichess:
    """What the server publishes, how it behaves, and what it was asked."""

    files: dict[str, bytes] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    honour_ranges: bool = True
    cut_after: int | None = None
    """Bytes of a file sent before the connection is dropped, once, to stand for a failed one."""
    cut_checksums: bool = False
    """Whether the checksums are sent short of the length the response says they have."""
    garbled: bool = False
    """Whether a dump is answered with something that is not HTTP."""
    requests: list[tuple[str, str | None]] = field(default_factory=list)
    """Every request's path and ``Range`` header, in order."""
    url: str = ""

    def publish(self, name: str, data: bytes, checksum: str | None = None) -> None:
        self.files[name] = data
        self.checksums[name] = checksum or hashlib.sha256(data).hexdigest()

    def asked_for(self, name: str) -> list[str | None]:
        """The ``Range`` headers of the requests for ``name``."""
        return [range_ for path, range_ in self.requests if path == f"/standard/{name}"]


@pytest.fixture
def lichess() -> Iterator[Lichess]:
    state = Lichess()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            state.requests.append((self.path, self.headers.get("Range")))
            name = self.path.removeprefix("/standard/")
            if name == "sha256sums.txt":
                body = "".join(f"{sha}  {file}\n" for file, sha in state.checksums.items())
                if state.cut_checksums:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body) + 1000))
                    self.end_headers()
                    self.wfile.write(body.encode())
                    return
                self._send(200, body.encode())
                return
            if state.garbled:
                self.wfile.write(b"this is not HTTP at all\r\n\r\n")
                return
            if name not in state.files:
                self._send(404, b"not found")
                return
            data = state.files[name]
            start = 0
            asked = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range") or "")
            if asked and state.honour_ranges:
                start = int(asked[1])
                if start >= len(data):
                    self._send(416, b"")
                    return
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            body = data[start:]
            if state.cut_after is not None:
                body = body[: state.cut_after]
                state.cut_after = None
            self.wfile.write(body)

        def _send(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}/standard/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def dump(size: int, seed: int = 0) -> bytes:
    """``size`` bytes that are not all the same, so that a misplaced piece changes the hash."""
    return hashlib.shake_256(str(seed).encode()).digest(size)


def fetch(lichess: Lichess, directory: Path, *months: str, **options) -> list[Path]:
    return download_months(list(months), directory=directory, base_url=lichess.url, **options)


def test_a_month_is_downloaded_under_its_own_name_once_it_is_checked(lichess, tmp_path):
    lichess.publish(JANUARY, data := dump(3 << 20))

    (path,) = fetch(lichess, tmp_path / "dumps", "2017-01")

    assert path == tmp_path / "dumps" / JANUARY
    assert path.read_bytes() == data
    assert not path.with_name(JANUARY + ".part").exists()


def test_several_months_come_down_in_the_order_asked_for(lichess, tmp_path):
    lichess.publish(JANUARY, january := dump(1000, 1))
    lichess.publish(FEBRUARY, february := dump(1000, 2))

    paths = fetch(lichess, tmp_path, "2017-02", "2017-01", "2017-02")

    assert [path.name for path in paths] == [FEBRUARY, JANUARY], "and each of them once"
    assert paths[0].read_bytes() == february and paths[1].read_bytes() == january


def test_a_partial_download_is_carried_on_from_where_it_stopped(lichess, tmp_path):
    lichess.publish(JANUARY, data := dump(3 << 20))
    (tmp_path / (JANUARY + ".part")).write_bytes(data[:1_000_000])

    (path,) = fetch(lichess, tmp_path, "2017-01")

    assert path.read_bytes() == data
    assert lichess.asked_for(JANUARY) == ["bytes=1000000-"], "only the rest was asked for"


def test_a_download_whose_connection_drops_is_kept_and_carried_on_next_time(lichess, tmp_path):
    lichess.publish(JANUARY, data := dump(3 << 20))
    lichess.cut_after = 1 << 20

    with pytest.raises(DownloadError, match="carries on from there"):
        fetch(lichess, tmp_path, "2017-01")
    part = tmp_path / (JANUARY + ".part")
    assert part.stat().st_size == 1 << 20
    assert not (tmp_path / JANUARY).exists()

    (path,) = fetch(lichess, tmp_path, "2017-01")

    assert path.read_bytes() == data
    assert lichess.asked_for(JANUARY) == [None, f"bytes={1 << 20}-"]


def test_a_server_that_sends_the_whole_file_again_starts_the_download_again(lichess, tmp_path):
    # Appending a whole file to the start of itself would only be caught by the checksum, an
    # hour later; the server saying it sent everything is reason enough to start again.
    lichess.publish(JANUARY, data := dump(1 << 20))
    lichess.honour_ranges = False
    (tmp_path / (JANUARY + ".part")).write_bytes(data[:5000])

    (path,) = fetch(lichess, tmp_path, "2017-01")

    assert path.read_bytes() == data


def test_a_download_that_was_whole_when_it_stopped_is_only_checked(lichess, tmp_path):
    lichess.publish(JANUARY, data := dump(1 << 20))
    (tmp_path / (JANUARY + ".part")).write_bytes(data)

    (path,) = fetch(lichess, tmp_path, "2017-01")

    assert path.read_bytes() == data
    assert lichess.asked_for(JANUARY) == [f"bytes={len(data)}-"]


def test_a_download_that_does_not_match_its_checksum_is_thrown_away(lichess, tmp_path):
    lichess.publish(JANUARY, dump(1 << 20), checksum="0" * 64)

    with pytest.raises(DownloadError, match="does not match the checksum"):
        fetch(lichess, tmp_path, "2017-01")

    assert list(tmp_path.iterdir()) == [tmp_path / f".{JANUARY}.lock"], (
        "nothing under the dump's name, and no part to carry on from: it is all suspect"
    )


def test_a_resumed_download_is_checked_as_a_whole(lichess, tmp_path):
    # A part file whose start is not the file's -- left over from a different month's dump, or
    # damaged on the disk -- carries on into a file that is wrong only at the start.
    lichess.publish(JANUARY, data := dump(1 << 20))
    (tmp_path / (JANUARY + ".part")).write_bytes(b"x" * 1000)

    with pytest.raises(DownloadError, match="does not match the checksum"):
        fetch(lichess, tmp_path, "2017-01")

    (path,) = fetch(lichess, tmp_path, "2017-01")
    assert path.read_bytes() == data


def test_a_month_already_downloaded_is_not_downloaded_again(lichess, tmp_path):
    lichess.publish(JANUARY, dump(1000))
    (tmp_path / JANUARY).write_bytes(b"what is there")

    (path,) = fetch(lichess, tmp_path, "2017-01")

    assert path.read_bytes() == b"what is there"
    assert lichess.asked_for(JANUARY) == []


def test_a_month_that_is_not_published_is_refused_before_anything_is_downloaded(lichess, tmp_path):
    lichess.publish(JANUARY, dump(1000))

    with pytest.raises(DownloadError, match=f"has not published {FEBRUARY}"):
        fetch(lichess, tmp_path, "2017-01", "2017-02")

    assert lichess.asked_for(JANUARY) == []


def test_checksums_cut_off_part_way_say_so_rather_than_raise(lichess, tmp_path):
    lichess.publish(JANUARY, dump(1000))
    lichess.cut_checksums = True

    with pytest.raises(DownloadError, match="cannot fetch the published checksums"):
        fetch(lichess, tmp_path, "2017-01")


def test_a_reply_that_is_not_http_says_so_rather_than_raise(lichess, tmp_path):
    lichess.publish(JANUARY, dump(1000))
    lichess.garbled = True

    with pytest.raises(DownloadError, match=f"cannot download .*{JANUARY}"):
        fetch(lichess, tmp_path, "2017-01")


def test_a_server_that_cannot_be_reached_says_so(tmp_path):
    with pytest.raises(DownloadError, match="cannot fetch the published checksums"):
        download_months(["2017-01"], directory=tmp_path, base_url="http://127.0.0.1:9/standard/")


def test_a_month_being_downloaded_elsewhere_is_not_downloaded_twice(lichess, tmp_path):
    fcntl = pytest.importorskip("fcntl")
    lichess.publish(JANUARY, dump(1000))
    lock = os.open(tmp_path / f".{JANUARY}.lock", os.O_CREAT | os.O_RDONLY, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(DownloadError, match="already being downloaded"):
            fetch(lichess, tmp_path, "2017-01")
    finally:
        os.close(lock)
    assert lichess.asked_for(JANUARY) == []


def test_progress_is_reported_from_what_was_already_there(lichess, tmp_path):
    lichess.publish(JANUARY, data := dump(3 << 20))
    (tmp_path / (JANUARY + ".part")).write_bytes(data[:1000])
    reports: list[tuple[str, int, int | None]] = []

    fetch(lichess, tmp_path, "2017-01", progress=lambda *report: reports.append(report))

    assert reports[0] == (JANUARY, 1000, len(data))
    assert reports[-1] == (JANUARY, len(data), len(data))


def test_the_progress_line_names_the_file_and_how_far_it_has_got():
    stream = io.StringIO()
    printer = DownloadPrinter(stream, interval=0)

    printer(JANUARY, 0, 2_000_000_000)
    printer(JANUARY, 2_000_000_000, 2_000_000_000)

    lines = stream.getvalue().splitlines()
    assert lines[0].startswith(f"{JANUARY}, 0% of 2.0 GB")
    assert lines[-1].startswith(f"{JANUARY}, 100% of 2.0 GB")


@pytest.mark.parametrize("text", ["2017-1", "2017-13", "17-01", "2017-01-01", "January"])
def test_a_month_is_written_year_dash_month(text):
    with pytest.raises(ValueError, match="YYYY-MM"):
        parse_month(text)


def test_the_command_downloads_into_the_data_directory(lichess, tmp_path, capsys, monkeypatch):
    lichess.publish(JANUARY, data := dump(1000))
    monkeypatch.setattr(download, "LICHESS_URL", lichess.url)

    assert main(["dataset", "download", "2017-01"]) == 0

    assert (tmp_path / "data" / "lichess" / JANUARY).read_bytes() == data
    assert f"data/lichess/{JANUARY}" in capsys.readouterr().out


def test_the_command_refuses_a_month_it_cannot_read(capsys):
    with pytest.raises(SystemExit) as exit:
        main(["dataset", "download", "2017-13"])

    assert exit.value.code == 2
    assert "YYYY-MM" in capsys.readouterr().err


def test_the_command_says_why_a_download_failed(lichess, capsys, monkeypatch):
    monkeypatch.setattr(download, "LICHESS_URL", lichess.url)
    lichess.publish(JANUARY, dump(1000))

    with pytest.raises(SystemExit) as exit:
        main(["dataset", "download", "2017-02"])

    assert exit.value.code == 2
    assert "has not published" in capsys.readouterr().err
