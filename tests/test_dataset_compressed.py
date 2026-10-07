"""Building datasets out of zstd-compressed PGN, which is how Lichess publishes its dumps."""

import hashlib
import logging
from pathlib import Path

import pytest
import zstandard
from dataset_helpers import (
    FIXTURES,
    SEPARATED,
    CountingPath,
    build,
    fixture,
    flat_pgn,
    separated_pgn,
    shard_bytes,
)

from chess_ai.dataset import DatasetError, Progress, builder, dataset_path
from chess_ai.dataset.manifest import Filters
from chess_ai.dataset.sources import (
    CompressedTail,
    PgnReader,
    TextPiece,
    games_in_piece,
    resolve_sources,
    text_pieces,
)


def compressed(path: Path, *, frames: int = 1) -> Path:
    """``path`` compressed beside itself as ``.zst``, in ``frames`` frames one after another."""
    text = path.read_bytes()
    out = path.with_name(path.name + ".zst")
    step = -(-len(text) // frames)
    out.write_bytes(
        b"".join(
            zstandard.ZstdCompressor().compress(text[at : at + step])
            for at in range(0, len(text), step)
        )
    )
    return out


def varied_pgn(path: Path, games: int) -> Path:
    """``games`` games that do not compress to nearly nothing, as real ones do not.

    A file of one game repeated compresses into a few large blocks, and text only comes out of a
    block once all of it is in -- so damage anywhere loses everything, and a build's progress
    has nowhere to be between the start and the end.
    """
    path.write_text(
        "".join(
            SEPARATED.replace(
                '[Event "x"]', f'[Event "{hashlib.sha256(str(game).encode()).hexdigest()}"]'
            )
            for game in range(games)
        )
    )
    return path


def mixed_pgn(path: Path) -> Path:
    """Every fixture file one after another, broken games and all, several times over."""
    path.write_bytes(
        b"\n".join(
            (FIXTURES / name).read_bytes()
            for name in ["lichess.pgn", "malformed.pgn", "unrated.pgn", "custom-start.pgn"] * 5
        )
    )
    return path


def parallel(patch, chunk: int = 300, most: int | None = None) -> None:
    """Make a build of the small files here read in several processes, in small pieces."""
    patch.setattr(builder, "PARALLEL_FROM_BYTES", 0)
    patch.setattr(builder, "CHUNK_BYTES", chunk)
    if most is not None:
        patch.setattr(builder, "MAX_PIECE_BYTES", most)


def same_manifest(one, other) -> None:
    """Two manifests that say the same thing, apart from when, and which file it was."""
    exclude = {"created": True, "sources": {"__all__": {"path", "bytes"}}}
    assert one.model_dump(exclude=exclude) == other.model_dump(exclude=exclude)


def test_a_compressed_file_builds_the_same_dataset_as_the_file_it_was_made_from(tmp_path):
    plain = mixed_pgn(tmp_path / "games.pgn")
    packed = compressed(plain)

    expected = build(tmp_path / "plain", str(plain), workers=1)
    actual = build(tmp_path / "packed", str(packed), workers=1)

    assert expected.games > 0 and expected.games_skipped > 0, "kept and broken games alike"
    assert shard_bytes(dataset_path(tmp_path / "packed", "test")) == shard_bytes(
        dataset_path(tmp_path / "plain", "test")
    )
    same_manifest(actual, expected)
    assert actual.sources[0].bytes == packed.stat().st_size, "sized as the file on the disk"


@pytest.mark.parametrize("chunk", [1, 300, 5000])
def test_reading_a_compressed_file_in_several_processes_gives_the_same_dataset(tmp_path, chunk):
    plain = mixed_pgn(tmp_path / "games.pgn")
    packed = compressed(plain)

    expected = build(tmp_path / "plain", str(plain), workers=1)
    with pytest.MonkeyPatch.context() as patch:
        parallel(patch, chunk)
        actual = build(tmp_path / "packed", str(packed), workers=3)

    assert shard_bytes(dataset_path(tmp_path / "packed", "test")) == shard_bytes(
        dataset_path(tmp_path / "plain", "test")
    )
    same_manifest(actual, expected)


def test_a_file_of_several_frames_is_read_through_all_of_them(tmp_path):
    # What zstd writes for input it was given in parts, and what concatenating two .zst files
    # makes. Stopping at the end of the first frame would look like a shorter file.
    plain = mixed_pgn(tmp_path / "games.pgn")
    packed = compressed(plain, frames=4)

    expected = build(tmp_path / "plain", str(plain), workers=1)
    serial = build(tmp_path / "one", str(packed), workers=1)
    with pytest.MonkeyPatch.context() as patch:
        parallel(patch)
        many = build(tmp_path / "many", str(packed), workers=3)

    same_manifest(serial, expected)
    same_manifest(many, expected)


def test_lichess_games_in_a_compressed_dump_are_rated_by_lichess(tmp_path):
    copy = tmp_path / "lichess.pgn"
    copy.write_bytes(Path(fixture("lichess.pgn")).read_bytes())

    manifest = build(tmp_path / "data", str(compressed(copy)), workers=1)

    assert manifest.games == 4
    assert manifest.statistics.rating_sources == {"lichess": 4}


def test_a_directory_or_a_glob_picks_up_compressed_files_with_the_plain_ones(tmp_path):
    separated_pgn(tmp_path / "a.pgn", games=1)
    compressed(separated_pgn(tmp_path / "b.pgn", games=1))
    (tmp_path / "b.pgn").unlink()
    (tmp_path / "c.zst").write_bytes(b"not a PGN file by its name")

    by_directory = resolve_sources([str(tmp_path)])
    by_glob = resolve_sources([str(tmp_path / "*")])

    assert [source.path.name for source in by_directory] == ["a.pgn", "b.pgn.zst"]
    assert by_glob == by_directory
    assert [source.compressed for source in by_directory] == [False, True]


@pytest.mark.parametrize("named", ["directory", "glob", "files"])
def test_a_dump_beside_its_decompressed_copy_is_refused_rather_than_read_twice(tmp_path, named):
    # `zstd -d` keeps the file it decompressed, so a dump looked at where it was downloaded
    # leaves both behind -- and a build of the directory would train on that month twice.
    plain = separated_pgn(tmp_path / "2017-01.pgn", games=1)
    packed = compressed(plain)
    patterns = {
        "directory": [str(tmp_path)],
        "glob": [str(tmp_path / "2017-01*")],
        "files": [str(packed), str(plain)],
    }[named]

    with pytest.raises(DatasetError, match="read their games twice") as refused:
        resolve_sources(patterns)

    assert f"{packed} and {plain}" in str(refused.value), "naming both copies"


@pytest.mark.parametrize("linked", ["dump", "copy"])
def test_a_symlinked_dump_beside_its_decompressed_copy_is_refused_too(tmp_path, linked):
    # Dumps are big, so the one in the data directory may well be a link to another disk. The
    # copy that matters is the one beside the name the build was given, not beside the target.
    elsewhere, here = tmp_path / "elsewhere", tmp_path / "here"
    elsewhere.mkdir()
    here.mkdir()
    plain = separated_pgn(elsewhere / "2017-01.pgn", games=1)
    packed = compressed(plain)
    if linked == "dump":
        (here / packed.name).symlink_to(packed)
        plain.rename(here / plain.name)
    else:
        (here / plain.name).symlink_to(plain)
        packed.rename(here / packed.name)

    with pytest.raises(DatasetError, match="read their games twice"):
        resolve_sources([str(here)])


@pytest.mark.parametrize("target_first", [True, False])
def test_a_copy_beside_a_links_target_is_caught_in_either_order(tmp_path, target_first):
    # Naming the data directory and the disk its links point to, with the dump decompressed on
    # that disk: the dump on the disk is dropped as the link's duplicate when the link comes
    # first, so the link is all that is left to find the copy from, beside its target.
    big, links = tmp_path / "big", tmp_path / "lichess"
    big.mkdir()
    links.mkdir()
    packed = compressed(separated_pgn(big / "2017-01.pgn", games=1))
    (links / packed.name).symlink_to(packed)
    patterns = [str(big), str(links)] if target_first else [str(links), str(big)]

    with pytest.raises(DatasetError, match="read their games twice"):
        resolve_sources(patterns)


def test_a_dump_on_its_own_or_with_other_months_is_not_a_copy(tmp_path):
    compressed(separated_pgn(tmp_path / "2017-01.pgn", games=1)).with_name("2017-01.pgn").unlink()
    separated_pgn(tmp_path / "2017-02.pgn", games=1)

    names = [source.path.name for source in resolve_sources([str(tmp_path)])]

    assert names == ["2017-01.pgn.zst", "2017-02.pgn"]


@pytest.mark.parametrize("workers", [1, 3])
def test_a_truncated_compressed_file_says_so_rather_than_ending_early(tmp_path, workers):
    # zstandard's own stream reader ends quietly where the file does, so a download that stopped
    # part way would have read as a smaller month. It is the contents that stopped making sense,
    # not the file that went away: those games do not exist to be missed.
    packed = compressed(varied_pgn(tmp_path / "games.pgn", games=20000))
    data = packed.read_bytes()
    packed.write_bytes(data[: len(data) * 2 // 3])

    with pytest.MonkeyPatch.context() as patch:
        parallel(patch, chunk=2000)
        manifest = build(tmp_path / "data", str(packed), workers=workers, validation_fraction=0)

    (source,) = manifest.sources
    assert 0 < manifest.games < 20000, "the games before the cut are kept"
    assert source.error is not None and "ends part way through" in source.error
    assert not source.went_away
    assert manifest.skipped == {"unreadable": 1}, "the game it stopped in"


@pytest.mark.parametrize("workers", [1, 3])
def test_a_corrupt_compressed_file_keeps_the_games_before_the_damage(tmp_path, workers):
    packed = compressed(varied_pgn(tmp_path / "games.pgn", games=20000))
    data = bytearray(packed.read_bytes())
    for at in range(len(data) // 2, len(data) // 2 + 64):
        data[at] ^= 0x5A
    packed.write_bytes(bytes(data))

    with pytest.MonkeyPatch.context() as patch:
        parallel(patch, chunk=2000)
        manifest = build(tmp_path / "data", str(packed), workers=workers, validation_fraction=0)

    (source,) = manifest.sources
    assert source.error is not None and "ZstdError" in source.error
    assert not source.went_away
    assert 0 < manifest.games < 20000


def test_a_compressed_file_that_cannot_be_opened_is_counted_as_gone(tmp_path):
    packed = compressed(separated_pgn(tmp_path / "games.pgn", games=10))
    readable = compressed(separated_pgn(tmp_path / "other.pgn", games=10))
    packed.chmod(0o000)
    try:
        with pytest.MonkeyPatch.context() as patch:
            parallel(patch)
            manifest = build(
                tmp_path / "data", str(packed), str(readable), workers=2, validation_fraction=0
            )
    finally:
        packed.chmod(0o600)

    gone, other = manifest.sources
    assert gone.went_away and gone.games_read == 0
    assert gone.error is not None and "cannot read" in gone.error
    assert other.games_kept == 10


def test_a_build_that_is_full_stops_decompressing(tmp_path, monkeypatch):
    # A build of the first thousand games of a 90-million-game month must not decompress the
    # month to find out where it ends.
    packed = compressed(varied_pgn(tmp_path / "games.pgn", games=20000))
    counter = CountingPath(packed)
    counter.install(monkeypatch)
    parallel(monkeypatch, chunk=2000)

    manifest = build(tmp_path / "data", str(packed), workers=2, filters=Filters(max_games=50))

    assert manifest.games == 50 and manifest.reached_max_games
    assert counter.bytes_read < packed.stat().st_size // 4, (
        f"read {counter.bytes_read:,} of {packed.stat().st_size:,} bytes for 50 games"
    )


def test_progress_is_counted_in_compressed_bytes(tmp_path):
    packed = compressed(varied_pgn(tmp_path / "games.pgn", games=20000))
    size = packed.stat().st_size
    reports: list[Progress] = []

    with pytest.MonkeyPatch.context() as patch:
        parallel(patch, chunk=4096)
        build(tmp_path / "data", str(packed), workers=2, progress=reports.append)

    during = [report.bytes_read for report in reports if not report.done]
    assert all(report.bytes_total == size for report in reports)
    assert any(0 < read < size for read in during), "it moves through the file"
    assert during == sorted(during), "and never goes backwards"
    assert reports[-1].bytes_read == size


def test_a_compressed_file_with_no_boundaries_is_read_here_from_where_cutting_stopped(
    tmp_path, caplog
):
    # Cutting a compressed file holds its text up to the next boundary, so a file with none
    # would be held whole. Past the largest piece, the rest of it is read in this process.
    plain = separated_pgn(tmp_path / "games.pgn", games=50)
    with plain.open("a") as out:
        out.write(flat_pgn(tmp_path / "flat.pgn", games=400).read_text())
    packed = compressed(plain)

    expected = build(tmp_path / "plain", str(plain), workers=1)
    with pytest.MonkeyPatch.context() as patch:
        parallel(patch, chunk=1024, most=4096)
        actual = build(tmp_path / "packed", str(packed), workers=3)

    assert shard_bytes(dataset_path(tmp_path / "packed", "test")) == shard_bytes(
        dataset_path(tmp_path / "plain", "test")
    )
    same_manifest(actual, expected)
    warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1 and "the rest of the file" in warnings[0]


def test_the_pieces_of_a_compressed_file_are_its_text_cut_between_games(tmp_path):
    plain = separated_pgn(tmp_path / "games.pgn", games=200)
    source = resolve_sources([str(compressed(plain))])[0]

    pieces = list(text_pieces(source, 3, 1000, 1 << 20))

    assert len(pieces) > 10
    assert b"".join(piece.text for piece in pieces) == plain.read_bytes()
    assert all(piece.text.startswith(b"[Event ") for piece in pieces)
    assert all(piece.source == 3 and piece.error is None for piece in pieces)
    assert all(len(piece.text) < 1000 + len(SEPARATED) for piece in pieces)
    ends = [piece.end for piece in pieces]
    assert ends == sorted(ends) and ends[-1] == source.bytes


def test_cutting_a_compressed_file_holds_no_more_than_the_largest_piece(tmp_path):
    plain = flat_pgn(tmp_path / "flat.pgn", games=2000)
    source = resolve_sources([str(compressed(plain))])[0]

    pieces = list(text_pieces(source, 0, 1000, 8192))

    (tail,) = pieces
    assert isinstance(tail, CompressedTail)
    assert tail.start == 0 and tail.end == source.bytes
    assert tail.bytes <= 8192 + (1 << 20), "the text in hand when it gave up, plus one read"
    games = [game for game, _ in games_in_piece(tail)]
    assert len(games) == 2000


def test_a_worker_does_not_send_the_text_it_read_back(tmp_path):
    # It has been read, and the parent has no use for it: sending it back would double what
    # crosses between the processes for every piece of every compressed file.
    plain = separated_pgn(tmp_path / "games.pgn", games=20)
    source = resolve_sources([str(compressed(plain))])[0]
    (piece,) = text_pieces(source, 0, 1 << 20, 1 << 21)

    read = builder._read_piece(
        builder._Job(piece=piece, rating_source=None, validation_fraction=0.0)
    )

    assert isinstance(read.piece, TextPiece) and read.piece.text == b""
    assert read.piece.end == piece.end and len(read.outcomes) == 20


def test_reading_a_compressed_file_counts_its_compressed_bytes(tmp_path):
    packed = compressed(separated_pgn(tmp_path / "games.pgn", games=500))
    source = resolve_sources([str(packed)])[0]

    with PgnReader(source) as reader:
        games = sum(1 for _ in reader.games())
        read = reader.bytes_read

    assert games == 500
    assert read == source.bytes


def test_the_serial_path_reads_a_compressed_file_with_its_ratings(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    packed = compressed(separated_pgn(tmp_path / "games.pgn", games=30))

    manifest = build(tmp_path / "data", str(packed), workers=1, validation_fraction=0)

    assert manifest.games == 30
    assert manifest.statistics.rating_sources == {"lichess": 30}
    assert not caplog.records
