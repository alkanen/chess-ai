"""A dataset written out for a person to read, and the progress a build reports while running."""

import io

from dataset_helpers import GOOD_GAMES, build

from chess_ai.dataset import Progress, ProgressPrinter, summarize
from chess_ai.dataset.progress import format_duration, format_progress
from chess_ai.dataset.summary import format_bytes


def test_a_summary_says_what_the_dataset_holds(tmp_path):
    manifest = build(tmp_path, validation_fraction=0.25)

    summary = summarize(manifest)

    assert "dataset test" in summary
    assert f"games      {manifest.games:,}" in summary
    assert f"positions  {manifest.positions:,}" in summary
    assert f"train {manifest.splits['train'].games:,}" in summary
    assert f"validation {manifest.splits['validation'].games:,}" in summary
    assert "version 1, move vocabulary 1968" in summary
    assert "filters    none" in summary


def test_a_summary_draws_every_distribution(tmp_path):
    manifest = build(tmp_path)

    summary = summarize(manifest)

    for heading in ("results", "time controls", "rating sources", "ratings"):
        assert f"\n{heading}" in summary
    assert "1/2-1/2" in summary
    assert "correspondence" in summary
    assert "lichess" in summary
    assert "2400-2499" in summary
    assert "unknown" in summary
    assert "█" in summary


def test_a_summary_names_the_sources_and_what_was_skipped(tmp_path):
    manifest = build(tmp_path)

    summary = summarize(manifest)

    assert "lichess.pgn" in summary
    assert "4 games read, 4 kept" in summary
    assert f"skipped games  {manifest.games_skipped}" in summary
    assert "unsupported variant" in summary, "reasons read as words, not as field names"


def test_a_summary_of_a_dataset_with_no_ratings_at_all(tmp_path):
    manifest = build(tmp_path, "unrated.pgn", validation_fraction=0.0)

    summary = summarize(manifest)

    assert "unknown" in summary
    assert f"games      {GOOD_GAMES - 5:,}" in summary


def test_progress_works_out_throughput_and_what_is_left():
    halfway = Progress(
        games_read=1000,
        games_kept=990,
        positions=80_000,
        bytes_read=500,
        bytes_total=2000,
        seconds=10.0,
        done=False,
    )

    assert halfway.games_per_second == 100.0
    assert halfway.fraction == 0.25
    assert halfway.seconds_remaining == 30.0

    line = format_progress(halfway)

    assert "25%" in line
    assert "990 games" in line
    assert "80,000 positions" in line
    assert "100 games/s" in line
    assert "10 skipped" in line
    assert "30s left" in line


def test_progress_says_nothing_about_time_left_when_it_cannot_tell():
    nothing_read = Progress(
        games_read=0,
        games_kept=0,
        positions=0,
        bytes_read=0,
        bytes_total=2000,
        seconds=0.0,
        done=False,
    )

    assert nothing_read.seconds_remaining is None
    assert nothing_read.games_per_second == 0.0
    assert "left" not in format_progress(nothing_read)


def test_the_final_report_is_a_summary_rather_than_an_estimate():
    done = Progress(
        games_read=10,
        games_kept=10,
        positions=800,
        bytes_read=2000,
        bytes_total=2000,
        seconds=90.0,
        done=True,
    )

    assert done.seconds_remaining is None
    assert format_progress(done) == "built 10 games, 800 positions, 0 games/s in 1m 30s"


def test_progress_is_printed_no_more_often_than_asked_for():
    out = io.StringIO()
    printer = ProgressPrinter(out, interval=1.0, rewrite=False)

    for second in (0.0, 0.2, 0.5, 1.5, 1.6):
        printer(_at(second))
    printer(_at(1.7, done=True))

    assert len(out.getvalue().splitlines()) == 3


def test_a_terminal_gets_one_line_that_rewrites_itself():
    out = io.StringIO()
    printer = ProgressPrinter(out, interval=0.0, rewrite=True)

    printer(_at(1.0))
    printer(_at(2.0, done=True))

    written = out.getvalue()
    assert written.count("\r") == 2
    assert written.endswith("\n")
    assert written.count("\n") == 1


def _at(seconds: float, done: bool = False) -> Progress:
    return Progress(
        games_read=1,
        games_kept=1,
        positions=1,
        bytes_read=1,
        bytes_total=10,
        seconds=seconds,
        done=done,
    )


def test_a_duration_reads_at_a_glance():
    assert format_duration(0) == "0s"
    assert format_duration(45.7) == "45s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3600) == "1h 00m"
    assert format_duration(7 * 3600 + 25 * 60) == "7h 25m"


def test_a_size_reads_at_a_glance():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024**3) == "5.0 GB"
