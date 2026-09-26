import errno
import os
import sys
from pathlib import Path

import pytest
import uvicorn
from dataset_helpers import FIXTURES, GOOD_GAMES, fixture
from fastapi.testclient import TestClient

from chess_ai.cli import main
from chess_ai.dataset import load_manifest


@pytest.fixture
def served(monkeypatch):
    """Capture what ``chess-ai serve`` would run instead of starting a server."""
    calls = {}

    def fake_run(app, host, port):
        calls.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


def test_serve_uses_configured_address_and_prefix(tmp_path, served, capsys):
    config = tmp_path / "custom.toml"
    config.write_text('[server]\nhost = "0.0.0.0"\nport = 9000\npath_prefix = "/chess"\n')

    assert main(["--config", str(config), "serve"]) == 0

    assert (served["host"], served["port"]) == ("0.0.0.0", 9000)
    assert TestClient(served["app"]).get("/chess/api/start-position").status_code == 200
    assert "http://0.0.0.0:9000/chess/" in capsys.readouterr().out


def test_invalid_config_exits_with_message(tmp_path, served, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(tmp_path / "missing.toml"), "serve"])

    assert exit_info.value.code == 2
    assert "cannot read config file" in capsys.readouterr().err
    assert served == {}


def test_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2


def test_dataset_build_makes_a_dataset_in_the_data_directory(tmp_path, capsys):
    assert main(["dataset", "build", "games", fixture("lichess.pgn")]) == 0

    out = capsys.readouterr().out
    manifest = load_manifest(tmp_path / "data" / "datasets" / "games")
    assert manifest.name == "games"
    assert manifest.games == 4
    assert f"{manifest.positions:,} positions" in out
    assert "data/datasets/games" in out, "the build says where it put the dataset"


def test_dataset_build_uses_the_configured_data_directory(tmp_path, capsys):
    config = tmp_path / "custom.toml"
    config.write_text(f'[paths]\ndata = "{tmp_path / "elsewhere"}"\n')

    assert main(["--config", str(config), "dataset", "build", "games", str(FIXTURES)]) == 0

    assert load_manifest(tmp_path / "elsewhere" / "datasets" / "games").games == GOOD_GAMES


def test_dataset_build_takes_a_glob_the_shell_did_not_expand(tmp_path):
    assert main(["dataset", "build", "games", str(FIXTURES / "*rated*.pgn")]) == 0

    manifest = load_manifest(tmp_path / "data" / "datasets" / "games")
    assert [Path(source.path).name for source in manifest.sources] == ["unrated.pgn"]


def test_dataset_build_passes_on_the_options_it_is_given(tmp_path):
    assert (
        main(
            [
                "dataset",
                "build",
                "games",
                str(FIXTURES),
                "--validation-fraction",
                "0.5",
                "--rating-source",
                "lichess",
            ]
        )
        == 0
    )

    manifest = load_manifest(tmp_path / "data" / "datasets" / "games")
    assert manifest.validation_fraction == 0.5
    assert manifest.rating_source == "lichess"
    assert manifest.statistics.rating_sources == {"lichess": GOOD_GAMES}
    assert manifest.splits["validation"].games > 0


def test_dataset_build_reports_progress_while_it_works(tmp_path, capsys):
    assert main(["dataset", "build", "games", str(FIXTURES)]) == 0

    assert "games/s" in capsys.readouterr().err


def test_dataset_build_over_an_existing_dataset_needs_overwrite(tmp_path, capsys):
    assert main(["dataset", "build", "games", fixture("lichess.pgn")]) == 0

    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "build", "games", fixture("unrated.pgn")])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "already in" in error
    assert "--overwrite" in error, "the message names the way out"

    assert main(["dataset", "build", "games", fixture("unrated.pgn"), "--overwrite"]) == 0
    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 3


def test_dataset_build_refuses_a_source_that_matches_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "build", "games", str(tmp_path / "nothing.pgn")])

    assert exit_info.value.code == 2
    assert "no such PGN file" in capsys.readouterr().err


def test_dataset_stats_summarises_a_built_dataset(tmp_path, capsys):
    main(["dataset", "build", "games", str(FIXTURES)])
    capsys.readouterr()

    assert main(["dataset", "stats", "games"]) == 0

    out = capsys.readouterr().out
    assert "dataset games" in out
    assert "results" in out and "time controls" in out and "ratings" in out
    assert "lichess.pgn" in out


def test_dataset_stats_of_a_dataset_that_is_not_there_says_what_is(tmp_path, capsys):
    main(["dataset", "build", "games", fixture("lichess.pgn")])
    capsys.readouterr()

    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "stats", "elsewhere"])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "manifest.json is missing" in error
    assert "datasets in data: games" in error


def test_dataset_stats_with_no_datasets_at_all_says_how_to_make_one(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "stats", "games"])

    assert exit_info.value.code == 2
    assert "chess-ai dataset build" in capsys.readouterr().err


def test_dataset_needs_a_command(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["dataset"])

    assert exit_info.value.code == 2


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a file of any mode"
)
def test_dataset_build_says_so_when_a_source_could_not_be_read(tmp_path, capsys):
    # Built, and not the dataset that was asked for. A cron job reads the exit status.
    gone = tmp_path / "gone.pgn"
    gone.write_text('[Event "x"]\n[Site "s"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    gone.chmod(0o000)
    try:
        status = main(["dataset", "build", "games", fixture("lichess.pgn"), str(gone)])
    finally:
        gone.chmod(0o600)

    assert status == 1, "a partial build is not a success"
    error = capsys.readouterr().err
    assert "missing everything in 1 source(s)" in error
    assert str(gone) in error
    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a file of any mode"
)
def test_dataset_build_refuses_to_replace_a_dataset_with_less_than_it_has(tmp_path, capsys):
    assert main(["dataset", "build", "games", fixture("lichess.pgn")]) == 0
    gone = tmp_path / "gone.pgn"
    gone.write_text('[Event "x"]\n[Site "s"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    gone.chmod(0o000)
    capsys.readouterr()

    try:
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "dataset",
                    "build",
                    "games",
                    fixture("unrated.pgn"),
                    str(gone),
                    "--overwrite",
                ]
            )
    finally:
        gone.chmod(0o600)

    assert exit_info.value.code == 2
    assert "left alone" in capsys.readouterr().err
    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


def test_dataset_build_closes_its_progress_line_when_it_fails(tmp_path, capsys):
    # In a terminal the progress line is rewritten and left open, so a build that fails has to
    # close it or whatever is printed next lands on the end of it.
    import chess_ai.dataset
    from chess_ai.dataset import ProgressPrinter, builder

    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)
        patch.setattr(os, "fsync", out_of_space)
        patch.setattr(
            chess_ai.dataset,
            "ProgressPrinter",
            lambda *args, **kwargs: ProgressPrinter(sys.stderr, interval=0.0, rewrite=True),
        )

        with pytest.raises(OSError, match="No space left"):
            main(["dataset", "build", "games", fixture("lichess.pgn")])

    written = capsys.readouterr().err
    assert "\r" in written, "it really was a line being rewritten"
    assert written.endswith("\n"), "and it was closed"
    assert "built" not in written, "without saying the build finished"
