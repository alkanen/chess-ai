import errno
import io
import os
import subprocess
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


def test_dataset_build_passes_on_how_many_workers_to_read_with(tmp_path):
    import chess_ai.dataset as dataset_package

    asked: list[int | None] = []
    real = dataset_package.build_dataset

    def record(*args, **kwargs):
        asked.append(kwargs.get("workers"))
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dataset_package, "build_dataset", record)
        assert main(["dataset", "build", "one", str(FIXTURES), "--workers", "3"]) == 0
        assert main(["dataset", "build", "two", str(FIXTURES)]) == 0

    assert asked == [3, None], "what was asked for, and None for one per CPU"


def test_dataset_build_refuses_fewer_than_one_worker(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "build", "games", str(FIXTURES), "--workers", "0"])

    assert exit_info.value.code == 2
    assert "at least one worker" in capsys.readouterr().err


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
    assert "missing games from 1 source(s)" in error
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


def test_dataset_build_does_not_call_an_empty_dataset_a_success(tmp_path, capsys):
    # The nightly over a directory of dumps that turn out to be gzip, or an export that landed as
    # HTML: every source read, no chess in any of it, and no dataset of this name to lose. An
    # exit status of 0 there is a build the next stage opens and trains on.
    empty = tmp_path / "empty.pgn"
    empty.write_text("")

    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "build", "games", str(empty)])

    assert exit_info.value.code == 2
    assert "kept no games" in capsys.readouterr().err
    assert not (tmp_path / "data" / "datasets" / "games").exists()


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

        with pytest.raises(SystemExit):
            main(["dataset", "build", "games", fixture("lichess.pgn")])

    written = capsys.readouterr().err
    assert "\r" in written, "it really was a line being rewritten"
    assert "built" not in written, "without saying the build finished"
    # Closed before the error, rather than the error landing on the end of it.
    progress, _, reported = written.partition("chess-ai: error:")
    assert progress.endswith("\n"), written
    assert reported


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a file of any mode"
)
def test_dataset_build_does_not_call_a_file_it_never_opened_partly_read(tmp_path, capsys):
    # One line about one file, and the right one: "not read whole" is for a source that was.
    gone = tmp_path / "gone.pgn"
    gone.write_text('[Event "x"]\n[Site "s"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    gone.chmod(0o000)
    try:
        main(["dataset", "build", "games", fixture("lichess.pgn"), str(gone)])
    finally:
        gone.chmod(0o600)

    written = capsys.readouterr()
    assert "not read whole" not in written.out
    assert "missing games from 1 source(s)" in written.err


def test_dataset_build_says_a_disk_that_filled_up_plainly(tmp_path, capsys):
    # The most ordinary way a long build dies, and every other failure in this command says
    # "chess-ai: error: ..." rather than showing a traceback.
    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", out_of_space)

        with pytest.raises(SystemExit) as exit_info:
            main(["dataset", "build", "games", fixture("lichess.pgn")])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "chess-ai: error:" in error
    assert "No space left on device" in error
    assert "Traceback" not in error


class DeadStream(io.StringIO):
    """A stream that goes away after its first write, as a tty does when its terminal closes.

    After the first one, so that a line has been written and is waiting to be closed: a stream
    that dies on its very first write leaves nothing open and nothing to trip over.
    """

    def __init__(self) -> None:
        super().__init__()
        self.writes = 0

    def write(self, text: str) -> int:
        self.writes += 1
        if self.writes > 1:
            raise OSError(errno.EIO, "Input/output error")
        return super().write(text)


def test_dataset_build_survives_its_terminal_going_away(tmp_path, capsys):
    # A nohuped build whose tty closes: the reporting stops, and the build it was reporting on
    # is finished, published and reported as a success.
    import chess_ai.dataset
    from chess_ai.dataset import ProgressPrinter, builder

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "REPORT_EVERY", 1)
        patch.setattr(
            chess_ai.dataset,
            "ProgressPrinter",
            lambda *args, **kwargs: ProgressPrinter(DeadStream(), interval=0.0, rewrite=True),
        )

        assert main(["dataset", "build", "games", fixture("lichess.pgn")]) == 0

    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


def test_dataset_build_reports_why_it_failed_even_with_its_terminal_gone(tmp_path, capsys):
    # The failure path of the same thing: closing the progress line must not take the place of
    # the message saying what went wrong.
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
            lambda *args, **kwargs: ProgressPrinter(DeadStream(), interval=0.0, rewrite=True),
        )

        with pytest.raises(SystemExit) as exit_info:
            main(["dataset", "build", "games", fixture("lichess.pgn")])

    assert exit_info.value.code == 2
    assert "No space left on device" in capsys.readouterr().err


def test_dataset_build_does_not_call_a_lost_source_a_success(tmp_path, capsys):
    # A build that lost most of a dump to a mount that dropped is not a success a cron job
    # should read past, even though it built something.
    import chess.pgn

    real_read_game = chess.pgn.read_game
    calls = {"n": 0}

    def goes_away(handle, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_read_game(handle, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chess.pgn, "read_game", goes_away)

        status = main(["dataset", "build", "games", fixture("unrated.pgn")])

    assert status == 1
    written = capsys.readouterr()
    assert "missing games from 1 source(s)" in written.err
    assert "not read whole" not in written.out, "that is for a file that was there throughout"


def dead_everywhere(patch, stream):
    """Point the command's output and its progress reporting at one stream that will go away."""
    import chess_ai.dataset
    from chess_ai.dataset import ProgressPrinter, builder

    patch.setattr(builder, "REPORT_EVERY", 1)
    patch.setattr(sys, "stdout", stream)
    patch.setattr(sys, "stderr", stream)
    patch.setattr(
        chess_ai.dataset,
        "ProgressPrinter",
        lambda *args, **kwargs: ProgressPrinter(stream, interval=0.0, rewrite=True),
    )


def test_dataset_build_survives_losing_every_stream_at_once(tmp_path):
    # What a closed terminal or a dropped ssh session really does: the progress line, the summary
    # and the error all go to the same place, and it goes away all at once.
    with pytest.MonkeyPatch.context() as patch:
        dead_everywhere(patch, DeadStream())

        assert main(["dataset", "build", "games", fixture("lichess.pgn")]) == 0

    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


def test_dataset_build_still_exits_two_with_nowhere_to_say_why(tmp_path):
    def out_of_space(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        dead_everywhere(patch, DeadStream())
        patch.setattr(os, "fsync", out_of_space)

        with pytest.raises(SystemExit) as exit_info:
            main(["dataset", "build", "games", fixture("lichess.pgn")])

    assert exit_info.value.code == 2, "the exit status stands even with nobody to tell"


def test_dataset_stats_survives_a_reader_that_went_away(tmp_path):
    main(["dataset", "build", "games", fixture("lichess.pgn")])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "stdout", DeadStream())

        assert main(["dataset", "stats", "games"]) == 0


RUN_MAIN = "import sys; from chess_ai.cli import main; sys.exit(main(sys.argv[1:]))"
"""What the installed ``chess-ai`` script does, for the tests that need a process of their own."""


def with_nobody_reading(*args, cwd, stdout=False, stderr=False, script=None):
    """Run ``chess-ai args`` in a real process, with the named streams on an unread pipe.

    A real process on a real pipe, because what goes wrong here happens on the way out of the
    interpreter: what a failed write left in a stream's buffer is flushed a second time at
    shutdown, and that failure is reported as "Exception ignored" and replaces the exit status
    with 120. A stream substituted in-process has no such buffer, so it cannot show any of it.
    """
    read, write = os.pipe()
    os.close(read)
    try:
        return subprocess.run(
            [sys.executable, "-c", script or RUN_MAIN, *args],
            cwd=cwd,
            stdout=write if stdout else subprocess.DEVNULL,
            stderr=write if stderr else subprocess.PIPE,
            text=True,
        )
    finally:
        os.close(write)


def test_a_finished_build_keeps_its_exit_status_when_nothing_reads_its_stdout(tmp_path):
    # "chess-ai dataset build ... | head": a finished, published, correct build whose reader has
    # gone. It answers the cron job that ran it, and what it answers must be that it worked.
    finished = with_nobody_reading(
        "dataset", "build", "games", fixture("lichess.pgn"), cwd=tmp_path, stdout=True
    )

    assert finished.returncode == 0, finished.stderr
    assert "Exception ignored" not in finished.stderr
    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


def test_a_finished_build_keeps_its_exit_status_when_nothing_reads_anything(tmp_path):
    # A dropped ssh session takes stdout and stderr together, and the progress line is written to
    # stderr by something that is right not to care whether it lands. The status still stands.
    finished = with_nobody_reading(
        "dataset", "build", "games", fixture("lichess.pgn"), cwd=tmp_path, stdout=True, stderr=True
    )

    assert finished.returncode == 0
    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


def test_a_failed_command_keeps_its_exit_status_when_nothing_reads_anything(tmp_path):
    failed = with_nobody_reading("dataset", "stats", "nope", cwd=tmp_path, stdout=True, stderr=True)

    assert failed.returncode == 2, "not 120, which is what a shutdown flush would make of it"


def test_a_rejected_command_line_keeps_its_exit_status_when_nothing_reads_anything(tmp_path):
    # argparse prints its own usage and error and exits without ever coming back through main.
    rejected = with_nobody_reading("--nope", cwd=tmp_path, stdout=True, stderr=True)

    assert rejected.returncode == 2, "not 120, which is what a shutdown flush would make of it"


CRASHING = """
import sys
import chess_ai.cli as cli


def crash(config, args):
    raise RuntimeError("nothing expected this")


cli._dataset_stats = crash
sys.exit(cli.main(["dataset", "stats", "anything"]))
"""
"""A command whose handler raises something main does not catch, as a real bug in it would."""


def test_a_crash_keeps_its_exit_status_when_nothing_reads_anything(tmp_path):
    # The traceback for an exception that escapes main is printed by the interpreter, after any
    # last chance main had at its streams, and it refills the buffer that the shutdown flush then
    # fails on. 1 says a command crashed; 120 says nothing at all to whatever ran it.
    crashed = with_nobody_reading(cwd=tmp_path, stdout=True, stderr=True, script=CRASHING)

    assert crashed.returncode == 1, "the status for a crash, not 120"


EMBEDDED = """
import os
import sys
from chess_ai.cli import main

# stdout on a pipe nobody is reading, as a caller's own can be, and a real descriptor rather
# than a substitute for it, so that silencing it would really silence something.
read, write = os.pipe()
os.close(read)
os.dup2(write, 1)
os.close(write)
sys.stdout = open(1, "w", closefd=False)

status = main(sys.argv[1:])

try:
    os.write(1, b"anyone still there?")
except OSError:
    sys.stderr.write(f"{status} untouched")
else:
    sys.stderr.write(f"{status} redirected")
"""
"""One call to ``main`` whose stdout has gone, and then a look at whose descriptor it was."""


def test_main_does_not_redirect_the_descriptors_of_whoever_called_it(tmp_path):
    # main is an ordinary function, and tests and other programs call it in the same process.
    # Pointing descriptor 1 at the null device to save one command's exit status would silence
    # everything that ran after it, with nothing said. Only the interpreter's own exit may.
    called = subprocess.run(
        [sys.executable, "-c", EMBEDDED, "dataset", "build", "games", fixture("lichess.pgn")],
        cwd=tmp_path,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert called.stderr.endswith("0 untouched"), called.stderr
    assert load_manifest(tmp_path / "data" / "datasets" / "games").games == 4


CLOSES_STDOUT = "import sys, chess_ai.cli; sys.stdout.close()"
"""A program that has only imported the module, and tidied up after itself before exiting."""


def test_a_program_that_closed_stdout_is_not_given_a_traceback_on_the_way_out(tmp_path):
    # Importing this module registers a callback at interpreter exit, and an exception out of an
    # atexit callback is printed as a traceback. A test harness or an embedding program that
    # pointed sys.stdout at a file and closed it before exiting has done nothing to earn one.
    quiet = subprocess.run(
        [sys.executable, "-c", CLOSES_STDOUT], cwd=tmp_path, capture_output=True, text=True
    )

    assert quiet.stderr == "", "nothing of ours on the way out"
    assert quiet.returncode == 0


def test_a_program_that_detached_stdout_is_not_given_one_either(tmp_path):
    # The interpreter has its own complaint about a detached stream and its own exit status for
    # it, and neither is ours to fix: a stream in that state has no descriptor left to point
    # anywhere. Not adding a second traceback to it is.
    detached = subprocess.run(
        [sys.executable, "-c", "import sys, chess_ai.cli; sys.stdout.detach()"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert "atexit" not in detached.stderr
    assert "_drain_streams" not in detached.stderr


def test_train_writes_a_run_in_the_configured_runs_directory(tmp_path, capsys):
    from training_helpers import dataset, experiment

    from chess_ai.training import RunStatus, open_run

    dataset(tmp_path / "data")
    config = experiment(tmp_path / "tiny.toml")

    assert main(["train", str(config)]) == 0

    out = capsys.readouterr().out
    reader = open_run(tmp_path / "runs", "tiny")
    assert reader.status.status == RunStatus.FINISHED
    assert "mlp," in out and "parameters" in out, "the parameter count, before it starts"
    assert "positions/s measured" in out, "and how fast a step actually was"
    assert "runs/tiny" in out, "and where the run went"


def test_train_uses_the_configured_runs_directory(tmp_path):
    from training_helpers import dataset, experiment

    from chess_ai.training import list_runs

    dataset(tmp_path / "elsewhere")
    config = tmp_path / "custom.toml"
    config.write_text(
        f'[paths]\ndata = "{tmp_path / "elsewhere"}"\nruns = "{tmp_path / "somewhere"}"\n'
    )

    assert main(["--config", str(config), "train", str(experiment(tmp_path / "tiny.toml"))]) == 0

    assert list_runs(tmp_path / "somewhere") == ["tiny"]


def test_train_can_be_given_the_run_name_on_the_command_line(tmp_path):
    from training_helpers import dataset, experiment

    from chess_ai.training import list_runs

    dataset(tmp_path / "data")
    config = experiment(tmp_path / "tiny.toml")

    assert main(["train", str(config), "--name", "second-try"]) == 0

    assert list_runs(tmp_path / "runs") == ["second-try"]


def test_train_refuses_to_write_over_a_run_unless_told_to(tmp_path, capsys):
    from training_helpers import dataset, experiment

    dataset(tmp_path / "data")
    config = experiment(tmp_path / "tiny.toml")
    assert main(["train", str(config)]) == 0

    with pytest.raises(SystemExit) as exit_info:
        main(["train", str(config)])

    assert exit_info.value.code == 2
    assert "already in" in capsys.readouterr().err
    assert main(["train", str(config), "--overwrite"]) == 0


def test_train_reports_a_config_it_cannot_use_without_a_traceback(tmp_path, capsys):
    broken = tmp_path / "broken.toml"
    broken.write_text('[dataset]\nname = "games"\n[training]\nbatch_size = -1\n')

    with pytest.raises(SystemExit) as exit_info:
        main(["train", str(broken)])

    assert exit_info.value.code == 2
    assert "invalid experiment config" in capsys.readouterr().err


def test_train_reports_a_missing_config_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", str(tmp_path / "nowhere.toml")])

    assert exit_info.value.code == 2
    assert "cannot read experiment config" in capsys.readouterr().err


def test_a_stopped_run_says_how_to_carry_on_and_resume_does(tmp_path, capsys, monkeypatch):
    import signal

    from training_helpers import dataset, experiment

    from chess_ai.training import RunStatus, open_run, trainer

    dataset(tmp_path / "data")
    config = experiment(tmp_path / "tiny.toml")
    real = trainer._step
    calls = {"n": 0}

    def stop_on_the_run_s_second_step(*args, **kwargs):
        result = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == trainer.PROBE_STEPS + 1 + 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(trainer, "_step", stop_on_the_run_s_second_step)

    assert main(["train", str(config)]) == 128 + signal.SIGTERM, "a stop is not a finish"
    err = capsys.readouterr().err
    assert "stopped by SIGTERM at step 2; 'chess-ai resume tiny' carries on from there" in err
    monkeypatch.setattr(trainer, "_step", real)

    assert main(["resume", "tiny"]) == 0
    assert open_run(tmp_path / "runs", "tiny").status.status == RunStatus.FINISHED
    assert "resuming at step 2" in capsys.readouterr().out


def test_resume_says_why_a_run_cannot_be_resumed_without_a_traceback(tmp_path, capsys):
    from training_helpers import dataset, experiment

    dataset(tmp_path / "data")
    assert main(["train", str(experiment(tmp_path / "tiny.toml"))]) == 0

    with pytest.raises(SystemExit) as exit_info:
        main(["resume", "tiny"])

    assert exit_info.value.code == 2
    assert "already finished" in capsys.readouterr().err


def annotated_run(tmp_path, name: str = "tiny") -> Path:
    """A run directory with nothing in it but its name, which is all annotating it needs."""
    directory = tmp_path / "runs" / name
    directory.mkdir(parents=True)
    (directory / "run.json").write_text("{}")
    return directory


def test_runs_annotate_gives_a_run_a_title_tags_and_notes(tmp_path, capsys):
    from chess_ai.training.run_store import RunNotes, open_run

    annotated_run(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("Width 2048.\nBetter than 1024.\n")

    assert (
        main(
            [
                "runs",
                "annotate",
                "tiny",
                "--title",
                "Wide MLP",
                "--tag",
                "mlp",
                "--tag",
                "wide",
                "--notes-file",
                str(notes),
            ]
        )
        == 0
    )

    assert open_run(tmp_path / "runs", "tiny").notes == RunNotes(
        title="Wide MLP", tags=["mlp", "wide"], notes="Width 2048.\nBetter than 1024.\n"
    )
    out = capsys.readouterr().out
    assert "title: Wide MLP" in out and "tags: mlp, wide" in out and "Better than 1024." in out


def test_runs_annotate_changes_only_what_it_is_told_to(tmp_path):
    from chess_ai.training.run_store import RunNotes, open_run, save_notes

    save_notes(
        open_run(tmp_path / "runs", annotated_run(tmp_path).name),
        RunNotes(title="Kept", tags=["a", "b"], notes="kept"),
    )

    assert main(["runs", "annotate", "tiny", "--tag", "c", "--untag", "a"]) == 0

    assert open_run(tmp_path / "runs", "tiny").notes == RunNotes(
        title="Kept", tags=["b", "c"], notes="kept"
    )

    assert main(["runs", "annotate", "tiny", "--title", "", "--notes", ""]) == 0

    assert open_run(tmp_path / "runs", "tiny").notes == RunNotes(tags=["b", "c"])


def test_runs_annotate_with_no_options_shows_the_notes_and_writes_nothing(tmp_path, capsys):
    directory = annotated_run(tmp_path)

    assert main(["runs", "annotate", "tiny"]) == 0

    assert "title: –" in capsys.readouterr().out
    assert not (directory / "notes.json").exists()


def test_runs_annotate_refuses_a_tag_that_is_not_one_word(tmp_path, capsys):
    directory = annotated_run(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["runs", "annotate", "tiny", "--tag", "two words"])

    assert exit_info.value.code == 2
    assert "tags: invalid tag 'two words'" in capsys.readouterr().err
    assert not (directory / "notes.json").exists()


def test_runs_annotate_refuses_a_run_that_is_not_there(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["runs", "annotate", "ghost", "--tag", "x"])

    assert exit_info.value.code == 2
    assert "no run called 'ghost'" in capsys.readouterr().err


def test_runs_list_shows_every_run_or_those_with_a_tag(tmp_path, capsys):
    from chess_ai.training.run_store import RunNotes, open_run, save_notes

    for name, tags in [("a", ["mlp", "wide"]), ("b", ["mlp"]), ("c", [])]:
        annotated_run(tmp_path, name)
        save_notes(open_run(tmp_path / "runs", name), RunNotes(title=f"Run {name}", tags=tags))

    assert main(["runs", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "a  starting [mlp, wide]  Run a",
        "b  starting [mlp]  Run b",
        "c  starting  Run c",
    ]

    assert main(["runs", "list", "--tag", "mlp", "--tag", "wide"]) == 0
    assert capsys.readouterr().out.splitlines() == ["a  starting [mlp, wide]  Run a"]

    assert main(["runs", "list", "--tag", "resnet"]) == 0
    assert "no runs tagged resnet" in capsys.readouterr().err


def test_runs_list_still_lists_a_run_whose_notes_cannot_be_read(tmp_path, capsys):
    (annotated_run(tmp_path) / "notes.json").write_text("{not json")

    assert main(["runs", "list"]) == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["tiny  starting"]
    assert "notes.json" in captured.err

    assert main(["runs", "list", "--tag", "mlp"]) == 0
    assert "tiny" not in capsys.readouterr().out, "its tags cannot be known, so no tag matches"
