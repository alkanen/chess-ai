import errno
import io
import logging
import os
from datetime import datetime

import pytest

from chess_ai import output
from chess_ai.cli import _build_parser, _verbosity
from chess_ai.output import DETAIL, MILESTONE, RESULT, Timestamps, configure, say

LOGGER = logging.getLogger("chess_ai.somewhere")

WHEN = datetime(2026, 10, 12, 14, 3, 7)

to_the_journal = output.to_the_journal
"""The real one, which conftest replaces for every test so that lines are compared unstamped."""


@pytest.fixture
def stamped(monkeypatch):
    """Lines stamped as they would be outside the journal, at a fixed time."""
    monkeypatch.setattr(output, "to_the_journal", lambda stream: False)
    monkeypatch.setattr(output, "_now", lambda: WHEN)


def said_at(verbosity, capsys):
    configure(verbosity=verbosity)
    for kind in (DETAIL, MILESTONE, RESULT):
        say(kind.name.lower(), kind=kind)
    return capsys.readouterr().out.split()


def test_everything_is_said_by_default_and_more_verbose_says_no_more(capsys):
    assert said_at(0, capsys) == ["detail", "milestone", "result"]
    assert said_at(2, capsys) == ["detail", "milestone", "result"]


def test_quiet_leaves_out_the_details_and_quieter_everything_but_the_results(capsys):
    assert said_at(-1, capsys) == ["milestone", "result"]
    assert said_at(-2, capsys) == ["result"]


def logged_at(verbosity, capsys):
    configure(verbosity=verbosity)
    LOGGER.debug("debug")
    LOGGER.info("info")
    LOGGER.warning("warning")
    LOGGER.error("error")
    return capsys.readouterr().err.splitlines()


def test_warnings_and_errors_reach_stderr_by_default_with_one_prefix(capsys):
    assert logged_at(0, capsys) == ["chess-ai: warning: warning", "chess-ai: error: error"]


def test_a_single_quiet_still_shows_warnings_and_a_second_only_errors(capsys):
    assert logged_at(-1, capsys) == ["chess-ai: warning: warning", "chess-ai: error: error"]
    assert logged_at(-2, capsys) == ["chess-ai: error: error"]


def test_verbose_adds_info_then_debug_records_with_where_they_came_from(capsys):
    assert logged_at(1, capsys) == [
        "chess-ai: info [chess_ai.somewhere]: info",
        "chess-ai: warning: warning",
        "chess-ai: error: error",
    ]
    assert logged_at(2, capsys)[0] == "chess-ai: debug [chess_ai.somewhere]: debug"


def test_logging_set_up_twice_still_writes_each_record_once(capsys):
    configure()
    configure()
    LOGGER.warning("once")

    assert capsys.readouterr().err == "chess-ai: warning: once\n"


def test_a_logged_exception_keeps_its_traceback(capsys):
    configure()
    try:
        raise ValueError("what went wrong")
    except ValueError:
        LOGGER.warning("it failed", exc_info=True)

    err = capsys.readouterr().err
    assert err.startswith("chess-ai: warning: it failed\nTraceback")
    assert "ValueError: what went wrong" in err


def test_lines_and_records_are_stamped_with_the_time(stamped, capsys):
    configure()
    say("a line")
    LOGGER.warning("a record")

    said = capsys.readouterr()
    assert said.out == "2026-10-12 14:03:07 a line\n"
    assert said.err == "2026-10-12 14:03:07 chess-ai: warning: a record\n"


def test_a_query_is_not_stamped_unless_asked_to_be(stamped, capsys):
    configure(query=True)
    say("a listing")
    configure(query=True, timestamps=Timestamps.ALWAYS)
    say("a stamped listing")

    assert capsys.readouterr().out == "a listing\n2026-10-12 14:03:07 a stamped listing\n"


def test_never_means_never(stamped, capsys):
    configure(timestamps=Timestamps.NEVER)
    say("a line")

    assert capsys.readouterr().out == "a line\n"


def test_the_journal_is_the_stream_journal_stream_names(monkeypatch, tmp_path):
    with open(tmp_path / "journal", "w") as journal, open(tmp_path / "file", "w") as other:
        status = os.fstat(journal.fileno())
        monkeypatch.setenv("JOURNAL_STREAM", f"{status.st_dev}:{status.st_ino}")

        assert to_the_journal(journal)
        # Inherited by a command whose output the service sent somewhere else.
        assert not to_the_journal(other)
        assert not to_the_journal(io.StringIO()), "a stream with no descriptor"


def test_journal_stream_that_is_not_a_device_and_an_inode_is_no_journal(monkeypatch, tmp_path):
    monkeypatch.setenv("JOURNAL_STREAM", "nonsense")

    with open(tmp_path / "file", "w") as stream:
        assert not to_the_journal(stream)


class GoneStream(io.StringIO):
    def write(self, text: str) -> int:
        raise OSError(errno.EIO, "Input/output error")


def test_a_record_for_a_stderr_that_has_gone_is_dropped_without_a_traceback(monkeypatch):
    gone = GoneStream()
    reports = io.StringIO()
    monkeypatch.setattr("sys.stderr", gone)
    # Where logging would report a failed write, which is a traceback.
    monkeypatch.setattr(logging, "raiseExceptions", True)
    monkeypatch.setattr(logging.Handler, "handleError", lambda self, record: reports.write("!"))
    configure()

    LOGGER.warning("nobody hears this")
    say("nor this", err=True)

    assert reports.getvalue() == ""


@pytest.mark.parametrize(
    ("argv", "verbosity"),
    [
        (["runs", "list"], 0),
        (["-v", "runs", "list"], 1),
        (["runs", "list", "-v"], 1),
        (["-v", "runs", "list", "-v"], 2),
        (["-vvv", "runs", "list"], 2),
        (["runs", "list", "-q"], -1),
        (["-q", "runs", "list", "-qq"], -2),
        (["-v", "runs", "list", "-q"], 0),
    ],
)
def test_verbose_and_quiet_count_wherever_they_are_given(argv, verbosity):
    assert _verbosity(_build_parser().parse_args(argv)) == verbosity


def test_timestamps_can_be_given_after_the_command_and_win(capsys):
    parser = _build_parser()

    assert parser.parse_args(["train", "x.toml"]).timestamps is Timestamps.AUTO
    assert parser.parse_args(["train", "x.toml", "--timestamps", "never"]).timestamps == "never"
    late = parser.parse_args(["--timestamps", "never", "train", "x.toml", "--timestamps", "always"])
    assert late.timestamps == "always"


def test_every_command_takes_the_output_options():
    parser = _build_parser()
    commands = [
        ["serve"],
        ["dataset", "build", "name", "a.pgn"],
        ["dataset", "append", "name", "a.pgn"],
        ["dataset", "stats", "name"],
        ["dataset", "download", "2026-01"],
        ["train", "x.toml"],
        ["resume", "name"],
        ["runs", "list"],
        ["runs", "annotate", "name"],
        ["match", "a@1", "b@1"],
        ["ladder", "a@1"],
        ["evaluator"],
        ["evaluate", "a"],
    ]
    for command in commands:
        assert _verbosity(parser.parse_args([*command, "-q"])) == -1, command


def test_only_the_commands_that_list_or_show_are_queries():
    parser = _build_parser()

    assert parser.parse_args(["runs", "list"]).query
    assert parser.parse_args(["runs", "annotate", "name"]).query
    assert parser.parse_args(["dataset", "stats", "name"]).query
    assert not getattr(parser.parse_args(["train", "x.toml"]), "query", False)
    assert not getattr(parser.parse_args(["evaluator"]), "query", False)


def test_a_command_s_error_is_stamped_and_prefixed_as_every_record_is(stamped, capsys):
    from chess_ai.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["dataset", "build", "nothing", "--resume", "-qq"])

    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("2026-10-12 14:03:07 chess-ai: error: "), err


def closed_stream() -> io.StringIO:
    stream = io.StringIO()
    stream.close()
    return stream


def test_a_record_for_a_stderr_that_was_closed_is_dropped_and_the_caller_carries_on(monkeypatch):
    # Closed or detached by whoever called main, which raises ValueError rather than OSError: a
    # handler must never raise into the code that was logging.
    monkeypatch.setattr("sys.stderr", closed_stream())
    configure()

    LOGGER.warning("nobody hears this")
    say("nor this", err=True)


def test_a_line_for_a_stdout_that_was_closed_is_dropped_too(monkeypatch):
    monkeypatch.setattr("sys.stdout", closed_stream())
    configure()

    say("nobody hears this")


def test_every_line_of_a_text_of_several_is_stamped(stamped, capsys):
    configure()
    say("a summary\n  and its second line")

    assert capsys.readouterr().out == (
        "2026-10-12 14:03:07 a summary\n2026-10-12 14:03:07   and its second line\n"
    )


def test_every_line_of_a_logged_traceback_is_stamped(stamped, capsys):
    configure()
    try:
        raise ValueError("what went wrong")
    except ValueError:
        LOGGER.warning("it failed", exc_info=True)

    lines = capsys.readouterr().err.splitlines()
    assert len(lines) > 2
    assert all(line.startswith("2026-10-12 14:03:07 ") for line in lines), lines


def ascii_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding="ascii"), raw


def test_a_line_a_stream_cannot_encode_is_still_written_with_what_it_cannot_escaped(
    monkeypatch,
):
    # PYTHONIOENCODING=ascii, or a locale that is not UTF-8: the stream has not gone, so the line
    # is not to go either, and a dash in it is no reason for a traceback.
    stream, raw = ascii_stream()
    monkeypatch.setattr("sys.stdout", stream)
    configure()

    say("RUN@30000 sample games 3–22 of 102", kind=RESULT)

    stream.flush()
    assert raw.getvalue() == b"RUN@30000 sample games 3\\u201322 of 102\n"


def test_a_record_a_stream_cannot_encode_is_still_written(monkeypatch):
    stream, raw = ascii_stream()
    monkeypatch.setattr("sys.stderr", stream)
    configure()

    LOGGER.warning("cannot lock /data/Ström")

    stream.flush()
    assert raw.getvalue() == b"chess-ai: warning: cannot lock /data/Str\\xf6m\n"


def test_a_line_for_a_stdout_that_was_detached_is_dropped(monkeypatch):
    detached, _ = ascii_stream()
    detached.detach()
    monkeypatch.setattr("sys.stdout", detached)
    configure()

    say("nobody hears this")


class Refusing(io.StringIO):
    """A stream that is open and still refuses a write, for a reason nothing here knows."""

    def write(self, text: str) -> int:
        raise ValueError("something nothing expected")


def test_a_stream_that_is_still_there_and_refuses_a_line_is_not_passed_over(monkeypatch):
    monkeypatch.setattr("sys.stdout", Refusing())
    configure()

    with pytest.raises(ValueError, match="nothing expected"):
        say("a line")


def test_a_record_a_stream_refuses_is_reported_as_logging_reports_it_and_not_raised(monkeypatch):
    reported = []
    monkeypatch.setattr("sys.stderr", Refusing())
    monkeypatch.setattr(
        logging.Handler, "handleError", lambda self, record: reported.append(record)
    )
    configure()

    LOGGER.warning("refused")

    assert [record.getMessage() for record in reported] == ["refused"]
