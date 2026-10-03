"""The ``chess-ai`` command-line tool."""

import argparse
import atexit
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TextIO

from chess_ai.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_FILE, Config, ConfigError, load_config

AUTO = "auto"
"""What ``--rating-source`` is when each game's own headers should say."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        return args.handler(config, args)
    except (ConfigError, _UserError) as e:
        _say(f"{parser.prog}: error: {e}", err=True)
        parser.exit(2)
        raise  # parser.exit has already left; this is only here to say so.


class _UserError(Exception):
    """Something the person running the command can fix, reported without a traceback."""


def _say(text: str, *, err: bool = False) -> None:
    """Print ``text``, or stop trying to, if there is nobody left to print it to.

    Every stream this command writes to can go away while it is running, and they go together: a
    closed terminal, a dropped ssh session, a pipe into ``head``. None of that is worth turning a
    finished build into a traceback, and on the way out of a failed one it must not take the place
    of the reason it failed. The exit status says what happened whether or not anyone hears it —
    which takes :func:`_drain_streams` as well, because what a failed write leaves behind is as
    costly as the write itself.
    """
    with suppress(OSError):
        print(text, file=sys.stderr if err else sys.stdout, flush=True)


def _drain_streams() -> None:
    """Leave the standard streams in a state the interpreter can shut down quietly.

    What a failed write left in a stream's buffer is flushed once more by the interpreter on its
    way out, where the failure is reported as ``Exception ignored in: <_io.TextIOWrapper ...>``
    and replaces the process's exit status with **120** — see :func:`_silence`. A finished build
    answering 120 because its reader had gone is the one thing :func:`_say`'s promise cannot
    afford, and 120 is the one status that tells the cron job reading it nothing at all.

    Registered with :mod:`atexit` rather than done on the way out of :func:`main`, for two
    reasons. It has to run after everything that writes, and not everything does so through
    :func:`_say`: the progress line comes from :class:`~chess_ai.dataset.ProgressPrinter`, the
    usage from :mod:`argparse`, and the traceback for an exception that escapes ``main`` from the
    interpreter itself — that last one printed after any ``finally`` here has already had its
    turn. And :func:`main` is an ordinary function that a test or another program calls in the
    same process, where :func:`_silence` would outlive the call; at the interpreter's own exit
    there is nothing left for it to take with it.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue  # Started without one, as a windowed process is; nothing to flush or lose.
        try:
            stream.flush()
        except OSError:
            _silence(stream)
        except ValueError:
            # Closed or detached by whoever called main. Both of those flush on their way, so
            # there is nothing left here to cost anyone an exit status, and nothing that could
            # be done about it if there were: a stream in either state has no descriptor left
            # to point anywhere. Passed over quietly, because an exception out of an atexit
            # callback is printed as a traceback, and a program that only imported this module
            # has done nothing to earn one.
            pass


atexit.register(_drain_streams)


def _silence(stream: TextIO) -> None:
    """Point ``stream`` at the null device, so that nothing it is still holding is written again.

    Ignoring the write error is not enough on its own. What :func:`print` handed the stream is
    still in its buffer, and the interpreter flushes the standard streams once more on the way
    out, where the second failure is reported as ``Exception ignored in: ...`` and — the part
    that matters — replaces the process's exit status with 120.

    Redirecting the descriptor rather than the Python object is what makes it stick: the buffer
    drains into the null device, and so does anything written afterwards by a library that never
    heard the stream had gone. That is a process-wide change that cannot be taken back, which is
    why the only caller is :func:`_drain_streams` and the only moment it runs is the interpreter
    shutting down, when there is nothing left in the process to silence.

    Nothing it can be asked to do may raise: it is called from an :mod:`atexit` callback, where
    an exception is a traceback printed at a program that has not done anything. A stream with no
    descriptor of its own raises :exc:`io.UnsupportedOperation`, which is an :exc:`OSError`; one
    that has been closed or detached raises :exc:`ValueError`. Neither has a buffer this could
    reach anyway.
    """
    with suppress(OSError, ValueError):
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, stream.fileno())
        finally:
            os.close(null)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-ai", description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"configuration file (default: ${CONFIG_PATH_ENV}, else ./{DEFAULT_CONFIG_FILE} "
        "if it exists)",
    )
    commands = parser.add_subparsers(title="commands", required=True, metavar="COMMAND")

    serve = commands.add_parser("serve", help="serve the web app")
    serve.set_defaults(handler=_serve)

    _add_dataset_commands(commands)
    _add_train_command(commands)
    _add_runs_commands(commands)
    return parser


def _add_dataset_commands(commands: argparse._SubParsersAction) -> None:
    """``chess-ai dataset ...``: making datasets out of PGN files, and looking at them."""
    from chess_ai.dataset import DEFAULT_VALIDATION_FRACTION, RatingSource
    from chess_ai.dataset.builder import default_workers, most_workers

    dataset = commands.add_parser("dataset", help="build and inspect training datasets")
    actions = dataset.add_subparsers(title="dataset commands", required=True, metavar="COMMAND")

    build = actions.add_parser(
        "build",
        help="build a dataset from PGN files",
        description="Build a named dataset in the data directory from PGN files, directories "
        "of PGN files, or globs. Games that cannot be read are skipped and counted.",
    )
    build.add_argument("name", help="what to call the dataset")
    build.add_argument(
        "sources",
        nargs="+",
        metavar="PGN",
        help="PGN files, directories of them, or glob patterns",
    )
    build.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        metavar="FRACTION",
        help="share of games held back for validation, chosen by a hash of each game "
        f"(default: {DEFAULT_VALIDATION_FRACTION})",
    )
    build.add_argument(
        "--rating-source",
        choices=[AUTO, *(source.name.lower() for source in RatingSource)],
        default=AUTO,
        help="rating pool to record for every game (default: auto, from each game's headers)",
    )
    build.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"processes parsing the PGN files at once (default here: {default_workers()}, "
        f"at most {most_workers()}). The default is one per usable CPU core up to a cap, since "
        "each process holds part of a file while it reads it. 1 reads them in this process, and "
        "so does any build with little enough to do that starting processes would cost more "
        "than the reading",
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a dataset of this name that is already there",
    )
    build.set_defaults(handler=_build_dataset)

    stats = actions.add_parser(
        "stats",
        help="show what a dataset contains",
        description="Summarise a built dataset: its sources, counts, and the distribution of "
        "ratings, results and time controls.",
    )
    stats.add_argument("name", help="the dataset to summarise")
    stats.set_defaults(handler=_dataset_stats)


def _add_train_command(commands: argparse._SubParsersAction) -> None:
    """``chess-ai train``: one experiment config file, one run directory."""
    train = commands.add_parser(
        "train",
        help="train a model from an experiment config file",
        description="Train the architecture an experiment config names, on the dataset it "
        "names, and write everything the run produces to a run directory. The config file is "
        "copied into the run, so a run can always be traced back to what produced it.",
    )
    train.add_argument("config_file", type=Path, metavar="CONFIG", help="the experiment config")
    train.add_argument(
        "--name",
        metavar="NAME",
        help="what to call the run, overriding the config (default: the config's name, "
        "else the config file's own name)",
    )
    train.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a run of this name that is already there, metrics and checkpoints and all",
    )
    train.set_defaults(handler=_train)

    resume = commands.add_parser(
        "resume",
        help="carry on training a stopped or crashed run",
        description="Carry on training a run from its latest checkpoint, with the config, "
        "weights, optimizer state, schedule and data order it had, to the end of its schedule. "
        "A run stops at the end of a step and saves a checkpoint on ctrl-c or SIGTERM; a run "
        "that crashed carries on from the last checkpoint it saved.",
    )
    resume.add_argument("name", help="the run")
    resume.set_defaults(handler=_resume)


def _add_runs_commands(commands: argparse._SubParsersAction) -> None:
    """``chess-ai runs ...``: finding training runs, and naming, tagging and annotating them."""
    runs = commands.add_parser("runs", help="list training runs and annotate them")
    actions = runs.add_subparsers(title="runs commands", required=True, metavar="COMMAND")

    listing = actions.add_parser(
        "list",
        help="list the training runs",
        description="List the training runs in the runs directory, with their state, tags "
        "and titles. Runs tagged archived are left out unless asked for.",
    )
    listing.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="TAG",
        help="only the runs with this tag; given more than once, only those with all of them",
    )
    listing.add_argument(
        "--archived",
        action="store_true",
        help="include the runs tagged archived, which are otherwise left out unless "
        "--tag archived asks for them",
    )
    listing.set_defaults(handler=_list_runs)

    annotate = actions.add_parser(
        "annotate",
        help="give a run a title, tags and notes",
        description="Change a run's title, tags and notes, which are kept in its run directory "
        "and shown on the runs dashboard. With no options, show what it has. The run's name "
        "never changes; the title is what it is shown as instead.",
    )
    annotate.add_argument("name", help="the run")
    annotate.add_argument(
        "--title", metavar="TEXT", help="what to show the run as; an empty one shows its name"
    )
    annotate.add_argument(
        "--tag", action="append", default=[], metavar="TAG", help="add a tag; may be repeated"
    )
    annotate.add_argument(
        "--untag",
        action="append",
        default=[],
        metavar="TAG",
        help="remove a tag; may be repeated",
    )
    notes = annotate.add_mutually_exclusive_group()
    notes.add_argument("--notes", metavar="TEXT", help="replace the notes with this")
    notes.add_argument(
        "--notes-file",
        type=Path,
        metavar="PATH",
        help="replace the notes with what this file says; - reads them from standard input",
    )
    annotate.set_defaults(handler=_annotate_run)


def _list_runs(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.training.run_store import RunError, RunNotes, RunReader, list_runs, listed

    shown = archived = 0
    for name in list_runs(config.paths.runs):
        run = RunReader(config.paths.runs / name)
        try:
            notes = run.notes
        except RunError as e:
            # Listed all the same, as the web list does: a run is still a run without its
            # notes. Only a filter by tag has to leave it out, since its tags cannot be known.
            _say(f"chess-ai: warning: {e}", err=True)
            notes = RunNotes()
        if not listed(notes.tags, wanted=args.tag, archived=args.archived):
            # Counted, so that a list emptied by archiving does not read as if the runs had gone.
            if listed(notes.tags, wanted=args.tag, archived=True):
                archived += 1
            continue
        try:
            beat = run.status
        except RunError:
            beat = None
        state = beat.status.value if beat is not None else "starting"
        tags = f" [{', '.join(notes.tags)}]" if notes.tags else ""
        title = f"  {notes.title}" if notes.title else ""
        _say(f"{name}  {state}{tags}{title}")
        shown += 1
    if shown == 0:
        wanted = f" tagged {', '.join(args.tag)}" if args.tag else ""
        hidden = f" ({archived} archived; --archived shows them)" if archived else ""
        _say(f"chess-ai: no runs{wanted} in {config.paths.runs}{hidden}", err=True)
    return 0


def _annotate_run(config: Config, args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from chess_ai.training.run_store import RunError, RunNotes, open_run, save_notes

    try:
        run = open_run(config.paths.runs, args.name)
        notes = run.notes
    except RunError as e:
        raise _UserError(e) from e
    text = notes.notes
    if args.notes is not None:
        text = args.notes
    elif args.notes_file is not None:
        text = _read_notes(args.notes_file)
    changing = args.title is not None or args.tag or args.untag or text != notes.notes
    if changing:
        # Removing wins over adding, so that "--tag a --untag a" leaves the run without it.
        tags = [tag for tag in [*notes.tags, *args.tag] if tag not in args.untag]
        try:
            edited = RunNotes(
                title=notes.title if args.title is None else args.title, tags=tags, notes=text
            )
        except ValidationError as e:
            problems = "; ".join(
                f"{error['loc'][0]}: {str(error['msg']).removeprefix('Value error, ')}"
                for error in e.errors()
            )
            raise _UserError(problems) from e
        try:
            notes = save_notes(run, edited)
        except RunError as e:
            raise _UserError(e) from e
    _say(f"run: {run.name}")
    _say(f"title: {notes.title or '–'}")
    _say(f"tags: {', '.join(notes.tags) or '–'}")
    _say("notes:" + (f"\n{notes.notes.rstrip()}" if notes.notes.strip() else " –"))
    return 0


def _read_notes(path: Path) -> str:
    """What ``path`` says, or standard input for ``-``."""
    if str(path) == "-":
        return sys.stdin.read()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise _UserError(f"cannot read {path}: {e.strerror}") from e


def _train(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.training import ExperimentError, load_experiment, train

    try:
        experiment = load_experiment(args.config_file, name=args.name)
    except ExperimentError as e:
        raise _UserError(e) from e
    try:
        config_text = args.config_file.read_text(encoding="utf-8")
    except OSError as e:
        raise _UserError(f"cannot read {args.config_file}: {e.strerror}") from e
    return _training(
        experiment.name,
        lambda: train(
            experiment,
            data_dir=config.paths.data,
            runs_dir=config.paths.runs,
            config_text=config_text,
            overwrite=args.overwrite,
            say=_say,
        ),
    )


def _resume(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.training import resume

    return _training(
        args.name,
        lambda: resume(args.name, data_dir=config.paths.data, runs_dir=config.paths.runs, say=_say),
    )


def _training(name: str, run: Callable[[], object]) -> int:
    """Run a training job, and answer for how it ended in words and in the exit status.

    A run that was stopped exits with the usual status for the signal that stopped it, so that a
    script waiting on it can tell a stop from a finish.
    """
    from chess_ai.training import RunError, TrainingError, TrainingStopped

    try:
        run()
    except (TrainingError, RunError) as e:
        # The trainer turns the run store's errors into TrainingError; RunError is caught as
        # well so that a path it does not cover cannot become a traceback either.
        raise _UserError(e) from e
    except TrainingStopped as e:
        _say(
            f"chess-ai: {e}; 'chess-ai resume {name}' carries on from there",
            err=True,
        )
        return 128 + e.signal
    except KeyboardInterrupt:
        # Not an error: ctrl-c twice, or once before the run had started. No hint about
        # resuming, since in the second case there may be no run to resume; a run that had
        # started has already said it stopped.
        _say("chess-ai: interrupted", err=True)
        return 130
    except OSError as e:
        raise _UserError(f"could not finish the run: {e.strerror or e}") from e
    return 0


def _serve(config: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from chess_ai.web import create_app

    server = config.server
    _say(f"chess-ai: serving on http://{server.host}:{server.port}{server.path_prefix}/")
    uvicorn.run(create_app(config), host=server.host, port=server.port)
    return 0


def _build_dataset(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset import (
        DatasetError,
        ProgressPrinter,
        RatingSource,
        build_dataset,
        dataset_path,
    )

    source = None if args.rating_source == AUTO else RatingSource[args.rating_source.upper()]
    printer = ProgressPrinter()
    try:
        manifest = build_dataset(
            args.name,
            args.sources,
            data_dir=config.paths.data,
            validation_fraction=args.validation_fraction,
            rating_source=source,
            progress=printer,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    except DatasetError as e:
        raise _UserError(e) from e
    except OSError as e:
        # A full disk, a mount that went away: the most ordinary way a long build dies, and the
        # one thing here that was still answered with a traceback.
        raise _UserError(f"could not build dataset {args.name!r}: {e.strerror or e}") from e
    finally:
        # A build that failed never printed that it was done, so the line it was rewriting is
        # still open and the error would otherwise be written onto the end of it. build_dataset
        # ends it for everything it gets as far as starting; this covers what it does not.
        printer.finish()
    lost = [source.path for source in manifest.sources if source.went_away]
    # Only the sources that were there throughout and whose contents gave up: a source that went
    # away took an unknown number of games with it, and saying "not read whole" of that on stdout
    # while the warning says the rest of it understates it in the line a log gets read for.
    unread = [
        source.path
        for source in manifest.sources
        if source.error is not None and not source.went_away
    ]
    _say(
        f"chess-ai: dataset {manifest.name} in {dataset_path(config.paths.data, manifest.name)}: "
        f"{manifest.games:,} games, {manifest.positions:,} positions, "
        f"{manifest.games_skipped:,} skipped"
        # A source that could not be read leaves the dataset short of its games, which is not
        # something to leave to whoever thinks to run "dataset stats" afterwards.
        + (f"; {len(unread)} source(s) not read whole: {', '.join(unread)}" if unread else "")
    )
    if lost:
        # Built, and not the dataset that was asked for. Said on its own line and answered for in
        # the exit status, because a build in a cron job is read by a script before a person.
        _say(
            f"chess-ai: warning: dataset {manifest.name} is missing games from "
            f"{len(lost)} source(s) that could not be read whole: {', '.join(lost)}",
            err=True,
        )
        return 1
    return 0


def _dataset_stats(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset import DatasetError, ManifestError, open_dataset, summarize

    try:
        dataset = open_dataset(args.name, data_dir=config.paths.data)
    except (DatasetError, ManifestError) as e:
        raise _UserError(_with_available(e, config.paths.data)) from e
    _say(summarize(dataset.manifest).rstrip("\n"))
    return 0


def _with_available(error: Exception, data_dir: Path) -> str:
    """``error``, plus the datasets there actually are, which is usually the next question."""
    from chess_ai.dataset import list_datasets

    names = list_datasets(data_dir)
    if not names:
        return f"{error} (no datasets in {data_dir}; build one with 'chess-ai dataset build')"
    return f"{error} (datasets in {data_dir}: {', '.join(names)})"
