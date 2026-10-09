"""The ``chess-ai`` command-line tool."""

import argparse
import asyncio
import atexit
import dataclasses
import math
import os
import secrets
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from chess_ai.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_FILE, Config, ConfigError, load_config

if TYPE_CHECKING:
    # Only named in annotations: importing them for real would bring in what they import
    # every time any command runs, torch-adjacent run store and all.
    from chess_ai.match import MatchGame
    from chess_ai.players import Player, SelectionStrategy
    from chess_ai.training.run_store import CheckpointChoice, CheckpointInfo, RunReader

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
    _add_match_command(commands)
    _add_ladder_command(commands)
    return parser


def _add_dataset_commands(commands: argparse._SubParsersAction) -> None:
    """``chess-ai dataset ...``: making datasets out of PGN files, and looking at them."""
    from chess_ai.dataset import DEFAULT_VALIDATION_FRACTION, RatingSource, TimeControl
    from chess_ai.dataset.builder import default_workers, most_workers
    from chess_ai.dataset.filters import Termination

    dataset = commands.add_parser("dataset", help="build and inspect training datasets")
    actions = dataset.add_subparsers(title="dataset commands", required=True, metavar="COMMAND")

    build = actions.add_parser(
        "build",
        help="build a dataset from PGN files",
        description="Build a named dataset in the data directory from PGN files, directories "
        "of PGN files, or globs, keeping the games the filters let through. Games that cannot be "
        "read, and games ended for a rules infraction (mostly cheating), are always left out; "
        "everything left out is counted. The filters can also come from a definition file, "
        "which the options here override.",
    )
    build.add_argument("name", help="what to call the dataset")
    build.add_argument(
        "sources",
        nargs="*",
        metavar="PGN",
        help="PGN files, directories of them, or glob patterns; these replace the definition "
        "file's sources",
    )
    build.add_argument(
        "--definition",
        type=Path,
        metavar="FILE",
        help="a TOML file of sources, filters and options to build from",
    )
    build.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        metavar="FRACTION",
        help="share of games held back for validation, chosen by a hash of each game "
        f"(default: {DEFAULT_VALIDATION_FRACTION})",
    )
    build.add_argument(
        "--rating-source",
        choices=[AUTO, *(source.name.lower() for source in RatingSource)],
        default=None,
        help="rating pool to record for every game (default: auto, from each game's headers)",
    )
    filters = build.add_argument_group(
        "filters",
        "Which games are kept and which of their positions are trained on. None are set by "
        "default. The rating and clock filters are about the player to move: a game is kept if "
        "either player passes, and the other player's positions are stored but not trained on.",
    )
    filters.add_argument(
        "--min-rating", type=int, metavar="RATING", help="lowest rating of the player to move"
    )
    filters.add_argument(
        "--max-rating", type=int, metavar="RATING", help="highest rating of the player to move"
    )
    filters.add_argument(
        "--unknown-rating-passes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="whether a player without a rating passes the rating limits (default: no)",
    )
    filters.add_argument(
        "--time-controls",
        type=_names,
        metavar="LIST",
        help="comma-separated time-control classes to keep: "
        + ", ".join(member.name.lower() for member in TimeControl),
    )
    filters.add_argument(
        "--exclude-terminations",
        type=_names,
        metavar="LIST",
        help="comma-separated ways of ending that leave a game out: "
        + ", ".join(
            member.value for member in Termination if member is not Termination.RULES_INFRACTION
        ),
    )
    filters.add_argument(
        "--from",
        dest="from_date",
        metavar="DATE",
        help="first day kept: YYYY, YYYY-MM or YYYY-MM-DD; undated games are then left out",
    )
    filters.add_argument(
        "--until",
        metavar="DATE",
        help="last day kept, the same way; a year or a month runs to its end",
    )
    filters.add_argument(
        "--min-clock",
        type=float,
        metavar="SECONDS",
        help="time the player to move must have left for the position to be trained on, from "
        "the file's [%%clk] comments; positions without one pass",
    )
    filters.add_argument(
        "--sample",
        type=float,
        metavar="FRACTION",
        help="keep this fraction of the games that pass, chosen by a hash of each game",
    )
    filters.add_argument(
        "--max-games",
        type=int,
        metavar="N",
        help="stop once this many games are kept, the first in the order the sources give them",
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
    build.add_argument(
        "--allow-repeat",
        action="store_true",
        help="read a file even if another source has the same contents or is the same "
        "Lichess month",
    )
    _interrupted_options(build, "build")
    build.set_defaults(handler=_build_dataset)

    append = actions.add_parser(
        "append",
        help="add games from more PGN files to a dataset, as a new version",
        description="Read more PGN files into a dataset through the filters it was built with, "
        "and add the games they keep to the end of it as its next version. Runs on an earlier "
        "version are unaffected, and a config can name any version with [dataset] version. A "
        "file already in the dataset, by its contents or as a Lichess month, is refused.",
    )
    append.add_argument("name", help="the dataset to append to")
    append.add_argument(
        "sources",
        nargs="*",
        metavar="PGN",
        help="PGN files, directories of them, or glob patterns; none with --resume",
    )
    append.add_argument(
        "--max-games",
        type=int,
        metavar="N",
        help="raise the dataset's cap on its total games to N, for a dataset that has reached it",
    )
    append.add_argument(
        "--allow-repeat",
        action="store_true",
        help="append a file even if the dataset already has one with the same contents or of "
        "the same Lichess month",
    )
    _interrupted_options(append, "append")
    append.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"processes parsing the PGN files at once (default here: {default_workers()})",
    )
    append.set_defaults(handler=_append_dataset)

    stats = actions.add_parser(
        "stats",
        help="show what a dataset contains",
        description="Summarise a built dataset: its sources, counts, and the distribution of "
        "ratings, results and time controls.",
    )
    stats.add_argument("name", help="the dataset to summarise")
    stats.add_argument(
        "--version",
        type=int,
        metavar="N",
        help="summarise the dataset as it was at version N (default: the latest)",
    )
    stats.set_defaults(handler=_dataset_stats)

    download = actions.add_parser(
        "download",
        help="download Lichess monthly dumps of rated games",
        description="Download Lichess's monthly dumps of rated standard games into the data "
        "directory, compressed, as a build reads them. A download that stopped is carried on from "
        "where it stopped, and every file is checked against the checksum Lichess publishes "
        "before it is given its name.",
    )
    download.add_argument(
        "months", nargs="+", type=_month, metavar="YYYY-MM", help="the months to download"
    )
    download.set_defaults(handler=_download_dumps)


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


_PLAYER_HELP = """\
a player is a checkpoint or Stockfish, followed by settings as KEY=VALUE, all separated by commas:

  RUN[@CHECKPOINT][,rating=R][,strategy=argmax|sample][,temperature=T]
      a checkpoint of a training run. CHECKPOINT is best (the default), latest or a step.
      rating is the rating it is asked to play like, for both sides; left out, it claims none.
      strategy is how it picks its move: argmax plays its most likely one, sample draws from
      its distribution. A temperature samples at it; below 1 sharpens towards the best move.
  stockfish:ELO[,move-time=SECONDS]
      Stockfish at a calibrated strength, which it raises or lowers to the range it supports.
      Stockfish's own play is not seeded, so a match against it does not replay exactly.

examples:
  chess-ai match resnet10x128 resnet6x64 --games 100
  chess-ai match resnet10x128@latest,rating=1500 stockfish:1350,move-time=0.5
  chess-ai match mlp@24000 mlp@48000 --temperature 0.25 --seed 7
"""


def _add_match_command(commands: argparse._SubParsersAction) -> None:
    """``chess-ai match``: two players, a number of games, an opening set, and the score."""
    from chess_ai.openings import DEFAULT_OPENING_SET, bundled_opening_sets

    match = commands.add_parser(
        "match",
        help="play a match between two checkpoints, or a checkpoint and Stockfish",
        description=(
            "Play a number of games between two players and print the score from each side's\n"
            "point of view. The games start from the lines of an opening set, each line played\n"
            "twice with the colours swapped, so that players who always pick their most likely\n"
            "move still play different games. Every game is saved as it ends, all of them in\n"
            "one PGN file in the games directory, which the replay view opens."
        ),
        epilog=_PLAYER_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    match.add_argument("first", type=_match_player, metavar="FIRST", help="the first player")
    match.add_argument("second", type=_match_player, metavar="SECOND", help="the second player")
    match.add_argument(
        "--games",
        type=_positive,
        default=DEFAULT_MATCH_GAMES,
        metavar="N",
        help=f"how many games to play (default: {DEFAULT_MATCH_GAMES}); fifty give a score "
        "good to about ±14 points either way, two hundred to about ±7",
    )
    match.add_argument(
        "--openings",
        default=DEFAULT_OPENING_SET,
        metavar="SET",
        help=f"the opening set: one that comes with chess-ai ({', '.join(bundled_opening_sets())})"
        f", or the path of a file of your own (default: {DEFAULT_OPENING_SET})",
    )
    match.add_argument(
        "--temperature",
        type=_temperature,
        metavar="T",
        help="sample the moves of every checkpoint that names no strategy or temperature of its "
        "own at this temperature, rather than playing the most likely one",
    )
    match.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="seed the sampling, so that the match can be played again move for move "
        "(default: a random seed, which is printed)",
    )
    match.set_defaults(handler=_match)


DEFAULT_MATCH_GAMES = 50


@dataclass(frozen=True)
class _ModelSide:
    """A checkpoint as the command line names it, before anything has been loaded."""

    text: str
    run: str
    checkpoint: "CheckpointChoice"
    rating: int | None
    strategy: "SelectionStrategy | None"
    """``None`` when the command line said nothing, which ``--temperature`` then decides."""
    temperature: float | None


@dataclass(frozen=True)
class _StockfishSide:
    text: str
    elo: int
    move_time: float


_MAX_RATING = 4000
_MAX_TEMPERATURE = 10.0
_STOCKFISH = "stockfish:"


def _match_player(text: str) -> _ModelSide | _StockfishSide:
    """A player as ``chess-ai match`` takes it; see :data:`_PLAYER_HELP`."""
    head, *rest = text.split(",")
    settings: dict[str, str] = {}
    for setting in rest:
        key, equals, value = setting.partition("=")
        if not equals or not key or not value:
            raise argparse.ArgumentTypeError(f"{setting!r} in {text!r} is not KEY=VALUE")
        if key in settings:
            raise argparse.ArgumentTypeError(f"{key} is given twice in {text!r}")
        settings[key] = value
    if head.startswith(_STOCKFISH):
        side: _ModelSide | _StockfishSide = _stockfish_side(text, head, settings)
    else:
        side = _model_side(text, head, settings)
    if settings:
        raise argparse.ArgumentTypeError(
            f"{text!r}: {', '.join(settings)} is not a setting of this kind of player"
        )
    return side


def _stockfish_side(text: str, head: str, settings: dict[str, str]) -> _StockfishSide:
    from chess_ai.stockfish import DEFAULT_MOVE_TIME, MAX_MOVE_TIME, MIN_MOVE_TIME

    elo = _number(int, head.removeprefix(_STOCKFISH), "an Elo", text, 0, _MAX_RATING)
    move_time = DEFAULT_MOVE_TIME
    if "move-time" in settings:
        move_time = _number(
            float, settings.pop("move-time"), "a move time", text, MIN_MOVE_TIME, MAX_MOVE_TIME
        )
    return _StockfishSide(text=text, elo=elo, move_time=move_time)


def _model_side(text: str, head: str, settings: dict[str, str]) -> _ModelSide:
    from chess_ai.players import MIN_TEMPERATURE

    run, at, which = head.partition("@")
    if not run:
        raise argparse.ArgumentTypeError(f"{text!r} names no run")
    checkpoint: CheckpointChoice = "best"
    if at:
        if which in ("best", "latest"):
            checkpoint = which  # type: ignore[assignment]
        else:
            checkpoint = _number(int, which, "a checkpoint", text, 0, None)
    rating = None
    if "rating" in settings:
        rating = _number(int, settings.pop("rating"), "a rating", text, 0, _MAX_RATING)
    strategy = settings.pop("strategy", None)
    if strategy not in (None, "argmax", "sample"):
        raise argparse.ArgumentTypeError(
            f"{text!r}: the strategy is argmax or sample, not {strategy!r}"
        )
    temperature = None
    if "temperature" in settings:
        temperature = _number(
            float,
            settings.pop("temperature"),
            "a temperature",
            text,
            MIN_TEMPERATURE,
            _MAX_TEMPERATURE,
        )
        if strategy == "argmax":
            raise argparse.ArgumentTypeError(
                f"{text!r}: a checkpoint playing its most likely move has no temperature"
            )
        strategy = "sample"
    return _ModelSide(
        text=text,
        run=run,
        checkpoint=checkpoint,
        rating=rating,
        strategy=strategy,  # type: ignore[arg-type]
        temperature=temperature,
    )


def _number[N: (int, float)](
    kind: type[N], value: str, what: str, text: str, low: N, high: N | None
) -> N:
    """``value`` as a number from ``low`` to ``high``, or a message saying what is wrong."""
    try:
        number = kind(value)
    except ValueError:
        number = None
    if (
        number is None
        or not math.isfinite(number)
        or number < low
        or (high is not None and number > high)
    ):
        bounds = f"from {low:g} to {high:g}" if high is not None else f"of at least {low:g}"
        raise argparse.ArgumentTypeError(f"{text!r}: {what} is a number {bounds}, not {value!r}")
    return number


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"a whole number of at least 1, not {value!r}")
    return number


def _temperature(value: str) -> float:
    from chess_ai.players import MIN_TEMPERATURE

    return _number(float, value, "the temperature", value, MIN_TEMPERATURE, _MAX_TEMPERATURE)


def _match(config: Config, args: argparse.Namespace) -> int:
    import chess.engine

    from chess_ai.match import append_game, distinct_games, match_filename
    from chess_ai.openings import OpeningSetError, load_opening_set
    from chess_ai.players import Player
    from chess_ai.stockfish import StockfishError

    try:
        openings = load_opening_set(args.openings)
    except OpeningSetError as e:
        raise _UserError(e) from e
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    path = config.paths.games / match_filename()
    played: list[MatchGame] = []
    names: list[str] = []
    engines: dict[tuple[str, int], Any] = {}

    def on_game(game: "MatchGame") -> None:
        try:
            append_game(path, game)
        except OSError as e:
            raise _UserError(f"could not save game {game.number} to {path}: {e.strerror}") from e
        played.append(game)
        _say(_game_line(game, args.games, names))

    async def play() -> None:
        from chess_ai.match import play_match
        from chess_ai.players import close_players, wait_players_closed

        sides = (args.first, args.second)
        made: dict[int, tuple[Player, str]] = {}
        try:
            # The checkpoints first, and all of them chosen before any is loaded: a mistyped run
            # is the likeliest way a match fails, and it then fails before an engine process has
            # been started. Choosing them together means two sides naming the same checkpoint of
            # a run that is still training get the same one, whatever it saves meanwhile.
            chosen = _choose_checkpoints(sides, config)
            for index, side in enumerate(sides):
                if isinstance(side, _ModelSide):
                    made[index] = _model_player(
                        side,
                        *chosen[(side.run, side.checkpoint)],
                        args.temperature,
                        config,
                        seed + index,
                        engines,
                    )
            for index, side in enumerate(sides):
                if isinstance(side, _StockfishSide):
                    made[index] = await _stockfish_player(side, config)
            (first, first_label), (second, second_label) = made[0], made[1]
            names.extend(_side_names((first_label, args.first), (second_label, args.second)))
            _say(
                f"chess-ai: {names[0]} against {names[1]}, {args.games} games from opening set "
                f"{openings.label}, seed {seed}"
            )
            if args.games > distinct_games(openings):
                _say(
                    f"chess-ai: warning: opening set {openings.label} has "
                    f"{_count(len(openings.openings), 'line')}, so from game "
                    f"{distinct_games(openings) + 1} "
                    "on the games start as earlier ones did, and players that always play their "
                    "most likely move will repeat them",
                    err=True,
                )
            try:
                await play_match(
                    first, second, games=args.games, openings=openings, on_game=on_game
                )
            except (StockfishError, TimeoutError, chess.engine.EngineError) as e:
                raise _UserError(
                    f"game {len(played) + 1} could not be finished: {_engine_failure(e)}"
                ) from e
        finally:
            # Waited for as well as closed, however the match ended, so that an engine's process
            # has been reaped before the event loop that would reap it is gone.
            players = [player for player, _ in made.values()]
            close_players(*players)
            with suppress(TimeoutError):
                await asyncio.wait_for(wait_players_closed(*players), _CLOSE_TIMEOUT)

    try:
        asyncio.run(play())
    except KeyboardInterrupt:
        _say(f"chess-ai: interrupted after {len(played)} of {args.games} games", err=True)
        return 130
    finally:
        if played:
            _say(_match_summary(played, names))
            _say(f"chess-ai: games saved in {path}")
    return 0


_CLOSE_TIMEOUT = 5.0
"""How long a finished match waits for its Stockfish processes to exit."""


def _count(number: int, thing: str) -> str:
    return f"{number} {thing}" if number == 1 else f"{number} {thing}s"


def _engine_failure(error: Exception) -> str:
    """What went wrong with Stockfish in the middle of a game, for the person who ran the match.

    The engine hanging and the engine dying are the two that need no mistake by anyone: a
    machine that is also training can starve it of CPU or run out of memory under it.
    """
    import chess.engine

    from chess_ai import stockfish

    if isinstance(error, TimeoutError):
        return (
            f"Stockfish took more than {stockfish.MOVE_GRACE:g} seconds longer than its move time "
            "over a move, and was taken to have hung"
        )
    if isinstance(error, chess.engine.EngineTerminatedError):
        return f"Stockfish stopped: {error}"
    if isinstance(error, chess.engine.EngineError):
        return f"Stockfish failed: {error}"
    return str(error)


def _choose_checkpoints(
    sides: "Sequence[_ModelSide | _StockfishSide]", config: Config
) -> "dict[tuple[str, CheckpointChoice], tuple[RunReader, CheckpointInfo]]":
    """The run and checkpoint each checkpoint side names, each one looked up once."""
    from chess_ai.training.run_store import RunError, choose_checkpoint, open_run

    chosen: dict[tuple[str, CheckpointChoice], tuple[RunReader, CheckpointInfo]] = {}
    for side in sides:
        if not isinstance(side, _ModelSide) or (side.run, side.checkpoint) in chosen:
            continue
        try:
            run = open_run(config.paths.runs, side.run)
            chosen[(side.run, side.checkpoint)] = (run, choose_checkpoint(run, side.checkpoint))
        except RunError as e:
            raise _UserError(e) from e
    return chosen


async def _stockfish_player(side: _StockfishSide, config: Config) -> tuple["Player", str]:
    """Stockfish at the strength ``side`` asks for, and what to call it."""
    from chess_ai.stockfish import StockfishError, start_stockfish

    try:
        player = await start_stockfish(
            config.stockfish.path, elo=side.elo, move_time=side.move_time
        )
    except StockfishError as e:
        raise _UserError(e) from e
    return player, player.name


def _model_player(
    side: _ModelSide,
    run: "RunReader",
    chosen: "CheckpointInfo",
    temperature: float | None,
    config: Config,
    seed: int,
    engines: dict[tuple[str, int], Any],
) -> tuple["Player", str]:
    """A checkpoint, loaded once however many sides play it; see ``web.app._players``.

    Called by its run's title where it has one, as the runs list shows it, since a run's name
    can be as long as a line of the terminal. The PGN names it in full regardless.
    """
    from chess_ai.inference import DEFAULT_TEMPERATURE, InferenceError, ModelPlayer, load_engine
    from chess_ai.training.run_store import RunError

    engine = engines.get((run.name, chosen.step))
    if engine is None:
        try:
            engine = load_engine(
                run.checkpoint_path(chosen),
                device=config.inference.device,
                batch_size=config.inference.batch_size,
            )
        except InferenceError as e:
            raise _UserError(e) from e
        engines[(run.name, chosen.step)] = engine
    # A side that says nothing about how it chooses plays its best move, unless the match
    # as a whole was given a temperature to sample at.
    strategy = side.strategy or ("sample" if temperature is not None else "argmax")
    try:
        title = run.notes.title
    except RunError:
        title = ""  # Only the label is lost; the run itself has been read.
    player = ModelPlayer(
        engine,
        run=run.name,
        checkpoint=chosen.step,
        rating=side.rating,
        strategy=strategy,
        temperature=side.temperature or temperature or DEFAULT_TEMPERATURE,
        seed=seed,
    )
    return player, f"{title or run.name} step {chosen.step}"


def _side_names(
    first: tuple[str, "_ModelSide | _StockfishSide"],
    second: tuple[str, "_ModelSide | _StockfishSide"],
) -> list[str]:
    """What to call the two players in what the match prints, telling them apart if need be."""
    (first_label, first_side), (second_label, second_side) = first, second
    if first_label != second_label:
        return [first_label, second_label]
    # One checkpoint against itself, at two temperatures or two ratings, say: the settings typed
    # for each are what tell them apart, if they differ at all.
    first_settings = first_side.text.partition(",")[2]
    second_settings = second_side.text.partition(",")[2]
    if first_settings == second_settings:
        first_settings, second_settings = "first", "second"
    # A side with no settings of its own is the plain label, which the other no longer is.
    return [
        f"{label} ({settings})" if settings else label
        for label, settings in ((first_label, first_settings), (second_label, second_settings))
    ]


def _game_line(game: "MatchGame", games: int, names: list[str]) -> str:
    """One finished game: who had which colour, and how it went."""
    state = game.game
    over = state.position.game_over
    result = over.result if over is not None else "*"
    how = f"{over.reason.replace('_', ' ')}, " if over is not None else ""
    moves = (len(state.moves) + 1) // 2
    white, black = names if game.first_plays_white else reversed(names)
    return (
        f"game {game.number}/{games}, {game.opening.name}: {white} – {black} {result} "
        f"({how}{moves} moves)"
    )


def _match_summary(played: "list[MatchGame]", names: list[str]) -> str:
    """The score from each side's point of view, overall and by colour."""
    import chess

    from chess_ai.match import score

    lines = []
    for first, name in zip((True, False), names, strict=True):
        overall = score(played, first=first)
        white = score(played, first=first, color=chess.WHITE)
        black = score(played, first=first, color=chess.BLACK)
        fraction = overall.fraction or 0.0
        margin = f" ± {100 * overall.margin:.1f}" if overall.margin is not None else ""
        lines.append(
            f"{name}: {overall.points:g}/{overall.games} = {100 * fraction:.1f}%{margin} "
            f"(+{overall.wins} ={overall.draws} -{overall.losses}); "
            f"as white {white.points:g}/{white.games}, as black {black.points:g}/{black.games}"
        )
    if len(played) >= 2:
        lines.append("± is a rough 95% range for the score, from how the games went")
    return "\n".join(lines)


def _add_ladder_command(commands: argparse._SubParsersAction) -> None:
    """``chess-ai ladder``: a checkpoint against Stockfish at several levels, and its Elo."""
    ladder = commands.add_parser(
        "ladder",
        help="measure a checkpoint's Elo against Stockfish at several strengths",
        description=(
            "Play a checkpoint against Stockfish at each level of a ladder in turn, as matches\n"
            "from the lines of an opening set, and estimate its Elo, with a 95% range, from how\n"
            "it scored. A checkpoint that loses every game, or whose estimate is below the\n"
            "lowest level, is reported as below it rather than given a number; likewise above\n"
            "the highest. The result and the games are saved in the run directory, under\n"
            "evaluations/, replacing any earlier ladder of the same checkpoint.\n"
            "\n"
            "The levels, games, rating, move time and opening set default to the [ladder]\n"
            "section of the configuration. The Elo is on Stockfish's scale at the move time\n"
            "played, not a Lichess rating: compare ladders played at the same move time."
        ),
        epilog=(
            "examples:\n"
            "  chess-ai ladder resnet10x256\n"
            "  chess-ai ladder resnet10x256@48000 --levels 1350,1500,1700 --games 10\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ladder.add_argument(
        "checkpoint",
        type=_ladder_checkpoint,
        metavar="RUN[@CHECKPOINT]",
        help="the checkpoint: best (the default), latest or a step",
    )
    ladder.add_argument(
        "--levels",
        type=_levels,
        metavar="ELO,ELO,...",
        help="Stockfish's levels, which are played weakest first",
    )
    ladder.add_argument(
        "--games", type=_positive, metavar="N", help="how many games to play against each level"
    )
    ladder.add_argument(
        "--rating",
        type=lambda value: _number(int, value, "a rating", value, 0, _MAX_RATING),
        metavar="R",
        help="the rating the checkpoint is asked to play like",
    )
    ladder.add_argument(
        "--move-time",
        type=_move_time,
        metavar="SECONDS",
        help="how long Stockfish thinks about each move",
    )
    ladder.add_argument("--openings", metavar="SET", help="the opening set the games start from")
    ladder.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        help="where the checkpoint's network runs; auto is the GPU if there is one",
    )
    ladder.set_defaults(handler=_ladder)


def _ladder_checkpoint(text: str) -> _ModelSide:
    if "," in text:
        raise argparse.ArgumentTypeError(
            f"{text!r}: a ladder's checkpoint takes no settings; see --rating"
        )
    return _model_side(text, text, {})


def _levels(text: str) -> tuple[int, ...]:
    from pydantic import ValidationError

    from chess_ai.config import LadderConfig

    try:
        return LadderConfig(levels=text).levels  # type: ignore[arg-type]
    except ValidationError as e:
        raise argparse.ArgumentTypeError(
            f"{text!r}: {'; '.join(error['msg'] for error in e.errors())}"
        ) from e


def _move_time(value: str) -> float:
    from chess_ai.stockfish import MAX_MOVE_TIME, MIN_MOVE_TIME

    return _number(float, value, "a move time", value, MIN_MOVE_TIME, MAX_MOVE_TIME)


class _LevelNotStarted(Exception):
    """Stockfish could not be started for a level of a ladder, which ``__cause__`` says why."""

    def __init__(self, level: int, error: Exception) -> None:
        super().__init__(level, error)
        self.level = level


def _ladder(config: Config, args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    import chess.engine

    from chess_ai.ladder import (
        games_path,
        ladder_result,
        play_ladder,
        result_path,
        save_ladder,
        start_games_file,
    )
    from chess_ai.match import append_game, distinct_games
    from chess_ai.openings import OpeningSetError, load_opening_set
    from chess_ai.stockfish import StockfishError, start_stockfish
    from chess_ai.training.run_store import RunError, code_version

    settings = config.ladder
    levels = args.levels or settings.levels
    games = args.games or settings.games_per_level
    rating = args.rating if args.rating is not None else settings.rating
    move_time = args.move_time if args.move_time is not None else settings.move_time
    try:
        openings = load_opening_set(args.openings or settings.openings)
    except OpeningSetError as e:
        raise _UserError(e) from e
    side = dataclasses.replace(args.checkpoint, rating=rating)
    ((run, chosen),) = _choose_checkpoints([side], config).values()
    inference = config.inference.model_copy(update={"device": args.device or settings.device})
    model, label = _model_player(
        side, run, chosen, None, config.model_copy(update={"inference": inference}), 0, {}
    )
    try:
        partial = start_games_file(run, chosen.step)
    except OSError as e:
        raise _UserError(f"cannot write the ladder's games in {run.directory}: {e}") from e
    played = 0
    total = games * len(levels)

    def on_game(stockfish, game: "MatchGame") -> None:
        nonlocal played
        try:
            append_game(partial, game)
        except OSError as e:
            raise _UserError(f"could not save a game to {partial}: {e.strerror}") from e
        played += 1
        line = _game_line(game, games, [label, f"Stockfish {stockfish.elo}"])
        _say(f"{played}/{total} {line}")

    def games_note() -> str:
        """Where the games played so far are, for a ladder that stopped before its end.

        A ladder that played none deletes its empty games file and has nothing to tell. The
        directory it was in stays, empty or not: another ladder or save of the same checkpoint
        may have just made sure it is there, and be about to write into it.
        """
        if played == 0:
            partial.unlink(missing_ok=True)
            return ""
        # A save that failed may have done so after the games were moved into place.
        place = partial if partial.exists() else games_path(run, chosen.step)
        return f"; the {_count(played, 'game')} played {'is' if played == 1 else 'are'} in {place}"

    async def opponent(level: int) -> "Player":
        try:
            stockfish = await start_stockfish(config.stockfish.path, elo=level, move_time=move_time)
        except StockfishError as e:
            raise _LevelNotStarted(level, e) from e
        if stockfish.stockfish.clamped:
            _say(
                f"chess-ai: warning: Stockfish plays from {stockfish.stockfish.min_elo} to "
                f"{stockfish.stockfish.max_elo}, so level {level} is played at "
                f"{stockfish.stockfish.elo}",
                err=True,
            )
        return stockfish

    _say(
        f"chess-ai: {label} at rating {rating} against Stockfish at "
        f"{', '.join(str(level) for level in levels)}, {_count(games, 'game')} each at "
        f"{move_time:g} s a move, from opening set {openings.label}"
    )
    if games > distinct_games(openings):
        _say(
            f"chess-ai: warning: opening set {openings.label} has "
            f"{_count(len(openings.openings), 'line')}, so from game "
            f"{distinct_games(openings) + 1} of each level on the games start as earlier ones "
            "did, and the checkpoint, which plays its most likely move, will repeat them",
            err=True,
        )
    started = datetime.now(UTC)
    try:
        levels_played = asyncio.run(
            play_ladder(
                model, levels, games=games, openings=openings, opponent=opponent, on_game=on_game
            )
        )
    except KeyboardInterrupt:
        _say(
            f"chess-ai: interrupted after {played} of {total} games, so no result was saved"
            f"{games_note()}",
            err=True,
        )
        return 130
    except _LevelNotStarted as e:
        raise _UserError(
            f"Stockfish could not be started for level {e.level}, so no result was saved: "
            f"{e.__cause__}{games_note()}"
        ) from e
    except (StockfishError, TimeoutError, chess.engine.EngineError) as e:
        raise _UserError(
            f"game {played + 1} could not be finished: {_engine_failure(e)}{games_note()}"
        ) from e
    result = ladder_result(
        model.model,
        levels_played,
        openings=openings,
        games_per_level=games,
        started=started,
        finished=datetime.now(UTC),
        code_version=code_version(),
    )
    for level in result.levels:
        fraction = level.points / level.games
        _say(
            f"Stockfish {level.elo}: {level.points:g}/{level.games} = {100 * fraction:.1f}% "
            f"(+{level.wins} ={level.draws} -{level.losses})"
        )
    _say(
        f"Elo: {result.estimate.describe()}, a {100 * result.estimate.confidence:g}% range, "
        f"on Stockfish's scale at {move_time:g} s a move"
    )
    try:
        save_ladder(run, result, partial)
    except RunError as e:
        raise _UserError(f"{e}, so no result was saved{games_note()}") from e
    _say(f"chess-ai: result saved in {result_path(run, chosen.step)}")
    return 0


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


_FILTER_OPTIONS = (
    "min_rating",
    "max_rating",
    "unknown_rating_passes",
    "time_controls",
    "exclude_terminations",
    "from_date",
    "until",
    "min_clock",
    "sample",
    "max_games",
)
"""The ``dataset build`` options that are filters, by the name they have in both places."""


def _names(text: str) -> list[str]:
    """A comma-separated list of names, as an option takes one."""
    return [name.strip().lower() for name in text.split(",") if name.strip()]


def _build_settings(args: argparse.Namespace):
    """What ``dataset build`` was asked for: the definition file's, with the options over it.

    Returns the sources, the validation fraction, the rating source and the filters.
    """
    from pydantic import ValidationError

    from chess_ai.dataset import DEFAULT_VALIDATION_FRACTION, DatasetError, RatingSource
    from chess_ai.dataset.definition import DatasetDefinition, load_definition
    from chess_ai.dataset.manifest import Filters

    try:
        definition = (
            load_definition(args.definition) if args.definition is not None else DatasetDefinition()
        )
    except DatasetError as e:
        raise _UserError(e) from e
    sources = args.sources or definition.sources
    if not sources:
        raise _UserError(
            "nothing to build from: name PGN files, or a --definition file that has sources"
        )
    fraction = args.validation_fraction
    if fraction is None:
        fraction = definition.validation_fraction
    if fraction is None:
        fraction = DEFAULT_VALIDATION_FRACTION
    rating_source = args.rating_source or definition.rating_source or AUTO
    if rating_source != AUTO and rating_source.upper() not in RatingSource.__members__:
        raise _UserError(f"no such rating source {rating_source!r}")
    given = {
        name: getattr(args, name) for name in _FILTER_OPTIONS if getattr(args, name) is not None
    }
    try:
        # By field name, which is what the options are called: "from" is only the file's alias.
        filters = Filters.model_validate(
            {name: getattr(definition.filters, name) for name in Filters.model_fields} | given
        )
    except ValidationError as e:
        messages = "; ".join(error["msg"].removeprefix("Value error, ") for error in e.errors())
        raise _UserError(f"invalid filters: {messages}") from e
    source = None if rating_source == AUTO else RatingSource[rating_source.upper()]
    return sources, fraction, source, filters


def _interrupted_options(parser: argparse.ArgumentParser, doing: str) -> None:
    """The options that say what to do about an interrupted build or append."""
    what = parser.add_mutually_exclusive_group()
    what.add_argument(
        "--resume",
        action="store_true",
        help=f"carry on an interrupted {doing} from its last checkpoint, with the sources and "
        "settings it was started with",
    )
    what.add_argument(
        "--discard-interrupted",
        action="store_true",
        help=f"throw away an interrupted {doing} of this dataset without asking, and start this "
        "one" + (" from the last version" if doing == "append" else ""),
    )


def _confirm() -> Callable[[str], bool] | None:
    """A question to put to the person running the command, if there is one at a terminal.

    ``None`` anywhere else, so that a script or a cron job is refused and told which flag says
    what it meant rather than left waiting on a question nobody will answer.
    """
    try:
        if not (sys.stdin.isatty() and sys.stderr.isatty()):
            return None
    except (AttributeError, ValueError):
        return None

    def ask(question: str) -> bool:
        print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
        try:
            answer = sys.stdin.readline()
        except (OSError, ValueError):
            return False
        return answer.strip().lower() in ("y", "yes")

    return ask


def _build_dataset(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset import (
        DEFAULT_VALIDATION_FRACTION,
        DatasetError,
        ProgressPrinter,
        build_dataset,
    )

    if args.resume:
        given = [
            option
            for option, value in (
                ("PGN files", args.sources),
                ("--definition", args.definition),
                ("--validation-fraction", args.validation_fraction),
                ("--rating-source", args.rating_source),
                *((f"--{name.replace('_', '-')}", getattr(args, name)) for name in _FILTER_OPTIONS),
            )
            # Every one of these is None, or no files, unless it was given: a filter of 0 is
            # given as much as one of 1500 is, and so is --no-unknown-rating-passes.
            if value is not None and value != []
        ]
        if given:
            raise _UserError(
                "--resume carries on with the sources and settings the interrupted build was "
                f"started with, so it takes no {', '.join(given)}"
            )
        sources, fraction, source, filters = [], DEFAULT_VALIDATION_FRACTION, None, None
    else:
        sources, fraction, source, filters = _build_settings(args)
    printer = ProgressPrinter()
    try:
        manifest = build_dataset(
            args.name,
            sources,
            data_dir=config.paths.data,
            validation_fraction=fraction,
            rating_source=source,
            progress=printer,
            overwrite=args.overwrite,
            workers=args.workers,
            filters=filters,
            allow_repeat=args.allow_repeat,
            resume=args.resume,
            discard_interrupted=args.discard_interrupted,
            confirm=_confirm(),
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
    return _report_built(config, manifest)


def _append_dataset(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset import DatasetError, ProgressPrinter
    from chess_ai.dataset.builder import append_dataset

    if args.resume and (args.sources or args.max_games is not None or args.allow_repeat):
        raise _UserError(
            "--resume carries on with the sources and maximum the interrupted append was started "
            "with, so it takes no PGN files, --max-games or --allow-repeat"
        )
    if not args.resume and not args.sources:
        raise _UserError("name the PGN files to append, or --resume an interrupted append")
    printer = ProgressPrinter()
    try:
        manifest = append_dataset(
            args.name,
            args.sources,
            data_dir=config.paths.data,
            max_games=args.max_games,
            progress=printer,
            workers=args.workers,
            allow_repeat=args.allow_repeat,
            discard_interrupted=args.discard_interrupted,
            resume=args.resume,
            confirm=_confirm(),
        )
    except DatasetError as e:
        raise _UserError(e) from e
    except OSError as e:
        raise _UserError(f"could not append to dataset {args.name!r}: {e.strerror or e}") from e
    finally:
        printer.finish()
    return _report_built(config, manifest)


def _report_built(config: Config, manifest) -> int:
    """Say what a build or an append made, and answer for sources it could not read whole.

    For an append, what is said is the version it added, then where the dataset now stands.
    """
    from chess_ai.dataset import dataset_path

    latest = manifest.versions[-1]
    lost = [source.path for source in latest.sources if source.went_away]
    # Only the sources that were there throughout and whose contents gave up: a source that went
    # away took an unknown number of games with it, and saying "not read whole" of that on stdout
    # while the warning says the rest of it understates it in the line a log gets read for.
    unread = [
        source.path
        for source in latest.sources
        if source.error is not None and not source.went_away
    ]
    skipped = sum(latest.skipped.values())
    filtered = sum(latest.filtered.values())
    added = (
        f"version {latest.version} added {latest.games:,} games, {latest.positions:,} positions "
        f"({manifest.games:,} games, {manifest.positions:,} positions in all), "
        if latest.version > 1
        else f"{manifest.games:,} games, {manifest.positions:,} positions, "
    )
    _say(
        f"chess-ai: dataset {manifest.name} in {dataset_path(config.paths.data, manifest.name)}: "
        + added
        + (
            f"{manifest.trained_on:,} of them trained on, "
            if manifest.trained_on != manifest.positions
            else ""
        )
        + f"{skipped:,} skipped"
        + (f", {filtered:,} filtered out" if filtered else "")
        + (" (stopped at --max-games)" if latest.reached_max_games else "")
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


def _month(text: str) -> str:
    from chess_ai.dataset.download import parse_month

    try:
        return parse_month(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _download_dumps(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset.download import (
        DownloadError,
        DownloadPrinter,
        download_months,
        downloads_dir,
    )

    printer = DownloadPrinter()
    try:
        paths = download_months(
            args.months, directory=downloads_dir(config.paths.data), progress=printer
        )
    except DownloadError as e:
        raise _UserError(e) from e
    finally:
        printer.finish()
    for path in paths:
        _say(f"chess-ai: {path}")
    return 0


def _dataset_stats(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset import DatasetError, ManifestError, open_dataset, summarize

    try:
        dataset = open_dataset(args.name, data_dir=config.paths.data, version=args.version)
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
