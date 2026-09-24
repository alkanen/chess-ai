"""The ``chess-ai`` command-line tool."""

import argparse
from collections.abc import Sequence
from pathlib import Path

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
        parser.exit(2, f"{parser.prog}: error: {e}\n")
        raise  # parser.exit has already left; this is only here to say so.


class _UserError(Exception):
    """Something the person running the command can fix, reported without a traceback."""


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
    return parser


def _add_dataset_commands(commands: argparse._SubParsersAction) -> None:
    """``chess-ai dataset ...``: making datasets out of PGN files, and looking at them."""
    from chess_ai.dataset import DEFAULT_VALIDATION_FRACTION, RatingSource

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


def _serve(config: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from chess_ai.web import create_app

    server = config.server
    print(
        f"chess-ai: serving on http://{server.host}:{server.port}{server.path_prefix}/", flush=True
    )
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
    try:
        manifest = build_dataset(
            args.name,
            args.sources,
            data_dir=config.paths.data,
            validation_fraction=args.validation_fraction,
            rating_source=source,
            progress=ProgressPrinter(),
            overwrite=args.overwrite,
        )
    except DatasetError as e:
        raise _UserError(e) from e
    print(
        f"chess-ai: dataset {manifest.name} in {dataset_path(config.paths.data, manifest.name)}: "
        f"{manifest.games:,} games, {manifest.positions:,} positions, "
        f"{manifest.games_skipped:,} skipped",
        flush=True,
    )
    return 0


def _dataset_stats(config: Config, args: argparse.Namespace) -> int:
    from chess_ai.dataset import DatasetError, ManifestError, open_dataset, summarize

    try:
        dataset = open_dataset(args.name, data_dir=config.paths.data)
    except (DatasetError, ManifestError) as e:
        raise _UserError(_with_available(e, config.paths.data)) from e
    print(summarize(dataset.manifest), end="")
    return 0


def _with_available(error: Exception, data_dir: Path) -> str:
    """``error``, plus the datasets there actually are, which is usually the next question."""
    from chess_ai.dataset import list_datasets

    names = list_datasets(data_dir)
    if not names:
        return f"{error} (no datasets in {data_dir}; build one with 'chess-ai dataset build')"
    return f"{error} (datasets in {data_dir}: {', '.join(names)})"
