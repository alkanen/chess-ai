"""The ``chess-ai`` command-line tool."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from chess_ai.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_FILE, Config, ConfigError, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        parser.exit(2, f"{parser.prog}: error: {e}\n")
    return args.handler(config)


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

    return parser


def _serve(config: Config) -> int:
    import uvicorn

    from chess_ai.web import create_app

    server = config.server
    print(
        f"chess-ai: serving on http://{server.host}:{server.port}{server.path_prefix}/", flush=True
    )
    uvicorn.run(create_app(config), host=server.host, port=server.port)
    return 0
