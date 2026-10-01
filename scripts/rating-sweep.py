#!/usr/bin/env python3
"""Ask one checkpoint the same positions at several ratings, and print what changes.

The rating a model player is given reaches the network as two numbers in the global feature
vector, and nothing makes the network use them: a run trained on games from one narrow band of
ratings has seen that feature barely move, and will have learned nothing from it. This says
whether a given checkpoint learned anything — fix a position, sweep the rating, and watch the
distribution over legal moves.

What to look for:

- **The top move changing** as the rating rises is the clearest sign the conditioning works.
- **The spread**, printed per position, is how far the distribution moves across the whole
  sweep, as total variation distance: 0% is a network ignoring the rating outright, and a few
  percent is a network that has learned something small.
- **The range the run trained on**, which ``chess-ai dataset stats <name>`` reports. Sweeping
  outside it measures the wrong thing: a run trained on 2500-and-up moves a lot when asked for
  800, but that is the network extrapolating off the end of its data rather than imitating an
  800, and it says nothing about whether the conditioning works.

Run it against a run in the configured runs directory::

    uv run scripts/rating-sweep.py mlp-baseline
    uv run scripts/rating-sweep.py mlp-baseline --checkpoint 12000 --ratings 800,1600,2400
    uv run scripts/rating-sweep.py mlp-baseline --fen "8/8/8/3k4/8/3K4/8/7R w - - 0 1"
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import chess

from chess_ai.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_FILE, Config, ConfigError, load_config
from chess_ai.inference import Evaluation, InferenceError, load_engine
from chess_ai.training.run_store import CheckpointChoice, RunError, choose_checkpoint, open_run

POSITIONS: dict[str, str] = {
    "the opening position": chess.STARTING_FEN,
    "a quiet Sicilian middlegame": (
        "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9"
    ),
    "a pawn to grab, or not": "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 b - - 0 9",
    "a king and rook endgame": "8/8/4k3/8/8/4K3/8/4R3 w - - 0 1",
}
"""Positions where a weaker and a stronger player plausibly differ, for a sweep with no ``--fen``.

Chosen so that the answer is not forced: an endgame with one good plan, a middlegame with
several reasonable moves, and a position whose tempting capture is a mistake. A position with
only one legal move would say nothing about anything.
"""

RATINGS: tuple[int, ...] = (800, 1200, 1600, 2000, 2400, 2800)
"""The sweep, unless ``--ratings`` says otherwise: the range a Lichess dump covers."""

TOP_MOVES = 3
"""How many moves of each distribution to print. The rest are in the spread."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        engine = _engine(config, args)
    except (ConfigError, RunError, InferenceError) as e:
        print(f"{parser.prog}: error: {e}", file=sys.stderr)
        return 2

    boards = [(fen, chess.Board(fen)) for fen in (args.fen or list(POSITIONS.values()))]
    names = {fen: name for name, fen in POSITIONS.items()}
    spreads: list[float | None] = []
    for fen, board in boards:
        print()
        print(names.get(fen, "a position of your own"))
        print(f"  {fen}")
        spreads.append(_sweep(engine, board, args.ratings, top=args.top))
    print()
    print(_verdict(spreads))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rating-sweep", description=__doc__.splitlines()[0], allow_abbrev=False
    )
    parser.add_argument("run", help="the training run to load a checkpoint from")
    parser.add_argument(
        "--checkpoint",
        default="best",
        metavar="WHICH",
        help='"best" (the default), "latest", or the step number of one checkpoint',
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"configuration file (default: ${CONFIG_PATH_ENV}, else ./{DEFAULT_CONFIG_FILE} "
        "if it exists), which is what says where runs are kept",
    )
    parser.add_argument(
        "--ratings",
        type=_ratings,
        default=RATINGS,
        metavar="N,N,…",
        help=f"the ratings to ask at (default: {','.join(str(r) for r in RATINGS)})",
    )
    parser.add_argument(
        "--fen",
        action="append",
        metavar="FEN",
        help="a position to ask about, instead of the built-in set; repeatable",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_MOVES,
        metavar="N",
        help=f"how many moves of each distribution to print (default: {TOP_MOVES})",
    )
    parser.add_argument(
        "--device",
        help="where to run the network: cpu, cuda, or auto (default: what the config says)",
    )
    return parser


def _ratings(text: str) -> tuple[int, ...]:
    """``--ratings`` as the numbers it names, or an error argparse can report."""
    try:
        ratings = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of ratings: {e}"
        ) from e
    if not ratings:
        raise argparse.ArgumentTypeError("name at least one rating")
    return ratings


def _engine(config: Config, args: argparse.Namespace):
    """The checkpoint named on the command line, loaded and ready to be asked."""
    run = open_run(config.paths.runs, args.run)
    chosen = choose_checkpoint(run, _checkpoint(args.checkpoint))
    engine = load_engine(
        run.checkpoint_path(chosen),
        device=args.device if args.device else config.inference.device,
        batch_size=config.inference.batch_size,
    )
    print(f"{run.name} step {engine.step} on {engine.device}")
    return engine


def _checkpoint(choice: str) -> CheckpointChoice:
    """What ``--checkpoint`` names: one of the two words, or a step number."""
    if choice in ("best", "latest"):
        return choice
    try:
        return int(choice)
    except ValueError as e:
        raise RunError(
            f'--checkpoint takes "best", "latest" or a step number, not {choice!r}'
        ) from e


def _sweep(engine, board: chess.Board, ratings: Sequence[int], *, top: int) -> float | None:
    """Print one position at every rating, and return how far the distribution moved.

    The unrated row is printed too and left out of the spread: leaving the rating empty sets
    the two unknown flags, which is a larger change to the input than any rating is, so a
    difference there says nothing about whether the rating itself is doing any work.
    """
    distributions = {}
    for rating in (None, *ratings):
        evaluation = engine.evaluate(board, mover_rating=rating, opponent_rating=rating)
        distributions[rating] = _by_uci(evaluation)
        moves = ", ".join(
            f"{board.san(move)} {probability:.1%}" for move, probability in evaluation.top(top)
        )
        print(f"  {'unrated' if rating is None else rating:>7}  {moves}")
    if len(ratings) < 2:
        # One rating is a reading, not a sweep: there is nothing for it to differ from.
        return None
    spread = _spread([distributions[rating] for rating in ratings])
    played = {next(iter(distributions[rating])) for rating in ratings}
    print(f"  {'spread':>7}  {spread:.1%}{'' if len(played) > 1 else ', the same move throughout'}")
    return spread


def _by_uci(evaluation: Evaluation) -> dict[str, float]:
    """One evaluation's probabilities by move, best first, which is how two are compared."""
    return {move.uci(): probability for move, probability in evaluation.moves}


def _spread(distributions: Sequence[dict[str, float]]) -> float:
    """How far apart the two furthest-apart distributions of a sweep are.

    Total variation distance, which is the share of the probability mass that would have to be
    moved to turn one distribution into the other: 0 for two that agree everywhere, 1 for two
    with nothing in common. Every pair is measured rather than just the ends of the sweep,
    because nothing says the effect of a rating has to be monotonic.
    """
    return max(
        (
            sum(abs(one.get(move, 0.0) - other.get(move, 0.0)) for move in one | other) / 2
            for index, one in enumerate(distributions)
            for other in distributions[index + 1 :]
        ),
        default=0.0,
    )


def _verdict(spreads: Sequence[float | None]) -> str:
    """One line on what the sweep showed, which is the thing worth acting on."""
    measured = [spread for spread in spreads if spread is not None]
    if not measured:
        return "Nothing to compare: pass two or more ratings to --ratings to sweep."
    worst = max(measured)
    if worst < 0.01:
        return (
            f"The rating moves nothing here (at most {worst:.1%}). Either this run trained on "
            "games from one narrow band of ratings, or the network has not learned to use the "
            "feature. Check the range with: chess-ai dataset stats <name>"
        )
    if worst < 0.10:
        return (
            f"The rating moves the distribution a little (up to {worst:.1%}). Worth comparing "
            "against a later checkpoint before reading much into it."
        )
    return (
        f"The rating changes what this checkpoint plays (up to {worst:.1%}). Check the range "
        "the run trained on (chess-ai dataset stats <name>) and sweep inside it before "
        "believing this: a rating the run never saw moves the feature off the end of its "
        "training data, and a network extrapolating is not a network imitating."
    )


if __name__ == "__main__":
    raise SystemExit(main())
