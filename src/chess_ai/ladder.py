"""The Stockfish ladder: a checkpoint against Stockfish at several strengths, and its Elo.

Each level of the ladder is a match, played by the match runner like any other: the same
opening set, the colours swapped every game. What the ladder adds is the opponents' ratings,
which Stockfish's calibrated strength limit gives it, and so a rating for the checkpoint from
how it scored against them, by :mod:`chess_ai.rating`.

The rating is on Stockfish's scale, not Lichess's. Stockfish's levels are anchored to an
engine rating list at a few seconds a move, and a ladder played at a fraction of that has
Stockfish playing below its levels, so two ladders are only comparable when they were played
at the same move time — which is why every result says what it was.

What a ladder finds out is kept in the run directory, beside what any other evaluation finds
out about the same checkpoint: ``<suite>.json`` with the results and the estimate, and
``<suite>.pgn`` with the games. The games are written as they end to a ``.partial`` file of the
ladder's own next to where they go, and moved into place only once the ladder is over, so that
the games beside a result are always the games it was worked out from.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from chess_ai.match import MatchGame, play_match, score
from chess_ai.openings import OpeningSet
from chess_ai.players import (
    ModelDescription,
    Player,
    StockfishBackedPlayer,
    StockfishDescription,
    close_players,
    wait_players_closed,
)
from chess_ai.rating import DEFAULT_CONFIDENCE, Beyond, Estimate, Result, estimate_rating
from chess_ai.training.run_store import RunReader, save_evaluation

SUITE: Final = "stockfish-ladder"
"""What the ladder is called among the evaluation suites, and what its files are named."""

EVENT: Final = "chess-ai ladder"
"""The ``Event`` of every ladder game's PGN, telling them from the games of a plain match."""

FORMAT_VERSION: Final = 1

PARTIAL_SUFFIX: Final = ".partial"
"""What the name of a games file ends in while its ladder is still playing."""

CLOSE_TIMEOUT: Final = 5.0
"""How long a level waits for its Stockfish process to exit before going on to the next."""


class LadderLevel(BaseModel):
    """How the checkpoint did against one level of the ladder."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    elo: int
    """The strength Stockfish played at, which is the rating the results are counted against."""
    requested_elo: int
    """The level asked for, which is :attr:`elo` unless Stockfish could not play at it."""
    move_time: float
    """Seconds Stockfish had for each move."""
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws


class LadderEstimate(BaseModel):
    """The checkpoint's rating, as :class:`~chess_ai.rating.Estimate` gives it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    rating: float | None
    """The most likely rating, or ``None`` when the checkpoint is :attr:`beyond` the ladder."""
    low: float | None
    high: float | None
    beyond: Beyond | None
    """Set when the checkpoint is below the ladder's lowest level or above its highest."""
    floor: float
    """The lowest level's rating."""
    ceiling: float
    """The highest level's rating."""
    confidence: float
    """How likely the range from :attr:`low` to :attr:`high` is to hold the true rating."""
    games: int
    points: float

    @classmethod
    def of(cls, estimate: Estimate) -> "LadderEstimate":
        return cls(**asdict(estimate))

    def describe(self) -> str:
        """Such as ``1523 (1460 to 1590)`` or ``below 1350, at most 1420``."""
        return Estimate(**self.model_dump()).describe()


class LadderResult(BaseModel):
    """Everything a ladder found out about a checkpoint, as ``stockfish-ladder.json`` holds it.

    The form every evaluation suite's result will take: which suite, which checkpoint and how it
    was played, when, by what code, and then what the suite has to say.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    format_version: Literal[1] = FORMAT_VERSION
    suite: Literal["stockfish-ladder"] = SUITE
    model: ModelDescription
    """The run, the checkpoint, and the rating it was asked to play like."""
    started: datetime
    finished: datetime
    code_version: str
    openings: str
    """The opening set the games started from, and its version."""
    games_per_level: int
    levels: list[LadderLevel]
    estimate: LadderEstimate


@dataclass(frozen=True)
class PlayedLevel:
    """One level of a ladder that has been played: who the opponent was, and the games."""

    stockfish: StockfishDescription
    games: list[MatchGame]


Opponent = Callable[[int], Awaitable[Player]]
"""Makes the Stockfish player for a level: given an Elo, a player that is
:class:`~chess_ai.players.StockfishBackedPlayer`, so that it can say what it plays at."""


async def play_ladder(
    model: Player,
    levels: Sequence[int],
    *,
    games: int,
    openings: OpeningSet,
    opponent: Opponent,
    on_game: Callable[[StockfishDescription, MatchGame], None] | None = None,
) -> list[PlayedLevel]:
    """Play ``model`` against Stockfish at each of ``levels`` in turn, ``games`` games each.

    Each level's opponent is made by ``opponent`` when its turn comes, and closed once its games
    are over, so that at most one engine runs at a time. ``on_game`` is handed each game as it
    ends, with the level it was played at. ``model`` is the first player of every match.

    Raises:
        ValueError: there are no levels, or ``games`` is less than one.
        TypeError: ``opponent`` made a player that cannot say how strong it plays.
        Whatever :func:`~chess_ai.match.play_match` or ``opponent`` raise.
    """
    if not levels:
        raise ValueError("a ladder needs at least one level")
    played = []
    for level in levels:
        stockfish = await opponent(level)
        try:
            if not isinstance(stockfish, StockfishBackedPlayer):
                raise TypeError(f"{stockfish.name} cannot be a level of a ladder")
            description = stockfish.stockfish

            def handed_over(game: MatchGame, description=description) -> None:
                if on_game is not None:
                    on_game(description, game)

            level_games = await play_match(
                model,
                stockfish,
                games=games,
                openings=openings,
                on_game=handed_over,
                event=EVENT,
            )
        finally:
            close_players(stockfish)
            with suppress(TimeoutError):
                await asyncio.wait_for(wait_players_closed(stockfish), CLOSE_TIMEOUT)
        played.append(PlayedLevel(stockfish=description, games=level_games))
    return played


def ladder_result(
    model: ModelDescription,
    played: Sequence[PlayedLevel],
    *,
    openings: OpeningSet,
    games_per_level: int,
    started: datetime,
    finished: datetime,
    code_version: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> LadderResult:
    """What ``played`` comes to: the score at each level, and the rating they add up to."""
    levels = []
    for level in played:
        counted = score(level.games)
        levels.append(
            LadderLevel(
                elo=level.stockfish.elo,
                requested_elo=level.stockfish.requested_elo,
                move_time=level.stockfish.move_time,
                wins=counted.wins,
                draws=counted.draws,
                losses=counted.losses,
            )
        )
    estimate = estimate_rating(
        (Result(opponent=level.elo, points=level.points, games=level.games) for level in levels),
        confidence=confidence,
    )
    return LadderResult(
        model=model,
        started=started,
        finished=finished,
        code_version=code_version,
        openings=openings.label,
        games_per_level=games_per_level,
        levels=levels,
        estimate=LadderEstimate.of(estimate),
    )


def result_path(run: RunReader, step: int) -> Path:
    """Where the ladder's result for the checkpoint from ``step`` is kept."""
    return run.evaluation_directory(step) / f"{SUITE}.json"


def games_path(run: RunReader, step: int) -> Path:
    """Where the games the result was worked out from are kept."""
    return run.evaluation_directory(step) / f"{SUITE}.pgn"


def start_games_file(run: RunReader, step: int) -> Path:
    """A new, empty file for the games of a ladder of the checkpoint from ``step``.

    One of its own for every ladder, so that two ladders of one checkpoint played at once do
    not write into each other's games; the last to finish is the result that stays. The games
    are added with :func:`~chess_ai.match.append_game` as they end. A ladder that is stopped
    leaves its file behind, for whoever stopped it to look at or delete.
    """
    directory = run.evaluation_directory(step)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{SUITE}.{uuid4().hex[:8]}.pgn{PARTIAL_SUFFIX}"
    path.touch(exist_ok=False)
    return path


def save_ladder(run: RunReader, result: LadderResult, partial: Path) -> None:
    """Keep ``result`` and its games, from ``partial``, for the checkpoint it is about.

    Saved as one set by :func:`~chess_ai.training.run_store.save_evaluation`, so that a result
    has its own games beside it, whichever other ladder of the checkpoint finishes alongside.

    Raises:
        RunError: the run directory cannot be written. The games are still in ``partial``,
            unless the failure came after they had been moved to :func:`games_path`.
    """
    step = result.model.checkpoint
    save_evaluation(
        run.evaluation_directory(step),
        {f"{SUITE}.json": result.model_dump_json(indent=2) + "\n"},
        {f"{SUITE}.pgn": partial},
    )
