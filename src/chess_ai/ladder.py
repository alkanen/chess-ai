"""The Stockfish ladder: a checkpoint against Stockfish at several strengths, and its Elo.

Each level of the ladder is a match, played by the match runner like any other: the same
opening set, the colours swapped every game. What the ladder adds is the opponents' ratings,
which Stockfish's calibrated strength limit gives it, and so a rating for the checkpoint from
how it scored against them, by :mod:`chess_ai.rating`.

The rating is on Stockfish's scale, not Lichess's. Stockfish's levels are anchored to an
engine rating list at a few seconds a move, and a ladder played at a fraction of that has
Stockfish playing below its levels, so two ladders are only comparable when they were played
at the same move time — which is why every result says what it was. Or at the same number of
nodes a move: what Stockfish searches in a tenth of a second depends on the machine and on how
busy it is, and what it makes of a number of nodes does not, so a ladder held to nodes compares
with any other held to as many.

Most of a ladder is Stockfish thinking, on one thread of a CPU that has many, so a ladder can
play several games at once, each against a Stockfish of its own.

What a ladder finds out is kept in the run directory, beside what any other evaluation finds
out about the same checkpoint: ``<suite>.json`` with the results and the estimate, and
``<suite>.pgn`` with the games. The games are written as they end to a ``.partial`` file of the
ladder's own next to where they go, and moved into place only once the ladder is over, so that
the games beside a result are always the games it was worked out from.

A ladder is played by ``chess-ai ladder`` on demand, and by the evaluator as a suite of its own
(:class:`LadderSuite`) for the runs that ask for it. A ladder takes minutes and runs save a
checkpoint far more often, so the evaluator ladders a run's checkpoints only every so many
steps, and a finished run's last; its games can be watched as they are played, as the sample
games can.
"""

import asyncio
import gc
import logging
import os
import threading
import traceback
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from chess_ai.config import LadderConfig
from chess_ai.evaluator import until_stopped
from chess_ai.game_session import GameSession
from chess_ai.match import (
    GameLines,
    LinesMatch,
    MatchGame,
    append_game,
    play_game,
    replaying,
    score,
)
from chess_ai.openings import OpeningSet
from chess_ai.output import Say
from chess_ai.output import say as said_on_stdout
from chess_ai.players import (
    ModelDescription,
    Player,
    StockfishBackedPlayer,
    StockfishDescription,
    close_and_wait,
    describe_limit,
)
from chess_ai.position_view import GameOverReason
from chess_ai.rating import DEFAULT_CONFIDENCE, Beyond, Estimate, Result, estimate_rating
from chess_ai.sample_games import (
    LiveGame,
    check_playable,
    clear_live_game,
    show_live_game,
    watched,
)
from chess_ai.training.run_store import (
    CheckpointChanged,
    CheckpointInfo,
    RunError,
    RunReader,
    RunStatus,
    code_version,
    created_with,
    prune_released,
    save_evaluation,
    start_partial_file,
    still_evaluated,
)

if TYPE_CHECKING:
    from chess_ai.inference import InferenceEngine

LOGGER = logging.getLogger(__name__)

SUITE: Final = "stockfish-ladder"
"""What the ladder is called among the evaluation suites, and what its files are named."""

EVENT: Final = "chess-ai ladder"
"""The ``Event`` of every ladder game's PGN, telling them from the games of a plain match."""

FORMAT_VERSION: Final = 1


class LadderLevel(BaseModel):
    """How the checkpoint did against one level of the ladder."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    elo: int
    """The strength Stockfish played at, which is the rating the results are counted against."""
    requested_elo: int
    """The level asked for, which is :attr:`elo` unless Stockfish could not play at it."""
    move_time: float | None
    """Seconds Stockfish had for each move, or ``None`` when it was held to :attr:`nodes`."""
    nodes: int | None = None
    """How many positions Stockfish searched for each move, or ``None`` when it was held to
    :attr:`move_time`, as every ladder from before ladders could be held to nodes was."""
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def pace(self) -> str:
        """What Stockfish was held to, such as ``0.1 s a move`` or ``100,000 nodes a move``."""
        return describe_limit(self.move_time, self.nodes)


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


class LadderGame(BaseModel):
    """One of the games of a ladder, as the result lists it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    index: int
    """Where the game is in ``stockfish-ladder.pgn``, counting from 0."""
    elo: int
    """The level it was played at."""
    model_color: Literal["white", "black"]
    """Which side the checkpoint played."""
    opening: str
    """The name of the line it started from."""
    result: str
    """``1-0``, ``0-1``, ``1/2-1/2``, or ``*`` for a game with no result."""
    termination: GameOverReason | None
    plies: int
    """How many moves were played, the opening's among them."""


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
    checkpoint_written: datetime | None = None
    """When the checkpoint file was written, which tells it from another of the same step;
    ``None`` from a ladder played before results said so."""
    started: datetime
    finished: datetime
    code_version: str
    openings: str
    """The opening set the games started from, and its version."""
    games_per_level: int
    levels: list[LadderLevel]
    estimate: LadderEstimate
    games: list[LadderGame] = []
    """The games, in the order they are in ``stockfish-ladder.pgn``; empty from a ladder played
    before results listed them, whose games are in the file all the same."""


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
    on_session: Callable[[StockfishDescription, int, GameSession], None] | None = None,
    parallel: int = 1,
) -> list[PlayedLevel]:
    """Play ``model`` against Stockfish at each of ``levels``, ``games`` games each, as a match
    at each level would, and return the games by level, weakest first, and by number.

    ``parallel`` games are played at once. Each of them is played against an engine of its own,
    which ``opponent`` makes for the level: whichever game ends first takes the next one, in
    order of level and number, and keeps its engine while the games it takes are of the same
    level, so at most ``parallel`` engines run at a time. One at a time, every game of a level is
    played against one engine, closed once they are over. ``model`` is the first player of every
    game, and is asked for moves in several games at once when several are played.

    ``on_game`` is handed each game as it ends, with the level it was played at, and
    ``on_session`` each game's session before it is played, with the level and the game's number
    in it, as :func:`~chess_ai.match.play_match` hands them. Played at once, games end, and may
    begin, in an order of their own; the games returned are in order all the same.

    A game in which Stockfish hangs or dies is played again from its start against an engine
    ``opponent`` makes in its place, as :func:`~chess_ai.match.replaying` has it, so that one
    late move does not throw away every other game of the ladder. ``on_session`` is then handed
    the game's session again.

    Raises:
        ValueError: there are no levels, or ``games`` or ``parallel`` is less than one.
        TypeError: ``opponent`` made a player that cannot say how strong it plays.
        GameFailed: a game's engine hung or died every time the game was played.
        Whatever else :func:`~chess_ai.match.play_game` or ``opponent`` raise. Every game still
            being played is stopped, and every engine closed, first.
    """
    if not levels:
        raise ValueError("a ladder needs at least one level")
    if games < 1:
        raise ValueError(f"a match needs at least one game, not {games}")
    if parallel < 1:
        raise ValueError(f"a ladder plays at least one game at a time, not {parallel}")
    ladder = _Ladder(
        levels=levels,
        openings=openings,
        opponent=opponent,
        on_game=on_game,
        on_session=on_session,
        per_level=games,
        games=iter(
            [(index, number) for index in range(len(levels)) for number in range(1, games + 1)]
        ),
        played=[[] for _ in levels],
    )
    at_once = min(parallel, games * len(levels))
    if at_once == 1:
        await _play_games(model, ladder)
    else:
        await _all_or_none(
            [asyncio.ensure_future(_play_games(model, ladder)) for _ in range(at_once)]
        )
    return [
        PlayedLevel(
            stockfish=ladder.described[index],
            games=sorted(ladder.played[index], key=lambda game: game.number),
        )
        for index in range(len(levels))
    ]


@dataclass
class _Ladder:
    """What the games of a ladder played at once share: what they are played with, the games
    still to be played, and those played so far.

    Not the model: whatever plays a ladder's games is handed it, rather than a closure being,
    so that the frames a failure comes through, which keep their function and its closure
    whatever clears them, keep no network from being let go of.
    """

    levels: Sequence[int]
    openings: OpeningSet
    opponent: Opponent
    on_game: Callable[[StockfishDescription, MatchGame], None] | None
    on_session: Callable[[StockfishDescription, int, GameSession], None] | None
    per_level: int
    """How many games each level plays."""
    games: Iterator[tuple[int, int]]
    """The games still to be played, each as its level's place among the levels and its number;
    one iterator for every game played at once, which hands each game out once."""
    played: list[list[MatchGame]]
    """The games played so far at each level, in the order they ended."""
    described: dict[int, StockfishDescription] = field(default_factory=dict)
    """How strong Stockfish played at each level that has begun."""


async def _play_games(model: Player, ladder: _Ladder) -> None:
    """Play the next game of ``ladder`` there is, and the next, until there are none."""
    engine = _LevelEngine(ladder)
    total = ladder.per_level * len(ladder.levels)
    try:
        for index, number in ladder.games:
            # The engine is started as part of the game, so that one that does not start in
            # time, on a machine still stalled, is a try of the game like a move too late.
            description, game = await replaying(
                lambda index=index, number=number: _play_one(model, ladder, engine, index, number),
                engine.close,
                game=lambda index=index, number=number: (
                    f"game {index * ladder.per_level + number}/{total} against "
                    f"Stockfish {_playing_at(ladder, index)}"
                ),
            )
            ladder.played[index].append(game)
            if ladder.on_game is not None:
                ladder.on_game(description, game)
    finally:
        await engine.close()


def _playing_at(ladder: _Ladder, index: int) -> int:
    """The Elo Stockfish plays the level at ``index`` at, as the game lines name it, or the level
    asked for if no engine has started for it yet."""
    described = ladder.described.get(index)
    return described.elo if described is not None else ladder.levels[index]


async def _play_one(
    model: Player, ladder: _Ladder, engine: "_LevelEngine", index: int, number: int
) -> tuple[StockfishDescription, MatchGame]:
    """Play game ``number`` of the level at ``index`` against ``engine``'s Stockfish for it, and
    say how strong that played."""
    stockfish = await engine.at(index)
    description = stockfish.stockfish
    on_session = ladder.on_session

    def begun(number: int, session: GameSession) -> None:
        if on_session is not None:
            on_session(description, number, session)

    game = await play_game(
        model, stockfish, number, openings=ladder.openings, on_session=begun, event=EVENT
    )
    return description, game


class _LevelEngine:
    """The Stockfish one of the games of a ladder played at once plays against: one level's,
    kept while its games are of that level, and started again once closed."""

    def __init__(self, ladder: _Ladder) -> None:
        self._ladder = ladder
        self._index: int | None = None
        self._player: Player | None = None

    async def at(self, index: int) -> StockfishBackedPlayer:
        """The engine for the level at ``index``, started if it is not running.

        Raises:
            TypeError: the ladder's ``opponent`` made a player that cannot say how strong it
                plays.
            Whatever ``opponent`` raises.
        """
        if index != self._index:
            await self.close()
            self._index = index
        if self._player is None:
            self._player = await self._ladder.opponent(self._ladder.levels[index])
            if not isinstance(self._player, StockfishBackedPlayer):
                raise TypeError(f"{self._player.name} cannot be a level of a ladder")
            self._ladder.described.setdefault(index, self._player.stockfish)
        assert isinstance(self._player, StockfishBackedPlayer)
        return self._player

    async def close(self) -> None:
        """Close the engine, if one is running."""
        player, self._player = self._player, None
        if player is not None:
            await close_and_wait(player)


async def _all_or_none(tasks: list["asyncio.Future[None]"]) -> None:
    """Wait for every one of ``tasks``, and if one fails, stop the rest and raise what it did.

    Every task has stopped by the time this returns or raises, whatever stopped it, so that
    whatever they hold, such as engines, has been let go of.
    """
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for task in tasks:
        if task in done and not task.cancelled() and task.exception() is not None:
            raise task.exception()  # type: ignore[misc]


def clear_failure_frames(failure: BaseException) -> None:
    """Clear the frames of ``failure``'s traceback, and of every failure it was raised from or
    while handling, so that none of them keeps alive what its frames held.

    Both links, and each failure once: Python breaks a cycle of contexts, but not one made with
    ``raise ... from``, and walking one for ever would hang whoever is handling the failure.
    """
    seen: set[int] = set()
    left: list[BaseException] = [failure]
    while left:
        each = left.pop()
        if id(each) in seen:
            continue
        seen.add(id(each))
        traceback.clear_frames(each.__traceback__)
        left.extend(link for link in (each.__cause__, each.__context__) if link is not None)


def game_index(
    levels: Sequence[int], games: int, stockfish: StockfishDescription, number: int
) -> int:
    """Where game ``number`` of the level ``stockfish`` was asked to play at is among the games
    of a ladder of ``levels``, ``games`` each, counting from 0: the order the result lists them
    in, and the order :func:`write_games` saves them in."""
    return levels.index(stockfish.requested_elo) * games + number - 1


class LadderLines:
    """Says the games of a ladder as each starts and ends, as
    :class:`~chess_ai.match.GameLines` does, each under the header of its own level and
    numbered through the whole ladder, whichever order they are played in."""

    def __init__(
        self, say: Say, title: str, *, levels: Sequence[int], games: int, openings: OpeningSet
    ) -> None:
        self._levels = levels
        self._games = games
        self._lines = GameLines(say, title, total=games * len(levels), openings=openings)
        self._matches: dict[int, LinesMatch] = {}

    def starting(self, stockfish: StockfishDescription, number: int) -> None:
        """Say that game ``number`` of the level ``stockfish`` plays is starting."""
        self._lines.starting(number, self._match(stockfish))

    def ended(self, stockfish: StockfishDescription, game: MatchGame) -> None:
        """Say how ``game``, of the level ``stockfish`` plays, went."""
        self._lines.ended(game, self._match(stockfish))

    def _match(self, stockfish: StockfishDescription) -> LinesMatch:
        position = self._levels.index(stockfish.requested_elo)
        if position not in self._matches:
            self._matches[position] = self._lines.match(
                ["model", f"Stockfish {stockfish.elo}"],
                games=self._games,
                before=position * self._games,
            )
        return self._matches[position]


def write_games(path: Path, played: Sequence[PlayedLevel]) -> None:
    """Write the games of ``played`` to ``path``, replacing what it held, in the order the result
    lists them: the games file of a ladder that played games at once has them in the order they
    ended.

    Written beside it and moved over it once whole, since it is the only copy of the games: a
    write that fails part way, such as on a disk that has filled up, leaves it as it was.

    Raises:
        OSError: the games could not be written, and ``path`` is unchanged.
    """
    ordered = path.with_name(f"{path.name}.ordered")
    try:
        with ordered.open("w", encoding="utf-8") as file:
            for level in played:
                for game in level.games:
                    # PGN separates the games of a file with a blank line.
                    file.write(f"{game.pgn}\n")
        os.replace(ordered, path)
    except BaseException:
        ordered.unlink(missing_ok=True)
        raise


def ladder_result(
    model: ModelDescription,
    played: Sequence[PlayedLevel],
    *,
    openings: OpeningSet,
    games_per_level: int,
    started: datetime,
    finished: datetime,
    code_version: str,
    checkpoint_written: datetime | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> LadderResult:
    """What ``played`` comes to: the score at each level, the rating they add up to, and the
    games, numbered in the order they were played, which is the order they are saved in."""
    levels = []
    for level in played:
        counted = score(level.games)
        levels.append(
            LadderLevel(
                elo=level.stockfish.elo,
                requested_elo=level.stockfish.requested_elo,
                move_time=level.stockfish.move_time,
                nodes=level.stockfish.nodes,
                wins=counted.wins,
                draws=counted.draws,
                losses=counted.losses,
            )
        )
    estimate = estimate_rating(
        (Result(opponent=level.elo, points=level.points, games=level.games) for level in levels),
        confidence=confidence,
    )
    games = [
        _listed(index, level.stockfish.elo, game)
        for index, (level, game) in enumerate(
            (level, game) for level in played for game in level.games
        )
    ]
    return LadderResult(
        model=model,
        checkpoint_written=checkpoint_written,
        started=started,
        finished=finished,
        code_version=code_version,
        openings=openings.label,
        games_per_level=games_per_level,
        levels=levels,
        estimate=LadderEstimate.of(estimate),
        games=games,
    )


def _listed(index: int, elo: int, game: MatchGame) -> LadderGame:
    over = game.game.position.game_over
    return LadderGame(
        index=index,
        elo=elo,
        # The checkpoint is the first player of every match of a ladder.
        model_color="white" if game.first_plays_white else "black",
        opening=game.opening.name,
        result=over.result if over is not None else "*",
        termination=over.reason if over is not None else None,
        plies=len(game.game.moves),
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
    return start_partial_file(run.evaluation_directory(step), f"{SUITE}.pgn")


def save_ladder(
    run: RunReader,
    result: LadderResult,
    partial: Path,
    *,
    current: Callable[[], bool] | None = None,
) -> None:
    """Keep ``result`` and its games, from ``partial``, for the checkpoint it is about.

    Saved as one set by :func:`~chess_ai.training.run_store.save_evaluation`, so that a result
    has its own games beside it, whichever other ladder of the checkpoint finishes alongside.
    ``current`` says whether the checkpoint is still the one played, as it does there.

    Raises:
        CheckpointChanged: ``current`` said no. Nothing was saved.
        RunError: the run directory cannot be written. The games are still in ``partial``,
            unless the failure came after they had been moved to :func:`games_path`.
    """
    step = result.model.checkpoint
    save_evaluation(
        run.evaluation_directory(step),
        {f"{SUITE}.json": result.model_dump_json(indent=2) + "\n"},
        {f"{SUITE}.pgn": partial},
        current=current,
    )


def read_ladder(run: RunReader, step: int) -> LadderResult | None:
    """The ladder's result for the checkpoint from ``step``, or ``None`` if there is none.

    Raises:
        RunError: there is one, and it cannot be read or is not a result this code knows.
    """
    text = run.evaluation(step, SUITE)
    if text is None:
        return None
    try:
        return LadderResult.model_validate_json(text)
    except ValidationError as e:
        raise RunError(f"{result_path(run, step)} is not a ladder result: {e}") from e


def read_games(run: RunReader, step: int) -> str | None:
    """The PGN of the ladder games of the checkpoint from ``step``, or ``None`` if there is none.

    Raises:
        RunError: there is one, and it cannot be read.
    """
    path = games_path(run, step)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        raise RunError(f"cannot read {path}: {e}") from e


def wanted(run: RunReader, checkpoint: CheckpointInfo, every: int) -> bool:
    """Whether the evaluator ladders ``checkpoint``: one from a step that is a multiple of
    ``every``, or the last of a run that has finished.

    Only of a run that has finished, rather than its latest checkpoint as it trains, which would
    be every checkpoint it saves for as long as the evaluator keeps up.
    """
    if checkpoint.step % every == 0:
        return True
    try:
        status = run.status
    except RunError:
        return False
    if status is None or status.status is not RunStatus.FINISHED:
        return False
    latest = run.latest_checkpoint()
    return latest is not None and latest.step == checkpoint.step


def awaiting_ladder(run: RunReader, every: int) -> Callable[[CheckpointInfo], bool]:
    """Which of ``run``'s checkpoints are due a ladder the evaluator has not played yet, for the
    trainer's retention to keep them until it has: :attr:`RunWriter.held
    <chess_ai.training.run_store.RunWriter.held>`.

    A ladder takes minutes, and an evaluator busy with another run's, or with this checkpoint's
    sample games, would otherwise find the checkpoint pruned when its turn came. Whether the run
    asks for the ladder is read from the run directory each time, as the evaluator reads it.
    """
    from chess_ai.evaluator import configured_suites

    def awaiting(checkpoint: CheckpointInfo) -> bool:
        if checkpoint.step % every != 0:
            return False
        try:
            if SUITE not in configured_suites(run):
                return False
            written = run.checkpoint_written(checkpoint)
        except RunError:
            return False
        return written is not None and not stands(run, checkpoint, written)

    return awaiting


def stands(run: RunReader, checkpoint: CheckpointInfo, written: datetime) -> bool:
    """Whether ``checkpoint``'s ladder was played with the file that is there now, written at
    ``written``.

    Not whether it was played as the settings now say: a ladder is played again for that on
    demand only, since it takes minutes a checkpoint. A ladder from before results said which
    file they were about stands if it began after the file was written, as ``chess-ai ladder``
    was played on the file that was there.
    """
    try:
        result = read_ladder(run, checkpoint.step)
    except RunError:
        return False
    return result is not None and result_stands(result, checkpoint.step, written)


def result_stands(result: LadderResult, step: int, written: datetime) -> bool:
    """Whether ``result`` is a ladder of the checkpoint from ``step`` as the file written at
    ``written`` has it, as :func:`stands` decides for the result on disk: for a result already
    read, so that what is decided about is what is used."""
    if result.model.checkpoint != step:
        return False
    if result.checkpoint_written is None:
        return result.started >= written
    return result.checkpoint_written == written


def summary(result: LadderResult) -> str:
    """One line about ``result``, such as ``Elo 1523 (1460 to 1590), 50/100 at 0.1 s a move``."""
    at = " and ".join(dict.fromkeys(level.pace for level in result.levels))
    estimate = result.estimate
    return f"Elo {estimate.describe()}, {estimate.points:g}/{estimate.games} points at {at}"


StockfishStarter = Callable[[int, float, int | None], Awaitable[Player]]
"""Starts a level's Stockfish, given the Elo, the move time and the number of nodes a move,
which is ``None`` for one held to the move time: a player that is
:class:`~chess_ai.players.StockfishBackedPlayer`, so that the result can say how strong it was."""

EngineLoader = Callable[[Path], "InferenceEngine"]
"""Loads a checkpoint's network where the ladder plays it, which need not be where the evaluator
loads it for the other suites."""


class LadderSuite:
    """The Stockfish ladder, as an evaluation suite.

    ``settings`` are the ``[ladder]`` section of the config, of which a ladder played by the
    evaluator takes all but nothing from the command line. ``load`` loads the checkpoint where
    the ladder plays it, or ``None`` to play the one the evaluator loaded; ``release`` is
    called once the ladder is done with what ``load`` loaded and nothing holds it any more,
    such as to hand a GPU's memory back to the training run beside it. ``live`` is the runs
    directory to show the game being played in, or ``None`` to show nothing, as a one-off
    evaluation does. ``say`` is handed a line as each game begins and ends, since a ladder
    plays for minutes.
    """

    name = SUITE

    def __init__(
        self,
        settings: LadderConfig,
        openings: OpeningSet,
        *,
        stockfish: StockfishStarter,
        load: EngineLoader | None = None,
        release: Callable[[], None] = lambda: None,
        live: Path | None = None,
        say: Say = said_on_stdout,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.settings = settings
        self.openings = openings
        self._stockfish = stockfish
        self._load = load
        self._release = release
        self._live = live
        self._say = say
        self._clock = clock

    def wants(self, run: RunReader, checkpoint: CheckpointInfo) -> bool:
        return wanted(run, checkpoint, self.settings.every)

    def stands(self, run: RunReader, checkpoint: CheckpointInfo, written: datetime) -> bool:
        return stands(run, checkpoint, written)

    def evaluate(
        self,
        run: RunReader,
        checkpoint: CheckpointInfo,
        written: datetime,
        engine: "InferenceEngine",
        *,
        stop: threading.Event | None = None,
    ) -> str:
        saved: list[LadderResult] = []
        try:
            playing = self._play(run, checkpoint, written, engine, saved)
            return summary(asyncio.run(until_stopped(playing, stop, saved)))
        except BaseException as e:
            # The frames a failure came through still hold the network the ladder loaded,
            # for as long as whoever catches it keeps it, which would keep it from release.
            clear_failure_frames(e)
            raise
        finally:
            if self._load is not None:
                # Once nothing holds what was loaded, so that there is something to release.
                del playing
                gc.collect()
                self._release()

    async def _play(
        self,
        run: RunReader,
        checkpoint: CheckpointInfo,
        written: datetime,
        engine: "InferenceEngine",
        saved: list[LadderResult],
    ) -> LadderResult:
        """Play the ladder and save it, adding the result to ``saved`` once it is saved."""
        from chess_ai.inference.player import ModelPlayer

        settings = self.settings
        label = f"{run.name}@{checkpoint.step}"
        total = settings.games_per_level * len(settings.levels)
        created = created_with(run, checkpoint, written)
        partial: Path | None = None
        try:
            if self._load is not None:
                path = run.checkpoint_path(checkpoint)
                try:
                    engine = await asyncio.to_thread(self._load, path)
                except Exception as e:
                    if path.exists():
                        raise
                    # Pruned since the evaluator loaded it: passed over, as it would have been
                    # had it been pruned before its turn came.
                    raise CheckpointChanged(f"{label} was pruned before its ladder") from e
            started = self._clock()
            await check_playable(engine, self.openings, settings.rating)
            model = ModelPlayer(
                engine, run=run.name, checkpoint=checkpoint.step, rating=settings.rating
            )
            partial = start_games_file(run, checkpoint.step)
            games_file = partial
            levels = settings.levels
            per_level = settings.games_per_level
            lines = LadderLines(
                self._say,
                f"{label} ladder games",
                levels=levels,
                games=per_level,
                openings=self.openings,
            )
            playing: dict[int, tuple[GameSession, int]] = {}
            """The games being played, by where they are among the ladder's, and the Elo of each."""
            # The game the live view follows: the first of those being played, until it ends.
            followed: int | None = None

            async def opponent(level: int) -> Player:
                return await self._stockfish(level, settings.move_time, settings.nodes)

            def show(index: int) -> None:
                if self._live is not None and index == followed:
                    session, elo = playing[index]
                    self._show(run, checkpoint.step, index, total, elo, session)

            def on_session(
                stockfish: StockfishDescription, number: int, session: GameSession
            ) -> None:
                nonlocal followed
                lines.starting(stockfish, number)
                index = game_index(levels, per_level, stockfish, number)
                playing[index] = (session, stockfish.elo)
                if followed is None:
                    followed = index
                show(index)
                session.listen(lambda event=None: show(index))

            def on_game(stockfish: StockfishDescription, game: MatchGame) -> None:
                nonlocal followed
                append_game(games_file, game)
                lines.ended(stockfish, game)
                index = game_index(levels, per_level, stockfish, game.number)
                del playing[index]
                if index == followed:
                    followed = min(playing, default=None)
                    if followed is not None:
                        show(followed)

            played = await play_ladder(
                model,
                levels,
                games=per_level,
                openings=self.openings,
                opponent=opponent,
                on_game=on_game,
                on_session=on_session,
                parallel=settings.parallel_games,
            )
            write_games(games_file, played)
            result = ladder_result(
                model.model,
                played,
                openings=self.openings,
                games_per_level=settings.games_per_level,
                started=started,
                finished=self._clock(),
                code_version=code_version(),
                checkpoint_written=written,
            )
            save_evaluation(
                run.evaluation_directory(checkpoint.step),
                {f"{SUITE}.json": result.model_dump_json(indent=2) + "\n"},
                {f"{SUITE}.pgn": partial},
                current=lambda: still_evaluated(run, checkpoint, written, created),
            )
            saved.append(result)
            return result
        finally:
            # Moved into place by a save that went through; anything else is games nobody
            # will look at, since the checkpoint plays the ladder again.
            if partial is not None:
                partial.unlink(missing_ok=True)
            if self._live is not None:
                with suppress(OSError):
                    clear_live_game(self._live)

    def after(self, run: RunReader) -> None:
        """Prune the checkpoints of ``run`` held for their ladders that no longer need holding,
        if it is no longer being trained: a trainer does so at its next save. Only once the
        checkpoint's other suites are done with it too, which is now."""
        try:
            prune_released(run.directory, awaiting_ladder(run, self.settings.every))
        except OSError as e:
            # Disk the run's retention was to free; the ladder itself is saved.
            LOGGER.warning("Cannot prune the checkpoints of %s: %s", run.name, e)

    def _show(
        self, run: RunReader, step: int, index: int, games: int, elo: int, session: GameSession
    ) -> None:
        """Write the game being played for the web server to send, as far as it can be."""
        assert self._live is not None
        live = LiveGame(
            suite=SUITE,
            run=run.name,
            step=step,
            index=index,
            games=games,
            opponent="stockfish",
            level=elo,
            updated=self._clock(),
            game=watched(session.state),
        )
        show_live_game(self._live, live)
