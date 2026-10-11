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

A ladder is played by ``chess-ai ladder`` on demand, and by the evaluator as a suite of its own
(:class:`LadderSuite`) for the runs that ask for it. A ladder takes minutes and runs save a
checkpoint far more often, so the evaluator ladders a run's checkpoints only every so many
steps, and a finished run's last; its games can be watched as they are played, as the sample
games can.
"""

import asyncio
import gc
import logging
import threading
import traceback
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from chess_ai.config import LadderConfig
from chess_ai.evaluator import until_stopped
from chess_ai.game_session import GameSession
from chess_ai.match import MatchGame, append_game, game_line, play_match, score
from chess_ai.openings import OpeningSet
from chess_ai.players import (
    ModelDescription,
    Player,
    StockfishBackedPlayer,
    StockfishDescription,
    close_players,
    wait_players_closed,
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
) -> list[PlayedLevel]:
    """Play ``model`` against Stockfish at each of ``levels`` in turn, ``games`` games each.

    Each level's opponent is made by ``opponent`` when its turn comes, and closed once its games
    are over, so that at most one engine runs at a time. ``on_game`` is handed each game as it
    ends, with the level it was played at, and ``on_session`` each game's session before it is
    played, with the level and the game's number in it, as
    :func:`~chess_ai.match.play_match` hands them. ``model`` is the first player of every match.

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

            def begun(number: int, session: GameSession, description=description) -> None:
                if on_session is not None:
                    on_session(description, number, session)

            level_games = await play_match(
                model,
                stockfish,
                games=games,
                openings=openings,
                on_game=handed_over,
                on_session=begun,
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
    if result is None or result.model.checkpoint != checkpoint.step:
        return False
    if result.checkpoint_written is None:
        return result.started >= written
    return result.checkpoint_written == written


def summary(result: LadderResult) -> str:
    """One line about ``result``, such as ``Elo 1523 (1460 to 1590), 50/100 at 0.1 s a move``."""
    move_times = sorted({level.move_time for level in result.levels})
    at = ", ".join(f"{move_time:g}" for move_time in move_times)
    estimate = result.estimate
    return (
        f"Elo {estimate.describe()}, {estimate.points:g}/{estimate.games} points at {at} s a move"
    )


StockfishStarter = Callable[[int, float], Awaitable[Player]]
"""Starts a level's Stockfish, given the Elo and the move time: a player that is
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
        say: Callable[[str], None] = LOGGER.info,
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
            traceback.clear_frames(e.__traceback__)
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
            begun = 0
            ended = 0

            async def opponent(level: int) -> Player:
                return await self._stockfish(level, settings.move_time)

            def on_session(
                stockfish: StockfishDescription, number: int, session: GameSession
            ) -> None:
                nonlocal begun
                begun += 1
                index = begun - 1
                level = f"Stockfish {stockfish.elo}"
                model_white = session.state.white.model is not None
                white, black = (label, level) if model_white else (level, label)
                self._say(f"{label} {SUITE}: game {begun}/{total} begins: {white} – {black}")
                if self._live is None:
                    return

                def show(event: object = None) -> None:
                    self._show(run, checkpoint.step, index, total, stockfish.elo, session)

                show()
                session.listen(show)

            def on_game(stockfish: StockfishDescription, game: MatchGame) -> None:
                nonlocal ended
                ended += 1
                append_game(games_file, game)
                names = [label, f"Stockfish {stockfish.elo}"]
                line = game_line(game, names, number=ended, games=total)
                self._say(f"{label} {SUITE}: {line}")

            played = await play_ladder(
                model,
                settings.levels,
                games=settings.games_per_level,
                openings=self.openings,
                opponent=opponent,
                on_game=on_game,
                on_session=on_session,
            )
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
