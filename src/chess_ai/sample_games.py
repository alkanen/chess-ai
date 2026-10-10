"""Sample games: a few games played by each checkpoint, to watch the model improve by.

The checkpoint plays itself and, if asked to, Stockfish, through the match runner like any
other match: every game from a line of the opening set, the same lines for every checkpoint so
that their games can be set side by side. Against itself it plays each line once, since a
deterministic model playing a line from both sides would play the same game twice; against
Stockfish it plays each line from both sides, as a match does.

What it played is kept in the run directory, beside what any other evaluation finds out about
the same checkpoint: ``sample-games.json`` lists the games and how they ended, and
``sample-games.pgn`` holds them. The games are written to a ``.partial`` file as they end and
moved into place with the result, as the ladder's are.

While a game is being played it can be watched: the evaluator writes the game as it stands to
``live-game.json`` in the runs directory after every move, and the web server, which polls the
runs directory anyway, sends it to whoever is looking at that run. There is one such file for
the whole runs directory, since only one evaluator works on a runs directory at a time and it
plays one game at a time. It is deleted when the games are over, and by an evaluator starting;
one left behind by an evaluator that was killed says when it was last written, which is how a
reader tells it from a game still being played.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import chess
from pydantic import BaseModel, ConfigDict, ValidationError

from chess_ai.config import SampleGamesConfig
from chess_ai.evaluator import SuiteStopped
from chess_ai.game_session import GameSession, GameState
from chess_ai.match import MatchGame, append_game, play_match
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
from chess_ai.probes import check_finite
from chess_ai.training.run_store import (
    TEMPORARY_SUFFIX,
    CheckpointInfo,
    RunError,
    RunReader,
    code_version,
    save_evaluation,
    start_partial_file,
)

if TYPE_CHECKING:
    from chess_ai.inference import InferenceEngine

LOGGER = logging.getLogger(__name__)

SUITE: Final = "sample-games"
"""What the sample games are called among the evaluation suites, and what their files are named."""

EVENT: Final = "chess-ai sample games"
"""The ``Event`` of every sample game's PGN, telling them from the games of a plain match."""

FORMAT_VERSION: Final = 1

LIVE_FILE: Final = "live-game.json"
"""In the runs directory: the sample game the evaluator is playing now, as it stands."""

STOP_POLL_SECONDS: Final = 0.1
"""How often the games look at whether the evaluator has been told to stop."""

CLOSE_TIMEOUT: Final = 5.0
"""How long the games wait for their Stockfish process to exit once they are over."""

Opponent = Literal["self", "stockfish"]


class SampleGame(BaseModel):
    """One of the games, as the result lists it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    index: int
    """Where the game is in ``sample-games.pgn``, counting from 0."""
    opponent: Opponent
    model_color: Literal["white", "black", "both"]
    """Which side the checkpoint played: both, against itself."""
    opening: str
    """The name of the line it started from."""
    white: str
    black: str
    result: str
    """``1-0``, ``0-1``, ``1/2-1/2``, or ``*`` for a game with no result."""
    termination: GameOverReason | None
    plies: int
    """How many moves were played, the opening's among them."""


class SampleGamesResult(BaseModel):
    """What a checkpoint's sample games came to, as ``sample-games.json`` holds it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    format_version: Literal[1] = FORMAT_VERSION
    suite: Literal["sample-games"] = SUITE
    model: ModelDescription
    """The run, the checkpoint, and the rating it was asked to play like."""
    checkpoint_written: datetime
    """When the checkpoint file was written, which tells it from another of the same step."""
    started: datetime
    finished: datetime
    code_version: str
    openings: str
    """The opening set the games started from, and its version."""
    stockfish: StockfishDescription | None
    """How strong Stockfish played, or ``None`` when it was not played."""
    games: list[SampleGame]


class LiveGame(BaseModel):
    """The sample game being played now, as ``live-game.json`` holds it."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    run: str
    step: int
    """The checkpoint playing it."""
    index: int
    """Where it will be in the checkpoint's ``sample-games.pgn``, counting from 0."""
    games: int
    """How many games the checkpoint plays in all."""
    opponent: Opponent
    updated: datetime
    """When the file was last written: after the last move, or when the game began."""
    game: GameState
    """The game as it stands. Only the last move says what the player was thinking, and the
    position offers no moves: it is there to be watched."""


def result_path(run: RunReader, step: int) -> Path:
    """Where the result for the checkpoint from ``step`` is kept."""
    return run.evaluation_directory(step) / f"{SUITE}.json"


def games_path(run: RunReader, step: int) -> Path:
    """Where the games the result lists are kept."""
    return run.evaluation_directory(step) / f"{SUITE}.pgn"


def read_sample_games(run: RunReader, step: int) -> SampleGamesResult | None:
    """The result for the checkpoint from ``step``, or ``None`` if there is none.

    Raises:
        RunError: there is one, and it cannot be read or is not a result this code knows.
    """
    text = run.evaluation(step, SUITE)
    if text is None:
        return None
    try:
        return SampleGamesResult.model_validate_json(text)
    except ValidationError as e:
        raise RunError(f"{result_path(run, step)} is not a sample-games result: {e}") from e


def read_games(run: RunReader, step: int) -> str | None:
    """The PGN of the sample games of the checkpoint from ``step``, or ``None`` if there is none.

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


def stands(run: RunReader, checkpoint: CheckpointInfo, written: datetime) -> bool:
    """Whether ``checkpoint``'s games were played with the file that is there now, written at
    ``written``.

    Not whether they were played as the settings now say: games are played again for that on
    demand only, since every checkpoint of every run would otherwise play them again. Nothing
    of the settings is needed to tell, so that whether a checkpoint lacks its games can be
    told without them, as by an evaluator whose settings cannot be read.
    """
    try:
        result = read_sample_games(run, checkpoint.step)
    except RunError:
        return False
    return (
        result is not None
        and result.model.checkpoint == checkpoint.step
        and result.checkpoint_written == written
    )


def save_live_game(runs_dir: Path, live: LiveGame) -> None:
    """Write ``live`` to the runs directory, in place of the game that was there.

    Renamed into place, so that a reader never sees half of it, but not synced to the disk: it
    is rewritten every move, and a machine that goes down has stopped the game anyway.
    """
    path = runs_dir / LIVE_FILE
    temporary = path.with_name(f"{path.name}{TEMPORARY_SUFFIX}")
    try:
        temporary.write_text(live.model_dump_json() + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def clear_live_game(runs_dir: Path) -> None:
    """Say that no sample game is being played, by deleting the file that says one is."""
    (runs_dir / LIVE_FILE).unlink(missing_ok=True)


def read_live_game(runs_dir: Path, *, now: datetime, stale_after: float) -> LiveGame | None:
    """The sample game being played in ``runs_dir`` now, or ``None`` if there is none.

    A game whose file has not been written for ``stale_after`` seconds is not being played: its
    evaluator was killed before it could delete it. One that cannot be read is none either.
    """
    path = runs_dir / LIVE_FILE
    try:
        live = LiveGame.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as e:
        # Debug: every viewer of a run asks about once a second.
        LOGGER.debug("Cannot read the live sample game %s: %s", path, e)
        return None
    if now - live.updated > timedelta(seconds=stale_after):
        return None
    return live


def watched(game: GameState) -> GameState:
    """``game`` with only what watching it takes: the thoughts behind the last move alone, and
    no legal moves, which would otherwise make the file larger with every move."""
    moves = [move.model_copy(update={"thoughts": None}) for move in game.moves[:-1]]
    return game.model_copy(
        update={
            "moves": [*moves, *game.moves[-1:]],
            "position": game.position.model_copy(update={"legal_moves": {}}),
        }
    )


StockfishStarter = Callable[[int, float], Awaitable[Player]]
"""Starts the Stockfish opponent, given the Elo and the move time: a player that is
:class:`~chess_ai.players.StockfishBackedPlayer`, so that the result can say how strong it was."""


class SampleGamesSuite:
    """The sample games, as an evaluation suite.

    ``stockfish`` starts the Stockfish opponent; it is only called for a config that asks for
    games against it. ``live`` is the runs directory to show
    the game being played in, or ``None`` to show nothing, as a one-off evaluation does: only
    the evaluator, of which there is one, may write the one file there is.
    """

    name = SUITE

    def __init__(
        self,
        settings: SampleGamesConfig,
        openings: OpeningSet,
        *,
        stockfish: StockfishStarter | None = None,
        live: Path | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if settings.stockfish_games and stockfish is None:
            raise ValueError("games against Stockfish need a way to start it")
        self.settings = settings
        self.openings = openings
        self._stockfish = stockfish
        self._live = live
        self._clock = clock

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
        saved: list[SampleGamesResult] = []
        playing = self._play(run, checkpoint, written, engine, saved)
        result = asyncio.run(self._until_stopped(playing, stop, saved))
        return summary(result)

    async def _until_stopped(
        self,
        playing: Coroutine[Any, Any, SampleGamesResult],
        stop: threading.Event | None,
        saved: list[SampleGamesResult],
    ) -> SampleGamesResult:
        """What ``playing`` comes to, unless ``stop`` is set before it has saved it in ``saved``.

        Games take minutes, and a stop that waited for them would be a stop a service manager
        gives up on and turns into a kill, which leaves the games' files behind. So the games
        are cancelled, in the middle of a move if need be, and their own cleaning up — the
        partial games file, the game shown as being played, Stockfish — is done on the way out.

        Raises:
            SuiteStopped: ``stop`` was set before the games were over; nothing was saved.
        """
        task = asyncio.create_task(playing)
        if stop is None:
            return await task
        while not task.done():
            # A threading event, set by a signal handler: there is nothing to await it with.
            await asyncio.wait({task}, timeout=STOP_POLL_SECONDS)
            if stop.is_set() and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                # Finished after all, the moment before it would have been cancelled.
                if not task.cancelled():
                    return task.result()
                # Cancelled in its cleaning up, such as waiting for Stockfish to exit, after the
                # games were saved: they are done, and the next evaluator would find them so.
                if saved:
                    return saved[0]
                raise SuiteStopped("stopped before the sample games were over")
        return task.result()

    async def _play(
        self,
        run: RunReader,
        checkpoint: CheckpointInfo,
        written: datetime,
        engine: "InferenceEngine",
        saved: list[SampleGamesResult],
    ) -> SampleGamesResult:
        """Play the games and save them, adding the result to ``saved`` once it is saved."""
        from chess_ai.inference.player import ModelPlayer

        settings = self.settings
        started = self._clock()
        # Before any game, as the probes do: a diverged network plays legal moves all the same,
        # chosen from NaNs, and games of that would be taken for the games of a model.
        board = chess.Board()
        for move in self.openings.openings[0].moves:
            board.push(move)
        check_finite(
            await asyncio.to_thread(
                engine.evaluate,
                board,
                mover_rating=settings.rating,
                opponent_rating=settings.rating,
            ),
            f"the position after {self.openings.openings[0].name}",
        )
        model = ModelPlayer(
            engine, run=run.name, checkpoint=checkpoint.step, rating=settings.rating
        )
        total = settings.games + settings.stockfish_games
        stockfish: Player | None = None
        partial: Path | None = None
        listed: list[SampleGame] = []
        try:
            if settings.stockfish_games:
                assert self._stockfish is not None
                # Before any game, so that an engine that will not start costs no games.
                stockfish = await self._stockfish(
                    settings.stockfish_elo, settings.stockfish_move_time
                )
                if not isinstance(stockfish, StockfishBackedPlayer):
                    raise TypeError(f"{stockfish.name} cannot say how strong it plays")
            partial = start_partial_file(run.evaluation_directory(checkpoint.step), f"{SUITE}.pgn")
            games_file = partial

            def kept(opponent: Opponent) -> Callable[[MatchGame], None]:
                def keep(game: MatchGame) -> None:
                    append_game(games_file, game)
                    listed.append(_listed(len(listed), opponent, game))

                return keep

            def watch(opponent: Opponent, before: int) -> Callable[[int, GameSession], None]:
                def follow(number: int, session: GameSession) -> None:
                    if self._live is None:
                        return
                    index = before + number - 1
                    self._show(run, checkpoint.step, index, total, opponent, session.state)
                    session.listen(
                        lambda event: self._show(
                            run, checkpoint.step, index, total, opponent, session.state
                        )
                    )

                return follow

            if settings.games:
                await play_match(
                    model,
                    model,
                    games=settings.games,
                    openings=self.openings,
                    on_game=kept("self"),
                    on_session=watch("self", 0),
                    event=EVENT,
                    paired=False,
                    move_delay=settings.move_delay,
                )
            if stockfish is not None:
                await play_match(
                    model,
                    stockfish,
                    games=settings.stockfish_games,
                    openings=self.openings,
                    on_game=kept("stockfish"),
                    on_session=watch("stockfish", settings.games),
                    event=EVENT,
                    move_delay=settings.move_delay,
                )
            result = SampleGamesResult(
                model=model.model,
                checkpoint_written=written,
                started=started,
                finished=self._clock(),
                code_version=code_version(),
                openings=self.openings.label,
                stockfish=(
                    stockfish.stockfish if isinstance(stockfish, StockfishBackedPlayer) else None
                ),
                games=listed,
            )
            save_evaluation(
                run.evaluation_directory(checkpoint.step),
                {f"{SUITE}.json": result.model_dump_json(indent=2) + "\n"},
                {f"{SUITE}.pgn": partial},
                current=lambda: run.checkpoint_written(checkpoint) == written,
            )
            saved.append(result)
            return result
        finally:
            # Moved into place by a save that went through; anything else is games nobody
            # will look at, since the checkpoint plays them all again.
            if partial is not None:
                partial.unlink(missing_ok=True)
            if self._live is not None:
                with suppress(OSError):
                    clear_live_game(self._live)
            if stockfish is not None:
                close_players(stockfish)
                with suppress(TimeoutError):
                    await asyncio.wait_for(wait_players_closed(stockfish), CLOSE_TIMEOUT)

    def _show(
        self,
        run: RunReader,
        step: int,
        index: int,
        games: int,
        opponent: Opponent,
        game: GameState,
    ) -> None:
        """Write the game being played for the web server to send, as far as it can be."""
        assert self._live is not None
        live = LiveGame(
            run=run.name,
            step=step,
            index=index,
            games=games,
            opponent=opponent,
            updated=self._clock(),
            game=watched(game),
        )
        try:
            save_live_game(self._live, live)
        except OSError as e:
            # Watching is a nicety; the games themselves go on, and are saved as they end.
            LOGGER.debug("Cannot show the sample game being played: %s", e)


def _listed(index: int, opponent: Opponent, game: MatchGame) -> SampleGame:
    over = game.game.position.game_over
    # Against itself the checkpoint is both sides, and the first player is White throughout.
    color: Literal["white", "black", "both"] = (
        "both" if opponent == "self" else "white" if game.first_plays_white else "black"
    )
    return SampleGame(
        index=index,
        opponent=opponent,
        model_color=color,
        opening=game.opening.name,
        white=game.game.white.name,
        black=game.game.black.name,
        result=over.result if over is not None else "*",
        termination=over.reason if over is not None else None,
        plies=len(game.game.moves),
    )


def summary(result: SampleGamesResult) -> str:
    """One line about ``result``, such as ``2 games against itself (1-0, 1/2-1/2)``."""
    parts = []
    against_itself = [game for game in result.games if game.opponent == "self"]
    if against_itself:
        results = ", ".join(game.result for game in against_itself)
        parts.append(f"{_games(len(against_itself))} against itself ({results})")
    against_stockfish = [game for game in result.games if game.opponent == "stockfish"]
    if against_stockfish and result.stockfish is not None:
        points = sum(_points(game) for game in against_stockfish)
        parts.append(
            f"{_games(len(against_stockfish))} against Stockfish {result.stockfish.elo} "
            f"(scored {points:g} of {len(against_stockfish)})"
        )
    return ", ".join(parts) or "no games"


def _games(count: int) -> str:
    return f"{count} game" if count == 1 else f"{count} games"


def _points(game: SampleGame) -> float:
    """What the checkpoint scored in a game against Stockfish; nothing for a game with no result."""
    white = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}.get(game.result)
    if white is None:
        return 0.0
    return white if game.model_color == "white" else 1.0 - white
