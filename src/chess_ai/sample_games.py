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
``live-game.json`` in the runs directory after every move, as it does a game of the Stockfish
ladder's, and the web server, which polls the runs directory anyway, sends it to whoever is
looking at that run. There is one such file for
the whole runs directory, since only one evaluator works on a runs directory at a time and it
plays one game at a time. It is deleted when the games are over, and by an evaluator starting;
one left behind by an evaluator that was killed says when it was last written, which is how a
reader tells it from a game still being played.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import chess
from pydantic import BaseModel, ConfigDict, ValidationError

from chess_ai.config import SampleGamesConfig
from chess_ai.evaluator import until_stopped
from chess_ai.game_session import GameSession, GameState
from chess_ai.match import GameLines, MatchGame, append_game, play_game, play_match, replaying
from chess_ai.openings import OpeningSet
from chess_ai.output import Say
from chess_ai.output import say as said_on_stdout
from chess_ai.players import (
    ModelDescription,
    Player,
    StockfishBackedPlayer,
    StockfishDescription,
    close_and_wait,
)
from chess_ai.position_view import GameOverReason
from chess_ai.probes import check_finite
from chess_ai.training.run_store import (
    TEMPORARY_SUFFIX,
    CheckpointInfo,
    RunError,
    RunReader,
    code_version,
    created_with,
    save_evaluation,
    start_partial_file,
    still_evaluated,
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
    """The game the evaluator is playing now, as ``live-game.json`` holds it: a sample game, or
    a game of the Stockfish ladder."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    suite: Literal["sample-games", "stockfish-ladder"] = SUITE
    """Which suite is playing it."""
    run: str
    step: int
    """The checkpoint playing it."""
    index: int
    """Where it will be in the checkpoint's games file of the suite, counting from 0."""
    games: int
    """How many games the checkpoint plays in all, in the suite."""
    opponent: Opponent
    level: int | None = None
    """The Elo Stockfish plays at, in a ladder; ``None`` for a sample game."""
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


def show_live_game(runs_dir: Path, live: LiveGame) -> None:
    """:func:`save_live_game`, as far as it can be: watching is a nicety, and the games
    themselves go on, and are saved as they end, whether or not they can be watched."""
    try:
        save_live_game(runs_dir, live)
    except OSError as e:
        LOGGER.debug("Cannot show the game being played: %s", e)


def clear_live_game(runs_dir: Path) -> None:
    """Say that no game is being played, by deleting the file that says one is."""
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


async def check_playable(engine: "InferenceEngine", openings: OpeningSet, rating: int) -> None:
    """Make sure ``engine`` has not diverged, before any game is played with it, as the probes
    do: a diverged network plays legal moves all the same, chosen from NaNs, and games of that
    would be taken for the games of a model.

    Raises:
        DivergedError: what it makes of the position after the first line of ``openings`` is
            not a number.
    """
    board = chess.Board()
    for move in openings.openings[0].moves:
        board.push(move)
    check_finite(
        await asyncio.to_thread(
            engine.evaluate, board, mover_rating=rating, opponent_rating=rating
        ),
        f"the position after {openings.openings[0].name}",
    )


StockfishStarter = Callable[[int, float], Awaitable[Player]]
"""Starts the Stockfish opponent, given the Elo and the move time: a player that is
:class:`~chess_ai.players.StockfishBackedPlayer`, so that the result can say how strong it was."""


class SampleGamesSuite:
    """The sample games, as an evaluation suite.

    ``stockfish`` starts the Stockfish opponent; it is only called for a config that asks for
    games against it. ``live`` is the runs directory to show
    the game being played in, or ``None`` to show nothing, as a one-off evaluation does: only
    the evaluator, of which there is one, may write the one file there is. ``say`` is handed a
    line as each game starts and ends, since a checkpoint's games can take an hour.
    """

    name = SUITE

    def __init__(
        self,
        settings: SampleGamesConfig,
        openings: OpeningSet,
        *,
        stockfish: StockfishStarter | None = None,
        live: Path | None = None,
        say: Say = said_on_stdout,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if settings.stockfish_games and stockfish is None:
            raise ValueError("games against Stockfish need a way to start it")
        self.settings = settings
        self.openings = openings
        self._stockfish = stockfish
        self._live = live
        self._say = say
        self._clock = clock

    def wants(self, run: RunReader, checkpoint: CheckpointInfo) -> bool:
        return True

    def after(self, run: RunReader) -> None:
        pass

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
        result = asyncio.run(until_stopped(playing, stop, saved))
        return summary(result)

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
        created = created_with(run, checkpoint, written)
        started = self._clock()
        await check_playable(engine, self.openings, settings.rating)
        model = ModelPlayer(
            engine, run=run.name, checkpoint=checkpoint.step, rating=settings.rating
        )
        total = settings.games + settings.stockfish_games
        lines = GameLines(
            self._say,
            f"{run.name}@{checkpoint.step} sample games",
            total=total,
            openings=self.openings,
        )
        stockfish: Player | None = None
        partial: Path | None = None
        listed: list[SampleGame] = []

        async def start() -> None:
            nonlocal stockfish
            assert self._stockfish is not None
            stockfish = await self._stockfish(settings.stockfish_elo, settings.stockfish_move_time)

        async def close() -> None:
            nonlocal stockfish
            player, stockfish = stockfish, None
            if player is not None:
                await close_and_wait(player)

        try:
            if settings.stockfish_games:
                # Before any game, so that an engine that will not start costs no games; and
                # tried again as a game's engine is, so that one that is only slow to start on a
                # stalled machine does not fail them all either.
                await replaying(
                    start,
                    close,
                    game=lambda: (
                        f"starting Stockfish {settings.stockfish_elo} for the sample games"
                    ),
                    again="it is started again in {seconds} seconds",
                )
                if not isinstance(stockfish, StockfishBackedPlayer):
                    raise TypeError(f"{stockfish.name} cannot say how strong it plays")
            partial = start_partial_file(run.evaluation_directory(checkpoint.step), f"{SUITE}.pgn")
            games_file = partial

            def kept(opponent: Opponent) -> Callable[[MatchGame], None]:
                def keep(game: MatchGame) -> None:
                    append_game(games_file, game)
                    listed.append(_listed(len(listed), opponent, game))
                    lines.ended(game)

                return keep

            def watch(opponent: Opponent, before: int) -> Callable[[int, GameSession], None]:
                def follow(number: int, session: GameSession) -> None:
                    lines.starting(number)
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
                lines.match(["model", "model"], games=settings.games, paired=False)
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
            if isinstance(stockfish, StockfishBackedPlayer):
                elo = stockfish.stockfish.elo
                lines.match(["model", f"Stockfish {elo}"], games=settings.stockfish_games)
                keep, follow = kept("stockfish"), watch("stockfish", settings.games)

                # Game by game rather than as a match, so that a game whose engine hangs is
                # played again against a new one, as a ladder's is.
                for number in range(1, settings.stockfish_games + 1):
                    game = await replaying(
                        lambda number=number: play_game(
                            model,
                            stockfish,
                            number,
                            openings=self.openings,
                            on_session=follow,
                            event=EVENT,
                            move_delay=settings.move_delay,
                        ),
                        close,
                        start=start,
                        game=lambda number=number: (
                            f"sample game {settings.games + number}/{total} against Stockfish {elo}"
                        ),
                    )
                    keep(game)
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
                current=lambda: still_evaluated(run, checkpoint, written, created),
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
            await close()

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
        show_live_game(self._live, live)


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
