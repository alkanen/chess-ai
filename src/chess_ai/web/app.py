"""The web server: the built frontend and the API, all under the configured path prefix.

Every route lives under the prefix so that a reverse proxy can forward requests
unchanged. The frontend build uses relative URLs only; the server tells the browser
the prefix at runtime by adding a ``<base href="{prefix}/">`` tag to ``index.html``.
"""

import asyncio
import html
import logging
import re
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, assert_never

import chess
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketClose, WebSocketState

from chess_ai import replay
from chess_ai.config import Config
from chess_ai.dataset import DatasetError, ManifestError
from chess_ai.game_session import (
    ActionRejectedError,
    ConsideringEvent,
    GameEvent,
    GameOverEvent,
    GameSession,
    GameState,
    GameStateEvent,
    MoveEvent,
    PausedEvent,
    ReplacedEvent,
    RequestEvent,
    TakebackEvent,
)
from chess_ai.pgn import MEDIA_TYPE as PGN_MEDIA_TYPE
from chess_ai.pgn import game_pgn, pgn_filename, save_game
from chess_ai.players import (
    DEFAULT_TEMPERATURE,
    MIN_TEMPERATURE,
    HumanPlayer,
    MoveRejectedError,
    Player,
    PlayerUnavailableError,
    RandomPlayer,
    SelectionStrategy,
    close_players,
)
from chess_ai.position_view import (
    InvalidFenError,
    PositionSnapshot,
    board_from_fen,
    snapshot,
)
from chess_ai.sample_games import SampleGamesResult, read_games, read_sample_games
from chess_ai.stockfish import (
    DEFAULT_MOVE_TIME,
    MAX_MOVE_TIME,
    MIN_MOVE_TIME,
    DeferredStockfishPlayer,
    StockfishError,
    StockfishInfo,
    describe_stockfish,
    start_stockfish,
)
from chess_ai.training.run_store import (
    CheckpointChoice,
    EvaluationEntry,
    RunError,
    RunNotes,
    RunReader,
    choose_checkpoint,
    list_runs,
    listed,
    open_run,
    save_notes,
)
from chess_ai.web.datasets import DEFAULT_PAGE as DEFAULT_DATASET_PAGE
from chess_ai.web.datasets import MAX_PAGE as MAX_DATASET_PAGE
from chess_ai.web.datasets import (
    DatasetGames,
    DatasetSummary,
    NoSuchDatasetError,
    dataset_game,
    dataset_games,
    describe_dataset,
    describe_datasets,
)
from chess_ai.web.engine_cache import CheckpointKey, EngineCache
from chess_ai.web.game_store import (
    GameStore,
    HumanRecord,
    ModelRecord,
    PlayerRecord,
    RandomRecord,
    StockfishRecord,
    StoredGame,
)
from chess_ai.web.games import (
    Access,
    GameLinks,
    GameRegistry,
    GamesClosedError,
    NoSuchGameError,
    Seat,
    TooManyGamesError,
)
from chess_ai.web.runs import (
    LatestMetricsCache,
    RunCheckpoints,
    RunEvent,
    RunStream,
    RunSummary,
    ShownProbes,
    describe_checkpoints,
    describe_run,
    show_probes,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
"""Where ``scripts/build-frontend.sh`` puts the built frontend."""

_HEAD_TAG = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)

NO_STORE = {"Cache-Control": "no-store"}
"""Kept by nobody, for a URL that means something else after every move.

Every answer the game routes give carries this, not only the file: a 404 is one of the
statuses a cache may keep of its own accord, and a stored "no such game" would go on being
served for a link that has since been given a game.
"""

CLIENT_GAVE_UP = 499
"""What a request whose sender went away is answered with, as nginx answers one.

No number of HTTP's own says it: the request was neither refused nor served, and the
one asking is no longer there to read whichever was chosen. 499 is what the proxy in
front of this server writes in its log for the same thing.
"""

IDLE_CHECKPOINT_POLL_SECONDS = 600.0
"""How often the loaded checkpoints are looked over for ones no game has played with lately."""

EXPIRY_POLL_SECONDS = 600.0
"""How often the games are looked over for ones nobody has moved in for too long. A game kept
ten minutes past its week is no matter."""

RUN_POLL_SECONDS = 1.0
"""How often a run being followed is looked at for new metrics and a new heartbeat.

The trainer is a process of its own that tells nobody when it has written something, so the
only way to learn it has is to look. Once a second costs a few small reads, and is as often as
anybody watching a chart could tell the difference.
"""

_WHICH_GAME = (
    "Which game of the file to replay, counting from zero. The headers of every game in "
    "it are returned whichever one this is."
)


class PlayerSpec(BaseModel):
    """What one side of a new game is to be played by.

    A tagged union rather than a name, because the kinds of player do not take the same
    settings: a checkpoint needs a run, a step, a rating and a way of choosing its move, and
    the engine that comes next needs a strength. ``kind`` is what tells them apart, so adding
    a kind adds a class here and takes nothing away.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class HumanSpec(PlayerSpec):
    """A person at a browser, whose moves are submitted over the game WebSocket."""

    kind: Literal["human"]


class RandomSpec(PlayerSpec):
    """A uniformly random legal move, which is what there was to play against first."""

    kind: Literal["random"]


class ModelSpec(PlayerSpec):
    """A checkpoint from a training run, playing by its policy."""

    kind: Literal["model"]

    run: str
    """The training run to take the weights from, by the name it is kept under."""
    checkpoint: CheckpointChoice = "best"
    """Which checkpoint of the run to play: "latest", "best", or a step number."""
    rating: int | None = Field(default=None, ge=0, le=4000)
    """The rating to play like, given to the network as both sides' rating. Left out, the
    position claims no rating at all, which is also something the model was trained on."""
    strategy: SelectionStrategy = "argmax"
    """"argmax" plays the most likely move every time; "sample" draws from the distribution."""
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=MIN_TEMPERATURE, le=10)
    """How flat the distribution is sampled from: below 1 sharpens towards the best move,
    above 1 flattens towards a coin toss. Not used when the model plays its best move."""
    seed: int | None = None
    """Fixes the sampling, so that the same game can be played twice. Left out, it is not."""


class StockfishSpec(PlayerSpec):
    """Stockfish, held to a strength by its calibrated limit."""

    kind: Literal["stockfish"]

    elo: int = Field(ge=0, le=4000)
    """The Elo to play at. Stockfish only supports a range (1320 to 3190 in Stockfish 19,
    for one); a strength outside it is played at the nearer end, and the game's state says so."""
    move_time: float = Field(default=DEFAULT_MOVE_TIME, ge=MIN_MOVE_TIME, le=MAX_MOVE_TIME)
    """Seconds Stockfish thinks about each move. Its levels assume a few seconds a move; much
    less and it plays below the level asked for."""


AnyPlayer = Annotated[
    HumanSpec | RandomSpec | ModelSpec | StockfishSpec, Field(discriminator="kind")
]
"""What either colour may be played by; see :class:`PlayerSpec`."""


class NewGameRequest(BaseModel):
    # The field docstrings below describe the request in the API docs, which is the
    # only place someone choosing a move delay has to go on.
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    white: AnyPlayer
    black: AnyPlayer
    move_delay: float = Field(default=0.5, ge=0, le=10)
    """The least number of seconds before a move a player works out for itself, so that a
    game between players that move instantly can be followed. A move a human submits is
    played as soon as it arrives."""
    fen: str | None = None
    """The position to start the game from, which the side it gives the move opens from.
    The standard starting position is used when this is left out."""


class ViewerAction(BaseModel):
    """Something a viewer asks of the game their link reaches, on behalf of the link's side.

    Which game, and which side, is the link's to say: the WebSocket is opened for one link,
    so nothing sent over it names either.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class SubmitMove(ViewerAction):
    """A move for the link's side, which has to be on move."""

    type: Literal["move"]
    uci: str


class Resign(ViewerAction):
    """The link's side resigns."""

    type: Literal["resign"]


class Abort(ViewerAction):
    """End the game with no result and delete it, or ask the other person to agree to that."""

    type: Literal["abort"]


class TakeBack(ViewerAction):
    """Take back the side's last move, or ask the other person to agree to that."""

    type: Literal["takeback"]


class Answer(ViewerAction):
    """Agree to, or decline, what the other person asked."""

    type: Literal["answer"]
    request: int
    """The ``id`` of the request being answered, so that an answer meant for one request
    cannot land on another asked after it."""
    accept: bool


ViewerMessage = Annotated[
    SubmitMove | Resign | Abort | TakeBack | Answer, Field(discriminator="type")
]
"""What a viewer may send over the game WebSocket."""

_VIEWER_MESSAGE = TypeAdapter(ViewerMessage)


class ErrorEvent(BaseModel):
    """The server would not act on what a viewer sent, and nothing has changed."""

    type: Literal["error"] = "error"
    message: str


class SeatView(BaseModel):
    """A game as the link it was reached through sees it: the first thing the WebSocket sends.

    ``access`` is what the link may do, which is all the browser has to go on in offering it.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    type: Literal["state"] = "state"
    game: GameState
    access: Access
    """"white" or "black" for a play link, "control" for the one link to a game nobody plays
    by hand, which can abort it, and "watch" for a link that can only follow the game."""
    watch: str
    """The game's watch link, to pass on to anyone who wants to follow it."""
    updated: datetime
    """When the game last changed: when it began, its last move or takeback, when a player was
    replaced, or when it ended."""


class NewGame(BaseModel):
    """A game that has just been started, and every link to it."""

    game: GameState
    links: GameLinks


ViewerEvent = (
    SeatView
    | MoveEvent
    | TakebackEvent
    | RequestEvent
    | GameOverEvent
    | PausedEvent
    | ReplacedEvent
    | ConsideringEvent
    | ErrorEvent
)
"""What the game WebSocket sends: the game's events, plus this viewer's own errors."""


def create_app(
    config: Config, static_dir: Path = STATIC_DIR, *, run_poll_seconds: float = RUN_POLL_SECONDS
) -> FastAPI:
    prefix = config.server.path_prefix
    stale_after = config.server.stale_after_seconds
    latest_metrics = LatestMetricsCache()

    def save_finished_game(game: GameState) -> None:
        """Keep a finished game, so that it can be looked at again or trained on."""
        logger.info("Saved the finished game to %s", save_game(game, config.paths.games))

    store = GameStore(config.paths.ongoing_games)
    games = GameRegistry(
        on_finished=save_finished_game,
        max_ongoing=config.games.max_ongoing,
        store=store,
        expire_after=timedelta(days=config.games.expire_after_days),
    )
    engines = EngineCache(
        lambda key: _load_checkpoint(key, config),
        capacity=config.games.max_loaded_checkpoints,
        idle_after=config.games.checkpoint_idle_hours * 3600,
        on_unload=_hand_back_device_memory,
    )
    # One game starts at a time. Making a model player can read a checkpoint off the disk,
    # which the handler suspends for, and the limit on games in progress is counted before
    # that: two starts interleaving there could both be let in under the limit.
    starting = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _restore_games(store, games, config, engines)
        unloading = asyncio.create_task(_unload_idle_checkpoints(engines))
        expiring = asyncio.create_task(_expire_games(games))
        try:
            yield
        finally:
            unloading.cancel()
            expiring.cancel()
            await games.close()
            engines.clear()

    app = FastAPI(
        title="chess-ai",
        lifespan=lifespan,
        docs_url=f"{prefix}/api/docs",
        redoc_url=None,
        openapi_url=f"{prefix}/api/openapi.json",
        # Defaults to /docs/oauth2-redirect, outside the prefix. The app has no
        # authentication, so the docs have no use for it.
        swagger_ui_oauth2_redirect_url=None,
    )

    api = APIRouter(prefix="/api")

    @api.get("/start-position")
    def start_position() -> PositionSnapshot:
        return snapshot(chess.Board())

    @api.post(
        "/games",
        responses={
            400: {
                "description": "The position cannot be played from, or a player cannot be "
                "made: no such run, no such checkpoint, or a checkpoint that cannot be loaded"
            },
            409: {"description": "As many games are in progress as the server holds"},
            500: {
                "description": "A model was asked for and the inference device this server "
                "is configured with is not there on this machine, or Stockfish was asked for "
                "and the configured binary is missing or is not Stockfish"
            },
            503: {"description": "The server is shutting down and is starting no more games"},
        },
    )
    async def new_game(request: NewGameRequest) -> NewGame:
        """Start a new game, alongside any others, and return the links that reach it.

        The game gets a watch link, a play link for each side a person plays, and, if nobody
        plays either side, a control link that can abort it. The links are the only way to the
        game: there is no list of games, and whoever holds a link can do what it allows.

        A game starts from the standard starting position unless a FEN says otherwise.
        Anything that would stop the game being started is refused: a FEN that cannot be
        played from, a checkpoint that is not there, or as many games in progress as the server
        holds. Not every refusal is the asker's doing — a configured inference device that this
        machine does not have, or a Stockfish that is not installed where the config says, is
        answered as the server's own fault, and a server on its way down starts nothing at all.

        Every Stockfish side is an engine process of its own, which is stopped when its game
        ends or the server goes down, and at once if the game is refused. A checkpoint is
        shared by every game playing it.
        """
        # The position is checked before the players are made, so that a mistyped FEN is
        # answered at once rather than after tens of megabytes of checkpoint have been read.
        if request.fen is not None:
            try:
                board_from_fen(request.fen)
            except InvalidFenError as invalid:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(invalid)) from invalid
        async with starting:
            return await start(request)

    async def start(request: NewGameRequest, follows: Seat | None = None) -> NewGame:
        """Start the game ``request`` asks for, or say why it cannot be started.

        Called holding ``starting``.
        """
        # Counted before the players are made as well as when the game starts, so that a
        # full server does not start an engine or load a checkpoint only to refuse.
        try:
            games.check_room()
        except TooManyGamesError as full:
            raise HTTPException(status.HTTP_409_CONFLICT, str(full)) from full
        (white, white_record), (black, black_record) = await _players(request, config, engines)
        try:
            session = GameSession(white, black, move_delay=request.move_delay, fen=request.fen)
            links = games.start(
                session, settings=request, follows=follows, players=(white_record, black_record)
            )
        except GamesClosedError as closed:
            close_players(white, black)
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "the server is shutting down"
            ) from closed
        except TooManyGamesError as full:
            close_players(white, black)
            raise HTTPException(status.HTTP_409_CONFLICT, str(full)) from full
        except BaseException:
            close_players(white, black)
            raise
        return NewGame(game=session.state, links=links)

    @api.post(
        "/games/{link}/rematch",
        responses={
            400: {"description": "A player cannot be made again, such as a run that is gone"},
            403: {"description": "The link only watches the game"},
            404: {"description": "No game has this link"},
            409: {
                "description": "As many games are in progress as the server holds, or the "
                "game's settings were not kept"
            },
            500: {"description": "As for starting a game"},
            503: {"description": "The server is shutting down and is starting no more games"},
        },
    )
    async def rematch(link: str) -> NewGame:
        """Start a new game with the settings the game ``link`` reaches was started with.

        The players are asked for as they were the first time: a checkpoint chosen as "latest"
        or "best" is chosen again, and may be a newer one now. The new game has links of its
        own. A watch link cannot start one: it is for following a game, not for playing.

        Once one has been started from this game, asking again joins it for as long as it is
        being played, rather than starting another: two people who both ask to play again want
        one game between them. Joining hands over only the asker's own link and the watch link.
        """
        seat = _seat(link)
        if seat.access == "watch":
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "a watch link cannot start a game from this one"
            )
        settings = seat.settings
        if not isinstance(settings, NewGameRequest):
            # Kept by an older server in a form this one cannot read.
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "this game cannot be played again: what it was started with was not kept",
            )
        # Under the lock, so that two people asking at once get one game, not one each.
        async with starting:
            joined = seat.next_game()
            if joined is not None:
                return NewGame(game=joined.state, links=joined.own_links)
            return await start(settings, follows=seat)

    @api.post(
        "/games/{link}/replace",
        responses={
            400: {"description": "The checkpoint cannot be played, as for starting a game"},
            403: {"description": "The link only watches the game"},
            404: {"description": "No game has this link"},
            409: {"description": "The game is not waiting for another player"},
            500: {"description": "As for starting a game"},
        },
    )
    async def replace_player(link: str, spec: ModelSpec) -> SeatView:
        """Hand the side a paused game is waiting on to another checkpoint, and play on.

        A game pauses when the checkpoint on move is gone: deleted, or replaced by other
        weights, since the game began with it. Any link but a watch link may choose another,
        which plays that side from the position the game is in. The game records the change,
        and its PGN names the checkpoint that finished the game and says where it took over.
        """
        seat = _seat(link)
        if seat.access == "watch":
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "a watch link cannot choose the players of a game"
            )
        # Asked before the checkpoint is loaded, so that nothing is read for a game that does
        # not need it; and again once it is, by `replace`, since the game may have moved on.
        if seat.state.paused is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "the game is not waiting for another player"
            )
        player, record = await _player(spec, config, engines)
        try:
            seat.replace(player, record)
        except ActionRejectedError as rejected:
            close_players(player)
            raise HTTPException(status.HTTP_409_CONFLICT, str(rejected)) from rejected
        return _seat_view(seat)

    @api.get(
        "/stockfish",
        responses={500: {"description": "The configured Stockfish is missing or is not Stockfish"}},
    )
    async def stockfish() -> StockfishInfo:
        """Which Stockfish games are played against here, and the strengths it plays at.

        Asked of the engine itself, by starting it, since the range differs between versions;
        so this is also how to find out, before starting a game, whether there is one.
        """
        try:
            return await describe_stockfish(config.stockfish.path)
        except StockfishError as unavailable:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(unavailable)
            ) from unavailable

    # The run routes below read run directories, which the trainer may be writing to at this
    # moment. Plain `def`, so that FastAPI runs them in a worker thread: the reads are small,
    # but they are file reads all the same, and the game is being played on the event loop.

    @api.get("/runs")
    def runs(
        response: Response,
        tag: Annotated[
            list[str] | None,
            Query(description="Only the runs tagged with this; given more than once, with all."),
        ] = None,
        archived: Annotated[
            bool,
            Query(
                description="Include the runs tagged archived, which are otherwise left out "
                "unless that tag is asked for."
            ),
        ] = False,
    ) -> list[RunSummary]:
        """Every training run here, newest first: where each has got to and what it measured.

        A run that says it is running and whose heartbeat is older than the configured
        ``stale_after_seconds`` is flagged ``stale``: its trainer has most likely died.

        Kept by nobody: a run saves a checkpoint and logs metrics at moments of its own, and
        a list that a browser had kept would go on showing those of an hour ago.
        """
        response.headers.update(NO_STORE)
        readers = [RunReader(config.paths.runs / name) for name in list_runs(config.paths.runs)]
        found = [
            describe_run(run, stale_after=stale_after, latest=latest)
            for run, latest in zip(readers, latest_metrics.follow(readers), strict=True)
        ]
        # Filtered after describing rather than before, so that the metrics of the runs left
        # out go on being followed and the next unfiltered list does not read them whole.
        found = [run for run in found if listed(run.tags, wanted=tag or (), archived=archived)]
        # Newest first, because the run someone wants to play against is almost always the
        # one they are training now. A run that does not say when it began sorts last.
        found.sort(key=lambda run: run.created.timestamp() if run.created else 0.0, reverse=True)
        return found

    @api.get(
        "/runs/{name}/checkpoints",
        responses={404: {"description": "No run of that name is kept here"}},
    )
    def run_checkpoints(name: str, response: Response) -> RunCheckpoints:
        """The checkpoints of one run, newest first, to choose which to play against."""
        response.headers.update(NO_STORE)
        try:
            run = open_run(config.paths.runs, name)
        except RunError as missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(missing)) from missing
        return describe_checkpoints(run)

    @api.get(
        "/runs/{name}/evaluations",
        responses={404: {"description": "No run of that name is kept here"}},
    )
    def run_evaluations(name: str, response: Response) -> list[EvaluationEntry]:
        """Which suites have a result about which of a run's checkpoints, by step and suite.

        Checkpoints the run has since pruned are among them: their results are kept.
        """
        response.headers.update(NO_STORE)
        return _open_run(name).evaluations()

    @api.get(
        "/runs/{name}/evaluations/{step}/probe-positions",
        responses={
            404: {"description": "No run of that name, or no probe result for that step"},
            500: {"description": "The result is there and cannot be read"},
        },
    )
    def run_probes(name: str, step: int, response: Response) -> ShownProbes:
        """What a checkpoint made of the probe positions, each with its position to draw.

        Replaced when the checkpoint is probed again, so it is kept by nobody.
        """
        response.headers.update(NO_STORE)
        run = _open_run(name)
        try:
            shown = show_probes(run, step)
        except RunError as unreadable:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(unreadable)
            ) from unreadable
        if shown is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"run {name!r} has no probe result for the checkpoint from step {step}",
            )
        return shown

    @api.get(
        "/runs/{name}/evaluations/{step}/sample-games",
        responses={
            404: {"description": "No run of that name, or no sample games for that step"},
            500: {"description": "The result is there and cannot be read"},
        },
    )
    def run_sample_games(name: str, step: int, response: Response) -> SampleGamesResult:
        """Which sample games a checkpoint played, and how each ended.

        Replaced when the checkpoint plays them again, so it is kept by nobody.
        """
        response.headers.update(NO_STORE)
        run = _open_run(name)
        try:
            result = read_sample_games(run, step)
        except RunError as unreadable:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(unreadable)
            ) from unreadable
        if result is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"run {name!r} has no sample games for the checkpoint from step {step}",
            )
        return result

    @api.get(
        "/runs/{name}/evaluations/{step}/sample-games/replay",
        responses={
            404: {"description": "No run of that name, no sample games, or no such game"},
            500: {"description": "The games are there and cannot be read"},
        },
    )
    def run_sample_game(
        name: str,
        step: int,
        response: Response,
        game: Annotated[int, Query(ge=0, description=_WHICH_GAME)] = 0,
    ) -> replay.ReplayFile:
        """Replay one of a checkpoint's sample games, as the replay viewer opens a file."""
        response.headers.update(NO_STORE)
        run = _open_run(name)
        try:
            pgn = read_games(run, step)
        except RunError as unreadable:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(unreadable)
            ) from unreadable
        if pgn is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"run {name!r} has no sample games for the checkpoint from step {step}",
            )
        return _replayed(pgn, game)

    @api.get(
        "/runs/{name}/notes",
        responses={
            404: {"description": "No run of that name is kept here"},
            500: {"description": "The run's notes are there and cannot be read"},
        },
    )
    def run_notes(name: str, response: Response) -> RunNotes:
        """A run's title, tags and notes, empty for a run nobody has written any for."""
        response.headers.update(NO_STORE)
        run = _open_run(name)
        try:
            return run.notes
        except RunError as unreadable:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(unreadable)
            ) from unreadable

    @api.put(
        "/runs/{name}/notes",
        responses={
            404: {"description": "No run of that name is kept here"},
            500: {"description": "The notes could not be written to the run directory"},
        },
    )
    def edit_run_notes(name: str, notes: RunNotes) -> RunNotes:
        """Replace a run's title, tags and notes, all three at once, and return them as kept.

        Kept in the run directory, where the command line reads and writes them too. Whatever
        is sent replaces what was there: a field left out is emptied, not kept.
        """
        run = _open_run(name)
        try:
            return save_notes(run, notes)
        except RunError as unwritable:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(unwritable)
            ) from unwritable

    # The dataset routes read files a build may be replacing at this moment, as the run routes
    # do, and are plain `def` for the same reason. A dataset is opened per request and let go
    # of at the end of it; see chess_ai.web.datasets.

    @api.get("/datasets")
    def datasets() -> list[DatasetSummary]:
        """Every dataset here, by name, each with its manifest or why that cannot be read."""
        return describe_datasets(config.paths.data)

    @api.get("/datasets/{name}", responses={404: {"description": "No dataset of that name"}})
    def dataset(name: str) -> DatasetSummary:
        """One dataset's manifest: its sources, filters, counts and statistics."""
        with _dataset_errors():
            return describe_dataset(config.paths.data, name)

    @api.get(
        "/datasets/{name}/{split}/games",
        responses={
            404: {"description": "No dataset of that name, or no such split in it"},
            500: {"description": "The dataset is there and cannot be read"},
        },
    )
    def dataset_games_page(
        name: str,
        split: str,
        offset: Annotated[int, Query(ge=0, description="The first game, counting from zero.")] = 0,
        limit: Annotated[
            int, Query(ge=1, le=MAX_DATASET_PAGE, description="How many games at most.")
        ] = DEFAULT_DATASET_PAGE,
    ) -> DatasetGames:
        """One page of a split's games, in the order they were stored."""
        with _dataset_errors():
            return dataset_games(config.paths.data, name, split, offset, limit)

    @api.get(
        "/datasets/{name}/{split}/games/{index}",
        responses={
            404: {"description": "No such dataset, split or game"},
            500: {"description": "The game's records cannot be read or played out"},
        },
    )
    def dataset_game_replay(name: str, split: str, index: int) -> replay.ReplayGame:
        """One game of a dataset, with the position before every move and after it."""
        with _dataset_errors():
            return dataset_game(config.paths.data, name, split, index)

    def _open_run(name: str) -> RunReader:
        try:
            return open_run(config.paths.runs, name)
        except RunError as missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(missing)) from missing

    def _seat(link: str) -> Seat:
        try:
            return games.seat(link)
        except NoSuchGameError as missing:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, str(missing), headers=NO_STORE
            ) from missing

    @api.get("/games/{link}", responses={404: {"description": "No game has this link"}})
    async def game(link: str, response: Response) -> SeatView:
        """A game as ``link`` sees it: its state, and what the link may do in it.

        Read on the event loop, as the PGN below is, because that is where moves are made.
        """
        response.headers.update(NO_STORE)
        return _seat_view(_seat(link))

    @api.get(
        "/games/{link}/pgn",
        response_class=Response,
        responses={
            200: {"content": {PGN_MEDIA_TYPE: {}}, "description": "The game as a PGN file"},
            404: {"description": "No game has this link"},
        },
    )
    async def game_pgn_file(link: str) -> Response:
        """Download a game as PGN, whether it has finished or not, through any of its links.

        A game in progress is described as far as it has been played, with the result "*"
        that PGN gives a game that has not ended.
        """
        # Read here on the event loop, which a plain `def` would not be: FastAPI runs
        # those in a worker thread, and a game read there while a move is being made can
        # come out claiming a checkmate that is not among its moves.
        current = _seat(link).state
        # One moment dates the file and names it, so that the two cannot disagree.
        now = datetime.now()
        return Response(
            game_pgn(current, now=now),
            media_type=PGN_MEDIA_TYPE,
            headers={
                "Content-Disposition": f'attachment; filename="{pgn_filename(current, now=now)}"',
                **NO_STORE,
            },
        )

    # The reading of a PGN file happens in a worker thread, either because the route is
    # a plain `def` or, where it has a body to take first, by being handed to one:
    # reading a file of a few thousand games takes long enough to be felt by the game
    # being played on the event loop meanwhile. It touches nothing that game touches.

    @api.get("/replay/saved")
    def saved_games(response: Response) -> list[replay.SavedGame]:
        """Every game saved here, the most recently played first.

        Kept by nobody: a game finishing adds to this list at a moment of its own.
        """
        response.headers.update(NO_STORE)
        return replay.saved_games(config.paths.games)

    @api.get("/replay/saved/{name}", responses={404: {"description": "No such saved game"}})
    def saved_game(
        name: str,
        game: Annotated[int, Query(ge=0, description=_WHICH_GAME)] = 0,
    ) -> replay.ReplayFile:
        """Open a saved game, by the ``name`` the list of saved games gives it."""
        try:
            return _replayed(replay.saved_game(config.paths.games, name), game)
        except replay.NoSuchGameError as missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(missing)) from missing

    @api.post(
        "/replay/pgn",
        openapi_extra={
            "requestBody": {
                "description": "A PGN file.",
                "required": True,
                "content": {PGN_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
            }
        },
        responses={
            400: {"description": "The file holds no game, or one that cannot be read"},
            404: {"description": "The file has no game with that number"},
            413: {"description": "The file is larger than the server will read"},
            499: {"description": "The upload was given up before it was all sent"},
        },
    )
    async def replay_pgn(
        request: Request,
        game: Annotated[int, Query(ge=0, description=_WHICH_GAME)] = 0,
    ) -> replay.ReplayFile:
        """Read a PGN file, and replay one of the games in it.

        The file itself is not kept: looking at a second game in it posts it again.

        The body is taken as it arrives rather than declared as a parameter, because a
        parameter is handed over whole: the body would be in memory before anything
        here could say it was too large, which is the one thing the limit is for.
        """
        pgn = await _read_at_most(request.stream(), replay.MAX_BYTES)
        # Back onto a worker thread for the reading itself, as the routes above, so that
        # a file of a few thousand games is not felt by the game being played meanwhile.
        # The bytes go over as they are: decoding them is part of that reading, and a
        # file that is not UTF-8 is two passes over every byte of it.
        return await run_in_threadpool(_replayed, pgn, game)

    @api.websocket("/games/{link}/ws")
    async def follow_game(websocket: WebSocket, link: str) -> None:
        """Send the game's full state as ``link`` sees it and every event after it, and do what
        the viewer asks, as far as the link allows.

        A link that reaches no game is answered with an ``error`` event, and the connection is
        closed. So is the connection to a game that stops: one that has ended has nothing more
        to send, and one that was aborted is gone.
        """
        await websocket.accept()
        connection = _Connection(websocket)
        try:
            seat = games.seat(link)
        except NoSuchGameError as missing:
            await connection.send(ErrorEvent(message=str(missing)))
            await connection.close(status.WS_1008_POLICY_VIOLATION)
            return
        # Either side can end the connection: the viewer goes away, or the game stops and
        # has nothing left to send.
        async with asyncio.TaskGroup() as tasks:
            sending = tasks.create_task(_send_game_events(connection, seat))
            receiving = tasks.create_task(_act_on_viewer_messages(connection, seat))
            await asyncio.wait({sending, receiving}, return_when=asyncio.FIRST_COMPLETED)
            sending.cancel()
            receiving.cancel()

    @api.websocket("/runs/{name}/ws")
    async def follow_run(websocket: WebSocket, name: str) -> None:
        """Send what a run is, its whole metrics log, which of its checkpoints have evaluation
        results, and then whatever changes in any of them.

        The first four messages are a ``run`` event, a ``metrics`` event with ``reset`` set, an
        ``evaluations`` event, which lists the results and says which probe set the evaluator
        probes with, and a ``live_game`` event with the sample game being played with one of the
        run's checkpoints, if any. After that a ``run`` event comes whenever the run's
        description, notes or heartbeat change or the heartbeat goes stale, a ``metrics`` event
        whenever the log has grown, an ``evaluations`` event, with every result again, whenever a
        result is added or replaced or the evaluator says it probes with another set, and a
        ``live_game`` event whenever the sample game moves on, another begins, or they are over.
        A run that is not here is answered with an ``error`` event, and the connection is closed.
        """
        await websocket.accept()
        connection = _Connection(websocket)
        try:
            run = open_run(config.paths.runs, name)
        except RunError as missing:
            await connection.send(ErrorEvent(message=str(missing)))
            await connection.close(status.WS_1008_POLICY_VIOLATION)
            return
        stream = RunStream(run, stale_after=stale_after)
        async with asyncio.TaskGroup() as tasks:
            sending = tasks.create_task(_send_run_events(connection, stream, run_poll_seconds))
            # Nothing a viewer sends is acted on; listening is how their going away is heard.
            waiting = tasks.create_task(_until_disconnected(connection))
            await asyncio.wait({sending, waiting}, return_when=asyncio.FIRST_COMPLETED)
            sending.cancel()
            waiting.cancel()

    # A WebSocket route matches only an upgrade, so a plain request on its path would fall
    # through to a 404 that reads as the API missing from under the prefix.
    for websocket_path in ("/games/{link}/ws", "/runs/{name}/ws"):
        api.add_api_route(
            websocket_path, _upgrade_required, methods=["GET"], include_in_schema=False
        )

    app.include_router(api, prefix=prefix)

    if prefix:

        @app.get(prefix, include_in_schema=False)
        def add_trailing_slash() -> RedirectResponse:
            # A path-only Location keeps the redirect correct behind a proxy.
            return RedirectResponse(f"{prefix}/")

    @app.get(f"{prefix}/", include_in_schema=False)
    def index() -> Response:
        index_file = static_dir / "index.html"
        if not index_file.is_file():
            return PlainTextResponse(
                "The frontend has not been built. Run scripts/build-frontend.sh.",
                status_code=503,
            )
        page = _with_base_href(index_file.read_text(encoding="utf-8"), f"{prefix}/")
        # Asset file names change with every build, so the page must not be cached.
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    app.mount(prefix or "/", _FrontendFiles(directory=static_dir, check_dir=False), name="static")

    return app


Made = tuple[Player, PlayerRecord]
"""A player, and what it takes to make it again after a restart."""


async def _players(
    request: NewGameRequest, config: Config, engines: EngineCache
) -> tuple[Made, Made]:
    """Both sides of a new game, made from what was asked for.

    Whatever the first side holds is let go of again if the second cannot be made, so that a
    refused game leaves no engine process running.

    Raises:
        HTTPException: a player cannot be made. A run, a checkpoint or the weights in it that
            are not what the request said are answered as a bad request rather than a fault of
            the server's: the person who asked chose the run and the checkpoint, and the
            message names what was wrong with the choice. The one refusal that is not theirs
            is an inference device this machine does not have, which nobody choosing a player
            picked; :func:`_model_player` answers that as the server's own fault. A Stockfish
            that is not there, or is not Stockfish, is the same kind of fault.
    """
    white = await _player(request.white, config, engines)
    try:
        black = await _player(request.black, config, engines)
    except BaseException:
        close_players(white[0])
        raise
    return white, black


async def _player(spec: AnyPlayer, config: Config, engines: EngineCache) -> Made:
    match spec:
        case HumanSpec():
            return HumanPlayer(), HumanRecord()
        case RandomSpec():
            return RandomPlayer(), RandomRecord()
        case ModelSpec():
            # On a worker thread: a model player reads a checkpoint off the disk and builds a
            # network from it, which the game being played meanwhile, and every viewer watching
            # it, should not be held up by.
            try:
                return await run_in_threadpool(_model_player, spec, config, engines)
            except RunError as missing:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(missing)) from missing
        case StockfishSpec():
            try:
                engine = await start_stockfish(
                    config.stockfish.path, elo=spec.elo, move_time=spec.move_time
                )
            except StockfishError as unavailable:
                # Nobody who clicked Start installed the engine or wrote the config.
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, str(unavailable)
                ) from unavailable
            record = StockfishRecord(
                elo=spec.elo, move_time=spec.move_time, description=engine.stockfish
            )
            return engine, record
        case _:
            # The union is closed, so this is unreachable until somebody adds a kind of
            # player and not the case that makes it. Said here, where the omission is, rather
            # than left to fail on the first move of a game with nobody playing it.
            assert_never(spec)


def _model_player(spec: ModelSpec, config: Config, engines: EngineCache) -> Made:
    """A checkpoint, loaded and sat down at the board.

    The checkpoint is borrowed from ``engines``, which every game shares: a checkpoint playing
    in several games, or playing itself at two ratings, is one set of weights and several ways
    of choosing from them. Everything a player was asked for — the rating, the strategy, its
    generator — lives in the player, so the engine has nothing of any game in it.

    It is loaded here, if it is not loaded already, so that a checkpoint that cannot be played
    is refused when the game is asked for rather than on its first move.
    """
    from chess_ai.inference import DeviceUnavailableError, InferenceError

    run = open_run(config.paths.runs, spec.run)
    chosen = choose_checkpoint(run, spec.checkpoint)
    key = CheckpointKey(run.name, chosen.step, _fingerprint(run.checkpoint_path(chosen)))
    record = ModelRecord(
        run=key.run,
        step=key.step,
        fingerprint=key.fingerprint,
        rating=spec.rating,
        strategy=spec.strategy,
        temperature=spec.temperature,
        seed=spec.seed,
    )
    try:
        engines.preload(key)
    except DeviceUnavailableError as misconfigured:
        # Nobody who clicked Start chose the device; the machine this server runs on did.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, str(misconfigured)
        ) from misconfigured
    except InferenceError as unusable:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(unusable)) from unusable
    return _model_player_from(record, engines), record


def _model_player_from(record: ModelRecord, engines: EngineCache) -> Player:
    """The checkpoint ``record`` names, sat down at the board without being loaded yet."""
    from chess_ai.inference import ModelPlayer

    return ModelPlayer(
        engines.engine(CheckpointKey(record.run, record.step, record.fingerprint)),
        run=record.run,
        checkpoint=record.step,
        rating=record.rating,
        strategy=record.strategy,
        temperature=record.temperature,
        seed=record.seed,
    )


def _revive(record: PlayerRecord, config: Config, engines: EngineCache) -> Player:
    """The player ``record`` was written of, holding nothing until its game next needs it.

    A checkpoint is loaded on its first move, and one that is gone by then pauses the game
    rather than keeping it from being taken back at all; Stockfish is started on its first move.
    """
    match record:
        case HumanRecord():
            return HumanPlayer()
        case RandomRecord():
            return RandomPlayer()
        case ModelRecord():
            return _model_player_from(record, engines)
        case StockfishRecord():
            return DeferredStockfishPlayer(
                config.stockfish.path,
                elo=record.elo,
                move_time=record.move_time,
                description=record.description,
            )
        case _:
            assert_never(record)


def _restore_games(
    store: GameStore, games: GameRegistry, config: Config, engines: EngineCache
) -> None:
    """Take back every game kept in ``store``, as the server starts, then let go of those that
    have expired meanwhile."""
    restored = 0
    for stored in store.read_all():
        try:
            _restore_game(stored, games, config, engines)
        except Exception:
            # One game that cannot be taken back is no reason to lose the others, or for the
            # server not to start. Its file is left where it is, for somebody to look at.
            logger.exception("Could not take back game %s, so it is left out", stored.id)
            continue
        restored += 1
    if restored:
        logger.info("Took back %d games from %s", restored, store.directory)
    games.expire()


def _restore_game(
    stored: StoredGame, games: GameRegistry, config: Config, engines: EngineCache
) -> None:
    settings = None
    if stored.settings is not None:
        try:
            settings = NewGameRequest.model_validate(stored.settings)
        except ValidationError:
            # Only playing it again needs them, which is a small thing to lose beside the game.
            logger.warning(
                "Cannot read what game %s was started with, so it cannot be played again",
                stored.id,
                exc_info=True,
            )
    white = _revive(stored.white, config, engines)
    try:
        black = _revive(stored.black, config, engines)
        games.restore(stored, white, black, settings)
    except BaseException:
        close_players(white)
        raise


class CheckpointGoneError(RunError, PlayerUnavailableError):
    """A game's checkpoint is no longer there to be loaded, or has been replaced by another.

    Both things at once: a run that is not as the game was asked to find it, which starting a
    game refuses, and a player that cannot go on, which pauses a game in progress.
    """


def _load_checkpoint(key: CheckpointKey, config: Config) -> Any:
    """Read checkpoint ``key`` off the disk and build an engine that plays with it.

    The file has to be the one ``key`` was taken from, before the read and after it: a game
    that began with one set of weights is not to go on with another under the same name.

    The inference package is imported here rather than at the top of this module, because it
    is what brings in torch: a server that is replaying games and playing people against the
    random mover has no use for it, and starting up is a second or two quicker without it.

    Raises:
        CheckpointGoneError: the run, or the checkpoint, is not there (any more), or has been
            replaced.
        InferenceError: the checkpoint cannot be played with, or the device is not there.
    """
    from chess_ai.inference import load_engine

    try:
        run = open_run(config.paths.runs, key.run)
        path = run.checkpoint_path(choose_checkpoint(run, key.step))
        _check_unchanged(path, key)
        engine = load_engine(
            path, device=config.inference.device, batch_size=config.inference.batch_size
        )
        _check_unchanged(path, key)
    except RunError as gone:
        raise CheckpointGoneError(str(gone)) from gone
    return engine


def _fingerprint(path: Path) -> tuple[int, ...]:
    """What tells this checkpoint file from another written in its place.

    Raises:
        RunError: the file is not there.
    """
    try:
        found = path.stat()
    except OSError as missing:
        raise RunError(f"cannot read the checkpoint {path}: {missing.strerror}") from missing
    return (found.st_ino, found.st_size, found.st_mtime_ns)


def _check_unchanged(path: Path, key: CheckpointKey) -> None:
    if _fingerprint(path) != key.fingerprint:
        raise RunError(
            f"{key.run} step {key.step} has been replaced since the game began with it, "
            "so the game cannot go on with it"
        )


def _hand_back_device_memory() -> None:
    """Give a GPU's memory back once a checkpoint on it has been let go of.

    PyTorch keeps the memory of freed tensors for itself, to hand out again, and a training run
    on the same card cannot have it until it is given back. Nothing to do if torch was never
    imported, or never touched a GPU.
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


async def _expire_games(games: GameRegistry) -> None:
    """Look over the games now and then, deleting those nobody has moved in for too long."""
    while True:
        await asyncio.sleep(EXPIRY_POLL_SECONDS)
        games.expire()


async def _unload_idle_checkpoints(engines: EngineCache) -> None:
    """Look over the loaded checkpoints now and then, letting go of those nobody plays."""
    while True:
        await asyncio.sleep(IDLE_CHECKPOINT_POLL_SECONDS)
        await run_in_threadpool(engines.unload_idle)


def _seat_view(seat: Seat, game: GameState | None = None) -> SeatView:
    """``seat``'s game as its link sees it, with ``game`` as its state if that is given."""
    return SeatView(
        game=game if game is not None else seat.state,
        access=seat.access,
        watch=seat.watch,
        updated=seat.updated,
    )


async def _read_at_most(body: AsyncIterator[bytes], limit: int) -> bytes:
    """The body being sent, refused the moment it passes ``limit`` bytes.

    Taken a piece at a time and counted as it goes, so that a body larger than the
    server will read costs the server the limit and not the size of the body: whoever
    is sending it decides how large it is, and nobody here has to believe the length
    they declared, or that they declared one at all.

    Raises:
        HTTPException: the body passed the limit, or whoever was sending it went away
            before it was all here. Neither is a fault of this server's.
    """
    read = bytearray()
    try:
        async for piece in body:
            read += piece
            if len(read) > limit:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"that file is larger than the {limit // 1024 // 1024} MB of PGN this "
                    "server will read at once",
                )
    except ClientDisconnect as gone:
        # Giving up on an upload is something people do on purpose, and at 8 MB over a
        # slow line it is the ordinary way to end one. Answered like any other request
        # the server will not act on, rather than raised through the server as a fault
        # and logged with a traceback nobody can do anything about.
        raise HTTPException(CLIENT_GAVE_UP, "the upload was given up") from gone
    return bytes(read)


@contextmanager
def _dataset_errors() -> Iterator[None]:
    """Answer a dataset that is not there with a 404, and one that cannot be read with a 500."""
    try:
        yield
    except NoSuchDatasetError as missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(missing)) from missing
    except (DatasetError, ManifestError) as damaged:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(damaged)) from damaged


def _replayed(pgn: str | bytes, selected: int) -> replay.ReplayFile:
    """``pgn`` read back, with what is wrong with it told to whoever sent it.

    A file that arrived as bytes is decoded here rather than by the caller, so that the
    decoding goes wherever the reading goes.
    """
    try:
        return replay.read(replay.decode(pgn) if isinstance(pgn, bytes) else pgn, selected=selected)
    except replay.MalformedPgnError as malformed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(malformed)) from malformed
    except replay.NoSuchGameError as missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(missing)) from missing


_UPGRADE_REQUIRED = (
    "This path is a WebSocket, and the request reached the server without the upgrade to "
    "one. A reverse proxy in front of the server is the most likely reason: it has to "
    "forward the Upgrade and Connection headers over HTTP/1.1. For nginx that is "
    "'proxy_http_version 1.1', 'proxy_set_header Upgrade $http_upgrade' and "
    "'proxy_set_header Connection \"upgrade\"', as in the deployment notes in the README."
)


def _upgrade_required() -> None:
    """Answer a plain request on a WebSocket path with what it is missing."""
    raise HTTPException(
        status.HTTP_426_UPGRADE_REQUIRED, _UPGRADE_REQUIRED, headers={"Upgrade": "websocket"}
    )


class _Connection:
    """One viewer's WebSocket, which the two tasks serving it take turns to write to."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._sending = asyncio.Lock()

    async def receive(self) -> Message:
        return await self._websocket.receive()

    async def send(self, event: ViewerEvent | RunEvent) -> None:
        async with self._sending:
            if self._websocket.application_state is WebSocketState.CONNECTED:
                await self._websocket.send_text(event.model_dump_json())

    async def close(self, code: int) -> None:
        async with self._sending:
            if self._websocket.application_state is WebSocketState.CONNECTED:
                await self._websocket.close(code)


async def _act_on_viewer_messages(connection: _Connection, seat: Seat) -> None:
    """Do what the viewer asks of the game, until the viewer goes away.

    Anything the game will not do, or the link may not, is reported to the viewer who asked
    and to nobody else, and the game goes on.
    """
    try:
        while (message := await connection.receive())["type"] != "websocket.disconnect":
            try:
                asked = _VIEWER_MESSAGE.validate_json(message.get("text") or "")
            except ValidationError:
                await connection.send(ErrorEvent(message="the server cannot read that message"))
                continue
            try:
                _act(seat, asked)
            except (ActionRejectedError, MoveRejectedError) as rejected:
                await connection.send(ErrorEvent(message=str(rejected)))
    except WebSocketDisconnect:
        pass  # The viewer has already gone.


def _act(seat: Seat, asked: ViewerMessage) -> None:
    match asked:
        case SubmitMove():
            seat.submit_move(asked.uci)
        case Resign():
            seat.resign()
        case Abort():
            seat.abort()
        case TakeBack():
            seat.take_back()
        case Answer():
            seat.answer(asked.request, accept=asked.accept)


async def _send_game_events(connection: _Connection, seat: Seat) -> None:
    """Send the game's events until it stops, then let the viewer go.

    The game's own first event, its full state, goes out as the link sees it.
    """
    try:
        with seat.subscribe() as events:
            async for event in events:
                await connection.send(_as_seen(event, seat))
        # Done for good once the game is over, which the browser need not ask about again.
        # Otherwise the server is going down, and the game will be there when it is back.
        over = seat.state.position.game_over is not None
        await connection.close(status.WS_1000_NORMAL_CLOSURE if over else status.WS_1001_GOING_AWAY)
    except WebSocketDisconnect:
        pass  # The viewer has already gone.


def _as_seen(event: GameEvent, seat: Seat) -> ViewerEvent:
    if isinstance(event, GameStateEvent):
        return _seat_view(seat, event.game)
    return event


async def _send_run_events(connection: _Connection, stream: RunStream, every: float) -> None:
    """Look at the run every ``every`` seconds, and send whatever has changed in it."""
    try:
        while True:
            # On a worker thread, as the run routes above, because these are file reads and the
            # game is being played on the event loop.
            try:
                events = await run_in_threadpool(stream.poll)
            except RunError:
                # A log this process may not read, for instance. Looked at again next time,
                # since whatever it was may well pass, and the viewer keeps what they have.
                logger.warning("cannot follow run %s", stream.name, exc_info=True)
                events = []
            for event in events:
                await connection.send(event)
            await asyncio.sleep(every)
    except WebSocketDisconnect:
        pass  # The viewer has already gone.


async def _until_disconnected(connection: _Connection) -> None:
    """Return once the viewer has gone, ignoring whatever they send meanwhile."""
    try:
        while (await connection.receive())["type"] != "websocket.disconnect":
            pass
    except WebSocketDisconnect:
        pass


class _FrontendFiles(StaticFiles):
    """The built frontend, whose directory may only appear after the server has started."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # A WebSocket to a path no route has, such as one a browser still running an older
        # frontend reconnects to, ends up here. Static files are HTTP only, and would fail
        # it with a server error; it is refused instead, as a route that is not there.
        if scope["type"] == "websocket":
            await WebSocketClose(status.WS_1008_POLICY_VIOLATION)(scope, receive, send)
            return
        await super().__call__(scope, receive, send)

    async def check_config(self) -> None:
        # StaticFiles raises on a missing directory. Until the frontend is built,
        # lookups 404 instead, and files built later are served without a restart.
        pass


def _with_base_href(page: str, href: str) -> str:
    base_tag = f'<base href="{html.escape(href)}">'
    page, count = _HEAD_TAG.subn(lambda m: m.group(0) + base_tag, page, count=1)
    if count == 0:
        raise ValueError("index.html has no <head> tag")
    return page
