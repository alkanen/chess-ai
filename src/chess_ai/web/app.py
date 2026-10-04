"""The web server: the built frontend and the API, all under the configured path prefix.

Every route lives under the prefix so that a reverse proxy can forward requests
unchanged. The frontend build uses relative URLs only; the server tells the browser
the prefix at runtime by adding a ``<base href="{prefix}/">`` tag to ``index.html``.
"""

import asyncio
import html
import logging
import re
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
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
from starlette.types import Message
from starlette.websockets import WebSocketState

from chess_ai import replay
from chess_ai.config import Config
from chess_ai.game_session import ActionRejectedError, GameSession, GameState
from chess_ai.pgn import MEDIA_TYPE as PGN_MEDIA_TYPE
from chess_ai.pgn import game_pgn, pgn_filename, save_game
from chess_ai.players import (
    DEFAULT_TEMPERATURE,
    MIN_TEMPERATURE,
    HumanPlayer,
    MoveRejectedError,
    Player,
    RandomPlayer,
    SelectionStrategy,
    close_players,
)
from chess_ai.position_view import (
    Color,
    InvalidFenError,
    PositionSnapshot,
    board_from_fen,
    snapshot,
)
from chess_ai.stockfish import (
    DEFAULT_MOVE_TIME,
    MAX_MOVE_TIME,
    MIN_MOVE_TIME,
    StockfishError,
    StockfishInfo,
    describe_stockfish,
    start_stockfish,
)
from chess_ai.training.run_store import (
    CheckpointChoice,
    RunError,
    RunNotes,
    RunReader,
    choose_checkpoint,
    list_runs,
    listed,
    open_run,
    save_notes,
)
from chess_ai.web.game_channel import ChannelEvent, GameChannel, GameChannelClosedError
from chess_ai.web.runs import (
    LatestMetricsCache,
    RunCheckpoints,
    RunEvent,
    RunStream,
    RunSummary,
    describe_checkpoints,
    describe_run,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
"""Where ``scripts/build-frontend.sh`` puts the built frontend."""

_HEAD_TAG = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)

NO_STORE = {"Cache-Control": "no-store"}
"""Kept by nobody, for a URL that means something else after every move.

Every answer the PGN route gives carries this, not only the file: a 404 is one of the
statuses a cache may keep of its own accord, and a stored "no game has been started"
would go on being served long after a game had started.
"""

CLIENT_GAVE_UP = 499
"""What a request whose sender went away is answered with, as nginx answers one.

No number of HTTP's own says it: the request was neither refused nor served, and the
one asking is no longer there to read whichever was chosen. 499 is what the proxy in
front of this server writes in its log for the same thing.
"""

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
    """Something a viewer asks of the game their browser is showing them.

    Every one of these names that game, because a new game can replace it in the moment
    before the message arrives, and the server acts on the game the viewer meant.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    game: str
    """The ``id`` of the game this is meant for, as its state gave it."""


class SubmitMove(ViewerAction):
    """A viewer plays a move for the human side to move."""

    type: Literal["move"]
    uci: str


class Resign(ViewerAction):
    """A viewer resigns on behalf of one side, which must be a side they play."""

    type: Literal["resign"]
    color: Color


class Abort(ViewerAction):
    """A viewer ends the game with no result."""

    type: Literal["abort"]


class TakeBack(ViewerAction):
    """A viewer takes back the last move, or the last pair of moves."""

    type: Literal["takeback"]


ViewerMessage = Annotated[SubmitMove | Resign | Abort | TakeBack, Field(discriminator="type")]
"""What a viewer may send over the game WebSocket."""

_VIEWER_MESSAGE = TypeAdapter(ViewerMessage)


class ErrorEvent(BaseModel):
    """The server would not act on what a viewer sent, and nothing has changed."""

    type: Literal["error"] = "error"
    message: str


ViewerEvent = ChannelEvent | ErrorEvent
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

    game_channel = GameChannel(on_finished=save_finished_game)
    # One game starts at a time. Making a model player reads a checkpoint off the disk, which
    # the handler suspends for, and two starts that interleave there would finish in whichever
    # order their players happened to be built: the game somebody asked for first could come
    # back and replace the game that replaced it, and the viewer who started that second game
    # would have been told it was theirs. Held for the whole of starting, so that the games
    # start in the order they were asked for — and so that two checkpoints are never loaded
    # at once on a machine that may also be training.
    starting = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await game_channel.close()

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
        "/game",
        responses={
            400: {
                "description": "The position cannot be played from, or a player cannot be "
                "made: no such run, no such checkpoint, or a checkpoint that cannot be loaded"
            },
            500: {
                "description": "A model was asked for and the inference device this server "
                "is configured with is not there on this machine, or Stockfish was asked for "
                "and the configured binary is missing or is not Stockfish"
            },
            503: {"description": "The server is shutting down and is starting no more games"},
        },
    )
    async def new_game(request: NewGameRequest) -> GameState:
        """Start a new game, replacing the current one for every viewer.

        A game starts from the standard starting position unless a FEN says otherwise.
        Anything that would stop the game being started is refused, and the current game
        plays on: a FEN that cannot be played from, or a checkpoint that is not there. Not
        every refusal is the asker's doing — a configured inference device that this machine
        does not have, or a Stockfish that is not installed where the config says, is answered
        as the server's own fault, and a server on its way down starts nothing at all.

        Every Stockfish side is an engine process of its own, which is stopped when its game
        ends, is replaced, or the server goes down, and at once if the game is refused.
        """
        # The position is checked before the players are made, so that a mistyped FEN is
        # answered at once rather than after tens of megabytes of checkpoint have been read.
        if request.fen is not None:
            try:
                board_from_fen(request.fen)
            except InvalidFenError as invalid:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(invalid)) from invalid
        async with starting:
            white, black = await _players(request, config)
            try:
                session = GameSession(white, black, move_delay=request.move_delay, fen=request.fen)
                game_channel.start(session)
            except GameChannelClosedError as closed:
                close_players(white, black)
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE, "the server is shutting down"
                ) from closed
            except BaseException:
                close_players(white, black)
                raise
        return session.state

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

    def _open_run(name: str) -> RunReader:
        try:
            return open_run(config.paths.runs, name)
        except RunError as missing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(missing)) from missing

    @api.get(
        "/game/pgn",
        response_class=Response,
        responses={
            200: {"content": {PGN_MEDIA_TYPE: {}}, "description": "The game as a PGN file"},
            404: {"description": "No game has been started yet"},
            409: {"description": "The game asked for has been replaced by another"},
        },
    )
    async def game_pgn_file(
        game: Annotated[
            str | None,
            Query(
                description="The id of the game to download, as its state gives it. A game "
                "that has been replaced since is refused rather than quietly swapped for "
                "its replacement. Left out, whatever game is on show is served."
            ),
        ] = None,
    ) -> Response:
        """Download a game as PGN, whether it has finished or not.

        A game in progress is described as far as it has been played, with the result "*"
        that PGN gives a game that has not ended.
        """
        # Read here on the event loop, which a plain `def` would not be: FastAPI runs
        # those in a worker thread, and a game read there while a move is being made can
        # come out claiming a checkmate that is not among its moves.
        current = game_channel.current_game
        if current is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "no game has been started", headers=NO_STORE
            )
        if game is not None and game != current.id:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "that game has been replaced", headers=NO_STORE
            )
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

    @api.websocket("/game/ws")
    async def follow_game(websocket: WebSocket) -> None:
        """Send the current game's full state and every event after it, and act on replies."""
        await websocket.accept()
        connection = _Connection(websocket)
        # Either side can end the connection: the viewer goes away, or the channel
        # closes and has nothing left to send.
        async with asyncio.TaskGroup() as tasks:
            sending = tasks.create_task(_send_game_events(connection, game_channel))
            receiving = tasks.create_task(_act_on_viewer_messages(connection, game_channel))
            await asyncio.wait({sending, receiving}, return_when=asyncio.FIRST_COMPLETED)
            sending.cancel()
            receiving.cancel()

    @api.websocket("/runs/{name}/ws")
    async def follow_run(websocket: WebSocket, name: str) -> None:
        """Send what a run is, its whole metrics log, and then whatever it adds to either.

        The first two messages are a ``run`` event and a ``metrics`` event with ``reset`` set.
        After that a ``run`` event comes whenever the heartbeat changes or goes stale, and a
        ``metrics`` event whenever the log has grown. A run that is not here is answered with
        an ``error`` event, and the connection is closed.
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
    for websocket_path in ("/game/ws", "/runs/{name}/ws"):
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


async def _players(request: NewGameRequest, config: Config) -> tuple[Player, Player]:
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
    # One engine per checkpoint rather than per player: a checkpoint playing itself at two
    # ratings, or its best move against its own sampling, is one set of weights and two ways
    # of choosing from them. Everything a player was asked for — the rating, the strategy,
    # its generator — lives in the player, so the engine has nothing of either side in it.
    engines: dict[tuple[str, int], Any] = {}
    white = await _player(request.white, config, engines)
    try:
        black = await _player(request.black, config, engines)
    except BaseException:
        close_players(white)
        raise
    return white, black


async def _player(spec: AnyPlayer, config: Config, engines: dict[tuple[str, int], Any]) -> Player:
    match spec:
        case HumanSpec():
            return HumanPlayer()
        case RandomSpec():
            return RandomPlayer()
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
                return await start_stockfish(
                    config.stockfish.path, elo=spec.elo, move_time=spec.move_time
                )
            except StockfishError as unavailable:
                # Nobody who clicked Start installed the engine or wrote the config.
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR, str(unavailable)
                ) from unavailable
        case _:
            # The union is closed, so this is unreachable until somebody adds a kind of
            # player and not the case that makes it. Said here, where the omission is, rather
            # than left to fail on the first move of a game with nobody playing it.
            assert_never(spec)


def _model_player(spec: ModelSpec, config: Config, engines: dict[tuple[str, int], Any]) -> Player:
    """A checkpoint, loaded and sat down at the board.

    The inference package is imported here rather than at the top of this module, because it
    is what brings in torch: a server that is replaying games and playing people against the
    random mover has no use for it, and starting up is a second or two quicker without it.

    ``engines`` is what the two sides of one game share, keyed by the checkpoint they came
    from; see :func:`_players`.
    """
    from chess_ai.inference import DeviceUnavailableError, InferenceError, ModelPlayer, load_engine

    run = open_run(config.paths.runs, spec.run)
    chosen = choose_checkpoint(run, spec.checkpoint)
    engine = engines.get((run.name, chosen.step))
    if engine is None:
        try:
            engine = load_engine(
                run.checkpoint_path(chosen),
                device=config.inference.device,
                batch_size=config.inference.batch_size,
            )
        except DeviceUnavailableError as misconfigured:
            # Nobody who clicked Start chose the device; the machine this server runs on did.
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(misconfigured)
            ) from misconfigured
        except InferenceError as unusable:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(unusable)) from unusable
        engines[(run.name, chosen.step)] = engine
    return ModelPlayer(
        engine,
        run=run.name,
        checkpoint=chosen.step,
        rating=spec.rating,
        strategy=spec.strategy,
        temperature=spec.temperature,
        seed=spec.seed,
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


async def _act_on_viewer_messages(connection: _Connection, game_channel: GameChannel) -> None:
    """Do what the viewer asks of the game, until the viewer goes away.

    Anything the game will not do is reported to the viewer who asked and to nobody
    else, and the game goes on.
    """
    try:
        while (message := await connection.receive())["type"] != "websocket.disconnect":
            try:
                asked = _VIEWER_MESSAGE.validate_json(message.get("text") or "")
            except ValidationError:
                await connection.send(ErrorEvent(message="the server cannot read that message"))
                continue
            try:
                _act(game_channel, asked)
            except (ActionRejectedError, MoveRejectedError) as rejected:
                await connection.send(ErrorEvent(message=str(rejected)))
    except WebSocketDisconnect:
        pass  # The viewer has already gone.


def _act(game_channel: GameChannel, asked: ViewerMessage) -> None:
    match asked:
        case SubmitMove():
            game_channel.submit_move(asked.game, asked.uci)
        case Resign():
            game_channel.resign(asked.game, asked.color)
        case Abort():
            game_channel.abort(asked.game)
        case TakeBack():
            game_channel.take_back(asked.game)


async def _send_game_events(connection: _Connection, game_channel: GameChannel) -> None:
    """Send the game channel's events until it closes, then let the viewer go."""
    try:
        async with aclosing(game_channel.events()) as events:
            async for event in events:
                await connection.send(event)
        await connection.close(status.WS_1001_GOING_AWAY)
    except WebSocketDisconnect:
        pass  # The viewer has already gone.


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
