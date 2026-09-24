"""The web server: the built frontend and the API, all under the configured path prefix.

Every route lives under the prefix so that a reverse proxy can forward requests
unchanged. The frontend build uses relative URLs only; the server tells the browser
the prefix at runtime by adding a ``<base href="{prefix}/">`` tag to ``index.html``.
"""

import asyncio
import html
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

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
from chess_ai.players import HumanPlayer, MoveRejectedError, Player, RandomPlayer
from chess_ai.position_view import Color, InvalidFenError, PositionSnapshot, snapshot
from chess_ai.web.game_channel import ChannelEvent, GameChannel, GameChannelClosedError

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

_WHICH_GAME = (
    "Which game of the file to replay, counting from zero. The headers of every game in "
    "it are returned whichever one this is."
)

PlayerKind = Literal["human", "random"]
_PLAYERS: dict[PlayerKind, Callable[[], Player]] = {
    "human": HumanPlayer,
    "random": RandomPlayer,
}


class NewGameRequest(BaseModel):
    # The field docstrings below describe the request in the API docs, which is the
    # only place someone choosing a move delay has to go on.
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    white: PlayerKind
    black: PlayerKind
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


def create_app(config: Config, static_dir: Path = STATIC_DIR) -> FastAPI:
    prefix = config.server.path_prefix

    def save_finished_game(game: GameState) -> None:
        """Keep a finished game, so that it can be looked at again or trained on."""
        logger.info("Saved the finished game to %s", save_game(game, config.paths.games))

    game_channel = GameChannel(on_finished=save_finished_game)

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

    @api.post("/game")
    async def new_game(request: NewGameRequest) -> GameState:
        """Start a new game, replacing the current one for every viewer.

        A game starts from the standard starting position unless a FEN says otherwise.
        A FEN that cannot be played from is refused, and the current game plays on.
        """
        try:
            session = GameSession(
                _PLAYERS[request.white](),
                _PLAYERS[request.black](),
                move_delay=request.move_delay,
                fen=request.fen,
            )
        except InvalidFenError as invalid:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(invalid)) from invalid
        try:
            game_channel.start(session)
        except GameChannelClosedError as closed:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "the server is shutting down"
            ) from closed
        return session.state

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


class _Connection:
    """One viewer's WebSocket, which the two tasks serving it take turns to write to."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._sending = asyncio.Lock()

    async def receive(self) -> Message:
        return await self._websocket.receive()

    async def send(self, event: ViewerEvent) -> None:
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
