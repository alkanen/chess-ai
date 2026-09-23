"""The web server: the built frontend and the API, all under the configured path prefix.

Every route lives under the prefix so that a reverse proxy can forward requests
unchanged. The frontend build uses relative URLs only; the server tells the browser
the prefix at runtime by adding a ``<base href="{prefix}/">`` tag to ``index.html``.
"""

import asyncio
import html
import re
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Literal

import chess
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from starlette.types import Message
from starlette.websockets import WebSocketState

from chess_ai.config import Config
from chess_ai.game_session import GameSession, GameState
from chess_ai.players import HumanPlayer, MoveRejectedError, Player, RandomPlayer
from chess_ai.position_view import PositionSnapshot, snapshot
from chess_ai.web.game_channel import ChannelEvent, GameChannel, GameChannelClosedError

STATIC_DIR = Path(__file__).parent / "static"
"""Where ``scripts/build-frontend.sh`` puts the built frontend."""

_HEAD_TAG = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)

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


class SubmitMove(BaseModel):
    """A viewer plays a move for the human side to move."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["move"]
    uci: str


ViewerMessage = SubmitMove
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
    game_channel = GameChannel()

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
        """Start a new game, replacing the current one for every viewer."""
        session = GameSession(
            _PLAYERS[request.white](), _PLAYERS[request.black](), move_delay=request.move_delay
        )
        try:
            game_channel.start(session)
        except GameChannelClosedError as closed:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "the server is shutting down"
            ) from closed
        return session.state

    @api.websocket("/game/ws")
    async def follow_game(websocket: WebSocket) -> None:
        """Send the current game's full state, then every move, and play submitted moves."""
        await websocket.accept()
        connection = _Connection(websocket)
        # Either side can end the connection: the viewer goes away, or the channel
        # closes and has nothing left to send.
        async with asyncio.TaskGroup() as tasks:
            sending = tasks.create_task(_send_game_events(connection, game_channel))
            receiving = tasks.create_task(_play_submitted_moves(connection, game_channel))
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


async def _play_submitted_moves(connection: _Connection, game_channel: GameChannel) -> None:
    """Play the moves the viewer submits, until the viewer goes away.

    A move the game will not take is reported to the viewer who submitted it and to
    nobody else, and the game plays on.
    """
    try:
        while (message := await connection.receive())["type"] != "websocket.disconnect":
            try:
                submitted = _VIEWER_MESSAGE.validate_json(message.get("text") or "")
            except ValidationError:
                await connection.send(ErrorEvent(message="the server cannot read that message"))
                continue
            try:
                game_channel.submit_move(submitted.uci)
            except MoveRejectedError as rejected:
                await connection.send(ErrorEvent(message=str(rejected)))
    except WebSocketDisconnect:
        pass  # The viewer has already gone.


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
