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
from pydantic import BaseModel, ConfigDict, Field

from chess_ai.config import Config
from chess_ai.game_session import GameSession, GameState
from chess_ai.players import Player, RandomPlayer
from chess_ai.position_view import PositionSnapshot, snapshot
from chess_ai.web.game_channel import GameChannel, GameChannelClosedError

STATIC_DIR = Path(__file__).parent / "static"
"""Where ``scripts/build-frontend.sh`` puts the built frontend."""

_HEAD_TAG = re.compile(r"<head(?:\s[^>]*)?>", re.IGNORECASE)

PlayerKind = Literal["random"]
_PLAYERS: dict[PlayerKind, Callable[[], Player]] = {"random": RandomPlayer}


class NewGameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    white: PlayerKind
    black: PlayerKind
    move_delay: float = Field(default=0.5, ge=0, le=10)
    """The least number of seconds between two moves."""


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
        """Send the current game's full state, then every move, and every new game."""
        await websocket.accept()
        # Either side can end the connection: the viewer goes away, or the channel
        # closes and has nothing left to send.
        async with asyncio.TaskGroup() as tasks:
            sending = tasks.create_task(_send_game_events(websocket, game_channel))
            waiting = tasks.create_task(_wait_for_disconnect(websocket))
            await asyncio.wait({sending, waiting}, return_when=asyncio.FIRST_COMPLETED)
            sending.cancel()
            waiting.cancel()

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


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass  # Viewers have nothing to say yet.


async def _send_game_events(websocket: WebSocket, game_channel: GameChannel) -> None:
    """Send the game channel's events until it closes, then let the viewer go."""
    try:
        async with aclosing(game_channel.events()) as events:
            async for event in events:
                await websocket.send_text(event.model_dump_json())
        await websocket.close(status.WS_1001_GOING_AWAY)
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
