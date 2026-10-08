"""The games the server holds: several at once, each reached only through its own links.

Starting a game replaces nothing. Every game has a watch link, which follows the game and can
do nothing to it, and a play link for each side a person plays, which can do what that side
may: move, take back, resign and abort. A game nobody plays by hand has a control link instead,
which can only abort it. A link is a random id nobody can guess, and that is all the protection
a game has: whoever holds a link can do what it allows.

A game that has ended stays to be looked at through its links; it no longer counts towards the
number of games in progress. An aborted game is gone at once, links and all, since an abort is
a game nobody wants kept. Any other game is gone once it has gone a while without a move,
finished or not.

Given a :class:`~chess_ai.web.game_store.GameStore`, the registry writes every game to it as the
game changes, and deletes it from there when it is gone, so that the server can carry on with
its games after a restart.
"""

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from chess_ai.game_session import ActionRejectedError, GameEvent, GameSession, GameState
from chess_ai.players import Player, close_players, wait_players_closed
from chess_ai.position_view import Color
from chess_ai.web.game_store import GameLinks, GameStore, PlayerRecord, StoredGame

logger = logging.getLogger(__name__)


CLOSE_TIMEOUT = 5.0
"""How long a game that has stopped waits for its players' engines to exit."""

LINK_BYTES = 16
"""How much randomness a link has: 128 bits, which nobody guesses or tries their way to."""

Access = Literal["white", "black", "control", "watch"]
"""What a link may do: play one side, abort a game nobody plays by hand, or only watch."""


class GamesClosedError(RuntimeError):
    """The server is going down, so it starts no further games."""


class TooManyGamesError(Exception):
    """As many games are in progress as the server holds, so a new one is refused."""


class NoSuchGameError(Exception):
    """No game has this link: it never had one, or the game was aborted."""


__all__ = [
    "Access",
    "GameLinks",
    "GameRegistry",
    "GamesClosedError",
    "NoSuchGameError",
    "Seat",
    "TooManyGamesError",
]


class _Hosted:
    """One game the server holds, and the task playing it, if it is being played."""

    def __init__(
        self,
        session: GameSession,
        links: GameLinks,
        settings: BaseModel | None,
        players: tuple[PlayerRecord, PlayerRecord] | None,
    ) -> None:
        self.session = session
        self.links = links
        self.task: asyncio.Task[None] | None = None
        self.settings = settings
        self.players = players
        """What it takes to make White and Black again, for a game that is kept on disk."""
        self.next: str | None = None
        """The id of the game started from this one once it ended, if one has been."""


class Seat:
    """One game as one link sees it: what it shows, and what that link may do to it."""

    def __init__(self, registry: "GameRegistry", hosted: _Hosted, access: Access) -> None:
        self._registry = registry
        self._hosted = hosted
        self.access: Access = access

    @property
    def state(self) -> GameState:
        return self._hosted.session.state

    @property
    def watch(self) -> str:
        """The game's watch link, which anybody holding any of its links may pass on."""
        return self._hosted.links.watch

    @property
    def own_links(self) -> GameLinks:
        """The links this seat may hand out: its own, and the game's watch link."""
        links = self._hosted.links
        own = getattr(links, self.access) if self.access != "watch" else None
        return GameLinks(watch=links.watch, **({self.access: own} if own is not None else {}))

    def next_game(self) -> "Seat | None":
        """This seat's place in the game started from this one, while that is being played.

        Two people who both ask to play again want one game between them, not one each.
        """
        following = self._registry._held(self._hosted.next)
        if following is None:
            return None
        if not _in_progress(following.session):
            return None
        link = getattr(following.links, self.access, None)
        return None if link is None else self._registry.seat(link)

    @property
    def settings(self) -> BaseModel | None:
        """What the game was started with, as whoever started it gave it."""
        return self._hosted.settings

    @property
    def updated(self) -> datetime:
        """When the game last changed."""
        return self._hosted.session.updated

    @contextmanager
    def subscribe(self) -> Iterator[AsyncIterator[GameEvent]]:
        """The game's full state, then everything that happens in it until it stops."""
        with self._hosted.session.subscribe() as events:
            yield events

    def submit_move(self, uci: str) -> None:
        """Play ``uci`` for this link's side.

        Raises:
            ActionRejectedError: the link plays no side, or it is the other side's move.
            MoveRejectedError: the game has ended, or the move is not legal.
        """
        side = self._side("move")
        session = self._hosted.session
        if session.position.game_over is None and session.position.turn != side:
            raise ActionRejectedError("it is not your move")
        session.submit_move(uci)

    def resign(self) -> None:
        """Resign the game for this link's side.

        Raises:
            ActionRejectedError: the link plays no side, or the game has ended.
        """
        self._hosted.session.resign(self._side("resign"))
        self._registry._after_action(self._hosted)

    def take_back(self) -> None:
        """Take back this side's last move, or ask the other person to agree to it.

        Raises:
            ActionRejectedError: the link plays no side, or the game will not take it back.
        """
        self._hosted.session.ask_to_take_back(self._side("take back"))

    def abort(self) -> None:
        """Abort the game, or ask the other person to agree to it.

        Raises:
            ActionRejectedError: the link only watches, or the game will not be aborted.
        """
        if self.access == "control":
            self._hosted.session.ask_to_abort(None)
        else:
            self._hosted.session.ask_to_abort(self._side("abort"))
        self._registry._after_action(self._hosted)

    def answer(self, request_id: int, *, accept: bool) -> None:
        """Agree to, or decline, request ``request_id`` the other person made of this side.

        Raises:
            ActionRejectedError: the link plays no side, or that request is not waiting for
                an answer.
        """
        self._hosted.session.answer(self._side("answer"), request_id, accept=accept)
        self._registry._after_action(self._hosted)

    def replace(self, player: Player, record: PlayerRecord | None = None) -> None:
        """Hand the side the game is waiting on to ``player``, which ``record`` makes again.

        Any link but a watch link may: in a game against a model the person playing it is the
        one waiting, and in a game nobody plays by hand it is whoever holds the control link.

        Raises:
            ActionRejectedError: the link only watches, the game has ended, or it is not
                waiting for another player.
        """
        if self.access == "watch":
            raise ActionRejectedError("you are watching this game and cannot choose its players")
        self._registry._replace(self._hosted, player, record)

    def _side(self, doing: str) -> Color:
        if self.access == "watch":
            raise ActionRejectedError(f"you are watching this game and cannot {doing} in it")
        if self.access == "control":
            raise ActionRejectedError(f"nobody plays this game by hand, so nobody can {doing}")
        return self.access


class GameRegistry:
    def __init__(
        self,
        on_finished: Callable[[GameState], None] | None = None,
        *,
        max_ongoing: int = 20,
        store: GameStore | None = None,
        expire_after: timedelta = timedelta(days=7),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """The games the server holds, keeping every game that reaches a result.

        ``on_finished`` is handed each game that reached a result, as it reaches it, which is
        where the server saves it as PGN. An aborted game reached no result of its own and is
        not kept. Whatever keeping a game raises is logged and the games play on.

        ``store`` is where every game started with the records of its players is written as it
        changes, and deleted from when it is gone. ``expire_after`` is how long a game is kept
        after it last changed, by ``clock``'s time, before :meth:`expire` deletes it.
        """
        self._on_finished = on_finished
        self._max_ongoing = max_ongoing
        self._store = store
        self._expire_after = expire_after
        self._clock = clock
        self._games: dict[str, tuple[_Hosted, Access]] = {}
        self._hosted: dict[str, _Hosted] = {}
        # Every game still on its way out, so that the server going down waits for each of
        # them, or their engine processes would outlive the server.
        self._playing: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def ongoing(self) -> int:
        """How many games are being played, which is what the limit counts."""
        return sum(1 for hosted in self._hosted.values() if _in_progress(hosted.session))

    def check_room(self) -> None:
        """Make sure another game may start.

        Raises:
            TooManyGamesError: as many games are in progress as the server holds.
        """
        if self.ongoing >= self._max_ongoing:
            raise TooManyGamesError(
                f"{self._max_ongoing} games are already being played here, which is as many "
                "as this server holds at once; try again once one of them has ended"
            )

    def start(
        self,
        session: GameSession,
        *,
        settings: BaseModel | None = None,
        follows: Seat | None = None,
        players: tuple[PlayerRecord, PlayerRecord] | None = None,
    ) -> GameLinks:
        """Start playing ``session``, and return the links that reach it.

        ``settings`` is what the game was asked for, kept with it so that another can be
        started the same way; the registry itself never reads it. ``follows`` is the game
        this one is played again from, which :meth:`Seat.next_game` then leads to it.
        ``players`` is what it takes to make White and Black again, without which the game is
        not written to the store, and so ends with the server.

        Raises:
            GamesClosedError: the server is going down.
            TooManyGamesError: as many games are in progress as the server holds.
        """
        if self._closed:
            raise GamesClosedError("the server is shutting down")
        self.check_room()
        sides = session.state
        links = GameLinks(
            watch=_new_link(),
            white=_new_link() if sides.white.accepts_moves else None,
            black=_new_link() if sides.black.accepts_moves else None,
        )
        if links.white is None and links.black is None:
            links.control = _new_link()
        hosted = _Hosted(session, links, settings, players)
        self._host(hosted)
        self._save(hosted)
        if follows is not None and self._held(follows._hosted.session.id) is follows._hosted:
            follows._hosted.next = session.id
            self._save(follows._hosted)
        return links

    def restore(
        self,
        stored: StoredGame,
        white: Player,
        black: Player,
        settings: BaseModel | None = None,
    ) -> None:
        """Take back a game kept in the store, as the server starts, with these players in it.

        A game still in progress is played on from where it was, and a game that had ended is
        there to be looked at, as it was before. Neither is counted against the limit on games:
        they were let in already, before the restart.

        Raises:
            GamesClosedError: the server is going down.
            ValueError: the game's moves cannot be played over again.
        """
        if self._closed:
            raise GamesClosedError("the server is shutting down")
        session = GameSession.restore(white, black, stored.session)
        hosted = _Hosted(session, stored.links, settings, (stored.white, stored.black))
        hosted.next = stored.next
        self._host(hosted)

    def expire(self) -> int:
        """Delete every game that has gone ``expire_after`` without changing, and say how many.

        A game in progress is stopped where it is, as if aborted, and its viewers let go of.
        """
        now = self._clock()
        expired = [
            hosted
            for hosted in self._hosted.values()
            if now - hosted.session.updated >= self._expire_after
        ]
        for hosted in expired:
            _stop(hosted)
            self._forget(hosted)
        if expired:
            logger.info(
                "Deleted %d games nobody had moved in for %s", len(expired), self._expire_after
            )
        return len(expired)

    def _host(self, hosted: _Hosted) -> None:
        """Hold ``hosted``, reached through its links, and play it if it is in progress."""
        session = hosted.session
        if _in_progress(session):
            task = asyncio.create_task(self._play(session))
            hosted.task = task
            task.add_done_callback(_log_failure)
            # A callback as well as a `finally` in the game, because a game stopped before it
            # got its first turn is cancelled without ever running, `finally` and all.
            task.add_done_callback(lambda _: _close_players(session))
            self._playing.add(task)
            task.add_done_callback(self._playing.discard)
        session.listen(lambda event: self._changed(hosted, event))
        self._hosted[session.id] = hosted
        for access in ("white", "black", "control", "watch"):
            link = getattr(hosted.links, access)
            if link is not None:
                self._games[link] = (hosted, access)

    def seat(self, link: str) -> Seat:
        """The game ``link`` reaches, with what the link may do in it.

        Raises:
            NoSuchGameError: no game has that link.
        """
        found = self._games.get(link)
        if found is None:
            raise NoSuchGameError("there is no such game: the link is wrong, or it was aborted")
        hosted, access = found
        return Seat(self, hosted, access)

    async def close(self) -> None:
        """Stop every game and end every viewer's events, as when the server shuts down."""
        self._closed = True
        for hosted in self._hosted.values():
            _stop(hosted)
        await asyncio.gather(*self._playing, return_exceptions=True)

    def _after_action(self, hosted: _Hosted) -> None:
        """Stop a game a viewer has just ended, and forget it if it was aborted."""
        over = hosted.session.position.game_over
        if over is None:
            return
        _stop(hosted)
        if over.result == "*":
            self._forget(hosted)

    def _held(self, game_id: str | None) -> _Hosted | None:
        return None if game_id is None else self._hosted.get(game_id)

    def _forget(self, hosted: _Hosted) -> None:
        """Let go of ``hosted`` and delete what is kept of it; doing it twice does nothing."""
        game_id = hosted.session.id
        if self._hosted.get(game_id) is not hosted:
            return
        del self._hosted[game_id]
        for link in [link for link, (held, _) in self._games.items() if held is hosted]:
            del self._games[link]
        if self._store is not None:
            try:
                self._store.remove(game_id)
            except OSError:
                logger.exception("Could not delete game %s from %s", game_id, self._store.directory)

    def _replace(self, hosted: _Hosted, player: Player, record: PlayerRecord | None) -> None:
        session = hosted.session
        paused = session.paused
        if session.position.game_over is not None or paused is None:
            raise ActionRejectedError("no player is waiting to be replaced")
        before = hosted.players
        # In place before the session says it has changed, which is when the game is written.
        if before is not None and record is not None:
            white, black = before
            hosted.players = (record, black) if paused.side == "white" else (white, record)
        try:
            old = session.replace(player)
        except BaseException:
            hosted.players = before
            raise
        _close_players_of(old)

    def _changed(self, hosted: _Hosted, event: GameEvent) -> None:
        """Keep a game that has just changed: as PGN once it has a result, and in the store."""
        if self._hosted.get(hosted.session.id) is not hosted:
            return
        if event.type == "considering":
            # What a player is considering is not kept for a restart, which asks it again, so
            # nothing the store holds has changed.
            return
        over = hosted.session.position.game_over
        if event.type in ("move", "game_over") and over is not None:
            if over.reason == "abort":
                # Nothing is kept of an aborted game, here or anywhere.
                self._forget(hosted)
                return
            # Before the store, so that a server that dies in between has the PGN and plays
            # the last move again, rather than having lost it.
            self._keep(hosted.session)
        self._save(hosted)

    def _save(self, hosted: _Hosted) -> None:
        if self._store is None or hosted.players is None:
            return
        white, black = hosted.players
        settings = hosted.settings
        record = hosted.session.record
        failed = record.game_over is not None and record.game_over.reason == "error"
        if failed and hosted.next is None:
            # A player failing is not how the game ended. An engine killed along with a server
            # going down for an update fails just the same, and a game that took days must not
            # be lost to that: it is kept as it stood, and played on after the restart with
            # players made afresh, where a player that is really broken fails again. Unless it
            # has been played again: whoever did has moved on, and it stays ended.
            record = record.model_copy(update={"game_over": None})
        stored = StoredGame(
            links=hosted.links,
            white=white,
            black=black,
            session=record,
            settings=settings.model_dump(mode="json") if settings is not None else None,
            next=hosted.next,
        )
        try:
            self._store.write(stored)
        except (OSError, ValueError):
            # The game goes on regardless; it is only a restart that it would not survive.
            logger.exception("Could not write game %s to %s", stored.id, self._store.directory)

    async def _play(self, session: GameSession) -> None:
        """Play ``session`` to wherever it stops, and keep the game if it was finished."""
        try:
            await session.play()
        finally:
            _close_players(session)
            # Waited for here, so that a server going down, which waits for its games, also
            # waits for their engines to have exited. Not for ever, though: a player that
            # could not be closed must not keep the server from going down.
            try:
                await asyncio.wait_for(wait_players_closed(*session.players), CLOSE_TIMEOUT)
            except TimeoutError:
                logger.warning("The players of a game had not let go after %gs", CLOSE_TIMEOUT)

    def _keep(self, session: GameSession) -> None:
        """Hand a game that reached a result to whoever keeps the games."""
        game = session.state
        over = game.position.game_over
        # "*" is a game that reached no result at all, which an abort leaves behind.
        if self._on_finished is None or over is None or over.result == "*":
            return
        try:
            self._on_finished(game)
        except Exception:
            # A game that cannot be kept is no reason to stop: the viewers are still
            # looking at it, and the other games have to be playable regardless.
            logger.exception("Could not keep the finished game")


def _new_link() -> str:
    return secrets.token_urlsafe(LINK_BYTES)


def _in_progress(session: GameSession) -> bool:
    return session.position.game_over is None


def _stop(hosted: _Hosted) -> None:
    """Close a game and let go of the task driving it, if one is.

    A game the players did not finish leaves ``play()`` waiting for a move that will never
    come, so closing the session is not enough to stop it.
    """
    hosted.session.close()
    if hosted.task is not None:
        hosted.task.cancel()


def _close_players(session: GameSession) -> None:
    """Let go of whatever the players of a game that has stopped hold, such as an engine."""
    _close_players_of(*session.players)


def _close_players_of(*players: Player) -> None:
    try:
        close_players(*players)
    except Exception:
        logger.exception("Could not close the players of a game")


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("A game stopped with an error", exc_info=error)
