"""Game sessions: one game held on the server and driven between two players.

A session asks each player for a move in turn, checks every move through python-chess
and tells any number of subscribers what happens. A subscriber first gets the full
current state and then everything that happens after it, so all viewers see the same
game however late they join. A game can also be ended off the board, by a resignation
or an abort.

Two people playing each other cannot undo each other's moves, or throw away a game they have
both put days into, on their own: between them a takeback, and an abort once the game is under
way, is a request that the other side answers.

A checkpoint that is gone by the time its side has to move, deleted or replaced while the game
waited, does not end the game: the game waits for another player to take its side over, which
it records, and carries on from where it was.

A session can be written down as a :class:`SessionRecord` and made again from one, which is how
a game outlives the server being restarted.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import chess
from pydantic import BaseModel

from chess_ai.players import (
    GameContext,
    ModelBackedPlayer,
    ModelDescription,
    MoveRejectedError,
    Player,
    PlayerMove,
    PlayerUnavailableError,
    StockfishBackedPlayer,
    StockfishDescription,
    SubmittedMovePlayer,
    Thoughts,
)
from chess_ai.position_view import (
    Color,
    GameOver,
    PositionSnapshot,
    board_from_fen,
    snapshot,
)


class IllegalMoveError(Exception):
    """A player chose a move that is not legal in the current position."""


class ActionRejectedError(Exception):
    """The session would not do what a viewer asked, so the game is left as it was.

    The message is shown to the person who asked, so it says what is wrong without
    naming the position: their browser is already looking at it.
    """


class MoveRecord(BaseModel):
    uci: str
    san: str
    thoughts: Thoughts | None = None


class PlayerInfo(BaseModel):
    name: str
    """Shown to viewers, such as "Random mover"."""
    accepts_moves: bool
    """Whether this side's moves are submitted by a viewer rather than played by itself."""
    model: ModelDescription | None = None
    """Which checkpoint is playing this side, for a side a checkpoint is playing.

    The name alone cannot carry it: a viewer comparing two checkpoints wants to see which run,
    which step and which rating each side is, and the PGN of the game has to say the same
    things six months later."""
    stockfish: StockfishDescription | None = None
    """How strong Stockfish plays this side, for a side Stockfish is playing."""


RequestKind = Literal["takeback", "abort"]


class PendingRequest(BaseModel):
    """One of two people at the board asked the other's consent, and has not had an answer."""

    id: int
    """Which request this is, which an answer quotes: an answer meant for a request that has
    since been declined must not land on the one asked after it."""
    kind: RequestKind
    by: Color
    """The side that asked."""


class Paused(BaseModel):
    """The game is waiting for a player to take over a side whose player cannot go on."""

    side: Color
    """The side to move, whose player is gone."""
    reason: str
    """Why it cannot play, in words for the people at the game."""


class Replacement(BaseModel):
    """One side's player was replaced in the middle of the game, and by which."""

    ply: int
    """How many moves had been played when the new player took over: it plays the next move
    of its side. A takeback to before that moves it back with the game."""
    side: Color
    old: PlayerInfo
    """The player that could not go on."""
    new: PlayerInfo
    """The player that took its place."""


class GameState(BaseModel):
    id: str
    """Tells this game apart from the one that replaces it; see ``GameSession.id``."""
    white: PlayerInfo
    """The player with the white pieces."""
    black: PlayerInfo
    """The player with the black pieces."""
    start_fen: str
    """The position the game began in, which says how its moves are numbered."""
    moves: list[MoveRecord]
    """Every move played so far."""
    position: PositionSnapshot
    request: PendingRequest | None = None
    """What one side has asked the other and is waiting to hear about, if anything."""
    paused: Paused | None = None
    """Set while the game waits for another player to take over a side that cannot go on."""
    replacements: list[Replacement] = []
    """Every player replaced so far, in the order it happened."""


class GameStateEvent(BaseModel):
    """The full state of the game: the first event every subscriber receives."""

    type: Literal["state"] = "state"
    game: GameState


class MoveEvent(BaseModel):
    type: Literal["move"] = "move"
    ply: int
    """The move's number in ``GameState.moves``, counting from 1."""
    move: MoveRecord
    position: PositionSnapshot
    """The position after the move."""


class TakebackEvent(BaseModel):
    """Moves were taken back, so the game stands in a position it has already been in."""

    type: Literal["takeback"] = "takeback"
    ply: int
    """How many moves are left in ``GameState.moves``; the rest are no longer played."""
    position: PositionSnapshot
    """The position the game is back in, exactly as it stood after ply ``ply``."""


class RequestEvent(BaseModel):
    """One side asked the other for a takeback or an abort, or the other side said no.

    A request is also over when the game moves on: a move, a takeback or the game ending clears
    it without an event of its own, since a request is about the position it was made in.
    """

    type: Literal["request"] = "request"
    request: PendingRequest | None
    """What is being asked now; ``None`` when the request was declined."""


class GameOverEvent(BaseModel):
    """The game ended without a move being played: a resignation or an abort."""

    type: Literal["game_over"] = "game_over"
    position: PositionSnapshot
    """The position the game stopped in, with its ``game_over`` set."""


class PausedEvent(BaseModel):
    """The player on move cannot go on, so the game waits for another to take its side."""

    type: Literal["paused"] = "paused"
    paused: Paused


class ReplacedEvent(BaseModel):
    """Another player has taken over a side, so the game is no longer paused."""

    type: Literal["replaced"] = "replaced"
    replacement: Replacement


GameEvent = (
    GameStateEvent
    | MoveEvent
    | TakebackEvent
    | RequestEvent
    | GameOverEvent
    | PausedEvent
    | ReplacedEvent
)


class SessionRecord(BaseModel):
    """Everything a session is, but its players: what it is made again from after a restart.

    The players are not in it, because what it takes to make one again (a checkpoint, an
    engine) is the business of whoever made them in the first place.
    """

    id: str
    start_fen: str
    moves: list[MoveRecord]
    move_delay: float
    game_over: GameOver | None = None
    """How the game ended, if it has; a game ended off the board says so nowhere else."""
    request: PendingRequest | None = None
    requests_made: int = 0
    """How many requests have been made, so that a request after a restart has a new id."""
    replacements: list[Replacement] = []
    started: datetime
    updated: datetime


class GameSession:
    def __init__(
        self,
        white: Player,
        black: Player,
        *,
        move_delay: float = 0.0,
        id: str | None = None,
        fen: str | None = None,
        opening: Sequence[chess.Move] = (),
    ) -> None:
        """A game from the starting position, or from ``fen`` when one is given.

        ``move_delay`` is the least number of seconds before a move a player worked out
        for itself, so that viewers can follow a game between players that move
        instantly. A move submitted from outside the game is played as soon as it
        arrives, so two submitted moves can follow each other at once.

        ``id`` names the game to viewers, who quote it back with everything they ask of
        it, so that a click meant for this game cannot land on the one that replaced it.
        It is made up unless given, which tests do to keep their games recognizable.

        ``fen`` must be a position a game can be played from; see
        ``position_view.board_from_fen``, which is what turns what a viewer typed into
        one. The side it gives the move has the first move of the game.

        ``opening`` is moves already played when the players sit down, as a match starts its
        games from a book line. They are the game's first moves like any other, in its history
        and its PGN, so that the game replays from where it really started and a repetition
        counts the positions they passed through.

        Raises:
            ValueError: a move of ``opening`` is not legal where it would be played.
        """
        self.id = id if id is not None else uuid4().hex
        self._players = {chess.WHITE: white, chess.BLACK: black}
        self._move_delay = move_delay
        self._board = chess.Board() if fen is None else board_from_fen(fen)
        self._start_fen = self._board.fen()
        self._moves: list[MoveRecord] = []
        for move in opening:
            if not self._board.is_legal(move):
                raise ValueError(f"{move.uci()} is not a legal move in {self._board.fen()}")
            self._moves.append(MoveRecord(uci=move.uci(), san=self._board.san(move)))
            self._board.push(move)
        self._position = snapshot(self._board)
        self._subscribers: set[asyncio.Queue[GameEvent | None]] = set()
        self._started = False
        self._closed = False
        # A takeback puts the game in a position the player being asked for a move knows
        # nothing about, so every question carries the count it was asked under and the
        # answer to a question that has since gone stale is thrown away.
        self._takebacks = 0
        self._asking: asyncio.Future[PlayerMove] | None = None
        self._request: PendingRequest | None = None
        self._requests_made = 0
        self._paused: Paused | None = None
        # Set while the game waits for another player, for that player to be handed over by.
        self._replaced: asyncio.Future[None] | None = None
        self._replacements: list[Replacement] = []
        self._listeners: list[Callable[[GameEvent], None]] = []
        self.started = datetime.now(UTC)
        """When the game began."""
        self.updated = self.started
        """When the game last changed: when it began, its last move or takeback, when a player
        was replaced, or when it ended other than by a player failing."""

    @classmethod
    def restore(cls, white: Player, black: Player, record: SessionRecord) -> "GameSession":
        """The session ``record`` was written from, with ``white`` and ``black`` in it.

        A game that had ended stays ended: it is closed already, and is not to be played.
        A request that was waiting for an answer is still waiting. A game that was paused is
        not: whether its player can play is found out again when it is next asked to.

        Raises:
            ValueError: a move of the record is not legal where it was played.
        """
        session = cls(
            white,
            black,
            move_delay=record.move_delay,
            id=record.id,
            fen=record.start_fen,
        )
        for played in record.moves:
            move = chess.Move.from_uci(played.uci)
            if not session._board.is_legal(move):
                raise ValueError(f"{played.uci} is not a legal move in {session._board.fen()}")
            session._board.push(move)
            session._moves.append(played)
        session._position = snapshot(session._board)
        session._request = record.request
        session._requests_made = record.requests_made
        session._replacements = list(record.replacements)
        session.started = record.started
        session.updated = record.updated
        if record.game_over is not None:
            # Ended off the board, which the position alone does not say.
            if session._position.game_over is None:
                session._position = session._position.model_copy(
                    update={"game_over": record.game_over, "legal_moves": {}}
                )
            session._started = True
            session._closed = True
        return session

    @property
    def record(self) -> SessionRecord:
        """What the session would be made again from, as it stands now."""
        return SessionRecord(
            id=self.id,
            start_fen=self._start_fen,
            moves=list(self._moves),
            move_delay=self._move_delay,
            game_over=self._position.game_over,
            request=self._request,
            requests_made=self._requests_made,
            replacements=list(self._replacements),
            started=self.started,
            updated=self.updated,
        )

    def listen(self, listener: Callable[[GameEvent], None]) -> None:
        """Have ``listener`` called with every event as it happens, before any subscriber
        hears of it, for as long as the session lasts.

        Called in the middle of whatever changed the game, so it must not change the game
        itself, and whatever it raises is the caller's: it is for keeping the game, which
        catches its own failures.
        """
        self._listeners.append(listener)

    @property
    def players(self) -> tuple[Player, Player]:
        """White and Black, for whoever made them to close once the game is over."""
        return self._players[chess.WHITE], self._players[chess.BLACK]

    @property
    def position(self) -> PositionSnapshot:
        """Where the game stands now, without the rest of its state."""
        return self._position

    @property
    def state(self) -> GameState:
        return GameState(
            id=self.id,
            white=_describe(self._players[chess.WHITE]),
            black=_describe(self._players[chess.BLACK]),
            start_fen=self._start_fen,
            moves=list(self._moves),
            position=self._position,
            request=self._request,
            paused=self._paused,
            replacements=list(self._replacements),
        )

    @property
    def paused(self) -> Paused | None:
        """What the game is waiting for, while it waits for another player."""
        return self._paused

    def submit_move(self, uci: str) -> None:
        """Play ``uci`` for the side to move, on behalf of a viewer.

        Raises:
            MoveRejectedError: the game has ended, the player to move plays its own
                moves, or the move is not legal. The game is unchanged either way.
        """
        if self._position.game_over is not None:
            raise MoveRejectedError("the game is over")
        player = self._players[self._board.turn]
        if not isinstance(player, SubmittedMovePlayer):
            raise MoveRejectedError(f"{player.name} plays this move")
        player.submit(uci)

    def resign(self, color: Color) -> None:
        """End the game as a resignation by ``color``, on behalf of a viewer.

        Raises:
            ActionRejectedError: the game has ended, or that side plays its own moves
                and so is nobody's to resign. The game is unchanged either way.
        """
        self._check_still_playing()
        player = self._players[_side(color)]
        if not isinstance(player, SubmittedMovePlayer):
            raise ActionRejectedError(f"{player.name} is not yours to resign")
        # The result names the winner, which is the side that did not resign.
        won_by = "1-0" if color == "black" else "0-1"
        self._end(GameOver(result=won_by, reason="resignation"))

    def abort(self) -> None:
        """End the game with no result at all, on behalf of a viewer.

        Raises:
            ActionRejectedError: the game has already ended, and is unchanged.
        """
        self._check_still_playing()
        self._end(GameOver(result="*", reason="abort"))

    def take_back(self, by: Color | None = None) -> None:
        """Undo the last move, or the last pair of moves, on behalf of a viewer.

        A game with one person in it comes back to a move of theirs: undoing only the
        opponent's reply would hand the move straight back to the opponent, which takes
        nothing back at all. Two people at one board take back the single move last
        played, whichever of them played it, unless ``by`` says whose move it is to be:
        then it is that side's last move, and the reply to it if there has been one.

        The position is restored exactly as it stood, castling rights, en passant
        square, move clocks and all, and the moves undone are gone from the game's
        history, so the repetitions it can be drawn by are those of the moves that are
        left.

        Only a game still being played can be taken back, checkmate included, so being
        mated is final here. That is a limit of the session's life rather than a rule
        of the game: an ended game is one ``play()`` has returned from and ``close()``
        has finished, dropping every subscriber, and giving a mated player their move
        back means a session that can be started a second time.

        Raises:
            ActionRejectedError: the game has ended, no side is played by hand, ``by`` is a
                side nobody plays by hand, or no move has been played yet. The game is
                unchanged either way.
        """
        self._check_still_playing()
        self._check_not_paused("take back")
        if not self._submitted_sides():
            raise ActionRejectedError("neither side is played by hand")
        if by is not None:
            self._check_played_by_hand(by, "take back")
        if not self._moves:
            raise ActionRejectedError("no move has been played yet")
        for _ in range(self._plies_to_take_back(by)):
            self._board.pop()
            self._moves.pop()
        self._position = snapshot(self._board)
        self._request = None
        # A player that took over is the one playing from the position the game is back in.
        played = len(self._moves)
        self._replacements = [
            replaced.model_copy(update={"ply": min(replaced.ply, played)})
            for replaced in self._replacements
        ]
        self.updated = datetime.now(UTC)
        # The player being asked for a move was asked about a position that is gone. A
        # person may never answer it at all, being busy with the position now on the
        # board, so the question is dropped rather than waited on.
        self._takebacks += 1
        _drop_question(self._asking)
        self._broadcast(TakebackEvent(ply=len(self._moves), position=self._position))

    def ask_to_take_back(self, by: Color) -> None:
        """Take back ``by``'s last move, or ask the other side to agree to it.

        Against a player that moves for itself the takeback is made at once, as
        :meth:`take_back` makes it. Between two people it is a request, which the other side
        answers with :meth:`answer`.

        Raises:
            ActionRejectedError: the game has ended, ``by`` is not played by hand, ``by`` has
                no move to take back, or a request is already waiting for an answer.
        """
        if not self._needs_consent("takeback"):
            self.take_back(by)
            return
        self._check_still_playing()
        self._check_played_by_hand(by, "take back")
        # Whose the moves are follows from who is on move now: the last move is the other
        # side's, and the one before it is the side on move's.
        on_move = self._board.turn == _side(by)
        if len(self._moves) < (2 if on_move else 1):
            raise ActionRejectedError("you have no move to take back")
        self._request_consent("takeback", by)

    def ask_to_abort(self, by: Color | None) -> None:
        """Abort the game, or ask the other side to agree to it.

        ``by`` is the side asking, or ``None`` for the one who holds a game that nobody plays
        by hand. Only two people who have both made a move need each other's say: before that,
        nothing either of them has put into the game is lost.

        Raises:
            ActionRejectedError: the game has ended, ``by`` is not played by hand, or a request
                is already waiting for an answer.
        """
        if by is not None:
            self._check_played_by_hand(by, "abort")
        if not self._needs_consent("abort"):
            self.abort()
            return
        assert by is not None, "a game two people play is aborted by one of them"
        self._check_still_playing()
        self._request_consent("abort", by)

    def answer(self, by: Color, request_id: int, *, accept: bool) -> None:
        """Agree to, or decline, what the other side asked ``by`` in request ``request_id``.

        Raises:
            ActionRejectedError: the game has ended, or that request is not waiting for
                ``by``'s answer: there is none, ``by`` made it, or it has been answered or
                overtaken by a move since. The game is unchanged either way.
        """
        self._check_still_playing()
        request = self._request
        if request is None or request.by == by or request.id != request_id:
            raise ActionRejectedError("nothing is waiting for your answer")
        if not accept:
            self._request = None
            self._broadcast(RequestEvent(request=None))
        elif request.kind == "takeback":
            self.take_back(request.by)
        else:
            self.abort()

    def _needs_consent(self, kind: RequestKind) -> bool:
        """Whether doing ``kind`` takes both people's say, which is so only between two."""
        if len(self._submitted_sides()) != 2:
            return False
        return kind == "takeback" or len(self._moves) >= 2

    def _request_consent(self, kind: RequestKind, by: Color) -> None:
        if self._request is not None:
            raise ActionRejectedError("a request is already waiting for an answer")
        self._requests_made += 1
        self._request = PendingRequest(id=self._requests_made, kind=kind, by=by)
        self._broadcast(RequestEvent(request=self._request))

    def _check_played_by_hand(self, color: Color, doing: str) -> None:
        player = self._players[_side(color)]
        if not isinstance(player, SubmittedMovePlayer):
            raise ActionRejectedError(f"{player.name} is not yours to {doing}")

    def replace(self, player: Player) -> Player:
        """Hand the side the game is paused for to ``player``, and carry on with it.

        Returns the player it replaces, for whoever made it to let go of.

        Raises:
            ActionRejectedError: the game has ended, or is not waiting for another player.
        """
        self._check_still_playing()
        paused, waiting = self._paused, self._replaced
        # A game closed while it waited, as one expiring is, has stopped waiting.
        if self._closed or paused is None or waiting is None or waiting.done():
            raise ActionRejectedError("no player is waiting to be replaced")
        side = _side(paused.side)
        old = self._players[side]
        self._players[side] = player
        replacement = Replacement(
            ply=len(self._moves), side=paused.side, old=_describe(old), new=_describe(player)
        )
        self._replacements.append(replacement)
        self._paused = None
        waiting.set_result(None)
        self.updated = datetime.now(UTC)
        self._broadcast(ReplacedEvent(replacement=replacement))
        return old

    def _check_not_paused(self, doing: str) -> None:
        if self._paused is not None:
            raise ActionRejectedError(
                f"the game is waiting for another player to take over, so you cannot {doing}"
            )

    def _broadcast(self, event: GameEvent) -> None:
        for listener in self._listeners:
            listener(event)
        for queue in self._subscribers:
            queue.put_nowait(event)

    def _submitted_sides(self) -> list[chess.Color]:
        """The sides whose moves viewers submit, which are the sides people play."""
        return [
            color
            for color, player in self._players.items()
            if isinstance(player, SubmittedMovePlayer)
        ]

    def _plies_to_take_back(self, by: Color | None) -> int:
        """How many moves one takeback undoes, in a game that has at least one to undo."""
        sides = self._submitted_sides()
        if by is not None:
            sides = [_side(by)]
        if len(sides) != 1 or sides[0] != self._board.turn:
            # Two people share the board, or the person has just moved and undoing that
            # one move is all it takes to give it back to them.
            return 1
        # It is the person's move, so the move before theirs is the opponent's reply to
        # a move of their own: both go. Unless the opponent opened the game, that is,
        # and undoing their opening move is as far back as the game goes.
        return min(2, len(self._moves))

    async def play(self) -> None:
        """Play the game to its end, asking each player for a move in turn.

        The session is closed when this returns, fails or is cancelled. A player that fails,
        or plays an illegal move, ends the game with no result, as an "error": nobody can go
        on with it, and every viewer is told so rather than left waiting for a move.

        Raises:
            IllegalMoveError: a player chose an illegal move. The game stays in the
                position before that move.
            Exception: whatever a player raised instead of choosing a move.
        """
        if self._started:
            raise RuntimeError("the game has already been played")
        self._started = True
        loop = asyncio.get_running_loop()
        try:
            earliest = loop.time() + self._move_delay
            while not self._closed and self._position.game_over is None:
                try:
                    choice = await self._ask(self._players[self._board.turn], earliest)
                except PlayerUnavailableError as gone:
                    await self._wait_for_another_player(gone)
                    earliest = loop.time() + self._move_delay
                    continue
                if self._closed:
                    break
                # A takeback while the player was thinking leaves nothing to play: the
                # position it answered is no longer the one on the board.
                if choice is not None:
                    self._make_move(choice)
                earliest = loop.time() + self._move_delay
        except Exception:
            if not self._closed and self._position.game_over is None:
                self._end(GameOver(result="*", reason="error"))
            raise
        finally:
            self.close()

    async def _wait_for_another_player(self, gone: PlayerUnavailableError) -> None:
        """Pause the game until :meth:`replace` hands the side on move to another player, or
        the game is closed."""
        color: Color = "white" if self._board.turn == chess.WHITE else "black"
        self._paused = Paused(side=color, reason=str(gone))
        self._replaced = asyncio.get_running_loop().create_future()
        self._broadcast(PausedEvent(paused=self._paused))
        try:
            await self._replaced
        finally:
            self._replaced = None

    async def _ask(self, player: Player, earliest: float) -> PlayerMove | None:
        """The move ``player`` chooses, or None if a takeback left the question behind.

        The question is a task of its own, so that a takeback can stop it: a player
        asked about a position that has since been taken back may never answer at all.
        The game waits on that task rather than awaiting it, because cancelling the
        question is not cancelling the game, and only the game stopping should stop it.
        """
        asked_after = self._takebacks
        # Started eagerly, so that the player is asked before anything else gets a turn,
        # just as it was when the game awaited the player itself. A person's browser can
        # have a move on its way already, and the player has to be there to take it.
        asking = asyncio.Task(
            self._choose(player, earliest), loop=asyncio.get_running_loop(), eager_start=True
        )
        self._asking = asking
        try:
            await asyncio.wait({asking})
        finally:
            self._asking = None
            # Bites only when the game itself was stopped while the player was thinking.
            _drop_question(asking)
        if asking.cancelled():
            return None
        # A player that failed rather than answered is read before anything else: a
        # takeback does not mend whatever broke, and nothing else would ever look at
        # the exception of a question that has been left behind. A player that is gone is
        # the exception: it is asked again when it is its move again, and fails again then,
        # which pauses the game for its side rather than for whichever side is on move now.
        if (failed := asking.exception()) is not None:
            if isinstance(failed, PlayerUnavailableError) and asked_after != self._takebacks:
                return None
            raise failed
        # A takeback can land after the player has answered but before the answer gets
        # here, and that answer is about a position just as gone as a cancelled one.
        return None if asked_after != self._takebacks else asking.result()

    async def _choose(self, player: Player, earliest: float) -> PlayerMove:
        """``player``'s move, held back until ``earliest`` if it worked the move out itself."""
        choice = await player.choose_move(GameContext(self._board.copy()))
        # The delay paces players that move instantly. Someone who submits a move has
        # already taken as long as they took, so their move is played at once. Sleeping
        # with nothing left to wait still yields to the event loop, so a game between
        # instant players doesn't hold up everything else.
        loop = asyncio.get_running_loop()
        pause = 0.0 if isinstance(player, SubmittedMovePlayer) else earliest - loop.time()
        await asyncio.sleep(max(0.0, pause))
        return choice

    def close(self) -> None:
        """Stop the game where it is and end every subscription.

        A running ``play()`` makes no further moves, but it may still be waiting for a
        player; cancel it to stop it at once.
        """
        if self._closed:
            return
        self._closed = True
        # A game waiting for another player has nothing more to wait for.
        if self._replaced is not None and not self._replaced.done():
            self._replaced.set_result(None)
        for queue in self._subscribers:
            queue.put_nowait(None)

    @contextmanager
    def subscribe(self) -> Iterator[AsyncIterator[GameEvent]]:
        """Follow the game: its full current state first, then every move as it is made.

        A game ended off the board finishes with the position it stopped in. The events
        end when the session is closed.
        """
        queue: asyncio.Queue[GameEvent | None] = asyncio.Queue()
        queue.put_nowait(GameStateEvent(game=self.state))
        if self._closed:
            queue.put_nowait(None)
        self._subscribers.add(queue)
        try:
            yield _events_until_closed(queue)
        finally:
            self._subscribers.discard(queue)

    def _check_still_playing(self) -> None:
        if self._position.game_over is not None:
            raise ActionRejectedError("the game is over")

    def _end(self, game_over: GameOver) -> None:
        """Stop the game in the position it stands in, ended by something off the board."""
        # The position itself is untouched, but a game that is over offers no moves.
        self._position = self._position.model_copy(
            update={"game_over": game_over, "legal_moves": {}}
        )
        self._request = None
        self._paused = None
        # A game that has just ended is kept its full time from now, for its result to be seen.
        # Not one a player failed in, which is kept as still being played, from its last move:
        # a player that fails again on every restart is no reason to keep it for ever.
        if game_over.reason != "error":
            self.updated = datetime.now(UTC)
        self._broadcast(GameOverEvent(position=self._position))
        self.close()

    def _make_move(self, choice: PlayerMove) -> None:
        move = choice.move
        if not self._board.is_legal(move):
            raise IllegalMoveError(f"{move.uci()} is not a legal move in {self._board.fen()}")
        record = MoveRecord(uci=move.uci(), san=self._board.san(move), thoughts=choice.thoughts)
        self._board.push(move)
        self._moves.append(record)
        self._position = snapshot(self._board)
        # A request was about the position before this move, which is no longer the one
        # anybody would be agreeing to.
        self._request = None
        self.updated = datetime.now(UTC)
        self._broadcast(MoveEvent(ply=len(self._moves), move=record, position=self._position))


def _drop_question(asking: asyncio.Future[PlayerMove] | None) -> None:
    """Stop waiting on a question, unless it has already been answered or has failed.

    A question the game has finished with is left exactly as it is, because cancelling
    it is not the nothing it looks like: ``cancel()`` clears a task's "never retrieved"
    flag before it looks at the state, so cancelling a question that has already failed
    silences that failure instead. Left alone, the failure is there for whoever reads
    the question next, and for asyncio to report if nobody ever does.
    """
    if asking is not None and not asking.done():
        asking.cancel()


def _side(color: Color) -> chess.Color:
    return chess.WHITE if color == "white" else chess.BLACK


def _describe(player: Player) -> PlayerInfo:
    return PlayerInfo(
        name=player.name,
        accepts_moves=isinstance(player, SubmittedMovePlayer),
        model=player.model if isinstance(player, ModelBackedPlayer) else None,
        stockfish=player.stockfish if isinstance(player, StockfishBackedPlayer) else None,
    )


async def _events_until_closed(
    queue: asyncio.Queue[GameEvent | None],
) -> AsyncIterator[GameEvent]:
    while (event := await queue.get()) is not None:
        yield event
