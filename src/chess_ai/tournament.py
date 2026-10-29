"""Round-robin tournaments: every entrant against every other, and how they rate against each other.

An entrant is a checkpoint, at a rating it is asked to play like, or Stockfish at an Elo. Each
pairing plays a match of its own through the match runner, from the lines of one opening set
with the colours swapped every game, and the results of all of them together rate the
entrants against each other by :func:`chess_ai.rating.rate_round_robin`.

A tournament is kept in a directory of its own under the tournaments directory:
``tournament.json`` with what it is, and ``games.jsonl`` with every game it has finished, one
line each, PGN and all. One file for the games rather than a PGN file beside a list of
results, so that the games and the results cannot disagree: a tournament stopped in the middle
of writing a line leaves a torn last line, which is not a game, rather than a game that is in
one file and not the other.

The games are played round by round, the first game of every pairing before the second of
any, so that a tournament stopped half-way has played every pairing about as often, and its
ratings mean what they would at the end, only with wider ranges. A tournament is resumed by
playing it again: only the games it has not finished are played.
"""

import errno
import fcntl
import hashlib
import os
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chess_ai.match import MatchGame, play_game
from chess_ai.openings import OpeningSetError, load_opening_set
from chess_ai.players import Player, close_and_wait
from chess_ai.position_view import GameOverReason
from chess_ai.rating import Pairing, rate_round_robin
from chess_ai.stockfish import Command, StockfishError, start_stockfish

if TYPE_CHECKING:
    # The engine brings torch with it, which reading a tournament has no need of.
    from chess_ai.inference import InferenceEngine
    from chess_ai.training.run_store import CheckpointInfo, RunReader

DEFINITION_FILE: Final = "tournament.json"
GAMES_FILE: Final = "games.jsonl"
LOCK_FILE: Final = "playing.lock"

EVENT_PREFIX: Final = "chess-ai tournament"
"""What the ``Event`` of every game of a tournament starts with, before the tournament's name."""


class TournamentError(Exception):
    """A tournament cannot be made, read, or played."""


class NoSuchTournamentError(TournamentError):
    """There is no tournament of the name asked for."""


class ModelEntrant(BaseModel):
    """A checkpoint, and how it is asked to play."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["model"] = "model"
    run: str
    step: int
    """The checkpoint, by its step: chosen when the tournament was made, so that a run that is
    still training does not swap its best checkpoint for another half-way through."""
    checkpoint_written: datetime | None = None
    """When the checkpoint's file was written, as the tournament was made: what tells it from
    another file of the same step, as a run resumed from an earlier step and trained past this
    one again saves, which is another network. ``None`` only in a tournament made before it
    was recorded."""
    rating: int
    """The rating it is asked to play like, and what its opponents are told it has."""
    strategy: Literal["argmax", "sample"] = "argmax"
    temperature: float = 1.0


class StockfishEntrant(BaseModel):
    """Stockfish held to an Elo, and the time it has for a move."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["stockfish"] = "stockfish"
    elo: int
    move_time: float


type Entrant = Annotated[ModelEntrant | StockfishEntrant, Field(discriminator="kind")]


class Anchor(BaseModel):
    """An entrant whose rating is known from elsewhere, which the others are rated from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entrant: int
    """The entrant, by its place among them."""
    rating: float


class TournamentDefinition(BaseModel):
    """What a tournament is: who plays, how many games each pairing, and from which openings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    entrants: tuple[Entrant, ...] = Field(min_length=2)
    games_per_pairing: int = Field(ge=1)
    openings: str
    """The opening set, as it was named: one that comes with chess-ai, or a file's path."""
    opening_set: str | None = None
    """The set's name and version when the tournament was made, such as ``standard v1``, which
    :func:`create_tournament` fills in: the games of a tournament resumed after the set has
    changed would not start from the lines the rest did."""
    seed: int
    """What every sampling checkpoint's moves are drawn from, game by game."""
    anchor: Anchor | None = None
    """The entrant the ratings are measured from, if any; else they are around an average of
    nothing."""
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TournamentGame(BaseModel):
    """One finished game of a tournament, a line of ``games.jsonl``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round: int
    """Which game of its pairing it was, counting from 1."""
    white: int
    """The entrant that had White, by its place among the entrants."""
    black: int
    result: Literal["1-0", "0-1", "1/2-1/2"]
    termination: GameOverReason
    opening: str
    """The name of the opening line it started from."""
    plies: int
    """How many moves both sides played, the opening line's among them."""
    pgn: str


class Tournament:
    """A tournament on disk: what it is, and the games it has finished."""

    def __init__(self, path: Path, definition: TournamentDefinition) -> None:
        self.path = path
        self.definition = definition
        self._held = False
        self._playing = False

    @contextmanager
    def held(self) -> Iterator[None]:
        """Make this process the one playing the tournament until the block ends, as
        :func:`play_tournament` does for itself when called outside one: so that whatever has
        to be ready before the games, such as the players, is not made by a second process
        only for it to be refused the games. :func:`play_tournament` called inside the block
        plays under this hold, once at a time.

        Raises:
            TournamentError: another process, or another hold here, is playing it.
        """
        if self._held:
            raise self._being_played_here()
        with _PlayingLock(self):
            self._held = True
            try:
                yield
            finally:
                self._held = False

    @contextmanager
    def _playing_games(self) -> Iterator[None]:
        """Play the games under the hold there is, or one taken for them, and only once at a
        time: two calls under one hold would both play every game left."""
        if self._playing:
            raise self._being_played_here()
        with nullcontext() if self._held else self.held():
            self._playing = True
            try:
                yield
            finally:
                self._playing = False

    def _being_played_here(self) -> TournamentError:
        return TournamentError(
            f"tournament {self.definition.name!r} is already being played here, by this process"
        )

    def _being_played(self) -> TournamentError:
        return TournamentError(
            f"tournament {self.definition.name!r} is being played by another process; stop that "
            "one first"
        )

    def games(self) -> list[TournamentGame]:
        """Every game finished so far, in the order they finished.

        A game is a whole line: one with no end to it is still being written, or was being
        written when whatever was writing it was stopped, and is not a game.
        """
        path = self.path / GAMES_FILE
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as e:
            raise TournamentError(f"cannot read {path}: {e.strerror or e}") from e
        whole = data[: data.rfind(b"\n") + 1]
        games = []
        for number, line in enumerate(whole.splitlines(), 1):
            try:
                games.append(TournamentGame.model_validate_json(line))
            except ValidationError as e:
                raise TournamentError(f"line {number} of {path} is not a game: {e}") from e
        return games

    def pgn(self) -> str:
        """Every game finished so far, as one PGN file."""
        return "".join(f"{game.pgn}\n" for game in self.games())

    def playing(self) -> bool:
        """Whether a process is playing the tournament's games now: not, as far as can be told,
        if its lock cannot be looked at."""
        try:
            descriptor = os.open(self.path / LOCK_FILE, os.O_RDONLY)
        except OSError:
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        finally:
            # Closing the file lets go of the lock taken to look, if it was.
            os.close(descriptor)
        return False

    def _drop_torn_line(self) -> None:
        """Cut off a line a stopped tournament was writing, which the next would be joined to.

        Raises:
            TournamentError: the games cannot be read or cut.
        """
        path = self.path / GAMES_FILE
        try:
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                with path.open("r+b") as file:
                    file.truncate(data.rfind(b"\n") + 1)
        except FileNotFoundError:
            return
        except OSError as e:
            raise TournamentError(f"cannot read {path}: {e.strerror or e}") from e

    def _append(self, game: TournamentGame) -> None:
        """Keep ``game``, as one write of a whole line.

        Raises:
            TournamentError: the games cannot be written.
        """
        path = self.path / GAMES_FILE
        try:
            with path.open("a", encoding="utf-8") as file:
                file.write(game.model_dump_json() + "\n")
        except OSError as e:
            raise TournamentError(f"cannot keep a game in {path}: {e.strerror or e}") from e


def valid_name(name: str) -> str:
    """``name`` if it can be a tournament's, else raise :exc:`TournamentError`.

    A tournament's name is also its directory, so it has to be a plain one, as a run's does.
    """
    if not _NAME.fullmatch(name):
        raise TournamentError(
            f"invalid tournament name {name!r}: use letters, digits and . _ -, "
            "starting with a letter or digit"
        )
    return name


_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def create_tournament(directory: Path, definition: TournamentDefinition) -> Tournament:
    """Make a new tournament in ``directory``, which has no games yet.

    Made whole in a hidden directory of its own and then moved into place, so that a
    tournament is there with what it is or not at all, and of two made under one name at
    once only one is.

    Raises:
        TournamentError: there is a tournament of that name already, or it cannot be written.
    """
    final = directory / valid_name(definition.name)
    try:
        label = load_opening_set(definition.openings).label
    except OpeningSetError as e:
        raise TournamentError(str(e)) from e
    definition = definition.model_copy(update={"opening_set": label})
    try:
        directory.mkdir(parents=True, exist_ok=True)
        making = Path(tempfile.mkdtemp(prefix=f".{definition.name}.", dir=directory))
    except OSError as e:
        raise TournamentError(f"cannot make tournament {definition.name!r}: {e}") from e
    try:
        (making / DEFINITION_FILE).write_text(
            definition.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except OSError as e:
        shutil.rmtree(making, ignore_errors=True)
        raise TournamentError(f"cannot make tournament {definition.name!r}: {e}") from e
    try:
        # Onto an empty directory of the name, if there is one, which is no tournament; never
        # onto one with anything in it.
        os.rename(making, final)
    except OSError as e:
        shutil.rmtree(making, ignore_errors=True)
        if e.errno in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR):
            raise TournamentError(
                f"there is a tournament called {definition.name!r} already"
            ) from e
        raise TournamentError(f"cannot make tournament {definition.name!r}: {e}") from e
    return Tournament(final, definition)


def open_tournament(directory: Path, name: str) -> Tournament:
    """The tournament called ``name`` in ``directory``.

    Raises:
        NoSuchTournamentError: there is no such tournament, nor could there be one by that name.
        TournamentError: what it is cannot be read.
    """
    try:
        path = directory / valid_name(name)
    except TournamentError as e:
        raise NoSuchTournamentError(str(e)) from None
    try:
        text = (path / DEFINITION_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NoSuchTournamentError(
            f"there is no tournament called {name!r} in {directory}"
        ) from None
    except OSError as e:
        raise TournamentError(f"cannot read tournament {name!r}: {e}") from e
    try:
        return Tournament(path, TournamentDefinition.model_validate_json(text))
    except ValidationError as e:
        raise TournamentError(f"cannot read tournament {name!r}: {e}") from e


def tournament_names(directory: Path) -> list[str]:
    """The names of the tournaments in ``directory``, in order."""
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return []
    return sorted(
        entry.name
        for entry in entries
        if _NAME.fullmatch(entry.name) and (entry / DEFINITION_FILE).is_file()
    )


def describe(entrant: Entrant) -> str:
    """An entrant in a few words, such as ``tiny step 2 (rating 1500)`` or ``Stockfish 1400``."""
    if isinstance(entrant, StockfishEntrant):
        return f"Stockfish {entrant.elo}"
    sampled = f", sample {entrant.temperature:g}" if entrant.strategy == "sample" else ""
    return f"{entrant.run} step {entrant.step} (rating {entrant.rating}{sampled})"


@dataclass(frozen=True)
class ScheduledGame:
    """A game of a tournament, as it comes in the order they are played."""

    number: int
    """Where it comes among all the tournament's games, counting from 1."""
    round: int
    """Which game of its pairing it is, counting from 1."""
    first: int
    """The entrant that comes first in the pairing, by its place: White in the odd rounds, and
    not always the lower of the two (see :func:`schedule`)."""
    second: int


def schedule(definition: TournamentDefinition) -> list[ScheduledGame]:
    """Every game of a tournament, in the order they are played: round by round, and in each
    round every pairing in order.

    Which of a pair comes first, and so has White in the odd rounds, goes by whether their
    places add up to an even number: the lower first if so, the higher if not. Every entrant
    then has White in half its games of a round, or one more or one fewer, rather than the
    first entrant White in all of them, which an odd number of games a pairing, or a
    tournament stopped after an odd round, would leave with the advantage of the move.
    """
    count = len(definition.entrants)
    pairs = [
        (low, high) if (low + high) % 2 == 0 else (high, low)
        for low in range(count)
        for high in range(low + 1, count)
    ]
    rounds = range(1, definition.games_per_pairing + 1)
    return [
        ScheduledGame(number=number, round=round_, first=first, second=second)
        for number, (round_, (first, second)) in enumerate(
            ((round_, pair) for round_ in rounds for pair in pairs), start=1
        )
    ]


class Cell(BaseModel):
    """What one entrant scored against another: a square of the crosstable."""

    points: float
    games: int


class EntrantStanding(BaseModel):
    """One entrant's line of the leaderboard."""

    entrant: int
    """The entrant, by its place among them."""
    rating: float
    low: float
    high: float
    """The range the rating is good to, as likely as :data:`~chess_ai.rating.DEFAULT_CONFIDENCE`
    to hold it."""
    points: float
    games: int


class Leaderboard(BaseModel):
    """How a tournament stands: every entrant's rating, best first, and the crosstable."""

    standings: list[EntrantStanding]
    crosstable: list[list[Cell | None]]
    """What the entrant of each row scored against the entrant of each column, by their places;
    nothing where they are the same."""
    games_played: int
    games_total: int


def leaderboard(tournament: Tournament, games: list[TournamentGame] | None = None) -> Leaderboard:
    """How ``tournament`` stands after the games it has finished, or after ``games`` of them
    read already."""
    definition = tournament.definition
    count = len(definition.entrants)
    crosstable: list[list[Cell | None]] = [
        [None if i == j else Cell(points=0, games=0) for j in range(count)] for i in range(count)
    ]
    if games is None:
        games = tournament.games()
    for game in games:
        white_points = _WHITE_POINTS[game.result]
        for entrant, opponent, points in (
            (game.white, game.black, white_points),
            (game.black, game.white, 1 - white_points),
        ):
            cell = crosstable[entrant][opponent]
            assert cell is not None
            cell.points += points
            cell.games += 1
    pairings = [
        Pairing(first=i, second=j, points=cell.points, games=cell.games)
        for i in range(count)
        for j in range(i + 1, count)
        if (cell := crosstable[i][j]) is not None
    ]
    anchor = definition.anchor
    rated = rate_round_robin(
        count, pairings, anchor=(anchor.entrant, anchor.rating) if anchor is not None else None
    )
    standings = [
        EntrantStanding(
            entrant=index,
            rating=standing.rating,
            low=standing.low,
            high=standing.high,
            points=standing.points,
            games=standing.games,
        )
        for index, standing in enumerate(rated)
    ]
    standings.sort(key=lambda standing: standing.rating, reverse=True)
    return Leaderboard(
        standings=standings,
        crosstable=crosstable,
        games_played=len(games),
        games_total=definition.games_per_pairing * count * (count - 1) // 2,
    )


_WHITE_POINTS: Final = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


type PlayerFor = Callable[[int, int, int], Player]
"""The player that plays for an entrant against an opponent, both by their places, in the game
of their pairing with the number given, counting from 1."""


@asynccontextmanager
async def entrant_players(
    definition: TournamentDefinition,
    *,
    runs: Path,
    stockfish: Command,
    device: str = "cpu",
    batch_size: int | None = None,
) -> AsyncIterator[PlayerFor]:
    """The players of ``definition``'s entrants, for :func:`play_tournament`, until the block
    ends: every checkpoint loaded once, from ``runs`` onto ``device``, and every Stockfish
    entrant started once, from ``stockfish``, and closed afterwards.

    A checkpoint plays each game as a player of its own, told the rating of the opponent it
    is playing: the rating a checkpoint is asked to play like, or Stockfish's Elo. Its moves
    are drawn, if it samples them, from a generator seeded by the tournament's seed and the
    game, so that a tournament played again, or resumed, plays the games it played before.

    Raises:
        TournamentError: a checkpoint cannot be found or loaded, its file is gone before it
            is loaded or has another time than when the tournament was made, or Stockfish
            cannot be started. Whatever was started before it is closed first.
    """
    from chess_ai.inference import InferenceError, ModelPlayer, load_engine
    from chess_ai.inference.engine import DEFAULT_BATCH_SIZE
    from chess_ai.training.run_store import RunError, choose_checkpoint, open_run

    engines: dict[tuple[str, int], InferenceEngine] = {}
    started: dict[int, Player] = {}
    try:
        for index, entrant in enumerate(definition.entrants):
            if isinstance(entrant, ModelEntrant) and (entrant.run, entrant.step) not in engines:
                try:
                    run = open_run(runs, entrant.run)
                    info = choose_checkpoint(run, entrant.step)
                    # Its file's time looked at on both sides of loading it. Before, it has to
                    # be the time recorded. After, only another time refuses it, as a file
                    # saved again during the load has; one pruned meanwhile is in memory as
                    # the look before found it.
                    _same_file(run, info, entrant, index)
                    engines[(entrant.run, entrant.step)] = load_engine(
                        run.checkpoint_path(info),
                        device=device,
                        batch_size=batch_size or DEFAULT_BATCH_SIZE,
                    )
                    _same_file(run, info, entrant, index, loaded=True)
                except (RunError, InferenceError) as e:
                    raise TournamentError(f"entrant {index + 1}: {e}") from e
        for index, entrant in enumerate(definition.entrants):
            if isinstance(entrant, StockfishEntrant):
                try:
                    started[index] = await start_stockfish(
                        stockfish, elo=entrant.elo, move_time=entrant.move_time
                    )
                except StockfishError as e:
                    raise TournamentError(f"entrant {index + 1}: {e}") from e

        def player(entrant: int, opponent: int, number: int) -> Player:
            playing = definition.entrants[entrant]
            if isinstance(playing, StockfishEntrant):
                return started[entrant]
            return ModelPlayer(
                engines[(playing.run, playing.step)],
                run=playing.run,
                checkpoint=playing.step,
                rating=playing.rating,
                opponent_rating=_rating(definition.entrants[opponent]),
                strategy=playing.strategy,
                temperature=playing.temperature,
                seed=_game_seed(definition.seed, entrant, opponent, number),
            )

        yield player
    finally:
        await close_and_wait(*started.values())


def _same_file(
    run: "RunReader",
    info: "CheckpointInfo",
    entrant: ModelEntrant,
    index: int,
    *,
    loaded: bool = False,
) -> None:
    """Raise :exc:`TournamentError` unless ``entrant``'s checkpoint file is the one the
    tournament began with, by its time; one from before files' times were recorded passes.

    Once the file is ``loaded``, its being gone passes too: what is in memory is what the look
    before the load found, as a run still training may prune it meanwhile, and only another
    time says another file was read.
    """
    if entrant.checkpoint_written is None:
        return
    written = run.checkpoint_written(info)
    where = f"entrant {index + 1}: the checkpoint of {entrant.run} from step {entrant.step}"
    if written is None:
        if loaded:
            return
        raise TournamentError(f"{where} is gone")
    if written != entrant.checkpoint_written:
        # The time is all that is known: a file saved again, which is another network, and
        # one copied without its times look the same.
        raise TournamentError(
            f"{where} has changed since the tournament started (saved again, or copied "
            "without its times), and may be another network; start a new tournament for it"
        )


def _game_seed(seed: int, entrant: int, opponent: int, number: int) -> int:
    """What one checkpoint's moves in one game are drawn from: the same for the same game in
    every process and every Python, which ``hash`` of a tuple is not, since the algorithm is
    CPython's to change."""
    text = f"{seed}/{entrant}/{opponent}/{number}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def _rating(entrant: Entrant) -> int:
    """What a checkpoint playing ``entrant`` is told its rating is."""
    return entrant.rating if isinstance(entrant, ModelEntrant) else entrant.elo


async def play_tournament(
    tournament: Tournament,
    player: PlayerFor,
    *,
    on_start: Callable[[ScheduledGame], None] | None = None,
    on_game: Callable[[ScheduledGame, MatchGame], None] | None = None,
) -> None:
    """Play every game of ``tournament`` it has not finished, and keep each as it ends.

    ``on_start`` is handed each game before a move of it is played, and ``on_game`` each game,
    as the match runner played it, once it is kept: which is how a tournament says how it is
    going.

    Raises:
        TournamentError: another process, or another call here, is playing the tournament, or
            its opening set is not the one it started with.
        Whatever ``player``, or a player it gives, raises. Every game finished before is kept.
    """
    definition = tournament.definition
    try:
        openings = load_opening_set(definition.openings)
    except OpeningSetError as e:
        raise TournamentError(str(e)) from e
    if definition.opening_set is not None and openings.label != definition.opening_set:
        raise TournamentError(
            f"tournament {definition.name!r} was started with opening set "
            f"{definition.opening_set}, and {definition.openings} is {openings.label} now"
        )
    with tournament._playing_games():
        tournament._drop_torn_line()
        finished = {(game.round, *sorted((game.white, game.black))) for game in tournament.games()}
        for scheduled in schedule(definition):
            if (scheduled.round, *sorted((scheduled.first, scheduled.second))) in finished:
                continue
            first = player(scheduled.first, scheduled.second, scheduled.round)
            second = player(scheduled.second, scheduled.first, scheduled.round)
            if on_start is not None:
                on_start(scheduled)
            played = await play_game(
                first,
                second,
                scheduled.round,
                openings=openings,
                event=f"{EVENT_PREFIX} {definition.name}",
            )
            game = _finished(played, scheduled)
            tournament._append(game)
            if on_game is not None:
                on_game(scheduled, played)


class _PlayingLock:
    """The lock that makes one process at a time the one playing a tournament.

    An ``flock`` on a file of its own, which the kernel lets go of however the process holding
    it ends, so a tournament killed outright is never left locked. The file itself stays, as a
    run's lock file does: deleting it races a process locking what it just opened.
    """

    def __init__(self, tournament: Tournament) -> None:
        self._tournament = tournament
        self._descriptor: int | None = None

    def __enter__(self) -> None:
        path = self._tournament.path / LOCK_FILE
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            raise TournamentError(f"cannot open {path}: {e.strerror or e}") from e
        try:
            _lock_unless_played(descriptor)
        except BlockingIOError:
            os.close(descriptor)
            raise self._tournament._being_played() from None
        except OSError as e:
            os.close(descriptor)
            raise TournamentError(f"cannot lock {path}: {e.strerror or e}") from e
        self._descriptor = descriptor

    def __exit__(self, *exception: object) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None


LOCK_TRIES: Final = 20
"""How many times the playing lock is tried, :data:`LOCK_PAUSE` apart, before a tournament is
taken to be played by another process: :meth:`Tournament.playing` holds it for a moment each
time it looks, as the web page does every few seconds, and a player holds it for hours."""
LOCK_PAUSE: Final = 0.01

_pause = time.sleep
"""How the lock is waited for between tries, which tests replace to end a look at it there."""


def _lock_unless_played(descriptor: int) -> None:
    """Lock ``descriptor`` exclusively, waiting out a moment's look at it.

    Raises:
        BlockingIOError: it is still locked after :data:`LOCK_TRIES` tries.
    """
    for attempt in range(LOCK_TRIES):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if attempt == LOCK_TRIES - 1:
                raise
            _pause(LOCK_PAUSE)


def _finished(played: MatchGame, scheduled: ScheduledGame) -> TournamentGame:
    """``played`` as it is kept.

    Raises:
        TournamentError: it has no result, as only a game stopped part way has, and is to be
            played again rather than kept.
    """
    over = played.game.position.game_over
    if over is None or over.result not in _WHITE_POINTS:
        raise TournamentError(f"game {scheduled.number} ended with no result")
    first, second = scheduled.first, scheduled.second
    white, black = (first, second) if played.first_plays_white else (second, first)
    return TournamentGame(
        round=scheduled.round,
        white=white,
        black=black,
        result=over.result,
        termination=over.reason,
        opening=played.opening.name,
        plies=len(played.game.moves),
        pgn=played.pgn,
    )
