"""The match runner: two players, a number of games and an opening set, played out.

It knows nothing about what the players are. A checkpoint, Stockfish, the random mover and a
search that does not exist yet all play through the same game session, and the PGN of each game
says what played it because the players say so themselves.

Two things make a match between deterministic players worth playing. Every game starts from a
line of the opening set rather than from the starting position, so that argmax play does not
play one game over and over. And the players swap colours every game, each line being played
once either way round, so that a player that is only better as White does not look better.

The players are asked for moves one game after another and never at once, which is also what
makes a match with sampling players reproducible: each player draws from its own generator in
the same order every time.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import chess

from chess_ai.game_session import GameSession, GameState
from chess_ai.openings import Opening, OpeningSet
from chess_ai.output import DETAIL, Say
from chess_ai.pgn import game_pgn
from chess_ai.players import Player

EVENT = "chess-ai match"
"""The ``Event`` every game of a match is recorded under, telling them from single games."""

_Z95 = 1.96
"""How many standard errors either side of a mean a 95% range spans, by the normal curve."""

_WHITE_SCORES = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}

HEADER_EVERY: Final = 20
"""How many games :class:`GameLines` says between two headers naming who is playing."""


@dataclass(frozen=True)
class MatchGame:
    """One game of a match, as it was played and as it was written out."""

    number: int
    """Which game of the match it was, counting from 1; the PGN's ``Round``."""
    opening: Opening
    first_plays_white: bool
    """Whether the first player of the match had White in this game."""
    game: GameState
    pgn: str
    """The game as PGN, headers and all, ending with a newline."""

    @property
    def first_score(self) -> float | None:
        """What the first player scored: 1, ½ or 0, or ``None`` for a game with no result."""
        over = self.game.position.game_over
        white = _WHITE_SCORES.get(over.result) if over is not None else None
        if white is None:
            return None
        return white if self.first_plays_white else 1.0 - white


def pairing(number: int, openings: OpeningSet, *, paired: bool = True) -> tuple[Opening, bool]:
    """The opening game ``number`` (from 1) starts from, and whether the first player is White.

    Games come in pairs, both from one line, the first player taking White in the first of them
    and Black in the second. After the last line the set starts again from its first.

    Unless ``paired`` is false: then every game starts from a line of its own, the first player
    always White. That is for a player against itself, which the swap would only make replay
    the game it has just played.
    """
    index = number - 1
    lines = openings.openings
    if not paired:
        return lines[index % len(lines)], True
    return lines[(index // 2) % len(lines)], index % 2 == 0


def distinct_games(openings: OpeningSet, *, paired: bool = True) -> int:
    """How many games a match can play before a game starts the way an earlier one did."""
    return (2 if paired else 1) * len(openings.openings)


async def play_match(
    first: Player,
    second: Player,
    *,
    games: int,
    openings: OpeningSet,
    on_game: Callable[[MatchGame], None] | None = None,
    on_session: Callable[[int, GameSession], None] | None = None,
    event: str = EVENT,
    paired: bool = True,
    move_delay: float = 0.0,
) -> list[MatchGame]:
    """Play ``games`` games between ``first`` and ``second``, and return every one of them.

    ``on_game`` is handed each game as soon as it is over, which is where a match saves its
    games and says how it is going, so that a match stopped half-way has kept what it played.
    ``on_session`` is handed each game's session, with the game's number, before a move of it is
    played, which is how a match is watched while it is being played. ``event`` is what the
    games' PGN says they were played for, such as a ladder. ``paired`` is as :func:`pairing`
    has it, and ``move_delay`` as :class:`~chess_ai.game_session.GameSession` has it.

    The players are not closed here: whoever made them may have more for them to play.

    Raises:
        ValueError: ``games`` is less than one.
        IllegalMoveError: a player chose an illegal move. ``on_game`` has had every game
            finished before it. Whatever a player raises is raised here likewise.
    """
    if games < 1:
        raise ValueError(f"a match needs at least one game, not {games}")
    played = []
    for number in range(1, games + 1):
        game = await play_game(
            first,
            second,
            number,
            openings=openings,
            on_session=on_session,
            event=event,
            paired=paired,
            move_delay=move_delay,
        )
        played.append(game)
        if on_game is not None:
            on_game(game)
    return played


async def play_game(
    first: Player,
    second: Player,
    number: int,
    *,
    openings: OpeningSet,
    on_session: Callable[[int, GameSession], None] | None = None,
    event: str = EVENT,
    paired: bool = True,
    move_delay: float = 0.0,
) -> MatchGame:
    """Play game ``number`` (from 1) of a match between ``first`` and ``second``, from the line
    and with the colours :func:`pairing` gives it, as :func:`play_match` plays each of its
    games. ``on_session`` is handed the game's session before a move of it is played.

    Raises:
        IllegalMoveError: a player chose an illegal move. Whatever a player raises is raised
            here likewise.
    """
    opening, first_plays_white = pairing(number, openings, paired=paired)
    white, black = (first, second) if first_plays_white else (second, first)
    session = GameSession(white, black, opening=opening.moves, move_delay=move_delay)
    if on_session is not None:
        on_session(number, session)
    await session.play()
    state = session.state
    headers = {
        "Event": event,
        "Round": str(number),
        "Opening": opening.name,
        # A tag of this project's own: which set, and which version of it, the line was
        # taken from, since two versions of a set can give one line different names.
        "OpeningSet": openings.label,
    }
    return MatchGame(
        number=number,
        opening=opening,
        first_plays_white=first_plays_white,
        game=state,
        pgn=game_pgn(state, headers=headers),
    )


@dataclass(frozen=True)
class Score:
    """How one player did over some games: wins, draws and losses, and what they add up to."""

    wins: int
    draws: int
    losses: int
    margin: float | None
    """Half the width of a rough 95% range around :attr:`fraction`, or ``None`` with too few
    games to say. See :func:`score`."""

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def fraction(self) -> float | None:
        """The points as a share of the games, or ``None`` before there are any."""
        return self.points / self.games if self.games else None


def score(
    games: Iterable[MatchGame], *, first: bool = True, color: chess.Color | None = None
) -> Score:
    """How the first player (or the second) did in ``games``, or in those it had ``color`` in.

    The range is the normal approximation to the score's sampling error, from the spread of the
    game results themselves: a match of fifty decisive games has a range of about ±14 points of
    score, and draws narrow it. It says how far a rerun of the match on other openings might
    land, not how the players compare in Elo, which is rating math's job and not this one's.

    The spread is measured with one win and one loss added to the games (the score itself is
    not), so that eight wins out of eight do not claim a range of nothing: identical results
    say little about how the next game goes. Over fewer than two games there is no range.

    Games with no result count for nobody.
    """
    results = []
    for game in games:
        if game.first_score is None:
            continue
        plays_white = game.first_plays_white == first
        if color is not None and plays_white != (color == chess.WHITE):
            continue
        results.append(game.first_score if first else 1.0 - game.first_score)
    margin = None
    if len(results) >= 2:
        spread = [*results, 0.0, 1.0]
        mean = sum(spread) / len(spread)
        variance = sum((result - mean) ** 2 for result in spread) / (len(spread) - 1)
        margin = _Z95 * math.sqrt(variance / len(results))
    return Score(
        wins=results.count(1.0),
        draws=results.count(0.5),
        losses=results.count(0.0),
        margin=margin,
    )


def game_line(game: MatchGame, names: Sequence[str], *, number: int, games: int) -> str:
    """One finished game in a line: which of how many it was, who had which colour, and how it
    went, such as ``game 3/20, Italian Game: Stockfish 1350 – tiny 1-0 (checkmate, 41 moves)``.

    ``names`` are the first player's and the second's, which the game puts in colour order.
    """
    state = game.game
    over = state.position.game_over
    result = over.result if over is not None else "*"
    how = f"{over.reason.replace('_', ' ')}, " if over is not None else ""
    moves = (len(state.moves) + 1) // 2
    white, black = names if game.first_plays_white else reversed(names)
    return (
        f"game {number}/{games}, {game.opening.name}: {white} – {black} {result} "
        f"({how}{moves} moves)"
    )


def starting_line(
    opening: Opening, first_plays_white: bool, names: Sequence[str], *, number: int, games: int
) -> str:
    """The line :func:`game_line` says of a game, as the game starts: ``game 3/20 starting,
    Italian Game: Stockfish 1350 – tiny``."""
    white, black = names if first_plays_white else reversed(names)
    return f"game {number}/{games} starting, {opening.name}: {white} – {black}"


@dataclass
class LinesMatch:
    """One match :class:`GameLines` says the games of, as :meth:`GameLines.match` starts it."""

    before: int
    """How many games the matches before this one play."""
    names: tuple[str, str]
    games: int
    paired: bool
    players: str
    header_due: int = 1
    """The number, in this match, of the next game to say the header before."""


class GameLines:
    """Says the games of one or more matches as each starts and ends, numbered from the first
    game of the first match, under a header that says who is playing.

    The game lines call the players by short names, such as ``model`` or ``Stockfish 1350``,
    since a run's name can be as long as a line of the terminal. The header says what they
    are, and comes before the first game, before the first game of each match, and otherwise
    every :data:`HEADER_EVERY` games, so that a log looked at from anywhere says it within a
    screenful: ``chess-ai: RUN@30000 sample games 3–22 of 102: model against Stockfish 1350``.
    ``title`` is what comes before the range, such as ``RUN@30000 sample games``, and ``total``
    is how many games all the matches play.

    Games are said of the match :meth:`match` started last, unless they are said of one it
    returned, as the games of matches played at once are.

    The game lines are :data:`~chess_ai.output.DETAIL` lines and the headers milestones.
    """

    def __init__(
        self,
        say: Say,
        title: str,
        *,
        total: int,
        openings: OpeningSet,
        every: int = HEADER_EVERY,
    ) -> None:
        self._say = say
        self._title = title
        self._total = total
        self._openings = openings
        self._every = every
        self._current = LinesMatch(before=0, names=("", ""), games=0, paired=True, players="")

    def match(
        self,
        names: Sequence[str],
        *,
        games: int,
        paired: bool = True,
        players: str | None = None,
        before: int | None = None,
    ) -> LinesMatch:
        """Start saying the games of the next match, between the players ``names`` calls
        the first and the second, as :func:`play_match` plays ``games`` of them, ``paired``
        or not. ``players`` is what the header says of them, by default ``first against
        second``, or ``first against itself``. ``before`` is how many games come before the
        match's first, by default those of every match started before it."""
        first, second = names
        if players is None:
            players = f"{first} against itself" if first == second else f"{first} against {second}"
        self._current = LinesMatch(
            before=before if before is not None else self._current.before + self._current.games,
            names=(first, second),
            games=games,
            paired=paired,
            players=players,
        )
        return self._current

    def starting(self, number: int, match: LinesMatch | None = None) -> None:
        """Say that game ``number`` of ``match``, counting from 1, is starting."""
        match = match if match is not None else self._current
        if number >= match.header_due:
            # From the game it was due before, which is this one unless games played at once
            # began out of order.
            first = match.header_due
            last = min(first + self._every - 1, match.games)
            first_number, last_number = match.before + first, match.before + last
            games = f"{first_number}" if last == first else f"{first_number}–{last_number}"
            self._say(f"chess-ai: {self._title} {games} of {self._total}: {match.players}")
            match.header_due = first + self._every
        opening, first_plays_white = pairing(number, self._openings, paired=match.paired)
        line = starting_line(
            opening,
            first_plays_white,
            match.names,
            number=match.before + number,
            games=self._total,
        )
        self._say(line, kind=DETAIL)

    def ended(self, game: MatchGame, match: LinesMatch | None = None) -> None:
        """Say how ``game``, of ``match``, went."""
        match = match if match is not None else self._current
        line = game_line(game, match.names, number=match.before + game.number, games=self._total)
        self._say(line, kind=DETAIL)


def match_filename(now: datetime | None = None) -> str:
    """A name for a match's PGN file, which sorts it among the saved games by when it began.

    The id tells apart two matches begun in the same second.
    """
    when = now if now is not None else datetime.now()
    return f"{when:%Y%m%d-%H%M%S}-match-{uuid4().hex[:8]}.pgn"


def append_game(path: Path, game: MatchGame) -> None:
    """Add ``game`` to the end of the match file at ``path``, making the file if need be.

    One file for the whole match, which the replay view opens as a list of its games. Each game
    is written whole as it ends, so the file holds every game a stopped match finished.

    The directory has to be there: one that has gone, such as the evaluation directory of a run
    renamed while its games were played, is not made again under the old name.

    Raises:
        OSError: the file cannot be written, or its directory is not there.
    """
    with path.open("a", encoding="utf-8") as file:
        # PGN separates the games of a file with a blank line.
        file.write(f"{game.pgn}\n")
