"""Replay: a PGN file read back as the positions its games pass through.

A game is replayed on the server, where the rules are, and sent to the browser as one
snapshot per ply. The browser steps between them; nothing is worked out there.

A file can hold any number of games, so reading it comes in two parts: every game's
headers, which are small enough to send for all of them at once, and the positions of
the one game being looked at, which are not. Both come from the same reading of the
file, so the text is posted again to look at a second game. That keeps nothing on the
server between requests, which is the point: an uploaded file is one person's business
for as long as they are looking at it.

A file with a game in it that cannot be read is refused whole, naming the game and what
is wrong with it, rather than quietly leaving that game out: a game missing from the
middle of a file is the kind of thing nobody notices until it matters.

What counts as a game is the tag pairs the file wrote for it. PGN files carry prose
around their games, and prose about chess reads as chess: a sentence leaves the parser a
move or two, sometimes a result, sometimes a move it cannot play at all. Nothing in the
record a note comes back as tells it from the record a game written without tag pairs
comes back as — python-chess fills in the seven tags and the result either way — so
weighing the moves or the result is guesswork, and guessing wrong either lists a note as
a game a viewer can pick, or refuses an ordinary annotated export whole. The tag pairs
are the one thing only the file can have written.

The price is paid by a game written with no tag pairs in a file that has them
elsewhere: it is read as a note and left out, silently, because it cannot be told from
one. Files of nothing but moves are the common way that shape turns up and are read as
the game they are; a game appended to an exported file without its tags is not.
"""

import io
import re
from pathlib import Path
from typing import NamedTuple

import chess
import chess.pgn
from pydantic import BaseModel

from chess_ai.position_view import PositionSnapshot, Result, snapshot

MAX_BYTES = 8 * 1024 * 1024
"""The largest PGN file that can be read, which is a few thousand games."""

SUFFIX = ".pgn"
"""What a file in the games directory is called, and all that is looked at there."""

_RESULTS: set[str] = {"1-0", "0-1", "1/2-1/2", "*"}

_NO_HEADERS = dict(chess.pgn.Headers())
"""The headers python-chess gives a record nothing was read for; see :func:`_names_a_game`."""

_TAG_PAIR = re.compile(r'^\[[A-Za-z0-9_]+\s+"', re.MULTILINE)
"""A PGN tag pair, as a file writes one: ``[Event "..."]`` at the start of a line."""


class _Record(NamedTuple):
    """One record of a file, and whether the file wrote it as a game."""

    game: chess.pgn.Game
    tagged: bool
    """Whether the file wrote tag pairs for this record, which is what makes it a game."""


class _RecordBuilder(chess.pgn.GameBuilder):
    """python-chess's reader, saying as well whether the record had tag pairs of its own.

    The headers a record comes back with cannot answer that. PGN asks every game for
    seven tags, and python-chess hands those seven back, as placeholders, for every
    record it read none from; it also writes the result it finds in the movetext into
    them, so a note ending "1-0" arrives looking as tagged as a game, while a genuinely
    unattributed game — the seven tags with nothing filled in — arrives looking like a
    note. The moves are no test either, since a sentence about chess leaves the parser a
    move or two. Only the reader knows what the file actually wrote, and this says so:
    ``visit_header`` is called for a tag pair the file wrote and for nothing filled in
    for one.
    """

    def begin_game(self) -> None:
        self.tagged = False
        return super().begin_game()

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.tagged = True
        return super().visit_header(tagname, tagvalue)

    def result(self) -> _Record:
        return _Record(super().result(), self.tagged)


class MalformedPgnError(ValueError):
    """A file does not hold a game that can be read.

    The message is shown to whoever uploaded the file, so it says which game is at fault
    and what is wrong with it.
    """


class NoSuchGameError(LookupError):
    """The game asked for is not there: no such file, or no game at that number."""


class GameSummary(BaseModel):
    """One game's headers: enough to pick it out of a file, and not a move of it."""

    index: int
    """Which game of the file this is, counting from zero; how it is asked for."""
    event: str
    site: str
    date: str
    """As PGN writes it ("2000.11.04"), unknown parts and all ("2000.??.??")."""
    round: str
    white: str
    black: str
    result: Result
    """Anything PGN does not recognise as a result is reported as "*", an unfinished game."""
    termination: str | None
    """How the game ended, where the file says; games saved here carry it."""
    plies: int
    """How many moves of the main line there are to step through."""


class ReplayMove(BaseModel):
    san: str
    uci: str
    position: PositionSnapshot
    """The position the move leads to, which carries the move as its last move."""


class ReplayGame(GameSummary):
    """One game of a file, with the position before every move and after it."""

    start_fen: str
    """The position the game began in, which says how its moves are numbered."""
    start_position: PositionSnapshot
    moves: list[ReplayMove]
    """The main line only: a variation in the file is not a game that was played."""


class ReplayFile(BaseModel):
    """What a PGN file holds: every game's headers, and one game to step through."""

    games: list[GameSummary]
    selected: ReplayGame


class SavedGame(BaseModel):
    """A game in the games directory, as the list of them describes it."""

    name: str
    """The file's name, which is what opens it."""
    event: str
    date: str
    white: str
    black: str
    result: Result


def decode(data: bytes) -> str:
    """``data`` as text, however the tool that wrote the PGN file spelled its names.

    PGN is written in UTF-8 by everything current, with or without a byte order mark,
    and in Latin-1 by what came before. Latin-1 reads any bytes at all, so a file that
    is not text in either ends up as a file with no game in it rather than an error
    about encodings, which is the more useful of the two things to be told.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read(text: str, *, selected: int = 0) -> ReplayFile:
    """Every game ``text`` holds, with game number ``selected`` replayed in full.

    The games are read one at a time and let go once they have been summarised, all but
    the one being replayed: a file of thousands of games is to cost the memory of the
    largest game in it, not of all of them at once.

    Raises:
        MalformedPgnError: there is no game in the file, or one of them cannot be read.
        NoSuchGameError: the file has no game with that number.
    """
    if selected < 0:
        raise NoSuchGameError(f"there is no game {selected}: the games are counted from zero")
    stream = io.StringIO(text)
    # A file with no tag pair anywhere in it is not PGN by the letter of the standard,
    # but it is what moves pasted into a file look like, and there is no game in such a
    # file for a note to sit beside: the first record with moves in it is read as the
    # game they are. Taken once, so that a note written under the moves is still a note.
    pasted_moves = _TAG_PAIR.search(text) is None
    summaries: list[GameSummary] = []
    replayed: ReplayGame | None = None
    unreadable: str | None = None
    while (record := chess.pgn.read_game(stream, Visitor=_RecordBuilder)) is not None:
        game, tagged = record
        holds_the_pasted_moves = pasted_moves and game.next() is not None
        pasted_moves = pasted_moves and not holds_the_pasted_moves
        # Written down as a game: its file gave it tag pairs of its own, or its file
        # gave none to anything and these are the moves it was written to hold.
        if not (tagged or holds_the_pasted_moves):
            # A note rather than a game: a PGN file carries prose around its games, and
            # prose about chess leaves the parser a move or two and often something it
            # cannot play at all. Neither is a reason to refuse the file. The first such
            # reason is kept in case the file turns out to hold no game, since then it
            # is all there is to say about why — said of the file, which is all that can
            # be said of it, rather than of a game it has not got.
            if game.errors and unreadable is None:
                unreadable = f"no game was found in that file: {game.errors[0]}"
            continue
        if game.errors:
            # A game, and not one that can be read: the file is refused for it.
            raise MalformedPgnError(
                f"{_named(game, len(summaries))} cannot be read: {game.errors[0]}"
            )
        summary = _summary(game, len(summaries))
        summaries.append(summary)
        if summary.index == selected:
            replayed = _replay(game, summary)
    if not summaries:
        raise MalformedPgnError(unreadable or "no game was found in that file")
    if replayed is None:
        raise NoSuchGameError(
            f"that file has no game {selected + 1}: there "
            f"{'is 1 game' if len(summaries) == 1 else f'are {len(summaries)} games'} in it"
        )
    return ReplayFile(games=summaries, selected=replayed)


def saved_games(directory: Path) -> list[SavedGame]:
    """The games saved in ``directory``, the most recent first.

    Games saved here are named after the moment they were written out, so their names
    put them in the order they were played. A directory with no games in it, or none
    yet, is simply empty.
    """
    if not directory.is_dir():
        return []
    saved = []
    for path in sorted(directory.glob(f"*{SUFFIX}"), reverse=True):
        try:
            headers = chess.pgn.read_headers(io.StringIO(decode(path.read_bytes())))
        except OSError:
            continue  # Taken away or unreadable since the directory was listed.
        if headers is None or not _names_a_game(headers):
            continue  # Not a game, whatever the file is called.
        saved.append(
            SavedGame(
                name=path.name,
                event=headers.get("Event", "?"),
                date=headers.get("Date", "????.??.??"),
                white=headers.get("White", "?"),
                black=headers.get("Black", "?"),
                result=_result(headers.get("Result", "*")),
            )
        )
    return saved


def saved_game(directory: Path, name: str) -> str:
    """The text of the game called ``name`` in ``directory``.

    ``name`` has to be a plain file name, as the listing gives it, so that a name made
    up elsewhere cannot reach out of the games directory.

    Raises:
        NoSuchGameError: there is no saved game by that name.
    """
    missing = NoSuchGameError(f"there is no saved game called {name!r}")
    if name != Path(name).name or name.startswith(".") or not name.endswith(SUFFIX):
        raise missing
    try:
        return decode((directory / name).read_bytes())
    except OSError as unreadable:
        raise missing from unreadable


def _named(game: chess.pgn.Game, index: int) -> str:
    """Which game of the file this is, for a message about what is wrong with it.

    Numbered among the games, since that is what somebody sent a message like this
    counts through the file to find, and named as well wherever the file says who
    played it: a number alone is a poor handle on a file of a thousand games.
    """
    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    # One side unattributed is common enough in exports, and the other side is still
    # the handle on the game: only a game the file names neither player of goes bare.
    played_by = f" ({white} – {black})" if (white, black) != ("?", "?") else ""
    return f"game {index + 1}{played_by}"


def _names_a_game(headers: chess.pgn.Headers) -> bool:
    """Whether ``headers`` were read off a game, for a listing that has only the headers.

    Only the games directory is read this way, where every file was written by this
    server and carries its tags. Reading a file somebody else wrote asks
    :class:`_RecordBuilder` instead, which knows what the file actually said.
    """
    written = dict(headers)
    return bool(written) and written != _NO_HEADERS


def _summary(game: chess.pgn.Game, index: int) -> GameSummary:
    headers = game.headers
    return GameSummary(
        index=index,
        event=headers.get("Event", "?"),
        site=headers.get("Site", "?"),
        date=headers.get("Date", "????.??.??"),
        round=headers.get("Round", "?"),
        white=headers.get("White", "?"),
        black=headers.get("Black", "?"),
        result=_result(headers.get("Result", "*")),
        termination=headers.get("Termination"),
        plies=sum(1 for _ in game.mainline_moves()),
    )


def _replay(game: chess.pgn.Game, summary: GameSummary) -> ReplayGame:
    """``game`` with the position it started in and the one every move leads to."""
    try:
        board = game.board()
        start_fen = board.fen()
        start = snapshot(board, legal_moves=False)
        moves = []
        for move in game.mainline_moves():
            # Named before it is played, since notation says what the position allowed.
            san = board.san(move)
            board.push(move)
            moves.append(
                ReplayMove(san=san, uci=move.uci(), position=snapshot(board, legal_moves=False))
            )
    except ValueError as unplayable:
        raise MalformedPgnError(
            f"game {summary.index + 1} cannot be read: {unplayable}"
        ) from unplayable
    return ReplayGame(
        **summary.model_dump(), start_fen=start_fen, start_position=start, moves=moves
    )


def _result(written: str) -> Result:
    """The result a file gives, or "*" for anything PGN has no result for."""
    return written if written in _RESULTS else "*"  # type: ignore[return-value]
