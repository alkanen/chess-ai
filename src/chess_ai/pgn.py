"""PGN: a game written out as the file every other chess tool reads.

A game that reaches a result is saved in the games directory, one file per game, and any
game can be downloaded from the browser at any point, finished or not. Either way the
file carries what it takes to replay the game: who played it, when, how it ended, and
the position it started from when that was not the usual one.
"""

import re
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from chess_ai.game_session import GameState
from chess_ai.position_view import GameOverReason, board_from_fen

MEDIA_TYPE = "application/x-chess-pgn"
"""What a PGN file is served as, so that a browser saves it rather than showing it."""

EVENT = "chess-ai game"
"""The ``Event`` every game played here belongs to; these games are nobody's tournament."""

COLUMNS = 80
"""How long a line of moves may be, which is the width PGN's export format asks for."""

TERMINATIONS: dict[GameOverReason, str] = {
    "checkmate": "checkmate",
    "stalemate": "stalemate",
    "insufficient_material": "insufficient material",
    "threefold_repetition": "threefold repetition",
    "fifty_move_rule": "fifty-move rule",
    "resignation": "resignation",
    "abort": "abandoned",
}
"""How the game ended, for the ``Termination`` header.

PGN's own vocabulary for this tag says no more than "normal" about every way a game of
chess can end on the board, so the reason is written out instead. "abandoned" is the one
word PGN has that says what it means: the game was given up rather than played out.
"""

_UNNAMEABLE = re.compile(r"[^A-Za-z0-9-]+")


def game_pgn(game: GameState, *, now: datetime | None = None) -> str:
    """``game`` as PGN, ready to be written to a file or sent to a browser.

    Any game can be written out, finished or not: a game still being played is given the
    result "*", and no ``Termination``, which is PGN for a game that has not ended. A
    game that began in a position of its own carries the ``FEN`` and ``SetUp`` headers,
    so that whoever reads it back starts where the game started.

    ``now`` is the moment the game is written out, which for a game saved as it ends is
    when it ended. It dates the game; tests give a fixed one.
    """
    when = now if now is not None else datetime.now()
    record = chess.pgn.Game()
    # Adds the FEN and SetUp headers, unless the game started where games start.
    record.setup(board_from_fen(game.start_fen))
    over = game.position.game_over
    record.headers.update(
        {
            "Event": EVENT,
            "Date": f"{when:%Y.%m.%d}",
            "Time": f"{when:%H:%M:%S}",
            # The games here are single games, not rounds of anything, which "-" is PGN
            # for. Left as the "?" it starts as, it would mean an unknown round instead.
            "Round": "-",
            "White": game.white.name,
            "Black": game.black.name,
            "WhiteType": _player_type(game.white.accepts_moves),
            "BlackType": _player_type(game.black.accepts_moves),
            "Result": over.result if over is not None else "*",
        }
    )
    if over is not None:
        record.headers["Termination"] = TERMINATIONS[over.reason]
    node: chess.pgn.GameNode = record
    for move in game.moves:
        node = node.add_main_variation(chess.Move.from_uci(move.uci))
    # Written out in the lines PGN asks for rather than python-chess's own single line
    # per game, so that the file reads in a text editor as well as in a chess program.
    written = record.accept(chess.pgn.StringExporter(columns=COLUMNS, headers=True))
    # A PGN file's last game ends with a newline, like every other line of it.
    return f"{written}\n"


def pgn_filename(game: GameState, *, now: datetime | None = None) -> str:
    """A name for ``game``'s PGN file: when it was written out, and which game it is.

    The time sorts the games in the order they were played, and the game's own id tells
    two games written out in the same second apart.
    """
    when = now if now is not None else datetime.now()
    # A game names its own file and the download it is served as, so the id is cut down
    # to the letters and digits it is made of.
    named = _UNNAMEABLE.sub("", game.id)[:8] or "game"
    return f"{when:%Y%m%d-%H%M%S}-{named}.pgn"


def save_game(game: GameState, directory: Path, *, now: datetime | None = None) -> Path:
    """Write ``game`` into ``directory`` as a PGN file of its own, and return the file.

    The directory is made if it is not there yet, so that a first game saves as readily
    as the thousandth.
    """
    when = now if now is not None else datetime.now()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / pgn_filename(game, now=when)
    path.write_text(game_pgn(game, now=when), encoding="utf-8")
    return path


def _player_type(accepts_moves: bool) -> str:
    """What PGN calls a player: a person at a board, or something computing its moves."""
    return "human" if accepts_moves else "program"
