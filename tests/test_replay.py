"""Reading PGN files back: the games in them, and the positions their moves pass through."""

import gc
import json
import weakref
from pathlib import Path

import chess.pgn
import pytest

from chess_ai.replay import (
    MalformedPgnError,
    NoSuchGameError,
    decode,
    read,
    saved_game,
    saved_games,
)

FOOLS_MATE = """[Event "chess-ai game"]
[Site "?"]
[Date "2026.09.24"]
[Round "-"]
[White "Human"]
[Black "Random mover"]
[Result "0-1"]
[Termination "checkmate"]

1. f3 e5 2. g4 Qh4# 0-1
"""

SCHOLARS_MATE = """[Event "Kitchen table"]
[Site "Trondheim"]
[Date "2001.04.03"]
[Round "2"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

FROM_A_POSITION = """[Event "Endgame"]
[Site "?"]
[Date "2026.01.01"]
[Round "-"]
[White "Alice"]
[Black "Bob"]
[Result "1/2-1/2"]
[SetUp "1"]
[FEN "7k/8/5K2/8/8/8/8/6R1 w - - 0 60"]

60. Rg2 Kh7 1/2-1/2
"""

TWO_GAMES = f"{FOOLS_MATE}\n{SCHOLARS_MATE}"

FRONTEND_FIXTURES = Path(__file__).parents[1] / "frontend" / "src" / "test" / "fixtures"

MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
"""Where fool's mate ends, which is the last of the positions that game steps through."""


def test_every_game_in_a_file_is_listed_by_the_headers_that_tell_them_apart():
    file = read(TWO_GAMES)

    assert [(game.index, game.white, game.black, game.result) for game in file.games] == [
        (0, "Human", "Random mover", "0-1"),
        (1, "Alice", "Bob", "1-0"),
    ]
    assert [(game.date, game.event, game.plies) for game in file.games] == [
        ("2026.09.24", "chess-ai game", 4),
        ("2001.04.03", "Kitchen table", 7),
    ]


def test_the_first_game_is_the_one_replayed_unless_another_is_asked_for():
    assert read(TWO_GAMES).selected.index == 0
    assert read(TWO_GAMES).selected.white == "Human"


def test_any_game_of_a_file_can_be_the_one_replayed():
    file = read(TWO_GAMES, selected=1)

    assert file.selected.index == 1
    assert [move.san for move in file.selected.moves] == [
        "e4",
        "e5",
        "Bc4",
        "Nc6",
        "Qh5",
        "Nf6",
        "Qxf7#",
    ]
    # The whole file is described whichever game of it is being looked at, so that a
    # viewer can move on to another game without sending the file up again.
    assert len(file.games) == 2


def test_a_replayed_game_carries_the_position_before_every_move_and_after_it():
    game = read(FOOLS_MATE).selected

    assert game.start_fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert game.start_position.turn == "white"
    assert game.start_position.last_move is None
    assert [move.san for move in game.moves] == ["f3", "e5", "g4", "Qh4#"]
    assert [move.uci for move in game.moves] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert [move.position.turn for move in game.moves] == ["black", "white", "black", "white"]
    assert game.moves[-1].position.fen == MATE_FEN


def test_every_step_shows_the_move_that_led_to_it():
    """The board draws the last move from the position it is showing, and nothing else."""
    game = read(FOOLS_MATE).selected

    assert [
        (move.position.last_move.from_square, move.position.last_move.to_square)
        for move in game.moves
        if move.position.last_move is not None
    ] == [("f2", "f3"), ("e7", "e5"), ("g2", "g4"), ("d8", "h4")]


def test_a_step_into_check_says_where_the_king_in_check_stands():
    game = read(FOOLS_MATE).selected

    assert [move.position.check_square for move in game.moves] == [None, None, None, "e1"]
    over = game.moves[-1].position.game_over
    assert over is not None
    assert (over.result, over.reason) == ("0-1", "checkmate")


def test_a_replayed_position_offers_no_moves_to_play():
    """Nobody plays from a game that has been played; the moves are most of the size."""
    game = read(FOOLS_MATE).selected

    assert game.start_position.legal_moves == {}
    assert all(move.position.legal_moves == {} for move in game.moves)


def test_a_game_that_began_in_a_position_of_its_own_is_replayed_from_there():
    game = read(FROM_A_POSITION).selected

    assert game.start_fen == "7k/8/5K2/8/8/8/8/6R1 w - - 0 60"
    assert game.start_position.pieces["g1"].type == "rook"
    assert [move.san for move in game.moves] == ["Rg2", "Kh7"]


def test_a_game_written_with_comments_and_variations_is_replayed_along_its_main_line():
    """A variation is a move somebody thought about, not a move that was played."""
    annotated = """[Event "Annotated"]
[White "Alice"]
[Black "Bob"]
[Result "*"]

1. e4 {a good start} e5 (1... c5 {the Sicilian} 2. Nf3) 2. Nf3 $1 Nc6 *
"""

    game = read(annotated).selected

    assert [move.san for move in game.moves] == ["e4", "e5", "Nf3", "Nc6"]


def test_a_result_the_file_does_not_spell_out_is_reported_as_an_unfinished_game():
    unfinished = '[Event "x"]\n[White "Alice"]\n[Black "Bob"]\n[Result "won by Alice"]\n\n1. e4 *\n'

    assert read(unfinished).selected.result == "*"


def test_the_termination_a_file_gives_says_how_the_game_ended():
    assert read(FOOLS_MATE).selected.termination == "checkmate"
    assert read(SCHOLARS_MATE).selected.termination is None


def test_a_file_that_holds_no_game_is_refused_with_a_reason():
    with pytest.raises(MalformedPgnError, match="no game was found in that file"):
        read("Dear diary, today I played some chess. It was lovely.\n")


def test_an_empty_file_is_refused_with_a_reason():
    with pytest.raises(MalformedPgnError, match="no game was found in that file"):
        read("")


def test_a_game_with_a_move_that_cannot_be_played_is_refused_and_named():
    """The game is named, because the file it is in may hold hundreds of others."""
    broken = (
        f'{FOOLS_MATE}\n[Event "Broken"]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 e5 2. Qh8 *\n'
    )

    with pytest.raises(MalformedPgnError, match=r"game 2 \(Alice – Bob\) cannot be read: illegal"):
        read(broken)


def test_a_game_that_starts_from_a_position_that_is_not_one_is_refused():
    nonsense = '[Event "x"]\n[SetUp "1"]\n[FEN "not a position"]\n\n1. e4 *\n'

    with pytest.raises(MalformedPgnError, match="game 1 cannot be read"):
        read(nonsense)


def test_asking_for_a_game_a_file_does_not_have_says_how_many_it_has():
    with pytest.raises(NoSuchGameError, match="no game 3: there are 2 games in it"):
        read(TWO_GAMES, selected=2)

    with pytest.raises(NoSuchGameError, match="no game 2: there is 1 game in it"):
        read(FOOLS_MATE, selected=1)


def test_a_game_before_the_first_one_reaches_no_game():
    """Counting from the end is Python's habit, not a viewer's: game -1 is not the last."""
    with pytest.raises(NoSuchGameError):
        read(TWO_GAMES, selected=-1)


def test_a_file_written_in_utf_8_is_read_with_the_names_its_players_have():
    written = FOOLS_MATE.replace("Human", "Magnus Carlsén")

    assert read(decode(written.encode("utf-8"))).selected.white == "Magnus Carlsén"
    # Written by a tool that marks its files as UTF-8, which is a mark and not a name.
    assert read(decode(written.encode("utf-8-sig"))).selected.white == "Magnus Carlsén"


def test_a_file_written_before_utf_8_is_read_rather_than_refused():
    """PGN from an older program is Latin-1, and its accented names still belong to it."""
    written = FOOLS_MATE.replace("Human", "Magnus Carlsén")

    assert read(decode(written.encode("latin-1"))).selected.white == "Magnus Carlsén"


def saved(directory, name: str, pgn: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(pgn, encoding="utf-8")


def test_saved_games_are_listed_with_the_date_the_players_and_the_result(tmp_path):
    saved(tmp_path, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)

    assert [game.model_dump() for game in saved_games(tmp_path)] == [
        {
            "name": "20260924-143005-3f9a1b2c.pgn",
            "event": "chess-ai game",
            "date": "2026.09.24",
            "white": "Human",
            "black": "Random mover",
            "result": "0-1",
        }
    ]


def test_the_most_recently_played_game_is_listed_first(tmp_path):
    """Games are named after the moment they were saved, so the names are that order."""
    saved(tmp_path, "20260924-143005-aaaaaaaa.pgn", FOOLS_MATE)
    saved(tmp_path, "20260925-090000-bbbbbbbb.pgn", SCHOLARS_MATE)
    saved(tmp_path, "20260101-120000-cccccccc.pgn", FOOLS_MATE)

    assert [game.name[:8] for game in saved_games(tmp_path)] == ["20260925", "20260924", "20260101"]


def test_a_games_directory_with_nothing_in_it_yet_lists_nothing(tmp_path):
    assert saved_games(tmp_path / "never-played") == []
    (tmp_path / "empty").mkdir()
    assert saved_games(tmp_path / "empty") == []


def test_a_file_in_the_games_directory_that_is_not_a_game_is_passed_over(tmp_path):
    saved(tmp_path, "notes.txt", "these are my notes, not a game")
    saved(tmp_path, "rubbish.pgn", "nor is this")
    saved(tmp_path, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)

    assert [game.name for game in saved_games(tmp_path)] == ["20260924-143005-3f9a1b2c.pgn"]


def test_a_saved_game_is_opened_by_the_name_the_listing_gives_it(tmp_path):
    saved(tmp_path, "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)

    game = read(saved_game(tmp_path, "20260924-143005-3f9a1b2c.pgn")).selected

    assert [move.san for move in game.moves] == ["f3", "e5", "g4", "Qh4#"]


@pytest.mark.parametrize(
    "name",
    [
        "../secrets.pgn",
        "../../etc/passwd",
        "/etc/passwd",
        "games/../../secrets.pgn",
        ".hidden.pgn",
        "notes.txt",
    ],
)
def test_a_name_that_is_not_a_saved_games_name_reaches_nothing(tmp_path, name):
    """Only a file in the games directory, called what a saved game is called, is read."""
    saved(tmp_path / "games", "20260924-143005-3f9a1b2c.pgn", FOOLS_MATE)
    (tmp_path / "secrets.pgn").write_text(FOOLS_MATE, encoding="utf-8")
    (tmp_path / "games" / ".hidden.pgn").write_text(FOOLS_MATE, encoding="utf-8")
    (tmp_path / "games" / "notes.txt").write_text(FOOLS_MATE, encoding="utf-8")

    with pytest.raises(NoSuchGameError):
        saved_game(tmp_path / "games", name)


def test_a_saved_game_that_is_not_there_says_so(tmp_path):
    with pytest.raises(NoSuchGameError, match="no saved game called 'gone.pgn'"):
        saved_game(tmp_path, "gone.pgn")


def test_a_file_whose_moves_cannot_be_played_says_what_is_wrong_with_them():
    """ "No game was found" alone hides the one thing the uploader can act on.

    A file of moves that will not play is not a game and is not named as one, but the
    move that stopped it being read is still worth handing back.
    """
    with pytest.raises(MalformedPgnError) as refused:
        read("1. Qh8 *\n")

    assert str(refused.value).startswith("no game was found in that file: illegal san: 'Qh8'")


def test_a_refusal_names_the_players_of_the_game_it_is_about():
    """A number alone is worth little in a file whose games a reader has to count."""
    named = '[Event "Broken"]\n[White "Carol"]\n[Black "Dave"]\n\n1. e4 e5 2. Qh8 *\n'
    broken = f"{FOOLS_MATE}\n{named}"

    with pytest.raises(MalformedPgnError, match=r"game 2 \(Carol – Dave\) cannot be read"):
        read(broken)


def test_a_note_after_the_last_game_does_not_refuse_the_file():
    """A PGN file carries prose, and prose about chess reads as chess to a PGN reader.

    python-chess makes a record of whatever follows the last game, so a footer that
    mentions a move is a record with an error in it and nothing else in it at all.
    Refusing the file for that turns an ordinary export into an unreadable one.
    """
    with_a_note = f"{SCHOLARS_MATE}\nNote: 1... Qh4 was possible.\n"

    file = read(with_a_note)

    assert [game.white for game in file.games] == ["Alice"]
    assert [move.san for move in file.selected.moves][:2] == ["e4", "e5"]


def test_a_note_whose_moves_can_be_played_does_not_refuse_the_file():
    """The commoner shape of note: what it mentions is legal, and it is still a note.

    "e4" out of a sentence is a move the parser can play, so the record it makes of the
    note has a move in it — and, a word later, something it cannot play at all.
    """
    with_a_note = f"{SCHOLARS_MATE}\nNote: e4 was better than d4 here.\n"

    file = read(with_a_note)

    assert [game.white for game in file.games] == ["Alice"]


def test_a_note_between_two_games_is_not_counted_as_one_of_them():
    """Worse than a message about the wrong game: nobody is told anything at all.

    A note that reads cleanly used to be listed between the games it sits between, so
    the game a viewer picked by number was the note rather than the game.
    """
    between = f"{SCHOLARS_MATE}\nAnnotation: Nf3 is the main line.\n\n{FOOLS_MATE}"

    file = read(between, selected=1)

    assert [(game.index, game.white) for game in file.games] == [(0, "Alice"), (1, "Human")]
    assert [move.san for move in file.selected.moves] == ["f3", "e5", "g4", "Qh4#"]


def test_a_note_that_mentions_the_result_is_not_counted_as_a_game():
    """A note ending in a result reads as headed as a game, and is still a note.

    python-chess writes the result it finds in the movetext into the headers, so the
    headers a record comes back with cannot say whether its file wrote any.
    """
    with_a_note = f"{SCHOLARS_MATE}\nThe match finished 1-0 to Alice.\n"

    file = read(with_a_note)

    assert [(game.index, game.white, game.black) for game in file.games] == [(0, "Alice", "Bob")]


def test_a_note_carrying_both_a_move_and_a_result_is_not_counted_as_a_game():
    """Every signal a note can borrow, in one note: a playable move and a result.

    A note is prose about a game, so the two turn up together as a matter of course.
    Counted as a game it takes a number, and the game after it answers to the number
    before its own — which is the reading a viewer picks a game by.
    """
    between = f"{SCHOLARS_MATE}\nAnnotation: Nf3 is the main line, 1-0.\n\n{FOOLS_MATE}"

    file = read(between, selected=1)

    assert [(game.index, game.white) for game in file.games] == [(0, "Alice"), (1, "Human")]
    assert [move.san for move in file.selected.moves] == ["f3", "e5", "g4", "Qh4#"]


def test_a_note_whose_result_comes_before_its_unplayable_move_does_not_refuse_the_file():
    """The same note, read the other way round, and the louder half of the same fault.

    A result early in a sentence and a move the parser cannot play after it is an
    ordinary thing to write. Taken for a game, the file it annotates is refused whole
    and the refusal names a game the file has not got.
    """
    annotated = (
        '[Event "B"]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 e5 1-0\n\n'
        "Alice won 1-0. Note: e4 was better than d4 here.\n"
    )

    file = read(annotated)

    assert [(game.index, game.white, game.plies) for game in file.games] == [(0, "Alice", 2)]


def test_a_game_whose_headers_are_all_placeholders_is_still_a_game():
    """An unattributed game is written as the seven tags with nothing filled in.

    Which is, to the letter, what python-chess hands back for a record that had no tags
    at all — so what tells them apart is that the file wrote the tags, not what they say.
    """
    unattributed = (
        '[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n'
        '[Round "?"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. e4 e5 *\n'
    )

    file = read(unattributed)

    assert [move.san for move in file.selected.moves] == ["e4", "e5"]
    assert (file.selected.white, file.selected.black) == ("?", "?")


def test_a_file_of_nothing_but_moves_is_read_as_the_game_it_is():
    """Moves pasted into a file with no tags around them: not PGN, and still a game.

    Nothing in such a file is a note, because there is no game for a note to sit beside.
    """
    file = read("1. e4 e5 2. Nf3 Nc6 *\n")

    assert [move.san for move in file.selected.moves] == ["e4", "e5", "Nf3", "Nc6"]


def test_a_file_of_nothing_but_moves_needs_no_result_at_the_end_of_them():
    """What ends the moves says nothing about whether they are a game.

    PGN ends a game with its result, but moves pasted out of somewhere else routinely
    arrive without one, and they are as much the game the file holds either way.
    """
    file = read("1. e4 e5 2. Nf3 Nc6\n")

    assert [move.san for move in file.selected.moves] == ["e4", "e5", "Nf3", "Nc6"]


def test_a_file_of_moves_that_stop_being_playable_is_refused_as_the_game_it_is():
    """Moves alone are the game such a file holds, so one that will not play names it."""
    with pytest.raises(MalformedPgnError, match="^game 1 cannot be read: illegal san: 'Qh8'"):
        read("1. e4 e5 2. Qh8 *\n")


def test_a_file_whose_first_move_will_not_play_is_described_rather_than_named():
    """The seam, and it falls where nothing was ever readable as a game.

    A record the parser got no move at all out of is not the moves a file of moves
    holds — there is nothing there to be them — so what can be said is said about the
    file. The message still carries what stopped the parser, which is what is wanted.
    """
    with pytest.raises(MalformedPgnError) as refused:
        read("1. Qh8 *\n")

    assert str(refused.value).startswith("no game was found in that file: illegal san: 'Qh8'")
    assert "game 1" not in str(refused.value)


def test_a_note_after_a_game_written_without_tags_is_not_a_game_of_its_own():
    """The moves in a note are legal as often as not, so they cannot be what says it."""
    file = read("1. e4 e5 2. Nf3 Nc6 *\n\nNote: e4 was better.\n")

    assert [(game.index, game.plies) for game in file.games] == [(0, 4)]


def test_a_game_written_without_tags_among_games_that_have_them_is_left_out():
    """The price of reading the tag pairs and nothing else, written down where it is paid.

    Such a game is the same record as a note that parses cleanly — python-chess fills
    in the seven tags and the result for both — so it cannot be kept without keeping
    every note beside it, which is the worse of the two: a note listed as a game is one
    somebody picks by number and gets prose for.
    """
    appended = f"{SCHOLARS_MATE}\n1. d4 d5 2. c4 e6 *\n"

    file = read(appended)

    assert [(game.index, game.white) for game in file.games] == [(0, "Alice")]


def test_a_file_that_is_not_pgn_at_all_is_not_given_a_game_to_name():
    """Someone who opened the wrong file is told that, not about its game 1."""
    notes = "Chess notes\n\nIn that line Qh4 is the move, and Nf6 loses.\n"

    with pytest.raises(MalformedPgnError) as refused:
        read(notes)

    assert str(refused.value).startswith("no game was found in that file")
    assert "game 1" not in str(refused.value)


def test_a_note_before_the_first_game_does_not_refuse_the_file():
    with_a_heading = f"Games from the club night. 1... Qh4 came up twice.\n\n{SCHOLARS_MATE}"

    assert [game.white for game in read(with_a_heading).games] == ["Alice"]


def test_a_broken_game_among_good_ones_is_still_refused():
    """What the reading of errors is for: a record that is a game, and is not readable."""
    broken = f'{SCHOLARS_MATE}\n[Event "B"]\n[White "Carol"]\n[Black "Dave"]\n\n1. e4 e5 2. Qh8 *\n'

    with pytest.raises(MalformedPgnError, match=r"game 2 \(Carol – Dave\) cannot be read"):
        read(broken)


def test_a_refusal_names_the_side_the_file_does_give():
    """One side unattributed is common in exports, and the other side still names it."""
    half_named = '[Event "B"]\n[White "Carol"]\n[Black "?"]\n\n1. e4 e5 2. Qh8 *\n'

    with pytest.raises(MalformedPgnError, match=r"game 1 \(Carol – \?\) cannot be read"):
        read(half_named)


def test_a_refusal_about_a_nameless_game_says_no_more_than_which_game_it_is():
    nameless = '[Event "Club night"]\n\n1. e4 e5 2. Qh8 *\n'

    with pytest.raises(MalformedPgnError, match="^game 1 cannot be read: illegal san"):
        read(nameless)


def test_only_one_game_of_a_file_is_held_while_it_is_read(monkeypatch):
    """A file of thousands of games must cost one game's memory, not the whole file's.

    Every game is read to be listed, but only the one being looked at is replayed, so
    there is no reason to keep the others' moves once they have been summarised.
    """
    three = f"{FOOLS_MATE}\n{SCHOLARS_MATE}\n{FROM_A_POSITION}"
    read_game = chess.pgn.read_game
    seen: list[weakref.ref] = []
    most_alive = 0

    def watched(*args, **kwargs):
        nonlocal most_alive
        # The nodes of a game point back at their parents, so a cycle collection is what
        # tells a game that has been let go from one that is still being held.
        gc.collect()
        most_alive = max(most_alive, sum(1 for game in seen if game() is not None))
        record = read_game(*args, **kwargs)
        if record is not None:
            # The record is a tuple of the game and what the file said about it; it is
            # the game that carries the nodes, and the game whose life is watched here.
            seen.append(weakref.ref(record.game))
        return record

    monkeypatch.setattr(chess.pgn, "read_game", watched)
    file = read(three, selected=1)

    assert [game.white for game in file.games] == ["Human", "Alice", "Alice"]
    # One: the game being read is still held while the next one is asked for.
    assert most_alive <= 1


def test_frontend_fixture_matches_the_file_the_server_reads_back():
    """The Vitest fixture must stay what the server actually sends: one entry per game."""
    fixture = json.loads((FRONTEND_FIXTURES / "replay-two-games.json").read_text(encoding="utf-8"))

    assert fixture == [read(TWO_GAMES, selected=n).model_dump(mode="json") for n in (0, 1)]
