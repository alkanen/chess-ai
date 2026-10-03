import hashlib
import textwrap

import chess
import pytest

from chess_ai.openings import (
    DEFAULT_OPENING_SET,
    OpeningSetError,
    bundled_opening_sets,
    load_opening_set,
)

STANDARD_V1 = "3fdb697699fa52ac8d34c1d0f7a2ecf73c6fcdf09b71c2a7216222dc413e7064"
"""The lines of ``standard`` v1, pinned. A change to the set that keeps its version would make
two matches that say they started from the same positions not have done so."""


def digest(openings) -> str:
    text = "\n".join(
        f"{opening.name}: {' '.join(move.uci() for move in opening.moves)}"
        for opening in openings.openings
    )
    return hashlib.sha256(text.encode()).hexdigest()


def write_set(path, lines: str, *, name: str = "mine", version: int = 1):
    path.write_text(f'name = "{name}"\nversion = {version}\n\n' + textwrap.dedent(lines))
    return path


def test_the_standard_set_comes_with_the_project_and_is_the_default():
    assert DEFAULT_OPENING_SET == "standard"
    assert "standard" in bundled_opening_sets()

    openings = load_opening_set("standard")

    assert openings.label == "standard v1"
    assert len(openings.openings) == 100


def test_the_standard_set_changes_only_with_its_version():
    openings = load_opening_set("standard")

    assert (openings.version, digest(openings)) == (1, STANDARD_V1), (
        "the standard set has changed: raise its version, and pin the new one here"
    )


def test_every_line_of_the_standard_set_is_its_own_position():
    openings = load_opening_set("standard").openings

    positions = {opening.board().epd() for opening in openings}

    assert len(positions) == len(openings)
    assert all(len(opening.moves) >= 8 for opening in openings), "a few moves deep, at least"


def test_a_set_of_ones_own_is_read_from_its_file(tmp_path):
    path = write_set(
        tmp_path / "mine.toml",
        """\
        [[openings]]
        name = "King's Pawn"
        moves = "e4 e5"

        [[openings]]
        name = "Queen's Pawn"
        moves = "d4 d5"
        """,
        version=3,
    )

    openings = load_opening_set(path)

    assert openings.label == "mine v3"
    assert [opening.name for opening in openings.openings] == ["King's Pawn", "Queen's Pawn"]
    assert openings.openings[0].moves == (chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5"))


def test_a_line_with_a_move_that_cannot_be_played_is_refused_by_name(tmp_path):
    path = write_set(
        tmp_path / "bad.toml",
        """\
        [[openings]]
        name = "Mistyped"
        moves = "e4 e5 Nf3 Nf6 Bb5 Ke7 Qh8"
        """,
    )

    with pytest.raises(OpeningSetError, match=r"'Mistyped' cannot play Qh8 after 1\. e4 e5"):
        load_opening_set(path)


def test_two_lines_that_reach_one_position_are_refused(tmp_path):
    path = write_set(
        tmp_path / "twice.toml",
        """\
        [[openings]]
        name = "One order"
        moves = "e4 e5 Nf3 Nc6"

        [[openings]]
        name = "The other"
        moves = "Nf3 Nc6 e4 e5"
        """,
    )

    with pytest.raises(
        OpeningSetError, match="'The other' reaches the same position as 'One order'"
    ):
        load_opening_set(path)


def test_a_set_that_is_not_there_says_which_sets_are(tmp_path):
    with pytest.raises(OpeningSetError, match=r"no opening set 'nonesuch'.*\(standard\)"):
        load_opening_set("nonesuch")


@pytest.mark.parametrize(
    "text",
    [
        "not toml at all [",
        'name = "empty"\nversion = 1\nopenings = []\n',
        'name = "unversioned"\n[[openings]]\nname = "x"\nmoves = "e4"\n',
        'name = "x"\nversion = 1\n[[openings]]\nname = "x"\nmoves = "e4"\neco = "B00"\n',
    ],
)
def test_a_file_that_is_not_an_opening_set_is_refused(tmp_path, text):
    path = tmp_path / "set.toml"
    path.write_text(text)

    with pytest.raises(OpeningSetError, match="set.toml"):
        load_opening_set(path)
