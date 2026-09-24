"""Building datasets out of the fixture PGN files: what is kept, what is not, and the split."""

from datetime import UTC, datetime

import chess
import pytest
from dataset_helpers import FIXTURES, GOOD_GAMES, build, fixture, move_sequences

from chess_ai.dataset import (
    SPLITS,
    TRAIN,
    VALIDATION,
    DatasetError,
    GameFlags,
    PositionFlags,
    Progress,
    RatingSource,
    Result,
    SkipReason,
    TimeControl,
    builder,
    load_manifest,
    open_dataset,
    unpack_board,
)
from chess_ai.dataset.builder import DEFAULT_VALIDATION_FRACTION
from chess_ai.dataset.games import split_of, time_control_class
from chess_ai.dataset.store import dataset_path
from chess_ai.move_codec import VOCABULARY_SIZE, move_at


def test_a_build_keeps_the_games_it_can_read(tmp_path):
    manifest = build(tmp_path)

    assert manifest.games == GOOD_GAMES
    assert manifest.positions == sum(counts.positions for counts in manifest.splits.values())
    assert manifest.positions > 100


def test_broken_games_are_skipped_and_counted_by_reason(tmp_path):
    manifest = build(tmp_path, "malformed.pgn")

    assert manifest.games == 0
    assert manifest.positions == 0
    assert manifest.skipped == {
        SkipReason.NO_MOVES.value: 1,
        SkipReason.NO_RESULT.value: 2,
        SkipReason.ILLEGAL_MOVE.value: 1,
        SkipReason.BAD_POSITION.value: 1,
        SkipReason.UNSUPPORTED_VARIANT.value: 1,
    }
    assert manifest.games_skipped == 6


def test_a_file_of_nothing_but_broken_games_does_not_end_the_build(tmp_path):
    manifest = build(tmp_path, "malformed.pgn", "lichess.pgn")

    assert manifest.games == 4
    assert [(source.games_read, source.games_kept) for source in manifest.sources] == [
        (6, 0),
        (4, 4),
    ]


def test_a_file_that_is_not_pgn_at_all_is_read_without_crashing(tmp_path):
    junk = tmp_path / "junk.pgn"
    junk.write_bytes(bytes(range(256)) * 50)

    manifest = build(tmp_path / "data", str(junk))

    assert manifest.games == 0
    assert manifest.games_skipped > 0


def test_games_with_missing_ratings_are_kept_and_flagged(tmp_path):
    build(tmp_path, "unrated.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)

    unrated = [
        split.game(index)
        for index in range(split.games)
        if int(split.game(index)["flags"]) & GameFlags.WHITE_RATING_UNKNOWN
    ]

    # The club-night game gives no ratings at all; the correspondence game gives black's only.
    assert len(unrated) == 2
    assert {(int(game["white_rating"]), int(game["black_rating"])) for game in unrated} == {
        (0, 0),
        (0, 2100),
    }
    both_unknown = [game for game in unrated if int(game["flags"]) & GameFlags.BLACK_RATING_UNKNOWN]
    assert len(both_unknown) == 1


def test_a_position_says_whether_its_own_players_were_rated(tmp_path):
    build(tmp_path, "unrated.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)
    half_rated = _game_with(split, lambda game: int(game["black_rating"]) == 2100)

    positions = split.game_positions(half_rated)

    for position in positions:
        white_to_move = bool(position["flags"] & PositionFlags.WHITE_TO_MOVE)
        mover_unknown = bool(position["flags"] & PositionFlags.MOVER_RATING_UNKNOWN)
        # White's rating is the missing one, so it is the mover's on white's turns.
        assert mover_unknown == white_to_move
        assert bool(position["flags"] & PositionFlags.OPPONENT_RATING_UNKNOWN) != white_to_move


def test_the_rating_pool_is_read_from_each_game(tmp_path):
    manifest = build(tmp_path)

    assert manifest.rating_source == "auto"
    assert manifest.statistics.rating_sources == {
        RatingSource.UNKNOWN.name.lower(): 2,
        RatingSource.LICHESS.name.lower(): 5,
        RatingSource.CHESSCOM.name.lower(): 1,
    }


def test_the_rating_pool_can_be_given_for_every_game(tmp_path):
    manifest = build(tmp_path, rating_source=RatingSource.FIDE)

    assert manifest.rating_source == "fide"
    assert manifest.statistics.rating_sources == {"fide": GOOD_GAMES}


def test_the_split_puts_a_whole_game_on_one_side(tmp_path):
    build(tmp_path, validation_fraction=0.5)
    dataset = open_dataset("test", data_dir=tmp_path)

    train = move_sequences(dataset[TRAIN])
    validation = move_sequences(dataset[VALIDATION])

    assert set(train).isdisjoint(validation)
    assert len(train) + len(validation) == GOOD_GAMES
    assert train and validation, "a half-and-half split of eight games should use both sides"


def test_every_position_belongs_to_a_game_of_its_own_split(tmp_path):
    build(tmp_path, validation_fraction=0.5)
    dataset = open_dataset("test", data_dir=tmp_path)

    for name in SPLITS:
        split = dataset[name]
        seen = 0
        for index in range(split.games):
            game = split.game(index)
            for ply, position in enumerate(split.game_positions(index)):
                assert int(position["game"]) == index
                assert int(position["ply"]) == ply
                assert int(position["result"]) == (
                    int(game["result"])
                    if position["flags"] & PositionFlags.WHITE_TO_MOVE
                    else Result(int(game["result"])).opponent
                )
                seen += 1
        assert seen == len(split), "the split's positions are exactly its games' positions"


def test_the_split_is_the_same_whichever_dataset_a_game_is_built_into(tmp_path):
    # The hash is of the game, not of the build, so a game validated against in one dataset
    # cannot be trained on in another.
    everything = build(tmp_path / "all", validation_fraction=0.5)
    lichess_only = build(tmp_path / "some", "lichess.pgn", validation_fraction=0.5)

    held_back = set(move_sequences(open_dataset("test", data_dir=tmp_path / "all")[VALIDATION]))
    also_held_back = set(
        move_sequences(open_dataset("test", data_dir=tmp_path / "some")[VALIDATION])
    )

    assert everything.games == GOOD_GAMES and lichess_only.games == 4
    assert also_held_back <= held_back


def test_the_validation_fraction_decides_how_much_is_held_back():
    identities = [f"game {number}" for number in range(2000)]

    held_back = sum(split_of(identity, 0.25) for identity in identities)

    assert 0.2 < held_back / len(identities) < 0.3
    assert not any(split_of(identity, 0.0) for identity in identities)
    assert all(split_of(identity, 1.0) for identity in identities)


def test_nothing_is_held_back_when_nothing_should_be(tmp_path):
    manifest = build(tmp_path, validation_fraction=0.0)

    assert manifest.splits[VALIDATION].games == 0
    assert manifest.splits[TRAIN].games == GOOD_GAMES


def test_a_fraction_that_is_not_one_is_refused(tmp_path):
    with pytest.raises(DatasetError, match="between 0 and 1"):
        build(tmp_path, validation_fraction=1.5)


def test_the_moves_recorded_are_the_moves_played(tmp_path):
    build(tmp_path)
    dataset = open_dataset("test", data_dir=tmp_path)

    replayed = 0
    for name in SPLITS:
        split = dataset[name]
        for index in range(split.games):
            positions = split.game_positions(index)
            moves = split.move_sequence(index)
            # Replaying a game's own moves from its first position has to walk through the
            # rest of its positions exactly, which is the whole dataset checked against itself.
            board = unpack_board(positions[0])
            for ply, position in enumerate(positions):
                assert board.fen() == unpack_board(position).fen()
                assert moves[ply] == position["move"]
                move = move_at(int(position["move"]))
                assert board.is_legal(move), f"{move} is not legal in {board.fen()}"
                board.push(move)
                replayed += 1
    assert replayed == len(dataset[TRAIN]) + len(dataset[VALIDATION])


def test_a_game_that_started_somewhere_unusual_keeps_where_it_started(tmp_path):
    build(tmp_path, "custom-start.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)
    game = split.game(0)

    assert int(game["flags"]) & GameFlags.CUSTOM_START
    assert split.board(int(game["ply_offset"])).fen() == "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert not int(split.game(0)["flags"]) & GameFlags.WHITE_RATING_UNKNOWN


def test_promotions_are_recorded_as_promotions(tmp_path):
    build(tmp_path, "custom-start.pgn", validation_fraction=0.0)
    split = _all_games(tmp_path)

    promotions = [
        move_at(int(position["move"]))
        for position in split.game_positions(0)
        if move_at(int(position["move"])).promotion is not None
    ]

    assert [move.uci() for move in promotions] == ["e7e8q"]


def test_time_controls_are_sorted_into_classes(tmp_path):
    manifest = build(tmp_path)

    assert manifest.statistics.time_controls == {
        TimeControl.UNKNOWN.name.lower(): 1,
        TimeControl.BULLET.name.lower(): 1,
        TimeControl.BLITZ.name.lower(): 2,
        TimeControl.RAPID.name.lower(): 2,
        TimeControl.CLASSICAL.name.lower(): 1,
        TimeControl.CORRESPONDENCE.name.lower(): 1,
    }


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, TimeControl.UNKNOWN),
        ("", TimeControl.UNKNOWN),
        ("?", TimeControl.UNKNOWN),
        ("-", TimeControl.CORRESPONDENCE),
        ("1/86400", TimeControl.CORRESPONDENCE),
        ("15", TimeControl.BULLET),
        ("60+0", TimeControl.BULLET),
        ("120+1", TimeControl.BULLET),
        ("180+0", TimeControl.BLITZ),
        ("300+0", TimeControl.BLITZ),
        ("180+3", TimeControl.BLITZ),
        ("600+0", TimeControl.RAPID),
        ("600+5", TimeControl.RAPID),
        ("1800+0", TimeControl.CLASSICAL),
        ("40/7200+30", TimeControl.CLASSICAL),
        ("40/9000:1800+30", TimeControl.CLASSICAL),
        ("*180", TimeControl.UNKNOWN),
        ("nonsense", TimeControl.UNKNOWN),
    ],
)
def test_a_time_control_header_names_a_class(header, expected):
    assert time_control_class(header) == expected


def test_results_are_counted_from_whites_side(tmp_path):
    manifest = build(tmp_path)

    assert manifest.statistics.results == {"1-0": 4, "0-1": 2, "1/2-1/2": 2}
    assert sum(manifest.statistics.results.values()) == GOOD_GAMES


def test_ratings_are_counted_per_player_in_buckets(tmp_path):
    manifest = build(tmp_path)
    statistics = manifest.statistics

    assert sum(statistics.ratings.values()) + statistics.ratings_unknown == GOOD_GAMES * 2
    assert statistics.ratings_unknown == 3
    assert statistics.ratings["2400"] == 2
    assert [int(bucket) for bucket in statistics.ratings] == sorted(
        int(bucket) for bucket in statistics.ratings
    )


def test_the_manifest_records_what_the_dataset_was_built_from(tmp_path):
    when = datetime(2024, 5, 17, 9, 30, tzinfo=UTC)

    manifest = build(tmp_path, "lichess.pgn", "unrated.pgn", now=when)

    assert manifest.name == "test"
    assert manifest.created == when
    assert manifest.format_version == 1
    assert manifest.move_vocabulary_size == VOCABULARY_SIZE
    assert manifest.validation_fraction == DEFAULT_VALIDATION_FRACTION
    assert manifest.filters.model_dump() == {}
    assert [source.path for source in manifest.sources] == [
        fixture("lichess.pgn"),
        fixture("unrated.pgn"),
    ]
    assert [source.bytes for source in manifest.sources] == [
        (FIXTURES / "lichess.pgn").stat().st_size,
        (FIXTURES / "unrated.pgn").stat().st_size,
    ]
    assert load_manifest(dataset_path(tmp_path, "test")) == manifest


def test_a_directory_of_pgn_files_is_read_in_name_order(tmp_path):
    manifest = build(tmp_path)

    assert [source.path for source in manifest.sources] == [
        fixture(name)
        for name in ("custom-start.pgn", "lichess.pgn", "malformed.pgn", "unrated.pgn")
    ]


def test_a_glob_names_the_files_it_matches(tmp_path):
    manifest = build(tmp_path, str(FIXTURES / "*rated*.pgn"))

    assert [source.path for source in manifest.sources] == [fixture("unrated.pgn")]
    assert manifest.games == 3


def test_a_file_named_twice_is_read_once(tmp_path):
    manifest = build(tmp_path, "lichess.pgn", "lichess.pgn")

    assert len(manifest.sources) == 1
    assert manifest.games == 4


def test_a_source_that_matches_nothing_is_an_error(tmp_path):
    with pytest.raises(DatasetError, match="no PGN files match"):
        build(tmp_path, str(tmp_path / "*.pgn"))
    with pytest.raises(DatasetError, match="no such PGN file"):
        build(tmp_path, str(tmp_path / "nowhere.pgn"))


def test_a_name_that_cannot_be_a_directory_is_an_error(tmp_path):
    from chess_ai.dataset import build_dataset

    for name in ("../escape", "with/slash", ".hidden", ""):
        with pytest.raises(DatasetError, match="invalid dataset name"):
            build_dataset(name, [str(FIXTURES)], data_dir=tmp_path)


def test_building_over_a_dataset_needs_saying_so(tmp_path):
    build(tmp_path, "lichess.pgn")

    with pytest.raises(DatasetError, match="already in"):
        build(tmp_path, "unrated.pgn")

    replaced = build(tmp_path, "unrated.pgn", overwrite=True)

    assert replaced.games == 3
    assert [source.path for source in replaced.sources] == [fixture("unrated.pgn")]


def test_the_build_reports_progress_as_it_goes(tmp_path):
    reports: list[Progress] = []

    manifest = build(tmp_path, progress=reports.append)

    assert reports, "a build should say what it is doing"
    final = reports[-1]
    assert final.done
    assert final.games_kept == manifest.games == GOOD_GAMES
    assert final.positions == manifest.positions
    assert final.games_read == manifest.games + manifest.games_skipped
    assert final.bytes_read == final.bytes_total == sum(s.bytes for s in manifest.sources)
    assert final.fraction == 1.0
    assert final.seconds_remaining is None
    assert final.games_per_second > 0


def test_a_failure_nothing_expected_is_counted_and_said_out_loud(tmp_path, caplog):
    # Counting a game as unreadable is right for a broken game and would hide a broken build,
    # so the first unexpected failure has to be visible rather than only counted.
    def explode(*args, **kwargs):
        raise RuntimeError("something nothing expected")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builder, "game_records", explode)
        manifest = build(tmp_path, "lichess.pgn")

    assert manifest.games == 0
    assert manifest.skipped == {SkipReason.UNREADABLE.value: 4}
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "said once, not once per game"
    assert "something nothing expected" in caplog.text


def test_a_dataset_of_no_games_is_still_a_dataset(tmp_path):
    empty = tmp_path / "empty.pgn"
    empty.write_text("")

    manifest = build(tmp_path / "data", str(empty))

    assert manifest.games == 0
    dataset = open_dataset("test", data_dir=tmp_path / "data")
    assert len(dataset[TRAIN]) == 0
    assert dataset[TRAIN].games == 0


def _all_games(data_dir):
    """The training split of a build that held nothing back, so it has every kept game."""
    dataset = open_dataset("test", data_dir=data_dir)
    assert dataset.manifest.validation_fraction == 0.0, "this helper needs a build of all games"
    return dataset[TRAIN]


def _game_with(split, matches):
    """The index of the one game in ``split`` that ``matches``."""
    found = [index for index in range(split.games) if matches(split.game(index))]
    assert len(found) == 1, f"expected one matching game, found {len(found)}"
    return found[0]


def test_a_board_read_back_out_of_a_dataset_plays(tmp_path):
    build(tmp_path)
    dataset = open_dataset("test", data_dir=tmp_path)
    split = dataset[TRAIN]

    board = split.board(0)

    assert isinstance(board, chess.Board)
    assert board.is_valid()
