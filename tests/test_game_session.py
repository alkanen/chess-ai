import asyncio
import gc
import json
from contextlib import suppress
from itertools import pairwise
from pathlib import Path

import chess
import pytest
from game_helpers import (
    BrokenPlayer,
    PlayerBroke,
    ScriptedPlayer,
    collect,
    playing,
    replay,
    scripted_players,
    settled,
)

from chess_ai.game_session import ActionRejectedError, GameSession, IllegalMoveError
from chess_ai.players import (
    CandidateMove,
    HumanPlayer,
    MoveRejectedError,
    RandomPlayer,
    Thoughts,
    WinDrawLoss,
)
from chess_ai.position_view import InvalidFenError, snapshot

pytestmark = pytest.mark.anyio

FOOLS_MATE = "f2f3 e7e5 g2g4 d8h4"
FRONTEND_FIXTURES = Path(__file__).parents[1] / "frontend" / "src" / "test" / "fixtures"
FIXTURE_GAME_ID = "fools-mate"
"""The id the fixture game is given, since a real one is made up afresh every time."""


async def test_scripted_game_plays_to_checkmate():
    session = GameSession(*scripted_players(FOOLS_MATE))

    await session.play()

    state = session.state
    assert [move.san for move in state.moves] == ["f3", "e5", "g4", "Qh4#"]
    assert state.position.last_move is not None
    assert (state.position.last_move.from_square, state.position.last_move.to_square) == (
        "d8",
        "h4",
    )
    assert state.position.game_over is not None
    assert (state.position.game_over.result, state.position.game_over.reason) == (
        "0-1",
        "checkmate",
    )


async def test_frontend_fixture_matches_a_subscribers_events():
    """The Vitest fixture must stay what the server actually sends."""
    session = GameSession(*scripted_players(FOOLS_MATE), id=FIXTURE_GAME_ID)

    with session.subscribe() as events:
        await session.play()
        received = [event.model_dump(mode="json") for event in await collect(events)]

    fixture = (FRONTEND_FIXTURES / "fools-mate-events.json").read_text(encoding="utf-8")
    assert received == json.loads(fixture)


async def test_state_names_the_players_and_says_who_takes_submitted_moves():
    session = GameSession(RandomPlayer(), HumanPlayer())

    state = session.state
    assert state.white.model_dump() == {"name": "Random mover", "accepts_moves": False}
    assert state.black.model_dump() == {"name": "Human", "accepts_moves": True}


@pytest.mark.parametrize("seed", range(10))
async def test_random_games_end_legally_at_the_first_game_over(seed):
    session = GameSession(RandomPlayer(seed), RandomPlayer(seed + 1000))

    await session.play()

    state = session.state
    board = chess.Board()
    for record in state.moves:
        assert snapshot(board).game_over is None
        move = chess.Move.from_uci(record.uci)
        assert board.is_legal(move)
        assert board.san(move) == record.san
        board.push(move)
    assert state.position == snapshot(board)
    assert state.position.game_over is not None


async def test_seeded_random_games_are_reproducible():
    # Two runs of one game, so they are named alike; only their play is under test.
    games = [GameSession(RandomPlayer(1), RandomPlayer(2), id="seeded") for _ in range(2)]

    for game in games:
        await game.play()

    assert games[0].state == games[1].state


async def test_subscribers_receive_consistent_event_streams():
    session = GameSession(RandomPlayer(1), RandomPlayer(2))

    with session.subscribe() as early:
        game = asyncio.create_task(session.play())
        early_events = [await anext(early) for _ in range(6)]
        with session.subscribe() as middle:
            early_events += await collect(early)
            middle_events = await collect(middle)
    await game
    with session.subscribe() as late:
        late_events = await collect(late)

    final = session.state
    assert final.position.game_over is not None
    assert early_events[0].type == "state"
    assert early_events[0].game.moves == []
    assert middle_events[0].type == "state"
    assert 0 < len(middle_events[0].game.moves) < len(final.moves)
    assert len(late_events) == 1
    assert replay(early_events) == replay(middle_events) == replay(late_events) == final


async def test_players_thoughts_reach_subscribers():
    thoughts = Thoughts(
        candidates=[
            CandidateMove(uci="f2f3", probability=0.5),
            CandidateMove(uci="e2e4", probability=0.3),
        ],
        wdl=WinDrawLoss(win=0.2, draw=0.3, loss=0.5),
    )
    white = ScriptedPlayer(["f2f3", "g2g4"], thoughts=thoughts)
    black = ScriptedPlayer(["e7e5", "d8h4"])
    session = GameSession(white, black)

    with session.subscribe() as events:
        await session.play()
        received = await collect(events)

    moves = [event.move for event in received if event.type == "move"]
    assert [move.thoughts for move in moves] == [thoughts, None, thoughts, None]
    assert replay(received) == session.state


@pytest.mark.parametrize(
    "illegal",
    [
        "e2e5",  # too far
        "e7e5",  # the opponent's piece
        "e1g1",  # castling through pieces
        "0000",  # a null move
    ],
)
async def test_illegal_move_is_rejected_without_changing_state(illegal):
    white = ScriptedPlayer(["e2e4", illegal])
    black = ScriptedPlayer(["e7e5"])
    session = GameSession(white, black)

    with session.subscribe() as events:
        with pytest.raises(IllegalMoveError):
            await session.play()
        received = await collect(events)

    state = session.state
    assert [move.uci for move in state.moves] == ["e2e4", "e7e5"]
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert state.position == snapshot(board)
    assert replay(received) == state


async def test_moves_are_at_least_the_move_delay_apart():
    delay = 0.05
    session = GameSession(*scripted_players(FOOLS_MATE), move_delay=delay)
    loop = asyncio.get_running_loop()

    times = [loop.time()]
    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        async for event in events:
            if event.type == "move":
                times.append(loop.time())
    await game

    assert len(times) == 5
    gaps = [later - earlier for earlier, later in pairwise(times)]
    # Timer resolution can wake a sleeper a hair early.
    assert min(gaps) >= delay * 0.9, gaps


async def test_close_stops_the_game_and_ends_subscriptions():
    session = GameSession(RandomPlayer(1), RandomPlayer(2))

    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        received = [await anext(events) for _ in range(4)]
        session.close()
        received += await collect(events)
    await game

    state = session.state
    assert state.position.game_over is None
    assert replay(received) == state
    with session.subscribe() as later:
        later_events = await collect(later)
    assert len(later_events) == 1
    assert replay(later_events) == state


async def test_a_session_closed_before_it_starts_makes_no_moves():
    session = GameSession(RandomPlayer(1), RandomPlayer(2))

    with session.subscribe() as events:
        session.close()
        await session.play()
        received = await collect(events)

    assert session.state.moves == []
    assert [event.type for event in received] == ["state"]


async def test_a_human_plays_the_move_that_is_submitted():
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        played = [await anext(events), await anext(events)]

    assert [event.move.san for event in played] == ["e4", "e5"]
    assert [move.uci for move in session.state.moves] == ["e2e4", "e7e5"]


async def test_a_submitted_underpromotion_makes_the_piece_it_asks_for():
    """The promotion piece in the UCI is the one that lands on the board.

    The four promotions of a pawn share a destination square, so the last letter of the
    UCI the browser's promotion picker submits is all that tells them apart.
    """
    # 1. h4 g5 2. hxg5 Na6 3. g6 Nb8 4. gxh7 Na6, and the h-pawn takes on g8.
    session = GameSession(HumanPlayer(), ScriptedPlayer(["g7g5", "b8a6", "a6b8", "b8a6"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        for uci in ["h2h4", "h4g5", "g5g6", "g6h7"]:
            session.submit_move(uci)
            played, replied = await anext(events), await anext(events)
            assert (played.type, replied.type) == ("move", "move")

        session.submit_move("h7g8n")
        promotion = await anext(events)

    assert promotion.type == "move"
    assert promotion.move.uci == "h7g8n" and promotion.move.san == "hxg8=N"
    assert session.state.position.pieces["g8"].model_dump() == {
        "color": "white",
        "type": "knight",
    }


async def test_both_sides_can_be_human():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        session.submit_move("e7e5")
        assert (await anext(events)).type == "move"

    assert [move.san for move in session.state.moves] == ["e4", "e5"]


@pytest.mark.parametrize(
    ("uci", "problem"),
    [
        ("e2e5", "not a legal move"),  # too far
        ("e7e5", "not a legal move"),  # the opponent's piece
        ("e1g1", "not a legal move"),  # castling through pieces
        ("0000", "not a legal move"),  # a null move
        ("hello", "is not a move"),
        ("", "is not a move"),
    ],
)
async def test_an_illegal_submission_is_rejected_and_the_game_waits_on(uci, problem):
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"

        with pytest.raises(MoveRejectedError, match=problem):
            session.submit_move(uci)

        assert session.state.moves == []
        assert session.state.position == snapshot(chess.Board())
        session.submit_move("e2e4")
        played = [await anext(events), await anext(events)]

    assert [event.move.uci for event in played] == ["e2e4", "e7e5"]
    assert [move.uci for move in session.state.moves] == ["e2e4", "e7e5"]


async def test_a_move_cannot_be_submitted_for_a_player_that_plays_its_own():
    session = GameSession(ScriptedPlayer(["e2e4"]), HumanPlayer())

    async with playing(session):
        with pytest.raises(MoveRejectedError, match="Scripted plays this move"):
            session.submit_move("e2e4")


async def test_a_second_submission_out_of_turn_is_rejected():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")

        with pytest.raises(MoveRejectedError, match="it is not your turn"):
            session.submit_move("d2d4")

        assert (await anext(events)).type == "move"

    assert [move.uci for move in session.state.moves] == ["e2e4"]


async def test_no_move_can_be_submitted_once_the_game_is_over():
    session = GameSession(*scripted_players(FOOLS_MATE))
    await session.play()

    with pytest.raises(MoveRejectedError, match="the game is over"):
        session.submit_move("e2e4")


async def test_a_submitted_move_is_not_held_back_by_the_move_delay():
    """The delay paces players that move instantly; a person is already slow enough."""
    delay = 0.5
    session = GameSession(HumanPlayer(), HumanPlayer(), move_delay=delay)
    loop = asyncio.get_running_loop()

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        submitted = loop.time()
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        waited = loop.time() - submitted

    assert waited < delay / 2, waited


async def test_a_rejection_does_not_put_the_position_in_its_message():
    """The message reaches the person who submitted the move, so it carries no FEN."""
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session):
        with pytest.raises(MoveRejectedError) as rejected:
            session.submit_move("e2e5")

    assert str(rejected.value) == "e2e5 is not a legal move here"


async def test_the_move_delay_still_paces_the_opponent_that_moves_instantly():
    delay = 0.2
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]), move_delay=delay)
    loop = asyncio.get_running_loop()

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        submitted = loop.time()
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        human = loop.time() - submitted
        assert (await anext(events)).type == "move"
        opponent = loop.time() - submitted

    assert human < delay / 2, human
    # Timer resolution can wake a sleeper a hair early.
    assert opponent >= delay * 0.9, opponent


async def test_two_submitted_moves_are_not_kept_the_move_delay_apart():
    """The delay paces players that move instantly, not the people using the board."""
    delay = 0.5
    session = GameSession(HumanPlayer(), HumanPlayer(), move_delay=delay)
    loop = asyncio.get_running_loop()

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        started = loop.time()
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        session.submit_move("e7e5")
        assert (await anext(events)).type == "move"
        waited = loop.time() - started

    assert [move.san for move in session.state.moves] == ["e4", "e5"]
    assert waited < delay / 2, waited


async def test_a_resignation_ends_the_game_and_reaches_every_subscriber():
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.resign("white")
        ended = [event async for event in events]
        with session.subscribe() as late:
            late_events = await collect(late)

    game_over = session.state.position.game_over
    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("0-1", "resignation")
    assert [event.type for event in ended] == ["game_over"]
    assert ended[0].position == session.state.position
    # Someone who arrives after the resignation is told the same as everyone else.
    assert replay(late_events) == session.state


async def test_black_resigning_hands_the_game_to_white():
    session = GameSession(RandomPlayer(1), HumanPlayer(), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        # Black resigns while it is White's move, which is a thing people do.
        session.resign("black")

    game_over = session.state.position.game_over
    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("1-0", "resignation")


async def test_an_abort_ends_the_game_with_no_result():
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.abort()
        ended = [event async for event in events]

    game_over = session.state.position.game_over
    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("*", "abort")
    assert [event.type for event in ended] == ["game_over"]


async def test_a_game_ended_off_the_board_keeps_its_position_but_offers_no_moves():
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        played = session.state.position
        session.abort()

    stopped = session.state.position
    assert stopped.fen == played.fen
    assert stopped.pieces == played.pieces
    assert stopped.last_move == played.last_move
    assert played.legal_moves != {}
    assert stopped.legal_moves == {}


async def test_no_move_is_taken_after_a_resignation():
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.resign("white")

        with pytest.raises(MoveRejectedError, match="the game is over"):
            session.submit_move("e2e4")

    assert session.state.moves == []


@pytest.mark.parametrize("ending", ["resign", "abort"])
async def test_a_game_that_has_ended_cannot_end_again(ending):
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.abort()

        with pytest.raises(ActionRejectedError, match="the game is over"):
            session.resign("white") if ending == "resign" else session.abort()

    game_over = session.state.position.game_over
    assert game_over is not None and game_over.reason == "abort"


async def test_a_finished_game_is_neither_resigned_nor_aborted():
    session = GameSession(*scripted_players(FOOLS_MATE))

    await session.play()

    for ending in [lambda: session.resign("white"), session.abort]:
        with pytest.raises(ActionRejectedError, match="the game is over"):
            ending()
    game_over = session.state.position.game_over
    assert game_over is not None and game_over.reason == "checkmate"


async def test_a_side_that_plays_its_own_moves_is_nobody_s_to_resign():
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"

        with pytest.raises(ActionRejectedError, match="Random mover is not yours to resign"):
            session.resign("black")

    assert session.state.position.game_over is None


async def test_either_human_side_can_resign_for_itself():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.resign("black")

    game_over = session.state.position.game_over
    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("1-0", "resignation")


async def test_a_takeback_gives_the_person_their_move_back():
    """Against a player that moves for itself, its reply goes back with your move."""
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5", "b8c6"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]

        session.take_back()
        taken_back = await anext(events)
        # The move the takeback gave back can be played again, differently this time.
        await settled()
        session.submit_move("d2d4")
        played, replied = await anext(events), await anext(events)

    assert taken_back.type == "takeback"
    assert (taken_back.ply, taken_back.position) == (0, snapshot(chess.Board()))
    assert (played.type, replied.type) == ("move", "move")
    assert [move.san for move in session.state.moves] == ["d4", "Nc6"]


async def test_a_takeback_between_two_people_undoes_the_one_move_last_played():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        session.submit_move("e7e5")
        assert (await anext(events)).type == "move"

        session.take_back()
        taken_back = await anext(events)
        await settled()
        # It is Black's move again, so White's move is not theirs to play.
        with pytest.raises(MoveRejectedError, match="not a legal move"):
            session.submit_move("d2d4")
        session.submit_move("c7c5")
        assert (await anext(events)).type == "move"

    assert taken_back.type == "takeback"
    assert taken_back.ply == 1
    assert [move.san for move in session.state.moves] == ["e4", "c5"]


async def test_a_takeback_restores_castling_rights_and_the_clocks_exactly():
    """Everything a position is judged by comes back, not just where the pieces stand."""
    # 1. Nf3 Nf6 2. Rg1 Rg8, which spends White's kingside castling rights and four
    # moves of the fifty-move clock without moving a pawn or taking anything.
    session = GameSession(HumanPlayer(), ScriptedPlayer(["g8f6", "h8g8"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("g1f3")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        before = session.state.position
        assert before.fen.split()[2:] == ["KQkq", "-", "2", "2"]
        session.submit_move("h1g1")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        assert session.state.position.fen.split()[2:] == ["Qq", "-", "4", "3"]

        session.take_back()
        assert (await anext(events)).type == "takeback"

    assert session.state.position == before


async def test_a_takeback_restores_the_en_passant_square():
    # 1. e4 e6 2. e5 d5, and the pawn on e5 may take on d6 in passing.
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e6", "d7d5", "g8f6"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        await settled()
        session.submit_move("e4e5")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        with_capture = session.state.position
        assert any(move.en_passant for move in with_capture.legal_moves["e5"])

        # Declining the capture spends it, and taking the decline back offers it again.
        await settled()
        session.submit_move("g1f3")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        declined = session.state.position.legal_moves
        assert not any(move.en_passant for move in declined.get("e5", []))
        session.take_back()
        assert (await anext(events)).type == "takeback"

    assert session.state.position == with_capture


async def test_a_takeback_leaves_the_repetitions_of_the_moves_that_are_left():
    """The moves that are gone are gone from the repetition count as well."""
    # The knights shuffle back to the starting position twice over; the third time it
    # stands is a draw, which is still there to be claimed after a takeback. White is
    # asked once more the moment Black declines the draw, and the takeback throws that
    # answer away, but a player with nothing left to say fails rather than waits.
    session = GameSession(ScriptedPlayer(["g1f3", "f3g1", "g1f3", "f3g1", "d2d4"]), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        for uci in ["g8f6", "f6g8", "g8f6"]:
            assert (await anext(events)).type == "move"
            await settled()
            session.submit_move(uci)
            assert (await anext(events)).type == "move"
        assert (await anext(events)).type == "move"

        # Ng8 now would be the starting position a third time. This is not that.
        await settled()
        session.submit_move("b8c6")
        assert (await anext(events)).type == "move"
        assert session.state.position.game_over is None
        session.take_back()
        assert (await anext(events)).type == "takeback"

        # The repetitions the taken-back move was counted against are still counted.
        await settled()
        session.submit_move("f6g8")
        assert (await anext(events)).type == "move"

    game_over = session.state.position.game_over
    assert game_over is not None
    assert (game_over.result, game_over.reason) == ("1/2-1/2", "threefold_repetition")


async def test_a_takeback_reaches_every_subscriber_and_whoever_arrives_after_it():
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        with session.subscribe() as watching:
            session.take_back()
            seen = [await anext(events), await anext(watching), await anext(watching)]
        with session.subscribe() as late:
            arrived = await anext(late)

    assert [event.type for event in seen] == ["takeback", "state", "takeback"]
    assert seen[0] == seen[2]
    assert arrived.type == "state"
    assert arrived.game == session.state
    assert session.state.moves == []


async def test_a_takeback_drops_the_move_a_player_was_still_working_out():
    """A move chosen in a position that has been taken back is never played."""
    session = GameSession(HumanPlayer(), RandomPlayer(1), move_delay=0.05)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        # Black is working its reply out; taking back now must leave nothing behind.
        session.take_back()
        assert (await anext(events)).type == "takeback"
        await asyncio.sleep(0.15)

        assert session.state.moves == []
        session.submit_move("d2d4")
        assert (await anext(events)).type == "move"

    assert session.state.moves[0].san == "d4"


async def test_a_move_submitted_in_the_same_breath_as_a_takeback_is_refused():
    """The game has stopped waiting on the person, so their move is nobody's to play."""
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"
        session.take_back()

        with pytest.raises(MoveRejectedError, match="it is not your turn"):
            session.submit_move("d2d4")

        # The game comes round to asking again, and takes the move then.
        await settled()
        session.submit_move("d2d4")
        assert (await anext(events)).type == "takeback"
        assert (await anext(events)).type == "move"

    assert [move.san for move in session.state.moves] == ["d4"]


async def test_a_takeback_puts_the_person_who_is_now_on_move_back_in_play():
    session = GameSession(HumanPlayer(), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"

        session.take_back()
        assert (await anext(events)).type == "takeback"
        # White is to move again, and the game has to be waiting on White to take it.
        await settled()
        session.submit_move("d2d4")
        played = await anext(events)

    assert played.type == "move" and played.move.san == "d4"


@pytest.mark.parametrize("colors", [(HumanPlayer, RandomPlayer), (RandomPlayer, HumanPlayer)])
async def test_a_game_with_no_moves_has_nothing_to_take_back(colors):
    white, black = colors
    session = GameSession(white(), black(), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"

        with pytest.raises(ActionRejectedError, match="no move has been played yet"):
            session.take_back()


async def test_the_opening_move_of_a_player_you_only_watch_can_be_taken_back():
    """Playing Black, there is no move of your own to come back to until you have one."""
    session = GameSession(ScriptedPlayer(["e2e4", "d2d4"]), HumanPlayer())

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        assert (await anext(events)).type == "move"

        session.take_back()
        taken_back = await anext(events)
        # White is asked again, and opens differently this time.
        opened = await anext(events)

    assert taken_back.type == "takeback" and taken_back.ply == 0
    assert opened.type == "move" and opened.move.san == "d4"


async def test_a_game_nobody_plays_by_hand_has_no_takebacks():
    session = GameSession(RandomPlayer(1), RandomPlayer(2), move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        assert (await anext(events)).type == "move"

        with pytest.raises(ActionRejectedError, match="neither side is played by hand"):
            session.take_back()

    assert len(session.state.moves) == 1


@pytest.mark.parametrize("ending", ["resign", "abort"])
async def test_a_game_that_has_ended_cannot_be_taken_back(ending):
    session = GameSession(HumanPlayer(), ScriptedPlayer(["e7e5"]))

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert [(await anext(events)).type for _ in range(2)] == ["move", "move"]
        session.resign("white") if ending == "resign" else session.abort()

        with pytest.raises(ActionRejectedError, match="the game is over"):
            session.take_back()

    assert len(session.state.moves) == 2


async def test_a_game_can_start_from_a_position_given_as_a_fen():
    # Black to move in the Scandinavian, with the queen ready to take on d5.
    fen = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    session = GameSession(RandomPlayer(1), HumanPlayer(), fen=fen, move_delay=10)

    async with playing(session) as events:
        state = await anext(events)
        session.submit_move("d8d5")
        played = await anext(events)

    assert state.type == "state"
    assert state.game.start_fen == fen
    assert state.game.position.fen == fen
    assert state.game.position.turn == "black"
    assert state.game.moves == []
    assert played.type == "move" and played.move.san == "Qxd5"


async def test_a_game_from_a_fen_takes_back_to_the_position_it_started_from():
    fen = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    session = GameSession(RandomPlayer(1), HumanPlayer(), fen=fen, move_delay=10)

    async with playing(session) as events:
        assert (await anext(events)).type == "state"
        session.submit_move("d8d5")
        assert (await anext(events)).type == "move"

        session.take_back()
        taken_back = await anext(events)

    assert taken_back.type == "takeback"
    assert (taken_back.ply, taken_back.position) == (0, snapshot(chess.Board(fen)))
    assert taken_back.position.last_move is None
    assert session.state.start_fen == fen


async def test_a_game_from_a_fen_is_over_before_it_starts_if_the_position_is():
    """Nothing is played from a position the rules have already finished with."""
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    session = GameSession(HumanPlayer(), RandomPlayer(1), fen=mate)

    await session.play()

    state = session.state
    assert state.moves == []
    assert state.position.game_over is not None
    assert state.position.game_over.reason == "checkmate"


async def test_a_fen_that_cannot_be_played_from_starts_no_game():
    with pytest.raises(InvalidFenError):
        GameSession(HumanPlayer(), RandomPlayer(1), fen="not a fen at all")


async def test_a_player_that_fails_is_not_lost_to_a_takeback_landing_at_once():
    """A stale answer is thrown away; a stale failure is still a failure.

    The question a takeback leaves behind is dropped unread, so a player that failed
    rather than answered would have nobody to report it: asyncio would notice the
    unretrieved exception at garbage collection, long after the game had moved on.
    """
    broken = BrokenPlayer()
    session = GameSession(HumanPlayer(), broken)

    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        await asyncio.sleep(0)
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"

        # The player fails, and the takeback lands before the game has read the failure.
        broken.fail()
        await asyncio.sleep(0)
        session.take_back()

        with pytest.raises(PlayerBroke):
            async with asyncio.timeout(2):
                await game


async def test_a_player_that_fails_stops_the_game():
    broken = BrokenPlayer()
    session = GameSession(broken, HumanPlayer())

    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        await asyncio.sleep(0)
        assert (await anext(events)).type == "state"

        broken.fail()

        with pytest.raises(PlayerBroke):
            async with asyncio.timeout(2):
                await game

    assert session.state.moves == []


async def test_a_player_that_fails_as_the_game_is_stopped_is_still_reported():
    """Stopping the game must not mark a failure nobody has read as handled.

    A game is stopped from outside on a resignation, a new game or a shutdown. A player
    that failed in the moment before that is never read by the game at all, so asyncio's
    own report of it is the last thing standing between the failure and silence.
    """
    reported: list[BaseException | None] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: reported.append(context.get("exception")))
    broken = BrokenPlayer()
    session = GameSession(HumanPlayer(), broken)

    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        await asyncio.sleep(0)
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"

        # The player fails, and the game is stopped before it has read the failure.
        broken.fail()
        await asyncio.sleep(0)
        game.cancel()
        # Nothing may hold the traceback afterwards, or the question it names stays alive.
        with suppress(asyncio.CancelledError):
            await game

    gc.collect()
    await asyncio.sleep(0)
    assert [type(failure) for failure in reported] == [PlayerBroke]


async def test_a_takeback_does_not_silence_a_failure_the_game_never_reads():
    """Taking back leaves an unread failure for whoever reads it next, or for asyncio.

    A takeback stops the question the game was waiting on, which for a turn or two is a
    question that has already been answered or has already failed. Stopping a failed one
    must not be what marks it read, because the game may be stopped before it reads it.
    """
    reported: list[BaseException | None] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: reported.append(context.get("exception")))
    broken = BrokenPlayer()
    session = GameSession(HumanPlayer(), broken)

    with session.subscribe() as events:
        game = asyncio.create_task(session.play())
        await asyncio.sleep(0)
        assert (await anext(events)).type == "state"
        session.submit_move("e2e4")
        assert (await anext(events)).type == "move"

        # The player fails; the takeback and the stop both land before the game reads it.
        broken.fail()
        await asyncio.sleep(0)
        session.take_back()
        game.cancel()
        with suppress(asyncio.CancelledError):
            await game

    gc.collect()
    await asyncio.sleep(0)
    assert [type(failure) for failure in reported] == [PlayerBroke]
