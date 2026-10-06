import asyncio

import pytest
from game_helpers import BrokenPlayer, ScriptedPlayer, scripted_players, settled

from chess_ai.game_session import ActionRejectedError, GameSession, GameState
from chess_ai.players import HumanPlayer, MoveRejectedError, RandomPlayer
from chess_ai.web.games import (
    GameRegistry,
    GamesClosedError,
    NoSuchGameError,
    TooManyGamesError,
)

pytestmark = pytest.mark.anyio

FOOLS_MATE = "f2f3 e7e5 g2g4 d8h4"


def waiting_game(**sides) -> GameSession:
    """A person playing White against a random mover that takes its time."""
    players = {"white": HumanPlayer(), "black": RandomPlayer(1), **sides}
    return GameSession(players["white"], players["black"], move_delay=10)


async def ended(session: GameSession) -> None:
    async with asyncio.timeout(5):
        while session.position.game_over is None:
            await asyncio.sleep(0)


async def test_every_game_gets_a_watch_link_and_a_play_link_per_person():
    games = GameRegistry()

    one_person = games.start(waiting_game())
    two_people = games.start(waiting_game(black=HumanPlayer()))

    assert one_person.white is not None and one_person.black is None
    assert one_person.control is None
    assert two_people.white is not None and two_people.black is not None
    assert two_people.control is None
    await games.close()


async def test_a_game_nobody_plays_by_hand_gets_a_control_link_instead():
    games = GameRegistry()

    links = games.start(GameSession(RandomPlayer(1), RandomPlayer(2), move_delay=10))

    assert (links.white, links.black) == (None, None)
    assert links.control is not None
    assert games.seat(links.control).access == "control"
    await games.close()


async def test_every_link_is_its_own_long_random_id():
    games = GameRegistry()
    links = [games.start(waiting_game(black=HumanPlayer())) for _ in range(3)]

    every = [link for game in links for link in (game.watch, game.white, game.black)]

    assert len(set(every)) == len(every)
    assert all(link is not None and len(link) >= 22 for link in every)
    await games.close()


async def test_a_link_reaches_its_game_with_its_own_rights():
    games = GameRegistry()
    session = waiting_game(black=HumanPlayer())
    links = games.start(session)

    assert links.white is not None and links.black is not None
    assert games.seat(links.white).access == "white"
    assert games.seat(links.black).access == "black"
    assert games.seat(links.watch).access == "watch"
    assert games.seat(links.watch).state.id == session.id
    assert games.seat(links.white).watch == links.watch
    await games.close()


async def test_a_link_nobody_was_given_reaches_no_game():
    games = GameRegistry()
    games.start(waiting_game())

    with pytest.raises(NoSuchGameError, match="no such game"):
        games.seat("not-a-link")
    await games.close()


async def test_two_games_are_played_at_once_and_neither_touches_the_other():
    games = GameRegistry()
    first, second = waiting_game(), waiting_game()
    first_links, second_links = games.start(first), games.start(second)
    await settled()

    assert first_links.white is not None and second_links.white is not None
    games.seat(first_links.white).submit_move("e2e4")
    games.seat(second_links.white).submit_move("d2d4")
    await settled()
    games.seat(second_links.white).resign()

    assert [move.san for move in first.state.moves] == ["e4"]
    assert first.position.game_over is None
    assert [move.san for move in second.state.moves] == ["d4"]
    assert second.position.game_over is not None
    await games.close()


@pytest.mark.parametrize("action", ["move", "resign", "takeback", "abort", "answer"])
async def test_a_watch_link_can_do_nothing_to_the_game(action):
    games = GameRegistry()
    session = waiting_game()
    watcher = games.seat(games.start(session).watch)
    await settled()

    with pytest.raises(ActionRejectedError, match="you are watching this game"):
        act(watcher, action)

    assert session.state.moves == []
    assert session.position.game_over is None
    await games.close()


def act(seat, action: str) -> None:
    match action:
        case "move":
            seat.submit_move("e2e4")
        case "resign":
            seat.resign()
        case "takeback":
            seat.take_back()
        case "abort":
            seat.abort()
        case "answer":
            seat.answer(1, accept=True)


async def test_each_person_moves_only_their_own_side():
    games = GameRegistry()
    session = waiting_game(black=HumanPlayer())
    links = games.start(session)
    assert links.white is not None and links.black is not None
    white, black = games.seat(links.white), games.seat(links.black)
    await settled()

    with pytest.raises(ActionRejectedError, match="not your move"):
        black.submit_move("e2e4")
    white.submit_move("e2e4")
    await settled()
    with pytest.raises(ActionRejectedError, match="not your move"):
        white.submit_move("d2d4")
    black.submit_move("e7e5")
    await settled()

    assert [move.san for move in session.state.moves] == ["e4", "e5"]
    await games.close()


async def test_each_person_resigns_only_their_own_side():
    games = GameRegistry()
    session = waiting_game(black=HumanPlayer())
    links = games.start(session)
    assert links.black is not None

    games.seat(links.black).resign()

    assert session.position.game_over is not None
    assert session.position.game_over.result == "1-0"
    await games.close()


async def test_the_control_link_aborts_and_moves_nothing():
    games = GameRegistry()
    session = GameSession(RandomPlayer(1), RandomPlayer(2), move_delay=10)
    links = games.start(session)
    assert links.control is not None
    control = games.seat(links.control)

    with pytest.raises(ActionRejectedError, match="nobody plays this game by hand"):
        control.submit_move("e2e4")
    control.abort()

    assert session.position.game_over is not None
    with pytest.raises(NoSuchGameError):
        games.seat(links.watch)
    await games.close()


async def test_an_aborted_game_is_gone_and_its_links_with_it():
    games = GameRegistry()
    links = games.start(waiting_game())
    assert links.white is not None
    await settled()

    games.seat(links.white).abort()

    for link in (links.white, links.watch):
        with pytest.raises(NoSuchGameError):
            games.seat(link)
    await games.close()


async def test_a_game_two_people_have_both_moved_in_is_aborted_only_by_both():
    games = GameRegistry()
    session = waiting_game(black=HumanPlayer())
    links = games.start(session)
    assert links.white is not None and links.black is not None
    white, black = games.seat(links.white), games.seat(links.black)
    await settled()
    white.submit_move("e2e4")
    await settled()
    black.submit_move("e7e5")
    await settled()

    white.abort()
    asked = session.state.request
    assert games.seat(links.watch).state.position.game_over is None
    assert asked is not None
    black.answer(asked.id, accept=True)

    assert asked is not None and (asked.kind, asked.by) == ("abort", "white")
    with pytest.raises(NoSuchGameError):
        games.seat(links.watch)
    await games.close()


async def test_a_game_that_has_ended_stays_to_be_looked_at():
    games = GameRegistry()
    session = GameSession(*scripted_players(FOOLS_MATE))
    links = games.start(session)

    await ended(session)

    seat = games.seat(links.watch)
    assert seat.state.position.game_over is not None
    assert seat.state.position.game_over.reason == "checkmate"
    await games.close()


async def test_nothing_more_can_be_done_in_a_game_that_has_ended():
    games = GameRegistry()
    links = games.start(waiting_game())
    assert links.white is not None
    player = games.seat(links.white)
    player.resign()

    with pytest.raises(MoveRejectedError, match="the game is over"):
        player.submit_move("e2e4")
    with pytest.raises(ActionRejectedError, match="the game is over"):
        player.take_back()
    await games.close()


async def test_a_new_game_beyond_the_limit_is_refused():
    games = GameRegistry(max_ongoing=2)
    games.start(waiting_game())
    games.start(waiting_game())

    with pytest.raises(TooManyGamesError, match="2 games are already being played"):
        games.start(waiting_game())
    await games.close()


async def test_a_game_that_has_ended_makes_room_for_another():
    games = GameRegistry(max_ongoing=1)
    links = games.start(waiting_game())
    assert links.white is not None
    games.seat(links.white).resign()

    games.start(waiting_game())

    assert games.ongoing == 1
    await games.close()


async def test_an_aborted_game_makes_room_for_another():
    games = GameRegistry(max_ongoing=1)
    links = games.start(waiting_game())
    assert links.white is not None
    games.seat(links.white).abort()

    games.start(waiting_game())

    assert games.ongoing == 1
    await games.close()


async def test_a_closed_registry_starts_no_games():
    games = GameRegistry()
    await games.close()

    with pytest.raises(GamesClosedError):
        games.start(waiting_game())


async def test_a_viewer_follows_the_game_until_it_ends():
    games = GameRegistry()
    session = GameSession(*scripted_players(FOOLS_MATE))
    links = games.start(session)

    with games.seat(links.watch).subscribe() as events:
        async with asyncio.timeout(5):
            seen = [event async for event in events]

    assert seen[0].type == "state"
    assert [event.type for event in seen[1:]] == ["move"] * 4
    await games.close()


async def test_a_game_played_to_a_result_is_kept():
    kept: list[GameState] = []
    games = GameRegistry(on_finished=kept.append)
    session = GameSession(*scripted_players(FOOLS_MATE))

    games.start(session)
    await ended(session)
    await games.close()

    assert kept == [session.state]


async def test_a_resigned_game_is_kept():
    kept: list[GameState] = []
    games = GameRegistry(on_finished=kept.append)
    session = waiting_game()
    links = games.start(session)
    assert links.white is not None
    await settled()

    games.seat(links.white).resign()
    await games.close()

    assert len(kept) == 1
    assert kept[0].position.game_over is not None
    assert kept[0].position.game_over.result == "0-1"


async def test_an_aborted_game_is_not_kept():
    """A game given up reached no result, so it is not a game that was played."""
    kept: list[GameState] = []
    games = GameRegistry(on_finished=kept.append)
    links = games.start(waiting_game())
    assert links.white is not None
    await settled()

    games.seat(links.white).abort()
    await games.close()

    assert kept == []


async def test_a_game_that_cannot_be_kept_is_reported_and_the_others_still_play(caplog):
    def refuse(game: GameState) -> None:
        raise OSError("the disk is full")

    games = GameRegistry(on_finished=refuse)
    first = GameSession(*scripted_players(FOOLS_MATE))
    second = GameSession(*scripted_players(FOOLS_MATE))
    games.start(first)
    games.start(second)

    await ended(first)
    await ended(second)
    await games.close()

    assert "Could not keep the finished game" in caplog.text
    assert "the disk is full" in caplog.text


class HeldPlayer(ScriptedPlayer):
    """A scripted player holding something to let go of, as an engine process is held.

    Out of moves to play, it waits to be stopped, as a player thinking for a long time would.
    """

    def __init__(self, moves: str = "") -> None:
        super().__init__(moves.split())
        self._left = len(moves.split())
        self.closed = False

    async def choose_move(self, context):
        if self._left == 0:
            await asyncio.Event().wait()
        self._left -= 1
        return await super().choose_move(context)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


async def test_the_players_of_a_game_played_to_its_end_are_let_go_of():
    white, black = HeldPlayer("f2f3 g2g4"), HeldPlayer("e7e5 d8h4")
    games = GameRegistry()
    session = GameSession(white, black)

    games.start(session)
    await ended(session)
    await settled()

    assert white.closed and black.closed
    await games.close()


async def test_the_players_of_an_aborted_game_are_let_go_of():
    held = HeldPlayer()
    games = GameRegistry()
    links = games.start(GameSession(HumanPlayer(), held))
    assert links.white is not None
    await settled()

    games.seat(links.white).submit_move("e2e4")
    await settled()
    games.seat(links.white).abort()
    await settled()

    assert held.closed
    await games.close()


async def test_the_players_of_every_game_are_let_go_of_when_the_server_goes_down():
    first, second = HeldPlayer(), HeldPlayer()
    games = GameRegistry()
    games.start(GameSession(first, HumanPlayer()))
    games.start(GameSession(second, HumanPlayer()))
    await settled()

    await games.close()

    assert first.closed and second.closed


async def test_a_game_whose_player_fails_ends_and_makes_room_for_another():
    """A player can fail mid-game, such as a checkpoint that cannot be loaded again."""
    broken = BrokenPlayer()
    games = GameRegistry(max_ongoing=1)
    session = GameSession(HumanPlayer(), broken)
    links = games.start(session)
    assert links.white is not None
    await settled()
    games.seat(links.white).submit_move("e2e4")
    await settled()

    broken.fail()
    await ended(session)
    await settled()

    assert games.ongoing == 0
    games.start(waiting_game())
    # Still there to look at, as any game that has ended is.
    stopped = games.seat(links.watch).state.position.game_over
    assert stopped is not None and stopped.reason == "error"
    await games.close()
