import asyncio
from datetime import UTC, datetime, timedelta

import chess
import pytest
from game_helpers import (
    BrokenPlayer,
    GoneModelPlayer,
    ScriptedModelPlayer,
    ScriptedPlayer,
    scripted_players,
    settled,
)

from chess_ai.game_session import ActionRejectedError, GameSession, GameState
from chess_ai.players import HumanPlayer, ModelDescription, MoveRejectedError, RandomPlayer
from chess_ai.web.game_store import (
    GameStore,
    HumanRecord,
    ModelRecord,
    PlayerRecord,
    RandomRecord,
    StoredGame,
)
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


PERSON_AND_RANDOM = (HumanRecord(), RandomRecord())


def on_disk(store: GameStore) -> dict[str, StoredGame]:
    return {game.id: game for game in store.read_all()}


def revive(record: PlayerRecord):
    """A player made again from its record, as the server makes one after a restart."""
    match record:
        case HumanRecord():
            return HumanPlayer()
        case RandomRecord():
            return RandomPlayer(3)
    raise AssertionError(f"no player for {record}")


def restore_all(games: GameRegistry, store: GameStore) -> None:
    for stored in store.read_all():
        games.restore(stored, revive(stored.white), revive(stored.black))


async def test_a_game_is_written_down_when_it_starts_and_as_it_is_played(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session = waiting_game()
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None

    assert on_disk(store)[session.id].session.moves == []
    await settled()
    games.seat(links.white).submit_move("e2e4")
    await settled()

    kept = on_disk(store)[session.id]
    assert [move.uci for move in kept.session.moves] == ["e2e4"]
    assert kept.links == links
    assert (kept.white, kept.black) == PERSON_AND_RANDOM
    await games.close()
    # The server going down is not the end of a game, which is still there to carry on with.
    assert on_disk(store)[session.id].session.game_over is None


async def test_a_game_with_nothing_to_make_its_players_again_from_is_not_written_down(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)

    games.start(waiting_game())

    assert on_disk(store) == {}
    await games.close()


async def test_an_aborted_game_is_deleted_from_the_disk_as_well(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    links = games.start(waiting_game(), players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()

    games.seat(links.white).abort()

    assert on_disk(store) == {}
    assert list((tmp_path / "ongoing").iterdir()) == []
    await games.close()


async def test_a_game_that_has_ended_stays_on_the_disk_with_how_it_ended(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session = waiting_game()
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()

    games.seat(links.white).resign()
    await games.close()

    over = on_disk(store)[session.id].session.game_over
    assert over is not None and over.reason == "resignation"


async def test_games_taken_back_after_a_restart_are_reached_through_the_same_links(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    before = GameRegistry(store=store)
    session = GameSession(HumanPlayer(), RandomPlayer(1))
    links = before.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()
    before.seat(links.white).submit_move("e2e4")
    async with asyncio.timeout(5):
        while len(session.state.moves) < 2:
            await asyncio.sleep(0)
    await before.close()

    after = GameRegistry(store=store)
    restore_all(after, store)
    white = after.seat(links.white)

    assert white.access == "white"
    assert white.state.moves == session.state.moves
    assert after.seat(links.watch).access == "watch"
    assert after.ongoing == 1
    await settled()
    white.submit_move("d2d4")
    await settled()
    assert [move.uci for move in on_disk(store)[session.id].session.moves][2] == "d2d4"
    await after.close()


async def test_a_finished_game_taken_back_is_there_to_be_looked_at_and_nothing_more(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    before = GameRegistry(store=store)
    links = before.start(waiting_game(), players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()
    before.seat(links.white).resign()
    await before.close()

    after = GameRegistry(store=store)
    restore_all(after, store)
    white = after.seat(links.white)

    assert white.state.position.game_over is not None
    assert after.ongoing == 0
    with pytest.raises(ActionRejectedError, match="game is over"):
        white.resign()
    await after.close()


async def test_playing_again_still_joins_the_same_new_game_after_a_restart(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    before = GameRegistry(store=store)
    links = before.start(waiting_game(), players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()
    before.seat(links.white).resign()
    again = before.start(
        waiting_game(), players=PERSON_AND_RANDOM, follows=before.seat(links.white)
    )
    await before.close()

    after = GameRegistry(store=store)
    restore_all(after, store)
    joined = after.seat(links.white).next_game()

    assert joined is not None and joined.own_links.white == again.white
    await after.close()


async def test_a_game_nobody_has_moved_in_for_the_expiry_is_deleted_links_and_all(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store, expire_after=timedelta(days=7))
    old, recent = waiting_game(), waiting_game()
    old_links = games.start(old, players=PERSON_AND_RANDOM)
    recent_links = games.start(recent, players=PERSON_AND_RANDOM)
    old.updated -= timedelta(days=7, seconds=1)
    recent.updated -= timedelta(days=6)

    assert games.expire() == 1

    with pytest.raises(NoSuchGameError):
        games.seat(old_links.watch)
    assert games.seat(recent_links.watch).state.id == recent.id
    assert set(on_disk(store)) == {recent.id}
    assert old.position.game_over is None  # Deleted, not ended: there is nothing to keep.
    await games.close()


async def test_a_move_starts_the_count_to_expiry_again(tmp_path):
    games = GameRegistry(expire_after=timedelta(days=7))
    session = waiting_game()
    links = games.start(session)
    assert links.white is not None
    session.updated -= timedelta(days=8)
    await settled()

    games.seat(links.white).submit_move("e2e4")
    await settled()

    assert games.expire() == 0
    await games.close()


async def test_a_finished_game_expires_too(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session = waiting_game()
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()
    games.seat(links.white).resign()
    session.updated -= timedelta(days=8)

    assert games.expire() == 1
    assert on_disk(store) == {}
    await games.close()


async def test_an_expired_game_in_progress_lets_go_of_its_players_and_viewers():
    held = HeldPlayer()
    games = GameRegistry()
    session = GameSession(HumanPlayer(), held)
    links = games.start(session)
    assert links.white is not None
    await settled()
    games.seat(links.white).submit_move("e2e4")
    await settled()

    with games.seat(links.watch).subscribe() as events:
        session.updated -= timedelta(days=8)
        games.expire()
        async with asyncio.timeout(5):
            seen = [event.type async for event in events]
        await settled()

    assert seen == ["state"]
    assert held.closed
    await games.close()


async def test_a_game_that_cannot_be_written_down_is_reported_and_played_on(tmp_path, caplog):
    (tmp_path / "ongoing").write_text("a file where the directory should be")
    games = GameRegistry(store=GameStore(tmp_path / "ongoing"))
    session = waiting_game()
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()

    games.seat(links.white).submit_move("e2e4")
    await settled()

    assert "Could not write game" in caplog.text
    assert len(session.state.moves) >= 1
    await games.close()


TINY = ModelDescription(run="tiny", checkpoint=2)
OTHER = ModelDescription(run="tiny", checkpoint=4)


def model_record(step: int) -> ModelRecord:
    return ModelRecord(
        run="tiny",
        step=step,
        fingerprint=(1, 2, 3),
        rating=None,
        strategy="argmax",
        temperature=1.0,
        seed=None,
    )


async def paused_game(games: GameRegistry) -> tuple[GameSession, str]:
    session = GameSession(HumanPlayer(), GoneModelPlayer(TINY))
    links = games.start(session, players=(HumanRecord(), model_record(2)))
    assert links.white is not None
    await settled()
    games.seat(links.white).submit_move("e2e4")
    async with asyncio.timeout(5):
        while session.paused is None:
            await asyncio.sleep(0)
    return session, links.white


async def test_the_person_at_a_paused_game_hands_its_side_to_another_checkpoint(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session, white = await paused_game(games)

    games.seat(white).replace(ScriptedModelPlayer(["e7e5"], OTHER), model_record(4))
    async with asyncio.timeout(5):
        while len(session.state.moves) < 2:
            await asyncio.sleep(0)

    kept = on_disk(store)[session.id]
    assert kept.black == model_record(4)
    assert [replaced.new.model for replaced in kept.session.replacements] == [OTHER]
    await games.close()


async def test_a_watch_link_cannot_hand_a_paused_side_to_anybody():
    games = GameRegistry()
    session, white = await paused_game(games)
    watch = games.seat(white).watch

    with pytest.raises(ActionRejectedError, match="you are watching this game"):
        games.seat(watch).replace(RandomPlayer(1), RandomRecord())

    assert session.paused is not None
    await games.close()


async def test_a_side_that_is_not_waiting_cannot_be_handed_over(tmp_path):
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session = waiting_game()
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None

    with pytest.raises(ActionRejectedError, match="no player is waiting"):
        games.seat(links.white).replace(RandomPlayer(1), model_record(4))

    assert on_disk(store)[session.id].black == RandomRecord()
    await games.close()


def test_a_file_that_cannot_be_read_is_left_out_and_left_alone(tmp_path, caplog):
    store = GameStore(tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    # Left by a server that died while writing: never read, whatever it holds.
    (tmp_path / "half.json.tmp").write_text("{")

    assert store.read_all() == []
    assert "Cannot read the game" in caplog.text
    assert (tmp_path / "broken.json").exists()


def test_a_game_id_that_could_name_a_file_elsewhere_is_refused(tmp_path):
    with pytest.raises(ValueError, match="names its file"):
        GameStore(tmp_path).remove("../escape")


async def failed_game(games: GameRegistry) -> tuple[GameSession, str]:
    """A game whose player failed after White's first move, and White's link to it."""
    broken = BrokenPlayer()
    session = GameSession(HumanPlayer(), broken)
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None
    await settled()
    games.seat(links.white).submit_move("e2e4")
    await settled()
    broken.fail()
    await ended(session)
    return session, links.white


async def test_a_game_whose_player_failed_is_kept_on_disk_as_still_being_played(tmp_path):
    """A player failing is not how the game ended: an engine killed as the server goes down
    for an update fails the same way, and the game has to carry on after the restart."""
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session, white = await failed_game(games)
    await games.close()

    kept = on_disk(store)[session.id]
    assert kept.session.game_over is None
    assert [move.uci for move in kept.session.moves] == ["e2e4"]
    after = GameRegistry(store=store)
    restore_all(after, store)
    assert after.seat(white).state.position.game_over is None
    await after.close()


async def test_a_failed_game_that_was_played_again_stays_ended_after_a_restart(tmp_path):
    """Whoever played it again has moved on, and the old game must not play itself out, hold
    a place among the games in progress, or be saved as a game nobody was watching."""
    store = GameStore(tmp_path / "ongoing")
    kept: list[GameState] = []
    games = GameRegistry(store=store)
    session, white = await failed_game(games)
    games.start(waiting_game(), players=PERSON_AND_RANDOM, follows=games.seat(white))
    await games.close()

    after = GameRegistry(on_finished=kept.append, store=store)
    restore_all(after, store)

    over = after.seat(white).state.position.game_over
    assert over is not None and over.reason == "error"
    assert after.ongoing == 1
    await after.close()
    assert kept == []


async def test_a_game_whose_player_fails_on_every_restart_still_expires(tmp_path):
    """A player that is really broken fails again each time the game is taken back. That is
    not a move, and must not keep the game from expiring however often the server restarts."""
    store = GameStore(tmp_path / "ongoing")
    games = GameRegistry(store=store)
    session = waiting_game()
    links = games.start(session, players=PERSON_AND_RANDOM)
    assert links.white is not None
    broken = BrokenPlayer()
    session._players[chess.BLACK] = broken
    await settled()
    games.seat(links.white).submit_move("e2e4")
    await settled()
    last_move = session.updated
    broken.fail()
    await ended(session)
    await games.close()

    assert on_disk(store)[session.id].session.updated == last_move


async def test_a_game_that_has_just_ended_is_kept_its_full_time_from_then(tmp_path):
    later = timedelta()
    games = GameRegistry(expire_after=timedelta(days=7), clock=lambda: datetime.now(UTC) + later)
    session = waiting_game()
    links = games.start(session)
    assert links.white is not None
    await settled()
    session.updated -= timedelta(days=6, hours=23)

    games.seat(links.white).resign()
    later = timedelta(hours=2)

    assert games.expire() == 0
    await games.close()
