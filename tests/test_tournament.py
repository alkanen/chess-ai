import anyio
import pytest
from pgn_helpers import read_back
from stockfish_helpers import fake_engine, started, wait_until_gone
from training_helpers import model_run

from chess_ai.openings import load_opening_set
from chess_ai.players import GameContext, PlayerMove
from chess_ai.rating import Pairing, rate_round_robin
from chess_ai.tournament import (
    Anchor,
    ModelEntrant,
    StockfishEntrant,
    TournamentDefinition,
    TournamentError,
    create_tournament,
    entrant_players,
    leaderboard,
    open_tournament,
    play_tournament,
    schedule,
)

pytestmark = pytest.mark.anyio


class RankedMovePlayer:
    """Plays the legal move at ``rank`` in UCI order, wrapping round: deterministic, quick, and
    a different player for every rank."""

    def __init__(self, name: str, rank: int) -> None:
        self.name = name
        self._rank = rank

    async def choose_move(self, context: GameContext) -> PlayerMove:
        moves = sorted(context.board.legal_moves, key=lambda move: move.uci())
        return PlayerMove(moves[self._rank % len(moves)])


def definition(name: str = "cup", entrants: int = 3, games: int = 2) -> TournamentDefinition:
    return TournamentDefinition(
        name=name,
        entrants=[StockfishEntrant(elo=1350 + 100 * i, move_time=0.01) for i in range(entrants)],
        games_per_pairing=games,
        openings="standard",
        seed=1,
    )


def fakes(entrant: int, opponent: int, number: int) -> RankedMovePlayer:
    return RankedMovePlayer(f"entrant {entrant}", rank=entrant)


async def test_every_pairing_plays_its_games_round_by_round_with_the_colours_swapped(tmp_path):
    create_tournament(tmp_path, definition(games=2))

    await play_tournament(open_tournament(tmp_path, "cup"), fakes)

    games = open_tournament(tmp_path, "cup").games()
    assert [(game.round, game.white, game.black) for game in games] == [
        (1, 1, 0), (1, 0, 2), (1, 2, 1),
        (2, 0, 1), (2, 2, 0), (2, 1, 2),
    ]  # fmt: skip
    standard = load_opening_set("standard")
    for game in games:
        pgn = read_back(game.pgn)
        assert pgn.headers["White"] == f"entrant {game.white}"
        assert pgn.headers["Black"] == f"entrant {game.black}"
        assert pgn.headers["OpeningSet"] == standard.label
        assert game.result == pgn.headers["Result"] != "*"
    # Game 1 and game 2 of a pairing: one line, played both ways round.
    first, second = games[0], games[3]
    assert read_back(first.pgn).headers["Opening"] == read_back(second.pgn).headers["Opening"]


class Stopped(Exception):
    """What a test stops a tournament with, as Ctrl-C would."""


def stopping_after(games: int):
    """Fake players, until ``games`` games have been started: then a stop."""
    started = 0

    def player(entrant: int, opponent: int, number: int) -> RankedMovePlayer:
        nonlocal started
        if entrant < opponent:  # Once a game: one of its two players is the lower.
            started += 1
            if started > games:
                raise Stopped
        return fakes(entrant, opponent, number)

    return player


async def test_a_stopped_tournament_plays_only_the_games_it_had_not_finished(tmp_path):
    create_tournament(tmp_path, definition(games=2))
    with pytest.raises(Stopped):
        await play_tournament(open_tournament(tmp_path, "cup"), stopping_after(4))
    before = open_tournament(tmp_path, "cup").games()
    # Killed in the middle of writing the fifth game's line.
    with (tmp_path / "cup" / "games.jsonl").open("a") as file:
        file.write('{"round": 2, "white": 2, "bla')

    assert len(open_tournament(tmp_path, "cup").games()) == 4, "a torn line is not a game"
    await play_tournament(open_tournament(tmp_path, "cup"), fakes)

    after = open_tournament(tmp_path, "cup").games()
    assert after[:4] == before
    assert [(game.round, game.white, game.black) for game in after[4:]] == [(2, 2, 0), (2, 1, 2)]


class WaitingPlayer(RankedMovePlayer):
    """Waits for ``go`` before its first move, so that a test can look at a tournament mid-game."""

    def __init__(self, name: str, rank: int, go: anyio.Event, waiting: anyio.Event) -> None:
        super().__init__(name, rank)
        self._go, self._waiting = go, waiting

    async def choose_move(self, context: GameContext) -> PlayerMove:
        self._waiting.set()
        await self._go.wait()
        return await super().choose_move(context)


async def test_a_tournament_being_played_cannot_be_played_by_another_at_once(tmp_path):
    create_tournament(tmp_path, definition(games=1))
    go, waiting = anyio.Event(), anyio.Event()

    def waiting_players(entrant: int, opponent: int, number: int) -> RankedMovePlayer:
        return WaitingPlayer(f"entrant {entrant}", entrant, go, waiting)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(play_tournament, open_tournament(tmp_path, "cup"), waiting_players)
        await waiting.wait()
        with pytest.raises(TournamentError, match="being played"):
            await play_tournament(open_tournament(tmp_path, "cup"), fakes)
        assert open_tournament(tmp_path, "cup").playing()
        go.set()

    assert not open_tournament(tmp_path, "cup").playing()
    assert len(open_tournament(tmp_path, "cup").games()) == 3


async def test_each_model_is_told_its_opponents_rating(tmp_path):
    runs = model_run(tmp_path / "runs").parent
    engine = fake_engine(tmp_path / "engine")
    entrants = [
        ModelEntrant(run="tiny", step=2, rating=1500),
        ModelEntrant(run="tiny", step=4, rating=1800),
        StockfishEntrant(elo=1400, move_time=0.01),
    ]
    told = {0: {1: 1800, 2: 1400}, 1: {0: 1500, 2: 1400}}
    create_tournament(
        tmp_path,
        TournamentDefinition(
            name="cup", entrants=entrants, games_per_pairing=2, openings="standard", seed=1
        ),
    )
    tournament = open_tournament(tmp_path, "cup")

    async with entrant_players(
        tournament.definition, runs=runs, stockfish=str(engine), device="cpu"
    ) as players:
        await play_tournament(tournament, players)

    games = tournament.games()
    assert len(games) == 6
    for game in games:
        headers = read_back(game.pgn).headers
        for color, entrant, opponent in (
            ("White", game.white, game.black),
            ("Black", game.black, game.white),
        ):
            if entrant == 2:
                assert headers[f"{color}Elo"] == "1400"
                continue
            assert headers[f"{color}Checkpoint"] == str(entrants[entrant].step)
            assert headers[f"{color}Rating"] == str(entrants[entrant].rating)
            assert headers[f"{color}OpponentRating"] == str(told[entrant][opponent])
    assert len(started(engine)) == 1, "one engine for every game Stockfish played"
    wait_until_gone(*started(engine))


async def test_the_leaderboard_rates_the_entrants_from_the_crosstable_best_first(tmp_path):
    create_tournament(tmp_path, definition(entrants=4, games=2))
    await play_tournament(open_tournament(tmp_path, "cup"), fakes)
    tournament = open_tournament(tmp_path, "cup")

    board = leaderboard(tournament)

    assert (board.games_played, board.games_total) == (12, 12)
    for i in range(4):
        assert board.crosstable[i][i] is None
        for j in range(4):
            if i != j:
                assert board.crosstable[i][j].games == 2
                assert board.crosstable[i][j].points + board.crosstable[j][i].points == 2
    expected = rate_round_robin(
        4,
        [
            Pairing(i, j, board.crosstable[i][j].points, 2)
            for i in range(4)
            for j in range(i + 1, 4)
        ],
    )
    ratings = {standing.entrant: standing.rating for standing in board.standings}
    assert ratings == {i: pytest.approx(expected[i].rating) for i in range(4)}
    assert [s.rating for s in board.standings] == sorted(ratings.values(), reverse=True)


async def test_an_anchored_tournament_puts_its_anchor_at_the_rating_it_was_given(tmp_path):
    create_tournament(
        tmp_path, definition(games=2).model_copy(update={"anchor": Anchor(entrant=2, rating=1600)})
    )
    await play_tournament(open_tournament(tmp_path, "cup"), fakes)

    board = leaderboard(open_tournament(tmp_path, "cup"))

    anchored = next(standing for standing in board.standings if standing.entrant == 2)
    assert (anchored.rating, anchored.low, anchored.high) == (1600, 1600, 1600)


@pytest.mark.parametrize("entrants", range(2, 9))
def test_white_and_black_are_shared_out_evenly_even_with_one_game_a_pairing(entrants):
    whites = [0] * entrants
    blacks = [0] * entrants
    for scheduled in schedule(definition(entrants=entrants, games=1)):
        # One game a pairing: the first entrant of the pairing has White.
        whites[scheduled.first] += 1
        blacks[scheduled.second] += 1

    for white, black in zip(whites, blacks, strict=True):
        assert abs(white - black) <= 1, (whites, blacks)


async def test_a_glance_at_whether_a_tournament_is_played_does_not_stop_it_starting(
    tmp_path, monkeypatch
):
    import fcntl
    import os

    from chess_ai import tournament as module

    tournament = create_tournament(tmp_path, definition(games=1))
    # As Tournament.playing() holds the lock for the moment it looks, from the web server.
    descriptor = os.open(tournament.path / "playing.lock", os.O_RDONLY | os.O_CREAT)
    fcntl.flock(descriptor, fcntl.LOCK_SH)
    pauses = []

    def look_ends(seconds: float) -> None:
        # The look is over by the first pause, however busy the machine: no thread to wait for.
        if not pauses:
            os.close(descriptor)
        pauses.append(seconds)

    monkeypatch.setattr(module, "_pause", look_ends)

    await play_tournament(open_tournament(tmp_path, "cup"), fakes)

    assert len(pauses) == 1, "it was refused once, and waited"
    assert len(open_tournament(tmp_path, "cup").games()) == 3


def test_a_tournament_that_cannot_be_written_leaves_nothing_behind(tmp_path, monkeypatch):
    import errno
    from pathlib import Path

    def full(self, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_text", full)

    with pytest.raises(TournamentError, match="No space left"):
        create_tournament(tmp_path / "tournaments", definition())

    assert list((tmp_path / "tournaments").iterdir()) == []


async def test_one_tournament_object_held_cannot_be_played_twice_at_once(tmp_path):
    create_tournament(tmp_path, definition(games=1))
    tournament = open_tournament(tmp_path, "cup")
    go, waiting = anyio.Event(), anyio.Event()

    def waiting_players(entrant: int, opponent: int, number: int) -> RankedMovePlayer:
        return WaitingPlayer(f"entrant {entrant}", entrant, go, waiting)

    with tournament.held():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(play_tournament, tournament, waiting_players)
            await waiting.wait()
            with pytest.raises(TournamentError, match="already being played here"):
                await play_tournament(tournament, fakes)
            with (
                pytest.raises(TournamentError, match="already being played here"),
                tournament.held(),
            ):
                pass
            go.set()

    games = open_tournament(tmp_path, "cup").games()
    assert len(games) == 3, "each game once"


def two_models(tmp_path):
    """A tournament of two checkpoints of the run ``tiny``, as ``tournament start`` makes it."""
    from training_helpers import model_run

    from chess_ai.training.run_store import RunReader, choose_checkpoint

    run = RunReader(model_run(tmp_path / "runs"))
    entrants = [
        ModelEntrant(
            run="tiny",
            step=step,
            checkpoint_written=run.checkpoint_written(choose_checkpoint(run, step)),
            rating=1500,
        )
        for step in (2, 4)
    ]
    create_tournament(
        tmp_path,
        TournamentDefinition(
            name="cup", entrants=entrants, games_per_pairing=1, openings="standard", seed=1
        ),
    )
    return open_tournament(tmp_path, "cup")


async def test_a_checkpoint_saved_again_while_it_is_loaded_is_refused(tmp_path, monkeypatch):
    import os

    from chess_ai import inference

    tournament = two_models(tmp_path)
    load = inference.load_engine

    def saved_again_meanwhile(path, **options):
        engine = load(path, **options)
        os.utime(path, (1_700_000_000, 1_700_000_000))
        return engine

    monkeypatch.setattr(inference, "load_engine", saved_again_meanwhile)

    with pytest.raises(TournamentError, match="has changed since the tournament started"):
        async with entrant_players(tournament.definition, runs=tmp_path / "runs", stockfish="none"):
            pass


async def test_a_checkpoint_gone_by_the_time_it_is_loaded_is_said_to_be_gone(tmp_path, monkeypatch):
    from chess_ai.training.run_store import RunReader

    tournament = two_models(tmp_path)
    # Pruned after the run's checkpoints were listed, before its file was looked at.
    monkeypatch.setattr(RunReader, "checkpoint_written", lambda self, info: None)

    with pytest.raises(TournamentError, match="entrant 1: .* is gone"):
        async with entrant_players(tournament.definition, runs=tmp_path / "runs", stockfish="none"):
            pass


async def test_a_checkpoint_pruned_once_it_is_loaded_still_plays(tmp_path, monkeypatch):
    from chess_ai import inference

    tournament = two_models(tmp_path)
    load = inference.load_engine

    def pruned_meanwhile(path, **options):
        engine = load(path, **options)
        path.unlink()  # As a run still training prunes it: the network is in memory already.
        return engine

    monkeypatch.setattr(inference, "load_engine", pruned_meanwhile)

    async with entrant_players(
        tournament.definition, runs=tmp_path / "runs", stockfish="none"
    ) as players:
        await play_tournament(tournament, players)

    assert len(tournament.games()) == 1
