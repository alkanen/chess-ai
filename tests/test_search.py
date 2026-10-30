"""The search player: a checkpoint that looks ahead before it moves.

The networks here answer the same thing in every position, so that what the search finds is
the search's own doing: a mate the policy does not point at is found by looking, and a draw by
the rules is a draw whatever the value head says.
"""

import asyncio
import dataclasses
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import chess
import pytest
import torch

from chess_ai.encoders import create_encoder
from chess_ai.inference import InferenceEngine
from chess_ai.inference.search import SearchPlayer
from chess_ai.models import ChessModel, ModelOutput
from chess_ai.move_codec import VOCABULARY_SIZE, move_index
from chess_ai.players import MIN_TEMPERATURE, GameContext, WinDrawLoss
from chess_ai.position_view import game_over


class FixedModel(ChessModel):
    """A model that answers the same logits whatever it is shown."""

    def __init__(self, spec, policy: torch.Tensor, value: torch.Tensor) -> None:
        super().__init__(spec)
        self.register_buffer("policy", policy)
        self.register_buffer("value", value)

    def forward(self, spatial: torch.Tensor, globals: torch.Tensor) -> ModelOutput:
        batch = spatial.shape[0]
        return ModelOutput(policy=self.policy.expand(batch, -1), value=self.value.expand(batch, -1))


def fixed_engine(
    policy: dict[str, float] | None = None, value=(0.0, 0.0, 0.0), **settings
) -> InferenceEngine:
    """An engine whose policy logits are 0 everywhere but the UCI moves named, and whose
    win/draw/loss logits are ``value`` in every position: an even game, unless it says not."""
    encoder = create_encoder("board-planes")
    logits = torch.zeros((1, VOCABULARY_SIZE))
    for uci, logit in (policy or {}).items():
        logits[0, move_index(chess.Move.from_uci(uci))] = logit
    return InferenceEngine(
        FixedModel(encoder.spec, logits, torch.tensor([value])), encoder, **settings
    )


class OneSideWinning(InferenceEngine):
    """An engine whose network says ``winner`` is winning every position, from either side's
    point of view: a value that agrees with itself however deep the tree, as a constant answer
    for the side to move, which flips at every move, does not."""

    def __init__(self, winner: chess.Color, engine: InferenceEngine) -> None:
        super().__init__(engine._model, engine._encoder)
        self._winner = winner

    def evaluate_many(self, boards, **ratings):
        winning = WinDrawLoss(win=1.0, draw=0.0, loss=0.0)
        losing = WinDrawLoss(win=0.0, draw=0.0, loss=1.0)
        return [
            dataclasses.replace(evaluation, wdl=winning if board.turn == self._winner else losing)
            for board, evaluation in zip(
                boards, super().evaluate_many(boards, **ratings), strict=True
            )
        ]


def chosen(player: SearchPlayer, board: chess.Board) -> str:
    return asyncio.run(player.choose_move(GameContext(board.copy()))).move.uci()


def test_one_node_plays_the_move_the_network_likes_best():
    """One node is the root alone, which is the network's own answer and nothing more."""
    board = chess.Board()

    player = SearchPlayer(fixed_engine({"g1f3": 2.0, "e2e4": 1.0}), nodes=1)

    assert chosen(player, board) == "g1f3"


def mates_in_one(board: chess.Board) -> set[str]:
    mates = set()
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mates.add(move.uci())
        board.pop()
    return mates


def mates_in_two(board: chess.Board) -> set[str]:
    """The moves after which every reply leaves a mate in one."""
    mates = set()
    for move in board.legal_moves:
        board.push(move)
        replies = list(board.legal_moves)
        forced = bool(replies)
        for reply in replies:
            board.push(reply)
            forced = forced and bool(mates_in_one(board))
            board.pop()
        if forced:
            mates.add(move.uci())
        board.pop()
    return mates


def test_a_mate_in_one_is_played_where_the_policy_likes_another_move():
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert mates_in_one(board) == {"h5f7"}

    player = SearchPlayer(fixed_engine({"h5h3": 3.0}), nodes=512)

    assert chosen(player, board) == "h5f7"


def test_a_forced_mate_in_two_is_found_at_a_modest_node_count():
    """Two rooks against a bare king: one to cut it off, the other to mate it. The network
    knows nothing, so every move looks alike to it until the search finds the mate: a few
    hundred nodes spread evenly over two dozen moves do not reach it, and 800 do."""
    board = chess.Board("7k/8/8/8/8/8/1R6/R1K5 w - - 0 1")
    assert not mates_in_one(board)
    mating = mates_in_two(board)
    assert mating

    player = SearchPlayer(fixed_engine(), nodes=800)

    assert chosen(player, board) in mating


def after(fen: str, *moves: str) -> chess.Board:
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    return board


def drawing(board: chess.Board) -> set[str]:
    """The moves after which the game is drawn by the rules, as a game played here ends: by
    the fifty-move rule and threefold repetition as soon as they can be claimed."""
    drawn = set()
    for move in board.legal_moves:
        board.push(move)
        over = game_over(board)
        if over is not None and over.reason != "checkmate":
            drawn.add(move.uci())
        board.pop()
    return drawn


DRAWS = {
    "stalemate": (chess.Board("k7/3Q4/1K6/8/8/8/8/8 w - - 0 1"), "d7c7"),
    "insufficient material": (chess.Board("7k/8/8/4p3/8/5N2/8/2K5 w - - 0 1"), "f3e5"),
    "the fifty-move rule": (chess.Board("7k/8/8/8/8/8/P7/R1K5 w - - 99 80"), "c1d1"),
    "threefold repetition": (
        after(chess.STARTING_FEN, "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"),
        "f6g8",
    ),
}


@pytest.mark.parametrize(("board", "liked"), DRAWS.values(), ids=list(DRAWS))
def test_a_draw_by_the_rules_is_a_draw_whatever_the_network_says(board, liked):
    """The network says the side to move is winning everywhere, so to it every move wins; the
    one that draws by the rules, which its policy likes best, is not played."""
    assert liked in drawing(board)

    player = SearchPlayer(OneSideWinning(board.turn, fixed_engine({liked: 1.0})), nodes=256)

    assert chosen(player, board) not in drawing(board)


def test_a_mate_wins_whatever_the_network_says():
    """The network says the side to move is losing everywhere, so to it every move loses, the
    mate among them."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")

    player = SearchPlayer(OneSideWinning(chess.BLACK, fixed_engine({"h5h3": 1.0})), nodes=256)

    assert chosen(player, board) == "h5f7"


@pytest.mark.parametrize("batch", [1, 4, 64])
def test_the_batch_the_network_is_asked_in_does_not_change_the_move(batch):
    """A different batch grows a slightly different tree, but not one that plays another move
    where the search has found the right one. Near the node count where it first finds it, the
    batch can decide whether it has: a mate in one the policy gives a sixtieth of what it gives
    another move is found at 256 nodes in batches of 16 and not one at a time."""
    mate_in_one = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    mate_in_two = chess.Board("7k/8/8/8/8/8/1R6/R1K5 w - - 0 1")

    for board, nodes in ((mate_in_one, 512), (mate_in_two, 1024)):
        batched = SearchPlayer(fixed_engine({"h5h3": 3.0}), nodes=nodes, leaf_batch=batch)
        one_at_a_time = SearchPlayer(fixed_engine({"h5h3": 3.0}), nodes=nodes, leaf_batch=1)
        assert chosen(batched, board) == chosen(one_at_a_time, board)


class Recording(InferenceEngine):
    """An engine that remembers every position it was asked about, and the ratings each was
    told."""

    def __init__(self, engine: InferenceEngine) -> None:
        super().__init__(engine._model, engine._encoder)
        self.asked: list[tuple[chess.Board, tuple[int | None, int | None]]] = []

    def evaluate_many(self, boards, *, ratings=None, **given):
        ratings = ratings or [(given.get("mover_rating"), given.get("opponent_rating"))] * len(
            boards
        )
        self.asked.extend(zip(boards, ratings, strict=True))
        return super().evaluate_many(boards, ratings=ratings)


def play(player: SearchPlayer, board: chess.Board, game: str = "") -> chess.Move:
    return asyncio.run(player.choose_move(GameContext(board.copy(), game=game))).move


def test_what_was_searched_below_the_move_played_is_kept_for_the_next_move():
    """Playing both sides, as in a game against itself: each move starts from the tree below
    the move before, so it asks the network about fewer positions than it searches."""
    engine = Recording(fixed_engine())
    player = SearchPlayer(engine, nodes=200)
    board = chess.Board()
    board.push(play(player, board))

    engine.asked.clear()
    play(player, board)

    assert 0 < len(engine.asked) < 200


def test_each_game_keeps_a_tree_of_its_own():
    """A ladder asks one player for moves in several games at once."""
    engine = Recording(fixed_engine())
    player = SearchPlayer(engine, nodes=200)
    board = chess.Board()
    board.push(play(player, board, game="first"))

    engine.asked.clear()
    play(player, board, game="second")

    assert len(engine.asked) == 200


def test_a_game_that_went_back_starts_a_new_tree():
    """As a takeback makes it: the position is not below the move the tree was kept for."""
    player = SearchPlayer(fixed_engine(), nodes=50)
    board = chess.Board()
    board.push(play(player, board))
    board.pop()
    board.push_uci("a2a3")

    assert board.is_legal(play(player, board))


def test_each_position_is_told_the_ratings_of_its_own_side_to_move():
    """The model's own where it is to move, and the opponent's, as the player it faces, where
    the opponent is."""
    engine = Recording(fixed_engine())
    player = SearchPlayer(engine, nodes=100, rating=1500, opponent_rating=2100)

    play(player, chess.Board())

    told = {board.turn: ratings for board, ratings in engine.asked}
    assert told == {chess.WHITE: (1500, 2100), chess.BLACK: (2100, 1500)}


def test_sampling_draws_from_how_often_each_move_was_looked_at():
    """Varied from one seed to the next, and the same again from the same seed."""
    board = chess.Board()

    def sampled(seed: int) -> chess.Move:
        return play(SearchPlayer(fixed_engine(), nodes=64, strategy="sample", seed=seed), board)

    assert len({sampled(seed) for seed in range(12)}) > 1
    assert sampled(3) == sampled(3)


def test_sampling_at_the_coldest_temperature_plays_the_move_looked_at_most():
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")

    for seed in range(5):
        player = SearchPlayer(
            fixed_engine({"h5h3": 1.0}),
            nodes=256,
            strategy="sample",
            temperature=MIN_TEMPERATURE,
            seed=seed,
        )
        assert play(player, board).uci() == "h5f7"


def test_sampling_with_one_node_draws_from_the_policy():
    board = chess.Board()
    drawn = {
        play(SearchPlayer(fixed_engine(), nodes=1, strategy="sample", seed=seed), board)
        for seed in range(12)
    }

    assert len(drawn) > 1


def test_a_game_that_is_over_leaves_no_tree_behind():
    """A match plays game after game through one player: a finished game's tree would only
    take up memory."""
    from chess_ai.game_session import GameSession
    from chess_ai.players import RandomPlayer

    player = SearchPlayer(fixed_engine(), nodes=512)
    session = GameSession(
        player,
        RandomPlayer(),
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    )

    asyncio.run(session.play())

    assert session.state.position.game_over is not None
    assert player.kept_games == 0


class Gated(Recording):
    """An engine that holds every batch at a gate until the test lets it through, so that the
    test knows which batch a search is waiting on whatever the machine is doing meanwhile.

    Each batch says it has come (``came``) and waits to be let through (``through``). Once
    ``give_up`` is set, every batch fails instead, which ends a search that did not stop.
    """

    WAIT: Final = 60.0
    """How long either side waits for the other before the test is taken to have failed:
    long enough that no machine is that busy, and only ever waited out by a broken test,
    since the test lets every batch it holds through on its way out."""

    def __init__(self, engine: InferenceEngine) -> None:
        super().__init__(engine)
        self.came = threading.Semaphore(0)
        self.through = threading.Semaphore(0)
        self.give_up = threading.Event()

    def evaluate_many(self, boards, **given):
        self.came.release()
        if not self.through.acquire(timeout=self.WAIT) or self.give_up.is_set():
            raise RuntimeError("given up")
        return super().evaluate_many(boards, **given)


class Gate:
    """The test's side of a :class:`Gated` engine's gate, for as long as a test's coroutine is in
    it: ``async with Gate(engine) as gate``.

    The search has the loop's only executor thread to itself, so that a no-op queued behind it
    runs exactly when the search has ended; the test waits at the gate in a thread of its own.
    On the way out, everything held is let go of: a search that did not stop fails at its next
    batch, rather than leaving ``asyncio.run`` to wait for it.
    """

    def __init__(self, engine: Gated) -> None:
        self.engine = engine
        self._waits = ThreadPoolExecutor(max_workers=1)
        self._waiting: set[asyncio.Future] = set()

    async def __aenter__(self) -> "Gate":
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        return self

    async def __aexit__(self, *exception: object) -> None:
        self.engine.give_up.set()
        self.engine.through.release()
        self.engine.came.release()
        if self._waiting:
            await asyncio.wait(self._waiting)
        self._waits.shutdown()

    def _came(self) -> asyncio.Future:
        waiting = asyncio.get_running_loop().run_in_executor(
            self._waits, functools.partial(self.engine.came.acquire, timeout=Gated.WAIT)
        )
        self._waiting.add(waiting)
        return waiting

    async def came(self) -> None:
        """Wait until a batch is at the gate."""
        assert await self._came()

    async def let_through(self, batches: int) -> None:
        """Let ``batches`` batches through, one at a time as each comes."""
        for _ in range(batches):
            await self.came()
            self.engine.through.release()

    async def stopped_after_letting_through(self) -> bool:
        """Let the batch at the gate through, and say whether the search then ended without
        asking for another. Whichever comes first decides, with no clock in it: a search that
        stops ends, and one that does not is at the gate again."""
        ended = asyncio.get_running_loop().run_in_executor(None, lambda: None)
        asked = self._came()
        self.engine.through.release()
        done, _ = await asyncio.wait({ended, asked}, return_when=asyncio.FIRST_COMPLETED)
        return ended in done and asked not in done


def test_a_search_that_is_cancelled_stops_and_keeps_nothing():
    """As ctrl-c, or a game given up on, cancels it: the search stops asking the network rather
    than running on to the end of its nodes, and keeps no tree that would take another's place.

    The search is cancelled while a batch is held at the gate. Once that batch comes back, a
    search that stops ends; one that does not asks for the next.
    """
    engine = Gated(fixed_engine())
    player = SearchPlayer(engine, nodes=100_000, leaf_batch=1)

    async def stopped_after_cancelling() -> bool:
        async with Gate(engine) as gate:
            searching = asyncio.ensure_future(
                player.choose_move(GameContext(chess.Board(), game="given up"))
            )
            await gate.let_through(3)
            await gate.came()
            searching.cancel()
            with pytest.raises(asyncio.CancelledError):
                await searching
            return await gate.stopped_after_letting_through()

    assert asyncio.run(stopped_after_cancelling())
    assert player.kept_games == 0


def test_a_search_whose_wait_ends_otherwise_than_by_cancelling_stops_too():
    """Closed rather than cancelled, as a coroutine nobody will finish is: the search stops all
    the same, rather than running on to the end of its nodes, and keeps nothing."""
    engine = Gated(fixed_engine())
    player = SearchPlayer(engine, nodes=100_000, leaf_batch=1)

    async def stopped_after_closing() -> bool:
        async with Gate(engine) as gate:
            waiting = player.choose_move(GameContext(chess.Board(), game="closed"))
            # As far as the wait for the search's thread, which is where it is closed: anything
            # awaited before it would leave the test waiting at the gate for nothing.
            assert isinstance(waiting.send(None), asyncio.Future)
            await gate.came()
            waiting.close()
            return await gate.stopped_after_letting_through()

    assert asyncio.run(stopped_after_closing())
    assert player.kept_games == 0


class Forgetting(Recording):
    """An engine that, as it is first asked, has the game the search is for forgotten, as a
    game closed while its player thinks has."""

    def __init__(self, engine: InferenceEngine, game: str) -> None:
        super().__init__(engine)
        self.game = game
        self.player: SearchPlayer | None = None

    def evaluate_many(self, boards, **given):
        if self.player is not None and not self.asked:
            self.player.forget(self.game)
        return super().evaluate_many(boards, **given)


def test_a_game_forgotten_while_its_move_is_searched_keeps_no_tree():
    """The game closes before the search hears it is no longer wanted: the search runs on, and
    what it finds must not be kept for a game that is over."""
    engine = Forgetting(fixed_engine(), game="closed")
    player = SearchPlayer(engine, nodes=32)
    engine.player = player

    play(player, chess.Board(), game="closed")

    assert player.kept_games == 0
