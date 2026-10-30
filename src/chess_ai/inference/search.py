"""The search player: a checkpoint that looks ahead before it moves, by Monte Carlo tree search.

The search is AlphaZero's and Leela Chess Zero's, PUCT. Each position in the tree is evaluated
once by the network: its policy gives every move a prior, and its win/draw/loss head says how
the position stands for the side to move. A position the game is over in is scored by the rules
instead, as the game itself would end there. The tree grows towards moves that score well, or
that the policy likes and have not been looked at much, and the move played is the one looked
at most.

How long it searches is a number of nodes rather than a time, for the reason the ladder holds
Stockfish to nodes: the same number plays the same moves on any machine, however busy.

Positions are evaluated in batches, since a network asked about one position at a time leaves
a GPU idle. A batch is gathered with virtual loss: each position on its way to the network
counts as a loss for whoever moved into it, so that the next one gathered goes elsewhere. The
batch is a constant rather than a setting, because a different batch grows a slightly different
tree, and a node count is meant to say how the player plays.

The constants are Leela's defaults. Leela's values run from -1 to 1, and so do the ones here,
so that its exploration constant means what it means there; the win/draw/loss head's expected
score, win + draw / 2, is that value halved and moved up by a half.
"""

import asyncio
import math
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Final

import chess
import numpy as np

from chess_ai.inference.engine import Evaluation, InferenceEngine
from chess_ai.inference.player import TOP_CANDIDATES, ModelPlayer
from chess_ai.inference.selection import sample
from chess_ai.players import (
    DEFAULT_TEMPERATURE,
    CandidateMove,
    GameContext,
    ModelDescription,
    PlayerMove,
    SearchDescription,
    SelectionStrategy,
    Thoughts,
    WinDrawLoss,
)
from chess_ai.position_view import game_over

LEAF_BATCH: Final = 16
"""How many positions are gathered for the network at once."""

CPUCT: Final = 1.745
"""Leela's exploration constant: how much a move's prior counts against its score."""
CPUCT_BASE: Final = 38739.0
CPUCT_FACTOR: Final = 3.894
"""How the exploration constant grows with the visits of a position, as Leela's does."""
FPU_REDUCTION: Final = 0.330
"""First-play urgency, Leela's default: a move not yet looked at is taken to score this much
below the position it is played from, more so the more of the policy has been looked at."""

KEPT_GAMES: Final = 64
"""How many games' trees a player keeps, the oldest let go of first: as many games as a ladder
plays at once, and more."""


class _Abandoned(Exception):
    """A search stopped because nobody waits for its move any more."""


@dataclass(eq=False)
class _Searching:
    """One search going on: whether it is to stop, and whether what it finds is to be kept.

    Equal only to itself, so that a search going on is told apart from another of the same
    game when it is taken off the list of them.
    """

    stop: threading.Event = field(default_factory=threading.Event)
    """Set once nobody waits for the move, which stops the search at its next batch."""
    wanted: bool = True
    """Cleared, under the player's lock, once the game it searches for is forgotten."""


class _Node:
    """A position in the tree, reached by one move from its parent."""

    __slots__ = ("children", "in_flight", "prior", "terminal", "total", "visits", "wdl")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.total = 0.0
        """The sum of the values backed up through here, for the side that moved into it."""
        self.in_flight = 0
        """Positions below here on their way to the network, each counted as a loss meanwhile."""
        self.children: dict[chess.Move, _Node] | None = None
        """The legal moves, best prior first; ``None`` until the position has been evaluated."""
        self.terminal: float | None = None
        """The value for the side to move where the rules have ended the game, else ``None``."""
        self.wdl: WinDrawLoss | None = None

    def q(self) -> float:
        """The average value for the side that moved into this position, losses in flight in."""
        return (self.total - self.in_flight) / (self.visits + self.in_flight)


@dataclass
class _Tree:
    """A game's tree, rooted at the position the game is in after ``moves``."""

    start: str
    """The FEN the game started from."""
    moves: list[chess.Move]
    root: _Node


class SearchPlayer:
    """A checkpoint that searches before each move until it has looked ``nodes`` times, and
    plays the move it looked at most, or draws one by how often it looked at each when
    ``strategy`` is ``sample``.

    Each look goes down the tree to one position, which the network evaluates or the rules
    decide: a mate found is looked at again and again, and counts each time. The looks kept
    from the move before count too, so a move the search expected may need none of its own.
    One node is the position moved from alone, which is the network's own move.

    The ratings are those :class:`~chess_ai.inference.ModelPlayer` takes. Each position in the
    tree is told them from its own side to move: the model's rating and the opponent's where
    the model is to move, and the other way around where the opponent is, who is modelled as
    the player the model really faces.

    What it found out about the position it moved to is kept for its next move, for each game it
    is playing, so that a move it expected starts with the search below it done.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        nodes: int,
        run: str | None = None,
        checkpoint: int | None = None,
        rating: int | None = None,
        opponent_rating: int | None = None,
        strategy: SelectionStrategy = "argmax",
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
        leaf_batch: int = LEAF_BATCH,
    ) -> None:
        if nodes < 1:
            raise ValueError(f"a search needs at least one node, not {nodes}")
        if leaf_batch < 1:
            raise ValueError(f"a batch is at least one position, not {leaf_batch}")
        self._engine = engine
        self._nodes = nodes
        self._leaf_batch = leaf_batch
        self._rating = rating
        self._opponent_rating = opponent_rating if opponent_rating is not None else rating
        self._strategy: SelectionStrategy = strategy
        self._temperature = temperature
        self._rng = random.Random(seed)
        self._trees: OrderedDict[str, _Tree] = OrderedDict()
        self._searching: dict[str, list[_Searching]] = {}
        """The searches going on, by game, which a game forgotten meanwhile tells not to keep
        their trees."""
        self._lock = threading.Lock()
        self._model = ModelDescription(
            run=run if run is not None else engine.run,
            checkpoint=checkpoint if checkpoint is not None else engine.step,
            rating=rating,
            opponent_rating=self._opponent_rating,
            strategy=strategy,
            temperature=temperature if strategy == "sample" else None,
            search=SearchDescription(nodes=nodes),
        )

    @property
    def name(self) -> str:
        return f"{self._model.run} step {self._model.checkpoint}, {self._nodes:,} nodes"

    @property
    def model(self) -> ModelDescription:
        return self._model

    @property
    def kept_games(self) -> int:
        """How many games' trees it is keeping for their next move."""
        with self._lock:
            return len(self._trees)

    def forget(self, game: str) -> None:
        """Let go of what was kept of ``game``, which is over or has been left, and of what a
        search for it still going on would keep.

        That search is not stopped: a game closed while its player thinks is cancelled as
        well where it is not to wait for the move, which stops it. It plays its move as it
        would have, for nobody, and keeps nothing of it.
        """
        with self._lock:
            self._trees.pop(game, None)
            for searching in self._searching.get(game, ()):
                searching.wanted = False

    async def choose_move(self, context: GameContext) -> PlayerMove:
        # In a thread, as a model player's forward pass is, and the whole search with it. A
        # thread cannot be cancelled, so a search that is told it is no longer wanted stops
        # at its next batch: ctrl-c, or a game given up on, would otherwise wait for it.
        searching = _Searching()
        with self._lock:
            self._searching.setdefault(context.game, []).append(searching)
        searched = asyncio.get_running_loop().run_in_executor(
            None, self._choose, context, searching
        )
        try:
            return await searched
        finally:
            # However the wait ends: cancelled, interrupted, or the coroutine closed. Once the
            # search has returned it is too late to matter, its tree kept or not already. A
            # wait that ended otherwise than by cancelling leaves nobody to read what the
            # search comes back with, so it is cancelled too rather than left to be reported
            # as an exception nobody retrieved.
            searching.stop.set()
            searched.cancel()
            with self._lock:
                searches = self._searching[context.game]
                searches.remove(searching)
                if not searches:
                    del self._searching[context.game]

    def _choose(self, context: GameContext, searching: _Searching) -> PlayerMove:
        board = context.board
        with self._lock:
            tree = self._trees.pop(context.game, None)
        root = _reached(tree, board)
        self._search(root, board, searching.stop)
        if searching.stop.is_set():
            raise _Abandoned
        move = self._pick(root, board)
        moves = list(board.move_stack)
        start = board.root().fen()
        assert root.children is not None
        kept = _Tree(start=start, moves=[*moves, move], root=root.children[move])
        with self._lock:
            # Decided under the lock that forget() takes: a game it has let go of meanwhile
            # keeps nothing, which would otherwise stand in for a game still going on.
            if searching.wanted and not searching.stop.is_set():
                self._trees[context.game] = kept
                while len(self._trees) > KEPT_GAMES:
                    self._trees.popitem(last=False)
        return PlayerMove(move, thoughts=_thoughts(root, board))

    def _search(self, root: _Node, board: chess.Board, stop: threading.Event) -> None:
        """Look below ``root`` until it has been looked at :attr:`_nodes` times, or until
        ``stop`` is set."""
        own = board.turn
        while root.visits < self._nodes and not stop.is_set():
            leaves: list[tuple[list[_Node], chess.Board]] = []
            # A way down to a position the rules decide counts towards the batch as one to the
            # network does. Scored at once, it would otherwise be found again and again while
            # everything else in the batch counts as lost.
            for _ in range(min(self._leaf_batch, self._nodes - root.visits)):
                path, leaf = _descend(root, board)
                node = path[-1]
                if node.in_flight:
                    # Already on its way to the network: the batch is as wide as the tree is.
                    break
                if node.terminal is None and node.children is None:
                    over = game_over(leaf)
                    if over is not None:
                        node.terminal = -1.0 if over.reason == "checkmate" else 0.0
                if node.terminal is not None:
                    _back_up(path, node.terminal, in_flight=False)
                    continue
                for passed in path:
                    passed.in_flight += 1
                leaves.append((path, leaf))
            if not leaves:
                continue
            ratings = [
                (self._rating, self._opponent_rating)
                if leaf.turn == own
                else (self._opponent_rating, self._rating)
                for _, leaf in leaves
            ]
            evaluations = self._engine.evaluate_many([leaf for _, leaf in leaves], ratings=ratings)
            for (path, _), evaluation in zip(leaves, evaluations, strict=True):
                node = path[-1]
                _expand(node, evaluation)
                _back_up(path, evaluation.wdl.win - evaluation.wdl.loss, in_flight=True)

    def _pick(self, root: _Node, board: chess.Board) -> chess.Move:
        """The move to play: the one looked at most, or one drawn by how often each was.

        With nothing below the root looked at, which is a search of one node, the policy is
        all there is to go on, and the move is the network's own: its best, or one drawn from
        its distribution.
        """
        assert root.children is not None
        moves = list(root.children)
        visits = np.array([child.visits for child in root.children.values()], dtype=np.float64)
        if not visits.any():
            weights = np.array([child.prior for child in root.children.values()])
        else:
            weights = visits
        if self._strategy == "argmax":
            # The children are in the policy's order, so ties go to the move it liked better.
            return moves[int(np.argmax(weights))]
        return sample(moves, weights / weights.sum(), temperature=self._temperature, rng=self._rng)


def _reached(tree: _Tree | None, board: chess.Board) -> _Node:
    """The node of ``tree`` that ``board`` is at, or a new one if the tree does not reach it."""
    if tree is None or board.root().fen() != tree.start:
        return _Node(1.0)
    played = board.move_stack
    if played[: len(tree.moves)] != tree.moves:
        return _Node(1.0)
    node = tree.root
    for move in played[len(tree.moves) :]:
        if node.children is None or move not in node.children:
            return _Node(1.0)
        node = node.children[move]
    return node


def _descend(root: _Node, board: chess.Board) -> tuple[list[_Node], chess.Board]:
    """Follow the best-scoring moves from ``root`` down to a position not yet evaluated, or
    one the game is over in, and return the way down and that position."""
    path = [root]
    node = root
    played = []
    while node.children is not None and node.terminal is None:
        move, node = _best_child(node)
        board.push(move)
        played.append(move)
        path.append(node)
    leaf = board.copy()
    for _ in played:
        board.pop()
    return path, leaf


def _best_child(node: _Node) -> tuple[chess.Move, _Node]:
    """The child PUCT picks: the best of value plus a bonus for its prior and few visits."""
    assert node.children is not None
    seen = node.visits + node.in_flight
    cpuct = CPUCT + CPUCT_FACTOR * math.log((node.visits + CPUCT_BASE) / CPUCT_BASE)
    bonus = cpuct * math.sqrt(max(seen - 1, 1))
    looked_at = sum(
        child.prior for child in node.children.values() if child.visits + child.in_flight
    )
    # The value of the position for its side to move, which is the side choosing among these.
    standing = -node.total / node.visits if node.visits else 0.0
    unvisited = standing - FPU_REDUCTION * math.sqrt(looked_at)
    best: tuple[chess.Move, _Node] | None = None
    best_score = -math.inf
    for move, child in node.children.items():
        count = child.visits + child.in_flight
        q = child.q() if count else unvisited
        score = q + bonus * child.prior / (1 + count)
        if score > best_score:
            best, best_score = (move, child), score
    assert best is not None
    return best


def _expand(node: _Node, evaluation: Evaluation) -> None:
    node.wdl = evaluation.wdl
    node.children = {move: _Node(prior) for move, prior in evaluation.moves}


def _back_up(path: list[_Node], value: float, *, in_flight: bool) -> None:
    """Add ``value``, for the side to move at the end of ``path``, to every node on it."""
    for node in reversed(path):
        node.visits += 1
        node.total -= value
        if in_flight:
            node.in_flight -= 1
        value = -value


def _thoughts(root: _Node, board: chess.Board) -> Thoughts:
    """The moves looked at most, each with its share of the looking, and how the network saw
    the position."""
    assert root.children is not None
    looked = sum(child.visits for child in root.children.values())
    if looked:
        shares = [(move, child.visits / looked) for move, child in root.children.items()]
    else:
        shares = [(move, child.prior) for move, child in root.children.items()]
    shares.sort(key=lambda pair: -pair[1])
    return Thoughts(
        candidates=[
            CandidateMove(uci=move.uci(), san=board.san(move), probability=share)
            for move, share in shares[:TOP_CANDIDATES]
        ],
        wdl=root.wdl,
    )


def checkpoint_player(
    engine: InferenceEngine,
    *,
    search: SearchDescription | None,
    run: str | None = None,
    checkpoint: int | None = None,
    rating: int | None = None,
    opponent_rating: int | None = None,
    strategy: SelectionStrategy = "argmax",
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = None,
) -> "ModelPlayer | SearchPlayer":
    """A checkpoint sat down at the board: one that searches as ``search`` says, or one that
    plays straight from its policy where ``search`` is ``None``. The rest is as
    :class:`~chess_ai.inference.ModelPlayer` takes it."""
    if search is not None:
        return SearchPlayer(
            engine,
            nodes=search.nodes,
            run=run,
            checkpoint=checkpoint,
            rating=rating,
            opponent_rating=opponent_rating,
            strategy=strategy,
            temperature=temperature,
            seed=seed,
        )
    return ModelPlayer(
        engine,
        run=run,
        checkpoint=checkpoint,
        rating=rating,
        opponent_rating=opponent_rating,
        strategy=strategy,
        temperature=temperature,
        seed=seed,
    )
