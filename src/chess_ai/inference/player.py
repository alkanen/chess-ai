"""The model player: a checkpoint sat down at the board.

It is the inference engine and a selection strategy and nothing else — no rules, no search, no
opening book. A game session cannot tell it from the random mover, which is the point: the same
session, the same PGN and the same browser serve a person, a checkpoint and, later, a search.
"""

import asyncio
import random
from typing import Final

from chess_ai.inference.engine import Evaluation, InferenceEngine
from chess_ai.inference.selection import DEFAULT_TEMPERATURE, select_move
from chess_ai.players import (
    CandidateMove,
    GameContext,
    ModelDescription,
    PlayerMove,
    SelectionStrategy,
    Thoughts,
)

TOP_CANDIDATES: Final = 5
"""How many moves a player says it was considering, which is what the overlay shows.

Five, as the top-5 validation metric is: it is the number that says whether the move played was
one the network had in mind at all.
"""


class ModelPlayer:
    """Plays whichever move a checkpoint's policy leads to.

    The rating is what the model is asked to play like, and it is given as both sides' rating:
    the games it learned from are games between players of about equal strength, so asking for
    a 1600's move means asking for the move a 1600 plays against another 1600. ``None`` asks for
    a position that claims no rating at all, which the training data also holds.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        run: str | None = None,
        checkpoint: int | None = None,
        rating: int | None = None,
        strategy: SelectionStrategy = "argmax",
        temperature: float = DEFAULT_TEMPERATURE,
        seed: int | None = None,
    ) -> None:
        self._engine = engine
        self._rating = rating
        self._strategy: SelectionStrategy = strategy
        self._temperature = temperature
        # Its own generator rather than the module's, so that one seeded player's moves are
        # the same whatever else in the process is drawing random numbers meanwhile.
        self._rng = random.Random(seed)
        self._model = ModelDescription(
            run=run if run is not None else engine.run,
            checkpoint=checkpoint if checkpoint is not None else engine.step,
            rating=rating,
            strategy=strategy,
            # A temperature is what a sampled move was drawn at. An argmax player has none,
            # rather than one that was never used and would read as a setting of the game.
            temperature=temperature if strategy == "sample" else None,
        )

    @property
    def name(self) -> str:
        """Which weights these moves came from, which is what viewers and PGN are shown."""
        return f"{self._model.run} step {self._model.checkpoint}"

    @property
    def model(self) -> ModelDescription:
        """Which checkpoint plays these moves, and how it was asked to choose them."""
        return self._model

    async def choose_move(self, context: GameContext) -> PlayerMove:
        # In a thread, because a forward pass is arithmetic that holds the event loop for as
        # long as it takes: a small model is a millisecond and a large one is not, and the
        # other games and viewers this server is serving should not wait on either.
        evaluation = await asyncio.to_thread(
            self._engine.evaluate,
            context.board,
            mover_rating=self._rating,
            opponent_rating=self._rating,
        )
        move = select_move(
            evaluation,
            strategy=self._strategy,
            temperature=self._temperature,
            rng=self._rng,
        )
        return PlayerMove(move, thoughts=_thoughts(evaluation))


def _thoughts(evaluation: Evaluation) -> Thoughts:
    """What the model was considering, for the browser to show over the board."""
    return Thoughts(
        candidates=[
            CandidateMove(uci=move.uci(), probability=probability)
            for move, probability in evaluation.top(TOP_CANDIDATES)
        ],
        wdl=evaluation.wdl,
    )
