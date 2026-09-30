"""The inference engine: a checkpoint, a position, and what the network makes of it.

Three things come out of one forward pass, and the reason they are separated is the reason this
module exists at all.

- *A distribution over legal moves.* The policy head scores every entry of the move vocabulary,
  legal or not, because nothing ever told it the rules. The mask is applied **before**
  normalizing, so what comes out is a probability distribution over the moves that can actually
  be played rather than a truncated one that sums to less than one.
- *A win/draw/loss estimate*, from the point of view of the side to move, which is how the
  dataset records a result and therefore what the value head was trained to say.
- *The probability mass the raw network put on illegal moves*, which masking would otherwise
  hide. It is the same thing the validation metrics watch, measured here on the positions
  actually being played.

Choosing which move to play is not in here; see :mod:`chess_ai.inference.selection`. That split
is what lets a search sit between the network and the move later on without anything else
changing.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import chess
import numpy as np
import torch

from chess_ai.dataset import Result
from chess_ai.encoders import Encoder, EncoderSpec, create_encoder
from chess_ai.models import ChessModel, create_model
from chess_ai.move_codec import VOCABULARY, legal_mask
from chess_ai.players import WinDrawLoss
from chess_ai.registry import RegistryError
from chess_ai.training import checkpoint
from chess_ai.training.checkpoint import CheckpointError
from chess_ai.training.hardware import HardwareError, resolve_device

DEFAULT_BATCH_SIZE: Final = 32
"""How many positions go through the network at once when several are asked about.

Small on purpose. Playing needs one position at a time, and the machine that plays is usually
the machine that is training: a batch that filled the card would be a batch that stopped a
two-day run. Throughput here is worth nothing next to that.
"""


class InferenceError(Exception):
    """A checkpoint cannot be played with, and why."""


class DeviceUnavailableError(InferenceError):
    """The device a checkpoint was to be played on is not there.

    Its own kind of failure because it is nobody's fault but the machine's: everything else
    here is something about the checkpoint that was asked for, and this is the configuration
    of the computer it was asked on. What answers a request decides what to say by it.
    """


@dataclass(frozen=True)
class Evaluation:
    """What the network makes of one position.

    ``moves`` holds every legal move with its probability, best first and ties broken by the
    move itself, so that reading the best move off the front is deterministic for a given set of
    weights. The probabilities sum to one across the legal moves: the illegal ones were masked
    out before normalizing, and what they had been given is ``illegal_mass``.
    """

    moves: tuple[tuple[chess.Move, float], ...]
    wdl: WinDrawLoss
    illegal_mass: float
    """What the raw policy put on moves that cannot be played here, before masking.

    Masking makes every move legal, so nothing downstream can tell a network that knows the
    rules from one that does not. This is that difference, per position: 1.0 from a network
    that has learned nothing, and a few percent from one that has.
    """

    @property
    def best(self) -> chess.Move:
        """The most probable legal move."""
        return self.moves[0][0]

    def probabilities(self) -> np.ndarray:
        """The probabilities alone, in ``moves``' order, for a selection strategy to work on."""
        return np.array([probability for _, probability in self.moves], dtype=np.float64)

    def top(self, count: int) -> tuple[tuple[chess.Move, float], ...]:
        """The ``count`` most probable legal moves, or all of them if there are fewer."""
        return self.moves[:count]


class InferenceEngine:
    """One loaded checkpoint, asked about positions.

    Holds the model in evaluation mode on one device and never trains it. Asking about several
    positions at once is a batch through the network; asking about one is a batch of one, which
    is what a game does on every move.
    """

    def __init__(
        self,
        model: ChessModel,
        encoder: Encoder,
        *,
        device: torch.device | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        step: int = 0,
        run: str = "",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {batch_size}")
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = batch_size
        self.run = run
        """The run the weights came from, as the checkpoint recorded it."""
        self.step = step
        """The step the checkpoint was saved at."""
        self._model = model.to(self.device).eval()
        self._encoder = encoder

    @property
    def spec(self) -> EncoderSpec:
        """The input the model expects, which is the encoder's contract with it."""
        return self._model.spec

    def evaluate(
        self,
        board: chess.Board,
        *,
        mover_rating: int | None = None,
        opponent_rating: int | None = None,
    ) -> Evaluation:
        """What the network makes of ``board``, for the side to move.

        Raises:
            ValueError: there is no move to be made in the position, so there is nothing to
                distribute probability over.
        """
        return self.evaluate_many(
            [board], mover_rating=mover_rating, opponent_rating=opponent_rating
        )[0]

    def evaluate_many(
        self,
        boards: Sequence[chess.Board],
        *,
        mover_rating: int | None = None,
        opponent_rating: int | None = None,
    ) -> list[Evaluation]:
        """Evaluate several positions, in batches of at most :attr:`batch_size`.

        The ratings are the same for every position, which is what a game or a suite asks for:
        one model playing at one strength. Every position is read from the side to move's point
        of view, so ``mover_rating`` is that side's whichever colour it is.
        """
        evaluations: list[Evaluation] = []
        for start in range(0, len(boards), self.batch_size):
            batch = boards[start : start + self.batch_size]
            policy, value = self._forward(batch, mover_rating, opponent_rating)
            evaluations.extend(
                _evaluation(board, self._encoder, policy[index], value[index])
                for index, board in enumerate(batch)
            )
        return evaluations

    @torch.no_grad()
    def _forward(
        self,
        boards: Sequence[chess.Board],
        mover_rating: int | None,
        opponent_rating: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One batch through the network, as float32 policy and value logits back on the CPU."""
        bundles = [
            self._encoder.encode_board(
                board, mover_rating=mover_rating, opponent_rating=opponent_rating
            )
            for board in boards
        ]
        spatial = torch.from_numpy(np.concatenate([bundle.spatial for bundle in bundles]))
        globals_ = torch.from_numpy(np.concatenate([bundle.globals for bundle in bundles]))
        out = self._model(spatial.to(self.device).float(), globals_.to(self.device))
        return out.policy.float().cpu().numpy(), out.value.float().cpu().numpy()


def _evaluation(
    board: chess.Board, encoder: Encoder, policy: np.ndarray, value: np.ndarray
) -> Evaluation:
    """One position's logits turned into what the caller asked about."""
    legal_moves = np.flatnonzero(legal_mask(board))
    if not len(legal_moves):
        raise ValueError(f"no legal move in {board.fen()}, so there is nothing to choose from")
    # Where the model keeps each legal move's logit. An encoder that turned the board around
    # turned the moves with it, and this is where they are turned back: everything from here
    # on is about moves on the real board.
    mask = encoder.model_moves(legal_moves, board.turn == chess.WHITE)
    # Two softmaxes rather than one and a division. The masked one is taken over the legal
    # logits alone, so it sums to one however little the network thought of all of them; the
    # unmasked one is only ever read for the mass that fell outside the mask.
    legal = _softmax(policy[mask])
    illegal = float(np.clip(1.0 - _softmax(policy)[mask].sum(), 0.0, 1.0))
    moves = sorted(
        zip((VOCABULARY[index] for index in legal_moves), legal.tolist(), strict=True),
        # Descending by probability, then by the move itself, so that two equally likely moves
        # are always ranked the same way and an argmax player is reproducible.
        key=lambda pair: (-pair[1], pair[0].uci()),
    )
    return Evaluation(moves=tuple(moves), wdl=_wdl(value), illegal_mass=illegal)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """``logits`` as probabilities, in float64 and shifted so that nothing overflows."""
    shifted = np.exp(logits.astype(np.float64) - logits.max())
    return shifted / shifted.sum()


def _wdl(value: np.ndarray) -> WinDrawLoss:
    """The value head's three logits as probabilities, in :class:`Result`'s order."""
    probabilities = _softmax(value)
    return WinDrawLoss(
        win=float(probabilities[Result.WIN]),
        draw=float(probabilities[Result.DRAW]),
        loss=float(probabilities[Result.LOSS]),
    )


def load_engine(
    path: Path,
    *,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> InferenceEngine:
    """Load the checkpoint at ``path`` and build an engine that plays with it.

    A checkpoint is enough on its own: it carries the architecture, its hyperparameters and the
    encoder spec, so nothing about the run that produced it has to still be on the disk. The
    optimizer state travels with it too and is two thirds of the file; it is dropped here rather
    than held, because nothing that plays will ever read it.

    ``device`` is ``cpu``, ``cuda`` or ``auto``, as a training config's is. It defaults to the
    CPU, because the machine that plays is usually the machine that is training and a game is
    one small position at a time.

    Raises:
        InferenceError: the file cannot be read, is not a checkpoint this code understands, or
            describes a model or encoder this code can no longer build.
        DeviceUnavailableError: ``device`` names a device this machine does not have.
    """
    try:
        resolved = resolve_device(device)
    except HardwareError as e:
        raise DeviceUnavailableError(str(e)) from e
    try:
        # Onto the CPU whatever the device is: a checkpoint saved from a GPU that is busy
        # training should not be loaded straight back onto it, and the model is moved once.
        payload = checkpoint.load(path, map_location="cpu")
    except CheckpointError as e:
        raise InferenceError(str(e)) from e
    # The weights are wanted and the optimizer state is not. Dropped before the model is built,
    # so that the two are never both in memory.
    payload.pop("optimizer_state", None)
    # The encoder first: when its shapes have moved, saying so is worth more than the size
    # mismatch that loading the weights would raise a moment later.
    encoder = build_encoder(payload)
    model = build_model(payload)
    return InferenceEngine(
        model,
        encoder,
        device=resolved,
        batch_size=batch_size,
        run=str(payload.get("run", "")),
        step=int(payload.get("step", 0)),
    )


def build_model(payload: dict[str, Any]) -> ChessModel:
    """The architecture a checkpoint holds, built and loaded with its weights."""
    spec = _spec(payload)
    architecture = payload.get("architecture")
    if not isinstance(architecture, str):
        raise InferenceError("the checkpoint does not say which architecture it was trained on")
    options = payload.get("model_options") or {}
    try:
        model = create_model(architecture, spec, **options)
    except (RegistryError, TypeError, ValueError) as e:
        raise InferenceError(
            f"cannot build {architecture!r} as the checkpoint describes it: {e}"
        ) from e
    try:
        model.load_state_dict(payload["model_state"])
    except (KeyError, RuntimeError) as e:
        raise InferenceError(f"the checkpoint's weights do not fit a {architecture!r}: {e}") from e
    return model.eval()


def build_encoder(payload: dict[str, Any]) -> Encoder:
    """The encoder a checkpoint was trained against, built as it was built then.

    The spec that comes back out of the rebuilt encoder has to be the one recorded in the
    checkpoint. If it is not, the encoder's code has changed since the run and would feed these
    weights something they were never trained on — which is worth saying rather than playing.
    """
    spec = _spec(payload)
    try:
        encoder = create_encoder(spec.encoder, **spec.options)
    except (RegistryError, TypeError, ValueError) as e:
        raise InferenceError(
            f"cannot build the {spec.encoder!r} encoder the checkpoint used: {e}"
        ) from e
    if encoder.spec != spec:
        raise InferenceError(
            f"the {spec.encoder!r} encoder is not what this checkpoint was trained on: "
            + "; ".join(_drift(spec, encoder.spec))
        )
    return encoder


def _drift(trained: EncoderSpec, now: EncoderSpec) -> list[str]:
    """What has changed about an encoder since a checkpoint was trained against it."""
    return [
        f"{field} was {getattr(trained, field)!r} and is now {getattr(now, field)!r}"
        for field in EncoderSpec.model_fields
        if getattr(trained, field) != getattr(now, field)
    ]


def _spec(payload: dict[str, Any]) -> EncoderSpec:
    try:
        return checkpoint.spec_of(payload)
    except CheckpointError as e:
        raise InferenceError(str(e)) from e
