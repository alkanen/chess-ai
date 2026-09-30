"""Validation metrics: how well the model predicts held-back games, and how legal it is.

Five numbers, measured on the same positions every time so that a curve across steps means
something:

- *policy loss* and *value loss*, the two halves of what is being minimized
- *top-1* and *top-5*, how often the move actually played is the model's first or top-five guess,
  which is the thing a move-prediction model is for and the number that compares to published
  work
- *illegal top move*, how often the model's own best move cannot be played at all

The last one is here because the policy is trained and scored unmasked. Nothing tells the network
the rules; it has to learn from the data that a bishop does not move like a rook, and watching
this fall from most of the time to almost never is the clearest early sign that it is learning
anything about chess rather than about move frequencies. At play time the distribution is masked
down to legal moves, so a high rate costs strength rather than legality — which is exactly why it
needs measuring separately.
"""

import numpy as np
import torch
from torch.nn import functional as F

from chess_ai.dataset import SplitReader, white_to_move
from chess_ai.dataset.records import unpack_board
from chess_ai.encoders import Encoder
from chess_ai.models import ChessModel
from chess_ai.move_codec import move_at

TOP_K = 5
"""How far down the model's ranking the played move still counts, for the top-5 metric."""


def validate(
    model: ChessModel,
    split: SplitReader,
    encoder: Encoder,
    *,
    positions: int,
    batch_size: int,
    device: torch.device,
    value_loss_weight: float,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Measure ``model`` on the first ``positions`` positions of ``split``.

    The front of the split rather than a sample of it, because the split is already a random
    selection — the dataset assigns whole games to it by a hash of the game — and taking the same
    prefix every time makes two validations of the same model give the same answer.
    """
    total = min(positions, len(split))
    if not total:
        raise ValueError("cannot validate on an empty split")
    was_training = model.training
    model.eval()
    sums = {"policy_loss": 0.0, "value_loss": 0.0}
    counts = {"top1": 0, "top5": 0, "illegal_top_move": 0}
    try:
        for start in range(0, total, batch_size):
            frames = split.position_history(
                np.arange(start, min(start + batch_size, total)), encoder.history
            )
            records = frames[:, 0]
            played, policy, value = _forward(
                model, encoder, frames, device=device, dtype=autocast_dtype
            )
            result = torch.as_tensor(records["result"].astype(np.int64), device=policy.device)
            sums["policy_loss"] += _total_loss(policy, played)
            sums["value_loss"] += _total_loss(value, result)
            ranked = policy.topk(TOP_K, dim=1).indices
            counts["top1"] += int((ranked[:, 0] == played).sum())
            counts["top5"] += int((ranked == played.unsqueeze(1)).any(dim=1).sum())
            top_moves = encoder.board_moves(ranked[:, 0].cpu().numpy(), white_to_move(records))
            counts["illegal_top_move"] += _illegal(records, top_moves.tolist())
    finally:
        model.train(was_training)

    policy_loss = sums["policy_loss"] / total
    value_loss = sums["value_loss"] / total
    return {
        "loss": policy_loss + value_loss_weight * value_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "top1": counts["top1"] / total,
        "top5": counts["top5"] / total,
        "illegal_top_move_rate": counts["illegal_top_move"] / total,
        "positions": float(total),
    }


@torch.no_grad()
def _forward(
    model: ChessModel,
    encoder: Encoder,
    frames: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The played moves and the model's two outputs for ``frames``, as float32 logits.

    The played moves are in the model's own terms, as its policy is, so the two can be compared
    directly whichever way the encoder turned the board.
    """
    records = frames[:, 0]
    bundle = encoder.encode(frames)
    spatial = torch.from_numpy(bundle.spatial).to(device).float()
    globals_ = torch.from_numpy(bundle.globals).to(device)
    played = torch.as_tensor(
        encoder.model_moves(records["move"], white_to_move(records)), device=device
    )
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        out = model(spatial, globals_)
    # Back to float32 before the losses and the ranking: a bfloat16 logit has eight bits of
    # mantissa, which is enough to train with and not enough to compare two close moves by.
    return played, out.policy.float(), out.value.float()


def _total_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Summed cross-entropy, so that batches of different sizes can be added together."""
    return float(F.cross_entropy(logits, targets, reduction="sum"))


def _illegal(records: np.ndarray, top_moves: list[int]) -> int:
    """How many of ``top_moves`` cannot be played in the position they were predicted for.

    ``top_moves`` are moves on the real board, which is where the rules apply.

    The one place validation leaves numpy for python-chess. Only the top move is checked rather
    than the whole distribution, which turns generating every legal move into a single legality
    question per position.
    """
    illegal = 0
    for record, index in zip(records, top_moves, strict=True):
        if not unpack_board(record).is_legal(move_at(index)):
            illegal += 1
    return illegal
