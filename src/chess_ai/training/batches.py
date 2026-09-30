"""Getting batches out of a dataset and into a model, encoded, shuffled and on the device.

Batched rather than shuffled per position, which is the one unusual thing here. A torch
``Dataset`` normally hands back one example and lets the loader collate them; this one hands back
a whole batch, because encoding is vectorised numpy over a batch of records and doing it a
position at a time would spend most of the run in Python. Gathering scattered records out of
memory-mapped shards is also cheaper in one call than in a thousand.

An index is a batch number over the *whole run*, not within an epoch, so the run needs one loader
rather than one per epoch. That matters at both ends of the scale. A dataset smaller than a few
batches would otherwise start a new set of worker processes every epoch — thousands of times over
a run, each fork costing more than the epoch it serves. And a dataset of tens of millions of
positions never reaches a second epoch in the first place.

Shuffling is a permutation per epoch, derived from the run's seed and the epoch number, so every
worker process computes the same one without being told and the same seed replays the same run.
It is built as the narrowest integer type that can index the split, because at Lichess scale each
worker holds one: 4 bytes a position rather than 8 is the difference between a gigabyte and two.
"""

import os
from typing import Any, Final, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from chess_ai.dataset import Dataset, SplitReader, white_to_move
from chess_ai.encoders import Encoder, create_encoder

PREFETCH_BATCHES: Final = 4
"""Batches each worker reads ahead, so that the GPU is not waiting on the disk."""


class Batch(NamedTuple):
    """One batch of encoded input and the targets that go with it."""

    spatial: torch.Tensor
    """(batch, channels, board, board) ``float32``."""
    globals: torch.Tensor
    """(batch, features) ``float32``."""
    move: torch.Tensor
    """(batch,) ``int64`` indices into the move vocabulary: the move actually played, as the
    encoder shows moves to the model; see :meth:`~chess_ai.encoders.Encoder.model_moves`."""
    result: torch.Tensor
    """(batch,) ``int64`` :class:`~chess_ai.dataset.Result`, from the mover's point of view."""

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "Batch":
        """This batch on ``device``."""
        return Batch(*(tensor.to(device, non_blocking=non_blocking) for tensor in self))

    def __len__(self) -> int:
        return len(self.move)


class PositionBatches(TorchDataset):
    """One split's positions as encoded batches, reshuffled at every epoch boundary.

    Opens the dataset lazily, once per process, so that the loader's worker processes each get
    their own memory maps whether they were forked or spawned. Nothing here is held across a fork
    that would be wrong on the other side of it.
    """

    def __init__(
        self,
        directory,
        split: str,
        *,
        encoder: str,
        encoder_options: dict[str, Any] | None = None,
        batch_size: int,
        batches: int,
        seed: int = 0,
    ) -> None:
        self._directory = directory
        self._split = split
        self._encoder_name = encoder
        self._encoder_options = dict(encoder_options or {})
        self.batch_size = batch_size
        self._batches = batches
        self._seed = seed
        self.positions = Dataset(directory).manifest.splits[split].positions
        if not self.positions:
            raise ValueError(f"the {split!r} split of {directory} has no positions")
        # A short last batch is dropped only when there is a full one to keep: a split smaller
        # than one batch still has to be trainable, which is what a test dataset is.
        self.batches_per_epoch = self.positions // batch_size or 1
        self._opened_by: int | None = None
        self._reader: SplitReader | None = None
        self._dataset: Dataset | None = None
        self._encoder: Encoder | None = None
        self._shuffled: tuple[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return self._batches

    def __getitem__(self, index: int) -> Batch:
        epoch, within = divmod(index, self.batches_per_epoch)
        start = within * self.batch_size
        order = self._order(epoch)[start : start + self.batch_size]
        encoder = self._position_encoder()
        frames = self._split_reader().position_history(order, encoder.history)
        records = frames[:, 0]
        bundle = encoder.encode(frames)
        return Batch(
            spatial=torch.from_numpy(bundle.spatial),
            globals=torch.from_numpy(bundle.globals),
            move=torch.from_numpy(encoder.model_moves(records["move"], white_to_move(records))),
            result=torch.from_numpy(records["result"].astype(np.int64)),
        )

    def _order(self, epoch: int) -> np.ndarray:
        """The order ``epoch`` visits the split in, shuffled from the seed and the epoch.

        One epoch's worth is kept at a time. A loader hands consecutive batches to its workers in
        turn, so each worker is always within a batch or two of the others and never needs two.
        """
        if self._shuffled is None or self._shuffled[0] != epoch:
            order = np.arange(self.positions, dtype=_index_dtype(self.positions))
            np.random.default_rng([self._seed, epoch]).shuffle(order)
            self._shuffled = (epoch, order)
        return self._shuffled[1]

    def close(self) -> None:
        """Drop everything a batch caused to be cached, leaving the object as it started.

        A loader pickles this object into each of its worker processes on any start method
        other than ``fork`` — macOS and Windows default to ``spawn``, and CPython 3.14 moves
        Linux to ``forkserver`` — so everything cached here is paid for once per worker. Two
        of those cost real memory at the scale this module is built for:

        - a :class:`numpy.memmap` pickles as the whole mapped file rather than as a mapping
        - a shuffled order is four bytes a position, which is 1.2 GB over a 300M-position
          split, and the worker keeps the copy rather than building its own — the memmap at
          least notices it arrived in another process and maps again

        So anything that takes a batch in the training process before the loader is built,
        such as the throughput probe, has to let go of all of it afterwards. Everything here
        is rebuilt on demand, so a batch asked for later maps and shuffles again.
        """
        if self._reader is not None:
            self._reader.close()
        self._reader = self._dataset = self._opened_by = None
        self._shuffled = self._encoder = None

    def _split_reader(self) -> SplitReader:
        """This process's own reader, opened the first time it asks for one."""
        if self._reader is None or self._opened_by != os.getpid():
            self._dataset = Dataset(self._directory)
            self._reader = self._dataset[self._split]
            self._opened_by = os.getpid()
        return self._reader

    def _position_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = create_encoder(self._encoder_name, **self._encoder_options)
        return self._encoder


def _index_dtype(positions: int) -> np.dtype:
    """The narrowest unsigned type that can index ``positions`` records."""
    for candidate in (np.uint16, np.uint32):
        if positions <= np.iinfo(candidate).max:
            return np.dtype(candidate)
    return np.dtype(np.uint64)


def batch_loader(batches: PositionBatches, *, workers: int, device: torch.device) -> DataLoader:
    """A loader over ``batches``, reading ahead in ``workers`` processes.

    ``batch_size=None`` turns off the loader's own batching: each item already is a batch, so
    there is nothing to collate. Memory is pinned for a GPU, which is what makes the copy to the
    device overlap with the next batch being read.
    """
    return DataLoader(
        batches,
        batch_size=None,
        shuffle=False,  # The dataset shuffles; see PositionBatches._order.
        num_workers=workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=PREFETCH_BATCHES if workers else None,
    )
