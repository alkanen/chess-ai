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
It is built as the narrowest integer type that can index the split: 4 bytes a position, which at
Lichess scale is 2.5 GB for an order. Held once per worker, as it was at first, that is 10 GB over
four of them for four copies of the same array, and twice that at an epoch boundary. So a loader
is given a directory to share it through: the first worker into an epoch writes its order to a
file there, and every worker maps that file. It is then page cache held once for all of them,
which the kernel can drop and read back rather than swap; see
:meth:`PositionBatches.sharing_orders`.
"""

import copy
import fcntl
import multiprocessing
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import resource_tracker
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from chess_ai.dataset import Dataset, SplitReader, white_to_move
from chess_ai.encoders import Encoder, create_encoder

PREFETCH_BATCHES: Final = 4
"""Batches each worker reads ahead, so that the GPU is not waiting on the disk."""

ORDER_SUFFIX: Final = ".order"
"""What an epoch's order is called in a directory orders are shared through."""
WRITING_SUFFIX: Final = ".writing"
"""What an order is called while it is being written, until it is whole."""
ORDER_LOCK: Final = "orders.lock"
"""Held by the worker writing an order, so that only one does and the rest wait for it."""
_FILL_CHUNK: Final = 1 << 24
"""Positions an order file is numbered in at a time, before it is shuffled, so that numbering
it never needs an array of the whole split in memory."""


class Batch(NamedTuple):
    """One batch of encoded input and the targets that go with it."""

    spatial: torch.Tensor
    """(batch, channels, board, board) ``uint8`` until :meth:`to` moves it, ``float32`` after.

    See :attr:`~chess_ai.encoders.InputBundle.spatial` for why it travels as bytes."""
    globals: torch.Tensor
    """(batch, features) ``float32``."""
    move: torch.Tensor
    """(batch,) ``int64`` indices into the move vocabulary: the move actually played, as the
    encoder shows moves to the model; see :meth:`~chess_ai.encoders.Encoder.model_moves`."""
    result: torch.Tensor
    """(batch,) ``int64`` :class:`~chess_ai.dataset.Result`, from the mover's point of view."""

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "Batch":
        """This batch on ``device``, with the planes as the floats a model reads.

        Converted after the copy rather than before, so that what crosses to the device is the
        quarter-size bytes and the conversion runs there.
        """
        spatial, globals_, move, result = (
            tensor.to(device, non_blocking=non_blocking) for tensor in self
        )
        return Batch(spatial.float(), globals_, move, result)

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
        version: int | None = None,
    ) -> None:
        """``version`` is the dataset's version to read, the latest when not given."""
        self._directory = directory
        self._version = version
        self._split = split
        self._encoder_name = encoder
        self._encoder_options = dict(encoder_options or {})
        self.batch_size = batch_size
        self._batches = batches
        self._seed = seed
        # The positions training draws from, which are the split's training targets: every
        # position, unless the dataset was built with a filter on positions.
        self.positions = Dataset(directory, version).manifest.splits[split].trained_on
        if not self.positions:
            raise ValueError(f"the {split!r} split of {directory} has no positions")
        # A short last batch is dropped only when there is a full one to keep: a split smaller
        # than one batch still has to be trainable, which is what a test dataset is.
        self.batches_per_epoch = self.positions // batch_size or 1
        self.positions_per_batch = min(batch_size, self.positions)
        """How many positions every batch has: the batch size, or the whole split if it is less."""
        self._opened_by: int | None = None
        self._reader: SplitReader | None = None
        self._dataset: Dataset | None = None
        self._encoder: Encoder | None = None
        self._shuffled: tuple[int, np.ndarray] | None = None
        self._orders: Path | None = None

    def sharing_orders(self, directory: Path) -> "PositionBatches":
        """These batches, with each epoch's order shared through files in ``directory``.

        Every process asking for a batch of an epoch maps the same file, written once by
        whichever got there first while the others wait on a lock, so the order is held once
        however many workers there are. The order itself is the one an unshared object
        shuffles, number for number, so a run started before orders were shared resumes with
        the batches it would have had.

        Writing an epoch's order removes those older than the epoch before it, which a worker
        a batch behind may still be reading, and whatever a writer that died left
        half-written; never a newer one, which with epochs of a few batches a worker ahead
        may be on. The directory is the caller's to remove once the workers are done with it.

        A copy, holding none of what this one cached, so that none of it is pickled into
        workers; this one is left as it is.
        """
        shared = copy.copy(self)
        shared._forget()
        shared._orders = Path(directory)
        return shared

    def __len__(self) -> int:
        return self._batches

    def __getitem__(self, index: int) -> Batch:
        epoch, within = divmod(index, self.batches_per_epoch)
        start = within * self.batch_size
        reader = self._split_reader()
        order = reader.target_positions(self._order(epoch)[start : start + self.batch_size])
        encoder = self._position_encoder()
        frames = reader.position_history(order, encoder.history)
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
        The last epoch's is let go before the next is built, rather than when it replaces it:
        at Lichess scale holding both is another 2.5 GB in every worker at every boundary.
        """
        if self._shuffled is None or self._shuffled[0] != epoch:
            self._shuffled = None
            if self._orders is None:
                order = _shuffle(np.arange(self.positions, dtype=self._dtype()), self._seed, epoch)
            else:
                order = self._shared_order(epoch)
            self._shuffled = (epoch, order)
        return self._shuffled[1]

    def _shared_order(self, epoch: int) -> np.ndarray:
        """``epoch``'s order mapped from its file, written first if no process has yet."""
        assert self._orders is not None
        path = self._orders / self._order_name(epoch)
        order = self._mapped(path)
        if order is not None:
            return order
        self._orders.mkdir(parents=True, exist_ok=True)
        with _locked(self._orders / ORDER_LOCK):
            # Looked for again: another worker may have written it while this one waited.
            order = self._mapped(path)
            if order is not None:
                return order
            for stale in self._orders.iterdir():
                # Only the lock's holder writes, so a half-written order is a dead writer's.
                # Older epochs only: with epochs of a few batches, a worker that is behind can
                # be writing one while the others are more than an epoch ahead of it.
                stale_epoch = self._epoch_of(stale.name)
                if stale.name.endswith(WRITING_SUFFIX) or (
                    stale_epoch is not None and stale_epoch < epoch - 1
                ):
                    stale.unlink(missing_ok=True)
            writing = path.with_name(path.name + WRITING_SUFFIX)
            try:
                self._write_order(writing, epoch)
            except BaseException:
                writing.unlink(missing_ok=True)
                raise
            os.replace(writing, path)
            order = self._mapped(path)
        assert order is not None, f"{path} was written and is not there"
        return order

    def _write_order(self, path: Path, epoch: int) -> None:
        """Write ``epoch``'s order to ``path``, whole and on the disk."""
        size = self.positions * self._dtype().itemsize
        with path.open("wb") as file:
            # Room for all of it first. A mapped file is otherwise sparse, and a page the
            # disk has no block for kills the process with SIGBUS rather than raising.
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(file.fileno(), 0, size)
            else:
                file.truncate(size)
        order = np.memmap(path, dtype=self._dtype(), mode="r+", shape=(self.positions,))
        for start in range(0, self.positions, _FILL_CHUNK):
            stop = min(start + _FILL_CHUNK, self.positions)
            order[start:stop] = np.arange(start, stop, dtype=order.dtype)
        # Shuffled through a plain view of the map: numpy only shuffles a base-class array in
        # place in C, and swaps a subclass's elements one at a time in Python.
        _shuffle(np.asarray(order), self._seed, epoch)
        order.flush()

    def _order_name(self, epoch: int) -> str:
        return f"{self._split}-seed{self._seed}-epoch{epoch}{ORDER_SUFFIX}"

    def _epoch_of(self, name: str) -> int | None:
        """The epoch an order of this split and seed called ``name`` is of, if it is one."""
        prefix = f"{self._split}-seed{self._seed}-epoch"
        if not (name.startswith(prefix) and name.endswith(ORDER_SUFFIX)):
            return None
        number = name[len(prefix) : -len(ORDER_SUFFIX)]
        return int(number) if number.isdigit() else None

    def _mapped(self, path: Path) -> np.ndarray | None:
        """The order in ``path``, or ``None`` if there is no whole one there."""
        try:
            if path.stat().st_size != self.positions * self._dtype().itemsize:
                return None
            # The lock's holder may remove an old order between the look and the open.
            return np.memmap(path, dtype=self._dtype(), mode="r", shape=(self.positions,))
        except FileNotFoundError:
            return None

    def _dtype(self) -> np.dtype:
        return _index_dtype(self.positions)

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
        self._forget()

    def _forget(self) -> None:
        """Let go of everything cached, without closing what another object may share."""
        self._reader = self._dataset = self._opened_by = None
        self._shuffled = self._encoder = None

    def _split_reader(self) -> SplitReader:
        """This process's own reader, opened the first time it asks for one."""
        if self._reader is None or self._opened_by != os.getpid():
            self._dataset = Dataset(self._directory, self._version)
            self._reader = self._dataset[self._split]
            self._opened_by = os.getpid()
        return self._reader

    def _position_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = create_encoder(self._encoder_name, **self._encoder_options)
        return self._encoder


def _shuffle(order: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """``order`` shuffled in place into ``epoch``'s order, and returned."""
    np.random.default_rng([seed, epoch]).shuffle(order)
    return order


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` for the block, waiting while another process has it."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _index_dtype(positions: int) -> np.dtype:
    """The narrowest unsigned type that can index ``positions`` records."""
    for candidate in (np.uint16, np.uint32):
        if positions <= np.iinfo(candidate).max:
            return np.dtype(candidate)
    return np.dtype(np.uint64)


def batch_loader(batches: TorchDataset[Batch], *, workers: int, device: torch.device) -> DataLoader:
    """A loader over ``batches``, reading ahead in ``workers`` processes.

    ``batch_size=None`` turns off the loader's own batching: each item already is a batch, so
    there is nothing to collate. Memory is pinned for a GPU, which is what makes the copy to the
    device overlap with the next batch being read.

    The loader gets a random generator of its own. It draws a seed for its workers every time
    it starts, and drawing that from torch's global generator would move the stream that
    dropout draws from — by a different amount in a run that resumed than in one that did not.
    """
    return DataLoader(
        batches,
        batch_size=None,
        shuffle=False,  # The dataset shuffles; see PositionBatches._order.
        num_workers=workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=PREFETCH_BATCHES if workers else None,
        worker_init_fn=_leave_stopping_to_the_trainer if workers else None,
        generator=torch.Generator(),
    )


def _leave_stopping_to_the_trainer(worker: int) -> None:
    """Make a worker process deaf to the signals that ask a run to stop.

    Ctrl-C in a terminal goes to every process in its group, and a service manager stopping a
    run sends SIGTERM to all of its processes too. The trainer answers either by finishing its
    step and saving a checkpoint, but a worker would simply die — and the loader reports a dead
    worker by raising in the trainer, at whatever it is doing, which may be writing that very
    checkpoint. The workers end when the trainer lets go of the loader, or when they notice it
    has gone.

    A worker started by :func:`iterate` was born with them held back; ignored first and only
    then let through, whatever arrived while it started is dropped here rather than acted on.
    """
    for number in STOP_SIGNALS:
        signal.signal(number, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


STOP_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)
"""The signals that ask a run to stop, which are the trainer's to answer and no worker's."""


def iterate(loader: DataLoader) -> Iterator[Batch]:
    """Every batch ``loader`` gives, from workers that a stop signal neither kills nor strands.

    The workers ignore the signals that stop a run (see :func:`_leave_stopping_to_the_trainer`),
    which has two consequences this takes care of:

    - A worker only gets to ignore them once it runs its ``worker_init_fn``. Under ``spawn``
      and ``forkserver`` that is seconds after it starts, spent importing in a fresh
      interpreter with the default handlers, and those seconds are the first of every run. So
      the workers are started with the signals blocked, which a child inherits through fork
      and exec alike, and they stay held back until the worker ignores them.
    - The loader's last resort for a worker that will not stop is SIGTERM, which they now
      ignore. A second ctrl-c while the loader shuts down can skip the message that tells
      them to go, and the trainer would then wait on its way out for workers that are waiting
      for it. So whatever is still alive once the loader has shut down is killed outright.
    """
    if not loader.num_workers:
        yield from loader
        return
    if _start_method(loader) != "fork":
        # Started now if it is not already, because how it is started is the one thing that
        # could undo what follows: Python 3.12.0 to 3.12.9 and 3.13.0 start it by blocking the
        # stop signals and then unblocking them, whatever the caller had blocked, and the
        # loader's queues start it the first time a process makes any.
        resource_tracker.ensure_running()
    with holding_stop_signals():
        before = set(multiprocessing.active_children())
        iterator = iter(loader)
        workers = [child for child in multiprocessing.active_children() if child not in before]
    try:
        yield from iterator
    finally:
        try:
            # The loader's own shutdown, which runs when its iterator goes.
            del iterator
        finally:
            with holding_stop_signals():
                _kill_stragglers(workers)


def _start_method(loader: DataLoader) -> str:
    """How ``loader`` starts its workers: ``fork``, ``spawn`` or ``forkserver``."""
    context = loader.multiprocessing_context
    return (context or multiprocessing.get_context()).get_start_method()


@contextmanager
def holding_stop_signals() -> Iterator[None]:
    """Hold the stop signals back until the block is done, from children and from this thread.

    Two holds, because a signal reaches two places:

    - **Children** started meanwhile inherit the signal mask, which blocks the signals in them
      until a worker ignores them itself. Blocked, never ignored, here: ignoring them would
      drop a signal that the kernel hands to another thread of this process — torch runs a
      pool of them once a model has run — and on macOS even in this one.
    - **This process's handlers** run in the main thread whichever thread the kernel handed
      the signal to, so blocking it in this thread does not keep them out. In the main thread,
      the handlers are swapped for ones that only note what arrived, and once the block is done
      the real ones are put back and given what was noted, in order. Elsewhere there is nothing
      to swap: Python neither runs handlers nor lets them be set off the main thread.

    A ``forkserver`` worker is a child of the fork server rather than of this process, and
    inherits what the server was started with. It is started by the first loader that needs
    one, which in a training process is this one, inside this block.
    """
    noted: list[int] = []
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    previous: dict[int, Any] = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for number in STOP_SIGNALS:
                # A handler set from C cannot be put back from Python, so it is left alone.
                if signal.getsignal(number) is not None:
                    previous[number] = signal.signal(number, lambda n, frame: noted.append(n))
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        for number in noted:
            signal.raise_signal(number)


def _kill_stragglers(workers: list[multiprocessing.process.BaseProcess]) -> None:
    """Kill the workers the loader's shutdown left alive."""
    for worker in workers:
        if worker.is_alive():
            worker.kill()
            worker.join()
