"""Batches: what order positions are visited in, and that a worker process sees the same one."""

import errno
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from signal_helpers import SignalledWhileStarting
from training_helpers import dataset

from chess_ai.dataset import Result, open_dataset
from chess_ai.dataset.manifest import Filters
from chess_ai.encoders import create_encoder
from chess_ai.move_codec import MIRRORED_INDEX, VOCABULARY_SIZE
from chess_ai.training import batches as batches_module
from chess_ai.training.batches import PositionBatches, batch_loader, iterate


@pytest.fixture
def data_dir(tmp_path):
    dataset(tmp_path / "data")
    return tmp_path / "data"


@pytest.fixture
def directory(data_dir):
    with open_dataset("test", data_dir=data_dir) as built:
        return built.directory


def batches(directory, **options) -> PositionBatches:
    options.setdefault("batch_size", 4)
    options.setdefault("batches", 10)
    return PositionBatches(directory, "train", encoder="board-planes", **options)


def moves(source: PositionBatches, count: int) -> list[int]:
    """The played moves of the first ``count`` batches, which is the order positions came in."""
    return [int(move) for index in range(count) for move in source[index].move]


def test_a_batch_is_encoded_input_and_the_targets_that_go_with_it(directory):
    source = batches(directory)

    batch = source[0]

    spec = create_encoder("board-planes").spec
    assert batch.spatial.shape == (4, *spec.spatial_shape)
    assert batch.globals.shape == (4, spec.global_features)
    assert batch.move.shape == batch.result.shape == (4,)
    assert batch.move.dtype == batch.result.dtype == torch.int64
    assert len(batch) == 4
    assert all(0 <= int(move) < VOCABULARY_SIZE for move in batch.move)
    assert all(int(result) in set(Result) for result in batch.result)


def test_a_run_is_as_many_batches_as_it_has_steps(directory):
    assert len(batches(directory, batches=37)) == 37


def test_an_epoch_visits_no_position_twice(directory):
    source = batches(directory, batch_size=4)

    seen = [int(index) for index in source._order(0)[: source.batches_per_epoch * 4]]

    assert len(set(seen)) == len(seen) == source.batches_per_epoch * 4
    assert set(seen) <= set(range(source.positions))


def test_the_positions_an_epoch_drops_are_the_short_last_batch_only(directory):
    """A partial batch is dropped rather than padded, and a later epoch picks those positions up."""
    source = batches(directory, batch_size=4)

    dropped = source.positions - source.batches_per_epoch * 4

    assert dropped == source.positions % 4 < 4
    first = set(int(index) for index in source._order(0)[: source.batches_per_epoch * 4])
    second = set(int(index) for index in source._order(1)[: source.batches_per_epoch * 4])
    assert first != second, "a different few are left out each time round"


def test_the_next_epoch_visits_them_in_another_order(directory):
    source = batches(directory)

    assert not np.array_equal(source._order(0), source._order(1))
    assert sorted(source._order(1).tolist()) == sorted(source._order(0).tolist())


def test_the_same_seed_gives_the_same_order(directory):
    one = batches(directory, seed=5)
    same = batches(directory, seed=5)
    other = batches(directory, seed=6)

    assert moves(one, 3) == moves(same, 3)
    assert moves(one, 3) != moves(other, 3)


def test_the_order_is_the_same_whichever_batch_is_asked_for_first(directory):
    """Loader workers take batches in turn, so they must agree without talking to each other."""
    forwards = batches(directory)
    backwards = batches(directory)

    ahead = [int(move) for move in backwards[5].move]
    assert [int(move) for move in forwards[5].move] == ahead
    assert moves(forwards, 3) == moves(batches(directory), 3)


def test_a_split_smaller_than_one_batch_is_still_one_batch(directory):
    source = batches(directory, batch_size=10_000)

    assert source.batches_per_epoch == 1
    assert len(source[0]) == source.positions


def test_a_split_with_no_positions_is_refused(tmp_path):
    dataset(tmp_path / "empty", name="unsplit", validation_fraction=0.0)
    with (
        open_dataset("unsplit", data_dir=tmp_path / "empty") as built,
        pytest.raises(ValueError, match="'validation' split.*has no positions"),
    ):
        PositionBatches(
            built.directory, "validation", encoder="board-planes", batch_size=4, batches=1
        )


def test_a_loader_yields_exactly_the_run_and_nothing_to_collate(directory):
    source = batches(directory, batches=7)

    loaded = list(batch_loader(source, workers=0, device=torch.device("cpu")))

    assert len(loaded) == 7
    assert [len(batch) for batch in loaded] == [4] * 7


@pytest.mark.parametrize("workers", [0, 2])
def test_worker_processes_produce_the_same_batches_as_the_training_process(directory, workers):
    expected = moves(batches(directory, batches=6), 6)

    loaded = list(
        batch_loader(batches(directory, batches=6), workers=workers, device=torch.device("cpu"))
    )

    assert [int(move) for batch in loaded for move in batch.move] == expected


def test_a_batch_moves_to_a_device_whole(directory):
    batch = batches(directory)[0]

    moved = batch.to(torch.device("cpu"))

    assert all(tensor.device.type == "cpu" for tensor in moved)
    assert torch.equal(moved.move, batch.move)


def test_the_shuffle_is_indexed_as_narrowly_as_the_split_allows(directory):
    """At Lichess scale every worker holds one of these, so the dtype is worth a byte or two."""
    source = batches(directory)

    assert source._order(0).dtype == np.uint16, "a few hundred positions fit in two bytes"


def test_closing_leaves_nothing_of_the_split_to_pickle(directory):
    """What a loader pickles into a worker must not be the dataset, or the order over it.

    On a spawn or forkserver start method the DataLoader pickles this object into every
    worker. A cached `np.memmap` pickles as the whole mapped file rather than as a mapping,
    and a cached shuffle is four bytes a position — a gigabyte at Lichess scale, which the
    worker then keeps rather than building its own. Anything that takes a batch in the
    training process has to be able to let go of both.

    Measured against a freshly built object rather than against a fixed budget, so this holds
    for any field that gets cached here later, not just the two that do today.
    """
    untouched = len(pickle.dumps(batches(directory)))
    source = batches(directory)
    source[0]
    opened = len(pickle.dumps(source))

    source.close()

    closed = len(pickle.dumps(source))
    assert closed < opened / 2, f"{closed} bytes after closing, {opened} before"
    assert closed <= untouched, (
        f"closing left {closed - untouched} bytes behind that a new object does not carry"
    )
    assert len(source[0]) == 4, "and it maps and shuffles again when asked for another batch"


@pytest.mark.parametrize("shared", [False, True])
def test_the_last_epochs_order_is_let_go_before_the_next_is_built(
    directory, tmp_path, monkeypatch, shared
):
    """At Lichess scale an order is 2.5 GB, and a worker holding two is what ran out of memory."""
    source = batches(directory)
    if shared:
        source = source.sharing_orders(tmp_path / "orders")
    real = batches_module._shuffle
    previous: list[weakref.ref] = []
    still_held = []

    def shuffle(order, seed, epoch):
        still_held.extend(epoch for ref in previous if ref() is not None)
        return real(order, seed, epoch)

    monkeypatch.setattr(batches_module, "_shuffle", shuffle)
    for epoch in range(3):
        previous.append(weakref.ref(source._order(epoch)))

    assert not still_held, f"epochs {still_held} were built while the last one was held"


def test_a_shared_order_is_the_order_an_unshared_one_has(directory, tmp_path):
    """So a run started before orders were shared resumes with the batches it would have had."""
    alone = batches(directory, batches=400)
    shared = batches(directory, batches=400).sharing_orders(tmp_path / "orders")
    assert 2 * alone.batches_per_epoch < 400, "into a third epoch"

    for epoch in range(3):
        assert np.array_equal(shared._order(epoch), alone._order(epoch)), epoch
        assert shared._order(epoch).dtype == alone._order(epoch).dtype
    assert moves(shared, 400) == moves(alone, 400)


def test_processes_sharing_orders_write_each_once(directory, tmp_path, monkeypatch):
    calls = []
    real = batches_module._shuffle
    monkeypatch.setattr(
        batches_module, "_shuffle", lambda *args: calls.append(args[1:]) or real(*args)
    )
    one = batches(directory).sharing_orders(tmp_path / "orders")
    other = batches(directory).sharing_orders(tmp_path / "orders")

    assert moves(one, 3) == moves(other, 3)
    assert calls == [(0, 0)]
    assert isinstance(other._order(0), np.memmap), "mapped from the file, not built again"


def test_only_the_current_and_the_last_epochs_orders_are_kept(directory, tmp_path):
    """The last one for a worker a batch behind the rest, and nothing older to fill the disk."""
    source = batches(directory).sharing_orders(tmp_path / "orders")

    for epoch in range(4):
        source._order(epoch)

    assert sorted(path.name for path in (tmp_path / "orders").glob("*.order")) == [
        "train-seed0-epoch2.order",
        "train-seed0-epoch3.order",
    ]


def test_writing_an_order_leaves_alone_the_newer_ones_workers_ahead_are_on(directory, tmp_path):
    """With epochs of a few batches, a worker can be more than an epoch behind the others."""
    ahead = batches(directory).sharing_orders(tmp_path / "orders")
    behind = batches(directory).sharing_orders(tmp_path / "orders")
    ahead._order(3)

    behind._order(0)

    assert sorted(path.name for path in (tmp_path / "orders").glob("*.order")) == [
        "train-seed0-epoch0.order",
        "train-seed0-epoch3.order",
    ]


def test_an_order_removed_as_it_is_mapped_is_written_again(directory, tmp_path, monkeypatch):
    """The lock's holder may remove an old order between a worker finding it and mapping it."""
    expected = batches(directory)._order(0)
    batches(directory).sharing_orders(tmp_path / "orders")._order(0)
    real = np.memmap
    removed = []

    def removed_first(path, *args, mode="r+", **options):
        if mode == "r" and not removed:
            removed.append(path)
            Path(path).unlink()
        return real(path, *args, mode=mode, **options)

    monkeypatch.setattr(np, "memmap", removed_first)

    order = batches(directory).sharing_orders(tmp_path / "orders")._order(0)

    assert removed, "the race was not set up"
    assert np.array_equal(order, expected)


def test_a_disk_with_no_room_for_an_order_says_so(directory, tmp_path, monkeypatch):
    """Rather than the writer dying of SIGBUS on the first page there is no block for."""

    def full(descriptor, offset, length):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    # Not raising where os has none, as on macOS: there the fake is what the code then finds.
    monkeypatch.setattr(os, "posix_fallocate", full, raising=False)
    source = batches(directory).sharing_orders(tmp_path / "orders")

    with pytest.raises(OSError, match="No space left on device"):
        source._order(0)

    assert not list((tmp_path / "orders").glob("*.writing")), "nor a half-made file left behind"


def test_an_order_is_written_where_space_cannot_be_reserved_up_front(
    directory, tmp_path, monkeypatch
):
    """macOS has no posix_fallocate, and the file is sized without it."""
    monkeypatch.delattr(os, "posix_fallocate", raising=False)

    order = batches(directory).sharing_orders(tmp_path / "orders")._order(0)

    assert np.array_equal(order, batches(directory)._order(0))


def test_what_a_writer_that_died_left_is_written_again(directory, tmp_path):
    expected = batches(directory)._order(0)
    orders = tmp_path / "orders"
    orders.mkdir()
    (orders / "train-seed0-epoch0.order.writing").write_bytes(b"half")
    (orders / "train-seed0-epoch0.order").write_bytes(b"not the right size")

    order = batches(directory).sharing_orders(orders)._order(0)

    assert np.array_equal(order, expected)
    assert not list(orders.glob("*.writing"))


def test_sharing_orders_leaves_the_original_alone_and_hands_over_nothing_cached(
    directory, tmp_path
):
    original = batches(directory)
    expected = moves(original, 2)
    untouched = len(pickle.dumps(batches(directory)))

    shared = original.sharing_orders(tmp_path / "orders")

    assert len(pickle.dumps(shared)) <= untouched + 200, "only the directory's name is added"
    assert moves(original, 2) == expected, "the original's reader was not closed under it"
    assert moves(shared, 2) == expected


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_workers_sharing_orders_produce_the_same_batches(directory, tmp_path, method):
    expected = moves(batches(directory, batches=8), 8)
    source = batches(directory, batches=8).sharing_orders(tmp_path / "orders")
    loader = batch_loader(source, workers=2, device=torch.device("cpu"))
    loader.multiprocessing_context = method

    loaded = list(iterate(loader))

    assert [int(move) for batch in loaded for move in batch.move] == expected
    assert [path.name for path in (tmp_path / "orders").glob("*.order")] == [
        "train-seed0-epoch0.order"
    ]


def test_a_batch_carries_the_history_its_encoder_asks_for(directory):
    options = {"history": 2}
    source = batches(directory, encoder_options=options)

    batch = source[0]

    spec = create_encoder("board-planes", **options).spec
    assert batch.spatial.shape == (4, *spec.spatial_shape) == (4, 36, 8, 8)
    assert torch.equal(batch.spatial[:, :12], batches(directory)[0].spatial), "the same positions"
    assert batch.spatial[:, 12:].sum() > 0, "with something behind them"


def test_an_oriented_batch_mirrors_the_targets_of_the_positions_it_turned(directory):
    """The target has to be the move on the board the model was shown."""
    absolute = batches(directory, batch_size=32)[0]
    turned = batches(directory, batch_size=32, encoder_options={"orientation": "side-to-move"})[0]

    white_to_move = absolute.globals[:, 0] == 1
    mirrored = torch.from_numpy(MIRRORED_INDEX.astype(np.int64))[absolute.move]
    assert white_to_move.any() and (~white_to_move).any()
    assert torch.equal(turned.move[white_to_move], absolute.move[white_to_move])
    assert torch.equal(turned.move[~white_to_move], mirrored[~white_to_move])
    assert not torch.equal(turned.move, absolute.move)
    assert torch.equal(turned.result, absolute.result), "the result is already the mover's"


def test_the_planes_travel_as_bytes_and_arrive_on_the_device_as_floats(directory):
    batch = batches(directory)[0]

    assert batch.spatial.dtype == torch.uint8, "a quarter of the size through the loader"
    moved = batch.to(torch.device("cpu"))
    assert moved.spatial.dtype == torch.float32
    assert torch.equal(moved.spatial, batch.spatial.float())
    assert moved.move.dtype == torch.int64, "only the planes are converted"


# The loader's workers and the signals that stop a run.


@pytest.fixture(params=["spawn", "forkserver"])
def start_method(request):
    """Start the loader's workers as macOS does, and as Linux does from Python 3.14 on.

    Both start each worker as a fresh interpreter, which spends seconds importing before its
    ``worker_init_fn`` runs, with whatever signal handlers it was born with.
    """
    before = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method(request.param, force=True)
    yield request.param
    multiprocessing.set_start_method(before, force=True)


def test_a_worker_signalled_while_it_starts_up_survives(start_method):
    """Ctrl-C in the first moments of a run reaches workers that are still starting."""
    loader = batch_loader(SignalledWhileStarting(), workers=2, device=torch.device("cpu"))

    assert sorted(int(item) for item in iterate(loader)) == list(range(8))


FRESH_START = """
import signal, sys
from multiprocessing import resource_tracker

import torch

from chess_ai.training.batches import batch_loader, iterate
from signal_helpers import SignalledWhileStarting

# Python 3.12.0 to 3.12.9 and 3.13.0 start the resource tracker by blocking SIGINT and SIGTERM
# and then unblocking them, whatever the caller had blocked; later releases put the caller's
# mask back. The older way is reproduced here, so that this means the same on every release.
class OlderSignal:
    def __getattr__(self, name):
        return getattr(signal, name)

    def pthread_sigmask(self, how, mask):
        if how == signal.SIG_SETMASK:
            return signal.pthread_sigmask(signal.SIG_UNBLOCK, resource_tracker._IGNORED_SIGNALS)
        return signal.pthread_sigmask(how, mask)

resource_tracker.signal = OlderSignal()
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method(sys.argv[1], force=True)
    loader = batch_loader(SignalledWhileStarting(), workers=2, device=torch.device("cpu"))
    print(sorted(int(item) for item in iterate(loader)))
"""


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_workers_are_protected_in_a_process_that_has_not_started_anything_yet(tmp_path, method):
    """The first loader of a training process is also the first thing to start the resource
    tracker, which some Python releases do by unblocking the very signals held back here.

    In a fresh interpreter, since in this one an earlier test has started the tracker already.
    """
    script = tmp_path / "fresh_start.py"
    script.write_text(FRESH_START)
    tests = Path(__file__).parent

    finished = subprocess.run(
        [sys.executable, str(script), method],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(tests)},
    )

    assert finished.returncode == 0, finished.stderr[-2000:]
    assert finished.stdout.strip() == str(list(range(8)))


def test_a_stop_signal_sent_while_the_workers_start_still_reaches_the_trainer(directory):
    """The trainer may hold a stop signal back while its workers are born, but never lose it.

    With another thread running, as torch's own pool is once a model has run, a signal the
    trainer blocks is delivered to that thread instead; were it ignored rather than only held
    back, that thread would drop it, and the run would never stop.
    """
    heard = []
    previous = signal.signal(signal.SIGINT, lambda number, frame: heard.append(number))
    real = torch.utils.data.DataLoader.__iter__
    done = threading.Event()
    other = threading.Thread(target=done.wait)
    other.start()

    def ctrl_c_while_starting(self):
        iterator = real(self)
        os.kill(os.getpid(), signal.SIGINT)
        return iterator

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.utils.data.DataLoader, "__iter__", ctrl_c_while_starting)
            loader = batch_loader(batches(directory), workers=2, device=torch.device("cpu"))
            assert len(list(iterate(loader))) == 10
    finally:
        done.set()
        other.join()
        signal.signal(signal.SIGINT, previous)

    assert heard == [signal.SIGINT]


def test_batches_can_be_read_off_the_main_thread(directory):
    """Only the main thread may set a signal handler, and a run need not be on it."""
    read = []

    def reader():
        loader = batch_loader(batches(directory), workers=2, device=torch.device("cpu"))
        read.extend(iterate(loader))

    thread = threading.Thread(target=reader)
    thread.start()
    thread.join()

    assert len(read) == 10


def test_no_worker_outlives_a_shutdown_that_was_interrupted(directory, monkeypatch):
    """A second ctrl-c as the loader shuts down skips the message that tells workers to go.

    The workers ignore SIGTERM, so the loader's own last resort of terminating them does
    nothing, and the trainer would hang on its way out, waiting for workers that wait for it.
    """
    from torch.utils.data import dataloader

    real = dataloader._MultiProcessingDataLoaderIter._shutdown_workers

    def interrupted(self):
        def ctrl_c():
            raise KeyboardInterrupt

        self._workers_done_event.set = ctrl_c
        real(self)

    monkeypatch.setattr(dataloader._MultiProcessingDataLoaderIter, "_shutdown_workers", interrupted)
    before = set(multiprocessing.active_children())
    stream = iterate(batch_loader(batches(directory), workers=2, device=torch.device("cpu")))
    next(stream)
    workers = [child for child in multiprocessing.active_children() if child not in before]
    assert len(workers) == 2

    try:
        stream.close()
        for worker in workers:
            worker.join(timeout=10)
        assert [worker.is_alive() for worker in workers] == [False, False]
    finally:
        for worker in workers:
            worker.kill()


def test_another_ctrl_c_while_stragglers_are_killed_waits_until_they_are(directory, monkeypatch):
    """With another thread running, blocking a signal in this one does not keep its handler
    out of it: Python runs handlers in the main thread whichever thread the kernel picked.

    An interrupt half-way through killing the workers left the rest alive, ignoring the SIGTERM
    that the interpreter's exit sends them, and the process hung on its way out.
    """
    from torch.utils.data import dataloader

    real_shutdown = dataloader._MultiProcessingDataLoaderIter._shutdown_workers

    def interrupted(self):
        def ctrl_c():
            raise KeyboardInterrupt

        self._workers_done_event.set = ctrl_c
        real_shutdown(self)

    def now(number, frame):
        raise KeyboardInterrupt

    real_kill = multiprocessing.process.BaseProcess.kill
    pressed = []

    def ctrl_c_while_killing(self):
        if not pressed:
            pressed.append(True)
            os.kill(os.getpid(), signal.SIGINT)
        real_kill(self)

    monkeypatch.setattr(dataloader._MultiProcessingDataLoaderIter, "_shutdown_workers", interrupted)
    previous = signal.signal(signal.SIGINT, now)
    done = threading.Event()
    other = threading.Thread(target=done.wait)
    other.start()
    before = set(multiprocessing.active_children())
    stream = iterate(batch_loader(batches(directory), workers=2, device=torch.device("cpu")))
    next(stream)
    workers = [child for child in multiprocessing.active_children() if child not in before]
    try:
        monkeypatch.setattr(multiprocessing.process.BaseProcess, "kill", ctrl_c_while_killing)
        with pytest.raises(KeyboardInterrupt):
            stream.close()
        for worker in workers:
            worker.join(timeout=10)
        assert pressed, "the interrupt was sent"
        assert [worker.is_alive() for worker in workers] == [False, False]
    finally:
        monkeypatch.setattr(multiprocessing.process.BaseProcess, "kill", real_kill)
        for worker in workers:
            worker.kill()
        done.set()
        other.join()
        signal.signal(signal.SIGINT, previous)


def test_an_epoch_of_a_filtered_dataset_visits_its_training_targets_and_nothing_else(tmp_path):
    dataset(tmp_path / "data", filters=Filters(min_rating=1700))
    with open_dataset("test", data_dir=tmp_path / "data") as built:
        reader = built["train"]
        assert 0 < reader.targets < len(reader), "some positions are stored and not trained on"
        targets = reader.positions(reader.target_positions(np.arange(reader.targets)))
        source = batches(built.directory, batch_size=1, batches=reader.targets)

        visited = sorted(int(source[index].move[0]) for index in range(reader.targets))

    assert source.positions == len(targets)
    assert visited == sorted(int(move) for move in targets["move"])


@pytest.mark.parametrize("filters", [None, Filters(min_rating=1700)])
def test_the_stored_batch_is_the_first_positions_trained_on_in_stored_order(
    tmp_path, monkeypatch, filters
):
    """What the throughput probe steps on: a real batch, for no epoch's order."""
    dataset(tmp_path / "data", filters=filters)
    with open_dataset("test", data_dir=tmp_path / "data") as built:
        unshuffled = batches(built.directory)
        monkeypatch.setattr(batches_module, "_shuffle", lambda order, seed, epoch: order)
        expected = unshuffled[0]
        source = batches(built.directory)
        asked = []
        monkeypatch.setattr(source, "_order", lambda epoch: asked.append(epoch))

        batch = source.stored_batch()

    assert not asked, "no epoch's order was built"
    assert len(batch) == source.positions_per_batch
    for got, wanted in zip(batch, expected, strict=True):
        assert torch.equal(got, wanted)


def test_the_stored_batch_of_a_split_smaller_than_a_batch_is_all_of_it(directory):
    source = batches(directory, batch_size=10_000)

    assert len(source.stored_batch()) == source.positions
