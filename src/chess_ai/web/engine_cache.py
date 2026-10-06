"""Loaded checkpoints, shared by every game that plays one.

Up to twenty games can be in progress at once, and several of them may well be playing the
same checkpoint: a copy each would be a copy each in memory, on a machine that may also be
training. So the games borrow from one cache instead, for one move at a time.

The cache holds a limited number of checkpoints. A checkpoint that has to be loaded when the
cache is full takes the place of the one used longest ago, and one that nobody has played with
for a long time is let go of by itself; either is loaded again when a game next needs it,
which costs that game a second or two and nothing else.

Borrowing happens on the worker thread a move is worked out on, so this is thread-safe rather
than asyncio-aware, and knows nothing about torch: what it holds is whatever ``load`` returns.
"""

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple

import chess

logger = logging.getLogger(__name__)


class CheckpointKey(NamedTuple):
    """Which checkpoint: the run it is from, the step it was saved at, and which file that was.

    The run and the step alone do not say which weights: a run restarted under the same name
    saves new ones at the same steps. The fingerprint tells the file a game began with from
    one written in its place since, so that the game is not handed different weights under
    the old name when its checkpoint has to be loaded again.
    """

    run: str
    step: int
    fingerprint: tuple[int, ...] = ()


@dataclass
class _Entry:
    engine: Any
    last_used: float
    borrowed: int = 0
    """How many moves are being worked out with it right now; it is not let go of meanwhile."""


class EngineCache:
    def __init__(
        self,
        load: Callable[[CheckpointKey], Any],
        *,
        capacity: int,
        idle_after: float,
        clock: Callable[[], float] = time.monotonic,
        on_unload: Callable[[], None] | None = None,
    ) -> None:
        """A cache of at most ``capacity`` checkpoints, each made by ``load``.

        ``idle_after`` is how many seconds a checkpoint nobody has borrowed is kept before
        :meth:`unload_idle` lets go of it. ``on_unload`` is called after any checkpoint has been
        let go of, which is where the memory a device keeps for itself can be handed back.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, not {capacity}")
        self._load = load
        self._capacity = capacity
        self._idle_after = idle_after
        self._clock = clock
        self._on_unload = on_unload
        # Oldest use first, so that the front is what goes when room has to be made.
        self._entries: OrderedDict[CheckpointKey, _Entry] = OrderedDict()
        self._loading: set[CheckpointKey] = set()
        self._changed = threading.Condition()
        # One checkpoint is read and built at a time, however many games want one: reading a
        # few hundred megabytes twice at once is no faster, and is twice the memory at its peak.
        self._one_load_at_a_time = threading.Lock()

    @property
    def loaded(self) -> list[CheckpointKey]:
        """The checkpoints held, the one used longest ago first."""
        with self._changed:
            return list(self._entries)

    def engine(self, key: CheckpointKey) -> "CachedEngine":
        """Something to play ``key`` with, which borrows it from here for every move."""
        return CachedEngine(self, key)

    def preload(self, key: CheckpointKey) -> None:
        """Load ``key`` now, if it is not held already, so that whatever is wrong with it is
        raised here rather than on the game's first move.

        Raises:
            Exception: whatever ``load`` raises.
        """
        with self.borrowed(key):
            pass

    @contextmanager
    def borrowed(self, key: CheckpointKey) -> Iterator[Any]:
        """The engine for ``key``, loaded if it has to be, held for as long as this lasts.

        Blocks while the cache is full of checkpoints that are all being borrowed, until one of
        them is given back.

        Raises:
            Exception: whatever ``load`` raises; nothing is held for ``key`` afterwards.
        """
        engine = self._acquire(key)
        try:
            yield engine
        finally:
            self._release(key)

    def unload_idle(self) -> None:
        """Let go of every checkpoint that nobody has borrowed for ``idle_after`` seconds."""
        with self._changed:
            now = self._clock()
            idle = [
                key
                for key, entry in self._entries.items()
                if entry.borrowed == 0 and now - entry.last_used >= self._idle_after
            ]
            for key in idle:
                del self._entries[key]
        if idle:
            logger.info("Let go of idle checkpoints: %s", ", ".join(map(_describe, idle)))
            self._unloaded()

    def clear(self) -> None:
        """Let go of everything nobody is borrowing, as when the server goes down."""
        with self._changed:
            for key in [key for key, entry in self._entries.items() if entry.borrowed == 0]:
                del self._entries[key]
        self._unloaded()

    def _acquire(self, key: CheckpointKey) -> Any:
        evicted: CheckpointKey | None = None
        with self._changed:
            while True:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.borrowed += 1
                    entry.last_used = self._clock()
                    self._entries.move_to_end(key)
                    return entry.engine
                if key in self._loading:
                    # Somebody else is loading it already; it is theirs to load, and ours to use.
                    self._changed.wait()
                    continue
                if len(self._entries) + len(self._loading) >= self._capacity:
                    evicted = self._evict_one()
                    if evicted is None:
                        # Every checkpoint held is working out a move. One will be given back
                        # within a move's time.
                        self._changed.wait()
                        continue
                self._loading.add(key)
                break
        if evicted is not None:
            logger.info("Let go of %s to make room for %s", _describe(evicted), _describe(key))
            self._unloaded()
        try:
            with self._one_load_at_a_time:
                engine = self._load(key)
        except BaseException:
            with self._changed:
                self._loading.discard(key)
                self._changed.notify_all()
            raise
        with self._changed:
            self._loading.discard(key)
            self._entries[key] = _Entry(engine, last_used=self._clock(), borrowed=1)
            self._changed.notify_all()
        return engine

    def _release(self, key: CheckpointKey) -> None:
        with self._changed:
            entry = self._entries[key]
            entry.borrowed -= 1
            entry.last_used = self._clock()
            self._changed.notify_all()

    def _evict_one(self) -> CheckpointKey | None:
        """Drop the checkpoint used longest ago that nobody is borrowing, and say which."""
        for key, entry in self._entries.items():
            if entry.borrowed == 0:
                del self._entries[key]
                return key
        return None

    def _unloaded(self) -> None:
        if self._on_unload is None:
            return
        try:
            self._on_unload()
        except Exception:
            logger.exception("Could not hand back the memory of an unloaded checkpoint")


class CachedEngine:
    """A checkpoint as a model player sees it, borrowed from the cache one move at a time.

    It answers the one question a model player asks of an engine, so that the player itself
    need not know there is a cache, and holds no engine between moves: a checkpoint the cache
    lets go of is really gone, rather than kept alive by a game that is waiting for a person.
    """

    def __init__(self, cache: EngineCache, key: CheckpointKey) -> None:
        self._cache = cache
        self.key = key
        self.run = key.run
        self.step = key.step

    def evaluate(
        self,
        board: chess.Board,
        *,
        mover_rating: int | None = None,
        opponent_rating: int | None = None,
    ) -> Any:
        with self._cache.borrowed(self.key) as engine:
            return engine.evaluate(
                board, mover_rating=mover_rating, opponent_rating=opponent_rating
            )


def _describe(key: CheckpointKey) -> str:
    return f"{key.run} step {key.step}"
