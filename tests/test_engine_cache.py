import threading
import time

import chess
import pytest

from chess_ai.web.engine_cache import CheckpointKey, EngineCache

A, B, C = CheckpointKey("a", 1), CheckpointKey("b", 2), CheckpointKey("c", 3)


class FakeEngine:
    def __init__(self, key: CheckpointKey) -> None:
        self.key = key
        self.asked: list[tuple[str, int | None]] = []

    def evaluate(self, board, *, mover_rating=None, opponent_rating=None):
        self.asked.append((board.fen(), mover_rating))
        return self.key


class Loader:
    """Loads fake engines and counts what it was asked to load."""

    def __init__(self) -> None:
        self.loads: list[CheckpointKey] = []

    def __call__(self, key: CheckpointKey) -> FakeEngine:
        self.loads.append(key)
        return FakeEngine(key)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def cache(loader=None, *, capacity=2, idle_after=100.0, clock=None, on_unload=None):
    return EngineCache(
        loader if loader is not None else Loader(),
        capacity=capacity,
        idle_after=idle_after,
        clock=clock if clock is not None else Clock(),
        on_unload=on_unload,
    )


def test_games_playing_the_same_checkpoint_share_one_load():
    loader = Loader()
    engines = cache(loader)

    first, second = engines.engine(A), engines.engine(A)
    first.evaluate(chess.Board(), mover_rating=1500)
    second.evaluate(chess.Board())

    assert loader.loads == [A]
    assert engines.loaded == [A]


def test_a_checkpoint_is_asked_what_the_player_asks():
    engines = cache()

    answer = engines.engine(A).evaluate(chess.Board(), mover_rating=1200, opponent_rating=1200)

    assert answer == A


def test_a_full_cache_lets_go_of_the_checkpoint_used_longest_ago():
    loader = Loader()
    engines = cache(loader, capacity=2)
    engines.preload(A)
    engines.preload(B)
    engines.preload(A)  # A is now the more recently used of the two.

    engines.preload(C)

    assert engines.loaded == [A, C]
    engines.preload(B)
    assert loader.loads == [A, B, C, B]


def test_a_checkpoint_working_out_a_move_is_not_let_go_of():
    engines = cache(capacity=2)
    engines.preload(A)
    engines.preload(B)

    # A is the one used longest ago, but it is thinking, so B makes room instead.
    with engines.borrowed(A):
        engines.preload(C)
        assert A in engines.loaded

    assert engines.loaded == [A, C]


def test_a_load_waits_while_every_checkpoint_held_is_working_out_a_move():
    engines = cache(capacity=1)
    loaded_c = threading.Event()

    def load_c():
        engines.preload(C)
        loaded_c.set()

    with engines.borrowed(A):
        waiting = threading.Thread(target=load_c)
        waiting.start()
        assert not loaded_c.wait(0.1), "C was loaded while A was still in use"
    waiting.join(5)

    assert loaded_c.is_set()
    assert engines.loaded == [C]


def test_a_checkpoint_nobody_has_played_with_for_long_enough_is_let_go_of():
    clock = Clock()
    unloads = []
    engines = cache(clock=clock, idle_after=100.0, on_unload=lambda: unloads.append(True))
    engines.preload(A)
    clock.now = 50.0
    engines.preload(B)

    clock.now = 120.0
    engines.unload_idle()

    assert engines.loaded == [B]
    assert unloads == [True]


def test_a_checkpoint_in_use_is_not_idle_however_long_the_move_takes():
    clock = Clock()
    engines = cache(clock=clock, idle_after=100.0)

    with engines.borrowed(A):
        clock.now = 500.0
        engines.unload_idle()
        assert engines.loaded == [A]


def test_a_checkpoint_let_go_of_is_loaded_again_when_a_game_next_needs_it():
    clock = Clock()
    loader = Loader()
    engines = cache(loader, clock=clock, idle_after=100.0)
    player = engines.engine(A)
    player.evaluate(chess.Board())

    clock.now = 200.0
    engines.unload_idle()
    player.evaluate(chess.Board())

    assert loader.loads == [A, A]


def test_a_checkpoint_that_cannot_be_loaded_says_why_and_holds_nothing():
    def broken(key):
        raise FileNotFoundError(f"{key.run} is gone")

    engines = cache(broken)

    with pytest.raises(FileNotFoundError, match="a is gone"):
        engines.preload(A)
    assert engines.loaded == []


def test_two_games_wanting_the_same_checkpoint_at_once_load_it_once():
    entered, release = threading.Event(), threading.Event()
    loads = []

    def slow(key):
        loads.append(key)
        entered.set()
        assert release.wait(5)
        return FakeEngine(key)

    engines = cache(slow)
    first = threading.Thread(target=engines.preload, args=(A,))
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=engines.preload, args=(A,))
    second.start()
    release.set()
    first.join(5)
    second.join(5)

    assert loads == [A]


class Gated:
    """Engines whose answers wait to be let through, and which say who is answering."""

    def __init__(self) -> None:
        self.answering: list[str] = []
        self.most_at_once = 0
        self.let_through = threading.Semaphore(0)
        self._inside = 0
        self._lock = threading.Lock()

    def __call__(self, key: CheckpointKey) -> "Gated._Engine":
        return Gated._Engine(self, key)

    def wait_until_answering(self, count: int) -> None:
        for _ in range(500):
            with self._lock:
                if len(self.answering) >= count:
                    return
            time.sleep(0.01)
        raise AssertionError(f"{count} answers never started: {self.answering}")

    class _Engine:
        def __init__(self, gated: "Gated", key: CheckpointKey) -> None:
            self._gated = gated
            self._key = key

        def _answer(self, asker: str) -> CheckpointKey:
            gated = self._gated
            with gated._lock:
                gated.answering.append(asker)
                gated._inside += 1
                gated.most_at_once = max(gated.most_at_once, gated._inside)
            try:
                assert gated.let_through.acquire(timeout=5), f"{asker} was never let through"
            finally:
                with gated._lock:
                    gated._inside -= 1
            return self._key

        def evaluate(self, board, *, mover_rating=None, opponent_rating=None):
            return self._answer(f"{self._key.run} one")

        def evaluate_many(self, boards, **ratings):
            return [self._answer(f"{self._key.run} {len(boards)}")]


def asking(ask) -> threading.Thread:
    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    return thread


def test_the_network_answers_one_question_at_a_time_whichever_checkpoint_asks():
    gated = Gated()
    engines = cache(gated)
    engines.preload(A)
    engines.preload(B)

    searching = asking(lambda: engines.engine(A).evaluate_many([chess.Board()] * 16))
    gated.wait_until_answering(1)
    playing = asking(lambda: engines.engine(B).evaluate(chess.Board()))
    time.sleep(0.1)
    assert gated.answering == ["a 16"], "B was answered while A's batch was"
    gated.let_through.release(2)
    searching.join(5)
    playing.join(5)

    assert gated.answering == ["a 16", "b one"]
    assert gated.most_at_once == 1


def wait_until_waiting(engines: EngineCache, count: int) -> None:
    for _ in range(500):
        if engines.waiting >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"{count} questions never waited, only {engines.waiting}")


def test_a_search_asking_batch_after_batch_lets_a_question_asked_meanwhile_go_first():
    gated = Gated()
    engines = cache(gated)
    engines.preload(A)
    engines.preload(B)

    def search():
        for _ in range(3):
            engines.engine(A).evaluate_many([chess.Board()] * 16)

    searching = asking(search)
    gated.wait_until_answering(1)
    playing = asking(lambda: engines.engine(B).evaluate(chess.Board()))
    wait_until_waiting(engines, 1)
    for answered in range(1, 4):
        gated.let_through.release()
        gated.wait_until_answering(answered + 1)
    gated.let_through.release()
    searching.join(5)
    playing.join(5)

    assert gated.answering == ["a 16", "b one", "a 16", "a 16"]


def test_clearing_lets_go_of_everything():
    engines = cache()
    engines.preload(A)
    engines.preload(B)

    engines.clear()

    assert engines.loaded == []


def test_a_cache_must_hold_something():
    with pytest.raises(ValueError, match="at least 1"):
        cache(capacity=0)
