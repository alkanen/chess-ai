"""What a GPU says about itself in the heartbeat, when NVML answers all, some or none of it.

No GPU is needed: CUDA's memory figures and NVML are both faked, since what is under test is
what the heartbeat makes of their answers and of their refusals.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from chess_ai.training import hardware
from chess_ai.training.hardware import gpu_stats
from chess_ai.training.run_store import GpuStats

GIB = 2**30

CUDA0 = torch.device("cuda", 0)


class FakeNvml:
    """Enough of pynvml to be asked about one card, refusing whatever ``refuse`` names."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, *, refuse: frozenset[str] = frozenset(), unavailable: bool = False):
        self.refuse = refuse
        self.unavailable = unavailable

    def nvmlInit(self):
        if self.unavailable:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return "card"

    def nvmlDeviceGetHandleByIndex(self, index):
        return "card"

    def _answer(self, question, value):
        if question in self.refuse:
            raise RuntimeError("Not Supported")
        return value

    def nvmlDeviceGetName(self, handle):
        return self._answer("name", "NVIDIA GeForce RTX 4090")

    def nvmlDeviceGetUtilizationRates(self, handle):
        return self._answer("utilization", SimpleNamespace(gpu=87, memory=40))

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return self._answer("temperature", 64)


@pytest.fixture
def cuda(monkeypatch):
    """A CUDA card with 24 GiB, 20 GiB of it in use and 6 GiB of that held by this process."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (4 * GIB, 24 * GIB))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "cuda's name for it")
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 6 * GIB)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid="1234")
    )
    # Every test asks NVML afresh, rather than taking the last test's answer from the cache.
    monkeypatch.setattr(hardware, "_HANDLES", {})


def use_nvml(monkeypatch, nvml: FakeNvml | None) -> None:
    """Make ``import pynvml`` find ``nvml``, or fail as it does when it is not installed."""
    monkeypatch.setitem(sys.modules, "pynvml", nvml)


def test_a_cpu_run_has_no_gpu_to_report():
    assert gpu_stats(torch.device("cpu")) is None


def test_a_gpu_that_nvml_answers_for_reports_everything(cuda, monkeypatch):
    use_nvml(monkeypatch, FakeNvml())

    assert gpu_stats(CUDA0) == GpuStats(
        name="NVIDIA GeForce RTX 4090",
        utilization_percent=87,
        memory_used_bytes=20 * GIB,
        memory_total_bytes=24 * GIB,
        process_memory_bytes=6 * GIB,
        temperature_celsius=64,
    )


@pytest.mark.parametrize(
    "nvml",
    [None, FakeNvml(unavailable=True)],
    ids=["not installed", "no driver"],
)
def test_without_nvml_the_memory_figures_are_still_reported(cuda, monkeypatch, nvml):
    use_nvml(monkeypatch, nvml)

    assert gpu_stats(CUDA0) == GpuStats(
        name="cuda's name for it",
        memory_used_bytes=20 * GIB,
        memory_total_bytes=24 * GIB,
        process_memory_bytes=6 * GIB,
    )


def test_a_question_nvml_declines_costs_only_its_own_answer(cuda, monkeypatch):
    use_nvml(monkeypatch, FakeNvml(refuse=frozenset({"utilization", "name"})))

    stats = gpu_stats(CUDA0)

    assert stats is not None
    assert stats.utilization_percent is None
    assert stats.name == "cuda's name for it"
    assert stats.temperature_celsius == 64
    assert stats.memory_used_bytes == 20 * GIB
