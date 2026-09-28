"""Which device to train on, and what the device says about itself while training.

Two things that both have to degrade rather than fail. A run asked to train on ``auto`` should
use the RTX 4090 when it is there and the CPU when it is not, because the same config has to work
on the training machine and in a test. And the hardware statistics in the heartbeat come from
NVML, which is a driver library that may not be installed, may have no driver to talk to, and on
some cards declines one question while answering the rest — none of which is a reason for a
two-day run to stop.
"""

import logging
from typing import Any, Final

import torch

from chess_ai.training.run_store import GpuStats

LOGGER = logging.getLogger(__name__)

CUDA: Final = "cuda"
CPU: Final = "cpu"


class HardwareError(Exception):
    """The device a config asked for is not there."""


def resolve_device(requested: str) -> torch.device:
    """The device to train on, from what the config asked for.

    ``auto`` prefers CUDA and falls back to the CPU. Naming a device explicitly is taken at its
    word: a config that says ``cuda`` on a machine with no GPU has a mistake in it somewhere, and
    quietly training two hundred times slower on the CPU is the least useful way to find out.
    """
    if requested == CPU:
        return torch.device(CPU)
    available = torch.cuda.is_available()
    if requested == CUDA and not available:
        raise HardwareError(
            "the config asks to train on cuda, but no CUDA device is available; "
            'use device = "auto" to fall back to the CPU'
        )
    return torch.device(CUDA if available else CPU)


def describe_device(device: torch.device) -> str:
    """What to print about the device before a run starts."""
    if device.type != CUDA:
        return f"{CPU} ({torch.get_num_threads()} threads)"
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    memory = properties.total_memory / 1024**3
    return f"{CUDA}:{index} {properties.name}, {memory:.0f} GiB"


def autocast_dtype(device: torch.device, *, mixed_precision: bool) -> torch.dtype | None:
    """The dtype to run the forward and backward passes in, or ``None`` for full precision.

    bfloat16 rather than float16 because it has float32's exponent range, so it needs no loss
    scaling and a run cannot quietly diverge into infinities. It needs an Ampere or newer card;
    anything older trains in float32, slower but correct.
    """
    if not mixed_precision or device.type != CUDA:
        return None
    if not torch.cuda.is_bf16_supported():
        LOGGER.warning("this GPU does not support bfloat16; training in float32")
        return None
    return torch.bfloat16


def gpu_stats(device: torch.device) -> GpuStats | None:
    """What ``device`` says about itself now, or ``None`` when it is not a GPU.

    Asked fresh every heartbeat rather than cached, since utilization and temperature are the
    point: what is wanted is whether the card is busy and how hot it is at this moment.

    Memory comes from CUDA rather than from NVML, which sounds like the wrong way round and is
    not. Under WSL2, NVML reports the whole Windows host's view of the card — on this machine it
    called a card with 21 GiB free almost full — while CUDA reports what a process here can
    actually allocate. The number is in the heartbeat to answer "will a bigger batch fit", so it
    has to be the one that decides that.
    """
    if device.type != CUDA:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    stats = GpuStats(
        name=torch.cuda.get_device_name(index),
        memory_used_bytes=total - free,
        memory_total_bytes=total,
        process_memory_bytes=torch.cuda.memory_reserved(index),
    )
    handle = _nvml_handle(device)
    if handle is None:
        return stats
    import pynvml

    return stats.model_copy(
        update={
            "name": _ask(pynvml.nvmlDeviceGetName, handle) or stats.name,
            "utilization_percent": _field(
                _ask(pynvml.nvmlDeviceGetUtilizationRates, handle), "gpu"
            ),
            "temperature_celsius": _ask(
                pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            ),
        }
    )


_HANDLES: dict[int, Any] = {}
"""NVML handles by torch device index, because initialising NVML on every heartbeat is waste."""

_NVML_UNAVAILABLE: Final = object()
"""What :data:`_HANDLES` holds for a device NVML could not be asked about, so it is asked once."""


def _nvml_handle(device: torch.device) -> Any | None:
    """The NVML handle for ``device``, or ``None`` if NVML cannot answer for it.

    Matched by UUID rather than by index: with ``CUDA_VISIBLE_DEVICES`` set, torch's device 0 is
    not NVML's device 0, and reporting another card's temperature would be worse than reporting
    none. Falls back to the index only when torch does not offer a UUID.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _HANDLES.get(index)
    if cached is _NVML_UNAVAILABLE:
        return None
    if cached is not None:
        return cached
    handle = _find_device(index)
    _HANDLES[index] = handle if handle is not None else _NVML_UNAVAILABLE
    return handle


def _find_device(index: int) -> Any | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        if uuid is not None:
            return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as e:  # noqa: BLE001 - NVML raises its own errors, and an import may fail
        LOGGER.info("GPU statistics unavailable: %s", e)
        return None


def _ask(question, *args: Any) -> Any | None:
    """Ask NVML one thing, and take "no" for an answer."""
    try:
        return question(*args)
    except Exception:  # noqa: BLE001 - one unsupported query must not cost the others
        return None


def _field(answer: Any, name: str) -> Any | None:
    """One field of an NVML struct, or ``None`` if the query it came from failed."""
    return getattr(answer, name, None) if answer is not None else None
