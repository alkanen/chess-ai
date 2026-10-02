"""Datasets for the loader's signal tests, importable by a worker started with ``spawn``.

A worker started with ``spawn`` or ``forkserver`` imports what it runs afresh rather than
inheriting it, so anything it unpickles has to live in a module it can import by name.
"""

import os
import signal

import torch
from torch.utils.data import Dataset

PARENT = "CHESS_AI_TEST_LOADER_PARENT"
"""The environment variable naming the process the dataset was made in, which it leaves alone."""


class SignalledWhileStarting(Dataset):
    """Sends its own process SIGINT and SIGTERM as a worker unpickles it.

    That is the window between a worker being started and its ``worker_init_fn`` running, when
    a freshly started interpreter still has the default handlers, and what a ctrl-c pressed
    while the workers start up would land in.
    """

    def __init__(self, size: int = 8) -> None:
        # Unpickling only calls __setstate__ when there is some state to set.
        self.size = size
        os.environ[PARENT] = str(os.getpid())

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if os.environ.get(PARENT) != str(os.getpid()):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
