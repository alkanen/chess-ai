import os

import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Keep the developer's own chess-ai.toml and CHESS_AI_* variables out of tests."""
    for name in os.environ:
        if name.startswith("CHESS_AI_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def anyio_backend():
    """Run ``@pytest.mark.anyio`` tests on asyncio only, as the server does."""
    return "asyncio"
