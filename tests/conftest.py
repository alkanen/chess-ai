import os

import pytest

from chess_ai import output


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Keep the developer's own chess-ai.toml and CHESS_AI_* variables out of tests."""
    for name in os.environ:
        if name.startswith("CHESS_AI_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def unstamped_output(monkeypatch):
    """Leave lines unstamped, as they are in the journal, so that tests can compare them whole;
    and leave no command's verbosity behind for the next test. Timestamp tests ask for them."""
    monkeypatch.delenv("JOURNAL_STREAM", raising=False)
    monkeypatch.setattr(output, "to_the_journal", lambda stream: True)
    yield
    output.reset()


@pytest.fixture
def anyio_backend():
    """Run ``@pytest.mark.anyio`` tests on asyncio only, as the server does."""
    return "asyncio"
