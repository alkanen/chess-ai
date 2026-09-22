import pytest
import uvicorn
from fastapi.testclient import TestClient

from chess_ai.cli import main


@pytest.fixture
def served(monkeypatch):
    """Capture what ``chess-ai serve`` would run instead of starting a server."""
    calls = {}

    def fake_run(app, host, port):
        calls.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


def test_serve_uses_configured_address_and_prefix(tmp_path, served, capsys):
    config = tmp_path / "custom.toml"
    config.write_text('[server]\nhost = "0.0.0.0"\nport = 9000\npath_prefix = "/chess"\n')

    assert main(["--config", str(config), "serve"]) == 0

    assert (served["host"], served["port"]) == ("0.0.0.0", 9000)
    assert TestClient(served["app"]).get("/chess/api/start-position").status_code == 200
    assert "http://0.0.0.0:9000/chess/" in capsys.readouterr().out


def test_invalid_config_exits_with_message(tmp_path, served, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--config", str(tmp_path / "missing.toml"), "serve"])

    assert exit_info.value.code == 2
    assert "cannot read config file" in capsys.readouterr().err
    assert served == {}


def test_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
