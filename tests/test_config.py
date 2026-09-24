from pathlib import Path

import pytest

from chess_ai.config import ConfigError, load_config


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_file_or_environment():
    server = load_config(environ={}).server

    assert (server.host, server.port, server.path_prefix) == ("127.0.0.1", 8000, "")


def test_reads_settings_from_file(tmp_path):
    path = write(
        tmp_path / "custom.toml",
        '[server]\nhost = "0.0.0.0"\nport = 9000\npath_prefix = "/chess"\n',
    )

    server = load_config(path, environ={}).server

    assert (server.host, server.port, server.path_prefix) == ("0.0.0.0", 9000, "/chess")


def test_finds_file_through_environment_variable(tmp_path):
    path = write(tmp_path / "custom.toml", "[server]\nport = 9001\n")

    config = load_config(environ={"CHESS_AI_CONFIG": str(path)})

    assert config.server.port == 9001


def test_finds_file_in_current_directory(tmp_path):
    write(tmp_path / "chess-ai.toml", "[server]\nport = 9002\n")

    config = load_config(environ={})

    assert config.server.port == 9002


def test_environment_variables_override_file(tmp_path):
    path = write(tmp_path / "custom.toml", '[server]\nport = 9000\npath_prefix = "/chess"\n')

    server = load_config(
        path,
        environ={
            "CHESS_AI_SERVER_HOST": "0.0.0.0",
            "CHESS_AI_SERVER_PORT": "9100",
            "CHESS_AI_SERVER_PATH_PREFIX": "",
        },
    ).server

    assert (server.host, server.port, server.path_prefix) == ("0.0.0.0", 9100, "")


def test_games_are_saved_in_a_directory_of_the_working_directory_by_default():
    assert load_config(environ={}).paths.games == Path("games")


def test_reads_the_games_directory_from_file(tmp_path):
    path = write(tmp_path / "custom.toml", '[paths]\ngames = "/srv/chess/games"\n')

    assert load_config(path, environ={}).paths.games == Path("/srv/chess/games")


def test_environment_variable_overrides_the_games_directory(tmp_path):
    path = write(tmp_path / "custom.toml", '[paths]\ngames = "/srv/chess/games"\n')

    config = load_config(path, environ={"CHESS_AI_PATHS_GAMES": "/mnt/big/games"})

    assert config.paths.games == Path("/mnt/big/games")


def test_expands_a_home_directory_in_the_games_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_config(environ={"CHESS_AI_PATHS_GAMES": "~/chess/games"})

    assert config.paths.games == tmp_path / "chess" / "games"


@pytest.mark.parametrize(
    ("given", "normalized"),
    [
        ("", ""),
        ("/", ""),
        ("chess", "/chess"),
        ("/chess/", "/chess"),
        ("/apps/chess-ai", "/apps/chess-ai"),
    ],
)
def test_normalizes_path_prefix(given, normalized):
    config = load_config(environ={"CHESS_AI_SERVER_PATH_PREFIX": given})

    assert config.server.path_prefix == normalized


@pytest.mark.parametrize("prefix", ["/a//b", "/../chess", "/chess game", '/"><script>'])
def test_rejects_unsafe_path_prefix(prefix):
    with pytest.raises(ConfigError, match="path_prefix"):
        load_config(environ={"CHESS_AI_SERVER_PATH_PREFIX": prefix})


def test_rejects_unknown_setting(tmp_path):
    path = write(tmp_path / "custom.toml", "[server]\nprot = 9000\n")

    with pytest.raises(ConfigError, match="server.prot"):
        load_config(path, environ={})


def test_rejects_invalid_port_from_environment():
    with pytest.raises(ConfigError, match="server.port"):
        load_config(environ={"CHESS_AI_SERVER_PORT": "http"})


def test_reports_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "missing.toml", environ={})


def test_reports_malformed_toml(tmp_path):
    path = write(tmp_path / "custom.toml", "[server\n")

    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path, environ={})
