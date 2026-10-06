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


def test_a_device_that_is_not_one_is_refused_when_the_config_is_read(tmp_path):
    """Said at startup rather than at the first model game, which is far from the cause."""
    path = write(tmp_path / "chess-ai.toml", '[inference]\ndevice = "gpu"\n')

    with pytest.raises(ConfigError, match="inference.device"):
        load_config(path, environ={})


def test_a_device_from_the_environment_is_checked_too(tmp_path):
    with pytest.raises(ConfigError, match="inference.device"):
        load_config(environ={"CHESS_AI_INFERENCE_DEVICE": "metal"})


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_the_devices_a_network_can_be_asked_to_run_on(tmp_path, device):
    path = write(tmp_path / "chess-ai.toml", f'[inference]\ndevice = "{device}"\n')

    assert load_config(path, environ={}).inference.device == device


def test_stockfish_is_looked_up_on_the_path_by_default():
    assert load_config(environ={}).stockfish.path == "stockfish"


def test_where_stockfish_is_can_be_set_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_config(environ={"CHESS_AI_STOCKFISH_PATH": "~/engines/stockfish"})

    assert config.stockfish.path == str(tmp_path / "engines" / "stockfish")


def test_the_server_holds_twenty_games_and_ten_checkpoints_by_default():
    games = load_config(environ={}).games

    assert (games.max_ongoing, games.max_loaded_checkpoints) == (20, 10)
    assert games.checkpoint_idle_hours == 24


def test_how_many_games_the_server_holds_can_be_set(tmp_path):
    path = write(tmp_path / "custom.toml", "[games]\nmax_ongoing = 5\n")

    games = load_config(path, environ={"CHESS_AI_GAMES_MAX_LOADED_CHECKPOINTS": "2"}).games

    assert (games.max_ongoing, games.max_loaded_checkpoints) == (5, 2)


def test_the_example_config_is_the_defaults():
    example = Path(__file__).parents[1] / "chess-ai.example.toml"

    assert load_config(example, environ={}) == load_config(environ={})


def test_ongoing_games_are_kept_apart_from_the_saved_ones_by_default():
    assert load_config(environ={}).paths.ongoing_games == Path("ongoing-games")


def test_the_ongoing_games_directory_can_be_set_and_expands_a_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_config(environ={"CHESS_AI_PATHS_ONGOING_GAMES": "~/chess/ongoing"})

    assert config.paths.ongoing_games == tmp_path / "chess" / "ongoing"


def test_a_game_is_kept_a_week_after_its_last_move_unless_the_config_says_otherwise(tmp_path):
    assert load_config(environ={}).games.expire_after_days == 7
    path = write(tmp_path / "custom.toml", "[games]\nexpire_after_days = 2.5\n")

    assert load_config(path, environ={}).games.expire_after_days == 2.5


def test_a_game_cannot_be_kept_for_no_time_at_all(tmp_path):
    path = write(tmp_path / "custom.toml", "[games]\nexpire_after_days = 0\n")

    with pytest.raises(ConfigError, match="expire_after_days"):
        load_config(path, environ={})
