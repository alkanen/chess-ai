"""``chess-ai tournament``: starting, resuming, showing and listing round-robin tournaments."""

import pytest
from pgn_helpers import read_back
from stockfish_helpers import fake_engine, started
from training_helpers import model_run

from chess_ai.cli import main
from chess_ai.tournament import open_tournament


def tournament_config(tmp_path, **sections: str):
    """A config whose runs directory has a two-checkpoint run, ``tiny``, and whose Stockfish is
    the stand-in engine."""
    model_run(tmp_path / "runs")
    engine = fake_engine(tmp_path / "engine")
    config = tmp_path / "tournament.toml"
    text = (
        f'[paths]\nruns = "{tmp_path / "runs"}"\ntournaments = "{tmp_path / "tournaments"}"\n'
        f'\n[stockfish]\npath = "{engine}"\n'
    )
    for section, body in sections.items():
        text += f"\n[{section}]\n{body}\n"
    config.write_text(text)
    return config


def test_a_tournament_is_started_played_and_its_leaderboard_printed(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='rating = 1700\ndevice = "cpu"')

    status = main(
        [
            "--config", str(config), "tournament", "start", "cup",
            "tiny@2,rating=1500", "tiny@latest", "stockfish:1400,move-time=0.01",
            "--games-per-pairing", "2",
        ]
    )  # fmt: skip

    out = capsys.readouterr().out
    assert status == 0
    tournament = open_tournament(tmp_path / "tournaments", "cup")
    games = tournament.games()
    assert len(games) == 6
    entrants = tournament.definition.entrants
    assert (entrants[1].step, entrants[1].rating) == (4, 1700), "latest, at the ladder's rating"
    first = read_back(games[0].pgn).headers
    assert (first["WhiteRating"], first["WhiteOpponentRating"]) == ("1700", "1500")
    assert (first["BlackRating"], first["BlackOpponentRating"]) == ("1500", "1700")
    assert "game 1/6, " in out and "game 6/6, " in out
    assert "cup: 6 of 6 games" in out
    for name in ("tiny step 2 (rating 1500)", "tiny step 4 (rating 1700)", "Stockfish 1400"):
        assert name in out


def start(config, *arguments: str) -> int:
    """The status ``chess-ai tournament start`` ends with, a refusal's included."""
    try:
        return main(["--config", str(config), "tournament", "start", *arguments])
    except SystemExit as e:
        return int(e.code or 0)


def test_resume_plays_only_what_a_stopped_tournament_had_left(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    assert start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "4") == 0
    games = tmp_path / "tournaments" / "cup" / "games.jsonl"
    lines = games.read_text().splitlines(keepends=True)
    games.write_text("".join(lines[:1]) + lines[1][:30])  # Stopped while writing the second.
    capsys.readouterr()

    status = main(["--config", str(config), "tournament", "resume", "cup"])

    out = capsys.readouterr().out
    assert status == 0
    assert "1 played already" in out
    assert "game 1/4" not in out and "game 2/4, " in out and "game 4/4, " in out
    played = open_tournament(tmp_path / "tournaments", "cup").games()
    assert [game.round for game in played] == [1, 2, 3, 4]


def test_show_and_list_print_only_results_under_qq(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    assert start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "2") == 0
    capsys.readouterr()

    assert main(["--config", str(config), "-qq", "tournament", "show", "cup"]) == 0
    shown = capsys.readouterr().out
    assert main(["--config", str(config), "-qq", "tournament", "list"]) == 0
    listed = capsys.readouterr().out

    assert shown.startswith("cup: 2 of 2 games, rated around an average of 0")
    assert "tiny step 2 (rating 2000)" in shown and "crosstable" in shown
    assert listed == "cup: 2 entrants, 2 of 2 games\n"


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        (["cup", "tiny@2"], "at least two entrants"),
        (["cup", "tiny@2", "nothing"], "nothing"),
        (["cup", "tiny@2", "tiny@4", "--anchor", "3=1500"], "there is no entrant 3"),
        (["cup", "tiny@2", "tiny@4", "--anchor", "1=ladder"], "no Stockfish ladder at step 2"),
        (["cup", "tiny@2", "stockfish:1400", "--anchor", "2=ladder"], "entrant 2 is Stockfish"),
        (["../cup", "tiny@2", "tiny@4"], "invalid tournament name"),
        (["cup", "tiny@2", "tiny@4", "--openings", "nowhere"], "nowhere"),
    ],
)
def test_start_refuses_what_it_cannot_play(tmp_path, capsys, arguments, complaint):
    config = tournament_config(tmp_path)

    assert start(config, *arguments) != 0

    assert complaint in capsys.readouterr().err
    assert not (tmp_path / "tournaments" / "cup").exists()


def test_a_name_that_is_taken_is_refused(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    assert start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "1") == 0

    assert start(config, "cup", "tiny@2", "tiny@4") != 0

    assert "there is a tournament called 'cup' already" in capsys.readouterr().err
    assert len(open_tournament(tmp_path / "tournaments", "cup").games()) == 1


def test_an_anchor_from_the_ladder_takes_the_checkpoints_ladder_elo(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    write_ladder(tmp_path / "runs" / "tiny", step=4, rating=1612.4)

    assert (
        start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "1", "--anchor", "2=ladder")
        == 0
    )

    definition = open_tournament(tmp_path / "tournaments", "cup").definition
    assert (definition.anchor.entrant, definition.anchor.rating) == (1, 1612.4)
    assert "rated from entrant 2 at 1612" in capsys.readouterr().out


def write_ladder(run_dir, *, step: int, rating: float, checkpoint_written=None) -> None:
    """A Stockfish ladder result for the checkpoint of ``run_dir`` from ``step``, as the ladder
    saves one, that rates it at ``rating``: played on the file there now, unless
    ``checkpoint_written`` says it was played on one written at another time."""
    from datetime import UTC, datetime

    from chess_ai.ladder import LadderEstimate, LadderResult, result_path
    from chess_ai.players import ModelDescription
    from chess_ai.training.run_store import RunReader

    now = datetime.now(UTC)
    result = LadderResult(
        model=ModelDescription(run=run_dir.name, checkpoint=step, rating=2000, strategy="argmax"),
        checkpoint_written=checkpoint_written,
        started=now,
        finished=now,
        code_version="test",
        openings="standard v1",
        games_per_level=20,
        levels=[],
        estimate=LadderEstimate(
            rating=rating,
            low=rating - 50,
            high=rating + 50,
            beyond=None,
            floor=1350,
            ceiling=2100,
            confidence=0.95,
            games=100,
            points=50,
        ),
    )
    path = result_path(RunReader(run_dir), step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json())


def test_a_stockfish_that_dies_stops_the_tournament_with_its_games_kept(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    fake_engine(tmp_path / "engine", "--exit-after", "20")

    status = start(
        config, "cup", "tiny@2", "stockfish:1400,move-time=0.01", "--games-per-pairing", "4"
    )

    err = capsys.readouterr().err
    assert status == 2
    assert "Stockfish stopped" in err
    assert "chess-ai tournament resume cup plays it again" in err
    kept = open_tournament(tmp_path / "tournaments", "cup").games()
    assert 0 < len(kept) < 4


def test_ctrl_c_stops_a_tournament_with_its_games_kept_and_says_how_to_resume(
    tmp_path, capsys, monkeypatch
):
    from chess_ai import tournament

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    play_game, games = tournament.play_game, 0

    async def interrupted_at_the_third(*args, **kwargs):
        nonlocal games
        games += 1
        if games == 3:
            raise KeyboardInterrupt
        return await play_game(*args, **kwargs)

    monkeypatch.setattr(tournament, "play_game", interrupted_at_the_third)

    status = start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "4")

    captured = capsys.readouterr()
    assert status == 130
    assert (
        "interrupted after 2 of 4 games; chess-ai tournament resume cup plays the rest"
        in captured.err
    )
    assert "cup: 2 of 4 games" in captured.out


def run(config, *arguments: str) -> int:
    """The status a ``chess-ai`` command ends with, a refusal's included."""
    try:
        return main(["--config", str(config), *arguments])
    except SystemExit as e:
        return int(e.code or 0)


def test_a_tournament_whose_games_cannot_be_read_is_reported_without_a_traceback(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    for name in ("broken", "fine"):
        assert start(config, name, "tiny@2", "tiny@4", "--games-per-pairing", "1") == 0
    with (tmp_path / "tournaments" / "broken" / "games.jsonl").open("a") as file:
        file.write("not a game\n")
    capsys.readouterr()

    shown = run(config, "tournament", "show", "broken")
    shown_err = capsys.readouterr().err
    resumed = run(config, "tournament", "resume", "broken")
    resumed_err = capsys.readouterr().err
    listed = run(config, "tournament", "list")
    listing = capsys.readouterr()

    assert (shown, resumed) == (2, 2)
    assert "line 2" in shown_err and "Traceback" not in shown_err
    assert "line 2" in resumed_err and "Traceback" not in resumed_err
    assert listed == 0
    assert "fine: 2 entrants, 1 of 1 games" in listing.out, "the rest are still listed"
    assert "line 2" in listing.err


def test_resuming_after_the_opening_file_has_gone_says_so_without_a_traceback(tmp_path, capsys):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    openings = tmp_path / "mine.toml"
    openings.write_text(
        'name = "mine"\nversion = 1\n\n[[openings]]\nname = "Open"\nmoves = "e4 e5"\n'
    )
    assert (
        start(
            config,
            "cup",
            "tiny@2",
            "tiny@4",
            "--games-per-pairing",
            "1",
            "--openings",
            str(openings),
        )
        == 0
    )
    openings.unlink()
    capsys.readouterr()

    status = run(config, "tournament", "resume", "cup")

    err = capsys.readouterr().err
    assert status == 2
    assert str(openings) in err and "Traceback" not in err


@pytest.mark.parametrize(
    ("entrant", "complaint"),
    [
        ("tiny@4,rating=1500", "rating 2000"),
        ("tiny@4,strategy=sample", "argmax"),
    ],
)
def test_an_anchor_from_a_ladder_played_another_way_is_refused(
    tmp_path, capsys, entrant, complaint
):
    config = tournament_config(tmp_path, ladder='device = "cpu"')
    write_ladder(tmp_path / "runs" / "tiny", step=4, rating=1612.4)

    status = start(config, "cup", "tiny@2", entrant, "--anchor", "2=ladder")

    err = capsys.readouterr().err
    assert status == 2
    assert complaint in err and "give its rating instead" in err
    assert not (tmp_path / "tournaments" / "cup").exists()


def test_an_anchor_from_a_ladder_of_another_file_of_the_checkpoint_is_refused(tmp_path, capsys):
    from datetime import UTC, datetime

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    # As a run resumed from an earlier step and trained past this one again leaves it.
    write_ladder(
        tmp_path / "runs" / "tiny",
        step=4,
        rating=1612.4,
        checkpoint_written=datetime(2026, 1, 1, tzinfo=UTC),
    )

    status = start(config, "cup", "tiny@2", "tiny@4", "--anchor", "2=ladder")

    err = capsys.readouterr().err
    assert status == 2
    assert "another file" in err and "give its rating instead" in err


def test_resuming_a_tournament_being_played_loads_nothing_before_it_is_refused(tmp_path, capsys):
    import fcntl
    import os

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    engine = tmp_path / "engine" / "fake-stockfish"
    assert (
        start(config, "cup", "tiny@2", "stockfish:1400,move-time=0.01", "--games-per-pairing", "1")
        == 0
    )
    before = started(engine)
    lock = tmp_path / "tournaments" / "cup" / "playing.lock"
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)  # As the process playing it holds it.
        capsys.readouterr()
        status = run(config, "tournament", "resume", "cup")
    finally:
        os.close(descriptor)

    captured = capsys.readouterr()
    assert status == 2
    assert "being played by another process" in captured.err
    assert captured.out == "", "nothing said as if it were about to be played"
    assert started(engine) == before, "no Stockfish was started for it"


def test_a_tournament_that_finishes_while_it_is_listed_is_not_listed_as_stopped_short(
    tmp_path, capsys, monkeypatch
):
    import fcntl
    import os

    from chess_ai import tournament

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    assert start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "2") == 0
    games = tmp_path / "tournaments" / "cup" / "games.jsonl"
    last = games.read_text().splitlines(keepends=True)[-1]
    games.write_text("".join(games.read_text().splitlines(keepends=True)[:-1]))
    descriptor = os.open(games.parent / "playing.lock", os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    read = tournament.Tournament.games

    def finishing(self):
        # The last game ends, and its player with it, just after the games are read.
        found = read(self)
        with games.open("a") as file:
            file.write(last)
        os.close(descriptor)
        return found

    monkeypatch.setattr(tournament.Tournament, "games", finishing)
    capsys.readouterr()

    assert run(config, "-qq", "tournament", "list") == 0

    assert capsys.readouterr().out in (
        "cup: 2 entrants, 1 of 2 games, being played\n",
        "cup: 2 entrants, 2 of 2 games\n",
    )


def test_an_anchor_is_the_elo_of_the_ladder_it_was_checked_against(tmp_path, capsys, monkeypatch):
    from datetime import UTC, datetime

    from chess_ai import ladder

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    tiny = tmp_path / "runs" / "tiny"
    write_ladder(tiny, step=4, rating=1500.0, checkpoint_written=datetime(2026, 1, 1, tzinfo=UTC))
    stale = ladder.read_ladder(ladder_run(tiny), 4)
    write_ladder(tiny, step=4, rating=1700.0)
    fresh = ladder.read_ladder(ladder_run(tiny), 4)
    reads = iter([stale])

    def evaluator_lays_a_new_one_meanwhile(run, step):
        # The stale ladder is read first; the evaluator's new one is there from then on.
        return next(reads, fresh)

    monkeypatch.setattr(ladder, "read_ladder", evaluator_lays_a_new_one_meanwhile)

    status = start(config, "cup", "tiny@2", "tiny@4", "--anchor", "2=ladder")

    # Decided on the stale ladder it read, which it refuses, rather than lent its Elo by a
    # check of the fresh one.
    assert status == 2
    assert "another file" in capsys.readouterr().err
    assert not (tmp_path / "tournaments" / "cup").exists()


def ladder_run(directory):
    from chess_ai.training.run_store import RunReader

    return RunReader(directory)


def test_resuming_after_a_checkpoint_has_been_written_again_is_refused(tmp_path, capsys):
    import os

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    assert start(config, "cup", "tiny@2", "tiny@4", "--games-per-pairing", "2") == 0
    games = tmp_path / "tournaments" / "cup" / "games.jsonl"
    games.write_text(games.read_text().splitlines(keepends=True)[0])
    # As a run resumed from an earlier step and trained past this one again leaves it.
    checkpoint = next((tmp_path / "runs" / "tiny").rglob("*000004*.pt"))
    os.utime(checkpoint, (1_700_000_000, 1_700_000_000))
    capsys.readouterr()

    status = run(config, "tournament", "resume", "cup")

    err = capsys.readouterr().err
    assert status == 2
    assert "entrant 2" in err and "has changed since the tournament started" in err
    assert len(open_tournament(tmp_path / "tournaments", "cup").games()) == 1


def test_a_checkpoint_pruned_as_the_tournament_starts_is_refused_there(
    tmp_path, capsys, monkeypatch
):
    from chess_ai.training.run_store import RunReader

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    # Gone between the run's checkpoints being listed and the file being looked at, as a
    # run still training prunes one.
    monkeypatch.setattr(RunReader, "checkpoint_written", lambda self, info: None)

    status = start(config, "cup", "tiny@2", "tiny@4")

    assert status == 2
    assert "is gone; run chess-ai tournament start again" in capsys.readouterr().err
    assert not (tmp_path / "tournaments" / "cup").exists()


def test_entrants_of_one_checkpoint_and_its_anchor_go_by_one_look_at_its_file(
    tmp_path, capsys, monkeypatch
):
    from datetime import timedelta

    from chess_ai.training.run_store import RunReader, choose_checkpoint

    config = tournament_config(tmp_path, ladder='device = "cpu"')
    tiny = RunReader(tmp_path / "runs" / "tiny")
    first = tiny.checkpoint_written(choose_checkpoint(tiny, 4))
    # The ladder was played on the file as the first look finds it; any later look finds a
    # later time, as a file saved again in between would have.
    write_ladder(tmp_path / "runs" / "tiny", step=4, rating=1612.4, checkpoint_written=first)
    looks = 0

    def later_each_look(self, info):
        nonlocal looks
        looks += 1
        return first + timedelta(microseconds=looks - 1)

    monkeypatch.setattr(RunReader, "checkpoint_written", later_each_look)

    status = start(
        config,
        "cup",
        "tiny@4,rating=2000",
        "tiny@4,rating=2000,strategy=sample",
        "--games-per-pairing",
        "1",
        "--anchor",
        "1=ladder",
    )

    definition = open_tournament(tmp_path / "tournaments", "cup").definition
    assert [entrant.checkpoint_written for entrant in definition.entrants] == [first, first]
    assert definition.anchor.rating == 1612.4, "the ladder of the file the entrants record"
    # Made, and then refused its games: the file's time moved on before they were played.
    assert status == 2
    assert "has changed since the tournament started" in capsys.readouterr().err
