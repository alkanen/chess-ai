"""Renaming a run, and everything in and outside its directory that names it."""

import errno
import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import chess
import pytest
from training_helpers import model_run

from chess_ai import run_rename
from chess_ai.evaluator import Evaluator
from chess_ai.game_session import GameState, SessionRecord
from chess_ai.inference import load_engine
from chess_ai.output import MILESTONE
from chess_ai.players import ModelDescription
from chess_ai.position_view import snapshot
from chess_ai.run_rename import RenameStopped, rename_run, renamed_toml
from chess_ai.sample_games import (
    LIVE_FILE,
    LiveGame,
    SampleGame,
    SampleGamesResult,
    read_sample_games,
    save_live_game,
    stands,
)
from chess_ai.training import checkpoint
from chess_ai.training.run_store import (
    EVALUATING_LOCK,
    RENAME_FILE,
    CheckpointChanged,
    CheckpointPolicy,
    Lineage,
    RunError,
    RunReader,
    RunWriter,
    _as_json,
    _replace_file,
    choose_checkpoint,
    evaluating,
    open_run,
    save_evaluation,
    start_partial_file,
)
from chess_ai.web.app import _fingerprint
from chess_ai.web.game_store import GameLinks, GameStore, ModelRecord, RandomRecord, StoredGame

CONFIG = '# The wide one.\nname = "old"  # what it is called\n\n[model]\narchitecture = "mlp"\n'
CHILD_CONFIG = (
    '[initialize_from]\nrun = "old"  # where the weights come from\ncheckpoint = "best"\n'
)

GAME = """\
[Event "chess-ai match"]
[Site "?"]
[White "old step 2"]
[Black "Random mover"]
[Result "*"]
[WhiteRun "old"]
[WhiteCheckpoint "2"]
[Opening "old"]

{old step 2 replaced by tiny step 2} 1. e4 e5 *
"""


RENAMED_GAME = """\
[Event "chess-ai match"]
[Site "?"]
[White "new step 2"]
[Black "Random mover"]
[Result "*"]
[WhiteRun "new"]
[WhiteCheckpoint "2"]
[Opening "old"]

{new step 2 replaced by tiny step 2} 1. e4 e5 *
"""
"""``GAME`` renamed: in the tags that name a player, and the comments, and nowhere else."""


def sample_games(run: RunReader, step: int) -> None:
    """Save sample games of ``run``'s checkpoint from ``step``, as the evaluator would."""
    info = choose_checkpoint(run, step)
    name = f"{run.name} step {step}"
    result = SampleGamesResult(
        model=ModelDescription(run=run.name, checkpoint=step),
        checkpoint_written=run.checkpoint_written(info),
        started=datetime.now(UTC),
        finished=datetime.now(UTC),
        code_version="test",
        openings="standard v1",
        stockfish=None,
        games=[
            SampleGame(
                index=0,
                opponent="self",
                model_color="both",
                opening="Ruy Lopez",
                white=name,
                black=name,
                result="*",
                termination=None,
                plies=2,
            )
        ],
    )
    pgn = (
        f'[Event "chess-ai sample games"]\n[White "{name}"]\n[Black "{name}"]\n'
        f'[WhiteRun "{run.name}"]\n[BlackRun "{run.name}"]\n\n1. e4 e5 *\n'
    )
    save_evaluation(
        run.evaluation_directory(step),
        {"sample-games.json": result.model_dump_json(indent=2) + "\n", "sample-games.pgn": pgn},
    )


def live_game(run: str, *, updated: datetime) -> LiveGame:
    return LiveGame(
        run=run,
        step=2,
        index=0,
        games=2,
        opponent="self",
        updated=updated,
        game=GameState.model_validate(
            {
                "id": "g",
                "white": {
                    "name": f"{run} step 2",
                    "accepts_moves": False,
                    "model": {"run": run, "checkpoint": 2},
                },
                "black": {"name": f"{run} step 2", "accepts_moves": False},
                "start_fen": chess.STARTING_FEN,
                "moves": [],
                "position": snapshot(chess.Board(), legal_moves=False).model_dump(),
            }
        ),
    )


def ongoing_game(runs: Path, run: str, *, game_id: str = "g1") -> StoredGame:
    path = runs / run / "checkpoints" / "step-000000002.pt"
    now = datetime.now(UTC)
    return StoredGame(
        links=GameLinks(watch=f"watch-{game_id}"),
        white=ModelRecord(
            run=run,
            step=2,
            fingerprint=_fingerprint(path),
            rating=None,
            strategy="argmax",
            temperature=1.0,
            seed=None,
        ),
        black=RandomRecord(),
        session=SessionRecord(
            id=game_id,
            start_fen=chess.STARTING_FEN,
            moves=[],
            move_delay=0,
            started=now,
            updated=now,
        ),
        settings={"white": {"kind": "model", "run": run}, "black": {"kind": "random"}},
    )


class Scene:
    """A runs directory with a run called ``old`` and everything that can name it."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.runs = tmp_path / "runs"
        self.games = tmp_path / "games"
        self.ongoing = tmp_path / "ongoing-games"
        old = model_run(self.runs, "old", steps=(2, 4), config={"name": "old"})
        (old / "config.toml").write_text(CONFIG)
        sample_games(RunReader(old), 2)
        child = model_run(
            self.runs,
            "child",
            steps=(2,),
            config={"name": "child", "initialize_from": {"run": "old", "checkpoint": "best"}},
        )
        info = RunReader(child).info.model_copy(
            update={"initialized_from": Lineage(run="old", step=2, checkpoint="best")}
        )
        _replace_file(child / "run.json", _as_json(info))
        (child / "config.toml").write_text(CHILD_CONFIG)
        model_run(self.runs, "unrelated", steps=(2,))
        save_live_game(self.runs, live_game("old", updated=datetime.now(UTC) - timedelta(hours=1)))
        self.games.mkdir()
        (self.games / "20261015-120000-aaaa.pgn").write_text(GAME)
        (self.games / "20261015-120001-bbbb.pgn").write_text(GAME.replace("old", "unrelated"))
        GameStore(self.ongoing).write(ongoing_game(self.runs, "old"))
        self.said: list[str] = []

    def rename(self, old: str = "old", new: str = "new", **options) -> run_rename.Renamed:
        options.setdefault("stale_after", 120.0)
        return rename_run(
            self.runs,
            old,
            new,
            games_dir=self.games,
            ongoing_dir=self.ongoing,
            say=lambda text, kind=MILESTONE: self.said.append(text),
            **options,
        )

    def files(self) -> dict[Path, tuple[bytes, int]]:
        """Every file under the scene, with what it holds and when it was written."""
        return {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
        }


@pytest.fixture
def scene(tmp_path) -> Scene:
    return Scene(tmp_path)


def written_at(directory: Path) -> dict[str, int]:
    return {path.name: path.stat().st_mtime_ns for path in directory.glob("*.pt")}


def test_a_rename_renames_the_run_and_everything_that_names_it(scene):
    before = written_at(scene.runs / "old" / "checkpoints")
    untouched = {
        path: path.read_bytes()
        for path in [
            *(scene.runs / "unrelated").rglob("*"),
            scene.games / "20261015-120001-bbbb.pgn",
        ]
        if path.is_file() and path.name != "writer.lock"
    }

    renamed = scene.rename()

    assert not (scene.runs / "old").exists()
    run = open_run(scene.runs, "new")
    assert run.info.name == "new" and run.info.config["name"] == "new"
    assert (run.directory / "config.toml").read_text() == CONFIG.replace('"old"', '"new"')
    # The checkpoints name the run, and keep the time they were written, which is what says
    # the results about them still stand.
    assert written_at(run.directory / "checkpoints") == before
    for path in (run.directory / "checkpoints").glob("*.pt"):
        payload = checkpoint.load(path)
        assert payload["run"] == "new" and payload["config"] == {}
    result = read_sample_games(run, 2)
    assert result is not None and result.model.run == "new"
    assert {result.games[0].white, result.games[0].black} == {"new step 2"}
    assert stands(run, choose_checkpoint(run, 2), run.checkpoint_written(choose_checkpoint(run, 2)))
    pgn = (run.evaluation_directory(2) / "sample-games.pgn").read_text()
    assert "old" not in pgn and '[WhiteRun "new"]' in pgn and '[Black "new step 2"]' in pgn
    # The run fine-tuned from it still says where its weights came from.
    child = open_run(scene.runs, "child")
    assert child.info.initialized_from == Lineage(run="new", step=2, checkpoint="best")
    assert child.info.config["initialize_from"] == {"run": "new", "checkpoint": "best"}
    assert child.config_text == CHILD_CONFIG.replace('"old"', '"new"')
    live = LiveGame.model_validate_json((scene.runs / LIVE_FILE).read_text())
    assert live.run == "new" and live.game.white.name == "new step 2"
    assert live.game.white.model == ModelDescription(run="new", checkpoint=2)
    assert (scene.games / "20261015-120000-aaaa.pgn").read_text() == RENAMED_GAME
    (held,) = GameStore(scene.ongoing).read_all()
    assert isinstance(held.white, ModelRecord) and held.white.run == "new"
    assert held.white.fingerprint == _fingerprint(run.checkpoint_path(choose_checkpoint(run, 2)))
    assert held.settings == {"white": {"kind": "model", "run": "new"}, "black": {"kind": "random"}}
    for path, text in untouched.items():
        assert path.read_bytes() == text, f"{path} does not name the run"
    assert json.loads((run.directory / RENAME_FILE).read_text())["finished"] is True
    assert renamed.moved and renamed.live and renamed.playing == 1
    assert (len(renamed.checkpoints), len(renamed.evaluations), renamed.runs) == (2, 2, ["child"])
    assert [path.name for path in renamed.games] == ["20261015-120000-aaaa.pgn"]
    assert [path.name for path in renamed.run_files] == ["config.toml", "run.json"]


def test_a_renamed_run_s_checkpoints_load_and_play_under_the_new_name(scene):
    scene.rename()

    run = open_run(scene.runs, "new")
    engine = load_engine(run.checkpoint_path(choose_checkpoint(run, "best")))

    assert engine.run == "new"
    assert engine.evaluate(chess.Board(), mover_rating=1500, opponent_rating=1500).wdl


def test_a_dry_run_changes_nothing_and_lists_what_would_change(scene):
    before = scene.files()

    renamed = scene.rename(dry_run=True)

    assert scene.files() == before
    assert len(renamed.checkpoints) == 2 and renamed.runs == ["child"] and renamed.live
    listed = [line for line in scene.said if line.startswith("would rewrite ")]
    assert any(line.endswith("step-000000004.pt") for line in listed)
    assert any(line.endswith("sample-games.pgn") for line in listed)
    assert any(line.endswith("20261015-120000-aaaa.pgn") for line in listed)
    assert not any("unrelated" in line or "bbbb" in line for line in listed)


@pytest.mark.parametrize(
    ("old", "new", "refusal"),
    [
        ("old", "unrelated", "is there already"),
        ("old", "../elsewhere", "invalid run name"),
        ("missing", "new", "no run called 'missing'"),
        ("old", "old", "called that already"),
    ],
)
def test_a_rename_that_cannot_be_made_is_refused_and_changes_nothing(scene, old, new, refusal):
    before = scene.files()

    with pytest.raises(RunError, match=refusal):
        scene.rename(old, new)

    assert scene.files() == before


def test_a_name_taken_by_an_empty_directory_is_refused(scene):
    (scene.runs / "new").mkdir()

    with pytest.raises(RunError, match="is there already"):
        scene.rename()

    assert (scene.runs / "old" / "run.json").is_file()


def test_a_run_being_trained_is_not_renamed(scene):
    before = scene.files()

    with (
        RunWriter.reopen(scene.runs / "old", policy=CheckpointPolicy()),
        pytest.raises(RunError, match="is being trained"),
    ):
        scene.rename()

    after = scene.files()
    assert {path: seen for path, seen in after.items() if path.suffix != ".jsonl"} == {
        path: seen for path, seen in before.items() if path.suffix != ".jsonl"
    }


def test_a_run_whose_games_the_evaluator_is_playing_is_not_renamed(scene):
    save_live_game(scene.runs, live_game("old", updated=datetime.now(UTC)))

    with pytest.raises(RunError, match="is playing the games of its checkpoint from step 2"):
        scene.rename()

    assert (scene.runs / "old" / "run.json").is_file()


def failing(*args, **kwargs):
    raise OSError(errno.EIO, "Input/output error")


def test_a_rename_stopped_before_the_move_leaves_the_run_working_under_its_old_name(
    scene, monkeypatch
):
    monkeypatch.setattr(run_rename, "_move", failing)

    with pytest.raises(RenameStopped, match="'old' is still called that.*again finishes it"):
        scene.rename()

    run = open_run(scene.runs, "old")
    engine = load_engine(run.checkpoint_path(choose_checkpoint(run, "latest")))
    assert engine.evaluate(chess.Board(), mover_rating=1500, opponent_rating=1500).wdl
    assert stands(run, choose_checkpoint(run, 2), run.checkpoint_written(choose_checkpoint(run, 2)))
    (held,) = GameStore(scene.ongoing).read_all()
    assert isinstance(held.white, ModelRecord) and held.white.run == "old"
    assert open_run(scene.runs, "child").info.initialized_from.run == "old"

    monkeypatch.undo()
    scene.rename()

    assert not (scene.runs / "old").exists()
    assert open_run(scene.runs, "child").info.initialized_from.run == "new"
    (held,) = GameStore(scene.ongoing).read_all()
    assert held.white.run == "new"


def test_a_rename_stopped_after_the_move_is_finished_by_running_it_again(scene, monkeypatch):
    monkeypatch.setattr(run_rename._Rename, "saved_games", failing)

    with pytest.raises(RenameStopped, match="'old' is 'new' now.*again finishes it"):
        scene.rename()

    assert open_run(scene.runs, "new").info.name == "new"
    assert "old" in (scene.games / "20261015-120000-aaaa.pgn").read_text()
    # Another rename of it would leave the games naming a run that is gone.
    with pytest.raises(RunError, match="still being renamed from 'old'"):
        scene.rename("new", "newer")

    monkeypatch.undo()
    renamed = scene.rename()

    assert any("is called new already" in line for line in scene.said)
    assert '[WhiteRun "new"]' in (scene.games / "20261015-120000-aaaa.pgn").read_text()
    assert [path.name for path in renamed.games] == ["20261015-120000-aaaa.pgn"]
    assert not renamed.checkpoints, "what says the new name already is left alone"
    assert json.loads((scene.runs / "new" / RENAME_FILE).read_text())["finished"] is True
    scene.rename("new", "newer")
    assert open_run(scene.runs, "newer").info.name == "newer"


def test_a_rename_to_another_name_that_stopped_is_overtaken_by_the_next(scene, monkeypatch):
    monkeypatch.setattr(run_rename, "_move", failing)
    with pytest.raises(RenameStopped):
        scene.rename("old", "wrong")
    monkeypatch.undo()

    scene.rename("old", "right")

    run = open_run(scene.runs, "right")
    text = "".join(path.read_text() for path in run.evaluation_directory(2).iterdir())
    assert "wrong" not in text and "right step 2" in text
    assert all(checkpoint.load(path)["run"] == "right" for path in run.directory.glob("*/*.pt"))


def test_running_a_finished_rename_again_renames_what_was_written_back_since(scene):
    scene.rename()
    # As a web server that was running would, playing the game on as the run it began with.
    GameStore(scene.ongoing).write(ongoing_game(scene.runs, "new").model_copy())
    stale = ongoing_game(scene.runs, "new", game_id="g2")
    GameStore(scene.ongoing).write(
        stale.model_copy(update={"white": stale.white.model_copy(update={"run": "old"})})
    )

    renamed = scene.rename()

    assert [path.name for path in renamed.ongoing] == ["g2.json"]
    assert all(game.white.run == "new" for game in GameStore(scene.ongoing).read_all())


def test_a_result_saved_after_its_run_was_renamed_is_not_saved_anywhere(scene):
    old = RunReader(scene.runs / "old")
    scene.rename()

    with pytest.raises(CheckpointChanged):
        save_evaluation(old.evaluation_directory(4), {"probe-positions.json": "{}"})
    with pytest.raises(CheckpointChanged):
        start_partial_file(old.evaluation_directory(4), "sample-games.pgn")

    assert not (scene.runs / "old").exists(), "the old name is not made again"


@pytest.mark.parametrize(
    ("text", "table", "key", "expected"),
    [
        ('name = "old"\n', (), "name", 'name = "new"\n'),
        (
            "name='old' # kept\n[a]\nname = 'old'\n",
            (),
            "name",
            "name='new' # kept\n[a]\nname = 'old'\n",
        ),
        ('[model]\nname = "old"\n', (), "name", '[model]\nname = "old"\n'),
        ('name = "other"\n', (), "name", 'name = "other"\n'),
        (
            '[initialize_from]\nrun = "old"\n',
            ("initialize_from",),
            "run",
            '[initialize_from]\nrun = "new"\n',
        ),
        (
            'initialize_from.run = "old"\n',
            ("initialize_from",),
            "run",
            'initialize_from.run = "new"\n',
        ),
        (
            'initialize_from = { run = "old", checkpoint = "best" }\n',
            ("initialize_from",),
            "run",
            'initialize_from = { run = "new", checkpoint = "best" }\n',
        ),
        (
            "[ initialize_from ]\r\nrun = 'old'\r\n",
            ("initialize_from",),
            "run",
            "[ initialize_from ]\r\nrun = 'new'\r\n",
        ),
    ],
)
def test_a_config_is_renamed_in_the_one_place_it_names_the_run(text, table, key, expected):
    assert renamed_toml(text, table, key, {"old"}, "new") == expected


def test_a_config_that_names_the_run_in_a_way_that_cannot_be_rewritten_says_so():
    with pytest.raises(ValueError, match="cannot find"):
        renamed_toml('name = """old"""\n', (), "name", {"old"}, "new")


def test_a_config_that_cannot_be_rewritten_is_left_as_it_is_and_the_rest_renamed(scene, caplog):
    (scene.runs / "old" / "config.toml").write_text('name = """old"""\n')

    scene.rename()

    assert (scene.runs / "new" / "config.toml").read_text() == 'name = """old"""\n'
    assert "cannot be rewritten" in caplog.text
    assert open_run(scene.runs, "new").info.name == "new"


def test_a_run_an_evaluation_is_going_on_in_is_not_renamed(scene):
    with evaluating(scene.runs / "old"), pytest.raises(RunError, match="is being evaluated"):
        scene.rename()

    assert (scene.runs / "old" / "run.json").is_file()
    assert not (scene.runs / "old" / RENAME_FILE).exists()


class Renaming:
    """A suite that tries to rename the run it is evaluating, as somebody might meanwhile."""

    name = "renaming"

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.refused: list[str] = []

    def wants(self, run, checkpoint) -> bool:
        return True

    def stands(self, run, checkpoint, written) -> bool:
        return bool(self.refused)

    def after(self, run) -> None:
        pass

    def evaluate(self, run, checkpoint, written, engine, *, stop=None) -> str:
        with pytest.raises(RunError) as refused:
            self.scene.rename()
        self.refused.append(str(refused.value))
        return "done"


def test_the_evaluator_keeps_a_run_from_being_renamed_while_it_evaluates_it(scene, monkeypatch):
    for name in ("child", "unrelated"):
        (scene.runs / name / "run.json").unlink()  # Only the run being renamed is evaluated.
    monkeypatch.setattr(
        "chess_ai.evaluator.configured_suites", lambda run: ["renaming"], raising=True
    )
    suite = Renaming(scene)
    evaluator = Evaluator(scene.runs, {suite.name: suite}, load=lambda path: None)

    evaluator.evaluate(evaluator.pending()[0])

    assert suite.refused and "is being evaluated" in suite.refused[0]
    assert (scene.runs / "old" / "run.json").is_file()


def test_an_evaluation_of_a_run_renamed_since_it_was_found_is_passed_over(scene, monkeypatch):
    monkeypatch.setattr("chess_ai.evaluator.configured_suites", lambda run: ["renaming"])
    said: list[str] = []
    evaluator = Evaluator(
        scene.runs,
        {"renaming": Renaming(scene)},
        load=load_engine,
        say=lambda text, kind=MILESTONE: said.append(text),
    )
    job = next(job for job in evaluator.pending() if job.run.name == "old")
    scene.rename()

    assert evaluator.evaluate(job)

    assert any("old@" in line and "renamed or deleted" in line for line in said)
    assert not (scene.runs / "old").exists(), "the old name is not made again"


def test_a_stopped_renames_name_is_only_replaced_in_the_run_it_was_written_in(scene, monkeypatch):
    monkeypatch.setattr(run_rename, "_move", failing)
    with pytest.raises(RenameStopped):
        scene.rename("old", "b")
    monkeypatch.undo()
    # A run of that name since, played and fine-tuned from, which is nothing to do with "old".
    model_run(scene.runs, "b", steps=(2,))
    (scene.games / "20261016-090000-cccc.pgn").write_text(GAME.replace("old", "b"))
    child = RunReader(scene.runs / "child")
    _replace_file(
        child.directory / "run.json",
        _as_json(
            child.info.model_copy(
                update={"initialized_from": Lineage(run="b", step=2, checkpoint="best")}
            )
        ),
    )

    scene.rename("old", "c")

    assert (scene.games / "20261016-090000-cccc.pgn").read_text() == GAME.replace("old", "b")
    assert open_run(scene.runs, "child").info.initialized_from.run == "b"
    run = open_run(scene.runs, "c")
    text = "".join(path.read_text() for path in run.evaluation_directory(2).iterdir())
    assert "b step 2" not in text and "c step 2" in text


def test_a_run_started_again_under_its_name_forgets_how_the_one_it_replaced_was_renamed(scene):
    (scene.runs / "old" / RENAME_FILE).write_text(
        json.dumps({"format_version": 1, "old": "x", "new": "old", "finished": False})
    )
    info = RunReader(scene.runs / "old").info

    RunWriter.create(
        scene.runs / "old", info, config_text="", policy=CheckpointPolicy(), overwrite=True
    ).close()

    assert not (scene.runs / "old" / RENAME_FILE).exists()
    scene.rename()
    assert open_run(scene.runs, "new").info.name == "new"


def test_an_evaluation_that_waited_for_a_rename_is_told_the_run_moved_away(scene):
    held = run_rename._lock_evaluations(scene.runs / "old", "old", dry_run=False)
    assert held is not None
    waited = threading.Event()
    outcome: list[BaseException | None] = []

    def evaluate() -> None:
        try:
            with evaluating(scene.runs / "old", waiting=waited.set):
                outcome.append(None)
        except BaseException as e:  # noqa: BLE001 - whatever it is, the test says
            outcome.append(e)

    evaluation = threading.Thread(target=evaluate)
    evaluation.start()
    assert waited.wait(5), "it says it is waiting before it waits"
    (scene.runs / "old").rename(scene.runs / "new")
    os.close(held)
    evaluation.join(5)

    assert len(outcome) == 1 and isinstance(outcome[0], CheckpointChanged)


def test_an_evaluation_that_has_nothing_to_wait_for_says_nothing(scene):
    waited: list[bool] = []

    with evaluating(scene.runs / "old", waiting=lambda: waited.append(True)):
        pass

    assert waited == []


def test_a_dry_run_holds_nothing_up(scene, monkeypatch):
    """Looking at what a rename would do keeps no evaluation waiting and no trainer out."""
    (scene.runs / "old" / EVALUATING_LOCK).touch()  # As any evaluation of it has left it.
    seen: list[str] = []
    real = run_rename._Rename.checkpoints

    def waited() -> None:
        raise AssertionError("an evaluation waited for a dry run")

    def look_meanwhile(self, home, renamed):
        with evaluating(home, waiting=waited):
            seen.append("evaluated")
        RunWriter.reopen(home, policy=CheckpointPolicy()).close()
        seen.append("written")
        real(self, home, renamed)

    monkeypatch.setattr(run_rename._Rename, "checkpoints", look_meanwhile)

    scene.rename(dry_run=True)

    assert seen == ["evaluated", "written"]
