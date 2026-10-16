"""Renaming a run: its directory, and everything that names it.

A run's name is its directory, and it is also written in its files and in files that are not its
own: in every checkpoint, every evaluation result and the games beside it, in the runs fine-tuned
from it, and in the games played against it. A directory renamed by hand leaves all of those
naming a run that is no longer there. :func:`rename_run` changes all of them, in an order that
keeps the run working whatever stops it part way:

1. Everything in the run directory first, while the directory still has the old name: the
   evaluation results and their games, the checkpoints, ``config.toml`` and ``run.json``. Each
   file is written beside the old one and moved over it, so none is ever half-written, and a
   run whose files say the new name while its directory has the old one still trains, evaluates
   and plays under the old one: the directory is what every reader goes by.
2. Then the directory is moved.
3. Then the files outside it: other runs' lineage, the game the evaluator shows, the games the
   web server holds and the PGN files of the games that were saved.

``rename.json`` in the run directory says what the run was last renamed from, and whether that
rename has finished. A rename that stops part way leaves it saying not, and running the same
rename again finds it and finishes the job, from wherever the run is by then. Running it again
once it has finished does it all again, which is how a game a running web server wrote the old
name back into is renamed after all. Every step leaves alone what already says the new name, so
doing one again is harmless.

A checkpoint is rewritten with the time it was first written, since that is what tells the
evaluator its results still stand and the web server that a game is still playing the same file.

The run's trainer is locked out for as long as the rename takes, by the lock it takes itself, and
so is every evaluation of the run's checkpoints, by the one each of them holds while it goes on
(:func:`~chess_ai.training.run_store.evaluating`): a rename is refused while one is going on, and
one that begins while the run is being renamed waits, then finds the run gone and passes it over
until the evaluator finds it under its new name. ``chess-ai serve`` holds the
games being played in memory, and writes each one's file again as it goes on, so a game against
the run that is still going on is only renamed for good if the server is stopped while the
rename is made.
"""

import fcntl
import json
import logging
import os
import re
import tomllib
from collections.abc import Callable, Collection
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from chess_ai.dataset.files import sync_directory, sync_file
from chess_ai.output import DETAIL, RESULT, Say
from chess_ai.output import say as said_on_stdout
from chess_ai.sample_games import LIVE_FILE, read_live_game
from chess_ai.training import checkpoint
from chess_ai.training.checkpoint import CheckpointError
from chess_ai.training.run_store import (
    CHECKPOINTS_DIR,
    CONFIG_FILE,
    EVALUATING_LOCK,
    EVALUATIONS_DIR,
    PARTIAL_SUFFIX,
    RENAME_FILE,
    RUN_FILE,
    TEMPORARY_SUFFIX,
    RunError,
    RunInfo,
    _as_json,
    _replace_file,
    _WriterLock,
    checkpoint_file,
    evaluation_lock,
    list_runs,
    read_checkpoints,
    valid_name,
)

LOGGER = logging.getLogger(__name__)

FORMAT_VERSION: Final = 1

GAME_SUFFIX: Final = ".pgn"
ONGOING_SUFFIX: Final = ".json"


class RenameStopped(RunError):
    """A rename made some of its changes and not the rest. Running it again finishes it."""


class _Marker(BaseModel):
    """``rename.json``: what a run was last renamed from, and whether that rename finished."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int = FORMAT_VERSION
    old: str
    new: str
    finished: bool = False


@dataclass
class Renamed:
    """What a rename changed, or would change: the files, by where they are."""

    checkpoints: list[Path] = field(default_factory=list)
    evaluations: list[Path] = field(default_factory=list)
    """Relative to the run directory, which moves in the middle of the rename."""
    run_files: list[Path] = field(default_factory=list)
    """``run.json`` and ``config.toml``, relative to the run directory."""
    runs: list[str] = field(default_factory=list)
    """The other runs that name it, as having been initialized from it."""
    games: list[Path] = field(default_factory=list)
    """The PGN files of games that were saved."""
    ongoing: list[Path] = field(default_factory=list)
    """The files of games the web server holds."""
    playing: int = 0
    """How many of :attr:`ongoing` are games still going on."""
    live: bool = False
    """Whether the game the evaluator shows, or showed last, was one of the run's."""
    moved: bool = False


def rename_run(
    runs_dir: Path,
    old: str,
    new: str,
    *,
    games_dir: Path,
    ongoing_dir: Path,
    stale_after: float,
    dry_run: bool = False,
    say: Say = said_on_stdout,
) -> Renamed:
    """Rename the run ``old`` in ``runs_dir`` to ``new``, and everything that names it.

    ``games_dir`` and ``ongoing_dir`` are where the saved games and the web server's games are.
    ``stale_after`` is how many seconds old a game in ``live-game.json`` can be and still be
    being played. ``dry_run`` says what would change and changes nothing.

    Raises:
        RunError: the rename is refused, and nothing has been changed: a name that cannot be a
            run's, no run called ``old``, ``new`` taken, or the run being trained or evaluated.
        RenameStopped: the rename stopped part way; running it again finishes it.
    """
    valid_name(old)
    valid_name(new)
    if old == new:
        raise RunError(f"run {old!r} is called that already")
    source, target = runs_dir / old, runs_dir / new
    if (source / RUN_FILE).is_file():
        home = source
        if os.path.lexists(target):
            raise RunError(f"cannot rename run {old!r} to {new!r}: {target} is there already")
        marker = _read_marker(source)
        if marker is not None and marker.new == old and not marker.finished:
            raise RunError(
                f"run {old!r} is still being renamed from {marker.old!r}: run 'chess-ai runs "
                f"rename {marker.old} {old}' to finish that first"
            )
        # A rename to another name that stopped before the move left that name in the files.
        unfinished = marker is not None and marker.old == old and not marker.finished
        names = {old} | ({marker.new} if marker is not None and unfinished else set())
    elif (marker := _read_marker(target)) is not None and (marker.old, marker.new) == (old, new):
        home = target
        names = {old}
        say(f"chess-ai: run {old} is called {new} already; renaming what still names it {old}")
    else:
        raise RunError(f"no run called {old!r} is kept in {runs_dir}")

    if dry_run:
        say(f"chess-ai: what renaming run {old} to {new} would change; nothing is changed")
    else:
        say(f"chess-ai: renaming run {old} to {new}")
    renamed = Renamed()
    # The name a stopped rename left is only in the files it rewrote, which are the run's own:
    # anything else that names it names some other run of that name.
    inside = _Rename(names, new, dry_run=dry_run, say=say)
    outside = _Rename({old}, new, dry_run=dry_run, say=say)
    lock = _lock_writer(home, old)
    evaluations: int | None = None
    try:
        evaluations = _lock_evaluations(home, old, dry_run=dry_run)
        _check_not_shown(runs_dir, {old}, stale_after)
        if dry_run:
            # Taken only to say whether the rename would be refused: a look at what it would
            # change, which reads every checkpoint, keeps no evaluation and no trainer waiting.
            if evaluations is not None:
                os.close(evaluations)
                evaluations = None
            lock.release()
        if not dry_run:
            _replace_file(home / RENAME_FILE, _as_json(_Marker(old=old, new=new)))
        try:
            inside.evaluations(home, renamed)
            inside.checkpoints(home, renamed)
            inside.run_files(home, renamed)
            if home == source:
                if not dry_run:
                    say(f"chess-ai: moving {source} to {target}", kind=DETAIL)
                    _move(source, target)
                home = target
                renamed.moved = True
        except (OSError, RunError) as e:
            raise _stopped(old, new, renamed, e, dry_run=dry_run) from e
    finally:
        # Let go of once the run has moved: an evaluation that waited for it finds it gone.
        if evaluations is not None:
            os.close(evaluations)
        lock.release()
    try:
        outside.other_runs(runs_dir, home, renamed)
        outside.live_game(runs_dir, renamed)
        outside.ongoing_games(ongoing_dir, home, renamed)
        outside.saved_games(games_dir, renamed)
        if not dry_run:
            _replace_file(home / RENAME_FILE, _as_json(_Marker(old=old, new=new, finished=True)))
    except (OSError, RunError) as e:
        raise _stopped(old, new, renamed, e, dry_run=dry_run) from e
    return renamed


def _stopped(old: str, new: str, renamed: Renamed, error: Exception, *, dry_run: bool) -> RunError:
    if dry_run:
        return RunError(f"cannot tell everything renaming run {old!r} would change: {error}")
    where = (
        f"run {old!r} is {new!r} now, but not everything that names it says so yet"
        if renamed.moved
        else f"run {old!r} is still called that"
    )
    return RenameStopped(
        f"the rename stopped part way ({error}); {where}. Running 'chess-ai runs rename "
        f"{old} {new}' again finishes it"
    )


def _read_marker(directory: Path) -> _Marker | None:
    try:
        return _Marker.model_validate_json((directory / RENAME_FILE).read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as e:
        LOGGER.warning("cannot read %s, so it is not followed: %s", directory / RENAME_FILE, e)
        return None


def _lock_writer(directory: Path, name: str) -> _WriterLock:
    try:
        return _WriterLock.acquire(directory, name)
    except RunError as e:
        if not (directory / RUN_FILE).is_file():
            raise
        raise RunError(
            f"run {name!r} is being trained, or written by another process: stop it before "
            "renaming it"
        ) from e


def _lock_evaluations(directory: Path, name: str, *, dry_run: bool) -> int | None:
    """Refuse a run one of whose checkpoints is being evaluated, by the evaluator or by
    ``chess-ai evaluate`` or ``chess-ai ladder``, and keep any from starting until let go of:
    the descriptor to close, or ``None`` when there is nothing to hold.

    Every evaluation holds :func:`~chess_ai.training.run_store.evaluating` while it goes on,
    which this takes for itself alone. A dry run makes no lock file where there is none yet,
    since nothing can have been evaluated holding one that is not there.
    """
    path = directory / EVALUATING_LOCK
    try:
        descriptor = os.open(path, os.O_RDWR | (0 if dry_run else os.O_CREAT), 0o644)
    except FileNotFoundError:
        return None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RunError(
            f"run {name!r} is being evaluated: the evaluator, 'chess-ai evaluate' or 'chess-ai "
            "ladder' is working on one of its checkpoints. Try again once it is done"
        ) from None
    except OSError as e:
        # A file system that cannot lock: nothing to tell from, as for the writer's lock.
        LOGGER.warning("cannot lock %s, so an evaluation could go on unseen: %s", path, e)
    return descriptor


def _check_not_shown(runs_dir: Path, names: Collection[str], stale_after: float) -> None:
    """Refuse a run the evaluator is playing games of, which it shows in ``live-game.json``."""
    live = read_live_game(runs_dir, now=datetime.now(UTC), stale_after=stale_after)
    if live is not None and live.run in names:
        raise RunError(
            f"run {live.run!r} is being evaluated: the evaluator is playing the games of its "
            f"checkpoint from step {live.step} ({live.suite}). Try again once they are over, or "
            "stop the evaluator"
        )


def _move(source: Path, target: Path) -> None:
    # Looked at again: a run of the new name may have been started since the first look, and
    # a rename onto an empty directory would quietly replace it.
    if os.path.lexists(target):
        raise RunError(f"{target} has appeared since the rename began")
    os.rename(source, target)
    sync_directory(source.parent)


class _Rename:
    """The steps of one rename, each of which leaves alone what says the new name already."""

    def __init__(self, names: Collection[str], new: str, *, dry_run: bool, say: Say) -> None:
        self.names = frozenset(names)
        """The names to replace: the old one, and one a rename that stopped part way left."""
        self.new = new
        self.dry_run = dry_run
        self._say = say
        self._patterns = [
            re.compile(rf"{re.escape(name)}( step \d+)") for name in sorted(self.names)
        ]
        self._in_comments = re.compile(
            rf"(?<![A-Za-z0-9._-])(?:{'|'.join(re.escape(name) for name in sorted(self.names))})"
            r"(?=\s+step\s+\d)"
        )

    def _changed(self, path: Path) -> None:
        """Say that ``path`` was changed, or would be."""
        if self.dry_run:
            self._say(f"would rewrite {path}", kind=RESULT)
        else:
            self._say(f"rewrote {path}", kind=DETAIL)

    def name(self, value: str) -> str:
        """``value`` with the new name, if it is the run's name or a player's name of it, such
        as ``RUN step 1000``; otherwise ``value``."""
        if value in self.names:
            return self.new
        for pattern in self._patterns:
            if match := pattern.fullmatch(value):
                return self.new + match[1]
        return value

    def json(self, value: Any) -> Any:
        """``value`` with the run renamed wherever it names it, at any depth.

        Only under the keys a name is kept under, such as a result's ``model.run`` and a
        player's ``name``: a run can be called ``start``, which a probe position is too.
        """
        if isinstance(value, dict):
            return {
                key: self.name(item)
                if key in _NAME_KEYS and isinstance(item, str)
                else self.json(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.json(item) for item in value]
        return value

    def pgn(self, text: str) -> str:
        """``text``, PGN, with the run renamed in the tags and in the comments.

        The tags are the players' names (``White "RUN step 1000"``) and the run's own
        (``WhiteRun "RUN"``). A comment can say a player was replaced by another, by name.
        The moves are left alone: a run can be called ``e4``.
        """

        def tag(match: re.Match[str]) -> str:
            if match[2] not in _PGN_NAME_TAGS:
                return match[0]
            value = self.name(match[3])
            return match[0] if value == match[3] else f'{match[1]}"{value}"{match[4]}'

        def comment(match: re.Match[str]) -> str:
            return self._in_comments.sub(self.new, match[0])

        text = _PGN_TAG.sub(tag, text)
        return _PGN_COMMENT.sub(comment, text)

    def _rewrite_text(self, path: Path, rewrite: Callable[[str], str], shown: Path) -> bool:
        """Rewrite ``path`` with ``rewrite`` if that changes it, saying so as ``shown``."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        except (OSError, UnicodeDecodeError) as e:
            LOGGER.warning("cannot read %s, so it is left as it is: %s", path, e)
            return False
        if not any(name in text for name in self.names):
            return False  # Most files, which there is no need to read any further.
        try:
            rewritten = rewrite(text)
        except ValueError as e:
            LOGGER.warning("cannot read %s, so it is left as it is: %s", path, e)
            return False
        if rewritten == text:
            return False
        if not self.dry_run:
            _replace_file(path, rewritten, writers="many")
        self._changed(shown)
        return True

    def _json_text(self, text: str) -> str:
        """JSON ``text`` with the run renamed in it, written as it was: indented or not."""
        data = json.loads(text)
        renamed = self.json(data)
        if renamed == data:
            return text
        if "\n " in text:
            written = json.dumps(renamed, indent=2, ensure_ascii=False)
        else:
            written = json.dumps(renamed, separators=(",", ":"), ensure_ascii=False)
        return written + ("\n" if text.endswith("\n") else "")

    def evaluations(self, home: Path, renamed: Renamed) -> None:
        """Rename the run in what each evaluation suite wrote about each checkpoint."""
        try:
            steps = sorted(
                entry.name
                for entry in os.scandir(home / EVALUATIONS_DIR)
                if entry.is_dir(follow_symlinks=False)
            )
        except FileNotFoundError:
            return
        self._say(
            f"chess-ai: looking through the evaluations of {len(steps)} checkpoints", kind=DETAIL
        )
        for step in steps:
            directory = home / EVALUATIONS_DIR / step
            if self.dry_run:
                self._evaluation(home, directory, renamed)
                continue
            # Held as every save holds it, though none should be going on: every evaluation is
            # kept out while the run is renamed.
            with evaluation_lock(directory):
                self._evaluation(home, directory, renamed)

    def _evaluation(self, home: Path, directory: Path, renamed: Renamed) -> None:
        try:
            files = sorted(entry.name for entry in os.scandir(directory) if entry.is_file())
        except FileNotFoundError:
            return
        for file in files:
            if file.endswith((TEMPORARY_SUFFIX, PARTIAL_SUFFIX)):
                continue  # Being written, or left by an evaluation that was stopped.
            if file.endswith(".json"):
                rewrite = self._json_text
            elif file.endswith(GAME_SUFFIX):
                rewrite = self.pgn
            else:
                continue
            shown = (directory / file).relative_to(home)
            rewritten = self._rewrite_text(directory / file, rewrite, home / shown)
            if rewritten:
                renamed.evaluations.append(shown)

    def checkpoints(self, home: Path, renamed: Renamed) -> None:
        """Rename the run in each of its checkpoints, which name it and its config."""
        directory = home / CHECKPOINTS_DIR
        found = read_checkpoints(directory)
        for number, info in enumerate(found, start=1):
            path = directory / info.file
            if not self.dry_run:
                self._say(
                    f"chess-ai: checkpoint {number}/{len(found)}: step {info.step:,}", kind=DETAIL
                )
            try:
                payload = checkpoint.load(path, mmap=True)
            except CheckpointError as e:
                LOGGER.warning("%s, so it is left naming the run as it does", e)
                continue
            config = payload.get("config")
            names_config = isinstance(config, dict) and config.get("name", self.new) != self.new
            if payload.get("run") == self.new and not names_config:
                continue
            if not self.dry_run:
                payload["run"] = self.new
                if names_config:
                    payload["config"] = {**config, "name": self.new}
                _replace_checkpoint(path, payload)
            self._changed(path)
            renamed.checkpoints.append(path.relative_to(home))

    def run_files(self, home: Path, renamed: Renamed) -> None:
        """Rename the run in its ``config.toml``, if it took its name from there, and in its
        ``run.json``, last of all, since it is what says the directory is a run."""
        path = home / CONFIG_FILE
        if self._rewrite_text(
            path, lambda text: self._toml(text, path, (), "name"), home / CONFIG_FILE
        ):
            renamed.run_files.append(Path(CONFIG_FILE))

        def run_json(text: str) -> str:
            info = RunInfo.model_validate_json(text)
            config = dict(info.config)
            if config.get("name", self.new) != self.new:
                config["name"] = self.new
            changed = info.model_copy(update={"name": self.new, "config": config})
            return text if changed == info else _as_json(changed)

        if self._rewrite_text(home / RUN_FILE, run_json, home / RUN_FILE):
            renamed.run_files.append(Path(RUN_FILE))

    def _toml(self, text: str, path: Path, table: tuple[str, ...], key: str) -> str:
        try:
            return renamed_toml(text, table, key, self.names, self.new)
        except ValueError as e:
            LOGGER.warning(
                "%s names the run in a way that cannot be rewritten (%s), so it is left as it "
                "is; it says what the run was started with, and nothing reads it but people",
                path,
                e,
            )
            return text

    def other_runs(self, runs_dir: Path, home: Path, renamed: Renamed) -> None:
        """Rename the run where the runs fine-tuned from it say where their weights came from."""
        for name in list_runs(runs_dir):
            directory = runs_dir / name
            if directory == home or name in self.names or name == self.new:
                continue
            changed = self._rewrite_text(directory / RUN_FILE, self._lineage, directory / RUN_FILE)
            config = directory / CONFIG_FILE
            changed |= self._rewrite_text(
                config,
                lambda text, path=config: self._toml(text, path, ("initialize_from",), "run"),
                config,
            )
            if changed:
                renamed.runs.append(name)

    def _lineage(self, text: str) -> str:
        info = RunInfo.model_validate_json(text)
        update: dict[str, Any] = {}
        lineage = info.initialized_from
        if lineage is not None and lineage.run in self.names:
            update["initialized_from"] = lineage.model_copy(update={"run": self.new})
        source = info.config.get("initialize_from")
        if isinstance(source, dict) and source.get("run") in self.names:
            update["config"] = {**info.config, "initialize_from": {**source, "run": self.new}}
        return _as_json(info.model_copy(update=update)) if update else text

    def live_game(self, runs_dir: Path, renamed: Renamed) -> None:
        """Rename the run in the game the evaluator showed last, if it was one of the run's."""
        path = runs_dir / LIVE_FILE
        renamed.live = self._rewrite_text(path, self._json_text, path)

    def ongoing_games(self, directory: Path, home: Path, renamed: Renamed) -> None:
        """Rename the run in the games the web server holds, which it plays them again from.

        A game also says which file its checkpoint was, so that it is not handed another one
        saved in its place; a checkpoint rewritten by the rename is the same one, and is known
        by the time it was written, which the rewriting keeps.
        """

        def game(text: str) -> str:
            data = json.loads(text)
            renamed_data = self.json(data)
            for side in ("white", "black"):
                record = renamed_data.get(side) if isinstance(renamed_data, dict) else None
                if isinstance(record, dict) and record.get("run") == self.new:
                    _follow_checkpoint(record, home)
            if renamed_data == data:
                return text
            return json.dumps(renamed_data, separators=(",", ":"), ensure_ascii=False)

        try:
            paths = sorted(directory.glob(f"*{ONGOING_SUFFIX}"))
        except OSError:
            return
        for path in paths:
            if self._rewrite_text(path, game, path):
                renamed.ongoing.append(path)
                if _still_going_on(path):
                    renamed.playing += 1

    def saved_games(self, directory: Path, renamed: Renamed) -> None:
        """Rename the run in the PGN files of the games that were saved."""
        try:
            paths = sorted(directory.glob(f"*{GAME_SUFFIX}"))
        except OSError:
            return
        if paths:
            self._say(f"chess-ai: looking through {len(paths)} saved games", kind=DETAIL)
        for path in paths:
            if self._rewrite_text(path, self.pgn, path):
                renamed.games.append(path)


_NAME_KEYS: Final = frozenset({"run", "name", "white", "black"})
"""The keys a run's name is kept under in JSON: by itself, or as a player's ``RUN step N``."""
_PGN_NAME_TAGS: Final = frozenset({"White", "Black", "WhiteRun", "BlackRun"})
_PGN_TAG: Final = re.compile(
    r'^(\[([A-Za-z0-9_]+)[ \t]+)"((?:[^"\\\n]|\\.)*)"([ \t]*\][ \t]*)$', re.MULTILINE
)
_PGN_COMMENT: Final = re.compile(r"\{[^}]*\}")


def _replace_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` in place of the checkpoint at ``path``, with the time it was written."""
    written = path.stat()
    temporary = path.with_name(path.name + TEMPORARY_SUFFIX)
    try:
        checkpoint.save(payload, temporary)
        with temporary.open("rb") as f:
            sync_file(f)
        os.utime(temporary, ns=(written.st_atime_ns, written.st_mtime_ns))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def _follow_checkpoint(record: dict[str, Any], home: Path) -> None:
    """Give a game's ``record`` of a player the fingerprint of its checkpoint as rewritten.

    Only for the file it began with, written at the same time: one saved in its place since is
    another checkpoint, which the game is not to go on with.
    """
    fingerprint = record.get("fingerprint")
    step = record.get("step")
    if not isinstance(fingerprint, list) or len(fingerprint) != 3 or not isinstance(step, int):
        return
    with suppress(OSError):
        found = (home / CHECKPOINTS_DIR / checkpoint_file(step)).stat()
        if found.st_mtime_ns == fingerprint[2]:
            record["fingerprint"] = [found.st_ino, found.st_size, found.st_mtime_ns]


def _still_going_on(path: Path) -> bool:
    """Whether the game in ``path`` has not ended."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    session = data.get("session") if isinstance(data, dict) else None
    return isinstance(session, dict) and session.get("game_over") is None


_TABLE: Final = re.compile(r"[ \t]*\[([^\[\]]*)\][ \t]*(?:#.*)?")
_ARRAY_OF_TABLES: Final = re.compile(r"[ \t]*\[\[.*")


def renamed_toml(
    text: str, table: tuple[str, ...], key: str, names: Collection[str], new: str
) -> str:
    """TOML ``text`` with the string ``key`` of ``table`` (``()`` for the top) set to ``new``,
    where it is one of ``names``, and every other line as it was: comments and all.

    Raises:
        ValueError: ``text`` is not TOML, or names the run in a way this cannot rewrite, such
            as a multi-line string.
    """
    try:
        before = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"not TOML: {e}") from e
    section: Any = before
    for part in table:
        section = section.get(part) if isinstance(section, dict) else None
    if not isinstance(section, dict) or section.get(key) not in names:
        return text
    old = section[key]
    expected = _with(before, (*table, key), new)
    value = rf"""(?P<quote>["']){re.escape(old)}(?P=quote)"""
    lines = text.splitlines(keepends=True)
    current: tuple[str, ...] | None = ()
    for number, line in enumerate(lines):
        content = line.rstrip("\r\n")
        if _ARRAY_OF_TABLES.match(content):
            current = None
            continue
        if header := _TABLE.fullmatch(content):
            current = tuple(part.strip().strip("\"'") for part in header[1].split("."))
            continue
        if current is None or current != table[: len(current)]:
            continue
        # The key as it is written under the current table: dotted for the rest of the table's
        # path, or inside an inline table of it, which the check below tells apart.
        rest = table[len(current) :]
        dotted = r"\s*\.\s*".join(re.escape(part) for part in (*rest, key))
        patterns = [rf"^(\s*{dotted}\s*=\s*){value}"]
        if rest:
            patterns.append(rf"(?<![A-Za-z0-9_.-])({re.escape(key)}\s*=\s*){value}")
        for pattern in patterns:
            candidate = re.sub(
                pattern, lambda m: f"{m[1]}{m['quote']}{new}{m['quote']}", content, count=1
            )
            if candidate == content:
                continue
            rewritten = "".join(
                [*lines[:number], candidate + line[len(content) :], *lines[number + 1 :]]
            )
            with suppress(tomllib.TOMLDecodeError):
                if tomllib.loads(rewritten) == expected:
                    return rewritten
    raise ValueError(f"cannot find where {'.'.join((*table, key))} = {old!r} is written")


def _with(data: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """``data`` with the value at ``path`` replaced by ``value``."""
    head, *rest = path
    if not rest:
        return {**data, head: value}
    return {**data, head: _with(data[head], tuple(rest), value)}
