"""The experiment config: what it means when a section is left out, and what it says when wrong."""

import json
import re
import textwrap
import tomllib
import typing
from pathlib import Path

import pytest

from chess_ai.encoders import ENCODERS, encoder_defaults
from chess_ai.models import MODELS, architecture_defaults
from chess_ai.training.experiment import (
    VALIDATION_METRICS,
    ExperimentConfig,
    ExperimentError,
    load_experiment,
)


def config(tmp_path, body: str, name: str = "run") -> ExperimentConfig:
    """Load a config written from ``body``, which may leave out everything but the dataset."""
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return load_experiment(path)


MINIMAL = """
    [dataset]
    name = "games"
"""


def test_only_the_dataset_has_to_be_named(tmp_path):
    loaded = config(tmp_path, MINIMAL)

    assert loaded.dataset.name == "games"
    assert loaded.encoder.name == "board-planes"
    assert loaded.model.architecture == "mlp"
    assert loaded.training.device == "auto"
    assert loaded.training.mixed_precision is True
    assert loaded.optimizer.learning_rate == 1e-3
    assert loaded.checkpoints.metric == "policy_loss"


def test_a_config_that_does_not_name_the_run_is_named_after_its_file(tmp_path):
    assert config(tmp_path, MINIMAL, name="mlp-baseline").name == "mlp-baseline"


def test_the_config_can_name_the_run_itself(tmp_path):
    assert config(tmp_path, f'name = "chosen"\n{MINIMAL}', name="ignored").name == "chosen"


def test_a_name_given_on_the_command_line_wins(tmp_path):
    path = tmp_path / "in-file.toml"
    path.write_text(f'name = "in-file"\n{textwrap.dedent(MINIMAL)}')

    assert load_experiment(path, name="from-the-command-line").name == "from-the-command-line"


def test_a_run_name_that_is_not_a_directory_name_is_refused(tmp_path):
    with pytest.raises(ExperimentError, match="invalid run name '../escape'"):
        config(tmp_path, f'name = "../escape"\n{MINIMAL}')


def test_encoder_and_model_options_are_whatever_the_named_thing_takes(tmp_path):
    loaded = config(
        tmp_path,
        """
        [dataset]
        name = "games"

        [encoder]
        name = "board-planes"
        rating_scale = 4000.0

        [model]
        architecture = "mlp"
        depth = 5
        width = 256
        """,
    )

    assert loaded.encoder.options == {"rating_scale": 4000.0}
    assert loaded.model.options == {"depth": 5, "width": 256}


def test_a_section_with_a_misspelt_setting_of_its_own_is_refused(tmp_path):
    """The open sections pass unknown keys on; the closed ones must not."""
    with pytest.raises(ExperimentError, match="training.batch_sixe"):
        config(
            tmp_path,
            f"{MINIMAL}\n[training]\nbatch_sixe = 64\n",
        )


def test_a_missing_dataset_section_says_so(tmp_path):
    with pytest.raises(ExperimentError, match="dataset: Field required"):
        config(tmp_path, 'name = "run"\n')


def test_a_setting_of_the_wrong_shape_says_which(tmp_path):
    with pytest.raises(ExperimentError, match="training.batch_size.*greater than 0"):
        config(tmp_path, f"{MINIMAL}\n[training]\nbatch_size = 0\n")


def test_warmup_longer_than_the_whole_run_is_refused(tmp_path):
    with pytest.raises(ExperimentError, match="warmup_steps 500 is longer than the whole run"):
        config(tmp_path, f"{MINIMAL}\n[schedule]\nsteps = 100\nwarmup_steps = 500\n")


def test_an_unknown_checkpoint_metric_lists_the_ones_there_are(tmp_path):
    with pytest.raises(ExperimentError, match="unknown metric 'accuracy'.*top1"):
        config(tmp_path, f'{MINIMAL}\n[checkpoints]\nmetric = "accuracy"\n')


@pytest.mark.parametrize("metric", sorted(VALIDATION_METRICS))
def test_every_metric_a_config_may_name_becomes_a_retention_policy(tmp_path, metric):
    loaded = config(tmp_path, f'{MINIMAL}\n[checkpoints]\nmetric = "{metric}"\nkeep = 5\n')

    policy = loaded.checkpoints.policy()
    assert (policy.metric, policy.keep) == (metric, 5)
    assert policy.higher_is_better == VALIDATION_METRICS[metric]


def test_losses_are_best_low_and_accuracies_best_high(tmp_path):
    """Which way a metric is read decides which checkpoint is kept, so it is worth pinning down."""
    assert VALIDATION_METRICS["policy_loss"] is False
    assert VALIDATION_METRICS["illegal_top_move_rate"] is False
    assert VALIDATION_METRICS["top1"] is True


def test_a_negative_seed_is_refused_at_load_time(tmp_path):
    """numpy's SeedSequence rejects negative entropy, so the shuffle would die on the first
    batch — after the run directory had been written and the throughput probe had swallowed
    the error into a warning. It is a config mistake, so it belongs here."""
    with pytest.raises(ExperimentError, match="seed.*greater than or equal to 0"):
        config(tmp_path, f"seed = -1\n{MINIMAL}")


def test_a_seed_beyond_what_the_generators_accept_is_refused_at_load_time(tmp_path):
    """The top of the range fails the same way the bottom did, in the other library.

    ``tomllib`` does not enforce TOML's 64-bit integer range, so a config really can carry a
    seed this large; ``torch.manual_seed`` then raises a bare ``ValueError`` out of ``_prepare``.
    """
    with pytest.raises(ExperimentError, match="seed"):
        config(tmp_path, f"seed = {2**64}\n{MINIMAL}")


def test_the_largest_seed_the_generators_accept_is_allowed(tmp_path):
    """torch's documented range tops out at 0xffff_ffff_ffff_ffff, not at 2**63 - 1."""
    assert config(tmp_path, f"seed = {2**64 - 1}\n{MINIMAL}").seed == 2**64 - 1


def test_a_device_that_is_not_a_device_is_refused(tmp_path):
    with pytest.raises(ExperimentError, match="training.device"):
        config(tmp_path, f'{MINIMAL}\n[training]\ndevice = "gpu"\n')


def test_invalid_toml_says_where(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[dataset\nname = 'games'\n")

    with pytest.raises(ExperimentError, match="invalid TOML"):
        load_experiment(path)


def test_a_config_that_is_not_there_says_so(tmp_path):
    with pytest.raises(ExperimentError, match="cannot read experiment config"):
        load_experiment(tmp_path / "missing.toml")


def test_the_resolved_config_has_every_default_filled_in(tmp_path):
    resolved = config(tmp_path, MINIMAL).resolved()

    assert resolved["training"]["batch_size"] == 1024
    assert resolved["schedule"]["warmup_steps"] == 500
    assert resolved["encoder"]["name"] == "board-planes"
    json.dumps(resolved)  # It goes into run.json and into every checkpoint.


def test_a_run_starts_from_its_seed_unless_told_otherwise(tmp_path):
    assert config(tmp_path, MINIMAL).initialize_from is None


@pytest.mark.parametrize(
    ("written", "chosen"),
    [("", "best"), ('checkpoint = "latest"', "latest"), ("checkpoint = 3000", 3000)],
)
def test_a_run_can_start_from_another_runs_checkpoint(tmp_path, written, chosen):
    loaded = config(tmp_path, MINIMAL + f'[initialize_from]\nrun = "pretrained"\n{written}\n')

    assert loaded.initialize_from.run == "pretrained"
    assert loaded.initialize_from.checkpoint == chosen


@pytest.mark.parametrize(
    ("written", "complaint"),
    [
        ("checkpoint = -1", "not negative"),
        ('checkpoint = "worst"', "initialize_from.checkpoint"),
        ('run = "../elsewhere"', "invalid run name"),
        ("", "initialize_from.run: Field required"),
    ],
)
def test_a_checkpoint_to_start_from_that_cannot_be_one_is_refused(tmp_path, written, complaint):
    body = (
        MINIMAL
        + "[initialize_from]\n"
        + ('run = "pretrained"\n' if "run" not in written and written else "")
        + written
        + "\n"
    )

    with pytest.raises(ExperimentError, match=re.escape(complaint)):
        config(tmp_path, body)


EXPERIMENTS = Path(__file__).parent.parent / "experiments"
"""The configs shipped with the repo, by absolute path: the suite runs elsewhere."""

DEFAULT_NOTE = re.compile(
    r"^(?P<key>[a-z_0-9]+) = (?P<value>.*?)\s*#\s*default:\s*(?P<default>.+)$"
)
SETTING = re.compile(r"^(?P<key>[a-z_0-9]+) = (?P<value>.*?)\s*(?:#.*)?$")
SECTION = re.compile(r"^\[(?P<name>[a-z_]+)\]")
COMMENTED_SECTION = re.compile(r"^# \[(?P<name>[a-z_]+)\]")
"""An optional section, shown commented out: a config documents it without asking for it."""

COMMENTED_SETTING = re.compile(r"^# [a-z_0-9]+ = ")
"""An option whose default cannot be written in TOML, shown commented out with an example."""

NO_DEFAULT = {"name", "dataset.name", "dataset.version", "initialize_from.run"}
"""Settings with nothing to compare against: the run's name follows the file, the dataset has to
be named, the latest version of it has no number, and a run to initialize from has to be named."""


def shipped() -> list[Path]:
    files = sorted(EXPERIMENTS.glob("*.toml"))
    assert files, f"no experiment configs in {EXPERIMENTS}"
    return files


def sections() -> dict[str, type]:
    """The config's sections, by the table name they appear under, the optional ones included."""
    found = {}
    for name, field in ExperimentConfig.model_fields.items():
        for candidate in (field.annotation, *typing.get_args(field.annotation)):
            if hasattr(candidate, "model_fields"):
                found[name] = candidate
    return found


def settable(section: str, data: dict) -> set[str]:
    """Every option a section accepts, including those the thing it names brings with it."""
    if not section:
        return {name for name in ExperimentConfig.model_fields if name not in sections()}
    own = set(sections()[section].model_fields)
    if section == "encoder":
        own |= set(ENCODERS.options(data["encoder"]["name"]))
    if section == "model":
        own |= set(MODELS.options(data["model"]["architecture"]))
    return own


def written(path: Path) -> dict[str, dict[str, str]]:
    """The settings each section of ``path`` writes, with the default each one notes."""
    found: dict[str, dict[str, str]] = {"": {}}
    section = ""
    commented = False
    for line in path.read_text().splitlines():
        if heading := SECTION.match(line) or COMMENTED_SECTION.match(line):
            section = heading.group("name")
            commented = line.startswith("#")
            found.setdefault(section, {})
            continue
        if commented or COMMENTED_SETTING.match(line):
            line = line.removeprefix("# ")
        if setting := SETTING.match(line):
            note = DEFAULT_NOTE.match(line)
            found[section][setting.group("key")] = note.group("default") if note else None
    return found


@pytest.mark.parametrize("path", shipped(), ids=lambda path: path.name)
def test_every_shipped_config_loads(path):
    assert load_experiment(path).dataset.name, path.name


@pytest.mark.parametrize("path", shipped(), ids=lambda path: path.name)
def test_every_shipped_config_mentions_every_option(path):
    """A config that stops naming an option stops documenting it.

    These files are how someone finds out what can be configured, so an option missing from one
    of them is an option nobody knows about.
    """
    data = tomllib.loads(path.read_text())
    present = written(path)

    for section in ["", *sections()]:
        missing = settable(section, data) - set(present.get(section, {}))
        where = f"[{section}]" if section else "the top level"
        assert not missing, f"{path.name}: {where} does not mention {sorted(missing)}"


@pytest.mark.parametrize("path", shipped(), ids=lambda path: path.name)
def test_every_shipped_config_notes_the_real_default(path):
    """The values are free to change; the default noted beside each one has to stay true.

    Without this the comments drift the first time somebody tries a different number, which is
    what these files are for.
    """
    data = tomllib.loads(path.read_text())
    present = written(path)

    for section, settings in present.items():
        model = sections()[section] if section else ExperimentConfig
        for key, noted in settings.items():
            qualified = f"{section}.{key}" if section else key
            if qualified in NO_DEFAULT:
                continue
            field = model.model_fields.get(key)
            if field is not None:
                default = field.default
            elif section == "encoder":
                default = encoder_defaults(data["encoder"]["name"])[key]
            elif section == "model":
                default = architecture_defaults(data["model"]["architecture"])[key]
            else:  # pragma: no cover - a closed section cannot hold an unknown key
                raise AssertionError(f"{path.name}: {qualified} is not an option")
            assert noted is not None, f"{path.name}: {qualified} does not say its default"
            assert tomllib.loads(f"x = {noted}")["x"] == default, (
                f"{path.name}: {qualified} says its default is {noted}, but it is {default!r}"
            )
