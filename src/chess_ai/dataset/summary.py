"""A dataset's manifest, written out for a person to read.

What a model learns is what its dataset contains, so the question this answers is "what is in
here?" — how many games and positions, how strong the players were, how the games ended, how
fast they were played, and what was thrown away on the way in. The distributions are drawn as
bars because the shape is the point: a rating histogram with two humps is a different training
set from one with a single peak, and no column of numbers says so at a glance.
"""

from typing import Final

from chess_ai.dataset.manifest import RATING_BUCKET, SPLITS, Manifest

BAR_WIDTH: Final = 28
"""How wide the longest bar of a distribution is drawn."""

BAR: Final = "█"

_UNITS: Final = ("B", "KB", "MB", "GB", "TB")


def summarize(manifest: Manifest) -> str:
    """``manifest`` as a page of text: what the dataset is, and what is in it."""
    statistics = manifest.statistics
    lines = [
        f"dataset {manifest.name}",
        f"  built      {manifest.created:%Y-%m-%d %H:%M:%S %Z}",
        f"  format     version {manifest.format_version}, "
        f"move vocabulary {manifest.move_vocabulary_size}",
        f"  ratings    {manifest.rating_source}",
        f"  games      {_counts(manifest, 'games')}",
        f"  positions  {_counts(manifest, 'positions')}",
        f"  held back  {manifest.validation_fraction:.1%} of games, by a hash of each game",
    ]
    if not manifest.filters.model_dump():
        lines.append("  filters    none: every game in every source")
    lines += ["", "sources"]
    for source in manifest.sources:
        lines.append(
            f"  {source.path}  {format_bytes(source.bytes)}, "
            f"{source.games_read:,} games read, {source.games_kept:,} kept"
        )
        if source.error is not None:
            # A source the build could not read whole leaves the dataset short of its games,
            # which is the first thing to know about a dataset that looks smaller than it should.
            # Not opening at all is the worse of the two: everything in that file is missing.
            what = "NOT READ AT ALL" if not source.opened else "NOT READ WHOLE"
            lines.append(f"    {what}: {source.error}")
    lines += ["", f"skipped games  {manifest.games_skipped:,}"]
    if manifest.skipped:
        lines += _bars(
            {reason.replace("_", " "): count for reason, count in manifest.skipped.items()}
        )
    lines += ["", "results"] + _bars(statistics.results)
    lines += ["", "time controls"] + _bars(statistics.time_controls)
    lines += ["", "rating sources"] + _bars(statistics.rating_sources)
    lines += ["", f"ratings  (per player, {manifest.games * 2:,} in all)"]
    ratings = {
        f"{int(bucket)}-{int(bucket) + RATING_BUCKET - 1}": count
        for bucket, count in statistics.ratings.items()
    }
    if statistics.ratings_unknown:
        ratings["unknown"] = statistics.ratings_unknown
    lines += _bars(ratings) if ratings else ["  no ratings"]
    return "\n".join(lines) + "\n"


def _counts(manifest: Manifest, what: str) -> str:
    """A total and its two splits, as "8,394 (train 8,240, validation 154)"."""
    total = getattr(manifest, what)
    splits = ", ".join(
        f"{split} {getattr(manifest.splits[split], what):,}"
        for split in SPLITS
        if split in manifest.splits
    )
    return f"{total:,} ({splits})" if splits else f"{total:,}"


def _bars(counts: dict[str, int]) -> list[str]:
    """One line per entry: its name, its count, its share, and a bar to compare them by."""
    total = sum(counts.values())
    if not total:
        return ["  none"]
    label_width = max(len(label) for label in counts)
    count_width = len(f"{max(counts.values()):,}")
    longest = max(counts.values())
    lines = []
    for label, count in counts.items():
        bar = BAR * max(1, round(BAR_WIDTH * count / longest)) if count else ""
        lines.append(
            f"  {label:<{label_width}}  {count:>{count_width},}  {count / total:>5.1%}  {bar}"
        )
    return lines


def format_bytes(count: int) -> str:
    """A size in the unit that makes it readable: "512 B", "1.2 MB", "27.4 GB"."""
    size = float(count)
    for unit in _UNITS:
        if size < 1024 or unit == _UNITS[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")
