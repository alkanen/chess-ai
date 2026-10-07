"""A dataset's manifest, written out for a person to read.

What a model learns is what its dataset contains, so the question this answers is "what is in
here?" — how many games and positions, how strong the players were, how the games ended, how
fast they were played, and what was thrown away on the way in. The distributions are drawn as
bars because the shape is the point: a rating histogram with two humps is a different training
set from one with a single peak, and no column of numbers says so at a glance.
"""

from typing import Final

from chess_ai.dataset.manifest import RATING_BUCKET, SPLITS, Filters, Manifest

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
    if any(counts.targets is not None for counts in manifest.splits.values()):
        lines.append(f"  trained on {_counts(manifest, 'trained_on')} positions")
    filters = describe_filters(manifest.filters)
    lines.append(f"  filters    {filters[0]}")
    lines += [f"             {line}" for line in filters[1:]]
    if manifest.reached_max_games:
        lines.append(f"  stopped    at the maximum of {manifest.filters.max_games:,} games")
    lines += ["", "sources"]
    for source in manifest.sources:
        lines.append(
            f"  {source.path}  {format_bytes(source.bytes)}, "
            f"{source.games_read:,} games read, {source.games_kept:,} kept"
        )
        if source.error is not None:
            # A source the build could not read whole leaves the dataset short of its games,
            # which is the first thing to know about a dataset that looks smaller than it should.
            # Three things worth telling apart: a file that went before it said anything, a
            # file that went away with an unknown number of its games still in it, and a file
            # that was there throughout and whose chess stopped making sense, which is all it
            # had to give.
            if source.left_nothing:
                what = "NOTHING READ FROM IT"
            elif source.went_away:
                what = "WENT AWAY PART-WAY THROUGH"
            else:
                what = "NOT READ WHOLE"
            lines.append(f"    {what}: {source.error}")
    lines += ["", f"skipped games  {manifest.games_skipped:,}"]
    if manifest.skipped:
        lines += _bars(
            {reason.replace("_", " "): count for reason, count in manifest.skipped.items()}
        )
    if manifest.filtered:
        lines += ["", f"games the filters left out  {manifest.games_filtered:,}"]
        lines += _bars(
            {reason.replace("_", " "): count for reason, count in manifest.filtered.items()}
        )
    if manifest.not_targets:
        lines += [
            "",
            f"positions stored but not trained on  {sum(manifest.not_targets.values()):,}",
        ]
        lines += _bars(
            {f"mover's {reason}": count for reason, count in manifest.not_targets.items()}
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


def describe_filters(filters: Filters) -> list[str]:
    """What ``filters`` let through, a line per filter, or a line saying there were none."""
    lines = []
    if filters.min_rating is not None or filters.max_rating is not None:
        if filters.max_rating is None:
            band = f"at least {filters.min_rating}"
        elif filters.min_rating is None:
            band = f"at most {filters.max_rating}"
        else:
            band = f"{filters.min_rating} to {filters.max_rating}"
        unknown = "passes" if filters.unknown_rating_passes else "does not"
        lines.append(f"player to move rated {band}; an unknown rating {unknown}")
    if filters.min_clock is not None:
        lines.append(f"player to move with at least {filters.min_clock:g}s on the clock")
    if filters.time_controls is not None:
        lines.append(f"time controls {', '.join(filters.time_controls)}")
    if filters.exclude_terminations:
        names = ", ".join(name.replace("_", " ") for name in filters.exclude_terminations)
        lines.append(f"not ended by {names}")
    if filters.from_date is not None or filters.until is not None:
        if filters.until is None:
            lines.append(f"played from {filters.from_date}")
        elif filters.from_date is None:
            lines.append(f"played until {filters.until}")
        else:
            lines.append(f"played from {filters.from_date} until {filters.until}")
    if filters.sample is not None:
        lines.append(f"a {filters.sample:.4g} sample, by a hash of each game")
    if filters.max_games is not None:
        lines.append(f"at most {filters.max_games:,} games")
    return lines or ["none: every game in every source"]


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
