"""``say`` callbacks for tests: what library code says, collected or dropped."""

from chess_ai.output import MILESTONE, Kind, Say


def heard(lines: list[str]) -> Say:
    """A ``say`` that adds each line to ``lines``, whatever kind it is."""

    def say(text: str, *, kind: Kind = MILESTONE) -> None:
        lines.append(text)

    return say


def unheard(text: str, *, kind: Kind = MILESTONE) -> None:
    """A ``say`` that drops every line."""
