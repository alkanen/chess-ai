"""What the tournaments page shows: the tournaments there are, and how one of them stands."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from chess_ai.position_view import GameOverReason
from chess_ai.tournament import (
    Leaderboard,
    Tournament,
    TournamentDefinition,
    TournamentError,
    describe,
    leaderboard,
    open_tournament,
    schedule,
    tournament_names,
)


class TournamentSummary(BaseModel):
    """One tournament, as the list of them shows it."""

    name: str
    created: datetime
    """When it was made, or for one that cannot be read, when its directory last changed."""
    entrants: list[str]
    """Each entrant in a few words, in their places."""
    games_played: int
    games_total: int
    playing: bool
    """Whether a process is playing its games now, so that the page is worth refreshing."""
    error: str | None = None
    """Why it cannot be read, for one that cannot; then it has no entrants or games to show."""


class TournamentGameListing(BaseModel):
    """One finished game, as the tournament page lists it: enough to pick it out."""

    index: int
    """Which game of the tournament's PGN it is, counting from zero; how it is replayed."""
    round: int
    white: int
    """The entrant that had White, by its place."""
    black: int
    result: str
    termination: GameOverReason
    opening: str
    plies: int


class TournamentPage(BaseModel):
    """One tournament: what it is, how it stands, and its games."""

    name: str
    definition: TournamentDefinition
    entrants: list[str]
    """Each entrant in a few words, in their places."""
    leaderboard: Leaderboard
    games: list[TournamentGameListing]
    playing: bool


def list_tournaments(directory: Path) -> list[TournamentSummary]:
    """Every tournament in ``directory``, the newest first: one that cannot be read with why."""
    summaries = []
    for name in tournament_names(directory):
        try:
            tournament = open_tournament(directory, name)
            # Asked before the games are read, for the reason tournament_page gives.
            playing = tournament.playing()
            games = tournament.games()
        except TournamentError as e:
            summaries.append(_unreadable(directory / name, str(e)))
            continue
        definition = tournament.definition
        summaries.append(
            TournamentSummary(
                name=name,
                created=definition.created,
                entrants=[describe(entrant) for entrant in definition.entrants],
                games_played=len(games),
                games_total=len(schedule(definition)),
                playing=playing,
            )
        )
    summaries.sort(key=lambda summary: summary.created, reverse=True)
    return summaries


def tournament_page(tournament: Tournament) -> TournamentPage:
    """What the page of ``tournament`` shows."""
    definition = tournament.definition
    # Whether it is being played asked first: a tournament that finishes in between is then
    # shown with every game and as still playing, which the next look puts right, rather than
    # short of its last game and as finished, which nothing would.
    playing = tournament.playing()
    games = tournament.games()
    return TournamentPage(
        name=definition.name,
        definition=definition,
        entrants=[describe(entrant) for entrant in definition.entrants],
        leaderboard=leaderboard(tournament, games),
        games=[
            TournamentGameListing(
                index=index,
                round=game.round,
                white=game.white,
                black=game.black,
                result=game.result,
                termination=game.termination,
                opening=game.opening,
                plies=game.plies,
            )
            for index, game in enumerate(games)
        ],
        playing=playing,
    )


def _unreadable(path: Path, error: str) -> TournamentSummary:
    try:
        changed = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        changed = datetime.fromtimestamp(0, UTC)
    return TournamentSummary(
        name=path.name,
        created=changed,
        entrants=[],
        games_played=0,
        games_total=0,
        playing=False,
        error=error,
    )
