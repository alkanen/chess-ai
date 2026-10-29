import { useEffect, useState } from 'react';
import {
  fetchTournament,
  fetchTournaments,
  tournamentPgnUrl,
  type TournamentPage,
  type TournamentSummary,
} from './api';
import { formatBuilt } from './DatasetsView';
import { describeResult } from './GameStatus';
import {
  describeCell,
  describePoints,
  describeRating,
  tournamentGameHash,
  tournamentHash,
} from './tournaments';
import './TournamentsView.css';

/**
 * How often a tournament, or the list of them, is asked for again while one is being played:
 * a game takes from seconds to minutes, and the page is something left open to watch.
 */
export const TOURNAMENT_REFRESH_MS = 5_000;

/**
 * What `fetch` gives, asked for again every so often while `again` says it is worth it, and
 * after a failure until it comes. What came last stays on show meanwhile, with the failure.
 */
function useRefreshed<T>(
  fetch: () => Promise<T>,
  again: (found: T) => boolean,
  key: string,
): [T | null, string | null] {
  const [found, setFound] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let dropped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    function ask() {
      fetch().then(
        (fetched) => {
          if (dropped) {
            return;
          }
          setFound(fetched);
          setError(null);
          if (again(fetched)) {
            timer = setTimeout(ask, TOURNAMENT_REFRESH_MS);
          }
        },
        (e: unknown) => {
          if (dropped) {
            return;
          }
          setError(e instanceof Error ? e.message : String(e));
          timer = setTimeout(ask, TOURNAMENT_REFRESH_MS);
        },
      );
    }
    ask();
    return () => {
      dropped = true;
      clearTimeout(timer);
    };
    // `fetch` and `again` are new functions every render: the key says what is fetched.
  }, [key]);

  return [found, error];
}

/** Every tournament on the server, the newest first, kept up to date while any is played. */
export function TournamentsView() {
  const [tournaments, error] = useRefreshed(
    fetchTournaments,
    (found) => found.some((one) => one.playing),
    'tournaments',
  );

  return (
    <section className="tournaments-view" aria-labelledby="tournaments-heading">
      <h2 id="tournaments-heading">Tournaments</h2>
      {error !== null && <p role="alert">Could not list the tournaments: {error}</p>}
      {error === null && tournaments === null && <p className="note">Looking…</p>}
      {tournaments !== null && tournaments.length === 0 && (
        <p className="note">
          No tournaments have been played here yet. Start one with{' '}
          <code>chess-ai tournament start</code>.
        </p>
      )}
      {tournaments !== null && tournaments.length > 0 && (
        <div className="table-frame">
          <table>
            <thead>
              <tr>
                <th scope="col">Name</th>
                <th scope="col">Started</th>
                <th scope="col">Entrants</th>
                <th scope="col" className="number">
                  Games
                </th>
              </tr>
            </thead>
            <tbody>
              {tournaments.map((tournament) => (
                <TournamentRow key={tournament.name} tournament={tournament} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function TournamentRow({ tournament }: { tournament: TournamentSummary }) {
  const link = (
    <a href={tournamentHash(tournament.name)} className="name">
      {tournament.name}
    </a>
  );
  if (tournament.error !== null) {
    return (
      <tr>
        <th scope="row">{link}</th>
        <td colSpan={3} className="unreadable">
          Cannot be read: {tournament.error}
        </td>
      </tr>
    );
  }
  return (
    <tr>
      <th scope="row">{link}</th>
      <td>{formatBuilt(tournament.created)}</td>
      <td className="wraps">{tournament.entrants.join(', ')}</td>
      <td className="number">
        {tournament.games_played} of {tournament.games_total}
        {tournament.playing && ' (being played)'}
      </td>
    </tr>
  );
}

/** An entrant by its number, counting from 1, and in a few words: "2 tiny step 4 (rating 1800)". */
function entrantName(page: TournamentPage, entrant: number): string {
  return `${entrant + 1} ${page.entrants[entrant]}`;
}

/** What the ratings are measured from, in a sentence. */
function describeScale(page: TournamentPage): string {
  const { anchor } = page.definition;
  if (anchor === null) {
    return 'Ratings are around an average of 0: only the gaps between them mean anything.';
  }
  return (
    `Ratings are measured from ${entrantName(page, anchor.entrant)} at ` +
    `${Math.round(anchor.rating)}.`
  );
}

/** The entrant's run, for a checkpoint, as a link to its page; else the name alone. */
function EntrantName({ page, entrant }: { page: TournamentPage; entrant: number }) {
  const played = page.definition.entrants[entrant];
  const name = entrantName(page, entrant);
  if (played.kind !== 'model') {
    return <>{name}</>;
  }
  return <a href={`#runs/${encodeURIComponent(played.run)}`}>{name}</a>;
}

/**
 * One tournament: its leaderboard, crosstable and games, asked for again while it is being
 * played.
 */
export function TournamentView({ name }: { name: string }) {
  const [page, error] = useRefreshed(
    () => fetchTournament(name),
    (found) => found.playing,
    name,
  );

  return (
    <section className="tournaments-view" aria-labelledby="tournament-heading">
      <p className="note">
        <a href="#tournaments">All tournaments</a>
      </p>
      <h2 id="tournament-heading">{name}</h2>
      {error !== null && <p role="alert">Could not show the tournament: {error}</p>}
      {error === null && page === null && <p className="note">Looking…</p>}
      {page !== null && <Tournament page={page} />}
    </section>
  );
}

function Tournament({ page }: { page: TournamentPage }) {
  const { definition, leaderboard } = page;
  return (
    <>
      <p>
        {leaderboard.games_played} of {leaderboard.games_total} games
        {page.playing && ', being played'}. {definition.games_per_pairing} games a pairing from
        opening set {definition.opening_set ?? definition.openings}, the colours swapped every game.
      </p>
      <h3 id="leaderboard-heading">Leaderboard</h3>
      <p className="note">
        {describeScale(page)} Each range is as likely as 95% to hold the rating.
      </p>
      <div className="table-frame">
        <table aria-labelledby="leaderboard-heading">
          <thead>
            <tr>
              <th scope="col" className="number">
                #
              </th>
              <th scope="col">Entrant</th>
              <th scope="col" className="number">
                Rating
              </th>
              <th scope="col">Range</th>
              <th scope="col" className="number">
                Score
              </th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.standings.map((standing, place) => (
              <tr key={standing.entrant}>
                <td className="number">{place + 1}</td>
                <td>
                  <EntrantName page={page} entrant={standing.entrant} />
                </td>
                <td className="number">{describeRating(standing.rating, definition)}</td>
                <td>
                  {describeRating(standing.low, definition)} to{' '}
                  {describeRating(standing.high, definition)}
                </td>
                <td className="number">
                  {describePoints(standing.points)}/{standing.games}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Crosstable page={page} />
      <Games page={page} />
    </>
  );
}

function Crosstable({ page }: { page: TournamentPage }) {
  return (
    <>
      <h3 id="crosstable-heading">Crosstable</h3>
      <p className="note">What each row's entrant scored against each column's.</p>
      <div className="table-frame">
        <table aria-labelledby="crosstable-heading" className="crosstable">
          <thead>
            <tr>
              <td />
              {page.entrants.map((_, entrant) => (
                <th key={entrant} scope="col" className="number" title={page.entrants[entrant]}>
                  {entrant + 1}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.leaderboard.crosstable.map((row, entrant) => (
              <tr key={entrant}>
                <th scope="row">{entrantName(page, entrant)}</th>
                {row.map((cell, opponent) => (
                  <td key={opponent} className="number">
                    {describeCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Games({ page }: { page: TournamentPage }) {
  return (
    <>
      <h3 id="games-heading">Games</h3>
      <p className="note">
        In the order they were played. <a href={tournamentPgnUrl(page.name)}>Download them all</a>{' '}
        as PGN.
      </p>
      {page.games.length === 0 ? (
        <p className="note">None finished yet.</p>
      ) : (
        <div className="table-frame">
          <table aria-labelledby="games-heading">
            <thead>
              <tr>
                <th scope="col" className="number">
                  #
                </th>
                <th scope="col" className="number">
                  Round
                </th>
                <th scope="col">White</th>
                <th scope="col">Black</th>
                <th scope="col">Result</th>
                <th scope="col">Opening</th>
              </tr>
            </thead>
            <tbody>
              {page.games.map((game) => (
                <tr key={game.index}>
                  <td className="number">
                    <a href={tournamentGameHash({ name: page.name, index: game.index })}>
                      {game.index + 1}
                    </a>
                  </td>
                  <td className="number">{game.round}</td>
                  <td className="wraps">{entrantName(page, game.white)}</td>
                  <td className="wraps">{entrantName(page, game.black)}</td>
                  <td className="wraps">
                    {describeResult(game.result)} (
                    {game.termination.replaceAll('_', ' ')}, {Math.ceil(game.plies / 2)} moves)
                  </td>
                  <td className="wraps">{game.opening}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
