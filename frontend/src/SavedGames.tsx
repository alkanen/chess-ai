import { useEffect, useState } from 'react';
import { fetchSavedGames, type SavedGame } from './api';
import { describeResult } from './GameStatus';
import './SavedGames.css';

interface SavedGamesProps {
  /** The saved game on show, so that the list can mark it, or null for none of them. */
  current: string | null;
  onOpen: (name: string) => void;
}

/** The games this server has played and kept, most recent first, each one to open. */
export function SavedGames({ current, onOpen }: SavedGamesProps) {
  const [games, setGames] = useState<SavedGame[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // The list is asked for once, when the viewer comes to this page: a game finishing
    // meanwhile is one page change away, and nothing here is worth polling the server for.
    let dropped = false;
    fetchSavedGames().then(
      (found) => !dropped && setGames(found),
      (e: unknown) => !dropped && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      dropped = true;
    };
  }, []);

  return (
    <section className="saved-games" aria-labelledby="saved-games-heading">
      <h3 id="saved-games-heading">Saved games</h3>
      {error !== null && <p role="alert">Could not list the saved games: {error}</p>}
      {error === null && games === null && <p className="note">Looking…</p>}
      {games !== null && games.length === 0 && (
        <p className="note">No games have been saved here yet.</p>
      )}
      {games !== null && games.length > 0 && (
        <ul>
          {games.map((game) => (
            <li key={game.name}>
              <button
                type="button"
                aria-current={game.name === current ? 'true' : undefined}
                onClick={() => onOpen(game.name)}
              >
                <span className="date">{game.date}</span>
                <span className="names">
                  {game.white} – {game.black}
                </span>
                <span className="result">{describeResult(game.result)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
