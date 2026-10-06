import { useEffect, useState } from 'react';
import { type Access, fetchGame, type SeatView } from './api';
import { describeGameOver } from './GameStatus';
import { forgetGame, gameHash, goToNewGame, rememberedGames } from './myGames';
import { NewGameForm } from './NewGameForm';
import './GameLobby.css';

const ROLES: Record<Access, string> = {
  white: 'you play White',
  black: 'you play Black',
  control: 'you started it',
  watch: 'watching',
};

/** How long ago `iso` was, the way people say it. */
export function ago(iso: string, now = Date.now()): string {
  const minutes = Math.round((now - new Date(iso).getTime()) / 60_000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? 'a day ago' : `${days} days ago`;
}

/** Where a game stands, in a few words. */
function standing(seen: SeatView): string {
  const over = seen.game.position.game_over;
  if (over !== null) return describeGameOver(over);
  const moves = seen.game.moves.length;
  const { turn } = seen.game.position;
  const onMove =
    turn === seen.access ? 'your move' : `${turn === 'white' ? 'White' : 'Black'} to move`;
  return `${moves} ${moves === 1 ? 'move' : 'moves'}, ${onMove}`;
}

interface Listed {
  link: string;
  access: Access;
  seen: SeatView;
}

/**
 * The games this browser has started or opened that the server still has, asked about
 * afresh: the opponent may have moved since. A game the server no longer has is forgotten.
 */
function useMyGames(): Listed[] | null {
  const [games, setGames] = useState<Listed[] | null>(null);

  useEffect(() => {
    let current = true;
    const remembered = rememberedGames();
    void Promise.all(
      remembered.map(async ({ link, access }) => {
        try {
          const seen = await fetchGame(link);
          if (seen === null) {
            forgetGame(link);
            return null;
          }
          return { link, access, seen };
        } catch {
          // A server that could not answer now may answer later; the game is not forgotten.
          return null;
        }
      }),
    ).then((found) => {
      if (current) {
        setGames(found.filter((game): game is Listed => game !== null));
      }
    });
    return () => {
      current = false;
    };
  }, []);

  return games;
}

/** The Game tab with no game open: starting one, and going back to one. */
export function GameLobby() {
  const games = useMyGames();

  return (
    <div className="game-lobby">
      <NewGameForm onStarted={(started) => goToNewGame(started)} />
      <section className="my-games" aria-labelledby="my-games-heading">
        <h2 id="my-games-heading">Your games</h2>
        <p className="note">
          The games started or opened in this browser. Keep the link to a game: it is the only way
          back to it from anywhere else.
        </p>
        {games === null && <p className="note">Looking…</p>}
        {games !== null && games.length === 0 && <p className="note">No games yet.</p>}
        {games !== null && games.length > 0 && (
          <ul>
            {games.map(({ link, access, seen }) => (
              <li key={link}>
                <a href={gameHash(link, access)}>
                  <span className="names">
                    {seen.game.white.name} – {seen.game.black.name}
                  </span>
                  <span className="role">{ROLES[access]}</span>
                  <span className="standing">{standing(seen)}</span>
                  <span className="updated">{ago(seen.updated)}</span>
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
