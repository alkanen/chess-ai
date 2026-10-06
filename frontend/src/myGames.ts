import type { Access, GameLinks, NewGame } from './api';

/**
 * The games this browser has started or opened, so that a link nobody bookmarked is not a
 * game lost. Kept in the browser alone: the server keeps no list of games, and anyone who
 * can see one has the links in it.
 */
export interface RememberedGame {
  /** The link the game was opened through, which is what reaches it again. */
  link: string;
  /** What that link may do in the game. */
  access: Access;
  /**
   * Every link to the game, for a game started in this browser: the other player's link
   * is here to be sent to them.
   */
  links?: GameLinks;
  /** When this browser first saw the game, as an ISO timestamp. */
  added: string;
}

const KEY = 'chess-ai.games';

/** The games this browser remembers, the one it saw most recently first. */
export function rememberedGames(): RememberedGame[] {
  try {
    const kept: unknown = JSON.parse(window.localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(kept) ? (kept as RememberedGame[]).filter(isRemembered) : [];
  } catch {
    // Storage this page may not use, or something in it that is not ours: nothing to show.
    return [];
  }
}

/** The game `link` reaches, as this browser remembers it, if it does. */
export function rememberedGame(link: string): RememberedGame | undefined {
  return rememberedGames().find((game) => game.link === link);
}

/**
 * Remember a game, or what more has been learned about one: a game already here keeps the
 * links it was started with, and moves to the front.
 */
export function rememberGame(game: Omit<RememberedGame, 'added'>): void {
  const kept = rememberedGames();
  const before = kept.find((known) => known.link === game.link);
  const merged: RememberedGame = {
    ...before,
    ...game,
    links: game.links ?? before?.links,
    added: before?.added ?? new Date().toISOString(),
  };
  store([merged, ...kept.filter((known) => known.link !== game.link)]);
}

/** Forget a game the server no longer has. */
export function forgetGame(link: string): void {
  store(rememberedGames().filter((known) => known.link !== link));
}

function store(games: RememberedGame[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(games));
  } catch {
    // A browser that will not keep anything for this page plays the game all the same.
  }
}

function isRemembered(entry: unknown): entry is RememberedGame {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    typeof (entry as RememberedGame).link === 'string' &&
    typeof (entry as RememberedGame).access === 'string'
  );
}

/**
 * Go to a game that has just been started here, remembering every link to it: to `side` if
 * the game has a person playing it, else to whichever side a person plays, else to its control.
 */
export function goToNewGame({ links }: NewGame, side?: Access): void {
  const preferred: Access[] = side === 'white' || side === 'black' ? [side] : [];
  const access =
    [...preferred, 'white' as const, 'black' as const, 'control' as const].find(
      (candidate) => links[candidate] !== null,
    ) ?? 'watch';
  const link = links[access] ?? links.watch;
  rememberGame({ link, access, links });
  window.location.hash = gameHash(link, access);
}

/** The address of the page a link opens: a play link plays, and a watch link watches. */
export function gameHash(link: string, access: Access): string {
  return `#${access === 'watch' ? 'watch' : 'game'}/${encodeURIComponent(link)}`;
}

/** The whole address of a link, to send to somebody. */
export function gameAddress(link: string, access: Access): string {
  return new URL(gameHash(link, access), window.location.href).href;
}
