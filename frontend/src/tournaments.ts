/**
 * What the tournament pages make of a tournament: where it and its games are, and how its
 * numbers are written.
 *
 * Games are numbered from 1 in an address, as they are on the page, and from 0 everywhere else.
 */

import type { CrosstableCell, TournamentDefinition } from './api';

/** One game of a tournament, by its place among the tournament's games. */
export interface TournamentGameRef {
  name: string;
  index: number;
}

const TOURNAMENT_PAGE = /^#tournaments\/([^/]+)$/;
const TOURNAMENT_GAME = /^#replay\/tournament\/([^/]+)\/(\d+)$/;

/** The address of a tournament's page. */
export function tournamentHash(name: string): string {
  return `#tournaments/${encodeURIComponent(name)}`;
}

/** The tournament whose page an address is, or null for any other address. */
export function tournamentInHash(hash: string): string | null {
  const found = TOURNAMENT_PAGE.exec(hash);
  return found === null ? null : decodeURIComponent(found[1]);
}

/** The address of one tournament game in the replay viewer. */
export function tournamentGameHash({ name, index }: TournamentGameRef): string {
  return `#replay/tournament/${encodeURIComponent(name)}/${index + 1}`;
}

/** The tournament game an address opens in the replay viewer, or null for any other address. */
export function tournamentGameInHash(hash: string): TournamentGameRef | null {
  const found = TOURNAMENT_GAME.exec(hash);
  if (found === null || Number(found[2]) < 1) {
    return null;
  }
  return { name: decodeURIComponent(found[1]), index: Number(found[2]) - 1 };
}

/**
 * Points as a score is written, a half as ½: "2½", so that "1/2" can only be one point of two;
 * mirrors _points in chess_ai.cli.
 */
export function describePoints(points: number): string {
  const halves = Math.round(points * 2);
  const whole = Math.floor(halves / 2);
  if (halves % 2 === 0) {
    return String(whole);
  }
  return whole === 0 ? '½' : `${whole}½`;
}

/** What one entrant scored against another, such as "1½/2", or "–" against itself. */
export function describeCell(cell: CrosstableCell | null): string {
  return cell === null ? '–' : `${describePoints(cell.points)}/${cell.games}`;
}

/**
 * A rating as the leaderboard shows it: signed when the ratings are around an average of 0,
 * since only the gaps mean anything then, and plain when an anchor puts them on a scale.
 */
export function describeRating(rating: number, definition: TournamentDefinition): string {
  const rounded = Math.round(rating);
  if (definition.anchor !== null) {
    return String(rounded);
  }
  // A typographic minus, which lines up with the plus in a column of numbers.
  return rounded > 0 ? `+${rounded}` : rounded < 0 ? `−${-rounded}` : '0';
}
