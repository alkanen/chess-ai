/**
 * Which of a model's thoughts the overlay shows, and the shapes it draws them in.
 *
 * The overlay shows what the model chose its move from: while the move is held back by the
 * move delay, what it is considering; once it is played, what it considered before playing it.
 * Both are the same thoughts, so the overlay does not change when the move lands. Once the
 * other side has moved they are about a position two moves ago, and nothing is shown until the
 * model has thought again.
 */

import type { Color, GameState, Thoughts, WinDrawLoss } from './api';

/** The thoughts the overlay shows, and whose they are. */
export interface ShownThoughts {
  thoughts: Thoughts;
  /** The side whose player thought them. */
  side: Color;
  /** The move it played from them, in UCI, or null while it is still considering. */
  played: string | null;
}

/** What the overlay shows for ``game``, or null when the model has nothing to show. */
export function shownThoughts(game: GameState): ShownThoughts | null {
  const { considering, moves, position } = game;
  // Only about the position the game stands in: anything else is about a position gone. Absent
  // altogether from a server older than this page, which is a frontend rebuilt under a server
  // still running.
  if (considering != null && considering.ply === moves.length) {
    return { thoughts: considering.thoughts, side: considering.side, played: null };
  }
  const last = moves.at(-1);
  if (last === undefined || last.thoughts === null) {
    return null;
  }
  // The side that made the last move is the one not on move now.
  const side: Color = position.turn === 'white' ? 'black' : 'white';
  return { thoughts: last.thoughts, side, played: last.uci };
}

/** Chances of each result, from White's side of the board. */
export interface WhiteOdds {
  white: number;
  draw: number;
  black: number;
}

/**
 * ``wdl`` from White's side. A model gives its chances from the side of the player moving, so
 * Black's win is White's loss.
 */
export function whiteOdds(wdl: WinDrawLoss, side: Color): WhiteOdds {
  return side === 'white'
    ? { white: wdl.win, draw: wdl.draw, black: wdl.loss }
    : { white: wdl.loss, draw: wdl.draw, black: wdl.win };
}

/** One candidate move as an arrow on the board. */
export interface Arrow {
  from: string;
  to: string;
  /** How likely the model thought the move, from 0 to 1, which is how heavily it is drawn. */
  weight: number;
  /** Whether it is the move the model played. */
  played: boolean;
}

/**
 * The candidates as arrows, least likely first, so that the likelier ones are drawn on top
 * where they cross.
 *
 * One arrow per pair of squares: the promotions of one pawn to one square differ only in the
 * piece, and the policy spreads its chances across them, so several are often among the
 * candidates. Drawn apart they would lie exactly on top of each other, the played one possibly
 * hidden under another. Together they are as likely as the pawn going there at all, and played
 * if any of them was; the list still tells them apart.
 */
export function candidateArrows(shown: ShownThoughts): Arrow[] {
  const arrows = new Map<string, Arrow>();
  for (const candidate of shown.thoughts.candidates) {
    const from = candidate.uci.slice(0, 2);
    const to = candidate.uci.slice(2, 4);
    const played = candidate.uci === shown.played;
    const before = arrows.get(from + to);
    arrows.set(from + to, {
      from,
      to,
      weight: (before?.weight ?? 0) + Math.max(0, candidate.probability),
      played: (before?.played ?? false) || played,
    });
  }
  return [...arrows.values()]
    .map((arrow) => ({ ...arrow, weight: Math.min(1, arrow.weight) }))
    .sort((one, other) => one.weight - other.weight);
}

/** A share as a whole percentage, for the list and the bar's labels. */
export function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}
