/**
 * What the run page makes of a run's sample games: the games each checkpoint plays against
 * itself and perhaps Stockfish, and where in the replay viewer one of them is.
 *
 * Games are numbered from 1 in an address, as they are on the page, and from 0 everywhere else.
 */

import type { SampleGame } from './api';
import { describeGameOver, describeResult } from './GameStatus';

/** What the sample games are called among a run's evaluation suites. */
export const SAMPLE_SUITE = 'sample-games';

/** One sample game of one checkpoint of a run. */
export interface SampleGameRef {
  run: string;
  step: number;
  /** Counted from 0, as the server counts it. */
  index: number;
}

const SAMPLE_GAME = /^#replay\/run\/([^/]+)\/(\d+)\/(\d+)$/;

/** The address of one sample game in the replay viewer. */
export function sampleGameHash({ run, step, index }: SampleGameRef): string {
  return `#replay/run/${encodeURIComponent(run)}/${step}/${index + 1}`;
}

/** The sample game an address opens in the replay viewer, or null for any other address. */
export function sampleGameInHash(hash: string): SampleGameRef | null {
  const found = SAMPLE_GAME.exec(hash);
  if (found === null || Number(found[3]) < 1) {
    return null;
  }
  return {
    run: decodeURIComponent(found[1]),
    step: Number(found[2]),
    index: Number(found[3]) - 1,
  };
}

/** Whom the checkpoint played, and with which pieces. */
export function describeOpponent(game: Pick<SampleGame, 'opponent' | 'model_color'>): string {
  if (game.opponent === 'self') {
    return 'Itself';
  }
  return `Stockfish, checkpoint as ${game.model_color === 'black' ? 'Black' : 'White'}`;
}

/** How the game ended, in words, or that it did not. */
export function describeEnding(game: Pick<SampleGame, 'result' | 'termination'>): string {
  if (game.termination === null) {
    return describeResult(game.result);
  }
  return describeGameOver({ result: game.result, reason: game.termination });
}
