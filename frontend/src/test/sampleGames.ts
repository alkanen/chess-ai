import type { LiveGame, PlayerInfo, SampleGamesResult } from '../api';
import { foolsMateMoves, foolsMateStart } from './foolsMate';

/** A run's checkpoint from `step`, as a side of a sample game names it. */
export function checkpointPlayer(step: number): PlayerInfo {
  return {
    name: `tiny@${step}`,
    accepts_moves: false,
    model: { run: 'tiny', checkpoint: step, rating: 2000, strategy: 'argmax', temperature: null },
    stockfish: null,
  };
}

export const STOCKFISH_PLAYER: PlayerInfo = {
  name: 'Stockfish 1350',
  accepts_moves: false,
  model: null,
  stockfish: { elo: 1350, requested_elo: 1350, min_elo: 1320, max_elo: 3190, move_time: 0.1 },
};

/**
 * A sample game of fool's mate two moves in, the last of them with what the checkpoint
 * thought of it, as the run channel sends it.
 */
export function liveGame(changes: Partial<LiveGame> = {}): LiveGame {
  const [first, second] = foolsMateMoves;
  const step = changes.step ?? 500;
  return {
    run: 'tiny',
    step,
    index: 0,
    games: 2,
    opponent: 'self',
    updated: '2026-10-10T12:00:00Z',
    game: {
      ...foolsMateStart.game,
      white: checkpointPlayer(step),
      black: checkpointPlayer(step),
      moves: [
        first.move,
        {
          ...second.move,
          thoughts: {
            candidates: [
              { uci: 'e7e5', san: 'e5', probability: 0.6 },
              { uci: 'd7d5', san: 'd5', probability: 0.3 },
            ],
            wdl: { win: 0.5, draw: 0.3, loss: 0.2 },
          },
        },
      ],
      position: second.position,
    },
    ...changes,
  };
}

/** What a checkpoint's two games against itself and one against Stockfish came to. */
export function sampleGamesResult(
  step: number,
  changes: Partial<SampleGamesResult> = {},
): SampleGamesResult {
  return {
    model: { run: 'tiny', checkpoint: step, rating: 2000, strategy: 'argmax', temperature: null },
    started: '2026-10-10T12:00:00Z',
    finished: '2026-10-10T12:05:00Z',
    openings: 'curated v1',
    stockfish: STOCKFISH_PLAYER.stockfish,
    games: [
      {
        index: 0,
        opponent: 'self',
        model_color: 'both',
        opening: 'Italian Game',
        white: `tiny@${step}`,
        black: `tiny@${step}`,
        result: '1-0',
        termination: 'checkmate',
        plies: 41,
      },
      {
        index: 1,
        opponent: 'self',
        model_color: 'both',
        opening: 'Sicilian Defence',
        white: `tiny@${step}`,
        black: `tiny@${step}`,
        result: '1/2-1/2',
        termination: 'threefold_repetition',
        plies: 60,
      },
      {
        index: 2,
        opponent: 'stockfish',
        model_color: 'black',
        opening: 'Italian Game',
        white: 'Stockfish 1350',
        black: `tiny@${step}`,
        result: '*',
        termination: null,
        plies: 12,
      },
    ],
    ...changes,
  };
}
