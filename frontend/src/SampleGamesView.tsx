import { memo, useId, useMemo } from 'react';
import {
  fetchSampleGames,
  type LiveGame,
  type EvaluationEntry,
  type SampleGamesResult,
} from './api';
import {
  CheckpointScrubber,
  resultKey,
  suiteSteps,
  useCheckpointResult,
  useChosenCheckpoint,
} from './CheckpointResults';
import { LiveGameBoard } from './LiveGameBoard';
import { formatCount } from './runFormat';
import { describeEnding, describeOpponent, SAMPLE_SUITE, sampleGameHash } from './sampleGames';
import './SampleGamesView.css';

/** What the live game is, among the games its checkpoint plays. */
function liveCaption(live: LiveGame): string {
  const against = live.opponent === 'self' ? 'against itself' : 'against Stockfish';
  const game = `game ${live.index + 1} of ${live.games}`;
  return `Step ${formatCount(live.step)} is playing ${game}, ${against}`;
}

/** How the checkpoint was asked to play, and against what. */
function describeSettings(result: SampleGamesResult): string {
  const parts = [
    result.model.rating === null ? 'no rating' : `both sides rated ${result.model.rating}`,
    `openings from ${result.openings}`,
  ];
  const stockfish = result.stockfish;
  if (stockfish !== null) {
    parts.push(`Stockfish at Elo ${stockfish.elo}, ${stockfish.move_time} s a move`);
  }
  return parts.join('; ');
}

interface SampleGameListProps {
  run: string;
  result: SampleGamesResult;
}

/** A checkpoint's games, each linked to the replay viewer. */
function SampleGameList({ run, result }: SampleGameListProps) {
  const step = result.model.checkpoint;
  return (
    <>
      <p className="sample-summary">Played with {describeSettings(result)}.</p>
      <table className="sample-game-list">
        <thead>
          <tr>
            <th scope="col">Game</th>
            <th scope="col">Opponent</th>
            <th scope="col">Opening</th>
            <th scope="col">Result</th>
            <th scope="col">Moves</th>
          </tr>
        </thead>
        <tbody>
          {result.games.map((game) => (
            <tr key={game.index}>
              <td>
                <a href={sampleGameHash({ run, step, index: game.index })}>
                  Game {game.index + 1}
                </a>
              </td>
              <td>{describeOpponent(game)}</td>
              <td>{game.opening}</td>
              <td>{describeEnding(game)}</td>
              <td className="number">{Math.ceil(game.plies / 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

interface SampleGamesViewProps {
  run: string;
  /** Which suites have a result about which checkpoints, as the run channel has it. */
  evaluations: EvaluationEntry[];
  /** The game being played with one of the run's checkpoints, or null for none: shown here
   * when it is a sample game. */
  liveGame: LiveGame | null;
  /** Whether the run's config has its checkpoints play sample games. */
  playing: boolean;
}

/**
 * The games the run's checkpoints play: the one being played now, live, and those a checkpoint
 * chosen with a scrubber has played, to replay. It shows the newest checkpoint's games until
 * another is chosen, and follows the newest as more are played.
 *
 * Kept apart from the rest of the run page, which changes every second, and only drawn again
 * when the games change.
 */
export const SampleGamesView = memo(function SampleGamesView({
  run,
  evaluations,
  liveGame: anyLiveGame,
  playing,
}: SampleGamesViewProps) {
  const liveGame = anyLiveGame?.suite === SAMPLE_SUITE ? anyLiveGame : null;
  const id = useId();
  const steps = useMemo(() => suiteSteps(evaluations, SAMPLE_SUITE), [evaluations]);
  const chosen = useChosenCheckpoint(steps);
  const { entry } = chosen;
  const { shown, failed } = useCheckpointResult(run, entry, fetchSampleGames);

  if (steps.length === 0 && liveGame === null && !playing) {
    return null;
  }
  const result = shown?.result;
  const current = entry !== undefined && shown?.key === resultKey(entry);
  const failure = entry !== undefined && failed?.key === resultKey(entry) ? failed : null;
  return (
    <section className="sample-games" aria-labelledby={`${id}-heading`}>
      <h3 id={`${id}-heading`}>Sample games</h3>
      {liveGame !== null && <LiveGameBoard live={liveGame} caption={liveCaption(liveGame)} />}
      {steps.length === 0 ? (
        <p className="note">
          No checkpoint has finished its sample games yet. <code>chess-ai evaluator</code> plays
          them as each checkpoint is saved.
        </p>
      ) : (
        <>
          {liveGame !== null && <h4>Games played</h4>}
          <CheckpointScrubber steps={steps} chosen={chosen} />
          {failure !== null && (
            <p role="alert">
              Cannot load step {formatCount(entry?.step)} ({failure.message}); trying again.
            </p>
          )}
          {!current && result === undefined && failure === null && (
            <p className="note">Loading…</p>
          )}
          {result !== undefined && (
            <div
              className={current ? 'sample-result' : 'sample-result loading'}
              aria-busy={!current}
            >
              {!current && (
                <p className="note">
                  Showing step {formatCount(result.model.checkpoint)} while step{' '}
                  {formatCount(entry?.step)} loads.
                </p>
              )}
              <SampleGameList run={run} result={result} />
            </div>
          )}
        </>
      )}
    </section>
  );
});
