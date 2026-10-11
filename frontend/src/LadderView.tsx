import { memo, useEffect, useId, useMemo, useState } from 'react';
import {
  fetchLadder,
  fetchLadderPoints,
  type EvaluationEntry,
  type LadderPoint,
  type LadderResult,
  type LiveGame,
} from './api';
import {
  CheckpointScrubber,
  resultKey,
  suiteSteps,
  useCheckpointResult,
  useChosenCheckpoint,
} from './CheckpointResults';
import { EloChart } from './EloChart';
import {
  describeEstimate,
  describeLadderOpponent,
  describeScore,
  LADDER_SUITE,
  ladderGameHash,
} from './ladder';
import { LiveGameBoard } from './LiveGameBoard';
import { formatCount } from './runFormat';
import { describeEnding } from './sampleGames';
import './SampleGamesView.css';
import './LadderView.css';

/** How long to wait before asking again for the chart's points, which could not be fetched. */
function retryDelayMs(failures: number): number {
  return Math.min(1000 * 2 ** (failures - 1), 10_000);
}

/**
 * Every ladder of `run`, to chart: asked for again whenever one is added or replaced, which
 * `entries` say, and again until it comes when it cannot be fetched. The last ones there were
 * stay on show while newer ones are on their way.
 */
function useLadderPoints(run: string, entries: EvaluationEntry[]): LadderPoint[] {
  const [points, setPoints] = useState<LadderPoint[]>([]);
  // As one string, so that the same results in a new array do not ask again.
  const key = entries.map(resultKey).join(',');
  useEffect(() => {
    if (key === '') {
      return;
    }
    let wanted = true;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ask = () => {
      fetchLadderPoints(run).then(
        (fetched) => {
          if (wanted) {
            setPoints(fetched);
          }
        },
        () => {
          if (wanted) {
            failures += 1;
            timer = setTimeout(ask, retryDelayMs(failures));
          }
        },
      );
    };
    ask();
    return () => {
      wanted = false;
      clearTimeout(timer);
    };
  }, [run, key]);
  return key === '' ? [] : points;
}

/** What the live game is, among the games of its checkpoint's ladder. */
function liveCaption(live: LiveGame): string {
  const against = live.level === null ? 'Stockfish' : `Stockfish ${live.level}`;
  return (
    `Step ${formatCount(live.step)} is playing ladder game ${live.index + 1} of ${live.games}, ` +
    `against ${against}`
  );
}

/** How the checkpoint was asked to play, and against what. */
function describeSettings(result: LadderResult): string {
  const moveTimes = [...new Set(result.levels.map((level) => level.move_time))];
  return [
    result.model.rating === null ? 'no rating' : `rated ${result.model.rating}`,
    `${result.games_per_level} games a level`,
    `Stockfish at ${moveTimes.map((time) => `${time} s`).join(' and ')} a move`,
    `openings from ${result.openings}`,
  ].join('; ');
}

interface LadderResultViewProps {
  run: string;
  result: LadderResult;
}

/** A checkpoint's ladder: its Elo, its score at each level, and its games. */
function LadderResultView({ run, result }: LadderResultViewProps) {
  const step = result.model.checkpoint;
  const { estimate } = result;
  return (
    <>
      <p className="ladder-estimate">
        Elo <strong>{describeEstimate(estimate)}</strong>, a{' '}
        {Math.round(estimate.confidence * 100)}% range, from {estimate.points}/{estimate.games}{' '}
        points.
      </p>
      <p className="sample-summary">Played {describeSettings(result)}.</p>
      <table className="sample-game-list ladder-levels">
        <thead>
          <tr>
            <th scope="col">Level</th>
            <th scope="col">Score</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {result.levels.map((level) => {
            const games = level.wins + level.draws + level.losses;
            const share = games === 0 ? 0 : (level.wins + level.draws / 2) / games;
            return (
              <tr key={level.elo}>
                <td>
                  Stockfish {level.elo}
                  {level.requested_elo !== level.elo && ` (asked for ${level.requested_elo})`}
                </td>
                <td className="number">{describeScore(level)}</td>
                <td className="number">{Math.round(share * 100)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {result.games.length === 0 ? (
        <p>
          <a href={ladderGameHash({ run, step, index: 0 })}>Replay the games</a>{' '}
          <span className="note">(this ladder was played before its games were listed)</span>
        </p>
      ) : (
        <details className="ladder-games">
          <summary>Games ({result.games.length})</summary>
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
                    <a href={ladderGameHash({ run, step, index: game.index })}>
                      Game {game.index + 1}
                    </a>
                  </td>
                  <td>{describeLadderOpponent(game)}</td>
                  <td>{game.opening}</td>
                  <td>{describeEnding(game)}</td>
                  <td className="number">{Math.ceil(game.plies / 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </>
  );
}

interface LadderViewProps {
  run: string;
  /** Which suites have a result about which checkpoints, as the run channel has it. */
  evaluations: EvaluationEntry[];
  /** The game being played with one of the run's checkpoints, or null for none: shown here
   * when it is a ladder game. */
  liveGame: LiveGame | null;
  /** Whether the run's config has its checkpoints laddered. */
  laddered: boolean;
}

/**
 * The run's Stockfish ladders: the Elo of its checkpoints over training, the ladder game being
 * played now, and how a checkpoint chosen with a scrubber did at each level, with its games to
 * replay. It shows the newest checkpoint's ladder until another is chosen.
 *
 * Kept apart from the rest of the run page, which changes every second, and only drawn again
 * when the ladders change.
 */
export const LadderView = memo(function LadderView({
  run,
  evaluations,
  liveGame: anyLiveGame,
  laddered,
}: LadderViewProps) {
  const id = useId();
  const liveGame = anyLiveGame?.suite === LADDER_SUITE ? anyLiveGame : null;
  const steps = useMemo(() => suiteSteps(evaluations, LADDER_SUITE), [evaluations]);
  const points = useLadderPoints(run, steps);
  const chosen = useChosenCheckpoint(steps);
  const { entry } = chosen;
  const { shown, failed } = useCheckpointResult(run, entry, fetchLadder);

  if (steps.length === 0 && liveGame === null && !laddered) {
    return null;
  }
  const result = shown?.result;
  const current = entry !== undefined && shown?.key === resultKey(entry);
  const failure = entry !== undefined && failed?.key === resultKey(entry) ? failed : null;
  return (
    <section className="sample-games ladder-view" aria-labelledby={`${id}-heading`}>
      <h3 id={`${id}-heading`}>Stockfish ladder</h3>
      {points.length > 0 && <EloChart points={points} />}
      {liveGame !== null && <LiveGameBoard live={liveGame} caption={liveCaption(liveGame)} />}
      {steps.length === 0 ? (
        <p className="note">
          No checkpoint has finished its ladder yet. <code>chess-ai evaluator</code> plays it
          every so many steps (<code>[ladder] every</code>), and <code>chess-ai ladder</code> on
          any checkpoint.
        </p>
      ) : (
        <>
          <h4>Ladder of a checkpoint</h4>
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
              <LadderResultView run={run} result={result} />
            </div>
          )}
        </>
      )}
    </section>
  );
});
