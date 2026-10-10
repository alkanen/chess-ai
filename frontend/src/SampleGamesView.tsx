import { memo, useId, useMemo, useState } from 'react';
import {
  fetchSampleGames,
  type Color,
  type LiveGame,
  type EvaluationEntry,
  type SampleGamesResult,
} from './api';
import { Board } from './board/Board';
import {
  CheckpointScrubber,
  resultKey,
  suiteSteps,
  useCheckpointResult,
  useChosenCheckpoint,
} from './CheckpointResults';
import { Player } from './GameView';
import { GameStatus } from './GameStatus';
import { MoveList } from './MoveList';
import { formatCount } from './runFormat';
import { describeEnding, describeOpponent, SAMPLE_SUITE, sampleGameHash } from './sampleGames';
import { candidateArrows, shownThoughts, whiteOdds } from './thoughts';
import { EvalBar, ThoughtsPanel, useShowThoughts } from './ThoughtsPanel';
// The board and the panel beside it are laid out as the game view's are.
import './App.css';
import './SampleGamesView.css';

function opposite(side: Color): Color {
  return side === 'white' ? 'black' : 'white';
}

/** What the live game is, among the games its checkpoint plays. */
function liveCaption(live: LiveGame): string {
  const against = live.opponent === 'self' ? 'against itself' : 'against Stockfish';
  const game = `game ${live.index + 1} of ${live.games}`;
  return `Step ${formatCount(live.step)} is playing ${game}, ${against}`;
}

/**
 * The sample game being played, move by move as the evaluator plays it, with what the model
 * thought of its last move when the viewer asks for that.
 */
function LiveSampleGame({ live }: { live: LiveGame }) {
  const id = useId();
  const { game } = live;
  // From the checkpoint's side, which against itself is both: White, as usual.
  const modelSide: Color = game.white.model !== null ? 'white' : 'black';
  // Kept as a flip rather than a side, so that it holds from one game to the next whichever
  // side the checkpoint plays in it.
  const [flipped, setFlipped] = useState(false);
  const orientation = flipped ? opposite(modelSide) : modelSide;
  const [showThoughts, setShowThoughts] = useShowThoughts();
  const shown = showThoughts ? shownThoughts(game) : null;
  const board = (
    <div className="board-frame">
      <Board
        snapshot={game.position}
        orientation={orientation}
        arrows={shown === null ? [] : candidateArrows(shown)}
      />
    </div>
  );
  return (
    <section className="live-sample-game" aria-labelledby={`${id}-heading`}>
      <h4 id={`${id}-heading`}>
        <span className="live-badge">Live</span> {liveCaption(live)}
      </h4>
      <GameStatus position={game.position} inGame />
      <div className="game-view">
        {showThoughts ? (
          <div className="board-row">
            <EvalBar
              odds={
                shown?.thoughts.wdl != null ? whiteOdds(shown.thoughts.wdl, shown.side) : null
              }
              orientation={orientation}
            />
            {board}
          </div>
        ) : (
          board
        )}
        <aside className="side-panel">
          <dl className="players">
            <dt>White</dt>
            <Player player={game.white} />
            <dt>Black</dt>
            <Player player={game.black} />
          </dl>
          <p className="flip">
            <button type="button" onClick={() => setFlipped((was) => !was)}>
              {orientation === 'white' ? 'Flip to Black' : 'Flip to White'}
            </button>
          </p>
          <ThoughtsPanel
            game={game}
            shown={shown}
            show={showThoughts}
            onShow={setShowThoughts}
          />
          <MoveList moves={game.moves} startFen={game.start_fen} />
        </aside>
      </div>
    </section>
  );
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
  /** The sample game being played with one of the run's checkpoints, or null for none. */
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
  liveGame,
  playing,
}: SampleGamesViewProps) {
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
      {liveGame !== null && <LiveSampleGame live={liveGame} />}
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
