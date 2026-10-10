import { useEffect, useState } from 'react';
import {
  type Access,
  type PlayerInfo,
  type PositionSnapshot,
  rematch,
  type Replacement,
} from './api';
import { Board } from './board/Board';
import type { Orientation } from './board/geometry';
import { GameControls } from './GameControls';
import { GameLinks } from './GameLinks';
import { GameStatus } from './GameStatus';
import { MoveList, type MoveMark } from './MoveList';
import { forgetGame, goToNewGame, rememberGame } from './myGames';
import { PausedPanel } from './PausedPanel';
import { RequestPanel } from './RequestPanel';
import { candidateArrows, shownThoughts, whiteOdds } from './thoughts';
import { EvalBar, modelPlays, ThoughtsPanel, useShowThoughts } from './ThoughtsPanel';
import { useGameChannel } from './useGameChannel';
import './App.css';

/** Whether the side to move is the one this link plays, so its moves can be made here. */
function yourTurn(access: Access | null, position: PositionSnapshot): boolean {
  return position.game_over === null && access === position.turn;
}

/**
 * The side of the board a viewer belongs on: the one their link plays. Watching leaves them
 * behind White as usual.
 */
function playersSide(access: Access | null): Orientation {
  return access === 'black' ? 'black' : 'white';
}

/**
 * Which side is at the bottom of the board, and the flip that turns it round.
 *
 * The board faces the side you are given to play, and stays wherever you last put it.
 */
function useOrientation(access: Access | null): [Orientation, () => void] {
  const facing = playersSide(access);
  const [orientation, setOrientation] = useState<Orientation>(facing);
  const [shown, setShown] = useState<Orientation>(facing);
  if (shown !== facing) {
    setShown(facing);
    setOrientation(facing);
  }
  const flip = () => setOrientation((side) => (side === 'white' ? 'black' : 'white'));
  return [orientation, flip];
}

/**
 * How a checkpoint was asked to play, for a side a checkpoint is playing.
 *
 * The name says which run and which step, and this says the rest: two sides of a game can be
 * the same checkpoint at two ratings, and nothing else on the page would tell them apart.
 */
function describeModel(player: PlayerInfo): string | null {
  const model = player.model;
  if (model === null) {
    return null;
  }
  const rating = model.rating !== null ? `plays like ${model.rating}` : 'no rating';
  const choosing =
    model.strategy === 'sample' ? `samples at ${model.temperature}` : 'plays its best move';
  return `${rating}, ${choosing}`;
}

/**
 * How long Stockfish thinks, for a side Stockfish is playing, and whether it plays at the
 * strength asked for. Its levels only cover a range, and a game against "800" that is really
 * played against its floor has to say so, or the viewer learns the wrong thing from it.
 */
function describeStockfish(player: PlayerInfo): string | null {
  const stockfish = player.stockfish;
  if (stockfish === null) {
    return null;
  }
  const pace = `${stockfish.move_time} s a move`;
  if (stockfish.elo === stockfish.requested_elo) {
    return pace;
  }
  const limit =
    stockfish.requested_elo < stockfish.elo
      ? `plays no weaker than ${stockfish.min_elo}`
      : `plays no stronger than ${stockfish.max_elo}`;
  return `${pace}; ${stockfish.requested_elo} was asked for, and Stockfish ${limit}`;
}

/** One side of the game: who is playing it, and how they were asked to. */
export function Player({ player }: { player: PlayerInfo }) {
  const note = describeModel(player) ?? describeStockfish(player);
  return (
    <dd>
      {player.name}
      {note !== null && <span className="model-note">{note}</span>}
    </dd>
  );
}

/** Where each player that took over did, for the move list to mark. */
function replacementMarks(replacements: Replacement[]): MoveMark[] {
  return replacements.map((replaced) => ({
    ply: replaced.ply,
    text: `${replaced.old.name} replaced by ${replaced.new.name}`,
  }));
}

interface GameViewProps {
  /** The link the game is reached through, which says what this viewer may do in it. */
  link: string;
}

/** One game, as a link reaches it: the board it is played on, and what is asked of it. */
export function GameView({ link }: GameViewProps) {
  const {
    view,
    connected,
    missing,
    error,
    movePending,
    submitMove,
    resign,
    abort,
    takeBack,
    answer,
  } = useGameChannel(link);
  const access = view?.access ?? null;
  const [orientation, flip] = useOrientation(access);
  const [showThoughts, setShowThoughts] = useShowThoughts();

  // Kept in this browser's list of games once the server has said what the link is, so that
  // the game can be found again from the Game tab; and dropped once it is gone.
  useEffect(() => {
    if (access !== null) {
      rememberGame({ link, access });
    }
  }, [link, access]);
  useEffect(() => {
    if (missing !== null) {
      forgetGame(link);
    }
  }, [link, missing]);

  if (missing !== null) {
    return (
      <div className="no-such-game">
        <p role="alert">{missing}</p>
        <p>
          <a href="#">Start a new game</a>
        </p>
      </div>
    );
  }
  if (view === null) {
    return <p>Connecting to the server…</p>;
  }
  const { game } = view;
  const aborted = view.position.game_over?.reason === 'abort';
  // Only where a model plays, which is the only player that has thoughts to show.
  const thinking = modelPlays(game);
  const overlay = thinking && showThoughts;
  const shown = overlay ? shownThoughts(game) : null;
  const board = (
    <div className="board-frame">
      <Board
        snapshot={view.position}
        orientation={orientation}
        interactive={connected && !movePending && yourTurn(access, view.position)}
        onMove={submitMove}
        arrows={shown === null ? [] : candidateArrows(shown)}
      />
    </div>
  );
  return (
    <>
      <GameStatus position={view.position} inGame />
      <div className="game-view">
        {overlay ? (
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
          {!connected && view.position.game_over === null && (
            <p role="alert">Lost the connection to the server. Reconnecting…</p>
          )}
          {connected && error !== null && <p role="alert">{error}</p>}
          {aborted && <p className="note">The game has been deleted, and nothing of it kept.</p>}
          {access === 'watch' && <p className="note">You are watching this game.</p>}
          <dl className="players">
            <dt>White</dt>
            <Player player={game.white} />
            <dt>Black</dt>
            <Player player={game.black} />
          </dl>
          {game.request !== null && view.position.game_over === null && (
            <RequestPanel
              request={game.request}
              access={view.access}
              disabled={!connected}
              onAnswer={answer}
            />
          )}
          {game.paused !== null && view.position.game_over === null && (
            <PausedPanel
              // A pause after another player took over is a new one, with a form of its own.
              key={game.replacements.length}
              link={link}
              paused={game.paused}
              player={game[game.paused.side]}
              access={view.access}
              disabled={!connected}
            />
          )}
          <GameControls
            orientation={orientation}
            onFlip={flip}
            link={link}
            game={game}
            access={view.access}
            disabled={!connected}
            onResign={resign}
            onAbort={abort}
            onTakeBack={takeBack}
            onPlayAgain={async () => goToNewGame(await rematch(link), view.access)}
          />
          {thinking && (
            <ThoughtsPanel
              game={game}
              shown={shown}
              show={showThoughts}
              onShow={setShowThoughts}
            />
          )}
          <MoveList
            moves={game.moves}
            startFen={game.start_fen}
            marks={replacementMarks(game.replacements)}
          />
          {!aborted && <GameLinks link={link} access={view.access} watch={view.watch} />}
        </aside>
      </div>
    </>
  );
}
