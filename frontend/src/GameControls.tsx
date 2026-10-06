import { useState } from 'react';
import { type Access, fetchPgn, type GameState, type PgnFile, PGN_MEDIA_TYPE, pgnUrl } from './api';
import type { Orientation } from './board/geometry';
import './GameControls.css';

/** Hands a file to the browser to save, which is what following a download link does. */
function save({ name, text }: PgnFile): void {
  const url = URL.createObjectURL(new Blob([text], { type: PGN_MEDIA_TYPE }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  // In the page while it is clicked, because not every browser follows a link that is
  // in no document, and out again at once: it is nothing for anyone to look at.
  document.body.append(link);
  link.click();
  link.remove();
  // Freed once the browser has taken the file. Not on the next tick: the blob is read
  // as the download starts rather than while the click runs, and a URL revoked first
  // fails the download outright, silently. Holding a few kilobytes meanwhile is nothing.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Whether both sides are people, between whom a takeback or an abort is asked for. */
export function betweenTwoPeople(game: GameState): boolean {
  return game.white.accepts_moves && game.black.accepts_moves;
}

/** Whether aborting takes the other person's say: two people who have both moved. */
function abortIsAsked(game: GameState): boolean {
  return betweenTwoPeople(game) && game.moves.length >= 2;
}

interface GameControlsProps {
  /** Which side is at the bottom of the board. */
  orientation: Orientation;
  onFlip: () => void;
  /** The link the game is downloaded as PGN through. */
  link: string;
  /** The game, which says what there is left to do in it. */
  game: GameState;
  /** What this viewer's link may do in the game. */
  access: Access;
  /** Whether the server is out of reach, so nothing can be asked of the game. */
  disabled: boolean;
  onResign: () => void;
  onAbort: () => void;
  onTakeBack: () => void;
  /** Starts a new game with this one's settings, once this one has ended. */
  onPlayAgain: () => Promise<void>;
}

/**
 * Turning the board round and downloading the game, which are this browser's business
 * alone, and what changes the game itself, which only a play link may: a takeback, a
 * resignation and an abort. Between two people, a takeback and an abort once both have
 * moved are asked of the other, who answers them.
 *
 * Resigning and aborting are asked about first, since a game may have taken days, and a
 * click is all it takes to end it.
 */
export function GameControls({
  orientation,
  onFlip,
  link,
  game,
  access,
  disabled,
  onResign,
  onAbort,
  onTakeBack,
  onPlayAgain,
}: GameControlsProps) {
  const [exportFailed, setExportFailed] = useState<string | null>(null);
  const [againFailed, setAgainFailed] = useState<string | null>(null);
  const [startingAgain, setStartingAgain] = useState(false);
  const playing = game.position.game_over === null;
  const side = access === 'white' || access === 'black' ? access : null;
  // Nothing more can be asked while something is waiting for an answer.
  const waiting = game.request !== null;
  const asking = betweenTwoPeople(game);

  async function exportPgn() {
    setExportFailed(null);
    try {
      save(await fetchPgn(link));
    } catch (e: unknown) {
      setExportFailed(e instanceof Error ? e.message : String(e));
    }
  }

  async function playAgain() {
    setAgainFailed(null);
    setStartingAgain(true);
    try {
      await onPlayAgain();
    } catch (e: unknown) {
      setAgainFailed(e instanceof Error ? e.message : String(e));
    } finally {
      setStartingAgain(false);
    }
  }

  // Whoever played the game can start it again; a watcher cannot, and an aborted game is
  // gone, settings and all.
  const canPlayAgain =
    !playing && access !== 'watch' && game.position.game_over?.reason !== 'abort';

  function resign() {
    if (window.confirm('Resign this game? It ends at once, as a loss for you.')) {
      onResign();
    }
  }

  function abort() {
    const question = abortIsAsked(game)
      ? 'Ask your opponent to abort this game? If they agree, it is deleted, and nothing ' +
        'of it is kept.'
      : 'Abort this game? It is deleted at once, and nothing of it is kept.';
    if (window.confirm(question)) {
      onAbort();
    }
  }

  return (
    <div className="game-controls">
      <button type="button" onClick={onFlip}>
        {orientation === 'white' ? 'Flip to Black' : 'Flip to White'}
      </button>
      {/* Downloading is a request of its own: a game that has ended is as downloadable
          as one in progress, and a dropped connection leaves the link alone. */}
      <a
        className="download"
        href={pgnUrl(link)}
        download
        onClick={(event) => {
          // A real link, which "save link as" still follows, but a plain click is
          // answered here: the server can refuse a game that is gone, and a navigation
          // has nowhere to show that.
          event.preventDefault();
          void exportPgn();
        }}
      >
        Export PGN
      </a>
      {playing && side !== null && (
        <button
          type="button"
          disabled={disabled || waiting || game.moves.length === 0}
          onClick={onTakeBack}
        >
          {asking ? 'Ask to take back' : 'Take back'}
        </button>
      )}
      {playing && side !== null && (
        <button type="button" disabled={disabled} onClick={resign}>
          Resign
        </button>
      )}
      {playing && (side !== null || access === 'control') && (
        <button
          type="button"
          disabled={disabled || (waiting && abortIsAsked(game))}
          onClick={abort}
        >
          {abortIsAsked(game) ? 'Ask to abort' : 'Abort'}
        </button>
      )}
      {canPlayAgain && (
        <button type="button" disabled={startingAgain} onClick={() => void playAgain()}>
          Play again
        </button>
      )}
      {canPlayAgain && asking && (
        <p className="note">
          Play again joins the new game if your opponent has started one already.
        </p>
      )}
      {exportFailed !== null && <p role="alert">Could not export the game: {exportFailed}</p>}
      {againFailed !== null && <p role="alert">Could not start the game again: {againFailed}</p>}
    </div>
  );
}
