import { useState } from 'react';
import { type Color, fetchPgn, type GameState, type PgnFile, pgnUrl } from './api';
import type { Orientation } from './board/geometry';
import './GameControls.css';

/** Hands a file to the browser to save, which is what following a download link does. */
function save({ name, text }: PgnFile): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/x-chess-pgn' }));
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

/** The sides a viewer may resign: the ones whose moves this browser plays. */
function resignable(game: GameState | null): Color[] {
  if (game === null) {
    return [];
  }
  const sides: Color[] = [];
  if (game.white.accepts_moves) sides.push('white');
  if (game.black.accepts_moves) sides.push('black');
  return sides;
}

interface GameControlsProps {
  /** Which side is at the bottom of the board. */
  orientation: Orientation;
  onFlip: () => void;
  /** The id of the game to download as PGN, or null when there is no game to download. */
  pgnGame: string | null;
  /** The game that can still be ended, or null when there is none to act on. */
  game: GameState | null;
  /** Whether the server is out of reach, so nothing can be asked of the game. */
  disabled: boolean;
  onResign: (color: Color) => void;
  onAbort: () => void;
  onTakeBack: () => void;
}

/**
 * Turning the board round and downloading the game, which are this browser's business
 * alone, and what changes the game itself, which is everyone's: a takeback, a
 * resignation and an abort all reach every viewer.
 */
export function GameControls({
  orientation,
  onFlip,
  pgnGame,
  game,
  disabled,
  onResign,
  onAbort,
  onTakeBack,
}: GameControlsProps) {
  const sides = resignable(game);
  const [exportFailed, setExportFailed] = useState<string | null>(null);

  async function exportPgn(id: string) {
    setExportFailed(null);
    try {
      save(await fetchPgn(id));
    } catch (e: unknown) {
      setExportFailed(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="game-controls">
      <button type="button" onClick={onFlip}>
        {orientation === 'white' ? 'Flip to Black' : 'Flip to White'}
      </button>
      {/* Downloading is a request of its own: a game that has ended is as downloadable
          as one in progress, and a dropped connection leaves the link alone. */}
      {pgnGame !== null && (
        <a
          className="download"
          href={pgnUrl(pgnGame)}
          download
          onClick={(event) => {
            // A real link, which "save link as" still follows, but a plain click is
            // answered here: the server can refuse a game that has been replaced, and
            // a navigation has nowhere to show that.
            event.preventDefault();
            void exportPgn(pgnGame);
          }}
        >
          Export PGN
        </a>
      )}
      {/* Only a game somebody is playing has a move to give back to them. */}
      {sides.length > 0 && (
        <button
          type="button"
          disabled={disabled || game === null || game.moves.length === 0}
          onClick={onTakeBack}
        >
          Take back
        </button>
      )}
      {sides.map((color) => (
        <button key={color} type="button" disabled={disabled} onClick={() => onResign(color)}>
          {/* One side to resign needs no saying which; two do. */}
          {sides.length === 1 ? 'Resign' : `${color === 'white' ? 'White' : 'Black'} resigns`}
        </button>
      ))}
      {game !== null && (
        <button type="button" disabled={disabled} onClick={onAbort}>
          Abort
        </button>
      )}
      {exportFailed !== null && <p role="alert">Could not export the game: {exportFailed}</p>}
    </div>
  );
}
