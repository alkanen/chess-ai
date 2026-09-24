import { useCallback, useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  openPgn,
  openSavedGame,
  PGN_MEDIA_TYPE,
  type PositionSnapshot,
  type ReplayFile,
} from './api';
import { Board } from './board/Board';
import type { Orientation } from './board/geometry';
import { describeResult, GameStatus } from './GameStatus';
import { MoveList } from './MoveList';
import { ReplayControls } from './ReplayControls';
import { SavedGames } from './SavedGames';
// The board and the panel beside it are laid out as the game view's are.
import './App.css';
import './ReplayView.css';

/**
 * Where the game on show came from.
 *
 * A file the viewer opened is kept here as its text, because the server keeps nothing:
 * moving to another game in the same file sends it up again.
 */
type Source = { kind: 'saved'; name: string } | { kind: 'file'; name: string; pgn: string };

/** What the arrow keys do, which is what the buttons under the board do. */
const KEYS: Record<string, (at: number, plies: number) => number> = {
  ArrowLeft: (at) => at - 1,
  ArrowRight: (at) => at + 1,
  Home: () => 0,
  End: (_at, plies) => plies,
};

/**
 * Input types an arrow key does nothing to, so the game may have it.
 *
 * The file button is the one that matters: choosing a file leaves the focus on it, and
 * it is the commonest way into this view. Anything left out of this list is a field the
 * keys belong to — a text box, a number, a slider, and a radio group, whose selection
 * is moved with the very keys this would otherwise take.
 */
const PASSES_KEYS = ['file', 'checkbox', 'button', 'submit', 'reset', 'image'];

/** Whether a key pressed here was meant for the game rather than for a field of a form. */
function forTheGame(event: KeyboardEvent): boolean {
  const held = event.altKey || event.ctrlKey || event.metaKey;
  return !typing(event.target) && !event.defaultPrevented && !held;
}

/** Whether ``target`` is a field that does something of its own with an arrow key. */
function typing(target: EventTarget | null): boolean {
  if (target instanceof HTMLInputElement) {
    return !PASSES_KEYS.includes(target.type);
  }
  return target instanceof HTMLElement && ['SELECT', 'TEXTAREA'].includes(target.tagName);
}

/** Nothing worth reading: what PGN writes where a file says nothing. */
function said(part: string): boolean {
  return part !== '?' && part !== '????.??.??' && part !== '-' && part !== '';
}

export function ReplayView() {
  const [source, setSource] = useState<Source | null>(null);
  const [file, setFile] = useState<ReplayFile | null>(null);
  /** How many moves of the game are on the board; 0 is the position it started from. */
  const [ply, setPly] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const [orientation, setOrientation] = useState<Orientation>('white');
  /** Which request the answer being waited for belongs to; see `openGame`. */
  const asked = useRef(0);

  const plies = file?.selected.moves.length ?? 0;
  const goTo = useCallback((at: number) => setPly(Math.min(Math.max(at, 0), plies)), [plies]);

  // Stepping through the game from the keyboard, which is how anyone reads a game: the
  // listener is on the window, because the board itself is not something to focus.
  useEffect(() => {
    if (file === null) {
      return;
    }
    function step(event: KeyboardEvent) {
      const to = KEYS[event.key];
      if (to === undefined || !forTheGame(event)) {
        return;
      }
      // Home and End would otherwise scroll the page out from under the board.
      event.preventDefault();
      setPly((at) => Math.min(Math.max(to(at, plies), 0), plies));
    }
    window.addEventListener('keydown', step);
    return () => window.removeEventListener('keydown', step);
  }, [file, plies]);

  /**
   * Take the next request, and say afterwards whether it is still the one being waited
   * for: only what comes back for the last thing asked for is shown, however long
   * anything asked for earlier takes to arrive.
   */
  function asking(): () => boolean {
    const request = ++asked.current;
    return () => request === asked.current;
  }

  /**
   * Open a game, and show it from its first position.
   *
   * A viewer clicking through a list of saved games must not end up looking at
   * whichever one the server finished reading last.
   */
  async function openGame(opened: Source, game: number): Promise<void> {
    const wanted = asking();
    setOpening(true);
    setError(null);
    try {
      const read =
        opened.kind === 'saved'
          ? await openSavedGame(opened.name, game)
          : await openPgn(opened.pgn, game);
      if (!wanted()) {
        return;
      }
      setSource(opened);
      setFile(read);
      setPly(0);
    } catch (e: unknown) {
      if (wanted()) {
        couldNotOpen(e);
      }
    } finally {
      if (wanted()) {
        setOpening(false);
      }
    }
  }

  /**
   * Say why a game could not be opened, leaving nothing behind that the message does
   * not describe — the panel is no longer opening anything either.
   *
   * Only ever called for the request being waited for, so it always has the say.
   */
  function couldNotOpen(e: unknown): void {
    setSource(null);
    setFile(null);
    setError(e instanceof Error ? e.message : String(e));
    setOpening(false);
  }

  function chooseFile(event: ChangeEvent<HTMLInputElement>): void {
    const input = event.currentTarget;
    const chosen = input.files?.[0];
    if (chosen === undefined) {
      return;
    }
    // Forgotten at once, so that the same file can be chosen again after it is changed.
    input.value = '';
    // Reading the file is the first half of opening it, and counts as the asking: a
    // viewer who gives up on a slow file and opens a saved game instead is to keep the
    // game they opened, whichever way the reading of the file ends afterwards. What the
    // panel says goes with the request, so this takes that on as well as the token —
    // otherwise the request it supersedes leaves "Opening…" behind with nobody to clear
    // it, and a large file is read with the panel saying nothing at all meanwhile.
    const wanted = asking();
    setOpening(true);
    setError(null);
    void chosen.text().then(
      (pgn) => {
        if (wanted()) {
          void openGame({ kind: 'file', name: chosen.name, pgn }, 0);
        }
      },
      // A file can be moved or replaced between the picker closing and the browser
      // reading it, and a viewer left looking at an unchanged panel learns nothing.
      (e: unknown) => {
        if (wanted()) {
          couldNotOpen(e);
        }
      },
    );
  }

  const game = file?.selected ?? null;
  const position: PositionSnapshot | null =
    game === null ? null : ply === 0 ? game.start_position : game.moves[ply - 1].position;
  const occasion = game === null ? [] : [game.event, game.site, game.date].filter(said);

  return (
    <section className="replay" aria-labelledby="replay-heading">
      <h2 id="replay-heading">Replay</h2>
      {error !== null && <p role="alert">Could not open that game: {error}</p>}
      {position !== null && <GameStatus position={position} inGame />}
      <div className="game-view">
        <div className={position === null ? 'board-frame empty' : 'board-frame'}>
          {position === null ? (
            <p className="note">Open a saved game or a PGN file to step through it.</p>
          ) : (
            <Board snapshot={position} orientation={orientation} />
          )}
        </div>
        <aside className="side-panel">
          <label className="open-pgn">
            Open a PGN file
            <input type="file" accept={`.pgn,${PGN_MEDIA_TYPE}`} onChange={chooseFile} />
          </label>
          {opening && <p className="note">Opening…</p>}
          {game !== null && source !== null && (
            <>
              <p className="opened">{source.name}</p>
              {/* A file of one game has nothing to choose between. */}
              {file !== null && file.games.length > 1 && (
                <label className="choose-game">
                  Game
                  <select
                    value={game.index}
                    onChange={(event) => void openGame(source, Number(event.target.value))}
                  >
                    {file.games.map((one) => (
                      <option key={one.index} value={one.index}>
                        {one.index + 1}. {one.white} – {one.black}, {describeResult(one.result)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <dl className="players">
                <dt>White</dt>
                <dd>{game.white}</dd>
                <dt>Black</dt>
                <dd>{game.black}</dd>
                <dt>Result</dt>
                <dd>
                  {describeResult(game.result)}
                  {game.termination !== null && ` (${game.termination})`}
                </dd>
              </dl>
              {occasion.length > 0 && <p className="occasion">{occasion.join(' · ')}</p>}
              <ReplayControls
                ply={ply}
                plies={plies}
                onGo={goTo}
                orientation={orientation}
                onFlip={() => setOrientation((side) => (side === 'white' ? 'black' : 'white'))}
              />
              <MoveList
                moves={game.moves}
                startFen={game.start_fen}
                current={ply}
                onSelect={goTo}
              />
            </>
          )}
          <SavedGames
            current={source?.kind === 'saved' ? source.name : null}
            onOpen={(name) => void openGame({ kind: 'saved', name }, 0)}
          />
        </aside>
      </div>
    </section>
  );
}
