import { useId, useState } from 'react';
import type { CandidateMove, Color, GameState } from './api';
import type { Orientation } from './board/geometry';
import { percent, type ShownThoughts, type WhiteOdds, whiteOdds } from './thoughts';
import './ThoughtsPanel.css';

const STORAGE_KEY = 'chess-ai.show-thoughts';

function storedChoice(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'yes';
  } catch {
    // Storage this browser will not let the page use: off, as it starts.
    return false;
  }
}

/**
 * Whether this viewer wants to see what the model is thinking, remembered in this browser.
 * Off until they ask, since it shows a person what the model is about to play.
 */
export function useShowThoughts(): [boolean, (show: boolean) => void] {
  const [show, setShow] = useState(storedChoice);
  return [
    show,
    (next: boolean) => {
      setShow(next);
      try {
        window.localStorage.setItem(STORAGE_KEY, next ? 'yes' : 'no');
      } catch {
        // Not remembered, which costs a click next time and nothing else.
      }
    },
  ];
}

/** Whether either side is played by a model, which is the only player that has thoughts. */
export function modelPlays(game: GameState): boolean {
  return game.white.model !== null || game.black.model !== null;
}

/** The chances of each result as words, for the bar's label and the panel. */
export function described(odds: WhiteOdds): string {
  const { white, draw, black } = odds;
  return `White wins ${percent(white)}, draw ${percent(draw)}, Black wins ${percent(black)}`;
}

interface EvalBarProps {
  odds: WhiteOdds | null;
  /** Which side is at the bottom of the board, so that its chances are at the bottom too. */
  orientation: Orientation;
}

/**
 * The model's chances of each result as a bar beside the board, from White's side: White's
 * wins in white, draws in grey and Black's wins in black. Empty when there is no estimate.
 */
export function EvalBar({ odds, orientation }: EvalBarProps) {
  const label = odds === null ? 'No estimate' : described(odds);
  // Drawn top to bottom, so the side at the bottom of the board comes last.
  const segments: { side: 'white' | 'draw' | 'black'; share: number }[] =
    odds === null
      ? []
      : [
          { side: 'black' as const, share: odds.black },
          { side: 'draw' as const, share: odds.draw },
          { side: 'white' as const, share: odds.white },
        ];
  if (orientation === 'black') {
    segments.reverse();
  }
  return (
    <div className="eval-bar" role="img" aria-label={`Evaluation: ${label}`} title={label}>
      {segments.map(({ side, share }) => (
        <div
          key={side}
          className={`eval-segment ${side}`}
          data-side={side}
          style={{ flexGrow: share }}
        />
      ))}
    </div>
  );
}

interface CandidateListProps {
  label: string;
  /** Most likely first. */
  candidates: CandidateMove[];
  /** Whether a move is singled out, such as the one the model played. */
  marked: (uci: string) => boolean;
  /** What a move singled out is, said to a screen reader after it, such as "played". */
  markedAs: string;
}

/** Candidate moves with how likely each was, as words and as a meter. */
export function CandidateList({ label, candidates, marked, markedAs }: CandidateListProps) {
  return (
    <ol className="candidates" aria-label={label}>
      {candidates.map((candidate) => {
        const singled = marked(candidate.uci);
        return (
          <li
            key={candidate.uci}
            className={singled ? 'candidate marked' : 'candidate'}
            data-uci={candidate.uci}
          >
            <span className="candidate-move">
              {candidate.san ?? candidate.uci}
              {singled && <span className="visually-hidden"> ({markedAs})</span>}
            </span>
            <span className="candidate-meter" aria-hidden="true">
              <span style={{ width: percent(candidate.probability) }} />
            </span>
            <span className="candidate-share">{percent(candidate.probability)}</span>
          </li>
        );
      })}
    </ol>
  );
}

interface ThoughtsPanelProps {
  game: GameState;
  shown: ShownThoughts | null;
  show: boolean;
  onShow: (show: boolean) => void;
}

/**
 * Which side thought it, and when, as the heading over the list. By side rather than by name:
 * the players are named just above, and a run's name is too long to say twice.
 */
function heading(game: GameState, shown: ShownThoughts): string {
  const side = shown.side === 'white' ? 'White' : 'Black';
  if (shown.played === null) {
    return `${side} is considering…`;
  }
  const san = game.moves.at(-1)?.san ?? shown.played;
  return `${side} before playing ${san}`;
}

/**
 * The switch for the overlay, and what the model is thinking as a list: its candidate moves
 * with how likely it thought each, and its chances of each result from White's side.
 */
export function ThoughtsPanel({ game, shown, show, onShow }: ThoughtsPanelProps) {
  const id = useId();
  const sideOf = (side: Color) => (side === 'white' ? 'White' : 'Black');
  return (
    <section className="thoughts" aria-labelledby={`${id}-toggle`}>
      <label id={`${id}-toggle`} className="thoughts-toggle">
        <input type="checkbox" checked={show} onChange={(event) => onShow(event.target.checked)} />
        Show what the model is thinking
      </label>
      {show && shown === null && (
        <p className="note">
          Shown when the model has just moved, and while it waits out the move delay.
        </p>
      )}
      {show && shown !== null && (
        <>
          <p className="thoughts-heading">{heading(game, shown)}</p>
          <CandidateList
            label={`${sideOf(shown.side)}'s candidate moves`}
            candidates={shown.thoughts.candidates}
            marked={(uci) => uci === shown.played}
            markedAs="played"
          />
          {shown.thoughts.wdl !== null && (
            <p className="thoughts-odds">
              {described(whiteOdds(shown.thoughts.wdl, shown.side))}
            </p>
          )}
        </>
      )}
    </section>
  );
}
