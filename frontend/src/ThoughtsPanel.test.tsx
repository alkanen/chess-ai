import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { GameState, Thoughts } from './api';
import { EvalBar, ThoughtsPanel, useShowThoughts } from './ThoughtsPanel';

const THOUGHTS: Thoughts = {
  candidates: [
    { uci: 'e7e5', san: 'e5', probability: 0.62 },
    { uci: 'c7c5', san: 'c5', probability: 0.25 },
    { uci: 'e7e6', probability: 0.05 },
  ],
  wdl: { win: 0.5, draw: 0.3, loss: 0.2 },
};

const GAME = {
  white: { name: 'Human', model: null },
  black: { name: 'tiny step 4', model: {} },
  moves: [
    { uci: 'e2e4', san: 'e4', thoughts: null },
    { uci: 'e7e5', san: 'e5', thoughts: THOUGHTS },
  ],
} as unknown as GameState;

function Toggled() {
  const [show, setShow] = useShowThoughts();
  return <ThoughtsPanel game={GAME} shown={null} show={show} onShow={setShow} />;
}

describe('ThoughtsPanel', () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => window.localStorage.clear());

  it('lists the candidates with how likely each was, and marks the one played', () => {
    render(
      <ThoughtsPanel
        game={GAME}
        shown={{ thoughts: THOUGHTS, side: 'black', played: 'e7e5' }}
        show
        onShow={() => {}}
      />,
    );

    expect(screen.getByText('Black before playing e5')).toBeInTheDocument();
    const items = screen.getAllByRole('listitem');
    expect(items.map((item) => item.textContent)).toEqual([
      'e5 (played)62%',
      'c525%',
      // Thoughts kept from before candidates were written out fall back to UCI.
      'e7e65%',
    ]);
    // From White's side: Black's win is White's loss.
    expect(screen.getByText('White wins 20%, draw 30%, Black wins 50%')).toBeInTheDocument();
  });

  it('says the model is considering while its move is held back', () => {
    render(
      <ThoughtsPanel
        game={GAME}
        shown={{ thoughts: THOUGHTS, side: 'black', played: null }}
        show
        onShow={() => {}}
      />,
    );

    expect(screen.getByText('Black is considering…')).toBeInTheDocument();
    expect(screen.queryByText(/played/)).not.toBeInTheDocument();
  });

  it('shows nothing but the switch until asked, and remembers being asked', () => {
    const { unmount } = render(<Toggled />);
    expect(screen.getByRole('checkbox')).not.toBeChecked();
    expect(screen.queryByText(/Shown when/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox'));
    expect(screen.getByText(/Shown when the model has just moved/)).toBeInTheDocument();
    unmount();

    render(<Toggled />);
    expect(screen.getByRole('checkbox')).toBeChecked();
  });
});

describe('EvalBar', () => {
  const odds = { white: 0.2, draw: 0.3, black: 0.5 };

  function segments(container: HTMLElement) {
    return Array.from(container.querySelectorAll<HTMLElement>('.eval-segment')).map(
      (segment) => [segment.dataset.side, Number(segment.style.flexGrow)],
    );
  }

  it("puts White's chances at White's end of the board", () => {
    const { container } = render(<EvalBar odds={odds} orientation="white" />);
    expect(segments(container)).toEqual([
      ['black', 0.5],
      ['draw', 0.3],
      ['white', 0.2],
    ]);
    expect(
      screen.getByRole('img', {
        name: 'Evaluation: White wins 20%, draw 30%, Black wins 50%',
      }),
    ).toBeInTheDocument();
  });

  it('turns round with the board', () => {
    const { container } = render(<EvalBar odds={odds} orientation="black" />);
    expect(segments(container).map(([side]) => side)).toEqual(['white', 'draw', 'black']);
  });

  it('is empty without an estimate', () => {
    const { container } = render(<EvalBar odds={null} orientation="white" />);
    expect(segments(container)).toEqual([]);
    expect(screen.getByRole('img', { name: 'Evaluation: No estimate' })).toBeInTheDocument();
  });
});
