import { describe, expect, it } from 'vitest';
import type { GameState, MoveRecord, PositionSnapshot, Thoughts } from './api';
import { candidateArrows, shownThoughts, whiteOdds } from './thoughts';

const THOUGHTS: Thoughts = {
  candidates: [
    { uci: 'e7e5', san: 'e5', probability: 0.6 },
    { uci: 'c7c5', san: 'c5', probability: 0.3 },
    { uci: 'e7e6', san: 'e6', probability: 0.1 },
  ],
  wdl: { win: 0.5, draw: 0.3, loss: 0.2 },
};

/** A game after ``moves``, with ``turn`` on move. */
function game(moves: MoveRecord[], turn: 'white' | 'black', considering = null): GameState {
  return {
    moves,
    position: { turn } as PositionSnapshot,
    considering,
  } as unknown as GameState;
}

const human: MoveRecord = { uci: 'e2e4', san: 'e4', thoughts: null };
const model: MoveRecord = { uci: 'e7e5', san: 'e5', thoughts: THOUGHTS };

describe('shownThoughts', () => {
  it('shows what the model considered before the move it has just played', () => {
    expect(shownThoughts(game([human, model], 'white'))).toEqual({
      thoughts: THOUGHTS,
      side: 'black',
      played: 'e7e5',
    });
  });

  it('shows what the model is considering while its move is held back', () => {
    const considering = { ply: 1, side: 'black', thoughts: THOUGHTS };
    expect(shownThoughts(game([human], 'black', considering as never))).toEqual({
      thoughts: THOUGHTS,
      side: 'black',
      played: null,
    });
  });

  it('shows nothing once the other side has moved since', () => {
    const reply: MoveRecord = { uci: 'g1f3', san: 'Nf3', thoughts: null };
    expect(shownThoughts(game([human, model, reply], 'black'))).toBeNull();
  });

  it('copes with a server that does not say what is being considered', () => {
    const older = { ...game([human, model], 'white') } as Partial<GameState>;
    delete older.considering;
    expect(shownThoughts(older as GameState)?.played).toBe('e7e5');
  });

  it('ignores what was considered in a position the game has left', () => {
    const considering = { ply: 0, side: 'white', thoughts: THOUGHTS };
    expect(shownThoughts(game([human], 'black', considering as never))).toBeNull();
  });
});

describe('whiteOdds', () => {
  it("turns a model's own chances into White's", () => {
    const wdl = { win: 0.5, draw: 0.3, loss: 0.2 };
    expect(whiteOdds(wdl, 'white')).toEqual({ white: 0.5, draw: 0.3, black: 0.2 });
    expect(whiteOdds(wdl, 'black')).toEqual({ white: 0.2, draw: 0.3, black: 0.5 });
  });
});

describe('candidateArrows', () => {
  it('draws the likeliest last, on top, and marks the move played', () => {
    const arrows = candidateArrows({ thoughts: THOUGHTS, side: 'black', played: 'c7c5' });
    expect(arrows.map((arrow) => [arrow.from, arrow.to, arrow.played])).toEqual([
      ['e7', 'e6', false],
      ['c7', 'c5', true],
      ['e7', 'e5', false],
    ]);
  });

  it('draws the promotions of one pawn to one square as one arrow, with their chances added', () => {
    // The policy spreads a promotion across the pieces, so several are often in the top five.
    // Drawn apart they would lie exactly on top of each other, with one React key between them.
    const promotions: Thoughts = {
      candidates: [
        { uci: 'e7e8q', san: 'e8=Q', probability: 0.5 },
        { uci: 'd2d4', san: 'd4', probability: 0.15 },
        { uci: 'e7e8n', san: 'e8=N', probability: 0.2 },
      ],
      wdl: null,
    };
    const arrows = candidateArrows({ thoughts: promotions, side: 'white', played: 'e7e8n' });
    expect(arrows.map((arrow) => [arrow.from, arrow.to, arrow.weight, arrow.played])).toEqual([
      ['d2', 'd4', 0.15, false],
      // Played if any of them was: the knight was, so the arrow says so.
      ['e7', 'e8', 0.7, true],
    ]);
  });

  it('draws a promotion from its square to its square', () => {
    const promotion: Thoughts = {
      candidates: [{ uci: 'a7a8q', san: 'a8=Q', probability: 0.9 }],
      wdl: null,
    };
    const [arrow] = candidateArrows({ thoughts: promotion, side: 'white', played: null });
    expect([arrow.from, arrow.to]).toEqual(['a7', 'a8']);
  });
});
