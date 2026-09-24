import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SavedGame } from './api';
import { SavedGames } from './SavedGames';

const GAMES: SavedGame[] = [
  {
    name: '20260925-090000-aaaaaaaa.pgn',
    event: 'chess-ai game',
    date: '2026.09.25',
    white: 'Human',
    black: 'Random mover',
    result: '1-0',
  },
  {
    name: '20260924-143005-3f9a1b2c.pgn',
    event: 'chess-ai game',
    date: '2026.09.24',
    white: 'Random mover',
    black: 'Random mover',
    result: '1/2-1/2',
  },
];

describe('SavedGames', () => {
  let fetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    fetch = vi.fn().mockResolvedValue(Response.json(GAMES));
    vi.stubGlobal('fetch', fetch);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('lists the saved games under the path prefix, with their dates and results', async () => {
    render(<SavedGames current={null} onOpen={vi.fn()} />);

    const games = await screen.findAllByRole('button');
    expect(fetch.mock.calls[0][0]).toBe(new URL('/chess/api/replay/saved', location.href).href);
    expect(games.map((game) => game.textContent)).toEqual([
      '2026.09.25Human – Random mover1–0',
      '2026.09.24Random mover – Random mover½–½',
    ]);
  });

  it('opens the game that is clicked, by the name the server gave it', async () => {
    const onOpen = vi.fn();
    render(<SavedGames current={null} onOpen={onOpen} />);

    fireEvent.click(await screen.findByRole('button', { name: /Human – Random mover/ }));

    expect(onOpen).toHaveBeenCalledExactlyOnceWith('20260925-090000-aaaaaaaa.pgn');
  });

  it('marks the game that is open', async () => {
    render(<SavedGames current="20260924-143005-3f9a1b2c.pgn" onOpen={vi.fn()} />);

    const games = await screen.findAllByRole('button');
    expect(games.map((game) => game.getAttribute('aria-current'))).toEqual([null, 'true']);
  });

  it('says so when no game has been saved yet', async () => {
    fetch.mockResolvedValue(Response.json([]));

    render(<SavedGames current={null} onOpen={vi.fn()} />);

    expect(await screen.findByText('No games have been saved here yet.')).toBeInTheDocument();
  });

  it('says why the list could not be shown', async () => {
    fetch.mockResolvedValue(Response.json({ detail: 'the games are elsewhere' }, { status: 500 }));

    render(<SavedGames current={null} onOpen={vi.fn()} />);

    expect(await screen.findByRole('alert')).toHaveTextContent('the games are elsewhere');
  });
});
