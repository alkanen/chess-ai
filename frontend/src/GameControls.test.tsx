import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GameState, PlayerKind } from './api';
import { GameControls } from './GameControls';
import startPosition from './test/fixtures/start-position.json';

const PGN = '[Event "chess-ai game"]\n[Result "*"]\n\n1. e4 *\n';

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true },
  random: { name: 'Random mover', accepts_moves: false },
} satisfies Record<PlayerKind, { name: string; accepts_moves: boolean }>;

const POSITION = startPosition as GameState['position'];
const GAME = 'a-game';
const PGN_HREF = new URL(`/api/game/pgn?game=${GAME}`, window.location.href).href;

/** The PGN file the server hands over, named as the server names it. */
function pgnResponse(): Response {
  return new Response(PGN, {
    headers: {
      'Content-Type': 'application/x-chess-pgn',
      'Content-Disposition': 'attachment; filename="20260924-143005-a-game.pgn"',
    },
  });
}

/** Every file the browser has been handed to save, as a download link would hand it. */
const saved: { name: string; text: Promise<string> }[] = [];

function gameOf(white: PlayerKind, black: PlayerKind, moves = 0): GameState {
  return {
    id: GAME,
    white: PLAYERS[white],
    black: PLAYERS[black],
    start_fen: POSITION.fen,
    // Only how many there are decides what can be taken back, not what they were.
    moves: Array.from({ length: moves }, () => ({ uci: 'e2e4', san: 'e4', thoughts: null })),
    position: POSITION,
  };
}

function renderControls(overrides: Partial<Parameters<typeof GameControls>[0]> = {}) {
  const props = {
    orientation: 'white' as const,
    onFlip: vi.fn(),
    pgnGame: GAME as string | null,
    game: gameOf('human', 'random', 1),
    disabled: false,
    onResign: vi.fn(),
    onAbort: vi.fn(),
    onTakeBack: vi.fn(),
    ...overrides,
  };
  render(<GameControls {...props} />);
  return props;
}

/** Every control on offer, in the order it is shown. */
function buttons(): string[] {
  return screen.getAllByRole('button').map((button) => button.textContent ?? '');
}

describe('GameControls', () => {
  let fetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    saved.length = 0;
    fetch = vi.fn().mockResolvedValue(pgnResponse());
    vi.stubGlobal('fetch', fetch);
    // jsdom has neither object URLs nor downloads: the blob behind the URL is kept so
    // that the test can read what the browser was handed, and the link is not followed.
    const blobs = new Map<string, Blob>();
    URL.createObjectURL = vi.fn((blob: Blob) => {
      const url = `blob:${blobs.size}`;
      blobs.set(url, blob);
      return url;
    });
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLElement.prototype, 'click').mockImplementation(function (this: HTMLElement) {
      const link = this as HTMLAnchorElement;
      const blob = blobs.get(link.href);
      if (blob !== undefined) {
        saved.push({ name: link.download, text: blob.text() });
      }
    });
  });

  it('offers to turn the board round to the other side', () => {
    const { onFlip } = renderControls({ orientation: 'white' });

    fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

    expect(onFlip).toHaveBeenCalledOnce();
  });

  it('names the side the board would turn to', () => {
    renderControls({ orientation: 'black' });

    expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();
  });

  it('resigns the one side the viewer plays', () => {
    const { onResign } = renderControls({ game: gameOf('human', 'random') });

    fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('white');
  });

  it('resigns for Black when Black is the side the viewer plays', () => {
    const { onResign } = renderControls({ game: gameOf('random', 'human') });

    fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('black');
  });

  it('asks which side resigns when the viewer plays both', () => {
    const { onResign } = renderControls({ game: gameOf('human', 'human', 1) });

    expect(buttons()).toEqual([
      'Flip to Black',
      'Take back',
      'White resigns',
      'Black resigns',
      'Abort',
    ]);

    fireEvent.click(screen.getByRole('button', { name: 'Black resigns' }));

    expect(onResign).toHaveBeenCalledExactlyOnceWith('black');
  });

  it('offers no resignation in a game the viewer only watches, but still an abort', () => {
    renderControls({ game: gameOf('random', 'random', 1) });

    // Nobody plays a game of two players that move for themselves, so nobody takes a
    // move back in one either.
    expect(buttons()).toEqual(['Flip to Black', 'Abort']);
  });

  it('takes back the last move of a game the viewer plays', () => {
    const { onTakeBack } = renderControls({ game: gameOf('human', 'random', 1) });

    fireEvent.click(screen.getByRole('button', { name: 'Take back' }));

    expect(onTakeBack).toHaveBeenCalledOnce();
  });

  it('has nothing to take back before the first move', () => {
    renderControls({ game: gameOf('human', 'random', 0) });

    expect(screen.getByRole('button', { name: 'Take back' })).toBeDisabled();
  });

  it('aborts the game', () => {
    const { onAbort } = renderControls();

    fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

    expect(onAbort).toHaveBeenCalledOnce();
  });

  it('offers only the flip when there is no game to end', () => {
    renderControls({ game: null });

    expect(buttons()).toEqual(['Flip to Black']);
  });

  it('offers the game as a PGN file the browser saves', () => {
    renderControls();

    const link = screen.getByRole('link', { name: 'Export PGN' });

    expect(link).toHaveAttribute('href', PGN_HREF);
    // Downloaded rather than opened, and named by the server rather than here.
    expect(link).toHaveAttribute('download', '');
  });

  it('offers the download of a game that has ended, which there is nothing left to end', () => {
    renderControls({ game: null });

    expect(buttons()).toEqual(['Flip to Black']);
    expect(screen.getByRole('link', { name: 'Export PGN' })).toBeInTheDocument();
  });

  it('offers no download when no game has been started', () => {
    renderControls({ game: null, pgnGame: null });

    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('saves the game under the name the server gave it', async () => {
    renderControls();

    fireEvent.click(screen.getByRole('link', { name: 'Export PGN' }));

    await vi.waitFor(() => expect(saved).toHaveLength(1));
    expect(fetch).toHaveBeenCalledExactlyOnceWith(PGN_HREF);
    expect(saved[0].name).toBe('20260924-143005-a-game.pgn');
    expect(await saved[0].text).toBe(PGN);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('frees the file it handed over only long after the browser took it', async () => {
    // The blob is read as the download starts, not while the click is still running:
    // a URL revoked on the next tick can be gone before the browser has looked.
    vi.useFakeTimers();
    try {
      renderControls();

      fireEvent.click(screen.getByRole('link', { name: 'Export PGN' }));
      await vi.advanceTimersByTimeAsync(0);

      expect(saved).toHaveLength(1);
      expect(URL.revokeObjectURL).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(10_000);

      expect(URL.revokeObjectURL).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });

  it('says why a game that has been replaced was not exported', async () => {
    // The one refusal a viewer can run into: they clicked before their browser heard
    // that a new game had replaced the one they were looking at.
    fetch.mockResolvedValue(
      Response.json({ detail: 'that game has been replaced' }, { status: 409 }),
    );
    renderControls();

    fireEvent.click(screen.getByRole('link', { name: 'Export PGN' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not export the game: that game has been replaced',
    );
    expect(saved).toEqual([]);
  });

  it('keeps the flip working, and ends nothing, while the server is out of reach', () => {
    const { onFlip } = renderControls({ disabled: true });

    fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

    expect(onFlip).toHaveBeenCalledOnce();
    expect(screen.getByRole('button', { name: 'Resign' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Abort' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Take back' })).toBeDisabled();
  });
});
