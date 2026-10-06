import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Access, GameState, PlayerInfo } from './api';
import { GameControls } from './GameControls';
import startPosition from './test/fixtures/start-position.json';

const PGN = '[Event "chess-ai game"]\n[Result "*"]\n\n1. e4 *\n';

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true, model: null, stockfish: null },
  random: { name: 'Random mover', accepts_moves: false, model: null, stockfish: null },
} satisfies Record<string, PlayerInfo>;

/** The kinds of player these tests set a game up between. */
type Playing = keyof typeof PLAYERS;

const POSITION = startPosition as GameState['position'];
const GAME = 'a-game';
const LINK = 'a-link';
const PGN_HREF = new URL(`/api/games/${LINK}/pgn`, window.location.href).href;

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

function gameOf(white: Playing, black: Playing, moves = 0): GameState {
  return {
    id: GAME,
    white: PLAYERS[white],
    black: PLAYERS[black],
    start_fen: POSITION.fen,
    // Only how many there are decides what can be taken back, not what they were.
    moves: Array.from({ length: moves }, () => ({ uci: 'e2e4', san: 'e4', thoughts: null })),
    position: POSITION,
    request: null,
    paused: null,
    replacements: [],
  };
}

/** The same game, ended. */
function ended(game: GameState): GameState {
  return {
    ...game,
    position: {
      ...game.position,
      legal_moves: {},
      game_over: { result: '0-1', reason: 'resignation' },
    },
  };
}

function renderControls(overrides: Partial<Parameters<typeof GameControls>[0]> = {}) {
  const props = {
    orientation: 'white' as const,
    onFlip: vi.fn(),
    link: LINK,
    game: gameOf('human', 'random', 1),
    access: 'white' as Access,
    disabled: false,
    onResign: vi.fn(),
    onAbort: vi.fn(),
    onTakeBack: vi.fn(),
    onPlayAgain: vi.fn().mockResolvedValue(undefined),
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
    // Ending a game is asked about first; these tests say yes unless they say otherwise.
    vi.spyOn(window, 'confirm').mockReturnValue(true);
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

  it('resigns the side the link plays, once that is confirmed', () => {
    const { onResign } = renderControls({ game: gameOf('human', 'random') });

    fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

    expect(window.confirm).toHaveBeenCalledOnce();
    expect(onResign).toHaveBeenCalledOnce();
  });

  it('resigns nothing when the confirmation is turned down', () => {
    vi.mocked(window.confirm).mockReturnValue(false);
    const { onResign, onAbort } = renderControls();

    fireEvent.click(screen.getByRole('button', { name: 'Resign' }));
    fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

    expect(onResign).not.toHaveBeenCalled();
    expect(onAbort).not.toHaveBeenCalled();
  });

  it('offers a watcher nothing to do to the game', () => {
    renderControls({ access: 'watch' });

    expect(buttons()).toEqual(['Flip to Black']);
    expect(screen.getByRole('link', { name: 'Export PGN' })).toBeInTheDocument();
  });

  it('offers only an abort through the link to a game nobody plays by hand', () => {
    renderControls({ game: gameOf('random', 'random', 1), access: 'control' });

    expect(buttons()).toEqual(['Flip to Black', 'Abort']);
  });

  it('asks the other person for a takeback, and for an abort once both have moved', () => {
    renderControls({ game: gameOf('human', 'human', 2) });

    expect(buttons()).toEqual(['Flip to Black', 'Ask to take back', 'Resign', 'Ask to abort']);
  });

  it('aborts outright between two people until both have moved', () => {
    renderControls({ game: gameOf('human', 'human', 1) });

    expect(buttons()).toEqual(['Flip to Black', 'Ask to take back', 'Resign', 'Abort']);
  });

  it('asks nothing more while a request is waiting for an answer', () => {
    const game = {
      ...gameOf('human', 'human', 2),
      request: { id: 1, kind: 'takeback', by: 'white' },
    };
    renderControls({ game: game as GameState });

    expect(screen.getByRole('button', { name: 'Ask to take back' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Ask to abort' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Resign' })).toBeEnabled();
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

  it('offers the game as a PGN file the browser saves', () => {
    renderControls();

    const link = screen.getByRole('link', { name: 'Export PGN' });

    expect(link).toHaveAttribute('href', PGN_HREF);
    // Downloaded rather than opened, and named by the server rather than here.
    expect(link).toHaveAttribute('download', '');
  });

  it('offers the download of a game that has ended, which there is nothing left to end', () => {
    renderControls({ game: ended(gameOf('human', 'random', 1)) });

    expect(buttons()).toEqual(['Flip to Black', 'Play again']);
    expect(screen.getByRole('link', { name: 'Export PGN' })).toBeInTheDocument();
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

  it('says why a game that is gone was not exported', async () => {
    // The one refusal a viewer can run into: they clicked before their browser heard
    // that the game had been aborted.
    fetch.mockResolvedValue(Response.json({ detail: 'there is no such game' }, { status: 404 }));
    renderControls();

    fireEvent.click(screen.getByRole('link', { name: 'Export PGN' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not export the game: there is no such game',
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

  describe('playing again', () => {
    it('is offered to a player once the game has ended', () => {
      const { onPlayAgain } = renderControls({ game: ended(gameOf('human', 'random', 1)) });

      fireEvent.click(screen.getByRole('button', { name: 'Play again' }));

      expect(onPlayAgain).toHaveBeenCalledOnce();
    });

    it('says that between two people it joins the game the other has started', () => {
      renderControls({ game: ended(gameOf('human', 'human', 2)) });

      expect(screen.getByText(/joins the new game if your opponent has started one/)).toBeVisible();
    });

    it('is offered to the control link of a game nobody played by hand', () => {
      renderControls({ game: ended(gameOf('random', 'random', 1)), access: 'control' });

      expect(screen.getByRole('button', { name: 'Play again' })).toBeInTheDocument();
    });

    it('is not offered while the game is being played', () => {
      renderControls();

      expect(screen.queryByRole('button', { name: 'Play again' })).not.toBeInTheDocument();
    });

    it('is not offered to a watcher', () => {
      renderControls({ game: ended(gameOf('human', 'random', 1)), access: 'watch' });

      expect(screen.queryByRole('button', { name: 'Play again' })).not.toBeInTheDocument();
    });

    it('is not offered after an abort, which kept nothing to play again from', () => {
      const game = gameOf('human', 'random', 1);
      renderControls({
        game: {
          ...game,
          position: {
            ...game.position,
            legal_moves: {},
            game_over: { result: '*', reason: 'abort' },
          },
        },
      });

      expect(screen.queryByRole('button', { name: 'Play again' })).not.toBeInTheDocument();
    });

    it('says why the game could not be started again', async () => {
      renderControls({
        game: ended(gameOf('human', 'random', 1)),
        onPlayAgain: vi.fn().mockRejectedValue(new Error('20 games are already being played')),
      });

      fireEvent.click(screen.getByRole('button', { name: 'Play again' }));

      expect(await screen.findByRole('alert')).toHaveTextContent(
        'Could not start the game again: 20 games are already being played',
      );
    });
  });
});
