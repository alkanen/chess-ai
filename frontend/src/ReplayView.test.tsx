import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SavedGame } from './api';
import { ReplayView } from './ReplayView';
import { checkSquare, lastMoveSquares } from './test/boardQueries';
import { foolsMateFile, scholarsMateFile } from './test/replayFile';

const SAVED: SavedGame[] = [
  {
    name: '20260924-143005-3f9a1b2c.pgn',
    event: 'chess-ai game',
    date: '2026.09.24',
    white: 'Human',
    black: 'Random mover',
    result: '0-1',
  },
];

/** The two games of the fixture file, by the number the viewer asks for them with. */
const FILE = [foolsMateFile, scholarsMateFile];

/** A PGN file as a browser hands it over: its text is never read here, the stub answers. */
function pgnFile(name = 'two-games.pgn'): File {
  return new File(['[Event "chess-ai game"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'], name, {
    type: 'application/x-chess-pgn',
  });
}

/** Opens a PGN file, as choosing one in the file picker does. */
async function open(file = pgnFile()): Promise<void> {
  fireEvent.change(screen.getByLabelText('Open a PGN file'), { target: { files: [file] } });
  await screen.findByRole('status');
}

/** The server refusing to read a file, while the list of saved games still works. */
function refuses(fetch: ReturnType<typeof vi.fn>, detail: string, status: number): void {
  fetch.mockImplementation((url: string) =>
    Promise.resolve(
      new URL(url).pathname === '/chess/api/replay/saved'
        ? Response.json(SAVED)
        : Response.json({ detail }, { status }),
    ),
  );
}

/** What the board is showing: the move that led here, and the king in check, if any. */
function board(container: HTMLElement): { lastMove: string[]; check: string | null } {
  return { lastMove: lastMoveSquares(container), check: checkSquare(container) };
}

describe('ReplayView', () => {
  let fetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    fetch = vi.fn((url: string) => {
      const asked = new URL(url);
      if (asked.pathname === '/chess/api/replay/saved') {
        return Promise.resolve(Response.json(SAVED));
      }
      return Promise.resolve(Response.json(FILE[Number(asked.searchParams.get('game') ?? 0)]));
    });
    vi.stubGlobal('fetch', fetch);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('has nothing on the board until a game is opened', () => {
    render(<ReplayView />);

    expect(screen.getByText(/Open a saved game or a PGN file/)).toBeInTheDocument();
    expect(screen.queryByRole('img', { name: 'white king on e1' })).not.toBeInTheDocument();
  });

  it('sends an opened PGN file to the server and shows where the game began', async () => {
    render(<ReplayView />);

    await open();

    const [url, init] = fetch.mock.calls.at(-1) as [string, RequestInit];
    expect(url).toBe(new URL('/chess/api/replay/pgn?game=0', location.href).href);
    expect(init.method).toBe('POST');
    expect(screen.getByRole('status')).toHaveTextContent('White to move');
    expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument();
    expect(screen.getByText('two-games.pgn')).toBeInTheDocument();
  });

  describe('stepping through a game', () => {
    it('goes forward and back a move at a time', async () => {
      const { container } = render(<ReplayView />);
      await open();

      fireEvent.click(screen.getByRole('button', { name: 'Forward' }));

      expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();
      expect(board(container)).toEqual({ lastMove: ['f2', 'f3'], check: null });
      expect(screen.getByRole('status')).toHaveTextContent('Black to move');
      expect(screen.getByText(/Move 1 of 4/)).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: 'Back' }));

      expect(screen.getByRole('img', { name: 'white pawn on f2' })).toBeInTheDocument();
      expect(board(container)).toEqual({ lastMove: [], check: null });
      expect(screen.getByRole('status')).toHaveTextContent('White to move');
    });

    it('goes to the end of the game and back to its start', async () => {
      const { container } = render(<ReplayView />);
      await open();

      fireEvent.click(screen.getByRole('button', { name: 'End' }));

      expect(screen.getByRole('img', { name: 'black queen on h4' })).toBeInTheDocument();
      // The mate: the move that gave it, and the king it is given to.
      expect(board(container)).toEqual({ lastMove: ['d8', 'h4'], check: 'e1' });
      expect(screen.getByRole('status')).toHaveTextContent('Black wins by checkmate (0–1)');
      expect(screen.getByText(/Move 4 of 4/)).toBeInTheDocument();

      fireEvent.click(screen.getByRole('button', { name: 'Start' }));

      expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument();
      expect(board(container)).toEqual({ lastMove: [], check: null });
    });

    it('walks the game with the arrow keys', async () => {
      const { container } = render(<ReplayView />);
      await open();

      fireEvent.keyDown(window, { key: 'ArrowRight' });
      fireEvent.keyDown(window, { key: 'ArrowRight' });

      expect(screen.getByRole('img', { name: 'black pawn on e5' })).toBeInTheDocument();
      expect(board(container)).toEqual({ lastMove: ['e5', 'e7'], check: null });

      fireEvent.keyDown(window, { key: 'ArrowLeft' });

      expect(screen.getByText(/Move 1 of 4/)).toBeInTheDocument();

      fireEvent.keyDown(window, { key: 'End' });

      expect(board(container)).toEqual({ lastMove: ['d8', 'h4'], check: 'e1' });

      fireEvent.keyDown(window, { key: 'Home' });

      expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument();
    });

    it('stays at the ends of the game however often they are stepped past', async () => {
      render(<ReplayView />);
      await open();

      fireEvent.keyDown(window, { key: 'ArrowLeft' });
      expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument();

      for (let step = 0; step < 6; step += 1) {
        fireEvent.keyDown(window, { key: 'ArrowRight' });
      }

      expect(screen.getByText(/Move 4 of 4/)).toBeInTheDocument();
    });

    it('jumps to a move clicked in the list, and marks it', async () => {
      const { container } = render(<ReplayView />);
      await open();

      fireEvent.click(screen.getByRole('button', { name: 'e5' }));

      expect(board(container)).toEqual({ lastMove: ['e5', 'e7'], check: null });
      expect(screen.getByRole('button', { name: 'e5' })).toHaveAttribute('aria-current', 'step');
      expect(screen.getByText(/Move 2 of 4/)).toBeInTheDocument();
    });

    it('steps with the arrow keys while the file button still has the focus', async () => {
      // Choosing a file leaves the focus on the input it was chosen with, which is the
      // commonest way into this view: the keys the controls advertise must work there.
      render(<ReplayView />);
      await open();

      fireEvent.keyDown(screen.getByLabelText('Open a PGN file'), { key: 'ArrowRight' });

      expect(screen.getByText(/Move 1 of 4/)).toBeInTheDocument();
    });

    it('leaves the arrow keys to a radio group, which moves its selection with them', async () => {
      render(<ReplayView />);
      await open();
      const radio = document.createElement('input');
      radio.type = 'radio';
      document.body.append(radio);

      fireEvent.keyDown(radio, { key: 'ArrowRight' });
      radio.remove();

      expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument();
    });

    it('leaves the arrow keys to the fields of a form', async () => {
      render(<ReplayView />);
      await open();

      fireEvent.keyDown(screen.getByLabelText('Game'), { key: 'ArrowRight' });

      expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument();
    });

    it('turns the board round without leaving the position', async () => {
      const { container } = render(<ReplayView />);
      await open();
      fireEvent.click(screen.getByRole('button', { name: 'Forward' }));

      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

      expect(screen.getByText(/Move 1 of 4/)).toBeInTheDocument();
      expect(board(container)).toEqual({ lastMove: ['f2', 'f3'], check: null });
      expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();
    });
  });

  describe('a file of several games', () => {
    it('offers every game in it', async () => {
      render(<ReplayView />);

      await open();

      expect(
        Array.from(screen.getByLabelText('Game').querySelectorAll('option'), (o) => o.textContent),
      ).toEqual(['1. Human – Random mover, 0–1', '2. Alice – Bob, 1–0']);
    });

    it('sends the file up again for the game that is chosen', async () => {
      render(<ReplayView />);
      await open();

      fireEvent.change(screen.getByLabelText('Game'), { target: { value: '1' } });

      await waitFor(() => expect(screen.getByText(/Move 0 of 7/)).toBeInTheDocument());
      const [url] = fetch.mock.calls.at(-1) as [string];
      expect(url).toBe(new URL('/chess/api/replay/pgn?game=1', location.href).href);
      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Qxf7#' })).toBeInTheDocument();
    });
  });

  describe('saved games', () => {
    it('opens one from the list, by the name the server gave it', async () => {
      render(<ReplayView />);

      fireEvent.click(await screen.findByRole('button', { name: /Human – Random mover/ }));

      await waitFor(() => expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument());
      const [url] = fetch.mock.calls.at(-1) as [string];
      expect(url).toBe(
        new URL('/chess/api/replay/saved/20260924-143005-3f9a1b2c.pgn?game=0', location.href).href,
      );
      expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    });

    it('names the game that ended it, where the file says', async () => {
      render(<ReplayView />);

      fireEvent.click(await screen.findByRole('button', { name: /Human – Random mover/ }));

      expect(await screen.findByText('0–1 (checkmate)')).toBeInTheDocument();
    });
  });

  describe('a file that takes a moment to read', () => {
    /** A file whose reading is settled by the test, standing in for a slow one. */
    function slowFile(): { file: File; read: (pgn: string) => void; fail: (e: Error) => void } {
      let read = (_pgn: string) => {};
      let fail = (_e: Error) => {};
      const text = () =>
        new Promise<string>((resolve, reject) => {
          read = resolve;
          fail = reject;
        });
      return {
        file: { name: 'slow.pgn', text } as unknown as File,
        read: (pgn) => read(pgn),
        fail: (e) => fail(e),
      };
    }

    /** Opens a slow file, then opens a saved game instead while it is still being read. */
    async function movedOn() {
      const slow = slowFile();
      fireEvent.change(screen.getByLabelText('Open a PGN file'), {
        target: { files: [slow.file] },
      });
      fireEvent.click(await screen.findByRole('button', { name: /Human – Random mover/ }));
      await waitFor(() => expect(screen.getByText(/Move 0 of 4/)).toBeInTheDocument());
      return slow;
    }

    it('says it is opening while the file is being read', async () => {
      render(<ReplayView />);
      const slow = slowFile();

      fireEvent.change(screen.getByLabelText('Open a PGN file'), {
        target: { files: [slow.file] },
      });

      expect(screen.getByText('Opening…')).toBeInTheDocument();
      await act(async () => slow.read('[Event "x"]\n[White "A"]\n[Black "B"]\n\n1. e4 *\n'));
      expect(screen.queryByText('Opening…')).not.toBeInTheDocument();
    });

    it('stops saying it is opening when the game it moved on to is dropped', async () => {
      // The saved game is slow to arrive, so the viewer gives up and chooses a file;
      // the saved game then answers and is dropped, and the file will not read either.
      let answer = (_answer: Response) => {};
      fetch.mockImplementation((url: string) =>
        new URL(url).pathname === '/chess/api/replay/saved'
          ? Promise.resolve(Response.json(SAVED))
          : new Promise<Response>((resolve) => {
              answer = resolve;
            }),
      );
      render(<ReplayView />);
      fireEvent.click(await screen.findByRole('button', { name: /Human – Random mover/ }));
      expect(screen.getByText('Opening…')).toBeInTheDocument();
      const slow = slowFile();
      fireEvent.change(screen.getByLabelText('Open a PGN file'), {
        target: { files: [slow.file] },
      });

      await act(async () => answer(Response.json(FILE[0])));
      await act(async () => slow.fail(new Error('the file has gone')));

      expect(await screen.findByRole('alert')).toHaveTextContent('the file has gone');
      expect(screen.queryByText('Opening…')).not.toBeInTheDocument();
    });

    it('says nothing when the file it has moved on from turns out to be unreadable', async () => {
      render(<ReplayView />);
      const slow = await movedOn();

      await act(async () => slow.fail(new Error('the file has gone')));

      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
      expect(screen.getByText('20260924-143005-3f9a1b2c.pgn')).toBeInTheDocument();
    });

    it('does not put the file on the board when it arrives after the game asked for', async () => {
      render(<ReplayView />);
      const slow = await movedOn();

      await act(async () => slow.read('[Event "Late"]\n\n1. e4 *\n'));

      expect(screen.getByText('20260924-143005-3f9a1b2c.pgn')).toBeInTheDocument();
      expect(screen.queryByText('slow.pgn')).not.toBeInTheDocument();
    });
  });

  describe('a file that cannot be read', () => {
    it('says so when the file itself cannot be read', async () => {
      // A file can go away between the picker closing and the browser reading it.
      const gone = {
        name: 'gone.pgn',
        text: () => Promise.reject(new Error('the file has gone')),
      } as unknown as File;
      render(<ReplayView />);

      fireEvent.change(screen.getByLabelText('Open a PGN file'), { target: { files: [gone] } });

      expect(await screen.findByRole('alert')).toHaveTextContent('the file has gone');
    });

    it('says what is wrong with it, and puts nothing on the board', async () => {
      refuses(fetch, 'no game was found in that file', 400);
      render(<ReplayView />);

      fireEvent.change(screen.getByLabelText('Open a PGN file'), {
        target: { files: [pgnFile('shopping-list.txt')] },
      });

      expect(await screen.findByRole('alert')).toHaveTextContent(
        'Could not open that game: no game was found in that file',
      );
      expect(screen.queryByRole('img', { name: 'white king on e1' })).not.toBeInTheDocument();
      expect(screen.getByText(/Open a saved game or a PGN file/)).toBeInTheDocument();
    });

    it('drops the game it was showing, so that nothing on the page is the wrong game', async () => {
      render(<ReplayView />);
      await open();
      refuses(fetch, 'that file has no game 1', 404);

      fireEvent.change(screen.getByLabelText('Open a PGN file'), {
        target: { files: [pgnFile('empty.pgn')] },
      });

      expect(await screen.findByRole('alert')).toHaveTextContent('that file has no game 1');
      expect(screen.queryByText(/Move 0 of 4/)).not.toBeInTheDocument();
    });
  });
});
