import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NewGameForm } from './NewGameForm';

/** Starts a game with the form as it stands and returns what was posted. */
async function start(fetch: ReturnType<typeof vi.fn>): Promise<unknown> {
  fireEvent.click(screen.getByRole('button', { name: 'Start' }));
  await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
  const [, init] = fetch.mock.calls[0] as [string, RequestInit];
  return JSON.parse(init.body as string);
}

describe('NewGameForm', () => {
  let fetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    fetch = vi.fn().mockResolvedValue(Response.json({}));
    vi.stubGlobal('fetch', fetch);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('starts a game with the chosen delay under the path prefix', async () => {
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Delay between moves'), { target: { value: '2' } });
    const posted = await start(fetch);

    const [url, init] = fetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(new URL('/chess/api/game', window.location.href).href);
    expect(init.method).toBe('POST');
    expect(posted).toEqual({ white: 'human', black: 'random', move_delay: 2, fen: null });
    expect(await screen.findByRole('button', { name: 'Start' })).toBeEnabled();
  });

  it('offers you the white pieces against a random mover to begin with', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('White')).toHaveDisplayValue('Human');
    expect(screen.getByLabelText('Black')).toHaveDisplayValue('Random mover');
    expect(screen.getByLabelText('Delay between moves')).toHaveDisplayValue('0.5 s');
  });

  it.each([
    ['human', 'human'],
    ['human', 'random'],
    ['random', 'random'],
  ])('starts %s against %s', async (white, black) => {
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('White'), { target: { value: white } });
    fireEvent.change(screen.getByLabelText('Black'), { target: { value: black } });

    expect(await start(fetch)).toEqual({ white, black, move_delay: 0.5, fen: null });
  });

  it('paces nothing when both sides are played by hand, and says so', () => {
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'human' } });

    expect(screen.getByLabelText('Delay between moves')).toBeDisabled();
    expect(screen.getByText(/nothing to pace/i)).toBeInTheDocument();
  });

  it('paces the moves of a player that moves for itself', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('Delay between moves')).toBeEnabled();
    expect(screen.queryByText(/nothing to pace/i)).not.toBeInTheDocument();
  });

  it('keeps the chosen delay while both sides are played by hand', async () => {
    render(<NewGameForm />);
    fireEvent.change(screen.getByLabelText('Delay between moves'), { target: { value: '2' } });

    fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'human' } });
    fireEvent.change(screen.getByLabelText('Black'), { target: { value: 'random' } });

    const delay = screen.getByLabelText('Delay between moves');
    expect(delay).toBeEnabled();
    expect(delay).toHaveDisplayValue('2 s');
    expect(await start(fetch)).toEqual({
      white: 'human',
      black: 'random',
      move_delay: 2,
      fen: null,
    });
  });

  it('reports a server error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));
    render(<NewGameForm />);

    fireEvent.click(screen.getByRole('button', { name: 'Start' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not start the game');
  });

  it('starts from a position typed into the FEN box', async () => {
    const fen = 'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2';
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Start from FEN'), { target: { value: ` ${fen} ` } });

    expect(await start(fetch)).toEqual({
      white: 'human',
      black: 'random',
      move_delay: 0.5,
      // Typed-in positions come with whatever was pasted around them.
      fen,
    });
  });

  it('says why the server would not start from the FEN', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(Response.json({ detail: 'White has no king' }, { status: 400 })),
    );
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Start from FEN'), { target: { value: '8/8/8' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not start the game: White has no king',
    );
  });

  it('keeps the FEN box empty for a game from the usual starting position', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('Start from FEN')).toHaveValue('');
  });
});
