import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NewGameForm } from './NewGameForm';

describe('NewGameForm', () => {
  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
  });

  it('starts a random game with the chosen delay under the path prefix', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json({}));
    vi.stubGlobal('fetch', fetch);
    render(<NewGameForm />);

    fireEvent.change(screen.getByLabelText('Delay between moves'), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));

    await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
    const [url, init] = fetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(new URL('/chess/api/game', window.location.href).href);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      white: 'random',
      black: 'random',
      move_delay: 2,
    });
    expect(await screen.findByRole('button', { name: 'Start' })).toBeEnabled();
  });

  it('defaults to half a second between moves', () => {
    render(<NewGameForm />);

    expect(screen.getByLabelText('Delay between moves')).toHaveDisplayValue('0.5 s');
  });

  it('reports a server error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));
    render(<NewGameForm />);

    fireEvent.click(screen.getByRole('button', { name: 'Start' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not start the game');
  });
});
