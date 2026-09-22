import { render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import startPosition from './test/fixtures/start-position.json';

describe('App', () => {
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

  it('shows the starting position from the server under its path prefix', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json(startPosition));
    vi.stubGlobal('fetch', fetch);

    render(<App />);

    expect(await screen.findByRole('status')).toHaveTextContent('White to move');
    expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      new URL('/chess/api/start-position', window.location.href).href,
      expect.anything(),
    );
  });

  it('reports a server error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 500 })));

    render(<App />);

    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load the position');
  });
});
