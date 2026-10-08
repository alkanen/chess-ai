import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Access, PlayerInfo, PositionSnapshot } from './api';
import { App } from './App';
import {
  checkSquare,
  destinations,
  lastMoveSquares,
  promotionChoice,
  square,
  squareAt,
} from './test/boardQueries';
import { FakeWebSocket } from './test/fakeWebSocket';
import startPosition from './test/fixtures/start-position.json';
import { foolsMateMoves, foolsMateStart, type StateEvent } from './test/foolsMate';
import { castling, check, drawnByFiftyMoves, promotion } from './test/positions';

// uPlot draws on a canvas, which jsdom does not have; see RunView.test.tsx for the charts.
vi.mock('uplot', async () => ({ default: (await import('./test/fakeUPlot')).FakeUPlot }));

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true, model: null, stockfish: null },
  random: { name: 'Random mover', accepts_moves: false, model: null, stockfish: null },
} satisfies Record<string, PlayerInfo>;

/** The kinds of player these tests set a game up between. */
type Playing = keyof typeof PLAYERS;

const GAME = 'the-game';
const LINK = 'the-link';

/**
 * What a link to a game between `white` and `black` may do: play the side a person plays,
 * White if both do, and abort a game nobody plays by hand.
 */
function accessTo(white: Playing, black: Playing): Access {
  if (white === 'human') return 'white';
  if (black === 'human') return 'black';
  return 'control';
}

/** A game at the given position, between the given kinds of player, as a link sees it. */
function gameOf(
  white: Playing,
  black: Playing,
  position: PositionSnapshot,
  access: Access = accessTo(white, black),
): StateEvent {
  return {
    type: 'state',
    game: {
      id: GAME,
      white: PLAYERS[white],
      black: PLAYERS[black],
      start_fen: position.fen,
      moves: [],
      position,
      request: null,
      paused: null,
      replacements: [],
      considering: null,
    },
    access,
    watch: 'the-watch-link',
    updated: '2026-10-05T12:00:00Z',
  };
}

/** The app, opened on the game `link` reaches, as a link sent by a friend opens it. */
function renderGame(link = LINK) {
  window.location.hash = `#game/${link}`;
  return render(<App />);
}

/** Everything the app has sent to the server, as the server would read it. */
function sent(socket: FakeWebSocket): unknown[] {
  return socket.sent.map((message) => JSON.parse(message));
}

/** The state a viewer who connects after `count` moves receives. */
function stateAfter(count: number): StateEvent {
  const moves = foolsMateMoves.slice(0, count);
  return {
    ...foolsMateStart,
    game: {
      ...foolsMateStart.game,
      moves: moves.map((event) => event.move),
      position: moves.at(-1)?.position ?? foolsMateStart.game.position,
    },
  };
}

describe('App', () => {
  beforeEach(() => {
    // What the server adds to index.html when serving under the /chess prefix.
    const base = document.createElement('base');
    base.href = '/chess/';
    document.head.append(base);
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
    // Ending a game is asked about first; these tests say yes.
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    window.localStorage.clear();
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
    // The view is in the address, and the next test starts wherever this one left it.
    window.location.hash = '';
  });

  describe('the views', () => {
    /** The replay viewer asks the server for the saved games as soon as it is shown. */
    function noSavedGames() {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json([])));
    }

    it('moves between the game and the replay viewer, and says which is on show', async () => {
      noSavedGames();
      renderGame();
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));

      fireEvent.click(screen.getByRole('button', { name: 'Replay' }));

      expect(await screen.findByText('No games have been saved here yet.')).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'Replay' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Replay' })).toHaveAttribute(
        'aria-current',
        'page',
      );
      // The game view is gone, and with it the connection it was following.
      expect(screen.queryByRole('img', { name: 'white king on e1' })).not.toBeInTheDocument();
      expect(window.location.hash).toBe('#replay');

      // The Game tab is where games are started, and found again.
      fireEvent.click(screen.getByRole('button', { name: 'Game' }));

      expect(screen.getByRole('heading', { name: 'New game' })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'Your games' })).toBeInTheDocument();
      expect(window.location.hash).toBe('');
    });

    it('opens the view the address names, so that a reload stays where it was', async () => {
      noSavedGames();
      window.location.hash = '#replay';

      render(<App />);

      expect(await screen.findByText('No games have been saved here yet.')).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'Replay' })).toBeInTheDocument();
    });

    it('opens the runs dashboard, and a run from it', async () => {
      const run = {
        name: 'mlp-big',
        architecture: 'mlp',
        created: null,
        status: 'finished',
        step: 10,
        steps: 10,
        checkpoints: 1,
      };
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json([run])));
      render(<App />);

      fireEvent.click(screen.getByRole('button', { name: 'Runs' }));

      expect(window.location.hash).toBe('#runs');
      const link = await screen.findByRole('link', { name: 'mlp-big' });
      // jsdom follows no links, so the address is changed as the browser would change it,
      // and the app follows the address.
      window.location.hash = link.getAttribute('href')!;

      expect(await screen.findByRole('heading', { name: 'mlp-big' })).toBeInTheDocument();
      expect(FakeWebSocket.latest.url).toMatch(/\/chess\/api\/runs\/mlp-big\/ws$/);
      expect(screen.getByRole('button', { name: 'Runs' })).toHaveAttribute('aria-current', 'page');
    });

    it('opens the run the address names', () => {
      window.location.hash = '#runs/mlp-big';

      render(<App />);

      expect(screen.getByRole('heading', { name: 'mlp-big' })).toBeInTheDocument();
      expect(FakeWebSocket.latest.url).toMatch(/\/chess\/api\/runs\/mlp-big\/ws$/);
    });

    it('opens the comparison the address names, under the runs dashboard', () => {
      window.location.hash = '#compare/mlp-big,mlp-small';

      render(<App />);

      expect(screen.getByRole('heading', { name: 'Comparing 2 runs' })).toBeInTheDocument();
      expect(FakeWebSocket.instances.map((socket) => socket.url)).toEqual([
        expect.stringMatching(/\/chess\/api\/runs\/mlp-big\/ws$/),
        expect.stringMatching(/\/chess\/api\/runs\/mlp-small\/ws$/),
      ]);
      expect(screen.getByRole('button', { name: 'Runs' })).toHaveAttribute('aria-current', 'page');
    });
  });

  it('follows the game channel under the path prefix', () => {
    renderGame();

    const url = new URL('/chess/api/games/the-link/ws', window.location.href);
    url.protocol = 'ws:';
    expect(FakeWebSocket.latest.url).toBe(url.href);
    expect(screen.getByText('Connecting to the server…')).toBeInTheDocument();
  });

  it('plays the moves onto the board as they arrive, then shows the result', () => {
    const { container } = renderGame();
    const socket = FakeWebSocket.latest;
    socket.open();

    socket.deliver(foolsMateStart);

    expect(screen.getByRole('status')).toHaveTextContent('White to move');
    expect(lastMoveSquares(container)).toEqual([]);

    socket.deliver(foolsMateMoves[0]);

    expect(screen.getByRole('status')).toHaveTextContent('Black to move');
    expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();
    expect(lastMoveSquares(container)).toEqual(['f2', 'f3']);

    for (const move of foolsMateMoves.slice(1)) {
      socket.deliver(move);
    }

    expect(screen.getByRole('status')).toHaveTextContent('Black wins by checkmate (0–1)');
    expect(screen.getByRole('img', { name: 'black queen on h4' })).toBeInTheDocument();
    expect(lastMoveSquares(container)).toEqual(['d8', 'h4']);
  });

  it('offers the game on show as a PGN download, named, under the path prefix', () => {
    renderGame();
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver(foolsMateStart);

    // Through the link the game was opened with, like everything else about it.
    const url = new URL('/chess/api/games/the-link/pgn', window.location.href);
    expect(screen.getByRole('link', { name: 'Export PGN' })).toHaveAttribute('href', url.href);
  });

  it('names the players', () => {
    renderGame();
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({
      ...foolsMateStart,
      game: {
        ...foolsMateStart.game,
        white: { name: 'Random mover', accepts_moves: false, model: null, stockfish: null },
        black: { name: 'Someone else', accepts_moves: false, model: null, stockfish: null },
      },
    });

    const players = screen.getAllByRole('definition').map((element) => element.textContent);
    expect(players).toEqual(['Random mover', 'Someone else']);
  });

  it('says how each checkpoint of a model against model game was asked to play', () => {
    renderGame();
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({
      ...foolsMateStart,
      game: {
        ...foolsMateStart.game,
        white: {
          name: 'mlp-baseline step 12000',
          accepts_moves: false,
          model: {
            run: 'mlp-baseline',
            checkpoint: 12000,
            rating: 1600,
            strategy: 'argmax',
            temperature: null,
          },
          stockfish: null,
        },
        black: {
          name: 'mlp-baseline step 6000',
          accepts_moves: false,
          model: {
            run: 'mlp-baseline',
            checkpoint: 6000,
            rating: null,
            strategy: 'sample',
            temperature: 1.5,
          },
          stockfish: null,
        },
      },
    });

    const players = screen.getAllByRole('definition').map((element) => element.textContent);
    expect(players).toEqual([
      'mlp-baseline step 12000plays like 1600, plays its best move',
      'mlp-baseline step 6000no rating, samples at 1.5',
    ]);
  });

  it('says how strong Stockfish plays, and when that is not the strength asked for', () => {
    renderGame();
    FakeWebSocket.latest.open();
    const stockfish = { min_elo: 1320, max_elo: 3190, move_time: 1 };

    FakeWebSocket.latest.deliver({
      ...foolsMateStart,
      game: {
        ...foolsMateStart.game,
        white: {
          name: 'Stockfish 1500',
          accepts_moves: false,
          model: null,
          stockfish: { ...stockfish, elo: 1500, requested_elo: 1500 },
        },
        black: {
          name: 'Stockfish 1320',
          accepts_moves: false,
          model: null,
          stockfish: { ...stockfish, elo: 1320, requested_elo: 800, move_time: 0.25 },
        },
      },
    });

    const players = screen.getAllByRole('definition').map((element) => element.textContent);
    expect(players).toEqual([
      'Stockfish 15001 s a move',
      'Stockfish 13200.25 s a move; 800 was asked for, and Stockfish plays no weaker than 1320',
    ]);
  });

  it('reconnects after losing the connection and catches up with the game', () => {
    vi.useFakeTimers();
    renderGame();
    const first = FakeWebSocket.latest;
    first.open();
    first.deliver(foolsMateStart);
    first.deliver(foolsMateMoves[0]);

    first.disconnect();

    expect(screen.getByRole('alert')).toHaveTextContent('Reconnecting');
    expect(screen.getByRole('img', { name: 'white pawn on f3' })).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(1000));
    const second = FakeWebSocket.latest;
    expect(second).not.toBe(first);
    second.open();
    expect(screen.getByRole('alert')).toHaveTextContent('Reconnecting');
    second.deliver(stateAfter(3));

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'white pawn on g4' })).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('Black to move');

    second.deliver(foolsMateMoves[3]);

    expect(screen.getByRole('status')).toHaveTextContent('Black wins by checkmate');
  });

  it('waits longer between attempts while the server stays unreachable', () => {
    vi.useFakeTimers();
    renderGame();

    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it('keeps backing off while connections open but close before sending anything', () => {
    vi.useFakeTimers();
    renderGame();

    FakeWebSocket.latest.open();
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.latest.open();
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it('starts backing off afresh once a connection delivers the game', () => {
    vi.useFakeTimers();
    renderGame();
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(2000));
    expect(FakeWebSocket.instances).toHaveLength(3);

    FakeWebSocket.latest.open();
    FakeWebSocket.latest.deliver(foolsMateStart);
    FakeWebSocket.latest.disconnect();
    act(() => vi.advanceTimersByTime(1000));

    expect(FakeWebSocket.instances).toHaveLength(4);
  });

  it('closes the connection for good when it goes away', () => {
    vi.useFakeTimers();
    const { unmount } = renderGame();
    const socket = FakeWebSocket.latest;

    unmount();

    expect(socket.closed).toBe(true);
    act(() => vi.advanceTimersByTime(60_000));
    expect(FakeWebSocket.instances).toEqual([socket]);
  });

  describe('making a move', () => {
    it('sends the move a player makes on the board', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(sent(socket)).toEqual([{ type: 'move', uci: 'e1g1' }]);
    });

    it('sends the piece chosen for a promotion, and nothing before it is chosen', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', promotion));

      fireEvent.mouseDown(square(container, 'b7'));
      fireEvent.mouseUp(square(container, 'b8'));
      expect(sent(socket)).toEqual([]);

      fireEvent.mouseDown(promotionChoice(container, 'knight'));

      expect(sent(socket)).toEqual([{ type: 'move', uci: 'b7b8n' }]);
    });

    it('takes no second move while the first is still on its way to the server', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      fireEvent.mouseEnter(square(container, 'a1'));
      fireEvent.mouseDown(square(container, 'a1'));
      fireEvent.mouseUp(square(container, 'a1'));
      fireEvent.mouseDown(square(container, 'b1'));

      expect(destinations(container)).toEqual([]);
      expect(sent(socket)).toEqual([{ type: 'move', uci: 'e1g1' }]);
    });

    it('takes moves again once the server has answered', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      socket.deliver({ type: 'error', message: 'e1g1 is not a legal move here' });
      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toContain('g1 castling');
    });

    it('shows the rejection of a move and leaves the board as it was', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      socket.deliver({ type: 'error', message: 'e1g1 is not a legal move in ...' });

      expect(screen.getByRole('alert')).toHaveTextContent('e1g1 is not a legal move');
      expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
      expect(screen.getByRole('status')).toHaveTextContent('White to move');
    });

    it('drops the rejection once the game moves on', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      socket.deliver({ type: 'error', message: 'it is not your turn' });
      expect(screen.getByRole('alert')).toBeInTheDocument();

      socket.deliver(gameOf('human', 'random', castling));

      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('takes moves again when the server goes quiet without closing the connection', () => {
      vi.useFakeTimers();
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));
      expect(destinations(container)).toEqual([]);

      // The socket never closes, so nothing else tells the board the move is lost.
      act(() => vi.advanceTimersByTime(30_000));

      expect(screen.getByRole('alert')).toHaveTextContent('has not answered');
      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');
    });

    it('takes moves again after a dropped connection has been remade', () => {
      vi.useFakeTimers();
      const { container } = renderGame();
      const first = FakeWebSocket.latest;
      first.open();
      first.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      first.disconnect();
      act(() => vi.advanceTimersByTime(1000));
      const second = FakeWebSocket.latest;
      second.open();
      second.deliver(gameOf('human', 'random', castling));
      fireEvent.mouseEnter(square(container, 'e1'));

      expect(destinations(container)).toContain('g1 castling');
    });

    it('takes no move while a player that moves for itself is to move', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'human', castling));

      fireEvent.mouseEnter(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(destinations(container)).toEqual([]);
      expect(sent(socket)).toEqual([]);
    });

    it('takes no move once the game is over, moves or no moves', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));

      fireEvent.mouseEnter(square(container, 'e2'));
      fireEvent.mouseDown(square(container, 'e2'));
      fireEvent.mouseUp(square(container, 'e3'));

      expect(screen.getByRole('status')).toHaveTextContent('Draw by the fifty-move rule');
      expect(destinations(container)).toEqual([]);
      expect(sent(socket)).toEqual([]);
    });

    it('takes no move while the connection is down', () => {
      vi.useFakeTimers();
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      socket.disconnect();
      fireEvent.mouseEnter(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(destinations(container)).toEqual([]);
      expect(sent(socket)).toEqual([]);
    });
  });
  describe('the game view', () => {
    it('lists the moves in algebraic notation as they arrive', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(foolsMateStart);

      expect(screen.getByText('No moves yet.')).toBeInTheDocument();

      socket.deliver(foolsMateMoves[0]);
      socket.deliver(foolsMateMoves[1]);

      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual(['f3e5']);

      for (const move of foolsMateMoves.slice(2)) {
        socket.deliver(move);
      }

      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual([
        'f3e5',
        'g4Qh4#',
      ]);
    });

    it('keeps the moves a reconnecting viewer had missed', () => {
      vi.useFakeTimers();
      renderGame();
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(foolsMateStart);
      FakeWebSocket.latest.disconnect();

      act(() => vi.advanceTimersByTime(1000));
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(stateAfter(3));

      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual([
        'f3e5',
        'g4',
      ]);
    });

    it('highlights the king that is in check', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver(gameOf('human', 'random', castling));
      expect(checkSquare(container)).toBeNull();

      socket.deliver(gameOf('random', 'human', check));

      expect(checkSquare(container)).toBe('e8');
    });
  });

  describe('turning the board round', () => {
    it('starts with White at the bottom', () => {
      const { container } = renderGame();
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(gameOf('human', 'random', castling));

      expect(squareAt(container, 'a1')).toBe('0,700');
    });

    it('puts Black at the bottom when the viewer plays Black', () => {
      const { container } = renderGame();
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('random', 'human', castling));

      expect(squareAt(container, 'a1')).toBe('700,0');
      expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();
    });

    it('leaves a viewer who plays both sides behind White', () => {
      const { container } = renderGame();
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('human', 'human', castling));

      expect(squareAt(container, 'a1')).toBe('0,700');
    });

    it('turns the board round when the viewer asks', () => {
      const { container } = renderGame();
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(gameOf('human', 'random', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

      expect(squareAt(container, 'a1')).toBe('700,0');
      expect(screen.getByRole('img', { name: 'white king on e1' })).toHaveAttribute('x', '300');
    });

    it('plays a move the same way round once the board has been flipped', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(sent(socket)).toEqual([{ type: 'move', uci: 'e1g1' }]);
    });

    it('keeps the board where the viewer put it while the game goes on', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(foolsMateStart);
      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

      socket.deliver(foolsMateMoves[0]);

      expect(squareAt(container, 'a1')).toBe('700,0');
    });
  });

  describe('ending a game', () => {
    it('resigns the side the viewer plays', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'human', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

      expect(sent(socket)).toEqual([{ type: 'resign' }]);
    });

    it('aborts the game', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'random', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

      expect(sent(socket)).toEqual([{ type: 'abort' }]);
    });

    it('shows the resignation the server sends back, to players and watchers alike', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      socket.deliver({
        type: 'game_over',
        position: {
          ...castling,
          legal_moves: {},
          game_over: { result: '0-1', reason: 'resignation' },
        },
      });

      expect(screen.getByRole('status')).toHaveTextContent('Black wins by resignation (0–1)');
      // The board is left as it stood, and takes nothing further.
      expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toEqual([]);
    });

    it('shows an abort as a game with no result', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(foolsMateStart);
      socket.deliver(foolsMateMoves[0]);

      socket.deliver({
        type: 'game_over',
        position: {
          ...foolsMateMoves[0].position,
          legal_moves: {},
          game_over: { result: '*', reason: 'abort' },
        },
      });

      expect(screen.getByRole('status')).toHaveTextContent('Game aborted');
      // The moves that were played are still there to look over.
      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual(['f3']);
    });

    it('offers nothing to end once the game is over', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));

      expect(screen.queryByRole('button', { name: 'Resign' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Abort' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Flip to Black' })).toBeInTheDocument();
    });
  });

  describe('taking a move back', () => {
    it('asks for the last move back, naming the game it is looking at', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      fireEvent.click(screen.getByRole('button', { name: 'Take back' }));

      expect(sent(socket)).toEqual([{ type: 'takeback' }]);
    });

    it('puts the board and the move list back as the server says they were', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'human', startPosition as PositionSnapshot));
      for (const move of foolsMateMoves.slice(0, 3)) {
        socket.deliver(move);
      }
      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual([
        'f3e5',
        'g4',
      ]);

      // The server takes the game back to the position after the first move.
      socket.deliver({ type: 'takeback', ply: 1, position: foolsMateMoves[0].position });

      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual(['f3']);
      expect(screen.getByRole('status')).toHaveTextContent('Black to move');
      expect(lastMoveSquares(container)).toEqual(['f2', 'f3']);
      expect(screen.queryByRole('img', { name: 'white pawn on g4' })).not.toBeInTheDocument();
    });

    it('plays on from the position a takeback went back to', () => {
      const { container } = renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      socket.deliver({ type: 'takeback', ply: 0, position: startPosition as PositionSnapshot });

      // White is on move again, and the board takes a move for White.
      fireEvent.mouseEnter(square(container, 'e2'));
      expect(destinations(container)).toContain('e4 quiet');
      fireEvent.mouseDown(square(container, 'e2'));
      fireEvent.mouseUp(square(container, 'e2'));
      fireEvent.mouseDown(square(container, 'e4'));
      expect(sent(socket).at(-1)).toEqual({ type: 'move', uci: 'e2e4' });
    });

    it('offers no takeback in a game nobody plays by hand', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      expect(screen.queryByRole('button', { name: 'Take back' })).not.toBeInTheDocument();
    });

    it('has nothing to take back before the first move, or once the game is over', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      expect(screen.getByRole('button', { name: 'Take back' })).toBeDisabled();

      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));
      expect(screen.queryByRole('button', { name: 'Take back' })).not.toBeInTheDocument();
    });

    it('shows the server refusing a takeback, and leaves the game alone', () => {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      socket.deliver({ type: 'error', message: 'no move has been played yet' });

      expect(screen.getByRole('alert')).toHaveTextContent('no move has been played yet');
      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual(['f3']);
    });
  });

  describe('links', () => {
    it('follows a watch link, offers nothing to do but watch, and says so', () => {
      window.location.hash = '#watch/a-watch-link';
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver(gameOf('human', 'random', castling, 'watch'));

      const url = new URL('/chess/api/games/a-watch-link/ws', window.location.href);
      url.protocol = 'ws:';
      expect(socket.url).toBe(url.href);
      expect(screen.getByText('You are watching this game.')).toBeInTheDocument();
      for (const name of ['Take back', 'Resign', 'Abort']) {
        expect(screen.queryByRole('button', { name })).not.toBeInTheDocument();
      }
      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toEqual([]);
    });

    it('puts Black at the bottom for the link that plays Black, in a game between two', () => {
      const { container } = renderGame();
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('human', 'human', castling, 'black'));

      expect(squareAt(container, 'a1')).toBe('700,0');
    });

    it('takes no move for the other person’s side', () => {
      const { container } = renderGame();
      FakeWebSocket.latest.open();

      // White is on move, and this link plays Black.
      FakeWebSocket.latest.deliver(gameOf('human', 'human', castling, 'black'));

      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toEqual([]);
    });

    it('says when a link reaches no game, and stops asking', () => {
      vi.useFakeTimers();
      window.localStorage.setItem(
        'chess-ai.games',
        JSON.stringify([{ link: LINK, access: 'white', added: '2026-10-01T00:00:00Z' }]),
      );
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver({
        type: 'error',
        message: 'there is no such game: the link is wrong',
      });
      socket.disconnect(1008);
      act(() => vi.advanceTimersByTime(60_000));

      expect(screen.getByRole('alert')).toHaveTextContent('there is no such game');
      expect(screen.getByRole('link', { name: 'Start a new game' })).toBeInTheDocument();
      expect(FakeWebSocket.instances).toEqual([socket]);
      expect(window.localStorage.getItem('chess-ai.games')).toBe('[]');
    });

    it('lets go of a game that has ended without trying to reconnect', () => {
      vi.useFakeTimers();
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));

      socket.disconnect(1000);
      act(() => vi.advanceTimersByTime(60_000));

      expect(FakeWebSocket.instances).toEqual([socket]);
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
      expect(screen.getByRole('status')).toHaveTextContent('Draw by the fifty-move rule');
    });

    it('shows the link to keep and the watch link to pass on', () => {
      renderGame();
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('human', 'random', castling));

      expect(screen.getByLabelText('Your link, playing White')).toHaveValue(
        new URL('#game/the-link', window.location.href).href,
      );
      expect(screen.getByLabelText('Watch link')).toHaveValue(
        new URL('#watch/the-watch-link', window.location.href).href,
      );
      expect(screen.queryByLabelText("Black's link")).not.toBeInTheDocument();
    });

    it('shows the other player’s link in the browser that started the game', () => {
      window.localStorage.setItem(
        'chess-ai.games',
        JSON.stringify([
          {
            link: LINK,
            access: 'white',
            added: '2026-10-01T00:00:00Z',
            links: {
              white: LINK,
              black: 'blacks-link',
              watch: 'the-watch-link',
              control: null,
            },
          },
        ]),
      );
      renderGame();
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('human', 'human', castling));

      expect(screen.getByLabelText("Black's link")).toHaveValue(
        new URL('#game/blacks-link', window.location.href).href,
      );
    });

    it('remembers a game it has opened, for the Game tab to list', () => {
      renderGame();
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('random', 'human', castling));

      const kept = JSON.parse(window.localStorage.getItem('chess-ai.games') ?? '[]');
      expect(kept).toEqual([expect.objectContaining({ link: LINK, access: 'black' })]);
    });
  });

  describe('asking the other person', () => {
    function twoPeopleAfter(moves: number, access: Access = 'white') {
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'human', startPosition as PositionSnapshot, access));
      for (const move of foolsMateMoves.slice(0, moves)) {
        socket.deliver(move);
      }
      return socket;
    }

    it('asks for a takeback and an abort rather than making them', () => {
      twoPeopleAfter(2);

      expect(screen.getByRole('button', { name: 'Ask to take back' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Ask to abort' })).toBeEnabled();
    });

    it('aborts outright until both have moved', () => {
      const socket = twoPeopleAfter(1);

      fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

      expect(sent(socket)).toEqual([{ type: 'abort' }]);
    });

    it('shows the one asked what was asked, and sends their answer', () => {
      const socket = twoPeopleAfter(1, 'black');

      socket.deliver({
        type: 'request',
        request: { id: 1, kind: 'takeback', by: 'white' },
      });
      expect(screen.getByRole('group', { name: 'Request' })).toHaveTextContent(
        'White asks to take back their last move.',
      );
      fireEvent.click(screen.getByRole('button', { name: 'Agree' }));

      expect(sent(socket)).toEqual([{ type: 'answer', request: 1, accept: true }]);
    });

    it('tells the one who asked to wait for the answer, and asks nothing more meanwhile', () => {
      const socket = twoPeopleAfter(2);

      socket.deliver({
        type: 'request',
        request: { id: 1, kind: 'abort', by: 'white' },
      });

      expect(screen.getByText(/You asked to abort the game/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Ask to take back' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Ask to abort' })).toBeDisabled();
    });

    it('drops a request once the game moves on', () => {
      const socket = twoPeopleAfter(1, 'black');
      socket.deliver({
        type: 'request',
        request: { id: 1, kind: 'takeback', by: 'white' },
      });

      socket.deliver(foolsMateMoves[1]);

      expect(screen.queryByRole('group', { name: 'Request' })).not.toBeInTheDocument();
    });

    it('drops a request that was declined', () => {
      const socket = twoPeopleAfter(1, 'black');
      socket.deliver({
        type: 'request',
        request: { id: 1, kind: 'takeback', by: 'white' },
      });

      socket.deliver({ type: 'request', request: null });

      expect(screen.queryByRole('group', { name: 'Request' })).not.toBeInTheDocument();
    });
  });

  describe('confirming the end of a game', () => {
    it('resigns nothing that was not confirmed', () => {
      vi.mocked(window.confirm).mockReturnValue(false);
      renderGame();
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Resign' }));
      fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

      expect(window.confirm).toHaveBeenCalledTimes(2);
      expect(sent(socket)).toEqual([]);
    });
  });

  describe('the Game tab', () => {
    const seen = (link: string, moves: number) => ({
      ...gameOf('human', 'random', castling),
      game: {
        ...gameOf('human', 'random', castling).game,
        moves: foolsMateMoves.slice(0, moves).map((event) => event.move),
      },
      updated: new Date(Date.now() - 3 * 3600_000).toISOString(),
      link,
    });

    it('lists the games the server still has, and forgets the rest', async () => {
      window.localStorage.setItem(
        'chess-ai.games',
        JSON.stringify([
          { link: 'kept', access: 'white', added: '2026-10-01T00:00:00Z' },
          { link: 'gone', access: 'black', added: '2026-10-01T00:00:00Z' },
        ]),
      );
      vi.stubGlobal(
        'fetch',
        vi.fn((url: string) =>
          Promise.resolve(
            url.endsWith('/api/games/kept')
              ? Response.json(seen('kept', 2))
              : Response.json({ detail: 'there is no such game' }, { status: 404 }),
          ),
        ),
      );

      render(<App />);

      const listed = await screen.findByRole('link', {
        name: /Human – Random mover/,
      });
      expect(listed).toHaveAttribute('href', '#game/kept');
      expect(listed).toHaveTextContent('you play White');
      expect(listed).toHaveTextContent('2 moves');
      expect(listed).toHaveTextContent('3 h ago');
      const kept = JSON.parse(window.localStorage.getItem('chess-ai.games') ?? '[]');
      expect(kept.map((game: { link: string }) => game.link)).toEqual(['kept']);
    });

    it('goes to the side it plays once a game has started, keeping every link to it', async () => {
      const links = {
        white: 'whites-link',
        black: 'blacks-link',
        watch: 'watching',
        control: null,
      };
      vi.stubGlobal(
        'fetch',
        vi.fn(() =>
          Promise.resolve(
            Response.json({
              game: gameOf('human', 'human', castling).game,
              links,
            }),
          ),
        ),
      );
      render(<App />);

      fireEvent.click(screen.getByRole('button', { name: 'Start' }));

      await vi.waitFor(() => expect(window.location.hash).toBe('#game/whites-link'));
      const kept = JSON.parse(window.localStorage.getItem('chess-ai.games') ?? '[]');
      expect(kept).toEqual([expect.objectContaining({ link: 'whites-link', links })]);
    });
  });

  describe('playing again', () => {
    it('starts the same game again and goes to the same side of it', async () => {
      const links = { white: 'new-white', black: 'new-black', watch: 'new-watch', control: null };
      const fetch = vi.fn(() =>
        Promise.resolve(Response.json({ game: gameOf('human', 'human', castling).game, links })),
      );
      vi.stubGlobal('fetch', fetch);
      renderGame();
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(gameOf('human', 'human', drawnByFiftyMoves, 'black'));

      fireEvent.click(screen.getByRole('button', { name: 'Play again' }));

      await vi.waitFor(() => expect(window.location.hash).toBe('#game/new-black'));
      const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toBe(new URL('/chess/api/games/the-link/rematch', window.location.href).href);
      expect(init.method).toBe('POST');
      // The other player's new link is here to send them, as for any game started here.
      const kept = JSON.parse(window.localStorage.getItem('chess-ai.games') ?? '[]');
      expect(kept[0]).toEqual(expect.objectContaining({ link: 'new-black', links }));
    });
  });
});
