import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PlayerKind, PositionSnapshot } from './api';
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

const PLAYERS = {
  human: { name: 'Human', accepts_moves: true },
  random: { name: 'Random mover', accepts_moves: false },
} satisfies Record<PlayerKind, { name: string; accepts_moves: boolean }>;

const GAME = 'the-game';

/** A game at the given position, between the given kinds of player. */
function gameOf(
  white: PlayerKind,
  black: PlayerKind,
  position: PositionSnapshot,
  id = GAME,
): StateEvent {
  return {
    type: 'state',
    game: {
      id,
      white: PLAYERS[white],
      black: PLAYERS[black],
      start_fen: position.fen,
      moves: [],
      position,
    },
  };
}

/** Everything the app has sent to the server, as the server would read it. */
function sent(socket: FakeWebSocket): unknown[] {
  return socket.sent.map((message) => JSON.parse(message));
}

/** The state a viewer who connects after `count` moves receives. */
function stateAfter(count: number): StateEvent {
  const moves = foolsMateMoves.slice(0, count);
  return {
    type: 'state',
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
  });

  afterEach(() => {
    document.head.querySelector('base')?.remove();
    vi.unstubAllGlobals();
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
      render(<App />);
      FakeWebSocket.latest.open();
      const atTheStart = { type: 'no_game', position: startPosition as PositionSnapshot } as const;
      FakeWebSocket.latest.deliver(atTheStart);

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

      fireEvent.click(screen.getByRole('button', { name: 'Game' }));
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(atTheStart);

      expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
      expect(window.location.hash).toBe('');
    });

    it('opens the view the address names, so that a reload stays where it was', async () => {
      noSavedGames();
      window.location.hash = '#replay';

      render(<App />);

      expect(await screen.findByText('No games have been saved here yet.')).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: 'Replay' })).toBeInTheDocument();
    });
  });

  it('follows the game channel under the path prefix', () => {
    render(<App />);

    const url = new URL('/chess/api/game/ws', window.location.href);
    url.protocol = 'ws:';
    expect(FakeWebSocket.latest.url).toBe(url.href);
    expect(screen.getByText('Connecting to the server…')).toBeInTheDocument();
  });

  it('shows the starting position before any game', () => {
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({ type: 'no_game', position: startPosition as PositionSnapshot });

    expect(screen.getByRole('status')).toHaveTextContent('No game in progress');
    expect(screen.getByRole('img', { name: 'white king on e1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start' })).toBeInTheDocument();
  });

  it('plays the moves onto the board as they arrive, then shows the result', () => {
    const { container } = render(<App />);
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
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver(foolsMateStart);

    // Named like every other message about the game, so that a game started in the
    // meantime is refused rather than downloaded in place of the one on the screen.
    const url = new URL('/chess/api/game/pgn?game=fools-mate', window.location.href);
    expect(screen.getByRole('link', { name: 'Export PGN' })).toHaveAttribute('href', url.href);
  });

  it('names the game that replaced the one before it in the download', () => {
    render(<App />);
    const socket = FakeWebSocket.latest;
    socket.open();
    socket.deliver(foolsMateStart);

    socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot, 'the-next-game'));

    const url = new URL('/chess/api/game/pgn?game=the-next-game', window.location.href);
    expect(screen.getByRole('link', { name: 'Export PGN' })).toHaveAttribute('href', url.href);
  });

  it('has no game to download before one has been started', () => {
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({ type: 'no_game', position: startPosition as PositionSnapshot });

    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('names the players', () => {
    render(<App />);
    FakeWebSocket.latest.open();

    FakeWebSocket.latest.deliver({
      type: 'state',
      game: {
        ...foolsMateStart.game,
        white: { name: 'Random mover', accepts_moves: false },
        black: { name: 'Someone else', accepts_moves: false },
      },
    });

    const players = screen.getAllByRole('definition').map((element) => element.textContent);
    expect(players).toEqual(['Random mover', 'Someone else']);
  });

  it('reconnects after losing the connection and catches up with the game', () => {
    vi.useFakeTimers();
    render(<App />);
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
    render(<App />);

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
    render(<App />);

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
    render(<App />);
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
    const { unmount } = render(<App />);
    const socket = FakeWebSocket.latest;

    unmount();

    expect(socket.closed).toBe(true);
    act(() => vi.advanceTimersByTime(60_000));
    expect(FakeWebSocket.instances).toEqual([socket]);
  });

  describe('making a move', () => {
    it('sends the move a player makes on the board', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));

      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(sent(socket)).toEqual([{ game: GAME, type: 'move', uci: 'e1g1' }]);
    });

    it('sends the piece chosen for a promotion, and nothing before it is chosen', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', promotion));

      fireEvent.mouseDown(square(container, 'b7'));
      fireEvent.mouseUp(square(container, 'b8'));
      expect(sent(socket)).toEqual([]);

      fireEvent.mouseDown(promotionChoice(container, 'knight'));

      expect(sent(socket)).toEqual([{ game: GAME, type: 'move', uci: 'b7b8n' }]);
    });

    it('takes no second move while the first is still on its way to the server', () => {
      const { container } = render(<App />);
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
      expect(sent(socket)).toEqual([{ game: GAME, type: 'move', uci: 'e1g1' }]);
    });

    it('takes moves again once the server has answered', () => {
      const { container } = render(<App />);
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
      const { container } = render(<App />);
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
      render(<App />);
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
      const { container } = render(<App />);
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
      const { container } = render(<App />);
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
      const { container } = render(<App />);
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
      const { container } = render(<App />);
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
      const { container } = render(<App />);
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
      render(<App />);
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
      render(<App />);
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
      const { container } = render(<App />);
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
      const { container } = render(<App />);
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(gameOf('human', 'random', castling));

      expect(squareAt(container, 'a1')).toBe('0,700');
    });

    it('puts Black at the bottom when the viewer plays Black', () => {
      const { container } = render(<App />);
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('random', 'human', castling));

      expect(squareAt(container, 'a1')).toBe('700,0');
      expect(screen.getByRole('button', { name: 'Flip to White' })).toBeInTheDocument();
    });

    it('leaves a viewer who plays both sides behind White', () => {
      const { container } = render(<App />);
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver(gameOf('human', 'human', castling));

      expect(squareAt(container, 'a1')).toBe('0,700');
    });

    it('turns the board round when the viewer asks', () => {
      const { container } = render(<App />);
      FakeWebSocket.latest.open();
      FakeWebSocket.latest.deliver(gameOf('human', 'random', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

      expect(squareAt(container, 'a1')).toBe('700,0');
      expect(screen.getByRole('img', { name: 'white king on e1' })).toHaveAttribute('x', '300');
    });

    it('plays a move the same way round once the board has been flipped', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));

      fireEvent.mouseEnter(square(container, 'e1'));
      expect(destinations(container)).toContain('g1 castling');

      fireEvent.mouseDown(square(container, 'e1'));
      fireEvent.mouseUp(square(container, 'e1'));
      fireEvent.mouseDown(square(container, 'g1'));

      expect(sent(socket)).toEqual([{ game: GAME, type: 'move', uci: 'e1g1' }]);
    });

    it('turns the board back to face the side a new game gives the viewer', () => {
      const { container } = render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling));
      fireEvent.click(screen.getByRole('button', { name: 'Flip to Black' }));
      expect(squareAt(container, 'a1')).toBe('700,0');

      socket.deliver(gameOf('random', 'human', castling));
      expect(squareAt(container, 'a1')).toBe('700,0');

      socket.deliver(gameOf('human', 'random', castling));

      expect(squareAt(container, 'a1')).toBe('0,700');
    });

    it('keeps the board where the viewer put it while the game goes on', () => {
      const { container } = render(<App />);
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
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'human', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

      expect(sent(socket)).toEqual([{ game: GAME, type: 'resign', color: 'black' }]);
    });

    it('aborts the game', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'random', castling));

      fireEvent.click(screen.getByRole('button', { name: 'Abort' }));

      expect(sent(socket)).toEqual([{ game: GAME, type: 'abort' }]);
    });

    it('names the game on screen, so a click cannot land on the one that replaced it', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', castling, 'the-first-game'));

      // Another viewer starts a new game, which the server sends to everyone.
      socket.deliver(gameOf('human', 'random', castling, 'the-second-game'));
      fireEvent.click(screen.getByRole('button', { name: 'Resign' }));

      expect(sent(socket)).toEqual([
        { game: 'the-second-game', type: 'resign', color: 'white' },
      ]);
    });

    it('shows the resignation the server sends back, to players and watchers alike', () => {
      const { container } = render(<App />);
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
      render(<App />);
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
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));

      expect(screen.queryByRole('button', { name: 'Resign' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Abort' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Flip to Black' })).toBeInTheDocument();
    });

    it('offers nothing to end before any game has started', () => {
      render(<App />);
      FakeWebSocket.latest.open();

      FakeWebSocket.latest.deliver({
        type: 'no_game',
        position: startPosition as PositionSnapshot,
      });

      expect(screen.queryByRole('button', { name: 'Abort' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Flip to Black' })).toBeInTheDocument();
    });
  });

  describe('taking a move back', () => {
    it('asks for the last move back, naming the game it is looking at', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      fireEvent.click(screen.getByRole('button', { name: 'Take back' }));

      expect(sent(socket)).toEqual([{ game: GAME, type: 'takeback' }]);
    });

    it('puts the board and the move list back as the server says they were', () => {
      const { container } = render(<App />);
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
      const { container } = render(<App />);
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
      expect(sent(socket).at(-1)).toEqual({ game: GAME, type: 'move', uci: 'e2e4' });
    });

    it('offers no takeback in a game nobody plays by hand', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('random', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      expect(screen.queryByRole('button', { name: 'Take back' })).not.toBeInTheDocument();
    });

    it('has nothing to take back before the first move, or once the game is over', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();

      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      expect(screen.getByRole('button', { name: 'Take back' })).toBeDisabled();

      socket.deliver(gameOf('human', 'random', drawnByFiftyMoves));
      expect(screen.queryByRole('button', { name: 'Take back' })).not.toBeInTheDocument();
    });

    it('shows the server refusing a takeback, and leaves the game alone', () => {
      render(<App />);
      const socket = FakeWebSocket.latest;
      socket.open();
      socket.deliver(gameOf('human', 'random', startPosition as PositionSnapshot));
      socket.deliver(foolsMateMoves[0]);

      socket.deliver({ type: 'error', message: 'no move has been played yet' });

      expect(screen.getByRole('alert')).toHaveTextContent('no move has been played yet');
      expect(screen.getAllByRole('listitem').map((item) => item.textContent)).toEqual(['f3']);
    });
  });
});
