import { act } from '@testing-library/react';
import type { GameEvent } from '../api';

/** Stands in for the browser's WebSocket; tests drive the server's side of it. */
export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  onopen: (() => void) | null = null;
  onmessage: ((message: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  /** The socket the app opened most recently. */
  static get latest(): FakeWebSocket {
    const socket = FakeWebSocket.instances.at(-1);
    if (socket === undefined) {
      throw new Error('no WebSocket has been opened');
    }
    return socket;
  }

  close() {
    this.disconnect();
  }

  /** The server accepts the connection. */
  open() {
    act(() => this.onopen?.());
  }

  /** The server sends an event. */
  send(event: GameEvent) {
    act(() => this.onmessage?.({ data: JSON.stringify(event) }));
  }

  /** The connection drops. */
  disconnect() {
    if (!this.closed) {
      this.closed = true;
      act(() => this.onclose?.());
    }
  }
}
