import { act } from '@testing-library/react';
import type { GameEvent } from '../api';

/** Stands in for the browser's WebSocket; tests drive the server's side of it. */
export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  onopen: (() => void) | null = null;
  onmessage: ((message: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  readyState: number = FakeWebSocket.CONNECTING;

  /** Every message the app has sent to the server, as it was serialized. */
  readonly sent: string[] = [];

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

  /**
   * The app sends a message. A socket that is not open discards it and says nothing,
   * as the WebSocket specification requires, so nothing reaches the server.
   */
  send(data: string) {
    if (this.readyState === FakeWebSocket.OPEN) {
      this.sent.push(data);
    }
  }

  /** The server accepts the connection. */
  open() {
    this.readyState = FakeWebSocket.OPEN;
    act(() => this.onopen?.());
  }

  /** The server sends an event. */
  deliver(event: GameEvent) {
    act(() => this.onmessage?.({ data: JSON.stringify(event) }));
  }

  /** The connection drops. */
  disconnect() {
    if (!this.closed) {
      this.closed = true;
      this.readyState = FakeWebSocket.CLOSED;
      act(() => this.onclose?.());
    }
  }
}
