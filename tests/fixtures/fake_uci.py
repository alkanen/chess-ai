"""A stand-in for Stockfish: just enough UCI to be played against, for tests without the engine.

It plays the first legal move in UCI order once its move time is up, or at once when told to
stop, and it offers the strength options Stockfish does. What it is told is appended to the
file given by ``--log``, so a test can see which strength it was asked for.

    python fake_uci.py [--log FILE] [--elo-range MIN MAX | --no-elo] [--mute]

``--no-elo`` leaves out the strength options, as an engine that is not Stockfish might, and
``--mute`` never answers at all, as a binary that is not a UCI engine would not.
"""

import argparse
import sys
import threading

import chess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--elo-range", nargs=2, type=int, default=[1320, 3190])
    parser.add_argument("--no-elo", action="store_true")
    parser.add_argument("--mute", action="store_true")
    args = parser.parse_args()
    log = open(args.log, "a", buffering=1) if args.log else None  # noqa: SIM115
    board = chess.Board()
    stop = threading.Event()
    thinking: threading.Thread | None = None

    def say(line: str) -> None:
        print(line, flush=True)

    def think(position: chess.Board, seconds: float) -> None:
        stop.wait(seconds)
        say(f"bestmove {min(move.uci() for move in position.legal_moves)}")

    for line in sys.stdin:
        if log is not None:
            log.write(line)
        words = line.split()
        if not words or args.mute:
            continue
        command = words[0]
        if command == "uci":
            say("id name Fake UCI")
            if not args.no_elo:
                low, high = args.elo_range
                say("option name UCI_LimitStrength type check default false")
                say(f"option name UCI_Elo type spin default {low} min {low} max {high}")
            say("uciok")
        elif command == "isready":
            say("readyok")
        elif command == "position":
            board = _position(words[1:])
        elif command == "go":
            seconds = int(words[words.index("movetime") + 1]) / 1000 if "movetime" in words else 0
            stop.clear()
            thinking = threading.Thread(target=think, args=(board.copy(), seconds))
            thinking.start()
        elif command == "stop":
            stop.set()
            if thinking is not None:
                thinking.join()
        elif command == "quit":
            break


def _position(words: list[str]) -> chess.Board:
    if words[0] == "startpos":
        board, rest = chess.Board(), words[1:]
    else:
        end = words.index("moves") if "moves" in words else len(words)
        board, rest = chess.Board(" ".join(words[1:end])), words[end:]
    for uci in rest[1:]:
        board.push_uci(uci)
    return board


if __name__ == "__main__":
    main()
