# chess-ai

A testbed for training neural-network chess players the way large language models are trained: show the network a position, have it predict the move a human actually played, and repeat over millions of games. The goal is to compare model architectures on equal terms (MLP, ResNet, a transformer over the 64 squares, and a GPT-style model over move sequences) and to watch them learn through a browser UI.

> **Status: early development.** The web server and the board are in place, and the server can play live games between random movers; the data pipeline, models and training come next. The full design is in the PRD: [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md).

## Planned features

- **Data pipeline.** Import any PGN files (chess.com exports, Lichess monthly database dumps including `.pgn.zst`, your own games) into compact, encoder-independent datasets. Filters on rating, time control and termination support a "pretrain on everything, fine-tune on masters" curriculum.
- **Pluggable models.** Every architecture gets the same structured input (spatial board planes, a separate vector of global features such as side to move and player ratings, and optionally the move sequence). Every architecture predicts the same outputs: a policy over one fixed move vocabulary and a win/draw/loss estimate. Adding a new architecture means implementing one interface.
- **Trainer.** Each experiment is described by one config file and uses mixed precision on a single NVIDIA GPU. Runs can be resumed, fine-tuned from another run's checkpoint, and started or stopped from the command line or the browser.
- **Evaluator.** A background worker evaluates new checkpoints without pausing training. It reports held-out move accuracy, plays sample games, records the model's predictions on probe positions, estimates Elo against a ladder of strength-limited Stockfish levels, measures Lichess puzzle accuracy, and runs tournaments between checkpoints and architectures.
- **Web UI.** A browser app that works from anywhere, including behind a reverse proxy under a URL path prefix. From it you can:
  - watch games live, with legal moves highlighted on hover
  - play against any checkpoint at a chosen "play like rating X", with an optional overlay of the model's candidate moves
  - follow training through live charts, run status and GPU statistics
  - browse evaluation results

## Architecture at a glance

```
            PGN files / Lichess dumps
                       │
                       ▼
              dataset builder ──► datasets/
                                      │
                                      ▼
   experiment config ──────────►  trainer  ──┐
                                             ▼
                                        runs/<run>/   ◄──── evaluator worker
                                  (config, metrics,        (sample games, probes,
                                   status, checkpoints,     Stockfish ladder,
                                   eval results, games)     puzzles, tournaments)
                                             │
                                             ▼
                          web server (FastAPI, REST + WebSocket)
                                             │
                                             ▼
                               browser (React + SVG board)
```

The trainer, the evaluator and the web server are independent processes that share run directories on disk. Restarting the web server never interrupts training.

## Planned tech stack

| Area | Choice |
|---|---|
| Language and packaging | Python 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Machine learning | PyTorch with CUDA (developed on an RTX 4090 under WSL2) |
| Chess rules, PGN and UCI | [python-chess](https://python-chess.readthedocs.io/) |
| Web server | FastAPI + uvicorn |
| Frontend | React + TypeScript, built with Vite; custom SVG board |
| Reference opponent | [Stockfish](https://stockfishchess.org/) (installed separately) |
| Data | [Lichess open database](https://database.lichess.org/) (games and puzzles, CC0) |

Running the system needs only Python and the built frontend. Node.js is only needed to build the frontend.

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.12 for the project by itself; the system Python is not used.
- [Node.js](https://nodejs.org/) 22.12 or later with npm, only to build the frontend.
- For training (not needed yet, and never needed to serve the UI): an NVIDIA GPU with a recent driver. PyTorch's wheels bring their own CUDA runtime, so no CUDA toolkit is needed. Under WSL2, install the NVIDIA driver on the Windows side only, never inside WSL; `nvidia-smi` in WSL should then list the GPU.

### Install

```sh
uv sync
```

This creates `.venv/` and installs the `chess-ai` command into it. Run it with `uv run chess-ai …`.

### Build the frontend

```sh
scripts/build-frontend.sh
```

This installs the frontend's npm packages and builds the React app into `src/chess_ai/web/static/`, where the server finds it. Run it again after changing or updating the frontend. The build uses relative URLs only, so the same build works under any path prefix.

### Configure

Settings are read from `chess-ai.toml` in the current directory, or from the file named by `--config` or the `CHESS_AI_CONFIG` environment variable. Without a file, the defaults apply. [chess-ai.example.toml](chess-ai.example.toml) lists every setting with its default:

```sh
cp chess-ai.example.toml chess-ai.toml
```

Every setting can also be overridden by an environment variable named `CHESS_AI_<SECTION>_<KEY>`:

| Setting | Environment variable | Default |
|---|---|---|
| `[server] host` | `CHESS_AI_SERVER_HOST` | `127.0.0.1` |
| `[server] port` | `CHESS_AI_SERVER_PORT` | `8000` |
| `[server] path_prefix` | `CHESS_AI_SERVER_PATH_PREFIX` | empty (serve at `/`) |

### Serve

```sh
uv run chess-ai serve
```

Then open the URL it prints, for example `http://127.0.0.1:8000/chess/` with `path_prefix = "/chess"`. The page, its assets, the API (`…/api/`, with interactive docs at `…/api/docs`) and the WebSocket that streams the game (`…/api/game/ws`) are all served under the prefix.

The server holds one game, which every open browser shows. Start a game between two random movers from the page, with a delay between moves so it can be followed; starting another game replaces it for everyone.

### Shortcuts with make

With `make` installed, one command builds the frontend if it is out of date and then serves the app:

```sh
make
```

`make build` only builds, `make check` runs the tests and the linter, and `make clean` removes the built frontend and the installed npm packages.

## Development

Run the tests and the linter (`make check` runs all three):

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
npm --prefix frontend test
```

For frontend work with hot reload, run the server with an empty prefix and the Vite dev server next to it. Vite forwards `/api` requests and WebSockets to the server:

```sh
CHESS_AI_SERVER_PATH_PREFIX= uv run chess-ai serve
npm --prefix frontend run dev
```

## Deployment notes

The web app serves everything (pages, assets, API and WebSockets) under the configured path prefix, so nginx can forward requests unchanged: set `path_prefix = "/chess"` and pass the path through as is, without a URI part in `proxy_pass`. The WebSocket upgrade headers must be forwarded for the live features to work. For example:

```nginx
location /chess/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 1h;
}
```

If nginx runs on another machine, or on the Windows side of a WSL2 setup, set `host = "0.0.0.0"` so the server accepts connections from outside, and point `proxy_pass` at an address nginx can reach.

The app has no authentication. It will have a read-only mode that disables starting and stopping jobs from the browser.

## License

GPL-3.0, matching the python-chess dependency.

The chess pieces are Colin M.L. Burnett's "cburnett" set, used under the GPL (version 2 or later); see [frontend/src/board/pieces/cburnett/LICENSE.md](frontend/src/board/pieces/cburnett/LICENSE.md).
