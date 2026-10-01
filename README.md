# chess-ai

A testbed for training neural-network chess players the way large language models are trained: show the network a position, have it predict the move a human actually played, and repeat over millions of games. The goal is to compare model architectures on equal terms (MLP, ResNet, a transformer over the 64 squares, and a GPT-style model over move sequences) and to watch them learn through a browser UI.

> **Status: early development.** The web server and the board are in place, the server plays live games between random movers and human players with legal moves shown on hover, the CLI builds training datasets out of PGN files, it trains an MLP or a residual CNN on them from an experiment config file, and a runs dashboard follows training live in the browser; the transformers and the evaluator come next. The full design is in the PRD: [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md).

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

## Tech stack

| Area | Choice |
|---|---|
| Language and packaging | Python 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Machine learning | PyTorch with CUDA, bfloat16 (developed on an RTX 4090 under WSL2) |
| Chess rules, PGN and UCI | [python-chess](https://python-chess.readthedocs.io/) |
| Web server | FastAPI + uvicorn |
| Frontend | React + TypeScript, built with Vite; custom SVG board; [uPlot](https://github.com/leeoniya/uPlot) for charts |
| Reference opponent | [Stockfish](https://stockfishchess.org/) (installed separately) |
| Data | [Lichess open database](https://database.lichess.org/) (games and puzzles, CC0) |

Running the system needs only Python and the built frontend. Node.js is only needed to build the frontend.

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.12 for the project by itself; the system Python is not used.
- [Node.js](https://nodejs.org/) 22.12 or later with npm, only to build the frontend.
- For training, an NVIDIA GPU with a recent driver. Not needed to serve the UI, and not needed to train either — training falls back to the CPU, slowly. PyTorch's wheels bring their own CUDA runtime, so no CUDA toolkit is needed. Under WSL2, install the NVIDIA driver on the Windows side only, never inside WSL; `nvidia-smi` in WSL should then list the GPU.

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
| `[server] stale_after_seconds` | `CHESS_AI_SERVER_STALE_AFTER_SECONDS` | `120` |
| `[paths] games` | `CHESS_AI_PATHS_GAMES` | `games`, in the working directory |
| `[paths] data` | `CHESS_AI_PATHS_DATA` | `data`, in the working directory |
| `[paths] runs` | `CHESS_AI_PATHS_RUNS` | `runs`, in the working directory |
| `[inference] device` | `CHESS_AI_INFERENCE_DEVICE` | `cpu` |
| `[inference] batch_size` | `CHESS_AI_INFERENCE_BATCH_SIZE` | `32` |

### Serve

```sh
uv run chess-ai serve
```

Then open the URL it prints, for example `http://127.0.0.1:8000/chess/` with `path_prefix = "/chess"`. The page, its assets, the API (`…/api/`, with interactive docs at `…/api/docs`) and the WebSocket that streams the game (`…/api/game/ws`) are all served under the prefix.

The server holds one game, which every open browser shows. Start a game from the page, choosing a human player, a random mover or a model for each colour, with a delay between moves so that a game between players that move instantly can be followed; starting another game replaces it for everyone.

### Follow training

**Runs** lists every training run with its state, architecture, dataset, progress and latest losses, top-1 accuracy and illegal-move rate, refreshed every few seconds. Opening a run shows its step, epoch, throughput, time left and GPU statistics, and charts its loss (on a log scale), validation accuracy, learning rate and illegal-move rate against the step. The charts follow the run over a WebSocket (`…/api/runs/<name>/ws`) as the trainer logs, so they update without reloading; a finished run shows the same charts for its whole history. Drag across a chart to zoom in, and double-click to let it follow the run again. Every chart can be drawn against the step, the positions seen (which lines up runs with different batch sizes) or the wall time; the choice is remembered in the browser.

Tick two or more runs in the list (up to eight) and choose **Compare** to overlay them on the same charts, one line per run: training and validation loss, top-1 and top-5 accuracy, illegal-move rate and learning rate, all followed live.

A run can be given a title, tags and notes, from its page in the browser or from the command line. They are kept in `notes.json` in the run directory, and the run list can be filtered by tag. The run's name never changes, since it is the directory; the title is what it is shown as instead.

```sh
uv run chess-ai runs annotate mlp-baseline --title "MLP baseline" --tag mlp --tag baseline --notes "First run on the 2024 games."
uv run chess-ai runs annotate mlp-baseline --untag baseline --notes-file notes.md   # - reads standard input
uv run chess-ai runs annotate mlp-baseline     # shows what it has
uv run chess-ai runs list --tag mlp
```

The web server never talks to a trainer: it reads the run directory, so it can be restarted at any time without affecting a run. A trainer that dies without saying so — killed, out of memory, or the machine gone — leaves a heartbeat that claims it is still running. Once that heartbeat is older than `[server] stale_after_seconds` (two minutes by default), the run is flagged **stale**. A trainer rewrites its heartbeat every couple of seconds, but not while it starts up, validates or saves a checkpoint, so keep the threshold well above those.

The charts are drawn with [uPlot](https://github.com/leeoniya/uPlot): it is about 50 kB, draws on a canvas, and redraws the tens of thousands of points a long run logs without slowing the page, which SVG charting libraries struggle with.

### Play a checkpoint

Choosing **Model** for a colour asks the server which training runs it keeps, and offers the run's checkpoints: its **best** one by the run's own validation metric, its **latest**, or any step it has saved. Two more settings say how it plays:

- **rating** is what the model is asked to play like, given to the network as both sides' rating — the whole point of training on rated games. Left empty, the position claims no rating at all, which is also something the model was trained on.
- **plays** is either its best move every time, which makes the same position give the same move, or a sample from its distribution at a **temperature**: below 1 sharpens towards the best move, above 1 flattens towards a coin toss.

Both colours can be models, so two checkpoints of one run, or two runs, can be watched against each other. A model's moves are masked to the legal ones before the probabilities are normalized, so an untrained checkpoint plays badly rather than illegally.

Whether the rating does anything is a property of the training data rather than of the model player: a run whose games all came from one narrow band of ratings has never seen that feature move, and will have learned nothing from it. To see which it is for a checkpoint, play it against itself twice from the same position with **plays** set to its best move, changing nothing between the two games but the rating: a run that learned something from the feature plays a different game, and one that did not plays the same one move for move. Ask only for ratings inside the range the run trained on, which `chess-ai dataset stats <name>` reports for its dataset — a rating the run never saw takes the feature off the end of its training data, and whatever the model does then says nothing about what it learned.

The network runs on the CPU unless `[inference] device` says otherwise, so a game can be played against a checkpoint while a run is training on the GPU. The PGN of a game a model played records the run, the checkpoint's step, the rating it was asked for and how it chose its moves, as `WhiteRun`, `WhiteCheckpoint`, `WhiteRating` and `WhiteSelection` (and the same for Black).

On a human player's turn, hovering one of its pieces highlights that piece's legal destinations, drawing captures, castling and en passant apart from quiet moves. Move by clicking the piece and then the destination, or by dragging it there. The server is the only judge of the rules: it rejects anything illegal and the piece goes back where it was. A pawn reaching the last rank asks which piece to promote it to, and nothing is submitted until you pick one, by clicking it or with Enter or Space on the choice the picker opens on. Clicking elsewhere on the board, or pressing Escape, puts the pawn back.

Every game that reaches a result is saved as PGN in the games directory, one file per game, named after the moment it ended: nothing has to be asked for, and the file replays in any other chess tool. A game that was aborted reached no result and is not kept. **Export PGN** downloads the game on show whenever you like, a game still being played included, with the moves played so far and the result `*` that PGN gives a game that has not ended.

### Build a dataset

A dataset is what a model is trained on: every position of every game, with the move that
was actually played in it. Build one from any PGN files — a chess.com export, a Lichess
dump, your own games — naming the dataset and the files, directories or glob patterns to
read:

```sh
uv run chess-ai dataset build my-games games/*.pgn
uv run chess-ai dataset build masters ~/pgn/lichess-2024-01.pgn
```

It reads the files in parallel, one process per usable CPU core up to a cap, by cutting each
into pieces of whole games and parsing those at the same time. A Lichess month of 10.6M games
takes about 15 minutes rather than about four hours.

For ordinary PGN the dataset is identical either way, to the byte: the split a game lands in is
a hash of the game itself, and where its records go is decided in one place as the pieces come
back in order. The one shape that differs is a game whose `{}` comment contains a blank line
followed by a line beginning `[Event ` — a model game pasted into the notes — which reads as a
game boundary and is cut there. That game is kept with its moves stopping early, and the rest
of it is counted as one unreadable skip. If your PGN is annotated that way and you want it read
exactly, build it with `--workers 1`.

```sh
uv run chess-ai dataset build lichess-2017-01 ~/lichess/2017-01.pgn --workers 8
```

`--workers` defaults to one per usable CPU core, capped at 32 — so a 96-core machine uses 32
unless you ask for more, and asking is what the flag is for. There is a ceiling as well, a few
times the core count, above which a number is refused outright rather than run: `--workers 200`
on a 32-core machine stops before a byte is read. `chess-ai dataset build --help` prints both
numbers as they are on the machine you are on. `--workers 1` reads everything in this process,
which is also what a build small enough that starting processes would cost more than the
reading does. Each process holds part of a file while it reads it, so
the number is a memory cost as well as a speed one; lower it if a build is squeezing the
machine. A file whose games are not separated by a blank line has nothing to cut at, so it is
read in one process and the build says so.

It streams, so the input's size is not limited by memory, and it reports throughput and
time remaining as it goes. Games it cannot read — an illegal move, a position that is not a
position, a variant that is not chess, a game with no result — are skipped and counted by
reason rather than ending the build. A whole *file* that cannot be read is counted too, and
named in the manifest, but not silently: the command exits non-zero, and it refuses outright to
replace an existing dataset with a build that was missing one of its sources. Games whose ratings the file does not give are kept and
marked as unrated, and the rating pool (Lichess, chess.com) is read from each game's headers,
or given for every game with `--rating-source`.

Each dataset lands in `<data>/datasets/<name>/`, as sharded record files that the trainer
memory-maps, plus a `manifest.json` recording the sources, the counts, the statistics and
when it was built. A share of the *games* — `--validation-fraction`, 2% by default — is held
back for validation, chosen by a hash of each game, so no position of a game can be trained
on and validated against. The hash is of the game itself, so the same game always lands on
the same side, in every dataset it is ever built into.

What a dataset holds:

```sh
uv run chess-ai dataset stats my-games
```

```
dataset my-games
  built      2024-05-17 09:30:00 UTC
  format     version 1, move vocabulary 1968
  games      9,631 (train 9,436, validation 195)
  positions  742,905 (train 727,884, validation 15,021)
...
results
  1-0      4,812  50.0%  ████████████████████████████
  0-1      4,301  44.7%  █████████████████████████
  1/2-1/2    518   5.4%  ███
```

`.pgn.zst` Lichess dumps, a command that downloads them, and filters on rating, time control,
termination and date are next.

### Train a model

A run is started from one experiment config file, which says everything it depends on: the
dataset, the input encoding, the architecture and its size, the optimizer and the schedule, what
to validate and how often, what to keep, and the seed. [experiments/mlp-baseline.toml](experiments/mlp-baseline.toml)
lists every option with its built-in default noted beside it, and only `[dataset] name` has to be
filled in:

```sh
uv run chess-ai train experiments/mlp-baseline.toml
```

It uses the GPU when there is one, in bfloat16, and the CPU when there is not. Before it starts
it says what it is about to do, including how many parameters the model has and how fast a real
step actually ran, so a two-day run can be recognised as one before it is two days in:

```
chess-ai: run mlp-baseline, seed 1234
  device     cuda:0 NVIDIA GeForce RTX 4090, 24 GiB, bf16
  dataset    carlsen: 4,500 games, 331,842 train positions, 6,772 validation
  encoder    board-planes: spatial 12x8x8, globals 11, policy 1968
  model      mlp, 4,918,195 parameters (depth=3, dropout=0.0, width=1024)
  schedule   10,000 steps of 1024 (30.9 epochs), lr 0.001 warmup 500 then cosine
  throughput 240,000 positions/s measured, about 43s for the run
```

The model is shown a position as 12 planes of 8×8 — one per piece type and colour — and a
separate vector of what is not on the board: side to move, castling rights, en passant, the
halfmove clock, and both players' ratings divided by 5000 with a flag for each rating the file
did not give. It answers with a score for every one of the 1,968 moves in the shared vocabulary
and a win/draw/loss judgement of the position from the mover's point of view. The loss is
cross-entropy on the move actually played, plus a weighted cross-entropy on how the game
actually ended.

Two `[encoder]` options change what the model is shown, and both are off by default:

- `history = N` adds the N positions before the current one as 12 more planes each, most recent
  first, so the model can see what was just played. Planes for positions before the game began
  are all zero.
- `orientation = "side-to-move"` turns the board around when black is to move, so the mover's
  pieces are always on the same planes and always play up the board. The castling features and
  the move indices are mirrored along with it, and predictions are mirrored back before a move
  is played, so nothing outside the encoder sees the difference.

`[model] architecture` chooses the network, and the rest of `[model]` is that architecture's own
size settings:

- `"mlp"` flattens the planes, appends the global features and puts `depth` dense layers of
  `width` units on them. It ignores the board's geometry, which makes it the floor every other
  architecture has to beat.
- `"resnet"` is the AlphaZero and Maia residual tower: a 3×3 convolution into `channels` planes,
  `blocks` residual blocks of two more, and small policy and value heads.
  [experiments/resnet-lichess.toml](experiments/resnet-lichess.toml) trains one on the same
  data and schedule as the MLP in [experiments/mlp-lichess.toml](experiments/mlp-lichess.toml).
  A convolution has nowhere to put the global features, so `globals` says how they get in:
  `"planes"` paints each one over a whole 8×8 plane next to the board's, and `"film"`
  (feature-wise linear modulation) has them scale and shift every channel of every block, so
  that a rating can steer the whole tower rather than only its first layer.

Validation runs on the held-back games at `[validation] every_steps`, on the same positions every
time so the curve means something, and prints and logs five numbers:

```
  step    500/10,000  policy 4.8213  value 1.0402  top1 14.2%  top5 36.1%  illegal 12.4%
```

`top1` and `top5` are how often the played move is the model's first or top-five guess, which is
what a move-prediction model is for. `illegal` is how often the model's own best move cannot be
played at all: nothing tells the network the rules, so watching that fall is the clearest early
sign it is learning chess rather than move frequencies. At play time the distribution is masked
down to legal moves, so a high rate costs strength rather than legality.

Each run lands in `<runs>/<name>/`:

```
runs/mlp-baseline/config.toml        the config as it was written, comments and all
runs/mlp-baseline/run.json           the same config resolved, the code version, the seed,
                                     the dataset, the encoder spec and the parameter count
runs/mlp-baseline/metrics.jsonl      append-only, one JSON object per measurement
runs/mlp-baseline/status.json        the heartbeat: step, epoch, positions/s, ETA, GPU
runs/mlp-baseline/checkpoints/       step-<step>.pt, plus an index naming the best
```

Nothing is rewritten in place: the metrics log is appended to and everything else is written to a
temporary name and renamed over the old one, so the run directory can be read at any moment —
which is how the web server follows a run without ever talking to the trainer. Checkpoints
hold the weights, the optimizer state, the config and the encoder spec, so one is enough on its
own; `[checkpoints] keep` bounds how many are kept, and the best one by `[checkpoints] metric` is
kept however old it gets.

A run directory is never written into twice. Give the run another name, in the config or with
`--name`, or pass `--overwrite` to replace one:

```sh
uv run chess-ai train experiments/mlp-baseline.toml --name mlp-baseline-lr3
```

Resuming a stopped run and fine-tuning from another run's checkpoint are next. To watch a run as
it trains, open **Runs** in the browser (see [Follow training](#follow-training)).

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
