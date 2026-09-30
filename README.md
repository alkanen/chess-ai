# chess-ai

A testbed for training neural-network chess players the way large language models are trained: show the network a position, have it predict the move a human actually played, and repeat over millions of games. The goal is to compare model architectures on equal terms (MLP, ResNet, a transformer over the 64 squares, and a GPT-style model over move sequences) and to watch them learn through a browser UI.

> **Status: early development.** The web server and the board are in place, the server plays live games between random movers and human players with legal moves shown on hover, the CLI builds training datasets out of PGN files, and it trains an MLP on them from an experiment config file; more architectures, the evaluator and the runs dashboard come next. The full design is in the PRD: [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md).

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
| Frontend | React + TypeScript, built with Vite; custom SVG board |
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
which is how the web server will follow a run without ever talking to the trainer. Checkpoints
hold the weights, the optimizer state, the config and the encoder spec, so one is enough on its
own; `[checkpoints] keep` bounds how many are kept, and the best one by `[checkpoints] metric` is
kept however old it gets.

A run directory is never written into twice. Give the run another name, in the config or with
`--name`, or pass `--overwrite` to replace one:

```sh
uv run chess-ai train experiments/mlp-baseline.toml --name mlp-baseline-lr3
```

Resuming a stopped run, fine-tuning from another run's checkpoint, and following a run in the
browser are next.

### Choosing training parameters

The defaults in [experiments/mlp-baseline.toml](experiments/mlp-baseline.toml) will train
something. Getting a *good* result out of a large dataset means choosing a handful of numbers,
and they interact, so the order you choose them in matters. What follows is the reasoning and
the numbers one sweep produced, on an RTX 4090 against 10.6M Lichess games — treat the shapes as
transferable and the exact figures as this setup's.

**Choose with short runs, deliver with long ones.** The single most useful thing to know is that
a run reaches its endpoint when its cosine schedule bottoms out, whatever length you gave it. A
15,000-step run and a 172,000-step run of the same config land in the same place relative to
their own schedules, so a nine-minute run answers most questions that a two-hour run answers.
Sweep at a length you are willing to repeat a dozen times, and spend the long run only once
something has actually moved.

The corollary is a trap worth naming: **a loss curve falling all the way to the end of a run
tells you nothing about whether the run was worth its length.** That fall is mostly the learning
rate annealing. Only the final numbers of two completed runs can be compared.

#### Learning rate

This is the one that repays attention. In one sweep, getting it wrong cost more than an
eleven-fold increase in training data was worth.

It scales with **batch size**, and upward, which surprises most people. A larger batch does not
give a larger gradient — it gives the same gradient measured more accurately, since the variance
of the estimate falls as 1/batch. Noise is what forces a small step at small batch sizes; remove
the noise and the limit becomes curvature instead, which is further away. Linear scaling
(`lr ∝ batch`) is the rule for SGD; for AdamW use **√batch**, because Adam's normalisation by
`√v̂` already absorbs part of the change. So 1024 → 4096 takes 1e-3 to about 2e-3.

It scales with **model size**, and downward. A wider or deeper network wants a smaller rate,
roughly as 1/width. A learning rate tuned on a small model will quietly hobble a larger one, and
the symptom is that the larger model looks *worse* early in training rather than better.

Bracket it rather than guessing — test above and below, and make sure the winner has losers on
both sides. Three runs of 15,000 steps, 26.6M parameters, batch 16,384, all annealing to 2e-4:

| peak learning rate | policy loss | top-1 |
| --- | --- | --- |
| 1e-3 | 1.9844 | 41.6% |
| **2e-3** | **1.9536** | **42.7%** |
| 4e-3 | 2.0305 | 41.3% |

Too high is loud and cheap to detect: the loss spikes or goes NaN within a few hundred steps of
warmup ending. Too low is silent, and looks like a model that has run out of capacity.

#### The floor, and comparing two peaks fairly

`min_learning_rate_fraction` is a fraction **of the peak**, so changing the peak silently moves
the endpoint too — and a run that anneals further will look better for that reason alone. To
compare two peaks, set the fraction so the final rates match: 4e-3 with `0.05` and 2e-3 with
`0.1` both end at 2e-4.

You cannot make that comparison perfectly clean. With the budget and the floor both fixed, the
peak, the span and the rate of decay are algebraically tied: match any two against your baseline
and the third differs. That is a reason to stop decomposing and pick a setting, not a reason for
another run.

#### Batch size

Raise it until throughput stops improving, then stop. A small model at a small batch is not
limited by arithmetic but by per-step overhead, and batch size is nearly free until it isn't:

| batch | positions/s | per step |
| --- | --- | --- |
| 4,096 | 440,285 | 9.30 ms |
| 16,384 | 523,804 | 31.28 ms |

Going 1,024 → 4,096 tripled throughput; 4,096 → 16,384 added 19%, because four times the work
now costs 3.4 times the time. That is the knee, and past it you are paying for arithmetic rather
than reclaiming overhead. Read the median `positions_per_second` out of `metrics.jsonl` rather
than the startup estimate, which measures a cold GPU and under-reports by 2–3×.

Every increase in batch size needs the learning rate raised with it, or you have simply made the
run shorter: the same number of steps at the same rate now travels the same distance through a
larger fraction of the data.

#### How long to train

`steps × batch_size / positions` is how many epochs you get, and the run prints it before it
starts. Watch that line — 30,000 steps of 1,024 against 704M positions is 0.04 epochs, which is
almost certainly not what was intended.

More data does help, but **only once the learning rate is right**. In the same sweep, a four-epoch
run at 4e-3 (2.8 billion positions) finished worse than a 15,000-step run at 2e-3 that saw
less than a tenth as much. If more data appears not
to be helping, suspect the learning rate before concluding the model is saturated.

The learning-rate schedule spans `steps`, and resuming a finished run is not supported yet, so
decide the total length up front. Two epochs is `steps = 2 × positions / batch_size`; running a
one-epoch config twice gives two runs that each annealed to their floor, which is a different and
worse thing.

#### What did not matter

Worth knowing so you do not spend runs on them. In this sweep, neither moved the result by more
than measurement noise:

- **Warmup length.** 500 and 3,000 steps gave the same answer. Warmup exists to protect Adam
  while its second-moment estimate is built from too few gradients, and that window is about
  `1/(1-beta2)` steps — 20 at the default `beta2 = 0.95`. Anything from 1–5% of the run is fine.
- **`value_loss_weight`.** Dropping it from 0.5 to 0.1 moved the policy by 0.003. The value head
  is a genuinely hard, noisy auxiliary task — every position in a game carries that game's final
  result — so it neither learns much nor costs the policy much. Leave it at 0.5 and get the
  better value head for free.

#### What to watch

- **`illegal`** is the clearest signal that the network is learning chess rather than move
  frequencies, and it needs no interpretation. Across one series it went 32.3% → 1.21%.
- **The train/validation gap** is your overfitting detector: `policy_loss` on the `train` rows of
  `metrics.jsonl` against the `validation` rows. Across this series it stayed under +0.02, and
  under +0.01 for the two-epoch run. If it opens up, you have found the data limit.
- **`top1` against published work.** Maia-class residual networks reach roughly 50% move
  matching; a dense MLP on these inputs plateaus in the low-to-mid 40s.

#### Two settings that are not about model quality

- **`[validation] positions`** is your error bar, and it is coarser than it looks: 16,384
  positions come from only ~240 games, and positions within a game are heavily correlated, so the
  effective sample is far smaller than the count suggests. 163,840 spans ~2,400 games and costs
  about three seconds per validation, nearly all of it the per-position legality check.
  Differences under about a point of `top1` are not resolvable at the smaller size.
- **`[training] data_workers`** costs memory twice. Each worker builds its own shuffle
  permutation — four bytes per position, so 2.8 GB against a 704M-position split — and each
  batch in flight is pinned host memory, `batch_size × data_workers × 4` batches' worth. On
  platforms where page-locked memory is scarce (WSL2 caps it near 1 GiB whatever `ulimit -l`
  reports) a large batch with four workers can exhaust it, which surfaces confusingly as
  `CUDA error: out of memory` while the GPU is nearly empty. Two workers cost about 7% throughput
  and a great deal of headroom.

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
