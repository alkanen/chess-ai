# chess-ai

A testbed for training neural-network chess players the way large language models are trained: show the network a position, have it predict the move a human actually played, and repeat over millions of games. The goal is to compare model architectures on equal terms (MLP, ResNet, a transformer over the 64 squares, and a GPT-style model over move sequences) and to watch them learn through a browser UI.

> **Status: early development.** The web server and the board are in place, the server plays live games between human players, random movers, trained checkpoints and Stockfish, with legal moves shown on hover, the CLI builds training datasets out of PGN files, it trains an MLP or a residual CNN on them from an experiment config file, a runs dashboard follows training live in the browser, and `chess-ai match` plays two checkpoints (or a checkpoint and Stockfish) against each other over a set of openings; the transformers and the evaluator come next. The full design is in the PRD: [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md).

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
- [Stockfish](https://stockfishchess.org/download/), only to play against it. `sudo apt install stockfish` on Debian and Ubuntu, or a release binary from its site; it needs a version with the `UCI_Elo` strength limit, which every recent one has. The server finds it on `PATH`, or wherever `[stockfish] path` says.
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
| `[stockfish] path` | `CHESS_AI_STOCKFISH_PATH` | `stockfish`, looked up on `PATH` |

### Serve

```sh
uv run chess-ai serve
```

Then open the URL it prints, for example `http://127.0.0.1:8000/chess/` with `path_prefix = "/chess"`. The page, its assets, the API (`…/api/`, with interactive docs at `…/api/docs`) and the WebSockets that stream the games (`…/api/games/<link>/ws`) are all served under the prefix.

Start a game from the **Game** tab, choosing a human player, a random mover, a model or Stockfish for each colour, with a delay between moves so that a game between players that move instantly can be followed. Each game is a game of its own: several people can play at once, up to `[games] max_ongoing` games in progress (20 by default), and a new game beyond that is refused until one has ended.

### Share a game with friends

A game is reached only through its links, which the game page shows with a button to copy each:

- a **play link** (`#game/<id>`) for each side a person plays, which moves that side, takes back, resigns and aborts. In a game between two people each gets their own, and can move only their own side; the browser that started the game goes to White's link and shows Black's, to send to whoever plays Black.
- a **watch link** (`#watch/<id>`), which follows the game live and can do nothing to it.
- a game nobody plays by hand, such as two models, has one link that can abort it instead of a play link.

**The links are the only protection.** There are no accounts: each link is a random id nobody can guess, and whoever has it can do what it allows. Send play links only to the person who is to play that side, and serve the app to people you trust.

Keep your play link: nothing else leads back to the game. The Game tab lists the games this browser has started or opened, so a link that was not bookmarked can still be found from the same browser.

Between two people, a takeback is asked for and the other side agrees or declines; so is an abort, once both have moved. Against a model, Stockfish or the random mover, both happen at once. Resigning and aborting ask for confirmation first. A game that has ended stays to be looked at through its links, and an aborted game is deleted at once, links and all. Once a game has ended, **Play again** starts a new one with the same players and settings, with links of its own; a checkpoint chosen as latest or best is chosen afresh, so it may be a newer one. While that new game is being played, **Play again** on the old one joins it on your side instead of starting another, so two people who both click it end up playing each other again. A game a player could not go on with, such as a Stockfish that died, ends as stopped, with no result; it is not kept that way, though, and after the next restart of the server it carries on from where it stopped, with the player made afresh, unless it has been played again meanwhile.

A checkpoint deleted, or replaced by a run restarted under the same name, while its game waited does not end the game: when it is next to move, the game pauses and says why. Anyone with a play link (or the control link of a game nobody plays by hand) can then pick another checkpoint, offered from the same run, to play on from where the game stands, or abort. The move list marks where it took over, the PGN names the checkpoint that finished the game in its headers and notes the change as a comment at that move, such as `{tiny step 2 replaced by tiny step 4}`.

Games survive a restart of the server. Each game is written as it changes to a JSON file of its own in `[paths] ongoing_games` (`ongoing-games` by default), and read back when the server starts: the links still work and the game carries on where it was. A model's checkpoint is loaded again, and Stockfish started again, only when the game next needs them. A game is deleted, links and all, `[games] expire_after_days` (7) days after its last move, or after it ended if it has; the PGN of a game that ended stays in `[paths] games`.

### Follow training

**Runs** lists every training run with its state, architecture, dataset, progress and latest losses, top-1 accuracy and illegal-move rate, refreshed every few seconds. Opening a run shows its step, epoch, throughput and the device it trains on, and while it is running, the time left and how busy, full and hot the GPU is: the memory the run holds and what the whole card has in use, both as CUDA sees them, with utilization and temperature from NVML when it can be asked (without it, those two show as a dash). It also charts its loss (on a log scale), validation accuracy, learning rate and illegal-move rate against the step, and the gradient norm before clipping, with the threshold it is clipped at, and how many steps were clipped. Runs logged before the trainer measured gradients have no gradient charts. The charts follow the run over a WebSocket (`…/api/runs/<name>/ws`) as the trainer logs, so they update without reloading; a finished run shows the same charts for its whole history. Drag across a chart to zoom in, and double-click to let it follow the run again. Every chart can be drawn against the step, the positions seen (which lines up runs with different batch sizes) or the wall time; the choice is remembered in the browser.

Tick two or more runs in the list (up to eight) and choose **Compare** to overlay them on the same charts, one line per run: training and validation loss, top-1 and top-5 accuracy, illegal-move rate and learning rate, all followed live.

A run can be given a title, tags and notes, from its page in the browser or from the command line. They are kept in `notes.json` in the run directory, and the run list can be filtered by tag. The run's name never changes, since it is the directory; the title is what it is shown as instead.

```sh
uv run chess-ai runs annotate mlp-baseline --title "MLP baseline" --tag mlp --tag baseline --notes "First run on the 2024 games."
uv run chess-ai runs annotate mlp-baseline --untag baseline --notes-file notes.md   # - reads standard input
uv run chess-ai runs annotate mlp-baseline     # shows what it has
uv run chess-ai runs list --tag mlp
```

To set a run aside without deleting it, tag it `archived`: `uv run chess-ai runs annotate NAME --tag archived`, and `--untag archived` to bring it back. An archived run is left out of `chess-ai runs list` (unless `--archived` or `--tag archived` asks for it), hidden on the dashboard behind a **Show archived runs** box, and never offered to play against; it can still be opened and compared. That is the place for a sweep's runs, whose curves are worth keeping and whose checkpoints nobody will play.

The web server never talks to a trainer: it reads the run directory, so it can be restarted at any time without affecting a run. A trainer that dies without saying so — killed, out of memory, or the machine gone — leaves a heartbeat that claims it is still running. Once that heartbeat is older than `[server] stale_after_seconds` (two minutes by default), the run is flagged **stale**. A trainer rewrites its heartbeat every couple of seconds, but not while it starts up, validates or saves a checkpoint, so keep the threshold well above those.

The charts are drawn with [uPlot](https://github.com/leeoniya/uPlot): it is about 50 kB, draws on a canvas, and redraws the tens of thousands of points a long run logs without slowing the page, which SVG charting libraries struggle with.

### Play a checkpoint

Choosing **Model** for a colour asks the server which training runs it keeps. Each run is offered by its title, if it has one (`chess-ai runs annotate NAME --title ...`, or the run's page), and otherwise by its name without the leading timestamp, cut down to the experiment when it is long; two runs that would look the same are told apart by when they started, and the chosen run's full name is shown under the list. The form then offers the run's checkpoints: its **best** one by the run's own validation metric, its **latest**, or any step it has saved. Two more settings say how it plays:

- **rating** is what the model is asked to play like, given to the network as both sides' rating — the whole point of training on rated games. Left empty, the position claims no rating at all, which is also something the model was trained on.
- **plays** is either its best move every time, which makes the same position give the same move, or a sample from its distribution at a **temperature**: below 1 sharpens towards the best move, above 1 flattens towards a coin toss.

Both colours can be models, so two checkpoints of one run, or two runs, can be watched against each other. A model's moves are masked to the legal ones before the probabilities are normalized, so an untrained checkpoint plays badly rather than illegally.

Whether the rating does anything is a property of the training data rather than of the model player: a run whose games all came from one narrow band of ratings has never seen that feature move, and will have learned nothing from it. To see which it is for a checkpoint, play it against itself twice from the same position with **plays** set to its best move, changing nothing between the two games but the rating: a run that learned something from the feature plays a different game, and one that did not plays the same one move for move. Ask only for ratings inside the range the run trained on, which `chess-ai dataset stats <name>` reports for its dataset — a rating the run never saw takes the feature off the end of its training data, and whatever the model does then says nothing about what it learned.

The network runs on the CPU unless `[inference] device` says otherwise, so a game can be played against a checkpoint while a run is training on the GPU. Games playing the same checkpoint share one loaded copy; at most `[games] max_loaded_checkpoints` (10) are held at once, the one used longest ago making room for another, and one no game has played with for `checkpoint_idle_hours` (24) is let go of. Either is loaded again when a game next needs it. The PGN of a game a model played records the run, the checkpoint's step, the rating it was asked for and how it chose its moves, as `WhiteRun`, `WhiteCheckpoint`, `WhiteRating` and `WhiteSelection` (and the same for Black).

On a human player's turn, hovering one of its pieces highlights that piece's legal destinations, drawing captures, castling and en passant apart from quiet moves. Move by clicking the piece and then the destination, or by dragging it there. The server is the only judge of the rules: it rejects anything illegal and the piece goes back where it was. A pawn reaching the last rank asks which piece to promote it to, and nothing is submitted until you pick one, by clicking it or with Enter or Space on the choice the picker opens on. Clicking elsewhere on the board, or pressing Escape, puts the pawn back.

Every game that reaches a result is saved as PGN in the games directory, one file per game, named after the moment it ended: nothing has to be asked for, and the file replays in any other chess tool. A game that was aborted reached no result and is not kept. **Export PGN** downloads the game whenever you like, through any of its links, a game still being played included, with the moves played so far and the result `*` that PGN gives a game that has not ended.

### Play Stockfish

Choosing **Stockfish** for a colour plays the engine at the **Elo** you type, held there by its own calibrated strength limit, thinking for the **time a move** you choose. Stockfish only plays a range of strengths, which the form shows as soon as Stockfish is chosen: 1350 to 2850 for the Stockfish 14.1 that `apt` installs on this kind of machine, 1320 to 3190 for Stockfish 19. An Elo outside it is played at the nearer end, and the game names Stockfish by the strength it actually plays at and says what was asked for. Stockfish can play either colour, or both at two strengths, against a person, the random mover or a checkpoint.

Stockfish's documentation says the levels were calibrated at two minutes a game plus a second a move and anchored to the CCRL 40/4 engine rating list. A move time well under a second plays below the level, and neither is a Lichess or FIDE rating.

Each Stockfish side is an engine process of its own, started with the game and stopped when the game ends or the server shuts down. A game that cannot find Stockfish is refused with a message saying where it looked. The PGN of a game Stockfish played records its strength as `WhiteElo` and its time a move as `WhiteMoveTime` (and the same for Black).

### Play a match

Validation metrics stop telling two checkpoints apart once they get close; games do. `chess-ai match` plays a number of games between two players and prints the score from each side's point of view:

```sh
uv run chess-ai match resnet10x128 resnet6x64 --games 100
uv run chess-ai match resnet10x128@latest,rating=1500 stockfish:1350,move-time=0.5
uv run chess-ai match mlp@24000 mlp@48000 --temperature 0.25 --seed 7
```

A player is a run, optionally with `@best` (the default), `@latest` or `@STEP`, followed by any of `rating=R`, `strategy=argmax|sample` and `temperature=T`, all separated by commas; or `stockfish:ELO`, optionally with `move-time=SECONDS`. `--temperature` makes every checkpoint that names no strategy of its own sample at that temperature. `chess-ai match --help` has the details.

Playing its best move, a checkpoint plays the same game from the same position every time, so the games start from the lines of a curated opening set instead: `standard`, 100 main lines a few moves deep, kept in `src/chess_ai/opening_sets/` and versioned, so that two matches on the same set started from the same positions. Each line is played twice in a row with the colours swapped, which also keeps a player that is better only as White from looking better outright, and the score is printed as White and as Black as well as overall. `--openings PATH` plays a set of your own, in the same TOML form. A match longer than twice the set starts repeating its openings, and warns that deterministic players will repeat their games too.

Sampling with `--temperature` or `temperature=` varies the games further, and `--seed` (printed when it was not given) replays a match move for move. A low temperature such as 0.25 only changes a game where the moves are close, so on its own it can give near-copies; the openings are what make the games differ. Stockfish's own play is not seeded, so a match against it does not replay exactly.

The `±` after each score is a rough 95% range from how the games went: about ±14 points of score over 50 games and ±7 over 200, less with draws. It says how much a rerun might differ, not an Elo difference. Every game is appended as it ends to one PGN file in the games directory, `YYYYMMDD-HHMMSS-match-ID.pgn`, so a match stopped with ctrl-c keeps the games it finished; the replay view opens the file as a list of its games. The games carry the usual model and Stockfish headers, plus `Event` "chess-ai match", the game's number as `Round`, the line as `Opening` and the set as `OpeningSet`.

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
  format     version 2, move vocabulary 1968
  games      9,631 (train 9,436, validation 195)
  positions  742,905 (train 727,884, validation 15,021)
...
results
  1-0      4,812  50.0%  ████████████████████████████
  0-1      4,301  44.7%  █████████████████████████
  1/2-1/2    518   5.4%  ███
```

#### Filters

A build keeps every game by default. Filters narrow it down, for fine-tuning on stronger
players or slower games:

```sh
uv run chess-ai dataset build masters-classical ~/lichess/2024-*.pgn \
    --min-rating 2200 --time-controls rapid,classical --exclude-terminations abandoned
```

| Option | Keeps |
|---|---|
| `--min-rating`, `--max-rating` | positions whose player to move is rated within the limits; either may be left out |
| `--unknown-rating-passes` | players without a rating as well, when a limit is set (by default they fail it) |
| `--min-clock SECONDS` | positions whose player to move had at least this long left, from the `[%clk]` comments |
| `--time-controls` | games of these classes: `bullet`, `blitz`, `rapid`, `classical`, `correspondence`, `unknown` |
| `--exclude-terminations` | games that did not end like this: `normal`, `time_forfeit`, `abandoned`, `unterminated`, `unknown` |
| `--from`, `--until` | games played in the range, each `YYYY`, `YYYY-MM` or `YYYY-MM-DD` and inclusive |
| `--sample FRACTION` | that fraction of the games that pass, chosen by a hash of each game |
| `--max-games N` | the first N games that pass, in the order the files give them; reading stops there |

The rating and the clock filters are about the *player to move*, so they decide which positions
are trained on rather than which games are kept. A game between a 2300 and an 1800 built with
`--min-rating 2000` is stored whole, since the history an encoder reads and the moves a
sequence model reads are the whole game, but only the 2300's positions are trained on and
validated against. A game none of whose positions is left to train on is not kept. A position
with no clock comment passes `--min-clock`; before a side's first move, the clock is the time
control's starting time.

The other filters are about the whole game and are read off its headers, so a game they leave
out is never parsed: a build that keeps a few percent of a dump is much faster than one that
keeps it all. A game with no date fails a date range, and one dated only to the month has to
fall inside it for the whole month. A sample is a hash of the game, independent of the
validation split: the same game is in or out of every sample of that size, and a 5% sample is
part of a 10% one. `--sample` and `--max-games` can be combined, filters first, then the
sample, then the maximum.

Games that a site ended for a rules infraction, which is mostly cheating, are left out of every
dataset whatever the filters say, and counted as skipped.

The same settings can be kept in a TOML file and given with `--definition`; options on the
command line override it, and sources on the command line replace its sources. Sources in the
file are relative to the file. The name stays on the command line, so one file can build both a
full dataset and a quick `--max-games 10000` one.

```toml
# datasets/masters-classical.toml
sources = ["../lichess/lichess_db_standard_rated_2024-0[1-6].pgn"]
validation_fraction = 0.02

[filters]
min_rating = 2200
time_controls = ["rapid", "classical"]
exclude_terminations = ["abandoned"]
from = "2024-01"
until = "2024-06"
min_clock = 30
```

```sh
uv run chess-ai dataset build masters-classical --definition datasets/masters-classical.toml
```

The manifest records the filters the build used, how many games each one left out, and how many
positions are stored but not trained on; `dataset stats` and the datasets page show them.
Datasets built before there were filters are still read, as unfiltered.

#### Lichess dumps

Lichess publishes every month of rated standard games as a zstd-compressed PGN file. `dataset
download` fetches the months you name into `lichess/` under the data directory, kept compressed:

```sh
uv run chess-ai dataset download 2024-01 2024-02
uv run chess-ai dataset build lichess-2024-q1 data/lichess/lichess_db_standard_rated_2024-0*.pgn.zst
```

A download in progress is a `.part` file. One that stops, or is stopped with ctrl-c, carries on
from where it got to the next time the month is asked for. Each file is checked against the
SHA-256 Lichess publishes before it gets its own name; one that does not match is deleted, to be
downloaded again from the start. A month already there is not downloaded again.

A build reads `.pgn.zst` files as they are, without decompressing them to disk, and picks them up
from directories and globs alongside `.pgn` files. The dataset is the same, to the byte, as one
built from the decompressed file. A compressed file cannot be cut where a plain one is, so the
build's own process decompresses it and hands the text to the other processes in pieces; that is
fast enough to keep them busy, and the progress is measured in compressed bytes. A file that is
cut short or damaged keeps the games before the damage, and the manifest records the error
against it.

#### Appending to a dataset

A dataset can grow, for example by the next Lichess month, without being rebuilt:

```sh
uv run chess-ai dataset append lichess-2024-q1 data/lichess/lichess_db_standard_rated_2024-04.pgn.zst
```

Each append is a new **version** of the dataset. The new games go through the filters the
dataset was built with (they belong to the dataset, not to an append), and are added after the
games already there, so version *n* is always the first so many games and positions of version
*n + 1*. `--max-games` raises the dataset's cap on its total games, for a dataset that has
reached it. Appending gives the same dataset, to the byte, as building from all the files at once.

A file already in the dataset is refused: recognised by the SHA-256 of its bytes, taken before
reading starts (a pass over the file, tens of seconds for a Lichess month), or by being the same
Lichess month, which catches a `.pgn` appended after its own `.pgn.zst`. `--allow-repeat` appends
it anyway. A build refuses the same file named twice under two names in the same way.

An append holds the dataset's lock, and runs reading the dataset are not disturbed by it. The
manifest is written last, so an append that is killed leaves records past the last version that
nothing reads; the next append refuses to start over them until it is told
`--discard-interrupted`, which cuts them off. `dataset stats` lists the versions, and
`dataset stats NAME --version N` describes an earlier one.

### Train a model

A run is started from one experiment config file, which says everything it depends on: the
dataset, the input encoding, the architecture and its size, the optimizer and the schedule, what
to validate and how often, what to keep, and the seed. [experiments/mlp-baseline.toml](experiments/mlp-baseline.toml)
lists every option with its built-in default noted beside it, and only `[dataset] name` has to be
filled in:

```sh
uv run chess-ai train experiments/mlp-baseline.toml
```

A run trains on the latest version of its dataset unless `[dataset] version` names one. The run
records the version it resolved to, the runs list and run page show it as `NAME vN`, and a resumed
run carries on with that version whatever has been appended since.

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
  optimizer  AdamW, weight decay 0.01 on 4,913,152 parameters, none on 5,043 biases and norms, clip 1
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
  data and schedule as the MLP in [experiments/mlp-lichess.toml](experiments/mlp-lichess.toml),
  at a higher learning rate that a sweep of short runs found suits it better.
  A convolution has nowhere to put the global features, so `globals` says how they get in:
  `"planes"` paints each one over a whole 8×8 plane next to the board's, and `"film"`
  (feature-wise linear modulation) has them scale and shift every channel of every block, so
  that a rating can steer the whole tower rather than only its first layer.

Weight decay shrinks the weights that multiply an input, and leaves biases and the
normalization layers' scales and shifts alone, as is usual. Runs made before the trainer told
the two apart decayed everything; `[optimizer] decay_biases_and_norms = true` brings that
back, to reproduce one of them.

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
runs/mlp-baseline/writer.lock        locked by the process training the run, while it does
```

Nothing is rewritten in place: the metrics log is appended to and everything else is written to a
temporary name and renamed over the old one, so the run directory can be read at any moment —
which is how the web server follows a run without ever talking to the trainer. Checkpoints
hold the weights, the optimizer state, the random state, the config and the encoder spec, so one
is enough on its own; `[checkpoints] keep` bounds how many are kept, and the best one by
`[checkpoints] metric` is kept however old it gets.

A run directory is never written into twice. Give the run another name, in the config or with
`--name`, or pass `--overwrite` to replace one:

```sh
uv run chess-ai train experiments/mlp-baseline.toml --name mlp-baseline-lr3
```

To watch a run as it trains, open **Runs** in the browser (see [Follow training](#follow-training)).

#### Stop and resume

Ctrl-C, or `kill -INT` or `kill -TERM` on the trainer's process, stops a run cleanly: it
finishes the step it is on, saves a checkpoint, marks the run **stopped**, and exits with
status 130 or 143. A second ctrl-c stops it at once, without that checkpoint. To carry on:

```sh
uv run chess-ai resume mlp-baseline
```

A resumed run picks up its latest checkpoint and carries on exactly where it was: the same
weights, optimizer state, learning rate schedule, random state and data order, so a run that was
stopped and resumed ends up with the same weights as one that never stopped. A run that crashed
or was killed resumes the same way, from the last checkpoint it saved; whatever it logged after
that checkpoint is dropped from its metrics, since those steps are trained again. The schedule
is the one the run started with, so resuming finishes a run but cannot make it longer.

A run is resumed on the dataset of the same name, and is refused if that dataset has been
rebuilt since, because its positions would come in a different order. It is also refused while
another process is still writing it.

#### Fine-tune from another run

An `[initialize_from]` section starts a new run from another run's weights instead of from its
seed:

```toml
[initialize_from]
run = "mlp-pretrained"
checkpoint = "best"    # or "latest", or a step number
```

Only the weights are taken. The new run has its own dataset, optimizer, schedule and step count,
which is how a curriculum goes from every game to the strongest players. The architecture, its
size and the encoder settings have to match the ones the other run trained with. A checkpoint
that does not match is refused, with a message saying what differs. Settings that do not change
the shape of the weights, such as `dropout`, may differ. `run.json` records the run, step and
dataset the weights came from, under `initialized_from`.

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
    # The port must be the one in [server] port.
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 1h;
}
```

If nginx runs on another machine, or on the Windows side of a WSL2 setup, set `host = "0.0.0.0"` so the server accepts connections from outside, and point `proxy_pass` at an address nginx can reach.

If the page loads but keeps saying it lost the connection to the server, and the server logs `GET …/api/games/<link>/ws` answered `426 Upgrade Required`, the WebSocket requests are reaching it without the upgrade: nginx is not forwarding the headers above (an `HTTP/1.0` in the log line means `proxy_http_version 1.1` is missing). A port in `proxy_pass` that does not match the server's shows up in nginx instead, as a refused connection.

The app has no authentication. It will have a read-only mode that disables starting and stopping jobs from the browser.

## License

GPL-3.0, matching the python-chess dependency.

The chess pieces are Colin M.L. Burnett's "cburnett" set, used under the GPL (version 2 or later); see [frontend/src/board/pieces/cburnett/LICENSE.md](frontend/src/board/pieces/cburnett/LICENSE.md).
