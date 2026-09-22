# PRD: Chess AI Trainer and Game Display

- **Status:** Draft
- **Date:** 2026-09-22
- **License:** GPL-3.0 (matches the python-chess dependency)

## Problem Statement

I want to find out how well different neural network architectures can learn to play chess when trained the way a large language model is trained: show the network a position, have it predict the move a human actually played, and repeat over a very large corpus of games. I want to compare several model families (plain MLPs, convolutional residual networks, attention-based transformers over the board, and GPT-style models over move sequences) on equal terms, using the same data, the same move vocabulary and the same evaluation.

Today I have no tooling for any of this. There is no pipeline to turn downloaded PGN files into training data, no common framework in which different architectures can be swapped without rewriting the training loop, no standard way to measure how strong a checkpoint is, and no way to *see* what a model is doing. Training runs take hours or days on my home machine (RTX 4090), and I want to follow them from work: watch loss curves, see the model play sample games, inspect what it thinks of particular positions, and play against it myself. Because the machine is only reachable through my nginx reverse proxy, anything I build must work under a URL path prefix.

## Solution

A single Python project with three cooperating parts, plus a browser UI:

1. **Data pipeline.** Imports any PGN files (chess.com exports, Lichess monthly database dumps including compressed `.zst` files, or my own games). It turns them into versioned, encoder-independent datasets, with or without filters on player rating, time control and game termination. This supports a curriculum: pretrain on everything, then fine-tune on progressively stronger players.
2. **Trainer.** Trains any registered architecture from a single experiment config file. Every model receives the same structured input: spatial board features, a separate vector of global features (side to move, castling rights, the mover's and opponent's ratings, and so on), and optionally the move sequence. Every model produces the same outputs: a probability distribution over a fixed move vocabulary (the *policy*) and a win/draw/loss estimate (the *value*). Runs write everything they produce to a self-describing run directory.
3. **Evaluator.** A background worker that picks up new checkpoints and evaluates them without slowing training:
   - sample games that can be watched live
   - the model's predictions on a fixed set of probe positions
   - matches against a ladder of strength-limited Stockfish levels, giving an Elo estimate
   - Lichess puzzle accuracy
   - tournaments between checkpoints and architectures
4. **Web display.** A browser app, served by a Python web server under a configurable path prefix, where I can:
   - watch games stream live, with legal moves shown on hover (including castling, en passant and promotion)
   - play against any checkpoint at a chosen "play like rating X" setting, optionally seeing the model's top candidate moves and win/draw/loss estimate
   - follow training runs with live charts, run status and GPU statistics
   - browse evaluation results
   - start, stop and resume training and evaluation jobs

The first deliverable is deliberately small: a board in the browser showing pieces moving in a server-hosted game between swappable players (human or random mover), with a turn indicator and legal-move hover highlighting. Everything else is added one vertical slice at a time.

## User Stories

Actors: the **experimenter** trains and evaluates models, the **player** plays games in the browser, the **viewer** watches games and runs (often remotely from work), and the **operator** deploys and runs the system. All four are the same person, wearing different hats.

### Game display and play

1. As a viewer, I want to open the app in any modern browser on any operating system, so that I don't need to install anything to use it.
2. As a viewer, I want to see a chessboard with clearly drawn pieces that scales to the available space, so that it is readable on a large monitor and on a laptop.
3. As a viewer, I want a clear indicator of whose turn it is, so that I always know which side is to move.
4. As a viewer, I want pieces to move on the board as moves are played, so that I can follow the game as it happens.
5. As a viewer, I want the most recent move highlighted, so that I can see what just happened even when moves arrive quickly.
6. As a viewer, I want a king in check to be highlighted, so that checks are obvious.
7. As a viewer, I want the game-over state shown with the result and the reason (checkmate, stalemate, insufficient material, threefold repetition, fifty-move rule, resignation, or abort), so that I know how the game ended.
8. As a viewer, I want a move list in standard algebraic notation next to the board, so that I can review the game so far.
9. As a player, when I hover over one of my pieces on my turn, I want its legal destination squares highlighted, so that I can see my options at a glance.
10. As a player, I want capture destinations drawn differently from quiet-move destinations, so that I can tell them apart.
11. As a player, I want castling shown as a legal king destination only when castling is actually legal (rights intact, path empty, and not castling out of, through or into check), so that the highlights never mislead me.
12. As a player, I want en passant captures shown as legal destinations when available, so that this special move is discoverable.
13. As a player, I want pinned pieces and moves that would leave my king in check to show no illegal destinations, so that the highlights reflect the real rules.
14. As a player, I want to make a move by clicking a piece and then a destination, or by dragging and dropping, so that I can use whichever I prefer.
15. As a player, I want a promotion picker (queen, rook, bishop, knight) when my pawn reaches the last rank, so that I can underpromote when I want to.
16. As a player, I want an illegal move attempt to be rejected without changing the game state, so that I cannot corrupt a game by mis-clicking.
17. As a viewer, I want to start a game between two players and choose who controls White and who controls Black (human, random mover, a model checkpoint, or Stockfish at a chosen strength), so that I can set up any matchup.
18. As a viewer, I want to watch a random mover play another random mover, streamed live from the server, so that I can confirm the display works before any model exists.
19. As a viewer, I want everyone viewing the same game to see the same state in real time, so that I can watch from work a game that is running on my home machine.
20. As a viewer, I want a page reload or a dropped connection to reconnect me to the ongoing game with its full state, so that I never lose track of a game.
21. As a player, I want to flip the board and play as Black with Black at the bottom, so that I see the board from my own side.
22. As a player, I want to take back moves in games against the AI, so that I can explore alternatives.
23. As a player, I want to start a game from a custom position (FEN), so that I can test the model in specific situations.
24. As a player, I want to resign or abort a game, so that I can end games I am no longer interested in.
25. As a player, I want to export any game as PGN, so that I can analyse it in other tools.
26. As a player, I want the games I play to be saved automatically as PGN, so that I can review them later or use them as data.
27. As an experimenter, I want to open any PGN file or any game from a dataset in a replay viewer and step forward and backward through it, so that I can inspect training data and saved games.

### Playing against the AI

28. As a player, I want to choose which training run and which checkpoint I play against, so that I can feel how the model's strength changes during training.
29. As a player, I want to set the rating the model should imitate (for example "play like a 1600"), so that I can explore the model's rating conditioning.
30. As a player, I want to choose between the model always playing its top move and sampling with a temperature, so that I can trade strength for variety.
31. As a player, I want an optional "thinking" overlay showing the model's top candidate moves with their probabilities (as arrows and a list), so that I can understand its choices.
32. As a player, I want an optional evaluation bar showing the model's win/draw/loss estimate for the current position, so that I can see how it judges the game.
33. As a player, I want to play against Stockfish at a chosen strength, so that I have a calibrated reference opponent.
34. As a viewer, I want to watch model vs Stockfish and model vs model (for example two checkpoints, or two architectures) in the same game view, so that I can compare them directly.
35. As a developer, I want every kind of player (human, random, model, Stockfish, and later search-based players) to implement one common player interface, so that new player types can be added without changing game sessions, matches or the UI.

### Data

36. As an experimenter, I want to import any PGN file regardless of source, so that I can use chess.com exports, Lichess dumps and my own games.
37. As an experimenter, I want compressed Lichess dumps (`.pgn.zst`) read as a stream without decompressing them to disk first, so that I don't need hundreds of gigabytes of scratch space.
38. As an experimenter, I want a helper command that downloads Lichess monthly database dumps, so that getting millions of games is one command.
39. As an experimenter, I want to build an unfiltered dataset containing every move from every game, so that I can pretrain on everything, as LLMs do.
40. As an experimenter, I want to build filtered datasets (minimum rating of the player to move, allowed time controls, excluded termination types such as abandonment or time forfeit, date ranges, maximum game count), so that I can fine-tune on stronger players.
41. As an experimenter, I want each dataset to record its sources, filters, counts and creation time in a manifest, so that every run can be traced back to exactly what it trained on.
42. As an experimenter, I want the train/validation split made by game rather than by position, so that positions from the same game never leak between the two sets.
43. As an experimenter, I want games with missing ratings kept and marked as "rating unknown" rather than discarded, so that unrated sources remain usable.
44. As an experimenter, I want the rating source or pool (for example chess.com or Lichess) recorded with each game, so that I can account for the fact that different rating pools are not directly comparable.
45. As an experimenter, I want malformed or illegal games skipped and counted rather than crashing the build, so that a multi-hour import of millions of games is robust.
46. As an experimenter, I want progress, throughput and time-remaining reporting during dataset builds, so that I know how long to wait.
47. As an experimenter, I want datasets stored compactly and independently of any encoder, so that the same dataset can feed every architecture and every encoder variant.
48. As an experimenter, I want summary statistics for a dataset (games, positions, rating distribution, result distribution, time-control mix), so that I understand what a model will learn from.
49. As a viewer, I want to see the available datasets and their statistics in the web UI, so that I can choose datasets when starting runs.

### Encoding and models

50. As an experimenter, I want to select the input encoder by name in the experiment config, so that I can compare input representations.
51. As an experimenter, I want a board-planes encoder (one plane per piece type and color, on an 8×8 grid) with an optional history of previous positions, so that convolutional and attention models get spatial input.
52. As an experimenter, I want an option to orient the board from the perspective of the side to move, so that I can test whether this canonical orientation helps learning.
53. As an experimenter, I want non-spatial information (side to move, castling rights, en passant availability, halfmove clock, the mover's rating, the opponent's rating, rating-unknown flags) delivered as a separate global feature vector rather than baked into the board planes, so that each architecture can consume it in the way that suits it.
54. As an experimenter, I want the rating normalization to be configurable (default: rating divided by 5000), so that I can experiment with scaling while leaving headroom above today's top ratings.
55. As an experimenter, I want a move-sequence encoder that turns the game so far into move tokens, so that GPT-style models can be trained on sequences exactly like a language model.
56. As an experimenter, I want to choose the architecture by name in the experiment config (MLP, ResNet, square transformer, GPT), so that switching architectures is a one-line change.
57. As an experimenter, I want to set each architecture's size hyperparameters (depth, width, attention heads, and so on) in the config, so that I can compare models of similar size or explore scaling.
58. As an experimenter, I want every architecture to output a policy over the same fixed move vocabulary and a win/draw/loss estimate, so that training, evaluation and play treat all models identically.
59. As an experimenter, I want to add a new architecture by implementing one interface and registering it under a name, without touching the trainer, evaluator or UI, so that trying new ideas is cheap.
60. As an experimenter, I want to see a model's parameter count and approximate throughput before a long run, so that I can size experiments sensibly.

### Training

61. As an experimenter, I want to start a training run from the command line with a single experiment config file, so that every run is fully described and repeatable.
62. As an experimenter, I want to start a training run from the browser by picking a saved experiment config, so that I can kick off experiments from work.
63. As an experimenter, I want to stop a running training job from the browser, with a checkpoint saved before it exits, so that I don't lose progress.
64. As an experimenter, I want to resume a stopped or crashed run from its last checkpoint, including optimizer state, so that interruptions cost nothing.
65. As an experimenter, I want to start a new run initialized from another run's checkpoint but trained on a different dataset, so that I can pretrain on everything and then fine-tune on masters.
66. As an experimenter, I want training to use mixed precision and keep the RTX 4090 busy, so that experiments finish as fast as the hardware allows.
67. As an experimenter, I want checkpoints saved periodically, keeping a configurable number of recent ones plus the best one, so that disk use stays bounded.
68. As an experimenter, I want validation metrics computed periodically (policy loss, value loss, top-1 and top-5 move accuracy, and how often the model's top move is illegal), so that I can track learning.
69. As an experimenter, I want each run directory to record its config, dataset reference, code version and random seed, so that results are reproducible and comparable.
70. As an experimenter, I want to name, tag and annotate runs, so that I can find and compare them later.
71. As an operator, I want only one GPU training job running at a time, with further start requests refused or queued, so that jobs don't fight over GPU memory.

### Watching training

72. As a viewer, I want a list of all runs with their status (queued, running, stopped, finished, crashed), architecture, dataset and latest key metrics, so that I get an overview at a glance.
73. As a viewer, I want live charts of loss, accuracy, learning rate and illegal-move rate that update without reloading the page, so that I can watch training progress from work.
74. As a viewer, I want to overlay several runs on the same chart, so that I can compare architectures and settings.
75. As a viewer, I want to see a running job's current step and epoch, positions per second, estimated time remaining, and GPU utilization, memory and temperature, so that I know it is healthy.
76. As a viewer, I want a run that has stopped reporting to be flagged as stale or crashed, so that silent failures don't go unnoticed.
77. As a viewer, I want the sample games played at each checkpoint streamed onto live boards, so that I can literally watch the model improve.
78. As a viewer, I want to see the model's top moves and probabilities on a fixed set of probe positions (openings, tactics, endgames), so that I can see what it "understands".
79. As a viewer, I want to scrub through checkpoints on the probe positions, so that I can see how the model's understanding changed during training.
80. As a viewer, I want evaluation results per checkpoint (Elo estimate, puzzle accuracy, tournament standing) charted over the course of training, so that I can see strength trends and not just loss.

### Evaluation

81. As an experimenter, I want a background evaluator that automatically evaluates new checkpoints, so that evaluation happens without my involvement and without pausing training.
82. As an experimenter, I want to configure which evaluation suites run for each run, so that expensive evaluations only run where they are useful.
83. As an experimenter, I want to re-run evaluations on old checkpoints on demand, from the CLI or the browser, so that I can apply new or changed suites retroactively.
84. As an experimenter, I want each checkpoint to play matches against a ladder of Stockfish levels with fixed Elo ratings and get an Elo estimate with a confidence interval, so that I can put a number on strength and know how much to trust it.
85. As an experimenter, I want puzzle accuracy on the Lichess puzzle database, overall and broken down by puzzle rating and theme, so that I can measure tactical ability separately from game play.
86. As an experimenter, I want round-robin tournaments between chosen checkpoints and architectures with a relative rating leaderboard, so that I can compare models head to head.
87. As an experimenter, I want evaluation matches to start from a varied set of opening positions with colors alternated, so that deterministic models don't replay the same game over and over.
88. As an experimenter, I want all evaluation games saved as PGN in the run directory, so that I can inspect any game later.
89. As an operator, I want evaluation to use mostly CPU and at most a small share of the GPU, so that it doesn't materially slow down training.

### Operations

90. As an operator, I want to serve the web app under a configurable URL path prefix, so that it can sit behind my nginx reverse proxy alongside other services.
91. As an operator, I want every page, asset, API and WebSocket URL to respect the path prefix, so that nothing breaks when proxied.
92. As an operator, I want a single command-line tool with subcommands for data, training, evaluation and serving, so that the system is easy to operate.
93. As an operator, I want the data directory, runs directory, Stockfish binary path, path prefix, bind address and port to be configurable, so that the system fits my machine and network.
94. As an operator, I want a read-only mode that disables starting and stopping jobs from the browser, so that I can lock it down while the app has no authentication.
95. As an operator, I want training and evaluation jobs to keep running when the web server restarts, so that deploying UI changes never interrupts a multi-day run.
96. As an operator, I want the frontend built once into static files served by the Python server, so that running the system needs only Python (Node is needed only to build it).
97. As an operator, I want the README to document setup, including GPU/CUDA prerequisites, installing Stockfish, and the nginx settings needed for WebSockets, so that I can deploy it without guesswork.

## Implementation Decisions

### Technology

- **Language and packaging:** Python 3.12, managed with `uv`. The system Python (3.10) is not used.
- **Machine learning:** PyTorch with CUDA on a single RTX 4090 under WSL2. Training uses bf16 mixed precision, with optional `torch.compile`.
- **Chess rules, PGN parsing and UCI:** `python-chess` (GPL-3.0). The project is licensed GPL-3.0 to match.
- **Web server:** FastAPI on uvicorn, using REST for queries and commands and WebSockets for live data.
- **Frontend:** React with TypeScript, built with Vite into static files that the Python server serves. The board is a custom SVG component, not a third-party board library, so that hover highlights, arrows and probability heatmaps are fully under our control. Pieces come from an openly licensed SVG piece set. Charts use a lightweight charting library, chosen when the dashboard is built.
- **External opponent and evaluator:** Stockfish as a separately installed binary, located through configuration and driven over UCI.
- **Data:** Lichess open database dumps (CC0) are the main bulk source. The Lichess puzzle database (CC0) is used for puzzle evaluation.

### Process topology

There are three kinds of long-lived processes, and none depends on another being up:

- **Web server:** always running.
- **Trainer:** one per training run.
- **Evaluator worker:** long-lived.

The **run directory** on disk is the contract between them. The trainer and the evaluator write to it, and the web server reads it and pushes changes to browsers. A restarted web server loses nothing, and past runs stay browsable forever.

When a job is started from the browser, the web server launches it as a **detached subprocess**. The job's state (process identity, status, heartbeat) lives in the run directory, so a web server restart neither kills the job nor loses track of it.

### Modules

1. **Move codec.**
   - Interface: a bidirectional mapping between chess moves and indices in a fixed, flat move vocabulary, plus a function that turns a position into a legal-move mask.
   - The vocabulary contains every geometrically possible (from-square, to-square, promotion piece) combination, on the order of 1,900 entries.
   - Castling is encoded as a two-square king move, as in UCI.
   - The codec handles mirroring when the board is oriented from the side to move.
   - All models, the trainer, the evaluator and the inference engine share this one vocabulary.
2. **Position view.**
   - Interface: position (plus move history) → a serializable snapshot.
   - The snapshot contains:
     - piece placement
     - side to move
     - check status and the checked king's square
     - last move
     - game-over state, result and reason
     - the full list of legal moves, grouped by origin square, with each move flagged as capture, castling, en passant, promotion and/or giving check
   - This is the only representation of a position the browser ever receives. The browser has **no rules engine**: the server is the single source of truth, and hover highlighting is instant because the legal moves arrive with every position.
3. **Game session and players.**
   - A session holds one game on the server, drives it between two players, validates submitted moves through python-chess, supports takebacks, resignation and aborts, and emits events (position changed, game over) to any number of subscribers.
   - Players implement one asynchronous interface: given the current game context, return a move, optionally with a "thoughts" payload (candidate moves with probabilities, a win/draw/loss estimate) for the UI overlay.
   - Implementations:
     - *human*: waits for a move submitted over the WebSocket
     - *random mover*
     - *model*: inference engine plus a move-selection strategy
     - *Stockfish*
   - Search-based players (MCTS, alpha-beta) will later implement the same interface by wrapping the inference engine. Move selection is kept separate from the network precisely so that search can replace it without changing anything else.
   - Finished games are saved as PGN.
4. **Match runner.**
   - Interface: (player A, player B, number of games, opening set) → per-game results and PGNs.
   - Alternates colors and draws starting positions from a curated set of openings, so that deterministic players still produce varied games.
   - Used by the sample-game, Stockfish-ladder and tournament suites.
5. **Dataset builder.**
   - Interface: (PGN sources, filters, output location) → a versioned dataset with a manifest recording sources, filters, counts, statistics and creation time.
   - Streams plain and `.zst` PGN.
   - Filters: minimum rating of the player to move, time-control classes, excluded termination types, date range, maximum number of games. No filters means everything.
   - The train/validation split is decided per game by a deterministic hash of the game's identity.
   - Records are compact and encoder-independent:
     - *per position:* packed board state, castling and en-passant state, move clocks, the index of the move played, the mover's and opponent's ratings with unknown flags, the rating source, the time-control class, the game result from the mover's perspective, the ply number, and a game reference
     - *per game:* the move-index sequence, which sequence models need
   - Records are stored in sharded, memory-mappable files, roughly 50–100 bytes per position.
   - A separate helper downloads Lichess monthly dumps.
6. **Encoders.**
   - Interface: a batch of dataset records (or live positions at inference time) → a model input bundle, plus an **encoder spec** describing its shapes.
   - The bundle has up to three parts:
     - `spatial`: shape [batch, channels, 8, 8]
     - `globals`: shape [batch, features]
     - `sequence`: move tokens, only for sequence encoders
   - The board-planes encoder supports an optional position history and an optional orientation from the side to move.
   - The global features are:
     - side to move
     - castling rights
     - en passant availability
     - halfmove clock
     - the mover's rating and the opponent's rating, each normalized by a configurable scale (default: divided by 5000)
     - rating-unknown flags
   - The sequence encoder uses the move vocabulary, plus special tokens, as its token set.
   - Encoding runs in data-loader workers.
7. **Model zoo.**
   - Interface: a registry mapping an architecture name to a constructor taking (encoder spec, hyperparameters). The constructed module maps an input bundle to policy logits over the move vocabulary and win/draw/loss logits.
   - Initial architectures, and how each consumes the global features:

     | Architecture | Structure | Global features |
     |---|---|---|
     | MLP | Flattened input into dense layers | Concatenated to the flattened input |
     | ResNet | AlphaZero/Maia-style residual tower | Broadcast planes or feature-wise (FiLM) conditioning |
     | Square transformer | 64 square tokens with self-attention | Additional tokens |
     | GPT | Decoder-only transformer over move tokens | Prefix tokens |
8. **Trainer.**
   - Interface: an experiment config file → a run directory.
   - The config fully specifies dataset, encoder, architecture, hyperparameters, optimizer and schedule, evaluation suites, checkpoint policy, seed, and an optional checkpoint to initialize from.
   - Loss: policy cross-entropy plus a configurable weight × value cross-entropy. The value target is the game result from the mover's perspective.
   - Optimizer: AdamW with warmup and cosine decay by default, with gradient clipping.
   - Checkpoints contain the weights, optimizer state, step, config and encoder spec.
   - *Resume* restores full state. *Initialize from* loads weights only, into a new run, which is how fine-tuning works.
   - Stopping is graceful: on a stop signal, the trainer saves a checkpoint and exits.
   - The trainer computes validation metrics itself (policy loss, value loss, top-1 and top-5 accuracy, illegal-top-move rate).
   - It reports hardware statistics through NVML in its heartbeat.
9. **Run store.**
   - Interface: read and write operations over run directories. This is the single place that knows the on-disk layout.
   - Contents:
     - config snapshot and code version
     - an append-only metrics log
     - a status heartbeat, including throughput and GPU statistics
     - checkpoints with retention
     - evaluation results per checkpoint and suite
     - saved games
     - run notes and tags
   - Writers use atomic replacement. Readers tolerate a run that is being written concurrently.
   - The web server detects changes by polling or file watching and pushes them to subscribed browsers.
10. **Inference engine.**
    - Interface: (checkpoint, position with history, desired rating) → a probability distribution over **legal** moves (illegal moves masked before normalizing) and a win/draw/loss estimate.
    - Supports batching for evaluation throughput.
    - Separately reports the probability mass the raw network put on illegal moves, which feeds the illegal-move metrics.
    - Move-selection strategies: argmax, and temperature sampling.
11. **Stockfish adapter.**
    - Interface: a player at a requested Elo (using Stockfish's calibrated strength limit), and an analysis function (evaluation, best move) for a position.
    - Manages the engine process lifecycle.
12. **Rating math.**
    - Interface: match results → rating estimates with confidence intervals.
    - Covers two cases:
      - maximum-likelihood Elo against opponents of known rating (the Stockfish ladder)
      - relative ratings for round-robin tournaments, optionally anchored to a known rating
    - Pure functions, independent of chess.
13. **Evaluator worker.**
    - Interface: watches the run store for checkpoints that lack results for their configured suites, runs those suites, and writes the results back.
    - Suites:
      - *sample games:* streamed live, with the checkpoint playing itself or Stockfish
      - *probe positions:* a curated, versioned set of FENs with labels
      - *Stockfish ladder:* levels configurable
      - *puzzles:* a solve means matching every one of the solver's moves in the solution line, with the opponent's replies given
      - *tournaments*
    - Re-evaluation can be requested on demand. The worker runs model inference on the GPU with small batches, or on the CPU if configured.
14. **Job control.**
    - Interface: list saved experiment configs; start a training run, resume a run, start a fine-tuning run from a checkpoint, start or trigger evaluations, stop a job.
    - Launches detached subprocesses and enforces at most one GPU training job at a time.
    - A read-only configuration switch disables every mutating endpoint.
15. **Web server.**
    - All routes (pages, static assets, REST, WebSockets) are mounted under the configured path prefix, so nginx passes paths through unchanged and the same URLs work with or without the proxy.
    - REST covers: runs, checkpoints, metrics, evaluation results, datasets, saved configs, saved games, and job-control commands.
    - WebSocket channels cover: a game session (snapshots, events, human moves, takebacks) and a run's live stream (new metrics, status changes, new evaluation results, sample games).
    - Game sessions live on the server, so any number of browsers can watch the same game, and reconnecting clients receive the full current state.
    - No authentication. Access control is left to the network and nginx, supported by read-only mode.
16. **React frontend.**
    - Views: the game view (board, turn indicator, move list, player selection, promotion picker, flip, takeback, "thinking" overlay, evaluation bar, PGN export), the replay viewer, the runs dashboard (list, charts with overlays, status and hardware), run detail (probe positions with a checkpoint scrubber, evaluation charts, sample games), datasets, and job control.
    - The build uses relative asset paths, so the same build works under any prefix. The prefix is provided at runtime by the server.
17. **CLI.**
    - One entry point with subcommands:
      - download Lichess data
      - build a dataset
      - show dataset statistics
      - train, resume and fine-tune
      - run the evaluator worker or a one-off evaluation
      - serve the web app
    - Global configuration comes from a config file, overridable by environment variables.

### Specific interactions

- **Hover:** hovering over a piece of the side to move highlights all its legal destinations from the latest snapshot, drawing captures, castling and en passant distinctly. Hovering over the opponent's pieces, or during a non-human player's turn, highlights nothing.
- **Promotion:** dropping a pawn on the last rank opens a four-piece picker before the move is submitted. Cancelling returns the pawn.
- **Move submission:** a human move is sent over the game WebSocket. The server validates it and broadcasts the new snapshot. Rejected moves return an error, and the board snaps back.
- **Live runs:** opening a run subscribes to its stream. The server sends a full initial state, then incremental updates.

## Testing Decisions

### What makes a good test here

- Tests exercise **external behavior through each module's public interface**: given inputs, check outputs or observable effects. They do not assert on internal helpers, private state or call sequences. A refactor that keeps behavior the same must not break any test.
- Prefer real collaborators over mocks. Use small real fixtures: hand-written PGN snippets, FEN positions with known properties, tiny datasets and tiny models. Mock only at genuine process boundaries, such as a missing Stockfish binary.
- The suite runs on CPU in well under a few minutes. Tests that need the GPU or a Stockfish binary are marked and skipped when those are unavailable.
- Tests are deterministic, with fixed seeds.

### Modules under test

- **Move codec:** every legal move in a large sample of positions (from real games and random playouts) round-trips through the codec. The legal-move mask matches python-chess's legal moves exactly. Mirroring is an involution and agrees with the mirrored board.
- **Position view:** hand-picked FEN fixtures for:
  - castling allowed and disallowed (rights lost, path blocked, through check, out of check)
  - en passant available and not available
  - promotion, including capture-promotion
  - pinned pieces
  - check, checkmate and stalemate
  - each game-over reason
- **Game session and match runner:** scripted and random players play complete games. Checks: illegal submissions are rejected without changing state, takebacks restore exact prior state, subscribers receive consistent event streams, colors alternate, openings vary, and saved PGN replays to the same final position.
- **Dataset builder:** small fixture PGN files, including malformed games, missing ratings, various time controls and terminations, and a `.zst` fixture. Checks: filters include and exclude correctly, the split is by game with no leakage, counts and manifest are correct, and malformed games are skipped and counted.
- **Encoders:** hand-verified positions produce expected planes and global features. Side-to-move orientation mirrors correctly. Rating normalization and unknown flags are right. The shapes in the encoder spec match the actual output. The sequence encoder round-trips.
- **Model zoo:** every registered architecture, built from each compatible encoder spec, returns correctly shaped policy and value outputs. Each can memorize a handful of positions to near-zero loss in a few CPU steps, which catches wiring errors.
- **Trainer:** a tiny run on a tiny dataset produces the expected run-directory artifacts. Stop-and-resume gives the same result as an uninterrupted run with the same seed. Initialize-from loads weights into a new run.
- **Run store:** write/read round-trips. Readers tolerate a concurrently written metrics log. Checkpoint retention keeps the right files.
- **Rating math:** synthetic match results with known true ratings are recovered within the reported confidence intervals. Edge cases (all wins, all losses, few games) behave sensibly.
- **Web API:** FastAPI's test client, run *under a non-empty path prefix*: REST endpoints, game-session WebSocket flows (subscribe, human move, rejection, takeback, reconnect), and read-only mode rejecting mutating calls.
- **Board UI:** Vitest with React Testing Library, fed position-view snapshots:
  - hover highlights exactly the legal destinations for castling, en passant and capture cases
  - no highlights for the opponent's pieces
  - the promotion picker appears and submits the chosen piece
  - the turn indicator and the flipped orientation are correct

### Prior art

None yet: this is a new repository. The first milestone establishes the conventions (pytest layout and fixtures, the FastAPI test client for HTTP and WebSocket tests, Vitest with React Testing Library for components), and later modules follow them.

## Out of Scope

- **Chess variants.** Only standard chess rules (no Chess960, no variants).
- **Search.** MCTS or alpha-beta search around the network. The player interface is designed for it, but implementing it belongs to a later PRD.
- **Self-play reinforcement learning.** Training is supervised next-move prediction on human games only.
- **A UCI engine front-end for our models,** for use in external GUIs or bot accounts. Wanted later, not now.
- **Online play** as a bot on Lichess or chess.com.
- **A chess.com API downloader.** chess.com exports are imported as ordinary PGN files.
- **Authentication and user accounts.** Access control is left to the network and nginx, supported by read-only mode.
- **Chess clocks and timed human games.**
- **Editing experiment configs in the browser.** The browser starts jobs from saved configs only.
- **Multi-GPU, distributed or cloud training,** and automated hyperparameter sweeps.
- **Opening books or endgame tablebases** assisting the model during play.
- **A dedicated mobile layout.** The UI should remain usable at reasonable widths, but it is designed for desktop browsers.

## Further Notes

### Suggested delivery order

Each step is a thin, end-to-end vertical slice and maps naturally to one or a few GitHub issues:

1. **Board and live game:**
   - project skeleton (uv, FastAPI under a path prefix, React build served by Python)
   - position view and SVG board with the turn indicator and legal-move hover (castling, en passant, promotion picker)
   - server-held game sessions with human and random players streaming over WebSocket
2. **Game view polish:** move list, last-move and check highlights, flip board, takebacks, custom FEN start, PGN export and saving, and the replay viewer.
3. **Datasets:** PGN and `.zst` import, filters, split by game, manifest and statistics, and the Lichess download helper.
4. **First model end to end:** move codec, board-planes encoder with global features, MLP baseline, trainer with run store, and CLI training.
5. **Runs dashboard:** run list, live metric charts with overlays, and status with GPU statistics.
6. **Play the model:** inference engine, model player with checkpoint and rating choice, and the "thinking" overlay and evaluation bar.
7. **Evaluator foundation:** Stockfish adapter and player, evaluator worker, sample games streamed live, and probe positions with the checkpoint scrubber.
8. **Strength measurement:** match runner with the opening set, rating math, the Stockfish ladder, puzzles, and tournaments.
9. **More architectures:** ResNet, square transformer, then the sequence encoder and GPT.
10. **Curriculum and control:** fine-tuning from a checkpoint, plus job control and read-only mode in the browser.

### Known caveats and considerations

- **Data scale.** Thousands of games are enough to exercise the pipeline, but published move-prediction models were trained on millions (for example, Maia). The Lichess monthly dumps each contain tens of millions of games, so the dataset builder must stream, and datasets may reach tens of gigabytes. Filtering and maximum-game limits keep experiments manageable.
- **Rating pools differ.** chess.com, Lichess and FIDE ratings are not on the same scale, and online ratings already exceed 3000 in fast time controls. The rating source is recorded per game so that experiments can account for this. Dividing by 5000 keeps all realistic values comfortably in range.
- **The Stockfish ladder has a floor.** Stockfish's calibrated strength limit only goes down to about 1320 Elo. Early, weak checkpoints fall below the ladder, and their estimates will be extrapolated or reported as "below floor". The random mover and earlier checkpoints can serve as additional low anchors.
- **Deterministic play.** Argmax play is deterministic, so without varied openings a match would repeat one game, and its results would be meaningless. The opening set in the match runner addresses this.
- **GPU sharing.** Evaluation inference is small next to training, but the evaluator can be configured to use the CPU only if it measurably slows training.
- **WSL2.** Training runs under WSL2 with CUDA passthrough. The web server's bind address must be reachable from wherever nginx runs.
- **Proxying WebSockets.** nginx must forward WebSocket upgrade headers for the live features. The README documents the required settings.
