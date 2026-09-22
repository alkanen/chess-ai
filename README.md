# chess-ai

A testbed for training neural-network chess players the way large language models are trained: show the network a position, have it predict the move a human actually played, and repeat over millions of games. The goal is to compare model architectures on equal terms (MLP, ResNet, a transformer over the 64 squares, and a GPT-style model over move sequences) and to watch them learn through a browser UI.

> **Status: planning.** No code yet. The full design is in the PRD: [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md).

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

Running the system will need Python and the built frontend. Node.js is only needed to build the frontend.

## Deployment notes

The web app will serve everything (pages, assets, API and WebSockets) under a configurable path prefix, so nginx can forward requests unchanged. The WebSocket upgrade headers must be forwarded for the live features to work. For example:

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

The app has no authentication. It will have a read-only mode that disables starting and stopping jobs from the browser.

## License

GPL-3.0, matching the python-chess dependency.
