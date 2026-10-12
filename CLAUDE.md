# chess-ai

The design is in the PRD, [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md) (GitHub issue #1). The work is split into GitHub issues, each labelled HITL (needs human review) or AFK.

## Git and GitHub: the user does it

- Never create branches or commits, and never push, open or edit pull requests, merge, or create or change issues or labels. The user does all of these by hand.
- The only things you write to GitHub are PR review comments and replies to them (see [Review rounds](#review-rounds)).
- Reading is fine: `git status`/`log`/`diff`/`fetch`, and read-only `gh` commands such as `gh issue view`, `gh pr view` and `gh api` GET requests.
- Leave all your work as uncommitted changes in the working tree, and hand it over with a suggested branch name and commit message (see [Handing over](#handing-over)).

## Working on an issue

- Read the issue and the parts of the PRD it references, for example with `gh issue view N`. Inline review comments are only available through `gh api` (see [Review rounds](#review-rounds)).
- Before changing anything, check with `git fetch` and `git status` that the working tree is clean and on an up-to-date `main`. If it isn't, ask the user.
- Implement, run the [checks](#checks), then stop and hand over.

## Handing over

When an issue's implementation is done, end with suggestions the user can paste:

- a **branch name**, e.g. `issue-3-live-random-game`
- a **commit title** in [Conventional Commits](https://www.conventionalcommits.org/) form: `type(optional scope): lowercase imperative summary`, e.g. `feat: add project skeleton with static board under a path prefix`
- a **commit body**: a short list of what changed, ending with `Closes #N`
- for HITL issues, the points from the issue's "Human review" section, for the PR description

The user squashes each branch into a single commit by hand and merges it into `main`, so the title is also what ends up on `main`.

## Review rounds

The user reviews PRs with inline review comments, then asks for the relevant ones to be fixed.

- Fetch the comments with `gh api repos/alkanen/chess-ai/pulls/N/comments` (and `/reviews`). The user may have replaced the PR with a new one from the same branch or rewritten its commits, so look up the open PR for the branch and `git fetch` first.
- Judge each comment on its merits. Reproduce the claim before changing anything, and write a failing regression test first. Where a comment is wrong, or its suggested fix is, push back with the argument or do something better and explain why.
- Leave the fixes uncommitted. The user makes one commit per review round and pushes it before starting the next round.
- Reply on every thread in the same round, through `gh api -X POST repos/alkanen/chess-ai/pulls/N/comments/ID/replies -f body=...`. Describe what changed (no commit hashes, since the commit doesn't exist yet) or give the reasoning for pushing back. Leave the threads unresolved; the user resolves them.
- Finish by summarizing, per comment, what was changed or pushed back on, and suggest a Conventional Commits title and body for the round's commit.

## Command-line output

Every command sets up its output once, in `chess_ai.output` (called from `cli.main`). New code follows the same patterns:

- **Two channels.** The command's own account of what it is doing (stages, games, progress, results) goes to stdout through a `say` callback (`chess_ai.output.say`, or a `say: Say` parameter that library code takes). Problems go to stderr through `logging`: `LOGGER = logging.getLogger(__name__)`, then `LOGGER.warning(...)`. Never `print`, and never `say(..., err=True)` for a warning. `err=True` is only for the line saying how a command ended ("interrupted", "evaluator stopped").
- **No prefix in log messages.** Write `LOGGER.warning("cannot lock %s: %s", path, e)`. The handler adds `chess-ai: warning: ` (and the logger name at info and debug). `say` lines start with `chess-ai: ` themselves when they are narration, but not when they are a result or a listing.
- **Every `say` line has a kind**, which is what `-q`/`-qq` hide: `kind=DETAIL` for one step of many (a stage starting, a game, a progress line, a checkpoint saved), the default `MILESTONE` for where a long job has got to (validation lines, a header naming what is being played), and `kind=RESULT` for what the command was run for (summaries, a suite's result, a listing, how it ended). `-qq` shows only results, so a query command's output must be `RESULT`.
- **Say when a slow stage starts**, not when it ends: one `DETAIL` line as it begins (`chess-ai: opening dataset NAME`). Every line is timestamped, so the gap to the next line is how long it took. A stage with a known size says how far along it is (`game 31/102`).
- **Plain lines, no redrawing in place.** The one exception is `ProgressPrinter`/`DownloadPrinter`, which rewrite one line on a terminal and write a line every `LOG_INTERVAL` anywhere else. Stamp anything written straight to a stream with `output.stamp(stream)`.
- **Timestamps** come from `output.stamp`/`output.say`, not from the message. They are on by default, but off for the journal (`JOURNAL_STREAM`) and for commands marked `query=True` in `set_defaults` (`runs list`, `runs annotate`, `dataset stats`). `--timestamps=always|never` overrides both.
- **Short names in per-game lines.** A run name can be a hundred characters long. Name the model(s) in a `MILESTONE` header (`RUN@30000 sample games 3–22 of 102: model against Stockfish 1350`), repeated whenever the opponent changes and otherwise every 20 games, and call them `model` (or `model 1`/`model 2`) in the game lines.
- **Tests** that pass `say=` need a callable that takes `kind=`, e.g. `lambda text, kind=MILESTONE: said.append(text)`. Lines are unstamped in tests (see `tests/conftest.py`). `tests/test_output.py` stamps them on purpose.

## Checks

Run these before handing over an issue or a review round:

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
npm --prefix frontend test
```

Changes to the frontend also need `scripts/build-frontend.sh` (it includes the TypeScript type check) before `uv run chess-ai serve` shows them.
