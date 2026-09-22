# chess-ai

The design is in the PRD, [docs/prd/chess-ai-trainer.md](docs/prd/chess-ai-trainer.md) (GitHub issue #1). The work is split into GitHub issues, each labelled HITL (needs human review) or AFK.

## Git and GitHub: the user does it

- Never create branches or commits, and never push, open or edit pull requests, merge, or create or change issues or labels. The user does all of these by hand.
- The only things you write to GitHub are PR review comments and replies to them (see [Review rounds](#review-rounds)).
- Reading is fine: `git status`/`log`/`diff`/`fetch`, and `gh api` GET requests.
- Leave all your work as uncommitted changes in the working tree, and hand it over with a suggested branch name and commit message (see [Handing over](#handing-over)).

## Working on an issue

- Read the issue and the parts of the PRD it references. Use `gh api repos/alkanen/chess-ai/issues/N`, because `gh issue view` fails with older gh versions.
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

## Checks

Run these before handing over an issue or a review round:

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
npm --prefix frontend test
```

Changes to the frontend also need `scripts/build-frontend.sh` (it includes the TypeScript type check) before `uv run chess-ai serve` shows them.
