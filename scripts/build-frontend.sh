#!/usr/bin/env bash
# Builds the React frontend into src/chess_ai/web/static, where `chess-ai serve` finds it.
# Needs Node.js; running the server afterwards needs only Python.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
# npm ci reinstalls from scratch, so only redo it when the lockfile has moved on.
if [ ! -f node_modules/.package-lock.json ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  npm ci
fi
npm run build
