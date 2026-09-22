#!/usr/bin/env bash
# Builds the React frontend into src/chess_ai/web/static, where `chess-ai serve` finds it.
# Needs Node.js; running the server afterwards needs only Python.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
npm ci
npm run build
