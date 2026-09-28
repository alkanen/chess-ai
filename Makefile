# Development shortcuts. Everything works without make too; see the README.
#
#   make          build the frontend if it is out of date, then serve the app
#   make build    build the frontend if it is out of date
#   make train    train from an experiment config: make train CONFIG=experiments/other.toml
#   make check    run the Python and frontend tests and the linter
#   make clean    remove the built frontend and the installed npm packages

# Which experiment config `make train` runs; override it on the command line.
CONFIG ?= experiments/mlp-baseline.toml

FRONTEND := frontend
PAGE := src/chess_ai/web/static/index.html
SOURCES := $(shell find $(FRONTEND)/src -type f) \
	$(FRONTEND)/index.html $(FRONTEND)/package.json $(FRONTEND)/package-lock.json \
	$(FRONTEND)/tsconfig.json $(FRONTEND)/vite.config.ts

.PHONY: serve build train check clean

serve: build
	uv run chess-ai serve

build: $(PAGE)

$(PAGE): $(SOURCES)
	scripts/build-frontend.sh

train:
	uv run chess-ai train $(CONFIG)

check:
	uv run pytest
	uv run ruff check . && uv run ruff format --check .
	npm --prefix $(FRONTEND) test

clean:
	rm -rf src/chess_ai/web/static $(FRONTEND)/node_modules
