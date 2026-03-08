.PHONY: setup run-cli sync-deps run-rag test down

setup:
	docker compose down && docker compose up -d

run-cli:
	uv run --package cli python -m cli.main $(SESSION_ARGS)

sync-deps:
	@echo "Syncing workspace dependencies..."
	uv sync

run-rag:
	uv run --package rag python -m rag.main

test:
	uv run pytest -v

down:
	docker compose down
