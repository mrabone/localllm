PROJECTS_DIRS = cli rag

.PHONY: setup run-cli sync-deps up down run-rag

setup:
	docker compose down && docker compose up

run-cli:
	uv run cli/src/cli/main.py

sync-deps:
	@echo "Syncing workspace dependencies..."
	uv sync

run-rag:
	uv run rag/src/rag/main.py
