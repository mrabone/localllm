.PHONY: setup dev prod run-cli sync-deps run-rag test down

dev:
	docker compose down
	SERVER_BUILD_TARGET=dev SERVER_RELOAD=true \
	docker compose up -d --build server

prod:
	SERVER_BUILD_TARGET=runtime \
	docker compose up -d --build

setup: prod

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
