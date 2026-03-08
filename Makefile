.PHONY: setup dev prod run-cli sync-deps run-rag test test-cli test-server test-rag down

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

test-cli:
	uv run pytest -v cli/tests

test-server:
	uv run pytest -v server/tests

test-rag:
	uv run pytest -v rag/tests

down:
	docker compose down
