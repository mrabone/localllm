PROJECTS_DIRS = cli rag

.PHONY: setup run-cli sync-deps up down run-rag

setup:
	docker compose down && docker compose up

run-cli:
	uv run cli/src/cli/main.py

sync-deps:
	@echo "Syncing workspace dependencies..."
	@$(foreach dir,$(PROJECTS_DIRS), \
		echo "Installing dependencies for $$dir..."; \
		(cd $(dir) && uv pip install -e .); \
	)

run-rag:
	uv run rag/src/rag/main.py
