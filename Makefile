PROJECTS = cli rag

.PHONY: setup run-cli sync-deps generate-requirements up down

setup:
	docker compose down && docker compose up

run-cli:
	uv run cli/main.py

sync-deps:
	@echo "Syncing dependencies from requirements.txt files..."
	@for project in $(PROJECTS); do \
		uv pip sync --quiet $$project/requirements.txt; \
	done

# Generates the requirements.txt lockfiles from pyproject.toml files.
generate-requirements:
	@echo "Generating requirements.txt for all projects..."
	@for project in $(PROJECTS); do \
		uv pip compile --quiet $$project/pyproject.toml -o $$project/requirements.txt -U; \
	done
