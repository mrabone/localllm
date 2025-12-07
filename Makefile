setup:
	docker compose down && docker compose up
run:
	uv run main.py
requirements:
	uv pip compile pyproject.toml -o requirements.txt