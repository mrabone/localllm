setup:
	docker compose down && docker compose up
run:
	export OLLAMA_HOST="http://172.25.192.1:11434" && uv run main.py