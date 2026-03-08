# LocalLLM

Personal project where I experiment with LLMs locally.

## Architecture

The project is split into four packages in a `uv` workspace:

- **`server/`** — FastAPI HTTP server. Owns all chat logic: session management, conversation memory (Mem0 with PGVector), RAG retrieval (pgvector), summarisation, and streaming responses via SSE. Runs in Docker.
- **`cli/`** — Thin terminal REPL. Sends user input to the server and renders the streamed response. Named sessions are stored in a local JSON registry at `~/.localllm_sessions.json` (these UUIDs are used as Mem0 user IDs on the server).
- **`rag/`** — One-shot pipeline that scrapes websites and ingests content into the pgvector store.
- **`common/`** — Shared utilities used by `server` and `rag`.

## Features

- **Interactive CLI chat interface** - Have conversations with a local LLM in your terminal, optionally enriched with your own data via RAG.
- **Named sessions** - Save and resume conversations by name with `--session <name>`. Each run without a name starts a fresh session.
- **Knowledge Base Integration** - Automatically import content from websites to let the assistant answer questions based on your custom data sources.
- **Conversation history management** - Automatically summarizes old messages when the conversation gets long to maintain context.
- **Persistent sessions** - Conversation memory is stored in Mem0 (backed by PGVector in PostgreSQL) and survives CLI restarts. Note: Mem0 stores extracted semantic memories rather than a verbatim transcript.
- **Customizable system prompt** - The assistant has a friendly, helpful personality.
- **No internet required** - Everything runs locally with Docker and Ollama.
- **Flexible model selection** - Easy to switch between different Ollama models.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- `uv` (recommended for dependency management)

### Running Locally

1. **Configure environment variables** — copy the example template and edit as needed:
   ```bash
   cp .env.example .env
   ```

2. **Start all Docker services** (Ollama, PostgreSQL, and the server):
   ```bash
   make setup
   ```
   On first run, Ollama will pull the embedding model and build the custom chat model from `Modelfile`. This can take a few minutes.

3. **Install local dependencies:**
   ```bash
   make sync-deps
   ```

4. *(Optional)* **Run the RAG pipeline** to populate the knowledge base:
   ```bash
   make run-rag
   ```

5. **Start chatting:**
   ```bash
   make run-cli
   ```
   The CLI connects to the server running in Docker and streams responses to your terminal.

   On first run (anonymous session), the application will print instructions for saving and resuming sessions.

   To resume or create a named session:
   ```bash
   make run-cli SESSION_ARGS='--session work'
   ```
   Replace `work` with any session name. The first run with a new name creates a new session; subsequent runs resume it.

### Stopping the Services

```bash
make down
```

## Using Named Sessions

Sessions are automatically persisted and can be resumed across CLI runs. Each session maps to a Mem0 user (the UUID stored in your local sessions registry) and maintains its own semantic memories in Mem0/PGVector.

**Anonymous sessions (no `--session` flag):**
- Create a fresh session each time you run the CLI
- Conversation history is stored but not easily accessible
- You'll be prompted with instructions on how to save it if needed

**Named sessions (with `--session <name>`):**
- Save and resume conversations by name
- First run with a new name creates the session
- Subsequent runs resume the same conversation
- Session list is printed when you start an anonymous session

**Examples:**
```bash
# Start a fresh session
make run-cli

# Create and use a "work" session
make run-cli SESSION_ARGS='--session work'

# Resume the "work" session later
make run-cli SESSION_ARGS='--session work'

# Create a "project-x" session
make run-cli SESSION_ARGS='--session project-x'
```

All sessions are stored in `~/.localllm_sessions.json` and cached for fast startup (default: 300 seconds). These UUIDs are used as Mem0 user IDs on the server. See `cli/README.md` for detailed documentation on session management, caching, and performance.

### Stopping the Services

```bash
make down
```

## Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Bring all Docker services up in the background (detached) |
| `make down` | Stop and remove all Docker containers |
| `make sync-deps` | Install / sync all workspace dependencies via `uv sync` |
| `make run-cli [SESSION_ARGS='--session <name>']` | Start the interactive CLI chat REPL. Optionally pass `SESSION_ARGS` to load or create a named session. |
| `make run-rag` | Run the RAG ingestion pipeline |
| `make test` | Run the full test suite with pytest |

## Configuration

Configured via environment variables in a `.env` file. Copy `.env.example` to get started.

### Common

- `OLLAMA_BASE_URL` — Base URL for the Ollama API (default: `http://127.0.0.1:11434`).

### PostgreSQL / Mem0 PGVector

- `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD` — database connection details.
- `PG_COLLECTION_NAME` — table name for pgvector embeddings (used by RAG pipeline).
- `MEM0_COLLECTION_NAME` — collection name used by Mem0 to store chat memories in the same Postgres/PGVector instance.

### RAG Pipeline

- `RAG_OLLAMA_MODEL` — Ollama model used to generate embeddings.
- `CONCURRENT_REQUESTS` — number of concurrent requests when scraping websites.
- `REQUEST_DELAY` — delay in seconds between scrape requests.

### Server

- `SERVER_OLLAMA_MODEL` — Ollama model used for chat responses.
- `SERVER_MAX_RECENT` — number of recent messages to retain before summarising.
- `SERVER_THRESHOLD` — total message count that triggers summarisation.
- `SERVER_ENABLE_RAG` — set to `true` to enable RAG context injection (requires the RAG pipeline to have been run first).
- `SERVER_RAG_MAX_DISTANCE` — maximum similarity distance for a document to be considered relevant.
- `SERVER_RAG_K` — number of top documents to retrieve per query.
- `SERVER_HOST` / `SERVER_PORT` — host and port the server binds to inside Docker.

### CLI

- `SERVER_URL` — base URL of the localLLM server (default: `http://127.0.0.1:8000`).
- `SESSIONS_REGISTRY` — path to the JSON file that maps session names to UUIDs (default: `~/.localllm_sessions.json`).
- `SESSION_CACHE_TTL` — seconds a validated session UUID is trusted locally before re-checking with the server (default: `300`). Set to `0` to always validate on startup.

## Development

### Dependency Management

The project is a `uv` workspace with four members: `cli`, `common`, `rag`, and `server`. To install all dependencies into the shared virtual environment:

```bash
make sync-deps
```

### Running Tests

```bash
make test
```

### Changing the Model

1. Find available models on [Ollama's model library](https://ollama.com/library).

2. Pull the model into the running container:
   ```bash
   docker compose exec ollama ollama pull <model-name>
   ```

3. Update `SERVER_OLLAMA_MODEL` in your `.env` file:
   ```
   SERVER_OLLAMA_MODEL=<model-name>
   ```

4. Restart the server container to pick up the change:
   ```bash
   docker compose restart server
   ```
