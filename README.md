# LocalLLM

Personal project where I experiment with LLMs locally.

## Architecture

The project is split into five packages in a `uv` workspace:

- **`server/`** — FastAPI HTTP server. Owns all chat logic: session management, conversation memory, RAG context injection, tool calling orchestration, and streaming responses via SSE. Runs in Docker.
- **`mcp/`** — FastMCP tool server. Exposes the RAG `search_knowledge_base` tool over StreamableHTTP and serves memory operations as plain REST endpoints. Holds the dual-memory system (verbatim sliding window in Postgres + Mem0 semantic fact extraction in PGVector) and the RAG knowledge base. Runs in Docker.
- **`cli/`** — Thin terminal REPL. Sends user input to the server and renders the streamed response. Named sessions are stored in a local JSON registry at `~/.localllm_sessions.json` (these UUIDs are used as Mem0 user IDs on the server).
- **`rag/`** — One-shot pipeline that scrapes websites and ingests content into the pgvector store.
- **`common/`** — Shared utilities (config base class, session store, DB pool, structured logging) used by `server`, `mcp`, and `rag`.

### Tool Calling

Each chat turn runs through a **LangGraph state machine** with two distinct model roles:

- **`functiongemma`** (`SERVER_FUNCTION_CALLING_MODEL`) — a dedicated function-calling model that silently decides whether to invoke tools and with what arguments. It never produces user-visible text. Runs with a smaller context window (`SERVER_FUNCTION_CALLING_NUM_CTX`, default 2048) since it only needs to select tools, not generate prose.
- **`custom-chatbot-model`** (`SERVER_OLLAMA_MODEL`) — the main chat model that generates and streams the final answer. It receives the enriched context (history + memories + any tool results) but never sees the tool schemas.

Tool results are injected into the conversation as `role: "system"` messages. Sending them as `role: "tool"` is not supported by gemma3's chat template and causes corrupted output (garbled text, instruction echoing), so this role is used instead.

The only tool currently exposed to `functiongemma` is **`search_knowledge_base`**, which searches the RAG vector store. Memory operations (loading the conversation window, loading long-term facts, persisting messages) are not MCP tools — they are plain REST endpoints on the MCP server called directly by the graph via HTTP. The model never sees them.

The decision loop can repeat up to `SERVER_TOOL_CALL_MAX_LOOPS` times, allowing chained tool calls before the chat model generates its final response.

### Memory Persistence

User and assistant messages are persisted to the MCP server **after** the full streaming response completes, not during the graph run. This prevents Mem0's LLM-based fact extraction from running concurrently with the streaming chat model on a single-GPU Ollama instance (`NUM_PARALLEL=1`), which would corrupt the KV cache. Fact extraction only runs for user-role messages — assistant responses are derived from context and contain no new user facts.

## Features

- **Interactive CLI chat interface** — Have conversations with a local LLM in your terminal, optionally enriched with your own data via RAG.
- **Named sessions** — Save and resume conversations by name with `--session <name>`. Each run without a name starts a fresh session.
- **Knowledge base integration** — Automatically import content from websites to let the assistant answer questions based on your custom data sources.
- **Tool calling** — A dedicated function-calling model silently decides whether to search the knowledge base to ground each answer. The main chat model then receives the enriched context and streams its reply.
- **Dual memory system** — Each turn loads a sliding window of recent verbatim turns (Postgres) and extracted long-term semantic facts (Mem0/PGVector). Memory persists across CLI restarts.
- **Persistent sessions** — Session UUIDs map to Mem0 user IDs, so memories survive restarts and are tied to named sessions.
- **Structured JSON logging** — Both `server` and `mcp` emit structured JSON log lines via a shared `UVICORN_LOG_CONFIG` from `common`.
- **Customizable system prompt** — The assistant has a friendly, helpful personality defined in `Modelfile`.
- **No internet required** — Everything runs locally with Docker and Ollama.
- **Flexible model selection** — Easy to switch between different Ollama models for each role (chat, function calling, embeddings).

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

2. **Start all Docker services** (Ollama, PostgreSQL, MCP server, and the chat server):
   ```bash
   make setup
   ```
   On first run, Ollama will pull the embedding model (`embeddinggemma`) and the function-calling model (`functiongemma`), and build the custom chat model from `Modelfile`. This can take a few minutes.

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

## Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Build and start all Docker services in production mode |
| `make dev` | Rebuild and start `server` and `mcp` in dev mode with hot-reload |
| `make down` | Stop and remove all Docker containers |
| `make sync-deps` | Install / sync all workspace dependencies via `uv sync` |
| `make run-cli [SESSION_ARGS='--session <name>']` | Start the interactive CLI chat REPL. Optionally pass `SESSION_ARGS` to load or create a named session. |
| `make run-mcp` | Run the MCP tool server locally (outside Docker) |
| `make run-rag` | Run the RAG ingestion pipeline |
| `make test` | Run the full test suite with pytest |
| `make test-cli` | Run CLI tests only |
| `make test-mcp` | Run MCP server tests only |
| `make test-server` | Run chat server tests only |
| `make test-rag` | Run RAG pipeline tests only |

## Configuration

Configured via environment variables in a `.env` file. Copy `.env.example` to get started.

### Common

- `OLLAMA_BASE_URL` — Base URL for the Ollama API (default: `http://127.0.0.1:11434`).

### PostgreSQL / Mem0 PGVector

- `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD` — database connection details.
- `PG_COLLECTION_NAME` — table name for pgvector embeddings (used by RAG pipeline and knowledge base search). Note: this variable is not injected into the Docker services, so it only takes effect when running services locally outside Docker (e.g. `make run-mcp`).
- `MEM0_COLLECTION_NAME` — PGVector collection used by Mem0 to store extracted semantic memories (default: `mem0_chat`). Same caveat applies: not passed to Docker services.
- `MEM0_LLM_MODEL` — Ollama model used by Mem0 internally to extract semantic facts from conversations (default: `custom-chatbot-model`).

### RAG Pipeline

- `RAG_OLLAMA_MODEL` — Ollama model used to generate embeddings (also used by Mem0 for vector storage). Must produce 768-dimensional vectors (default: `embeddinggemma`).
- `CONCURRENT_REQUESTS` — number of concurrent requests when scraping websites.
- `REQUEST_DELAY` — delay in seconds between scrape requests.
- `CHUNKER_BREAKPOINT_TYPE` — chunking strategy for the semantic chunker (default: `percentile`).
- `CHUNKER_BREAKPOINT_AMOUNT` — threshold value for the chunker (default: `60.0`).

### MCP Server

- `MCP_HOST` / `MCP_PORT` — host and port the MCP server binds to inside Docker.
- `MCP_ENABLE_RAG` — set to `true` to enable the RAG knowledge base (requires the RAG pipeline to have been run first).
- `MCP_SERVER_URL` — URL the chat server uses to connect to the MCP server.
- `RAG_K` — number of top documents to retrieve per knowledge base query.
- `RAG_MAX_DISTANCE` — maximum cosine similarity distance for a document to be considered relevant.

### Server

- `SERVER_OLLAMA_MODEL` — Ollama model used to generate chat responses (default: `custom-chatbot-model`).
- `SERVER_FUNCTION_CALLING_MODEL` — Ollama model used to decide tool calls (default: `functiongemma`).
- `SERVER_OLLAMA_NUM_CTX` — context window size for the main chat model (default: `8192`).
- `SERVER_FUNCTION_CALLING_NUM_CTX` — context window size for the function-calling model (default: `2048`). Kept smaller than the chat model since tool selection requires less context.
- `SERVER_MCP_POOL_SIZE` — number of persistent MCP client connections kept in the pool (default: `4`).
- `SERVER_MEMORY_WINDOW_SIZE` — number of recent verbatim turns loaded into context per request (default: `10`).
- `SERVER_MEMORY_LONG_TERM_MAX` — maximum number of Mem0 semantic facts injected as a system message (default: `3`).
- `SERVER_TOOL_CALL_MAX_LOOPS` — maximum number of decide→execute tool-call loop iterations before forcing a response (default: `3`).
- `SERVER_HOST` / `SERVER_PORT` — host and port the server binds to inside Docker.

### CLI

- `SERVER_URL` — base URL of the localLLM server (default: `http://127.0.0.1:8000`).
- `SESSIONS_REGISTRY` — path to the JSON file that maps session names to UUIDs (default: `~/.localllm_sessions.json`).
- `SESSION_CACHE_TTL` — seconds a validated session UUID is trusted locally before re-checking with the server (default: `300`). Set to `0` to always validate on startup.

## Development

### Dependency Management

The project is a `uv` workspace with five members: `cli`, `common`, `mcp`, `rag`, and `server`. To install all dependencies into the shared virtual environment:

```bash
make sync-deps
```

### Running Tests

```bash
make test
```

### Changing the Model

There are three distinct model roles, each configurable independently:

| Role | Variable | Default | Purpose |
|------|----------|---------|---------|
| Chat / response | `SERVER_OLLAMA_MODEL` | `custom-chatbot-model` | Generates the user-visible streamed answer |
| Tool calling | `SERVER_FUNCTION_CALLING_MODEL` | `functiongemma` | Decides whether and how to call tools |
| Embeddings | `RAG_OLLAMA_MODEL` | `embeddinggemma` | Generates embeddings for RAG ingestion, retrieval, and Mem0 memory storage |

To swap a model:

1. Find available models on [Ollama's model library](https://ollama.com/library).

2. Pull the model into the running container:
   ```bash
   docker compose exec ollama ollama pull <model-name>
   ```

3. Update the relevant variable in your `.env` file and restart the affected service:
   ```bash
   docker compose restart server   # for SERVER_OLLAMA_MODEL or SERVER_FUNCTION_CALLING_MODEL
   docker compose restart mcp      # for RAG_OLLAMA_MODEL or MEM0_LLM_MODEL
   ```

4. If swapping the embedding model, note that the vector dimensions must match `EMBEDDING_DIMS` (currently hardcoded to `768` in `mcp/src/mcp_server/services.py`). You will also need to re-run the RAG pipeline to regenerate embeddings.
