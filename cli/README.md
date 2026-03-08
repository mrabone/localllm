# CLI

## Session Management

### Sessions Registry

Sessions are stored in a JSON registry at `~/.localllm_sessions.json` (configurable via `SESSIONS_REGISTRY`). The registry maps human-readable session names to UUIDs (these UUIDs are used as Mem0 user IDs on the server):

```json
{
  "work": "550e8400-e29b-41d4-a716-446655440000",
  "project-x": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "_cache": {
    "work": {"validated_at": 1740000000.0},
    "project-x": {"validated_at": 1740000001.0}
  }
}
```

### Registry Structure

- **Session entries** — keys are human-readable session names (e.g. `"work"`, `"project-x"`), values are UUID strings. These sessions persist across CLI runs.
- **`_cache` key** — Reserved key (never used as a session name). Stores per-session validation timestamps to implement the TTL cache and avoid redundant server calls.
 - **`_cache` key** — Reserved key (never used as a session name). Stores per-session validation timestamps to implement the TTL cache and avoid redundant server calls.

Note: The UUID strings stored in this registry are directly used as Mem0 user IDs by the server; they identify which semantic memories belong to which CLI session.

## TTL Cache

The `_cache` key stores validation timestamps so that the CLI avoids a server round-trip on every startup. When you launch the CLI with `--session <name>`:

1. If the session is in the cache and the TTL is still fresh, trust the local UUID without calling the server.
2. If the TTL has expired, make a lightweight `HEAD` request to re-validate the session is still alive on the server.
3. If the session is missing or the server confirms it's gone, create a new session and update the registry.

The TTL is configurable via `SESSION_CACHE_TTL` (default: 300 seconds). Set to `0` to always validate on startup.

## Session Lifecycle

### Named Sessions (`localllm --session <name>`)

1. **First run** — if `<name>` does not exist:
   - Create a new session on the server
   - Store the UUID in the registry under `<name>`
   - Update the cache timestamp
   - Return to the REPL

2. **Subsequent runs** — if `<name>` exists:
   - Check if the cache is fresh (within TTL)
   - If fresh: return the UUID immediately (zero network latency)
   - If stale: `HEAD /sessions/{uuid}` to re-validate
   - If server confirms it's alive: update the cache, return the UUID
   - If server says it's gone (404): create a replacement session, update the registry

### Anonymous Sessions (`localllm` with no `--session`)

- Always create a fresh session on the server
- Do **not** store it in the registry
- Print a hint showing how to save it for later or resume previous sessions
- Return to the REPL with that temporary session ID

## Performance

- **First run with a named session**: 1 `POST /sessions` + file I/O (~50–100ms network + local I/O)
- **Subsequent runs with cache hit**: pure local file I/O (~1–5ms), no network call
- **Subsequent runs with cache miss**: 1 `HEAD /sessions/{id}` (~15–50ms network), no message history loaded
- **Anonymous session**: 1 `POST /sessions` + message print (~50–100ms network)

The lightweight `HEAD` endpoint avoids the old bottleneck of fetching the entire message history just to confirm the session exists.
