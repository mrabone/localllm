import asyncio
import logging
import uuid
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from common.db_pool import is_pool_healthy
from server.chat import run_chat_graph
from server.config import settings
from server.memory import create_session, session_exists
from server.services import (
    FunctionCallingModelDep,
    McpSessionPoolDep,
    McpToolsDep,
    MemoryHttpClientDep,
    OllamaClientDep,
    PgDsnDep,
    ServiceContainer,
    lifespan,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="LocalLLM Server", lifespan=lifespan)


class CreateSessionResponse(BaseModel):
    session_id: uuid.UUID


class MessageResponse(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: uuid.UUID
    messages: list[MessageResponse]


class ChatRequest(BaseModel):
    message: str


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
def create_new_session(pg_dsn: PgDsnDep) -> CreateSessionResponse:
    """Create a new conversation session and return its ID."""
    session_id = create_session(pg_dsn)
    logger.info("Created session %s.", session_id)
    return CreateSessionResponse(session_id=session_id)


@app.head("/sessions/{session_id}", status_code=200)
def check_session_exists_endpoint(session_id: uuid.UUID, pg_dsn: PgDsnDep) -> None:
    """Check whether a session exists without loading its history.

    Returns 200 if found, 404 if not.  Used by the CLI at startup to validate
    a cached session ID without paying the cost of fetching full history.
    """
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: uuid.UUID,
    pg_dsn: PgDsnDep,
    memory_http_client: MemoryHttpClientDep,
) -> SessionHistoryResponse:
    """Return the stored memories for an existing session."""
    exists = await asyncio.to_thread(session_exists, pg_dsn, session_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found.")

    response = await memory_http_client.get(f"/memory/long-term/{session_id}")
    response.raise_for_status()
    memories_text = response.json().get("content", "")
    messages = (
        [MessageResponse(role="system", content=memories_text)] if memories_text else []
    )
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.post("/sessions/{session_id}/chat")
async def chat(
    session_id: uuid.UUID,
    body: ChatRequest,
    pg_dsn: PgDsnDep,
    mcp_pool: McpSessionPoolDep,
    memory_http_client: MemoryHttpClientDep,
    ollama_client: OllamaClientDep,
    mcp_tools: McpToolsDep,
    function_calling_model: FunctionCallingModelDep,
) -> EventSourceResponse:
    """Send a message and stream the assistant response as SSE.

    The session existence check happens before the EventSourceResponse is
    created so that a missing session returns a proper HTTP 404 rather than
    an SSE error event.  An MCP session is checked out from the pool for the
    duration of the request and returned when the stream completes.

    Events:
      - ``token``  — one text fragment from the model.
      - ``done``   — signals the end of the stream (data: "[DONE]").
      - ``error``  — sent if an exception occurs at any point.
    """
    if not await asyncio.to_thread(session_exists, pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            async with mcp_pool.acquire() as mcp_session:
                token_stream = run_chat_graph(
                    session_id=session_id,
                    user_input=body.message,
                    mcp_session=mcp_session,
                    memory_http_client=memory_http_client,
                    ollama_client=ollama_client,
                    chat_model=settings.server_ollama_model,
                    function_calling_model=function_calling_model,
                    mcp_tools=mcp_tools,
                )

                async for token in token_stream:
                    yield {"event": "token", "data": token}

            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:
            logger.exception("Error during chat for session %s - %s", session_id, exc)
            yield {
                "event": "error",
                "data": str(exc) or "An unknown server error occurred.",
            }

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for container orchestration.

    Verifies that the service container is initialised and that PostgreSQL,
    Ollama, and the MCP server are all reachable.  The three dependency checks
    run in parallel to minimise latency.  Returns 200 when healthy, 503 when
    any dependency is unavailable.
    """
    if ServiceContainer.get_or_none() is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "unavailable", "detail": "services not yet initialised"},
        )

    async def _check_postgres() -> str:
        try:
            if is_pool_healthy():
                return "ok"
            return "error: pool unavailable"
        except Exception as exc:
            return f"error: {exc}"

    async def _check_ollama() -> str:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(f"{settings.ollama_base_url}/api/tags")
                response.raise_for_status()
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    async def _check_mcp() -> str:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(f"{settings.mcp_server_url}/health")
                response.raise_for_status()
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    postgres_result, ollama_result, mcp_result = await asyncio.gather(
        _check_postgres(), _check_ollama(), _check_mcp()
    )

    checks = {
        "postgres": postgres_result,
        "ollama": ollama_result,
        "mcp": mcp_result,
    }
    healthy = all(v == "ok" for v in checks.values())

    if not healthy:
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "checks": checks},
        )

    return {"status": "ok", "checks": checks}


def main() -> None:
    uvicorn.run(
        "server.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
    )


if __name__ == "__main__":
    main()
