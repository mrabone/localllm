import logging
import uuid
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from common.db_pool import is_pool_healthy
from server.chat import ChatSession
from server.config import settings
from server.memory import create_session, session_exists
from server.services import (
    McpSessionDep,
    McpToolsDep,
    OllamaClientDep,
    PgDsnDep,
    ServiceContainer,
    ThreadPoolDep,
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
async def create_new_session(pg_dsn: PgDsnDep) -> CreateSessionResponse:
    """Create a new conversation session and return its ID."""
    session_id = create_session(pg_dsn)
    logger.info("Created session %s.", session_id)
    return CreateSessionResponse(session_id=session_id)


@app.head("/sessions/{session_id}", status_code=200)
async def check_session_exists_endpoint(
    session_id: uuid.UUID, pg_dsn: PgDsnDep
) -> None:
    """Check whether a session exists without loading its history.

    Returns 200 if found, 404 if not.  Used by the CLI at startup to validate
    a cached session ID without paying the cost of fetching full history.
    """
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: uuid.UUID, pg_dsn: PgDsnDep, mcp_session: McpSessionDep
) -> SessionHistoryResponse:
    """Return the stored memories for an existing session."""
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    memories_result = await mcp_session.call_tool(
        "load_long_term_memory", arguments={"session_id": str(session_id)}
    )
    memories_text = memories_result.content[0].text if memories_result.content else ""
    messages = (
        [MessageResponse(role="system", content=memories_text)] if memories_text else []
    )
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.post("/sessions/{session_id}/chat")
async def chat(
    session_id: uuid.UUID,
    body: ChatRequest,
    pg_dsn: PgDsnDep,
    mcp_session: McpSessionDep,
    ollama_client: OllamaClientDep,
    mcp_tools: McpToolsDep,
    thread_pool: ThreadPoolDep,
) -> EventSourceResponse:
    """Send a message and stream the assistant response as SSE.

    Always returns an EventSourceResponse so the client always receives
    text/event-stream, even on error. Events:
      - ``token``  — one text fragment from the model.
      - ``done``   — signals the end of the stream (data: "[DONE]").
      - ``error``  — sent if an exception occurs at any point.
    """

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            if not session_exists(pg_dsn, session_id):
                yield {"event": "error", "data": "Session not found."}
                return

            session = ChatSession(
                session_id=session_id,
                mcp_session=mcp_session,
                ollama_client=ollama_client,
                thread_pool=thread_pool,
                mcp_tools=mcp_tools,
            )

            token_stream = await session.chat(body.message)

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
    Ollama, and the MCP server are all reachable.  Returns 200 when healthy,
    503 when any dependency is unavailable.
    """
    if ServiceContainer.get_or_none() is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "unavailable", "detail": "services not yet initialised"},
        )

    checks: dict[str, str] = {}
    healthy = True

    try:
        if is_pool_healthy():
            checks["postgres"] = "ok"
        else:
            checks["postgres"] = "error: pool unavailable"
            healthy = False
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"
        healthy = False

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:
        checks["ollama"] = f"error: {exc}"
        healthy = False

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{settings.mcp_server_url}/health")
            response.raise_for_status()
        checks["mcp"] = "ok"
    except Exception as exc:
        checks["mcp"] = f"error: {exc}"
        healthy = False

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
