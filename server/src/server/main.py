import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from server.chat import ChatSession
from server.config import settings
from server.memory import create_session, session_exists
from server.services import McpSessionDep, OllamaClientDep, PgDsnDep, lifespan

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
) -> EventSourceResponse:
    """Send a message and stream the assistant response as SSE.

    Events:
      - ``token``  — one text fragment from the model.
      - ``rag``    — emitted before tokens when RAG context was used.
      - ``done``   — signals the end of the stream (data: "[DONE]").
      - ``error``  — sent if an exception occurs mid-stream.
    """
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    session = ChatSession(
        session_id=session_id,
        mcp_session=mcp_session,
        ollama_client=ollama_client,
    )

    token_stream, rag_used = await session.chat(body.message)

    async def event_generator() -> AsyncGenerator[dict, None]:
        if rag_used:
            yield {"event": "rag", "data": json.dumps({"document_count": None})}

        try:
            async for token in token_stream:
                yield {"event": "token", "data": token}
        except Exception as exc:
            logger.error("Error during streaming for session %s: %s", session_id, exc)
            yield {"event": "error", "data": str(exc)}
            return

        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
    )


if __name__ == "__main__":
    main()
