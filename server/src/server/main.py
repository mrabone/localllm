import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from server.chat import ChatSession
from server.config import settings
from server.memory import create_session, load_messages, session_exists
from server.services import Mem0Dep, OllamaClientDep, PgDsnDep, RagStoreDep, lifespan

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
def check_session_exists(session_id: uuid.UUID, pg_dsn: PgDsnDep) -> None:
    """Check whether a session exists without loading its memories.

    Returns 200 if found, 404 if not.  Used by the CLI at startup to validate
    a cached session ID without paying the cost of fetching full history.
    """
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(
    session_id: uuid.UUID, mem0: Mem0Dep, pg_dsn: PgDsnDep
) -> SessionHistoryResponse:
    """Return the stored memories for an existing session."""
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = load_messages(mem0, session_id, query="")
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[MessageResponse(**m) for m in messages],
    )


@app.post("/sessions/{session_id}/chat")
def chat(
    session_id: uuid.UUID,
    body: ChatRequest,
    mem0: Mem0Dep,
    ollama_client: OllamaClientDep,
    rag_store: RagStoreDep,
    pg_dsn: PgDsnDep,
) -> EventSourceResponse:
    """Send a message and stream the assistant response as SSE.

    Events:
      - ``token``  — one text fragment from the model.
      - ``rag``    — JSON object with ``document_count`` (only sent when RAG
                     retrieved context).
      - ``done``   — signals the end of the stream (data: "[DONE]").
      - ``error``  — sent if an exception occurs mid-stream.
    """
    if not session_exists(pg_dsn, session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    session = ChatSession(
        session_id=session_id,
        mem0=mem0,
        ollama_client=ollama_client,
        pg_dsn=pg_dsn,
        pgvector_store=rag_store,
    )

    token_stream, rag_result = session.chat(body.message)

    async def event_generator() -> AsyncGenerator[dict, None]:
        # Send RAG metadata first if context was retrieved.
        if rag_result is not None:
            yield {
                "event": "rag",
                "data": json.dumps({"document_count": rag_result.document_count}),
            }

        try:
            # The synchronous Ollama generator uses httpx streaming internally.
            # We run it in a thread and feed tokens into an asyncio.Queue so
            # the async side only ever awaits queue.get(), keeping the sync
            # and async worlds cleanly separated.
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _done = object()

            def _producer() -> None:
                try:
                    for token in token_stream:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except RuntimeError as exc:
                    # PEP 479 converts StopIteration raised inside a generator
                    # into RuntimeError. Treat this as a normal end-of-stream
                    # rather than a real error.
                    if "StopIteration" not in str(exc):
                        loop.call_soon_threadsafe(queue.put_nowait, exc)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _done)

            loop.run_in_executor(None, _producer)

            while True:
                item = await queue.get()
                if item is _done:
                    break
                if isinstance(item, Exception):
                    raise item
                yield {"event": "token", "data": item}
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
        reload=False,
    )


if __name__ == "__main__":
    main()
