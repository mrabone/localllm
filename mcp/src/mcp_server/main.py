import asyncio
import functools
import uuid
from typing import Literal

import httpx
import uvicorn
from fastmcp import FastMCP
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from common.db_pool import is_pool_healthy
from common.logging_utils import setup_logging
from mcp_server.config import settings
from mcp_server.memory import (
    append_turn,
    load_long_term_memories,
    load_window,
    save_message,
)
from mcp_server.rag import build_rag_system_message, get_rag_context
from mcp_server.services import (
    ServiceContainer,
    get_mem0,
    get_pg_dsn,
    get_rag_store,
    lifespan,
)

setup_logging()

mcp = FastMCP(name="localllm-mcp", lifespan=lifespan)


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the RAG knowledge base and return relevant document snippets.

    Returns a formatted string of ranked document snippets, or an empty
    string if RAG is disabled, no relevant documents were found, or the
    query score exceeds the configured distance threshold.

    The returned string is ready to embed directly in an LLM system message
    via ``build_rag_system_message``.

    Args:
        query: The search query, typically the user's current message.
    """
    rag_result = get_rag_context(query, get_rag_store())
    if rag_result is None:
        return ""
    return build_rag_system_message(rag_result.context)


class PersistMessageRequest(BaseModel):
    session_id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str


def main() -> None:
    async def health_check(request: Request) -> JSONResponse:
        """Health check endpoint for container orchestration.

        Verifies that the service container is initialised and that both
        PostgreSQL and Ollama are reachable.  Returns 200 when healthy,
        503 when any dependency is unavailable.
        """
        if ServiceContainer.get_or_none() is None:
            return JSONResponse(
                {"status": "unavailable", "detail": "services not yet initialised"},
                status_code=503,
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

        status_code = 200 if healthy else 503
        return JSONResponse(
            {"status": "ok" if healthy else "degraded", "checks": checks},
            status_code=status_code,
        )

    async def get_conversation_window(request: Request) -> JSONResponse:
        """Return the most recent verbatim turns for a session, oldest first.

        Query parameters:
            window_size: Maximum number of turns to return (default 10).
        """
        try:
            session_id = uuid.UUID(request.path_params["session_id"])
            window_size = int(request.query_params.get("window_size", 10))
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        turns = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                load_window, get_pg_dsn(), session_id, window_size=window_size
            ),
        )
        return JSONResponse(turns)

    async def get_long_term_memory(request: Request) -> JSONResponse:
        """Return Mem0-extracted semantic facts for a session.

        Returns a JSON object with a ``content`` field containing a
        bullet-point string, or an empty string if no memories exist yet.

        Query parameters:
            long_term_max: Maximum number of facts to include (default 3).
        """
        try:
            session_id = uuid.UUID(request.path_params["session_id"])
            long_term_max = int(request.query_params.get("long_term_max", 3))
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        messages = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                load_long_term_memories,
                get_mem0(),
                session_id,
                long_term_max=long_term_max,
            ),
        )
        content = messages[0]["content"] if messages else ""
        return JSONResponse({"content": content})

    async def persist_message(request: Request) -> JSONResponse:
        """Persist a single conversation turn to both Mem0 and the verbatim window.

        Both writes are dispatched to the default thread-pool executor so that
        the blocking Mem0 and psycopg2 calls do not stall the event loop.  They
        run concurrently because they write to independent stores.
        """
        body = PersistMessageRequest.model_validate(await request.json())
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            loop.run_in_executor(
                None,
                functools.partial(
                    save_message, get_mem0(), body.session_id, body.role, body.content
                ),
            ),
            loop.run_in_executor(
                None,
                functools.partial(
                    append_turn, get_pg_dsn(), body.session_id, body.role, body.content
                ),
            ),
        )
        return JSONResponse(None, status_code=204)

    app = mcp.http_app(path="/mcp")
    app.routes.append(Route("/health", health_check, methods=["GET"]))
    app.routes.append(
        Route(
            "/memory/window/{session_id}",
            get_conversation_window,
            methods=["GET"],
        )
    )
    app.routes.append(
        Route(
            "/memory/long-term/{session_id}",
            get_long_term_memory,
            methods=["GET"],
        )
    )
    app.routes.append(Route("/memory/messages", persist_message, methods=["POST"]))

    uvicorn.run(
        app,
        host=settings.mcp_host,
        port=settings.mcp_port,
    )


if __name__ == "__main__":
    main()
