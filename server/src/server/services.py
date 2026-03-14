import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from ollama import Client

from common.db import build_pg_dsn
from common.db_pool import _close_pool, _init_pool
from server.config import settings
from server.memory import ensure_turns_table

logger = logging.getLogger(__name__)

# Module-level singletons populated during lifespan startup.
_ollama_client: Client | None = None
_mcp_session: ClientSession | None = None
_pg_dsn: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Connect to shared services on startup and disconnect on shutdown."""
    global _ollama_client, _mcp_session, _pg_dsn

    logger.info("Starting server, connecting to services...")

    _ollama_client = Client(host=settings.ollama_base_url)
    logger.info("Ollama client initialised (host=%s).", settings.ollama_base_url)

    _pg_dsn = build_pg_dsn(settings)
    _init_pool(_pg_dsn)
    ensure_turns_table(_pg_dsn)
    logger.info("PostgreSQL pool and schema ready.")

    mcp_url = f"{settings.mcp_server_url}/mcp"
    logger.info("Connecting to MCP server at %s...", mcp_url)
    try:
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _mcp_session = session
                logger.info("MCP client session established.")
                yield
    except Exception as exc:
        logger.error("Failed to connect to MCP server: %s", exc, exc_info=True)
        raise
    finally:
        _mcp_session = None
        _close_pool()
        logger.info("Server shut down.")


def get_ollama_client() -> Client:
    if _ollama_client is None:
        raise RuntimeError("Ollama client not initialised")
    return _ollama_client


def get_mcp_session() -> ClientSession:
    if _mcp_session is None:
        raise RuntimeError("MCP client session not initialised")
    return _mcp_session


def get_pg_dsn() -> str:
    if _pg_dsn is None:
        raise RuntimeError("PostgreSQL DSN not initialised")
    return _pg_dsn


OllamaClientDep = Annotated[Client, Depends(get_ollama_client)]
McpSessionDep = Annotated[ClientSession, Depends(get_mcp_session)]
PgDsnDep = Annotated[str, Depends(get_pg_dsn)]
