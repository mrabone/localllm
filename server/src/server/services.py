import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool
from ollama import Client

from common.db import build_pg_dsn
from common.db_pool import close_pool, init_pool
from common.session_store import ensure_turns_table
from server.config import settings

logger = logging.getLogger(__name__)


class ServiceContainer:
    _instance: "ServiceContainer | None" = None

    def __init__(
        self,
        ollama_client: Client,
        mcp_session: ClientSession,
        pg_dsn: str,
        mcp_tools: list[Tool],
    ) -> None:
        self.ollama_client = ollama_client
        self.mcp_session = mcp_session
        self.pg_dsn = pg_dsn
        self.mcp_tools = mcp_tools

    @classmethod
    def initialise(
        cls,
        ollama_client: Client,
        mcp_session: ClientSession,
        pg_dsn: str,
        mcp_tools: list[Tool],
    ) -> "ServiceContainer":
        cls._instance = cls(
            ollama_client=ollama_client,
            mcp_session=mcp_session,
            pg_dsn=pg_dsn,
            mcp_tools=mcp_tools,
        )
        return cls._instance

    @classmethod
    def get(cls) -> "ServiceContainer":
        if cls._instance is None:
            raise RuntimeError("Services not initialised")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @classmethod
    def get_or_none(cls) -> "ServiceContainer | None":
        return cls._instance


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Connect to shared services on startup and disconnect on shutdown."""
    logger.info("Starting server, connecting to services...")

    ollama_client = Client(host=settings.ollama_base_url)
    logger.info("Ollama client initialised (host=%s).", settings.ollama_base_url)

    pg_dsn = build_pg_dsn(settings)
    init_pool(pg_dsn)
    ensure_turns_table(pg_dsn)
    logger.info("PostgreSQL pool and schema ready.")

    mcp_url = f"{settings.mcp_server_url}/mcp"
    logger.info("Connecting to MCP server at %s...", mcp_url)
    try:
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP client session established.")

                tools_result = await session.list_tools()
                mcp_tools = tools_result.tools
                logger.info(
                    "Discovered %d MCP tools: %s",
                    len(mcp_tools),
                    [t.name for t in mcp_tools],
                )

                ServiceContainer.initialise(
                    ollama_client=ollama_client,
                    mcp_session=session,
                    pg_dsn=pg_dsn,
                    mcp_tools=mcp_tools,
                )

                yield
    except Exception as exc:
        logger.error("Failed to connect to MCP server: %s", exc, exc_info=True)
        raise
    finally:
        ServiceContainer.reset()
        close_pool()
        logger.info("Server shut down.")


def get_ollama_client() -> Client:
    return ServiceContainer.get().ollama_client


def get_mcp_session() -> ClientSession:
    return ServiceContainer.get().mcp_session


def get_pg_dsn() -> str:
    return ServiceContainer.get().pg_dsn


def _is_internal_tool(tool: Tool) -> bool:
    """Return True if the tool is tagged as internal infrastructure.

    Internal tools are called directly by the server and should never be
    described to the model.  FastMCP encodes tags in tool.meta under the
    key ``fastmcp.tags``; any tool carrying the ``"internal"`` tag is
    excluded from the model-facing tool list.
    """
    tags = (tool.meta or {}).get("fastmcp", {}).get("tags", [])
    return "internal" in tags


def get_mcp_tools() -> list[Tool]:
    container = ServiceContainer.get_or_none()
    if container is None:
        return []
    return [t for t in container.mcp_tools if not _is_internal_tool(t)]


OllamaClientDep = Annotated[Client, Depends(get_ollama_client)]
McpSessionDep = Annotated[ClientSession, Depends(get_mcp_session)]
PgDsnDep = Annotated[str, Depends(get_pg_dsn)]
McpToolsDep = Annotated[list[Tool], Depends(get_mcp_tools)]
