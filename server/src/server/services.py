import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

import httpx
from fastapi import Depends, FastAPI, HTTPException
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool
from ollama import AsyncClient

from common.db import build_pg_dsn
from common.db_pool import close_pool, init_pool
from common.session_store import ensure_turns_table
from server.config import settings

logger = logging.getLogger(__name__)


async def _open_mcp_session(
    mcp_url: str,
    max_retries: int = 5,
    initial_wait_seconds: float = 1.0,
) -> tuple[ClientSession, list[Tool], asyncio.Task]:
    """Open a single MCP session and return it alongside a keep-alive task.

    The keep-alive task holds the underlying HTTP transport open for the
    lifetime of the session.  Callers must cancel the task and await it when
    they are done with the session to clean up the transport.

    Returns:
        Tuple of (ClientSession, list of available tools, keep-alive Task).

    Raises:
        RuntimeError: If connection fails after all retry attempts.
    """
    last_exception: Exception | None = None
    session_ready: asyncio.Event = asyncio.Event()
    result_holder: list = []

    async def _run_session() -> None:
        nonlocal last_exception
        for attempt in range(1, max_retries + 1):
            try:
                async with streamablehttp_client(mcp_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        result_holder.append((session, tools_result.tools))
                        session_ready.set()
                        await asyncio.Event().wait()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait_time = initial_wait_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "MCP connection failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt,
                        max_retries,
                        exc,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "MCP connection failed after %d attempts: %s",
                        max_retries,
                        exc,
                        exc_info=True,
                    )
                    session_ready.set()

    task = asyncio.create_task(_run_session())
    await session_ready.wait()

    if not result_holder:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise RuntimeError(
            f"Failed to connect to MCP server after {max_retries} attempts"
        ) from last_exception

    session, tools = result_holder[0]
    return session, tools, task


class McpSessionPool:
    """A fixed-size pool of MCP client sessions.

    Opens `pool_size` independent MCP connections at startup so that
    concurrent requests can each use their own session, preventing the
    serialisation that occurs when multiple requests share one connection.

    Usage:
        async with pool.acquire() as session:
            await session.call_tool(...)
    """

    def __init__(self, pool_size: int) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {pool_size}")
        self._pool_size = pool_size
        self._sessions: list[ClientSession] = []
        self._tasks: list[asyncio.Task] = []
        self._queue: asyncio.Queue[ClientSession] = asyncio.Queue()

    async def start(self, mcp_url: str, max_retries: int = 5) -> list[Tool]:
        """Open all sessions concurrently and populate the pool.

        Returns the tool list from the first successfully opened session
        (all sessions connect to the same server, so their tool lists are
        identical).
        """
        open_results = await asyncio.gather(
            *[
                _open_mcp_session(mcp_url, max_retries=max_retries)
                for _ in range(self._pool_size)
            ],
            return_exceptions=True,
        )

        successes: list[tuple[ClientSession, list[Tool], asyncio.Task]] = []
        errors: list[BaseException] = []
        for result in open_results:
            if isinstance(result, BaseException):
                errors.append(result)
            else:
                successes.append(result)

        if errors:
            for _, _, task in successes:
                task.cancel()
            await asyncio.gather(
                *(task for _, _, task in successes), return_exceptions=True
            )
            error_summary = "; ".join(f"{type(err).__name__}: {err}" for err in errors)
            raise RuntimeError(
                f"Failed to open MCP session pool: {error_summary}"
            ) from errors[0]

        mcp_tools: list[Tool] = []
        for i, (session, tools, task) in enumerate(successes):
            self._sessions.append(session)
            self._tasks.append(task)
            await self._queue.put(session)
            if i == 0:
                mcp_tools = tools
                logger.info(
                    "MCP session pool ready (%d sessions). Tools: %s",
                    self._pool_size,
                    [t.name for t in tools],
                )
        return mcp_tools

    async def stop(self) -> None:
        """Cancel all keep-alive tasks, closing the underlying transports."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._sessions.clear()
        self._tasks.clear()
        logger.info("MCP session pool closed.")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[ClientSession, None]:
        """Check out a session from the pool, yield it, then return it."""
        session = await self._queue.get()
        try:
            yield session
        finally:
            await self._queue.put(session)


class ServiceContainer:
    _instance: "ServiceContainer | None" = None

    def __init__(
        self,
        ollama_client: AsyncClient,
        mcp_session_pool: McpSessionPool,
        memory_http_client: httpx.AsyncClient,
        pg_dsn: str,
        mcp_tools: list[Tool],
        function_calling_model: str,
    ) -> None:
        self.ollama_client = ollama_client
        self.mcp_session_pool = mcp_session_pool
        self.memory_http_client = memory_http_client
        self.pg_dsn = pg_dsn
        self.mcp_tools = mcp_tools
        self.function_calling_model = function_calling_model

    @classmethod
    def initialise(
        cls,
        ollama_client: AsyncClient,
        mcp_session_pool: McpSessionPool,
        memory_http_client: httpx.AsyncClient,
        pg_dsn: str,
        mcp_tools: list[Tool],
        function_calling_model: str,
    ) -> "ServiceContainer":
        cls._instance = cls(
            ollama_client=ollama_client,
            mcp_session_pool=mcp_session_pool,
            memory_http_client=memory_http_client,
            pg_dsn=pg_dsn,
            mcp_tools=mcp_tools,
            function_calling_model=function_calling_model,
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

    ollama_client = AsyncClient(host=settings.ollama_base_url)
    logger.info("Ollama async client initialised (host=%s).", settings.ollama_base_url)

    pg_dsn = build_pg_dsn(settings)
    init_pool(pg_dsn)
    ensure_turns_table(pg_dsn)
    logger.info("PostgreSQL pool and schema ready.")

    memory_http_client = httpx.AsyncClient(
        base_url=settings.mcp_server_url,
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )

    mcp_url = f"{settings.mcp_server_url}/mcp"
    logger.info(
        "Connecting to MCP server at %s (pool_size=%d)...",
        mcp_url,
        settings.server_mcp_pool_size,
    )
    pool = McpSessionPool(pool_size=settings.server_mcp_pool_size)
    try:
        mcp_tools = await pool.start(mcp_url)
        ServiceContainer.initialise(
            ollama_client=ollama_client,
            mcp_session_pool=pool,
            memory_http_client=memory_http_client,
            pg_dsn=pg_dsn,
            mcp_tools=mcp_tools,
            function_calling_model=settings.server_function_calling_model,
        )

        yield
    except Exception as exc:
        logger.error("Failed to connect to MCP server: %s", exc, exc_info=True)
        raise
    finally:
        await pool.stop()
        await memory_http_client.aclose()
        ServiceContainer.reset()
        close_pool()
        logger.info("Server shut down.")


def get_ollama_client() -> AsyncClient:
    container = ServiceContainer.get_or_none()
    if container is None:
        raise HTTPException(status_code=503, detail="services not initialised")
    return container.ollama_client


def get_mcp_session_pool() -> McpSessionPool:
    container = ServiceContainer.get_or_none()
    if container is None:
        raise HTTPException(status_code=503, detail="services not initialised")
    return container.mcp_session_pool


def get_memory_http_client() -> httpx.AsyncClient:
    container = ServiceContainer.get_or_none()
    if container is None:
        raise HTTPException(status_code=503, detail="services not initialised")
    return container.memory_http_client


def get_pg_dsn() -> str:
    container = ServiceContainer.get_or_none()
    if container is None:
        raise HTTPException(status_code=503, detail="services not initialised")
    return container.pg_dsn


def get_mcp_tools() -> list[Tool]:
    container = ServiceContainer.get_or_none()
    if container is None:
        raise HTTPException(status_code=503, detail="services not initialised")
    return list(container.mcp_tools)


def get_function_calling_model() -> str:
    container = ServiceContainer.get_or_none()
    if container is None:
        raise HTTPException(status_code=503, detail="services not initialised")
    return container.function_calling_model


OllamaClientDep = Annotated[AsyncClient, Depends(get_ollama_client)]
McpSessionPoolDep = Annotated[McpSessionPool, Depends(get_mcp_session_pool)]
MemoryHttpClientDep = Annotated[httpx.AsyncClient, Depends(get_memory_http_client)]
PgDsnDep = Annotated[str, Depends(get_pg_dsn)]
McpToolsDep = Annotated[list[Tool], Depends(get_mcp_tools)]
FunctionCallingModelDep = Annotated[str, Depends(get_function_calling_model)]
