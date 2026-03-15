import logging

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan as fastmcp_lifespan
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from mem0 import Memory

from common.db import build_pg_dsn
from common.db_pool import close_pool, init_pool
from common.session_store import ensure_turns_table
from mcp_server.config import settings

logger = logging.getLogger(__name__)


class ServiceContainer:
    _instance: "ServiceContainer | None" = None

    def __init__(self, mem0: Memory, pg_dsn: str, rag_store: PGVector | None) -> None:
        self.mem0 = mem0
        self.pg_dsn = pg_dsn
        self.rag_store = rag_store

    @classmethod
    def initialise(
        cls,
        mem0: Memory,
        pg_dsn: str,
        rag_store: PGVector | None,
    ) -> "ServiceContainer":
        cls._instance = cls(mem0=mem0, pg_dsn=pg_dsn, rag_store=rag_store)
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


@fastmcp_lifespan
async def lifespan(server: FastMCP):
    """Initialise shared services on startup and release them on shutdown."""
    logger.info("Starting MCP server, initialising services...")

    mem0_config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": settings.pg_host,
                "port": settings.pg_port,
                "user": settings.pg_user,
                "password": settings.pg_password,
                "dbname": settings.pg_database,
                "collection_name": settings.mem0_collection_name,
                "embedding_model_dims": 768,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": settings.rag_ollama_model,
                "ollama_base_url": settings.ollama_base_url,
                "embedding_dims": 768,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": settings.mem0_llm_model,
                "ollama_base_url": settings.ollama_base_url,
                "temperature": 0.1,
            },
        },
    }

    try:
        mem0 = Memory.from_config(mem0_config)
        logger.info("Mem0 initialised (collection=%s).", settings.mem0_collection_name)
    except Exception as exc:
        logger.error("Failed to initialise Mem0: %s", exc, exc_info=True)
        raise

    pg_dsn = build_pg_dsn(settings)
    init_pool(pg_dsn)
    ensure_turns_table(pg_dsn)

    rag_store: PGVector | None = None
    if settings.mcp_enable_rag:
        try:
            from sqlalchemy import create_engine

            _rag_engine = create_engine(settings.db_url)
            embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.rag_ollama_model,
            )
            rag_store = PGVector(
                connection=_rag_engine,
                embeddings=embeddings,
                collection_name=settings.pg_collection_name,
            )
            logger.info(
                "PGVector RAG store initialised (collection=%s).",
                settings.pg_collection_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialise RAG store, running without RAG: %s", exc
            )
    else:
        logger.info("RAG disabled (MCP_ENABLE_RAG=false).")

    ServiceContainer.initialise(mem0=mem0, pg_dsn=pg_dsn, rag_store=rag_store)

    try:
        yield
    finally:
        logger.info("MCP server shutting down.")
        ServiceContainer.reset()
        close_pool()


def get_mem0() -> Memory:
    return ServiceContainer.get().mem0


def get_rag_store() -> PGVector | None:
    container = ServiceContainer.get_or_none()
    if container is None:
        return None
    return container.rag_store


def get_pg_dsn() -> str:
    return ServiceContainer.get().pg_dsn
