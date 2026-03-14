import logging

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from mem0 import Memory

from common.db import build_pg_dsn
from common.db_pool import _close_pool, _init_pool
from mcp_server.config import settings
from mcp_server.memory import ensure_turns_table

logger = logging.getLogger(__name__)

# Module-level singletons populated during lifespan startup.
_mem0: Memory | None = None
_rag_store: PGVector | None = None
_pg_dsn: str | None = None


@lifespan
async def lifespan(server: FastMCP):
    """Initialise shared services on startup and release them on shutdown."""
    global _mem0, _rag_store, _pg_dsn

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
                "model": settings.rag_ollama_model,
                "ollama_base_url": settings.ollama_base_url,
                "temperature": 0.1,
            },
        },
    }

    try:
        _mem0 = Memory.from_config(mem0_config)
        logger.info("Mem0 initialised (collection=%s).", settings.mem0_collection_name)
    except Exception as exc:
        logger.error("Failed to initialise Mem0: %s", exc, exc_info=True)
        raise

    _pg_dsn = build_pg_dsn(settings)
    _init_pool(_pg_dsn)
    ensure_turns_table(_pg_dsn)

    if settings.mcp_enable_rag:
        try:
            from sqlalchemy import create_engine

            _rag_engine = create_engine(settings.db_url)
            embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.rag_ollama_model,
            )
            _rag_store = PGVector(
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
            _rag_store = None
    else:
        logger.info("RAG disabled (MCP_ENABLE_RAG=false).")

    try:
        yield
    finally:
        logger.info("MCP server shutting down.")
        _close_pool()


def get_mem0() -> Memory:
    if _mem0 is None:
        raise RuntimeError("Mem0 not initialised")
    return _mem0


def get_rag_store() -> PGVector | None:
    return _rag_store


def get_pg_dsn() -> str:
    if _pg_dsn is None:
        raise RuntimeError("PostgreSQL DSN not initialised")
    return _pg_dsn
