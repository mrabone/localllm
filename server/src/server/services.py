import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from mem0 import Memory
from ollama import Client

from server.config import settings
from server.memory import _close_pool, _init_pool, ensure_turns_table

logger = logging.getLogger(__name__)

# Module-level singletons populated during lifespan startup.
_mem0: Memory | None = None
_ollama_client: Client | None = None
_rag_store: PGVector | None = None
_pg_dsn: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create shared services on startup and clean up on shutdown."""
    global _mem0, _ollama_client, _rag_store, _pg_dsn

    logger.info("Starting server, initialising services...")

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
                "model": settings.server_ollama_model,
                "ollama_base_url": settings.ollama_base_url,
                "temperature": 0.1,
            },
        },
    }
    logger.debug("Initialising Mem0 with config: %s", mem0_config)
    try:
        _mem0 = Memory.from_config(mem0_config)
        logger.info(
            "Mem0 memory initialised successfully (collection=%s).",
            settings.mem0_collection_name,
        )
    except Exception as exc:
        logger.error("Failed to initialise Mem0: %s", exc, exc_info=True)
        raise

    # Ensure the verbatim sliding-window table exists in PostgreSQL.
    _pg_dsn = (
        f"host={settings.pg_host} port={settings.pg_port} "
        f"dbname={settings.pg_database} user={settings.pg_user} "
        f"password={settings.pg_password}"
    )
    _init_pool(_pg_dsn)
    ensure_turns_table(_pg_dsn)

    _ollama_client = Client(host=settings.ollama_base_url)
    logger.info("Ollama client initialised (host=%s).", settings.ollama_base_url)

    if settings.server_enable_rag:
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
        logger.info("RAG disabled (SERVER_ENABLE_RAG=false).")

    yield

    logger.info("Server shutting down.")
    _close_pool()


def get_mem0() -> Memory:
    if _mem0 is None:
        raise RuntimeError("Mem0 not initialised")
    return _mem0


def get_ollama_client() -> Client:
    if _ollama_client is None:
        raise RuntimeError("Ollama client not initialised")
    return _ollama_client


def get_rag_store() -> PGVector | None:
    return _rag_store


def get_pg_dsn() -> str:
    if _pg_dsn is None:
        raise RuntimeError("PostgreSQL DSN not initialised")
    return _pg_dsn


Mem0Dep = Annotated[Memory, Depends(get_mem0)]
OllamaClientDep = Annotated[Client, Depends(get_ollama_client)]
RagStoreDep = Annotated[PGVector | None, Depends(get_rag_store)]
PgDsnDep = Annotated[str, Depends(get_pg_dsn)]
