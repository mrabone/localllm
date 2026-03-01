import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from ollama import Client
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from server.config import settings
from server.db import create_tables

logger = logging.getLogger(__name__)

# Module-level singletons populated during lifespan startup.
_engine: Engine | None = None
_ollama_client: Client | None = None
_rag_store: PGVector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create shared services on startup and clean up on shutdown."""
    global _engine, _ollama_client, _rag_store

    logger.info("Starting server, initialising services...")

    _engine = create_engine(settings.db_url)
    create_tables(_engine)
    logger.info("Database tables ready.")

    _ollama_client = Client(host=settings.ollama_base_url)
    logger.info("Ollama client initialised (host=%s).", settings.ollama_base_url)

    if settings.server_enable_rag:
        try:
            embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.rag_ollama_model,
            )
            _rag_store = PGVector(
                connection=_engine,
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
    if _engine:
        _engine.dispose()


def get_engine() -> Engine:
    assert _engine is not None, "Engine not initialised"
    return _engine


def get_ollama_client() -> Client:
    assert _ollama_client is not None, "Ollama client not initialised"
    return _ollama_client


def get_rag_store() -> PGVector | None:
    return _rag_store


EngineDep = Annotated[Engine, Depends(get_engine)]
OllamaClientDep = Annotated[Client, Depends(get_ollama_client)]
RagStoreDep = Annotated[PGVector | None, Depends(get_rag_store)]
