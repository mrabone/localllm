import logging
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from ollama import Client
from sqlalchemy import create_engine

from cli.config import settings

logger = logging.getLogger(__name__)


def get_ollama_client() -> Client:
    """Create and return an Ollama client pointed at the configured base URL."""
    return Client(host=settings.ollama_base_url)


def get_rag_store() -> Optional[PGVector]:
    """Create and return a PGVector store, or None if RAG is disabled.

    Returns None without raising if RAG is disabled via CLI_ENABLE_RAG=false,
    or if the database connection fails — allowing the app to run in
    RAG-less mode gracefully.
    """
    if not settings.cli_enable_rag:
        return None

    try:
        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.rag_ollama_model,
        )
        engine = create_engine(settings.db_url)
        store = PGVector(
            connection=engine,
            embeddings=embeddings,
            collection_name=settings.pg_collection_name,
        )
        logger.info(
            "Connected to PGVector collection '%s'", settings.pg_collection_name
        )
        return store
    except Exception as e:
        logger.warning("Failed to initialise RAG store: %s", e)
        return None
