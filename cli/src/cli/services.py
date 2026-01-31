from sqlalchemy import create_engine
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from ollama import Client
from typing import Optional

from cli.config import settings


def _init_ollama_client() -> Client:
    """Initialize the Ollama client."""
    return Client(host=settings.ollama_base_url)


def _init_rag_store() -> Optional[PGVector]:
    """Initialize the PGVector store for RAG if enabled."""
    if not settings.cli_enable_rag:
        return None

    try:
        ollama_embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url, model=settings.rag_ollama_model
        )
        engine = create_engine(settings.db_url)
        pg_vector_store = PGVector(
            connection=engine,
            embeddings=ollama_embeddings,
            collection_name=settings.pg_collection_name,
        )
        print(
            f"RAG enabled: Connected to PGVector collection '{settings.pg_collection_name}'"
        )
        return pg_vector_store
    except Exception as e:
        print(f"Warning: Failed to initialize RAG: {e}")
        return None


# Initialize services at module level
ollama_client = _init_ollama_client()
pgvector_store = _init_rag_store()
