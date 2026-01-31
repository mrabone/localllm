from sqlalchemy import create_engine
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from ollama import Client
from typing import Optional

from cli.config import settings


class ServiceProvider:
    """A singleton provider for all external services."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceProvider, cls).__new__(cls)
            cls._instance._init_services()
        return cls._instance

    def _init_services(self) -> None:
        """Initialize all services."""
        self.ollama_client = Client(host=settings.ollama_base_url)
        self.pgvector_store = self._init_rag()

    def _init_rag(self) -> Optional[PGVector]:
        """Initialize the PGVector store for RAG if enabled."""
        if not settings.cli_enable_rag:
            return None

        try:
            ollama_embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url, model=settings.rag_ollama_model
            )
            engine = create_engine(settings.db_connection_string)
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

    def get_ollama_client(self) -> Client:
        return self.ollama_client

    def get_pgvector_store(self) -> Optional[PGVector]:
        return self.pgvector_store


services = ServiceProvider()
