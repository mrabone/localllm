from pydantic import Field
from pydantic_settings import SettingsConfigDict

from common.config import SharedSettings


class Settings(SharedSettings):
    """Settings for the chat server.

    Inherits all shared PostgreSQL and Ollama fields from SharedSettings.
    Server-specific settings control the chat model, conversation history
    management, RAG retrieval behaviour, and the HTTP server binding.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HTTP server
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")

    # Chat model
    server_ollama_model: str = Field(
        default="custom-chatbot-model",
        alias="SERVER_OLLAMA_MODEL",
    )

    # Conversation history management
    server_max_recent: int = Field(
        default=10,
        alias="SERVER_MAX_RECENT",
        description="Number of recent messages to retain verbatim before summarisation.",
    )
    server_threshold: int = Field(
        default=20,
        alias="SERVER_THRESHOLD",
        description="Total message count at which history summarisation is triggered.",
    )

    # RAG retrieval
    server_enable_rag: bool = Field(default=False, alias="SERVER_ENABLE_RAG")
    server_rag_k: int = Field(
        default=3,
        alias="SERVER_RAG_K",
        description="Number of candidate documents to retrieve from PGVector.",
    )
    server_rag_max_distance: float = Field(
        default=0.5,
        alias="SERVER_RAG_MAX_DISTANCE",
        description="Maximum L2 distance for a retrieved document to be included in context.",
    )


settings = Settings()
