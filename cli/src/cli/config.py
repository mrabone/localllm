from pydantic import Field
from pydantic_settings import SettingsConfigDict

from common.config import SharedSettings


class Settings(SharedSettings):
    """Settings for the CLI chat application.

    Inherits all shared PostgreSQL and Ollama fields from SharedSettings.
    CLI-specific settings control the chat model, conversation history
    management, and RAG retrieval behaviour.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chat model
    cli_ollama_model: str = Field(
        default="custom-chatbot-model",
        alias="CLI_OLLAMA_MODEL",
    )

    # Conversation history management
    cli_max_recent: int = Field(
        default=10,
        alias="CLI_MAX_RECENT",
        description="Number of recent messages to retain verbatim before summarisation.",
    )
    cli_threshold: int = Field(
        default=20,
        alias="CLI_THRESHOLD",
        description="Total message count at which history summarisation is triggered.",
    )

    # RAG retrieval
    cli_enable_rag: bool = Field(default=False, alias="CLI_ENABLE_RAG")
    cli_rag_k: int = Field(
        default=3,
        alias="CLI_RAG_K",
        description="Number of candidate documents to retrieve from PGVector.",
    )
    cli_rag_max_distance: float = Field(
        default=0.5,
        alias="CLI_RAG_MAX_DISTANCE",
        description="Maximum L2 distance for a retrieved document to be included in context.",
    )


settings = Settings()
