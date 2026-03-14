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
    server_reload: bool = Field(default=False, alias="SERVER_RELOAD")

    # MCP server connection
    mcp_server_url: str = Field(
        default="http://127.0.0.1:8001",
        alias="MCP_SERVER_URL",
        description="Base URL of the MCP server that owns RAG and memory tools.",
    )

    # Chat model
    server_ollama_model: str = Field(
        default="custom-chatbot-model",
        alias="SERVER_OLLAMA_MODEL",
    )
    server_ollama_num_ctx: int = Field(
        default=8192,
        alias="SERVER_OLLAMA_NUM_CTX",
        description="Context window size passed to Ollama on every chat call.",
    )

    # Memory retrieval
    server_memory_window_size: int = Field(
        default=10,
        alias="SERVER_MEMORY_WINDOW_SIZE",
        description=(
            "Number of verbatim turns (user + assistant messages) kept in the "
            "sliding window persisted in PostgreSQL. Older turns are discarded. "
            "Replaces the old SERVER_MEMORY_MAX_MESSAGES setting."
        ),
    )
    server_memory_long_term_max: int = Field(
        default=3,
        alias="SERVER_MEMORY_LONG_TERM_MAX",
        description=(
            "Maximum number of Mem0 long-term memory entries injected as system "
            "messages at the top of the context on every turn."
        ),
    )


settings = Settings()
