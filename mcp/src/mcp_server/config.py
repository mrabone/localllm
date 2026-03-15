from pydantic import Field

from common.config import SharedSettings


class Settings(SharedSettings):
    """Settings for the MCP server.

    Inherits all shared PostgreSQL and Ollama fields from SharedSettings,
    including the shared RAG retrieval settings (rag_k, rag_max_distance).
    MCP-specific settings control the HTTP binding and which optional
    services are enabled at startup.
    """

    # HTTP server
    mcp_host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    mcp_port: int = Field(default=8001, alias="MCP_PORT")

    # Feature flags
    mcp_enable_rag: bool = Field(
        default=False,
        alias="MCP_ENABLE_RAG",
        description="When true, initialise the PGVector RAG store on startup.",
    )


settings = Settings()
