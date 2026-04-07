from pydantic import Field

from common.config import SharedSettings


class Settings(SharedSettings):
    """Settings for the chat server.

    Inherits all shared PostgreSQL and Ollama fields from SharedSettings.
    Server-specific settings control the chat model, conversation history
    management, RAG retrieval behaviour, and the HTTP server binding.
    """

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
    server_function_calling_model: str = Field(
        default="functiongemma",
        alias="SERVER_FUNCTION_CALLING_MODEL",
    )
    server_ollama_num_ctx: int = Field(
        default=8192,
        alias="SERVER_OLLAMA_NUM_CTX",
        description="Context window size passed to Ollama for the main chat model.",
    )
    server_function_calling_num_ctx: int = Field(
        default=2048,
        alias="SERVER_FUNCTION_CALLING_NUM_CTX",
        description=(
            "Context window size passed to the function-calling model (FunctionGemma). "
            "Kept smaller than the main context since this model only needs to decide "
            "which tools to call, not generate a full response."
        ),
    )

    server_mcp_pool_size: int = Field(
        default=4,
        alias="SERVER_MCP_POOL_SIZE",
        description=(
            "Number of concurrent MCP client sessions to open at startup. "
            "Each session uses its own HTTP connection, allowing concurrent "
            "requests to avoid serialising their MCP tool calls."
        ),
    )

    # Memory retrieval
    server_memory_window_size: int = Field(
        default=10,
        alias="SERVER_MEMORY_WINDOW_SIZE",
        description=(
            "Number of verbatim turns (user + assistant messages) kept in the "
            "sliding window persisted in PostgreSQL. Older turns are discarded."
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
    server_tool_call_max_loops: int = Field(
        default=3,
        alias="SERVER_TOOL_CALL_MAX_LOOPS",
        description=(
            "Maximum number of tool-calling rounds the LLM is allowed per turn "
            "before being forced to produce a plain-text answer."
        ),
    )


settings = Settings()
