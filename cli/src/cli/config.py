from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the CLI chat application.

    The CLI is a thin presentation layer that delegates all chat logic to the
    server.  The only configuration it needs is the server's base URL.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server_url: str = Field(
        default="http://127.0.0.1:8000",
        alias="SERVER_URL",
        description="Base URL of the localLLM HTTP server.",
    )

    sessions_registry: str = Field(
        default="~/.localllm_sessions.json",
        alias="SESSIONS_REGISTRY",
        description="Path to the JSON file that maps session names to UUIDs.",
    )

    session_cache_ttl: int = Field(
        default=300,
        alias="SESSION_CACHE_TTL",
        description=(
            "Seconds a validated session UUID is trusted locally before "
            "re-checking with the server. Set to 0 to always validate."
        ),
    )


settings = Settings()
