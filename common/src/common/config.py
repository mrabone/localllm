from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class SharedSettings(BaseSettings):
    """
    Base settings shared across all workspace packages.

    Covers the PostgreSQL connection and the Ollama service URL/model used
    for both the RAG ingestion pipeline and the CLI chat application.
    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        alias="OLLAMA_BASE_URL",
    )
    rag_ollama_model: str = Field(
        default="embeddinggemma",
        alias="RAG_OLLAMA_MODEL",
    )

    # PostgreSQL / PGVector
    pg_host: str = Field(default="127.0.0.1", alias="PG_HOST")
    pg_port: str = Field(default="5432", alias="PG_PORT")
    pg_database: str = Field(default="rag_db", alias="PG_DATABASE")
    pg_user: str = Field(default="user", alias="PG_USER")
    pg_password: str = Field(default="password", alias="PG_PASSWORD")
    pg_collection_name: str = Field(
        default="reading_list_embs",
        alias="PG_COLLECTION_NAME",
    )
    mem0_collection_name: str = Field(
        default="mem0_chat",
        alias="MEM0_COLLECTION_NAME",
    )

    @property
    def db_url(self) -> URL:
        """Construct a SQLAlchemy URL object for the PostgreSQL connection.

        Uses URL.create() rather than f-string interpolation so that special
        characters in the password are handled correctly.

        WARNING: do NOT log or format this value directly — SQLAlchemy renders
        the password in plain text when the URL is coerced to a string.
        Use ``db_url_safe`` for any logging or display.
        """
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.pg_user,
            password=self.pg_password,
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_database,
        )

    @property
    def db_url_safe(self) -> str:
        """Return a redacted connection string safe for logging.

        The password is replaced with ``***`` so the string can be included in
        log messages or error output without leaking credentials.
        """
        return (
            f"postgresql+psycopg2://{self.pg_user}:***"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )
