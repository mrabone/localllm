from pydantic_settings import BaseSettings
from pydantic import Field
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL"
    )
    cli_ollama_model: str = Field(
        default="custom-chatbot-model", alias="CLI_OLLAMA_MODEL"
    )
    cli_max_recent: int = Field(default=10, alias="CLI_MAX_RECENT")
    cli_threshold: int = Field(default=20, alias="CLI_THRESHOLD")
    cli_enable_rag: bool = Field(default=False, alias="CLI_ENABLE_RAG")
    rag_ollama_model: str = Field(default="embeddinggemma", alias="RAG_OLLAMA_MODEL")
    cli_rag_threshold: float = Field(default=0.4, alias="CLI_RAG_THRESHOLD")
    cli_rag_k: int = Field(default=3, alias="CLI_RAG_K")

    # PGVector database settings (reused from RAG pipeline)
    pg_host: str = Field(default="127.0.0.1", alias="PG_HOST")
    pg_port: str = Field(default="5432", alias="PG_PORT")
    pg_database: str = Field(default="rag_db", alias="PG_DATABASE")
    pg_user: str = Field(default="user", alias="PG_USER")
    pg_password: str = Field(default="password", alias="PG_PASSWORD")
    pg_collection_name: str = Field(
        default="reading_list_embs", alias="PG_COLLECTION_NAME"
    )

    @property
    def db_connection_string(self) -> str:
        """Constructs the database connection string."""
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.pg_user,
            password=self.pg_password,
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_database,
        ).render_as_string(hide_password=False)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
