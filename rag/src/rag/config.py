from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    pg_host: str = Field(default="127.0.0.1", alias="PG_HOST")
    pg_port: str = Field(default="5432", alias="PG_PORT")
    pg_database: str = Field(default="rag_db", alias="PG_DATABASE")
    pg_user: str = Field(default="user", alias="PG_USER")
    pg_password: str = Field(default="password", alias="PG_PASSWORD")
    pg_collection_name: str = Field(
        default="reading_list_embs", alias="PG_COLLECTION_NAME"
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL"
    )
    rag_ollama_model: str = Field(default="embeddinggemma", alias="RAG_OLLAMA_MODEL")
    concurrent_requests: int = Field(default=5, alias="CONCURRENT_REQUESTS")
    request_delay: int = Field(default=1, alias="REQUEST_DELAY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
