from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL"
    )
    cli_ollama_model: str = Field(
        default="custom-chatbot-model", alias="CLI_OLLAMA_MODEL"
    )
    cli_max_recent: int = Field(default=10, alias="CLI_MAX_RECENT")
    cli_threshold: int = Field(default=20, alias="CLI_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
