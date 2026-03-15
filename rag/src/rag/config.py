from pydantic import Field

from common.config import SharedSettings


class Settings(SharedSettings):
    """Settings for the RAG ingestion pipeline.

    Inherits all shared PostgreSQL and Ollama fields from SharedSettings.
    Pipeline-specific settings control scraping concurrency, rate limiting,
    and the semantic chunker behaviour.
    """

    # Async scraping concurrency
    concurrent_requests: int = Field(
        default=5,
        alias="CONCURRENT_REQUESTS",
        description="Number of concurrent HTTP requests and consumer workers.",
    )
    request_delay: float = Field(
        default=1.0,
        alias="REQUEST_DELAY",
        description="Seconds to wait before each HTTP request (per worker).",
    )

    # SemanticChunker tuning — see experiments/chunking_strategy_evaluation.ipynb
    chunker_breakpoint_type: str = Field(
        default="percentile",
        alias="CHUNKER_BREAKPOINT_TYPE",
        description="Breakpoint threshold type for SemanticChunker (e.g. 'percentile', 'standard_deviation').",
    )
    chunker_breakpoint_amount: float = Field(
        default=60.0,
        alias="CHUNKER_BREAKPOINT_AMOUNT",
        description="Breakpoint threshold amount for SemanticChunker.",
    )


settings = Settings()
