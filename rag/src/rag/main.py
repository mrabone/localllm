import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import httpx
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

from rag.config import settings
from common.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# HTTP headers used for all scraping requests.
# A realistic browser UA reduces the chance of being blocked by simple filters.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


def load_reading_list(file_path: Path) -> list[dict]:
    """Load the reading list from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A list of item dicts, or an empty list if the file is missing, unreadable, or invalid.
    """
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from %s: %s", file_path, e)
        return []
    except (OSError, IOError) as e:
        logger.error("Failed to read file %s: %s", file_path, e)
        return []

    # Validate that the parsed JSON is a list of dict-like items
    if not isinstance(data, list):
        logger.error(
            "Expected JSON array in %s, but got %s", file_path, type(data).__name__
        )
        return []

    return data


async def scrape_url(
    url: str,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> str | None:
    """Fetch raw HTML from a URL, respecting the concurrency semaphore and delay.

    Args:
        url: The URL to fetch.
        semaphore: Limits the number of concurrent requests.
        delay: Seconds to wait before issuing the request (per worker).

    Returns:
        The response body as a string, or None on any error.
    """
    async with semaphore:
        await asyncio.sleep(delay)
        try:
            logger.info("Fetching: %s", url)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=HEADERS)
                response.raise_for_status()
                logger.info("Fetched successfully: %s", url)
                return response.text
        except httpx.RequestError as e:
            logger.error("Request error fetching %s: %s", url, e)
        except httpx.HTTPStatusError as e:
            logger.error("HTTP status error fetching %s: %s", url, e)
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", url, e)
        return None


def extract_text_from_html(html: str) -> str | None:
    """Parse HTML and return cleaned plain text, or None if no text was found.

    Uses ``trafilatura`` for main-content extraction, which filters out
    navigation bars, footers, ads, and other boilerplate automatically.
    Falls back to a BeautifulSoup heuristic (<main> → <body> → document) if
    trafilatura returns nothing, then collapses whitespace and normalises
    newlines on the result.
    """
    # Primary path: trafilatura gives much cleaner article/main-content text.
    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted:
        cleaned = re.sub(r"[ \t]+", " ", extracted)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip() or None

    # Fallback: BeautifulSoup heuristic for pages trafilatura cannot parse.
    soup = BeautifulSoup(html, "html.parser")
    content_block = soup.main or soup.body or soup
    raw_text = content_block.get_text(separator="\n", strip=True)

    cleaned = re.sub(r"[ \t]+", " ", raw_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned if cleaned else None


def chunk_documents(
    text: str,
    url: str,
    splitter: SemanticChunker,
) -> list[Document]:
    """Split text into semantic chunks using the provided SemanticChunker.

    Args:
        text: The cleaned plain text to split.
        url: The source URL, stored in each chunk's metadata.
        splitter: A pre-built SemanticChunker instance (constructed once per
            pipeline run to avoid repeated initialisation overhead).

    Returns:
        A list of Document chunks, or an empty list if splitting produced nothing.
    """
    document = Document(page_content=text, metadata={"source": url})
    return splitter.split_documents([document])


async def process_item(
    item: dict,
    pgvector_store: PGVector,
    splitter: SemanticChunker,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> None:
    """Process a single reading-list item end-to-end.

    Steps: validate → duplicate-check → scrape → extract text →
    chunk → embed → store.

    Args:
        item: A reading-list dict with 'url' and optional 'title' keys.
        pgvector_store: Destination vector store.
        splitter: Pre-built SemanticChunker (shared across all items to avoid
            repeated instantiation overhead).
        semaphore: Concurrency limiter passed through to the scraper.
        delay: Per-request delay in seconds passed through to the scraper.
    """
    url: str | None = item.get("url")
    title: str = item.get("title") or url or "Unknown Item"

    if not url:
        logger.warning("Skipping item with missing URL: %s", title)
        return

    logger.info("Processing: %s (%s)", title, url)

    # 1. Skip already-processed URLs
    try:
        existing = pgvector_store.similarity_search(
            query="",
            k=1,
            filter={"source": url},
        )
        if existing:
            logger.info("Already processed, skipping: %s", url)
            return
    except Exception as e:
        logger.error("Error checking for existing documents for %s: %s", url, e)

    # 2. Scrape
    html = await scrape_url(url, semaphore, delay)
    if not html:
        logger.error("Failed to fetch HTML for %s — skipping.", url)
        return

    # 3. Extract text
    text = extract_text_from_html(html)
    if not text:
        logger.warning("No meaningful text extracted from %s — skipping.", url)
        return

    # 4. Chunk
    try:
        chunks = chunk_documents(text, url, splitter)
    except Exception as e:
        logger.error("Error chunking %s: %s", url, e)
        return

    if not chunks:
        logger.warning("No chunks produced for %s — skipping.", url)
        return

    logger.info("Split into %d chunk(s): %s", len(chunks), url)

    # 5. Store
    try:
        pgvector_store.add_documents(chunks)
        logger.info("Stored %d chunk(s) from %s.", len(chunks), url)
    except Exception as e:
        logger.error("Error storing chunks for %s: %s", url, e)


async def producer(
    queue: asyncio.Queue,
    reading_list: list[dict],
    num_consumers: int,
) -> None:
    """Enqueue all reading-list items, then send one sentinel per consumer.

    Sending one ``None`` sentinel per consumer guarantees that every worker
    receives exactly one shutdown signal and exits cleanly without needing
    to be cancelled from outside.
    """
    for item in reading_list:
        await queue.put(item)
    for _ in range(num_consumers):
        await queue.put(None)  # poison pill


async def consumer(
    queue: asyncio.Queue,
    pgvector_store: PGVector,
    splitter: SemanticChunker,
    semaphore: asyncio.Semaphore,
) -> None:
    """Consume items from the queue until a ``None`` sentinel is received.

    A ``None`` value is the poison-pill shutdown signal written by the producer
    after all real items have been enqueued.  Unexpected errors are logged and
    the current item is skipped so that the worker remains alive for subsequent
    items.
    """
    while True:
        try:
            item = await queue.get()
            if item is None:
                # Poison pill — this worker is done.
                queue.task_done()
                break
            await process_item(
                item,
                pgvector_store,
                splitter,
                semaphore,
                settings.request_delay,
            )
            queue.task_done()
        except Exception:
            logger.exception("Unexpected error processing item — skipping.")
            queue.task_done()


def init_pgvector_store(embeddings: OllamaEmbeddings) -> PGVector:
    """Connect to PostgreSQL and return an initialised PGVector store.

    Raises on failure so that the caller can decide whether to abort the pipeline.
    """
    engine = create_engine(settings.db_url)
    return PGVector(
        connection=engine,
        embeddings=embeddings,
        collection_name=settings.pg_collection_name,
    )


async def run_pipeline(data_path: Path) -> None:
    """Load a reading list and process all items concurrently.

    Args:
        data_path: Path to the JSON file containing the reading list.
    """
    logger.info("Loading reading list from: %s", data_path)
    reading_list = load_reading_list(data_path)

    if not reading_list:
        logger.info("No items to process.")
        return

    logger.info("Loaded %d item(s).", len(reading_list))

    logger.info("Initialising embeddings model...")
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.rag_ollama_model,
    )
    logger.info("Embeddings model ready.")

    # Build the chunker once here so workers share the same instance rather
    # than each item constructing a new one inside chunk_documents().
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=settings.chunker_breakpoint_type,
        breakpoint_threshold_amount=settings.chunker_breakpoint_amount,
    )
    logger.info("SemanticChunker ready.")

    try:
        logger.info("Connecting to PGVector store...")
        pgvector_store = init_pgvector_store(embeddings)
        logger.info("PGVector store ready.")
    except Exception as e:
        logger.error("Failed to connect to PGVector store: %s", e)
        return

    semaphore = asyncio.Semaphore(settings.concurrent_requests)
    queue: asyncio.Queue = asyncio.Queue()
    num_consumers = settings.concurrent_requests

    producer_task = asyncio.create_task(producer(queue, reading_list, num_consumers))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, pgvector_store, splitter, semaphore))
        for _ in range(num_consumers)
    ]

    await producer_task
    await queue.join()
    # Consumers exit cleanly on their own poison-pill sentinels; just await them.
    await asyncio.gather(*consumer_tasks)

    logger.info("Pipeline complete.")


def main() -> None:
    """Parse CLI arguments and run the RAG ingestion pipeline."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Process a reading list from a JSON file."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to the JSON data file. Defaults to the bundled test data.",
    )
    args = parser.parse_args()

    if args.data_file:
        data_file_path = Path(args.data_file)
    else:
        data_file_path = Path(__file__).parent / "data" / "reading_list_test_data.json"

    asyncio.run(run_pipeline(data_file_path))


if __name__ == "__main__":
    main()
