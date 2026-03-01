import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import httpx
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

    Falls back from <main> → <body> → the whole document when selecting the
    primary content block. Collapses whitespace and normalises newlines.
    """
    soup = BeautifulSoup(html, "html.parser")
    content_block = soup.main or soup.body or soup
    raw_text = content_block.get_text(separator="\n", strip=True)

    # Collapse runs of spaces/tabs into a single space
    cleaned = re.sub(r"[ \t]+", " ", raw_text)
    # Normalise 3+ consecutive newlines into a standard paragraph break
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned if cleaned else None


def chunk_documents(
    text: str,
    url: str,
    embeddings: OllamaEmbeddings,
) -> list[Document]:
    """Split text into semantic chunks using the configured SemanticChunker.

    Args:
        text: The cleaned plain text to split.
        url: The source URL, stored in each chunk's metadata.
        embeddings: The embeddings model used to determine split points.

    Returns:
        A list of Document chunks, or an empty list if splitting produced nothing.
    """
    document = Document(page_content=text, metadata={"source": url})
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=settings.chunker_breakpoint_type,
        breakpoint_threshold_amount=settings.chunker_breakpoint_amount,
    )
    return splitter.split_documents([document])


async def process_item(
    item: dict,
    pgvector_store: PGVector,
    embeddings: OllamaEmbeddings,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> None:
    """Process a single reading-list item end-to-end.

    Steps: validate → duplicate-check → scrape → extract text →
    chunk → embed → store.

    Args:
        item: A reading-list dict with 'url' and optional 'title' keys.
        pgvector_store: Destination vector store.
        embeddings: Embeddings model for chunking and storage.
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
        if pgvector_store.similarity_search(query=" ", filter={"source": url}):
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
        chunks = chunk_documents(text, url, embeddings)
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
) -> None:
    """Enqueue all reading-list items for the consumers to process."""
    for item in reading_list:
        await queue.put(item)


async def consumer(
    queue: asyncio.Queue,
    pgvector_store: PGVector,
    embeddings: OllamaEmbeddings,
    semaphore: asyncio.Semaphore,
) -> None:
    """Consume items from the queue until cancelled.

    Unexpected errors are logged and the current item is skipped so that
    the worker remains alive for subsequent items.
    """
    while True:
        try:
            item = await queue.get()
            await process_item(
                item,
                pgvector_store,
                embeddings,
                semaphore,
                settings.request_delay,
            )
            queue.task_done()
        except asyncio.CancelledError:
            break
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

    try:
        logger.info("Connecting to PGVector store...")
        pgvector_store = init_pgvector_store(embeddings)
        logger.info("PGVector store ready.")
    except Exception as e:
        logger.error("Failed to connect to PGVector store: %s", e)
        return

    semaphore = asyncio.Semaphore(settings.concurrent_requests)
    queue: asyncio.Queue = asyncio.Queue()

    producer_task = asyncio.create_task(producer(queue, reading_list))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, pgvector_store, embeddings, semaphore))
        for _ in range(settings.concurrent_requests)
    ]

    await producer_task
    await queue.join()

    for task in consumer_tasks:
        task.cancel()
    await asyncio.gather(*consumer_tasks, return_exceptions=True)

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
