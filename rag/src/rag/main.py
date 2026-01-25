from sqlalchemy import create_engine
import re
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import asyncio
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from rag.config import settings
from rag.logging_utils import LoggingSetup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


LoggingSetup.setup_logging()
logger = logging.getLogger(__name__)


def load_reading_list(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads the reading list from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A list of dictionaries representing the reading list.
        Returns an empty list if the file is not found or invalid.
    """
    if not file_path.exists():
        logger.error(f"File not found at {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path}: {e}")
        return []


async def scrape_url(
    url: str, semaphore: asyncio.Semaphore, delay: float
) -> Optional[str]:
    """
    Fetches HTML content from a single URL with rate limiting.
    """
    async with semaphore:
        await asyncio.sleep(delay)  # Apply delay before request
        try:
            logger.info(f"Attempting to fetch URL: {url}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=HEADERS)
                response.raise_for_status()  # Raise an exception for 4xx/5xx responses
                logger.info(f"Successfully fetched URL: {url}")
                return response.text
        except httpx.RequestError as e:
            logger.error(f"HTTPX Request error fetching {url}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Status error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching {url}: {e}")
            return None


async def process_item(
    item: Dict[str, Any],
    pgvector_store: PGVector,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> None:
    """
    Processes a single item from the reading list: scrapes HTML, extracts content,
    generates embeddings, and stores in PgVector.
    """
    title = item.get("title")
    url = item.get("url")

    if not url:
        logger.warning(
            f"Skipping item with missing URL: {title if title else 'Unknown Item'}"
        )
        return

    if not title:
        title = url  # Use URL as title if title is missing
        logger.info(f"Title missing for item, using URL as title: {title}")

    logger.info(f"Processing: {title} ({url})")

    # 1. Check if URL has already been processed
    try:
        existing_docs = pgvector_store.similarity_search(
            query=" ", filter={"source": url}
        )
        if existing_docs:
            logger.info(f"URL {url} has already been processed. Skipping.")
            return
    except Exception as e:
        logger.error(f"Error checking for existing documents for {url}: {e}")
        pass

    # 2. Scrape HTML content
    html_content = await scrape_url(url, semaphore, delay)
    if not html_content:
        logger.error(f"Failed to get HTML content for {url}. Skipping processing.")
        return

    # 3. Load and split documents with Langchain
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        raw_text = soup.get_text()
        cleaned_text = re.sub(r"\s+", " ", raw_text).strip()

        if not cleaned_text:
            logger.warning(
                f"No meaningful text extracted from {url}. Skipping embeddings."
            )
            return

        documents = [Document(page_content=cleaned_text, metadata={"source": url})]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        if not texts:
            logger.warning(f"No text chunks to add for {url}. Skipping.")
            return
        logger.info(
            f"Split {len(documents)} documents from {url} into {len(texts)} chunks."
        )
    except Exception as e:
        logger.error(f"Error loading or splitting documents for {url}: {e}")
        return

    # 4. Generate embeddings and store in PgVector
    try:
        # Add documents to PgVector
        # PGVector automatically creates embeddings if an embeddings object is provided
        pgvector_store.add_documents(texts)
        logger.info(f"Successfully added {len(texts)} chunks from {url} to PgVector.")
    except Exception as e:
        logger.error(f"Error adding documents to PgVector for {url}: {e}")


async def producer(queue: asyncio.Queue, reading_list: List[Dict[str, Any]]) -> None:
    """Puts items from the reading list into the queue."""
    for item in reading_list:
        await queue.put(item)


async def consumer(
    queue: asyncio.Queue,
    pgvector_store: PGVector,
    semaphore: asyncio.Semaphore,
) -> None:
    """Consumes items from the queue and processes them."""
    while True:
        try:
            item = await queue.get()
            await process_item(item, pgvector_store, semaphore, settings.request_delay)
            queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in consumer")
            break


async def run_pipeline(data_path: Path) -> None:
    """
    Orchestrates the loading and processing of the reading list concurrently.
    """
    logger.info(f"Loading reading list from: {data_path}")
    reading_list = load_reading_list(data_path)

    if not reading_list:
        logger.info("No items to process.")
        return

    logger.info(f"Successfully loaded {len(reading_list)} items.")

    logger.info("Initializing OllamaEmbeddings...")
    ollama_embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url, model=settings.rag_ollama_model
    )
    logger.info("OllamaEmbeddings initialized.")

    CONNECTION_STRING = f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"
    try:
        logger.info("Connecting to PgVector store...")
        engine = create_engine(CONNECTION_STRING)
        pgvector_store = PGVector(
            connection=engine,
            embeddings=ollama_embeddings,
            collection_name=settings.pg_collection_name,
        )
        logger.info("Successfully connected to PgVector store.")
    except Exception as e:
        logger.error(f"Failed to connect to PgVector store: {e}")
        return

    # Create a semaphore and a queue
    semaphore = asyncio.Semaphore(settings.concurrent_requests)
    queue = asyncio.Queue()

    # Create producer and consumer tasks
    producer_task = asyncio.create_task(producer(queue, reading_list))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, pgvector_store, semaphore))
        for _ in range(settings.concurrent_requests)
    ]

    # Wait for the producer to finish
    await producer_task

    # Wait for all items in the queue to be processed
    await queue.join()

    # Cancel the consumer tasks
    for task in consumer_tasks:
        task.cancel()

    # Wait for the consumer tasks to finish cancelling
    await asyncio.gather(*consumer_tasks, return_exceptions=True)

    logger.info("Pipeline processing complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Process a reading list from a JSON file."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to the JSON data file containing the reading list.",
        default=None,
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.data_file:
        data_file_path = Path(args.data_file)
    else:
        # Default to the test data file if no argument is provided
        data_file_path = base_dir / "data" / "reading_list_test_data.json"

    asyncio.run(run_pipeline(data_file_path))


if __name__ == "__main__":
    main()
