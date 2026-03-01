import logging
from dataclasses import dataclass

from langchain_postgres import PGVector

from cli.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RagResult:
    """The result of a RAG retrieval operation.

    Separates the formatted context string from the document count so that
    callers can use the count for display without parsing the context string.
    """

    context: str
    document_count: int


def get_rag_context(query: str, pgvector_store: PGVector | None) -> RagResult | None:
    """Retrieve relevant context from PGVector for the given query.

    Returns a RagResult with the formatted context and document count, or
    None if RAG is disabled, no results were found, or an error occurred.

    Args:
        query: The user's input to search against the vector store.
        pgvector_store: An initialised PGVector store, or None if RAG is disabled.
    """
    if pgvector_store is None:
        return None

    try:
        results = pgvector_store.similarity_search_with_score(
            query, k=settings.cli_rag_k
        )

        if not results:
            return None

        # Fast path: if the best (lowest) score already exceeds the threshold,
        # all results are too distant — skip the filter entirely.
        best_score = results[0][1]
        if best_score > settings.cli_rag_max_distance:
            return None

        filtered = [
            (doc, score)
            for doc, score in results
            if score <= settings.cli_rag_max_distance
        ]

        if not filtered:
            return None

        context_parts = [
            f"[Document {i}] (relevance: {score:.3f})\n"
            f"Source: {doc.metadata.get('source', 'Unknown source')}\n"
            f"Content: {doc.page_content}"
            for i, (doc, score) in enumerate(filtered, 1)
        ]

        return RagResult(
            context="\n\n".join(context_parts),
            document_count=len(filtered),
        )

    except Exception as e:
        logger.warning("Error retrieving RAG context: %s", e)
        return None


def build_rag_prompt(user_query: str, context: str) -> str:
    """Wrap a user query with retrieved context to form a RAG-enriched prompt.

    Args:
        user_query: The original question from the user.
        context: The formatted context string from get_rag_context().
    """
    return (
        "Use the following relevant information to answer the user's question. "
        "If the context doesn't contain relevant information, answer based on "
        "your general knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        "Answer:"
    )
