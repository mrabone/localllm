from langchain_postgres import PGVector
from cli.config import settings


def get_rag_context(query: str, pgvector_store: PGVector) -> str:
    """Retrieve relevant context from PGVector based on the query."""
    if not pgvector_store:
        return ""

    try:
        # Use similarity_search_with_score to get documents with their distance scores
        results = pgvector_store.similarity_search_with_score(
            query, k=settings.cli_rag_k
        )

        # Filter results by threshold (lower score = more similar for L2 distance)
        filtered_results = [
            (doc, score)
            for doc, score in results
            if score <= settings.cli_rag_threshold
        ]

        if not filtered_results:
            return ""

        # Build context from filtered results
        context_parts = []
        for i, (doc, score) in enumerate(filtered_results, 1):
            source = doc.metadata.get("source", "Unknown source")
            context_parts.append(
                f"[Document {i}] (relevance: {score:.3f})\nSource: {source}\nContent: {doc.page_content}"
            )

        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"Warning: Error retrieving RAG context: {e}")
        return ""


def build_rag_prompt(user_query: str, context: str) -> str:
    """Build a prompt enriched with RAG context."""
    return (
        "Use the following relevant information to answer the user's question. "
        "If the context doesn't contain relevant information, answer based on your general knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        "Answer:"
    )
