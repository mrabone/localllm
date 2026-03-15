import uuid

import uvicorn
from fastmcp import FastMCP

from common.logging_utils import setup_logging
from mcp_server.config import settings
from mcp_server.memory import (
    append_turn,
    load_long_term_memories,
    load_window,
    save_message,
)
from mcp_server.rag import build_rag_system_message, get_rag_context
from mcp_server.services import get_mem0, get_pg_dsn, get_rag_store, lifespan

setup_logging()

mcp = FastMCP(name="localllm-mcp", lifespan=lifespan)


@mcp.tool(tags={"internal"})
def load_conversation_window(session_id: str, window_size: int = 10) -> list[dict]:
    """Return the most recent verbatim turns for a session, oldest first.

    Each entry is a dict with ``role`` and ``content`` keys, ready to pass
    directly into an LLM message list.

    Args:
        session_id: UUID string of the session.
        window_size: Maximum number of turns to return.
    """
    return load_window(get_pg_dsn(), uuid.UUID(session_id), window_size=window_size)


@mcp.tool(tags={"internal"})
def load_long_term_memory(session_id: str, long_term_max: int = 3) -> str:
    """Return Mem0-extracted semantic facts for a session as a formatted string.

    Facts are presented as a bullet-point system message.  Returns an empty
    string if no long-term memories have been stored yet.

    Args:
        session_id: UUID string of the session.
        long_term_max: Maximum number of memory facts to include.
    """
    messages = load_long_term_memories(
        get_mem0(), uuid.UUID(session_id), long_term_max=long_term_max
    )
    if not messages:
        return ""
    return messages[0]["content"]


@mcp.tool(tags={"internal"})
def persist_message(session_id: str, role: str, content: str) -> None:
    """Persist a single conversation turn to both Mem0 and the verbatim window.

    Mem0 runs an LLM extraction pass to distil semantic facts; the verbatim
    text is also inserted into the sliding-window PostgreSQL table so it
    appears in future ``load_conversation_window`` calls.

    Args:
        session_id: UUID string of the session.
        role: Message role — ``"user"`` or ``"assistant"``.
        content: The full text of the message.
    """
    parsed_id = uuid.UUID(session_id)
    save_message(get_mem0(), parsed_id, role, content)
    append_turn(get_pg_dsn(), parsed_id, role, content)


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the RAG knowledge base and return relevant document snippets.

    Returns a formatted string of ranked document snippets, or an empty
    string if RAG is disabled, no relevant documents were found, or the
    query score exceeds the configured distance threshold.

    The returned string is ready to embed directly in an LLM system message
    via ``build_rag_system_message``.

    Args:
        query: The search query, typically the user's current message.
    """
    rag_result = get_rag_context(query, get_rag_store())
    if rag_result is None:
        return ""
    return build_rag_system_message(rag_result.context)


def main() -> None:
    uvicorn.run(
        mcp.http_app(path="/mcp"),
        host=settings.mcp_host,
        port=settings.mcp_port,
    )


if __name__ == "__main__":
    main()
