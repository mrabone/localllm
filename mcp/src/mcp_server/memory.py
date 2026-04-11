import logging
import uuid

import psycopg2.extras
from mem0 import Memory

from common.db_pool import get_conn

logger = logging.getLogger(__name__)


def _normalise_mem0_results(results) -> list[dict]:
    """Unwrap the Mem0 API response into a plain list of memory dicts.

    Mem0's PGVector provider returns results in several shapes depending on
    version: a plain list, a dict with a ``"results"`` key, or a dict with a
    ``"data"`` key.  This helper normalises all three forms into a single flat
    list so callers don't need to repeat the branching logic.
    """
    if isinstance(results, dict):
        return results.get("results", results.get("data", []))
    if isinstance(results, list):
        return results
    return []


def _extract_memory_content(m: dict) -> str | None:
    """Return the content string from a Mem0 memory dict, or None if absent.

    Mem0 uses different field names across provider versions:
    ``memory`` (PGVector), ``content``, ``data``, and ``text``.
    The fields are tried in that priority order.
    """
    return m.get("memory") or m.get("content") or m.get("data") or m.get("text")


def append_turn(dsn: str, session_id: uuid.UUID, role: str, content: str) -> None:
    """Insert a single verbatim turn into the sliding-window table.

    Both user and assistant turns are stored here.  The window is trimmed to
    ``window_size`` at read time by ``load_window``.
    """
    str_session_id = str(session_id)
    logger.debug(
        "Appending turn: session=%s, role=%s, content_len=%d",
        str_session_id,
        role,
        len(content),
    )
    try:
        with get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_turns (session_id, role, content) "
                "VALUES (%s, %s, %s)",
                (str_session_id, role, content),
            )
        logger.debug("Turn appended successfully.")
    except Exception as exc:
        logger.error(
            "Error appending turn: session=%s, role=%s: %s",
            str_session_id,
            role,
            exc,
            exc_info=True,
        )
        raise


def load_window(
    dsn: str, session_id: uuid.UUID, window_size: int = 10
) -> list[dict[str, str]]:
    """Return the last ``window_size`` turns for the session, oldest first.

    Returns a list of ``{"role": ..., "content": ...}`` dicts ready to pass
    directly into the Ollama message list as layer 2 (verbatim recent turns).
    """
    str_session_id = str(session_id)
    logger.debug(
        "Loading window: session=%s, window_size=%d", str_session_id, window_size
    )
    try:
        with (
            get_conn(dsn) as conn,
            conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur,
        ):
            cur.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM conversation_turns
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) sub
                ORDER BY id ASC
                """,
                (str_session_id, window_size),
            )
            rows = cur.fetchall()
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]
        logger.info(
            "Loaded %d window turns (session=%s, window_size=%d)",
            len(messages),
            str_session_id,
            window_size,
        )
        return messages
    except Exception as exc:
        logger.error(
            "Error loading window: session=%s: %s", str_session_id, exc, exc_info=True
        )
        return []


def should_extract_memories(role: str) -> bool:
    """Decide whether a message should be sent through Mem0's fact extraction.

    Only user messages are worth extracting — assistant responses are derived
    from model + context and don't contain facts about the user.  Skipping
    them avoids a 30-60s LLM round-trip per assistant turn.

    Content-level filtering (greetings, short messages) is handled by mem0's
    fact extraction prompt, which instructs the LLM to return an empty facts
    list for trivial inputs.
    """
    return role == "user"


def save_message(mem0: Memory, session_id: uuid.UUID, role: str, content: str) -> None:
    """Persist a single message to Mem0.

    Mem0 extracts and stores semantic memories from the message rather than
    recording a verbatim transcript.
    """
    str_session_id = str(session_id)
    logger.info(
        "Saving message to Mem0: session=%s, role=%s, content_len=%d",
        str_session_id,
        role,
        len(content),
    )
    try:
        mem0.add(
            [{"role": role, "content": content}],
            user_id=str_session_id,
        )
        logger.debug("Message saved successfully to Mem0")
    except Exception as exc:
        logger.error(
            "Error saving message to Mem0: session=%s, role=%s: %s",
            str_session_id,
            role,
            exc,
            exc_info=True,
        )
        raise


def load_long_term_memories(
    mem0: Memory, session_id: uuid.UUID, long_term_max: int = 3
) -> list[dict[str, str]]:
    """Return a single consolidated system message containing Mem0-extracted facts.

    Semantic facts distilled by Mem0 from previous conversations are collapsed
    into one ``system`` role message with a descriptive header and bullet-point
    facts.  This gives the model clear attribution for where the facts came from
    rather than a sequence of bare system messages.

    Returns a list containing a single dict (so the return type is consistent
    with other context-layer helpers), or an empty list if no facts exist.

    Uses ``get_all`` rather than ``search`` so that all stored facts are
    available regardless of semantic proximity to the current query.
    """
    str_session_id = str(session_id)
    logger.debug(
        "Loading long-term memories: session=%s, long_term_max=%d",
        str_session_id,
        long_term_max,
    )
    try:
        raw = mem0.get_all(user_id=str_session_id)
        memories = _normalise_mem0_results(raw)

        facts: list[str] = []
        for m in memories:
            if not isinstance(m, dict):
                continue
            memory_content = _extract_memory_content(m)
            if memory_content:
                facts.append(str(memory_content))

        if len(facts) > long_term_max:
            facts = facts[:long_term_max]

        logger.info(
            "Loaded %d long-term memories (session=%s, long_term_max=%d)",
            len(facts),
            str_session_id,
            long_term_max,
        )

        if not facts:
            return []

        bullet_points = "\n".join(f"- {fact}" for fact in facts)
        content = (
            "The following facts have been remembered from previous conversations "
            f"with this user:\n{bullet_points}"
        )
        return [{"role": "system", "content": content}]

    except Exception as exc:
        logger.error(
            "Error loading long-term memories: session=%s: %s",
            str_session_id,
            exc,
            exc_info=True,
        )
        return []


__all__ = [
    "append_turn",
    "load_long_term_memories",
    "load_window",
    "save_message",
    "should_extract_memories",
]
