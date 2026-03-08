import logging
import uuid
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool
from mem0 import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mem0 response normalisation helpers
# ---------------------------------------------------------------------------
def _normalise_mem0_results(results: Any) -> list[dict]:
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


# ---------------------------------------------------------------------------
# PostgreSQL connection pool
# ---------------------------------------------------------------------------

# Module-level connection pool.  Initialised lazily on the first call to
# _get_conn() so that imports don't require a live database.  Replaced by
# _init_pool() during server startup for explicit control over pool sizing.
_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_dsn: str | None = None

_POOL_MIN_CONN = 2
_POOL_MAX_CONN = 10


def _init_pool(dsn: str) -> None:
    """Create (or recreate) the module-level connection pool for *dsn*.

    Safe to call multiple times; a new pool is only created when the DSN
    changes.  Intended to be called once during server lifespan startup so
    that a warm pool is ready before the first request arrives.
    """
    global _pool, _pool_dsn
    if _pool is not None and _pool_dsn == dsn:
        return
    if _pool is not None:
        try:
            _pool.closeall()
        except Exception:
            pass
    logger.info(
        "Initialising psycopg2 connection pool (min=%d, max=%d).",
        _POOL_MIN_CONN,
        _POOL_MAX_CONN,
    )
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=_POOL_MIN_CONN,
        maxconn=_POOL_MAX_CONN,
        dsn=dsn,
    )
    _pool_dsn = dsn
    logger.info("psycopg2 connection pool ready.")


def _close_pool() -> None:
    """Close all pooled connections.  Call during server lifespan shutdown."""
    global _pool, _pool_dsn
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("psycopg2 connection pool closed.")
        except Exception as exc:
            logger.warning("Error closing connection pool: %s", exc)
        finally:
            _pool = None
            _pool_dsn = None


class _PooledConn:
    """Context manager that checks out a connection and returns it to the pool.

    Falls back to a direct ``psycopg2.connect()`` when no pool is available
    (e.g. during tests that patch ``_get_conn`` or run without a live DB).
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = None
        self._from_pool = False

    def __enter__(self):
        if _pool is not None and _pool_dsn == self._dsn:
            self._conn = _pool.getconn()
            self._from_pool = True
        else:
            self._conn = psycopg2.connect(self._dsn)
            self._from_pool = False
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            if self._from_pool and _pool is not None:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                except Exception:
                    pass
                _pool.putconn(self._conn)
            else:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                finally:
                    self._conn.close()
        return False


def _get_conn(dsn: str) -> "_PooledConn":
    """Return a ``_PooledConn`` context manager for *dsn*.

    Usage::

        with _get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(...)

    Kept as a named function so that unit tests can continue to patch
    ``server.memory._get_conn`` and inject mock connections without needing
    to be aware of the pool internals.
    """
    return _PooledConn(dsn)


# ---------------------------------------------------------------------------
# PostgreSQL sliding-window helpers
# ---------------------------------------------------------------------------

_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_CREATE_TURNS_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID      NOT NULL,
    role        TEXT      NOT NULL,
    content     TEXT      NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_CREATE_TURNS_INDEX = """
CREATE INDEX IF NOT EXISTS conversation_turns_session_id_id_idx
    ON conversation_turns (session_id, id);
"""


def ensure_turns_table(dsn: str) -> None:
    """Create the chat_sessions, conversation_turns table and index if they do not exist.

    Safe to call multiple times (uses IF NOT EXISTS).  Intended to be called
    once during server lifespan startup.
    """
    logger.info("Ensuring chat_sessions and conversation_turns tables exist.")
    try:
        with _get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(_CREATE_SESSIONS_TABLE)
            cur.execute(_CREATE_TURNS_TABLE)
            cur.execute(_CREATE_TURNS_INDEX)
        logger.info("chat_sessions and conversation_turns tables ready.")
    except Exception as exc:
        logger.error("Failed to create tables: %s", exc, exc_info=True)
        raise


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
        with _get_conn(dsn) as conn, conn.cursor() as cur:
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
            _get_conn(dsn) as conn,
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


def create_session(dsn: str) -> uuid.UUID:
    """Create a new session row in chat_sessions and return its UUID."""
    session_id = uuid.uuid4()
    logger.info("Creating new session: %s", session_id)
    try:
        with _get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (id) VALUES (%s)",
                (str(session_id),),
            )
        logger.info("Session %s created in chat_sessions.", session_id)
    except Exception as exc:
        logger.error("Failed to create session %s: %s", session_id, exc, exc_info=True)
        raise
    return session_id


def session_exists(dsn: str, session_id: uuid.UUID) -> bool:
    """Return True if the session row exists in chat_sessions."""
    str_session_id = str(session_id)
    logger.debug("Checking if session exists: %s", str_session_id)
    try:
        with _get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chat_sessions WHERE id = %s LIMIT 1",
                (str_session_id,),
            )
            exists = cur.fetchone() is not None
        logger.info("Session %s exists: %s", str_session_id, exists)
        return exists
    except Exception as exc:
        logger.error("Error checking session existence: %s", exc, exc_info=True)
        return False


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


def load_messages(
    mem0: Memory, session_id: uuid.UUID, query: str, max_messages: int = 10
) -> list[dict[str, str]]:
    """Retrieve semantically relevant memories for the given query.

    NOTE: This function is used exclusively by the ``GET /sessions/{id}``
    history-viewing endpoint.  The chat path does **not** call it — it uses
    ``load_long_term_memories`` (layer 1) and ``load_window`` (layer 2)
    instead.

    Returns a list of dicts with ``role`` and ``content`` keys, ready to pass
    directly into the Ollama chat messages list.  If ``query`` is empty all
    stored memories are returned via ``get_all``.

    At most ``max_messages`` entries are returned.  Mem0 already ranks results
    by relevance, so truncating the tail discards the least-relevant memories
    and keeps the context window from growing without bound.
    """
    str_session_id = str(session_id)
    logger.debug(
        "Loading messages from Mem0: session=%s, query='%s'",
        str_session_id,
        query[:100] if query else "(empty)",
    )

    try:
        if not query:
            logger.debug("Query is empty, calling get_all()")
            raw = mem0.get_all(user_id=str_session_id)
            logger.debug(
                "get_all() returned: type=%s, len=%d",
                type(raw),
                len(raw) if isinstance(raw, (list, dict)) else 0,
            )
            memories = _normalise_mem0_results(raw)
            logger.debug(
                "After normalization: memories type=%s, len=%d",
                type(memories),
                len(memories),
            )
        else:
            logger.debug("Query is non-empty, calling search()")
            raw = mem0.search(query, user_id=str_session_id)
            logger.debug(
                "search() returned: type=%s, len=%d",
                type(raw),
                len(raw) if isinstance(raw, (list, dict)) else 0,
            )
            memories = _normalise_mem0_results(raw)

        # Build standard {role, content} dicts from normalised memory entries.
        messages = []
        for m in memories:
            if not isinstance(m, dict):
                logger.debug("Skipping non-dict memory item: %s", type(m))
                continue

            memory_content = _extract_memory_content(m)
            memory_role = m.get("role", "system")

            if memory_content:
                messages.append({"role": memory_role, "content": memory_content})
                logger.debug(
                    "Added memory: role=%s, content_len=%d",
                    memory_role,
                    len(str(memory_content)),
                )
            else:
                logger.debug(
                    "Could not extract memory content from item with keys: %s",
                    list(m.keys()),
                )

        logger.info(
            "Loaded %d memories from Mem0 (session=%s, query_len=%d)",
            len(messages),
            str_session_id,
            len(query),
        )
        if messages:
            logger.debug(
                "Sample memory: role=%s, content=%s...",
                messages[0].get("role"),
                str(messages[0].get("content"))[:50],
            )

        if len(messages) > max_messages:
            logger.debug(
                "Capping memories from %d to %d (max_messages=%d)",
                len(messages),
                max_messages,
                max_messages,
            )
            messages = messages[:max_messages]

        return messages
    except Exception as exc:
        logger.error(
            "Error loading messages from Mem0: session=%s, query='%s': %s",
            str_session_id,
            query[:100] if query else "(empty)",
            exc,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Long-term Mem0 memories (layer 1)
# ---------------------------------------------------------------------------


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
        logger.debug(
            "get_all() returned: type=%s, len=%d",
            type(raw),
            len(raw) if isinstance(raw, (list, dict)) else 0,
        )

        memories = _normalise_mem0_results(raw)

        facts: list[str] = []
        for m in memories:
            if not isinstance(m, dict):
                continue
            memory_content = _extract_memory_content(m)
            if memory_content:
                facts.append(str(memory_content))

        if len(facts) > long_term_max:
            logger.debug(
                "Capping long-term memories from %d to %d",
                len(facts),
                long_term_max,
            )
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
