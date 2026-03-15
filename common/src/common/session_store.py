import logging
import uuid

from common.db_pool import get_conn

logger = logging.getLogger(__name__)


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
        with get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(_CREATE_SESSIONS_TABLE)
            cur.execute(_CREATE_TURNS_TABLE)
            cur.execute(_CREATE_TURNS_INDEX)
        logger.info("chat_sessions and conversation_turns tables ready.")
    except Exception as exc:
        logger.error("Failed to create tables: %s", exc, exc_info=True)
        raise


def create_session(dsn: str) -> uuid.UUID:
    """Create a new session row in chat_sessions and return its UUID."""
    session_id = uuid.uuid4()
    logger.info("Creating new session: %s", session_id)
    try:
        with get_conn(dsn) as conn, conn.cursor() as cur:
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
    """Return True if the session row exists in chat_sessions.

    Raises on database errors rather than returning False, so that
    infrastructure failures surface as server errors rather than silent 404s.
    """
    str_session_id = str(session_id)
    logger.debug("Checking if session exists: %s", str_session_id)
    with get_conn(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM chat_sessions WHERE id = %s LIMIT 1",
            (str_session_id,),
        )
        exists = cur.fetchone() is not None
    logger.info("Session %s exists: %s", str_session_id, exists)
    return exists
