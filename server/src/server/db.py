import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine

metadata = MetaData()

sessions_table = Table(
    "chat_sessions",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    ),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

messages_table = Table(
    "chat_messages",
    metadata,
    Column("id", String, primary_key=True, default=lambda: str(uuid.uuid4())),
    Column(
        "session_id",
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

Index(
    "ix_chat_messages_session_created",
    messages_table.c.session_id,
    messages_table.c.created_at,
)


def create_tables(engine: Engine) -> None:
    """Create chat_sessions and chat_messages tables if they do not exist."""
    metadata.create_all(engine)


def load_messages(engine: Engine, session_id: uuid.UUID) -> list[dict[str, str]]:
    """Return all messages for a session ordered by creation time.

    Each message is a plain dict with 'role' and 'content' keys, ready to
    pass directly into the Ollama chat messages list.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            messages_table.select()
            .where(messages_table.c.session_id == session_id)
            .order_by(messages_table.c.created_at)
        ).fetchall()
    return [{"role": row.role, "content": row.content} for row in rows]


def save_message(
    engine: Engine, session_id: uuid.UUID, role: str, content: str
) -> None:
    """Persist a single message to the database."""
    with engine.begin() as conn:
        conn.execute(
            messages_table.insert().values(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=role,
                content=content,
            )
        )


def create_session(engine: Engine) -> uuid.UUID:
    """Insert a new session row and return its UUID."""
    session_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(sessions_table.insert().values(id=session_id))
    return session_id


def session_exists(engine: Engine, session_id: uuid.UUID) -> bool:
    """Return True if the given session ID exists in the database."""
    with engine.connect() as conn:
        row = conn.execute(
            sessions_table.select().where(sessions_table.c.id == session_id)
        ).fetchone()
    return row is not None
