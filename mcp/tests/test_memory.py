import uuid
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.memory import (
    append_turn,
    create_session,
    ensure_turns_table,
    load_long_term_memories,
    load_window,
    save_message,
    session_exists,
)


def _make_mock_conn():
    """Return a (mock_conn, mock_cur) pair wired up as context managers."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


class TestCreateSession:
    def test_returns_uuid(self):
        """create_session should return a UUID object."""
        mock_conn, _ = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        assert isinstance(session_id, uuid.UUID)

    def test_each_call_returns_unique_uuid(self):
        """create_session should return a different UUID each time."""
        mock_conn, _ = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id_1 = create_session("dsn")
            session_id_2 = create_session("dsn")
        assert session_id_1 != session_id_2

    def test_returned_uuid_is_valid(self):
        """The returned UUID should be valid and convertible to string."""
        mock_conn, _ = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        str_id = str(session_id)
        assert len(str_id) == 36
        assert str_id.count("-") == 4

    def test_inserts_row_into_chat_sessions(self):
        """create_session should INSERT the new UUID into chat_sessions."""
        mock_conn, mock_cur = _make_mock_conn()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_id = create_session("dsn")
        mock_cur.execute.assert_called_once()
        sql, params = mock_cur.execute.call_args[0]
        assert "INSERT INTO chat_sessions" in sql
        assert str(session_id) in params


class TestSessionExists:
    def test_returns_true_when_row_exists(self):
        """session_exists should return True when chat_sessions has a matching row."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (1,)

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is True

    def test_returns_false_when_no_row(self):
        """session_exists should return False when chat_sessions has no matching row."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            result = session_exists("dsn", session_id)

        assert result is False

    def test_raises_on_db_error(self):
        """session_exists should propagate DB exceptions to the caller."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("Connection error")

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            with pytest.raises(Exception, match="Connection error"):
                session_exists("dsn", session_id)

    def test_converts_session_id_to_string(self):
        """session_exists should pass the UUID as a string to the SQL query."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        session_id = uuid.uuid4()
        with patch("common.session_store.get_conn", return_value=mock_conn):
            session_exists("dsn", session_id)

        _, params = mock_cur.execute.call_args[0]
        assert str(session_id) in params


class TestSaveMessage:
    def test_calls_mem0_add_with_correct_parameters(self):
        """save_message should call mem0.add with the message in correct format."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()
        role = "user"
        content = "Hello, world!"

        save_message(mem0, session_id, role, content)

        mem0.add.assert_called_once_with(
            [{"role": role, "content": content}],
            user_id=str(session_id),
        )

    def test_saves_user_messages(self):
        """save_message should correctly save user messages."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()

        save_message(mem0, session_id, "user", "What is Paris?")

        mem0.add.assert_called_once()
        call_args = mem0.add.call_args
        assert call_args[0][0][0]["role"] == "user"
        assert call_args[0][0][0]["content"] == "What is Paris?"

    def test_saves_assistant_messages(self):
        """save_message should correctly save assistant messages."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()

        save_message(mem0, session_id, "assistant", "Paris is the capital of France.")

        mem0.add.assert_called_once()
        call_args = mem0.add.call_args
        assert call_args[0][0][0]["role"] == "assistant"
        assert call_args[0][0][0]["content"] == "Paris is the capital of France."

    def test_converts_session_id_to_string(self):
        """save_message should convert UUID to string when calling mem0.add."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()

        save_message(mem0, session_id, "user", "test")

        call_args = mem0.add.call_args
        assert call_args[1]["user_id"] == str(session_id)

    def test_raises_exception_on_mem0_failure(self):
        """save_message should propagate exceptions from mem0.add."""
        mem0 = MagicMock()
        mem0.add.side_effect = Exception("Storage error")
        session_id = uuid.uuid4()

        with pytest.raises(Exception, match="Storage error"):
            save_message(mem0, session_id, "user", "test")

    def test_handles_long_content(self):
        """save_message should handle long message content."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()
        long_content = "x" * 10000

        save_message(mem0, session_id, "user", long_content)

        call_args = mem0.add.call_args
        assert call_args[0][0][0]["content"] == long_content

    def test_handles_special_characters(self):
        """save_message should handle special characters in content."""
        mem0 = MagicMock()
        session_id = uuid.uuid4()
        content = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?\n\t"

        save_message(mem0, session_id, "user", content)

        call_args = mem0.add.call_args
        assert call_args[0][0][0]["content"] == content


class TestEnsureTurnsTable:
    def test_executes_create_table_and_index(self):
        """ensure_turns_table should execute CREATE TABLE (sessions), CREATE TABLE (turns) and CREATE INDEX statements."""
        mock_conn, mock_cur = _make_mock_conn()

        with patch("common.session_store.get_conn", return_value=mock_conn):
            ensure_turns_table("host=localhost dbname=test")

        assert mock_cur.execute.call_count == 3

    def test_raises_on_db_error(self):
        """ensure_turns_table should propagate database errors."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("DB error")

        with patch("common.session_store.get_conn", return_value=mock_conn):
            with pytest.raises(Exception, match="DB error"):
                ensure_turns_table("host=localhost dbname=test")


class TestAppendTurn:
    def test_inserts_row_with_correct_values(self):
        """append_turn should INSERT a row with session_id, role, and content."""
        mock_conn, mock_cur = _make_mock_conn()

        session_id = uuid.uuid4()
        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            append_turn("dsn", session_id, "user", "hello")

        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args
        sql, params = call_args[0]
        assert "INSERT INTO conversation_turns" in sql
        assert params == (str(session_id), "user", "hello")

    def test_raises_on_db_error(self):
        """append_turn should propagate database errors."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("write error")

        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            with pytest.raises(Exception, match="write error"):
                append_turn("dsn", uuid.uuid4(), "user", "hello")

    def test_converts_session_id_to_string(self):
        """append_turn should pass the session UUID as a string to the DB."""
        mock_conn, mock_cur = _make_mock_conn()

        session_id = uuid.uuid4()
        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            append_turn("dsn", session_id, "assistant", "reply")

        _, params = mock_cur.execute.call_args[0]
        assert params[0] == str(session_id)
        assert isinstance(params[0], str)


class TestLoadWindow:
    def _make_mock_conn(self, rows: list[dict]) -> MagicMock:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = rows
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_returns_rows_as_role_content_dicts(self):
        """load_window should map DB rows to {role, content} dicts."""
        rows = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        mock_conn = self._make_mock_conn(rows)
        session_id = uuid.uuid4()

        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            messages = load_window("dsn", session_id, window_size=10)

        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_returns_empty_list_when_no_rows(self):
        """load_window should return an empty list when there are no turns."""
        mock_conn = self._make_mock_conn([])
        session_id = uuid.uuid4()

        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            messages = load_window("dsn", session_id, window_size=10)

        assert messages == []

    def test_passes_window_size_and_session_id_to_query(self):
        """load_window should pass window_size (LIMIT) and session_id to the query."""
        mock_conn = self._make_mock_conn([])
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        session_id = uuid.uuid4()

        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            load_window("dsn", session_id, window_size=5)

        call_args = mock_cur.execute.call_args
        _, params = call_args[0]
        assert params == (str(session_id), 5)

    def test_returns_empty_list_on_db_error(self):
        """load_window should return empty list when a DB error occurs."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.side_effect = Exception("connection lost")

        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            messages = load_window("dsn", uuid.uuid4(), window_size=10)

        assert messages == []

    def test_default_window_size_is_ten(self):
        """The default window_size should be 10."""
        mock_conn = self._make_mock_conn([])
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        session_id = uuid.uuid4()

        with patch("mcp_server.memory.get_conn", return_value=mock_conn):
            load_window("dsn", session_id)

        _, params = mock_cur.execute.call_args[0]
        assert params[1] == 10


class TestLoadLongTermMemories:
    def test_returns_single_consolidated_system_message(self):
        """load_long_term_memories should return a single system message containing
        all facts as bullet points under a descriptive header."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "User likes Python"},
            {"memory": "User is a senior engineer"},
        ]

        session_id = uuid.uuid4()
        messages = load_long_term_memories(mem0, session_id)

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "User likes Python" in messages[0]["content"]
        assert "User is a senior engineer" in messages[0]["content"]
        assert "- User likes Python" in messages[0]["content"]
        assert "- User is a senior engineer" in messages[0]["content"]

    def test_consolidated_message_has_descriptive_header(self):
        """The consolidated message must have a header explaining the source."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": "A fact"}]

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.index("A fact") > 0

    def test_respects_long_term_max_cap(self):
        """load_long_term_memories should include at most long_term_max facts."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(10)]

        session_id = uuid.uuid4()
        messages = load_long_term_memories(mem0, session_id, long_term_max=3)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.count("- fact") == 3

    def test_default_long_term_max_is_three(self):
        """Default long_term_max should be 3."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [{"memory": f"fact {i}"} for i in range(10)]

        session_id = uuid.uuid4()
        messages = load_long_term_memories(mem0, session_id)

        assert len(messages) == 1
        assert messages[0]["content"].count("- fact") == 3

    def test_returns_empty_list_when_no_memories(self):
        """load_long_term_memories should return empty list when no memories exist."""
        mem0 = MagicMock()
        mem0.get_all.return_value = []

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert messages == []

    def test_handles_dict_with_results_key(self):
        """load_long_term_memories should unwrap dict with 'results' key."""
        mem0 = MagicMock()
        mem0.get_all.return_value = {"results": [{"memory": "A fact"}]}

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert len(messages) == 1
        assert "A fact" in messages[0]["content"]

    def test_returns_empty_list_on_exception(self):
        """load_long_term_memories should return empty list if get_all raises."""
        mem0 = MagicMock()
        mem0.get_all.side_effect = Exception("connection error")

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert messages == []

    def test_skips_entries_without_content(self):
        """load_long_term_memories should skip items with no extractable content."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "Valid fact"},
            {"other_field": "no content"},
            {"memory": "Another valid fact"},
        ]

        messages = load_long_term_memories(mem0, uuid.uuid4(), long_term_max=10)

        assert len(messages) == 1
        assert "Valid fact" in messages[0]["content"]
        assert "Another valid fact" in messages[0]["content"]
        assert "no content" not in messages[0]["content"]

    def test_always_uses_system_role_regardless_of_source_role(self):
        """load_long_term_memories should always assign role=system."""
        mem0 = MagicMock()
        mem0.get_all.return_value = [
            {"memory": "A fact", "role": "user"},
        ]

        messages = load_long_term_memories(mem0, uuid.uuid4())

        assert messages[0]["role"] == "system"
